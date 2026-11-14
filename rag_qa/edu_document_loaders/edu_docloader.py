"""
Word 文档（.docx）加载器：OCRDOCLoader

功能说明：
    1. 读取 .docx 文件，按文档中"块（block）"的原始顺序依次解析；
    2. 对普通段落，提取其中的文字；
    3. 对段落中内嵌的图片，调用 OCR（光学字符识别）把图片里的文字识别出来并拼接进正文；
    4. 对表格，逐行逐单元格提取其中的段落文字。

为什么要用 OCR：
    Word 文档中经常有截图、扫描件、公式图片等非文本内容，
    如果只读取 paragraph.text，这些图片里的信息就会丢失，
    因此这里借助 OCR 把图片中的文字也转成可检索的文本，便于后续构建 RAG 知识库。

依赖说明：
    - python-docx：解析 .docx 文档结构（段落、表格、图片）；
    - rapidocr：对图片做 OCR 文字识别（GPU 版或 CPU 版，见 _edu_ocr.py）；
    - Pillow：把图片二进制流转换成可供 OCR 处理的图像对象；
    - langchain_core：提供 Document / BaseLoader 基类，方便接入 LangChain 的文档处理流程；
    - tqdm：显示文档解析进度的进度条。
"""

from typing import Iterator  # 类型注解：表示本类会以迭代器方式逐个产出文档
from tqdm import tqdm  # 进度条，用于展示文档块（段落/表格）的解析进度
from docx.table import _Cell, Table  # _Cell：表格单元格；Table：表格对象
from docx.oxml.table import CT_Tbl  # 表格在 XML 层面对应的元素类型
from docx.oxml.text.paragraph import CT_P  # 段落在 XML 层面对应的元素类型
from docx.text.paragraph import Paragraph  # 段落在 python-docx 中的封装对象
from docx import Document as Docu1  # Document 工厂函数：Docu1(filepath) 读取 .docx 文件
from docx.document import Document as Docu2  # Document 类型：用于判断传进来的父对象是否是整个文档
from docx import ImagePart  # 图片部件类型，用于判断嵌入资源是否为图片
from PIL import Image  # Pillow 图像库，用于把字节流解析为图像
from io import BytesIO  # 把二进制字节流包装成类文件对象，交给 Pillow 打开
import numpy as np  # 用于把 PIL 图像转换成 OCR 库需要的 numpy 数组格式
from langchain_core.documents import Document  # LangChain 的文档对象（注意与 docx 的同名类区分）
from langchain_core.document_loaders import BaseLoader  # LangChain 文档加载器基类，需实现 lazy_load

# 从同目录的 _edu_ocr.py 中导入 OCR 对象创建函数
# 注意：这里使用的是绝对导入，依赖 __init__.py 中把当前目录加入 sys.path 的处理
from _edu_ocr import get_ocr


class OCRDOCLoader(BaseLoader):
    """Word（.docx）文档加载器。

    继承自 LangChain 的 BaseLoader，重写 lazy_load 方法，
    把整个 .docx 文档转换为一个包含全部文本（含图片 OCR 结果）的 LangChain Document。
    """

    def __init__(self, filepath: str) -> None:
        """初始化加载器。

        Args:
            filepath: 待加载的 .docx 文件路径。
        """
        # 保存文件路径，后续在 lazy_load / doc2text 中使用
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载：把文件内容逐个（这里只有一个）产出为 LangChain Document。

        LangChain 约定的方法：不接收参数，使用生成器逐个 yield 文档对象，
        这样可以避免一次性把所有文档都读进内存。
        """
        # 调用 doc2text 完成实际的文档解析，得到整篇文档的纯文本
        line = self.doc2text(self.filepath)
        # 产出 LangChain 的 Document 对象，metadata 中记录来源文件路径，便于后续溯源
        yield Document(page_content=line, metadata={"source": self.filepath})

    def doc2text(self, filepath):
        """把 .docx 文件解析成一段文本，包含段落文字、表格文字以及图片 OCR 结果。

        Args:
            filepath: .docx 文件的路径。

        Returns:
            解析得到的完整文本字符串（各段落、表格单元格、图片识别结果以换行分隔）。
        """

        # 创建 OCR 识别对象（内部会自动选择 GPU 版或 CPU 版 RapidOCR）
        ocr = get_ocr()
        # print(f'ocr--》{ocr}')  # 调试用：输出 OCR 对象信息

        # 读取 Word 文档，得到 python-docx 的 Document 对象
        doc = Docu1(filepath)
        # print(f'doc-->{doc}')  # 调试用：输出读取到的文档信息
        # 定义一个空字符串，用于累积最终的文本内容
        resp = ""

        # 定义内部生成器：按文档中的原始顺序遍历所有块（段落、表格）
        def iter_block_items(parent):
            """按顺序迭代 parent 容器下的顶层块（段落或表格）。

            python-docx 的 doc.paragraphs 和 doc.tables 是分开返回的，
            无法还原它们在文档中的真实先后顺序。这里直接遍历底层 XML 的子元素，
            从而保证段落和表格按照它们在 Word 中出现的顺序被处理。

            Args:
                parent: 整个文档对象（Document）或表格单元格（_Cell）。

            Yields:
                Paragraph 或 Table 对象。
            """
            # 如果 parent 是整个文档，取其正文（body）节点作为遍历起点
            if isinstance(parent, Docu2):
                parent_elm = parent.element.body
            # 如果 parent 是表格单元格，取其 XML 元素（_tc）作为遍历起点（支持表格中嵌套表格）
            elif isinstance(parent, _Cell):
                parent_elm = parent._tc
            else:
                # 其他类型不支持解析，直接抛出异常
                raise ValueError("OCRDOCLoader parse fail")
            # print(f'parent_elm--》{parent_elm}')
            # print('*' * 80)

            # 遍历父节点下的每一个子元素
            for child in parent_elm.iterchildren():
                # print(f'child--》{child}')
                # 如果是段落元素，封装成 Paragraph 对象产出
                if isinstance(child, CT_P):
                    yield Paragraph(child, parent)
                # 如果是表格元素，封装成 Table 对象产出
                elif isinstance(child, CT_Tbl):
                    yield Table(child, parent)

        # print(f'doc.paragraphs-->{doc.paragraphs}')
        # print(f'doc.tables-->{doc.tables}')

        # 创建进度条：总数 = 顶层段落数 + 顶层表格数
        # 描述文字会随着处理进度不断更新为当前处理的块编号
        b_unit = tqdm(total=len(doc.paragraphs) + len(doc.tables),
                      desc="OCRDOCLoader block index: 0")

        # 按文档中的真实顺序遍历所有块（段落和表格）
        for i, block in enumerate(iter_block_items(doc)):
            # 更新进度条描述，显示当前正在处理的块索引
            b_unit.set_description("OCRDOCLoader  block index: {}".format(i))
            # 刷新进度条显示
            b_unit.refresh()

            # 情况一：当前块是段落
            if isinstance(block, Paragraph):
                # 提取段落的纯文本（去掉首尾空白），追加到结果中并换行
                resp += block.text.strip() + "\n"

                # 查找该段落中嵌入的所有图片（pic:pic 是 DrawingML 中图片元素的标签）
                images = block._element.xpath('.//pic:pic')
                for image in images:
                    # 通过 a:blip 的 r:embed 属性拿到图片在文档部件中的关系 ID
                    for img_id in image.xpath('.//a:blip/@r:embed'):
                        # 根据关系 ID 从文档部件中取出对应的资源对象
                        part = doc.part.related_parts[img_id]
                        # 只处理图片类型的资源（跳过超链接等其他关系）
                        if isinstance(part, ImagePart):
                            # part._blob 是图片的原始二进制数据
                            # BytesIO 把它包装成类文件对象，Image.open 再解析成图像
                            image = Image.open(BytesIO(part._blob))
                            # 把 PIL 图像转换成 numpy 数组后交给 OCR 识别
                            # ocr 返回 (识别结果列表, 耗时信息)，这里只取识别结果
                            result, _ = ocr(np.array(image))
                            # 如果识别到了文字
                            if result:
                                # result 中每一项形如 [文本框坐标, 识别文字, 置信度]，这里取文字部分
                                ocr_result = [line[1] for line in result]
                                # 把该图片识别出的所有文字用换行拼接后追加到结果中
                                resp += "\n".join(ocr_result)

            # 情况二：当前块是表格
            elif isinstance(block, Table):
                # 逐行遍历表格
                for row in block.rows:
                    # 逐单元格遍历该行
                    for cell in row.cells:
                        # 逐段落读取单元格内的文字
                        for paragraph in cell.paragraphs:
                            # 去掉首尾空白后追加到结果中，每个段落一行
                            resp += paragraph.text.strip() + "\n"

            # 每处理完一个块，进度条前进一格
            b_unit.update(1)

        # 返回整篇文档提取出的文本内容
        return resp


if __name__ == '__main__':
    # 直接运行本文件时的简单自测：
    # 以相对路径加载示例文档（需在 edu_document_loaders 目录下运行）
    docx_loader = OCRDOCLoader(filepath='samples/ocr_02.docx')
    # 调用 LangChain 基类的 load()，内部会调用 lazy_load() 并把结果收集为列表
    doc = docx_loader.load()
    # 打印解析结果
    print(doc)
