"""
=================================================================
文档加载器 —— BaseLoader 自定义加载器 + Document 对象
=================================================================
所有文档加载器都继承 BaseLoader，返回 Document 对象。
Document 是 LangChain 内文档的统一载体。

项目中自定义了 4 个 OCR 加载器（继承 BaseLoader）：
    - OCRDOCLoader   → rag_qa/edu_document_loaders/edu_docloader.py
    - OCRPDFLoader   → rag_qa/edu_document_loaders/edu_pdfloader.py
    - OCRIMGLoader   → rag_qa/edu_document_loaders/edu_imgloader.py
    - OCRPPTLoader   → rag_qa/edu_document_loaders/edu_pptloader.py
每个加载器实现 lazy_load() 方法，yield Document 对象。
"""
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

# ==============================
# 1. Document 对象
# ==============================
document = Document(
    page_content="文档里的内容",
    metadata={"source": "正文的身份证、标签、备注，不参与模型回答，只用来标记、筛选、溯源"}
)
print(f"【Document】page_content={document.page_content}")
print(f"【Document】metadata={document.metadata}")


# ==============================
# 2. BaseLoader 自定义加载器示例
# ==============================
class MyCustomLoader(BaseLoader):
    """
    自定义加载器示例：继承 BaseLoader，实现 lazy_load 方法。
    项目中 4 个 OCR 加载器就是这种模式。
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self):
        """
        延迟加载：yield Document 对象，避免大文件内存溢出。
        实际项目中这里会调用 OCR 服务识别文档内容。
        """
        yield Document(
            page_content=f"这是从 {self.file_path} 加载的内容",
            metadata={"source": self.file_path}
        )

    def load(self):
        """
        一次性加载全部文档，内部调用 lazy_load。
        """
        return list(self.lazy_load())  # 懒加载


# 使用自定义加载器
loader = MyCustomLoader("test.txt")

# load() 全量加载 → 列表 [Document, ...]
docs = loader.load()
print(f"\n【load】加载了 {len(docs)} 个文档")
print(docs[0].page_content)

# lazy_load() 懒加载 → 迭代器
print("\n【lazy_load】")
for doc in loader.lazy_load():
    print(doc.page_content)

# 运行结果：
# 【Document】page_content=文档里的内容
# 【Document】metadata={'source': '正文的身份证、标签、备注，不参与模型回答，只用来标记、筛选、溯源'}
#
# 【load】加载了 1 个文档
# 这是从 test.txt 加载的内容
#
# 【lazy_load】
# 这是从 test.txt 加载的内容
