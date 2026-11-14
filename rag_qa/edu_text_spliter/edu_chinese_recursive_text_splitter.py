"""
中文递归文本分割器：ChineseRecursiveTextSplitter

背景与目的：
    LangChain 自带的 RecursiveCharacterTextSplitter 默认按 "\\n\\n"、"\\n"、" "、"" 这类
    英文语料常见的分隔符递归切分文本，但中文文档几乎没有空格分词，
    直接使用会把一整段中文硬切成固定长度，容易切断句子、丢失语义。
    因此这里继承 RecursiveCharacterTextSplitter，做两处定制：
        1. 提供一套"从大到小"的中文/英文标点分隔符（段落 → 换行 → 句号 → 分号 → 逗号）；
        2. 重写 _split_text 方法，支持用正则表达式作为分隔符，并把分隔符保留在切分片段的末尾。

核心思路（递归切分）：
    先用最"粗"的分隔符（如段落、换行）切分文本；
    如果某一段仍然超过 chunk_size，就换用更"细"的分隔符（如句号、逗号）继续切；
    直到片段长度满足要求，或者没有更细的分隔符可用为止。
    最后再把相邻的短片段合并成不超过 chunk_size、且带有 chunk_overlap 重叠的文本块。
"""

import re  # 正则表达式模块：用 re.split 切分文本、用 re.search 判断分隔符是否存在
from typing import List, Optional, Any  # 类型注解：List 列表、Optional 可选参数、Any 任意类型
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 继承 LangChain 的递归字符分割器


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """用正则表达式切分文本，并（可选）把分隔符保留在切分结果的"末尾"。

    与 Python 内置 re.split 的区别：
        re.split 默认会把分隔符丢掉，且无法控制分隔符归属于前一段还是后一段。
        本函数通过捕获组 + 切片配对的方式，把分隔符"拼接回"它前面的文本片段末尾，
        使每个片段形如 "句子内容。"，这样后续合并后的文本能保持标点完整。

    Args:
        text: 待切分的原始文本。
        separator: 分隔符（正则表达式的字符串形式）。
        keep_separator: 是否在结果中保留分隔符。

    Returns:
        切分后的非空文本片段列表。
    """
    # 如果传入了有效的分隔符，则按正则切分
    if separator:
        if keep_separator:
            # 关键点：正则里加括号形成"捕获组"后，re.split 的返回结果中会保留分隔符本身。
            # 例如 re.split("(。)", "a。b。c") -> ['a', '。', 'b', '。', 'c']
            _splits = re.split(f"({separator})", text)
            # _splits[0::2] 是每段正文，_splits[1::2] 是紧随其后的分隔符；
            # zip 后两两配对再 join，就把分隔符拼到了前一段正文的末尾：
            #   ('a', '。') -> 'a。'，('b', '。') -> 'b。'
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 若切分结果长度为奇数，说明最后还有一个不带分隔符的正文片段（如 'c'），需要补上
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # 下面这行是另一种设计（把分隔符放在下一段的开头）的写法，当前未启用：
            # splits = [_splits[0]] + splits
        else:
            # 不保留分隔符：直接丢弃所有匹配到的分隔符
            splits = re.split(separator, text)
    else:
        # 没有可用的分隔符时，退化为"按单个字符"切分（兜底策略）
        splits = list(text)
    # 过滤掉空字符串，避免产生空片段
    return [s for s in splits if s != ""]


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """面向中文文本的递归字符分割器。

    继承 LangChain 的 RecursiveCharacterTextSplitter：
        - 复用其 split_text / split_documents 等对外接口与参数校验；
        - 复用其 _merge_splits 完成"短片段合并 + 重叠"的逻辑；
        - 只重写分隔符默认值以及 _split_text 的切分策略。
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            separators: 分隔符列表，按"从粗到细"的顺序排列；为 None 时使用下面的中文默认列表。
            keep_separator: 是否保留分隔符；True 时把分隔符附加在前一个片段末尾。
            is_separator_regex: 是否把 separators 中的每一项当作正则表达式（默认 True）。
            **kwargs: 其余参数（如 chunk_size、chunk_overlap、length_function 等）透传给基类。
        """
        # 调用父类构造：把 keep_separator 和 chunk_size、chunk_overlap 等参数交给基类保存
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 默认分隔符列表：从"语义边界最明显"到"最细碎"依次排列，切分时优先使用靠前的分隔符
        self._separators = separators or [
            "\n\n",  # 1. 段落分隔：空行（最优先，尽量保持段落完整）
            "\n",  # 2. 换行：段内换行
            "。|！|？",  # 3. 中文句子结束标点：句号、感叹号、问号
            "\.\s|\!\s|\?\s",  # 4. 英文句子结束标点：点/叹号/问号 + 空白（要求后跟空白，避免切断 19.5、U.S. 这类内容）
            "；|;\s",  # 5. 中文分号，或英文分号 + 空白
            "，|,\s"  # 6. 中文逗号，或英文逗号 + 空白（最细，作为最后的分句手段）
        ]
        # 记录分隔符是否为正则表达式，后续切分时据此决定要不要 re.escape 转义
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks.

        递归切分的主流程：
            1. 从当前分隔符列表中选出第一个能在文本里匹配到的分隔符；
            2. 用该分隔符切分文本；
            3. 遍历切分结果：短片段先攒起来，等遇到过长片段时再统一合并；
            4. 过长的片段若无更细的分隔符可用，则原样保留，否则递归继续切分；
            5. 最后清理多余空行并丢弃空片段。

        Args:
            text: 待切分的文本。
            separators: 当前可用的分隔符列表（从粗到细）。

        Returns:
            切分并合并后的文本块列表。
        """
        # 存放最终切分结果的列表
        final_chunks = []
        # Get appropriate separator to use
        # 默认取列表最后一个（最细的）分隔符作为兜底；下面循环会尽量找到更粗的可用分隔符
        separator = separators[-1]
        # 记录"当前分隔符之后"的更细分隔符，供递归切分时使用
        new_separators = []
        # 从前到后遍历分隔符（顺序即"由粗到细"），选第一个能被匹配到的
        for i, _s in enumerate(separators):
            # 若分隔符不是正则形式，先转义，避免 "." 等特殊字符被当成通配符
            _separator = _s if self._is_separator_regex else re.escape(_s)
            # 空字符串是特殊标记：表示"没有分隔符可用，按字符切分"，直接采用并结束选择
            if _s == "":
                separator = _s
                break
            # 若当前分隔符在文本中确实存在，就选它，并把后面更细的分隔符留作递归使用
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        # 确定真正用于切分的正则表达式（非正则模式需要转义）
        _separator = separator if self._is_separator_regex else re.escape(separator)
        # 调用上面的辅助函数执行切分；分隔符会保留在每段末尾（keep_separator=True 时）
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        # 用于暂存"长度达标"的小片段，稍后由 _merge_splits 合并成大块
        _good_splits = []
        # 合并时使用的连接符：保留分隔符时，分隔符已经粘在片段末尾，故用空串连接；
        # 不保留分隔符时，则重新用当前分隔符连接各片段
        _separator = "" if self._keep_separator else separator
        # 遍历本次切分出来的每个片段
        for s in splits:
            # 片段长度小于 chunk_size：先攒起来，等待与相邻片段合并
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                # 遇到超长片段：先把之前攒下的短片段合并输出，避免顺序错乱
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                # 如果已经没有更细的分隔符，就无法再切，直接把该超长片段原样输出
                if not new_separators:
                    final_chunks.append(s)
                else:
                    # 否则用更细的分隔符递归切分这个超长片段
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        # 循环结束后，处理剩余尚未合并的短片段
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 收尾清理：
        #   - re.sub(r"\n{2,}", "\n", ...)：把两个及以上连续换行压缩成一个，去掉多余空行；
        #   - chunk.strip()：去掉每个文本块首尾空白；
        #   - if chunk.strip() != ""：过滤掉只含空白字符的空块。
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


if __name__ == "__main__":
    # 简单自测：创建一个中文递归分割器
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,  # 保留标点：让切出来的每句话仍以标点结尾，语义更完整
        is_separator_regex=True,  # 分隔符列表是正则表达式
        chunk_size=150,  # 每个文本块的最大长度（字符数）
        chunk_overlap=10  # 相邻文本块之间重叠 10 个字符，避免上下文被切断后丢失信息
    )
    # 测试文本列表（这里只有一段较长的中文示例文本）
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    ]
    # 逐个处理测试文本
    for inum, text in enumerate(ls):
        # 打印当前文本的序号，便于区分多段测试文本的结果
        print(inum)
        # 执行切分，得到文本块列表
        chunks = text_splitter.split_text(text)
        # 逐块打印切分结果，便于观察切分位置是否符合语义
        for chunk in chunks:
            print(chunk)
