"""
=================================================================
文档分割器 —— 语义文档分割器（SemanticChunker）
=================================================================
使用语义理解来分割文本，需要文本嵌入模型。
基于语义相似性分割，能更好地保持语义完整性。
计算成本较高，处理大量文本时效率较低。

注意：SemanticChunker 来自 langchain_experimental 包（实验性质），
项目中并未直接使用此分割器，这里仅做了解。
"""
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

api_key = os.environ.get("AliApiKey")

# 嵌入模型
model = DashScopeEmbeddings(
    model='qwen3.7-text-embedding',
    dashscope_api_key=api_key
)

# 语义分割器
text_splitter = SemanticChunker(
    embeddings=model,
    # 分段阈值类型：百分位（最通用、最好用）
    breakpoint_threshold_type='percentile',
    # 百分位数值：
    #   50  → 中等切割，切得较碎
    #   90~95 → 尽量不切，大块到底（中文文档推荐）
    breakpoint_threshold_amount=95.0,
    # 句子拆分正则：同时识别中英文终结符
    sentence_split_regex=r'(?<=[。！？.!?])\s*',
    # 每个 chunk 至少的字符数
    min_chunk_size=10
)

# 测试文本
text = """
人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
它们可以帮助人们进行写作、代码生成、甚至是科研探索。
相比之下，新能源的发展同样重要。
电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。
"""

# 分割
docs = text_splitter.split_text(text)
for i, d in enumerate(docs):
    print(f"------ Chunk {i+1} ------")
    print(d.strip())
    print()
# 运行结果：
# ------ Chunk 1 ------
# 人工智能正在快速发展，尤其是大语言模型的应用，正在改变人类的工作方式。
# 它们可以帮助人们进行写作、代码生成、甚至是科研探索。
#
# ------ Chunk 2 ------
# 相比之下，新能源的发展同样重要。
#
# ------ Chunk 3 ------
# 电动车和太阳能正在逐渐替代传统能源，减少碳排放，对全球环境保护至关重要。


"""
=================================================================
breakpoint_threshold_type 参数详解：
    1. percentile 百分位（默认首选，通用）
       amount=95 → 只在差距最大的前 5% 位置切分，块偏大
       amount=50 → 前 50% 都切，切得很碎

    2. standard_deviation 标准差
       amount=3 → 差距高出平均值+3倍波动值才切
       适用：论文、技术手册

    3. interquartile 四分位
       amount=1.5 → 只抓突兀跳话题的断点
       适用：长篇小说、散文

    4. gradient 梯度
       不看差距绝对值，差距突然猛地变大就切
       适用：演讲稿、访谈

选型建议：
    日常知识库     → percentile 95
    专业技术文档   → standard_deviation 2~3
    长篇网文小说   → interquartile 1.5
    采访/演讲稿    → gradient
"""
