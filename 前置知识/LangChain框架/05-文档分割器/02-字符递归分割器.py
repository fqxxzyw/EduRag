"""
=================================================================
文档分割器 —— 字符递归分割器（RecursiveCharacterTextSplitter）
=================================================================
递归字符分割器，按分隔符列表从左到右依次递归分割，直到满足 chunk_size。
优先尊重文本的自然结构（段落→句子→词），只有大块依然超长时才降级分割。

这是 RAG 系统中的绝对主力分割器！

项目中 ChineseRecursiveTextSplitter（edu_chinese_recursive_text_splitter.py）
继承了 RecursiveCharacterTextSplitter，使用中文标点作为分隔符。
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# 1. 加载文档
# ==============================
loader = TextLoader(
    file_path="../data/Python基础语法.txt",
    encoding='utf8'
)
doc = loader.load()

# ==============================
# 2. 创建递归分割器
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 分段的最大字符数
    chunk_overlap=50,       # 分段之间允许重叠字符数
    # 文本自然段落分隔的依据符号（按优先级从左到右依次尝试）
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    length_function=len,    # 统计字符的依据函数
)

# ==============================
# 3. 分割文档
# ==============================
split_docs = splitter.split_documents(doc) # doc ==> [Document]
print(f"分割后文档数：{len(split_docs)}")
print(f"类型：{type(split_docs)}")

for i, doc in enumerate(split_docs[:3]):  # 只打印前3个
    print(f"\n{'='*20} 第 {i+1} 块 {'='*20}")
    print(doc.page_content[:100])
    print(f"... (共 {len(doc.page_content)} 字符)")
# 运行结果：
# 分割后文档数：2
# 类型：<class 'list'>
#
# ==================== 第 1 块 ====================
# Python基础语法
#
# 一、变量和数据类型
#
# Python是一种动态类型语言，变量不需要声明类型。
# x = 10          # 整数
# name = "张三"    # 字符串
# pi = 3.1
# ... (共 479 字符)
#
# ==================== 第 2 块 ====================
# for fruit in fruits:
#     print(fruit)
#
# print(person["name"])
# ... (共 60 字符)


"""
=================================================================
运行流程：
    1. 首先尝试用第一个分隔符（如 "\\n\\n"）分割文本
    2. 如果分割后的块仍然过大，则用下一个分隔符继续分割
    3. 重复此过程，直到达到 chunk_size 或用完所有分隔符

核心参数：
    chunk_size     块大小，太大会超出模型输入限制，太小丢失上下文语义
    chunk_overlap  重叠度，建议设为 chunk_size 的 10%~20%，保持语义连贯
"""
