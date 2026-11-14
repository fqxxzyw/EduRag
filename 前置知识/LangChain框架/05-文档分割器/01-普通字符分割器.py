"""
=================================================================
文档分割器 —— 普通字符分割器（CharacterTextSplitter）
=================================================================
CharacterTextSplitter 是最简单的分割器，仅根据指定的单个分隔符来分割文本。
优势：速度极快，简单可控。
劣势：破坏语义，完全不关心文本的语义结构。

项目中 AliTextSplitter（edu_model_text_spliter.py）继承了 CharacterTextSplitter，
在其基础上增加了 BERT 语义段落分割能力。
"""
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

# 创建分割器
text_splitter = CharacterTextSplitter(
    separator=" ",        # 分隔符：空格符
    chunk_size=5,         # 每个文本块大小：最多 5 个字符
    chunk_overlap=1       # 块之间重叠 1 个字符
)

# ==============================
# 1. split_text —— 分割字符串 → 字符串列表
# ==============================
result1 = text_splitter.split_text("a b c d e f")
print(f"【split_text】{result1}")  # ['a b c', 'c d e', 'e f']

# ==============================
# 2. create_documents —— 创建 Document 列表
# ==============================
result2 = text_splitter.create_documents(["a b c d e f", "e f g h"])
print(f"\n【create_documents】")
for doc in result2:
    print(f"  {doc}")

# ==============================
# 3. split_documents —— 分割 Document 列表
# ==============================
result3 = text_splitter.split_documents([
    Document(page_content="a b c d e f", metadata={"id": "1"})
])
print(f"\n【split_documents】")
for doc in result3:
    print(f"  {doc}")

# 运行结果：
# 【split_text】['a b c', 'c d e', 'e f']
#
# 【create_documents】
#   page_content='a b c'
#   page_content='c d e'
#   page_content='e f'
#   page_content='e f g'
#   page_content='g h'
#
# 【split_documents】
#   page_content='a b c' metadata={'id': '1'}
#   page_content='c d e' metadata={'id': '1'}
#   page_content='e f' metadata={'id': '1'}
