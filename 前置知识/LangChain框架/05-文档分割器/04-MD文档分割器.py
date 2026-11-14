"""
=================================================================
文档分割器 —— MD文档分割器（MarkdownTextSplitter）
=================================================================
MarkdownHeaderTextSplitter：按 # ## ### 标题层级切分，
    切分后的 Document 的 metadata 中会记录标题层级信息。

MarkdownTextSplitter：优先按标题层级切分，
    章节太长超过 chunk_size 时自动再切小，
    代码块、表格不会被从中间劈开。

项目在 process_documents.py 中使用 MarkdownTextSplitter 做 parent-child 分层切割。
"""

# ==============================
# 1. MarkdownHeaderTextSplitter —— 按标题层级切分
# ==============================
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 手动指定分割规则
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_text = "# Header 11\nSome text\n## Header 22\nMore text\n### Header 33\nEven more text\n#### Header 44\nSome more text"
docs = markdown_splitter.split_text(markdown_text)

print("【MarkdownHeaderTextSplitter】")
for doc in docs:
    print(f"  metadata={doc.metadata}")
    print(f"  content={doc.page_content}")
    print()


# ==============================
# 2. MarkdownTextSplitter —— 标题层级 + chunk_size 控制
# ==============================
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownTextSplitter

# 加载 Markdown 文件
file_path = "../data/haha.md"
loader = UnstructuredMarkdownLoader(file_path)
docs_list = loader.load()
print(f"\n【加载文档】{len(docs_list)} 个文档")

# 切割器
markdown_splitter = MarkdownTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
# todo 注意：是 split_documents()，不是 split_text()
docs = markdown_splitter.split_documents(docs_list)
print(f"【分割后】{len(docs)} 个文档")

for doc in docs[:5]:
    print('-' * 10)
    print(doc)
# 运行结果：
# 【MarkdownHeaderTextSplitter】
#   metadata={'Header 1': 'Header 1'}
#   content=Some text
#   metadata={'Header 1': 'Header 1', 'Header 2': 'Header 2'}
#   content=More text
#   metadata={'Header 1': 'Header 1', 'Header 2': 'Header 2', 'Header 3': 'Header 3'}
#   content=Even more text
#
# 【加载文档】1 个文档
# 【分割后】5 个文档
# ----------
# page_content='LangChain学习笔记\n\n什么是LangChain\n\nLangChain是一个用于开发大语言模型应用的框架。\n...'
# ----------
# page_content='Prompts: 提示词模板管理\n\nChains: 链式调用组合\n\n快速开始\n...'
# ...（按标题层级 + chunk_size=100 自动切分）
