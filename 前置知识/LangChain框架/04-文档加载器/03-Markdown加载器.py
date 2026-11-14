"""
=================================================================
文档加载器 —— Markdown 加载器
=================================================================
UnstructuredMarkdownLoader 加载 Markdown 文件。

需要安装：
    pip install unstructured langchain-unstructured markdown

注意：
    unstructured 包依赖 NLTK 包，NLTK 包依赖 averaged_perceptron_tagger_eng 模型包。
    虽然是中文，但 NLTK 仍需要这个英文词性标注模型。
    首次使用需执行：
        import nltk
        nltk.download('averaged_perceptron_tagger_eng')

项目在 rag_qa/edu_document_loaders/__init__.py 中 re-export 了 UnstructuredMarkdownLoader。
"""

# ==============================
# 方式1：UnstructuredMarkdownLoader（langchain_community）
# ==============================
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 默认 single 模式：所有内容合并为一个 Document
file_path = "../data/haha.md"
loader = UnstructuredMarkdownLoader(file_path)
docs = loader.load() # [Document]
print(f"【UnstructuredMarkdownLoader】加载了 {len(docs)} 个文档")
print(docs[0].page_content[:200])
# 运行结果：
# 【UnstructuredMarkdownLoader】加载了 1 个文档
# LangChain学习笔记
#
# 什么是LangChain
#
# LangChain是一个用于开发大语言模型应用的框架。
#
# 核心概念
#
# Models: 大语言模型接口
#
# Prompts: 提示词模板管理
#
# Chains: 链式调用组合
#
# 快速开始
#
# 安装
#
# pip install langchain
#
# 基本用法
#
# from langchain.chat_models import init_chat_mo


# ==============================
# 方式2：UnstructuredLoader（langchain_unstructured）
# ==============================
# from langchain_unstructured import UnstructuredLoader
#
# # mode="elements" 会把文档拆分成标题、正文、列表等独立元素
# # mode="single" (默认) 会把所有文本合并成一个大的 Document
# loader = UnstructuredLoader(file_path, mode="elements")
# docs = loader.load()
# print(f"\n【UnstructuredLoader】加载了 {len(docs)} 个文档")
# for doc in docs[:3]:
#     print(doc)
