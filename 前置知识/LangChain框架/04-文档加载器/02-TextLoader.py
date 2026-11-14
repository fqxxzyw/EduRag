"""
=================================================================
文档加载器 —— TextLoader
=================================================================
TextLoader 读取纯文本文件，全部内容放入一个 Document 对象中。
项目在 rag_qa/edu_document_loaders/__init__.py 中 re-export 了 TextLoader。
"""
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    # 文件路径（根据实际数据文件位置调整）
    file_path="../data/Python基础语法.txt",
    encoding='utf8'
)

# load() 全量加载 → 列表 [Document]
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")
print(f"文档类型：{type(docs)}")  # <class 'list'>
print(f"第一个文档：{docs[0]}")
print(f"\n文档内容（前200字）：")
print(docs[0].page_content[:200])
print(type(docs[0]))
# 运行结果：
# 加载了 1 个文档
# 文档类型：<class 'list'>
# 第一个文档：page_content='Python基础语法\n\n一、变量和数据类型\n...' metadata={'source': '../data/Python基础语法.txt'}
#
# 文档内容（前200字）：
# Python基础语法
#
# 一、变量和数据类型
#
# Python是一种动态类型语言，变量不需要声明类型。
# x = 10          # 整数
# name = "张三"    # 字符串
# pi = 3.14       # 浮点数
# is_valid = True  # 布尔值
#
# 二、条件判断
#
# if x > 0:
#     print("正数")
# elif x == 0:
#     print("零")
