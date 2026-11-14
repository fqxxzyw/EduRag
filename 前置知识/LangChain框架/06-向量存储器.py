"""
=================================================================
向量存储器（概念章）
=================================================================
Vector Stores 向量存储器的作用：支持 RAG 系统最核心的 "存和取" 流程。

    存（离线阶段）：文档 → 嵌入模型 → 向量 → 存入向量库 → 建索引
    取（在线阶段）：查询文本 → 嵌入模型 → 查询向量 → 相似度搜索 → 返回 Top-k 结果

=================================================================
LangChain 的 VectorStore 是 LangChain 给向量库套的 "统一外壳/接口"，
提供了统一接口：from_documents / similarity_search / as_retriever，
切换库几乎不用改检索代码。

=================================================================
★ 本项目（EduRAG）不使用 LangChain 的 VectorStore 封装！
★ 项目直接使用 pymilvus 操作 Milvus 向量数据库。
=================================================================

常见向量库对比：

    | 向量库       | 部署方式          | 特点                          | 适用场景                    |
    |-------------|------------------|-------------------------------|---------------------------|
    | InMemory    | 内存              | 零安装、最快                    | 临时测试、小数据调试          |
    | Chroma      | 本地文件           | 开箱即用、自动落盘               | 本地开发、Demo              |
    | FAISS       | 本地库文件         | 检索速度天花板                   | 百万级本地离线 RAG           |
    | Milvus      | Docker 服务        | 企业级、HNSW/IVF、分区、字段过滤  | 十万~亿级生产（★ 本项目使用）  |
    | Qdrant/Redis| Docker 服务        | 实时读写、混合检索               | 线上生产、接口服务 RAG        |

=================================================================
项目中 Milvus 相关操作代码位置：
    - 连接/集合/索引/插入/删除/查询：rag_qa/core/vector_store.py
    - 使用 pymilvus 的 MilvusClient 直接操作
    - 嵌入模型使用 BGEM3EmbeddingFunction（milvus_model.hybrid），不走 LangChain
=================================================================
"""

# 以下代码仅做概念演示，项目实际不使用 LangChain VectorStore。
# 如需运行，请安装：pip install langchain-chroma faiss-cpu

"""
import os
from langchain_community.embeddings import DashScopeEmbeddings

api_key = os.environ.get("AliApiKey")
emb_model = DashScopeEmbeddings(model="text-embedding-v4", dashscope_api_key=api_key)

# 1. 内存向量库（进程结束数据消失）
from langchain_core.vectorstores import InMemoryVectorStore
store = InMemoryVectorStore(embedding=emb_model)
store.add_texts(["我喜欢Python", "函数用def定义", "类用class定义"])
docs = store.similarity_search(query='怎么写函数', k=1)
print(docs[0].page_content)  # 函数用def定义

# 2. FAISS
from langchain_community.vectorstores import FAISS
store = FAISS.from_texts(texts=["我喜欢Python", "函数用def定义"], embedding=emb_model)
docs = store.similarity_search(query='怎么写函数', k=1)
print(docs[0].page_content)
"""
