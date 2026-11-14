"""
=================================================================
Python操作Milvus —— 过滤搜索
=================================================================
向量搜索 + 标量过滤条件组合。
"""
import os
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings

api_key = os.environ.get("AliApiKey")
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'
client.load_collection(COLLECTION_NAME)

model = DashScopeEmbeddings(dashscope_api_key=api_key, model="text-embedding-v4")

# ===================== 3. 过滤搜索 =====================
# 需求：只在【喜剧片(genre=2)】中，搜索语义为"穿越回过去改变人生"的电影
query_text = "穿越回过去改变人生"
query_vector = model.embed_documents([query_text])[0]

res = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    limit=3,
    metric_type="IP",
    filter="genre == 2",  # 核心：标量过滤条件
    output_fields=["title", "genre", "plot", "duration_minutes"]
)

print("\n🔍 过滤搜索（喜剧片+语义检索）结果：")
for hits in res:
    for hit in hits:
        print(f"相似度分数：{hit['distance']:.4f}")
        print(f"电影名称：{hit['entity']['title']}")
        print(f"分类：{hit['entity']['genre']}（喜剧片）")
        print(f"时长：{hit['entity']['duration_minutes']}分钟\n")

"""
================== 运行输出 ==================

🔍 过滤搜索（喜剧片+语义检索）结果：
相似度分数：0.5214
电影名称：夏洛特烦恼
分类：2（喜剧片）
时长：104分钟

相似度分数：0.5065
电影名称：你好，李焕英
分类：2（喜剧片）
时长：128分钟

相似度分数：0.4621
电影名称：飞驰人生
分类：2（喜剧片）
时长：98分钟
===============================================
"""