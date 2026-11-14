"""
=================================================================
Python操作Milvus —— 计数及释放内存
=================================================================
统计数据条数、释放集合内存。
"""
from pymilvus import MilvusClient

client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'
# 加载集合到内存
client.load_collection(COLLECTION_NAME)

# ===================== 计数及释放内存 =====================
# 1. 统计满足条件的数据总数
count = client.query(
    collection_name=COLLECTION_NAME,
    filter="genre == 2",
    output_fields=["count(*)"]  # 统计数量
)
print(f"\n喜剧片总条数：{count[0]['count(*)']}")

# 2. 释放集合内存（查询完成后释放）
client.release_collection(COLLECTION_NAME)
print("✅ 集合已释放内存")

"""
================== 运行输出 ==================

喜剧片总条数：10
✅ 集合已释放内存
===============================================
"""