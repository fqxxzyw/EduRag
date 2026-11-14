"""
=================================================================
Python操作Milvus —— 修改数据
=================================================================
使用 upsert 实现数据的更新。
"""


# 初始化 Milvus 客户端和 Embedding 模型
import os
api_key=os.environ.get("AliApiKey")
from pymilvus import MilvusClient
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")

# 阿里云的text-embedding-v4模型，自动进行归一化
# from langchain_community.embeddings import DashScopeEmbeddings
# model = DashScopeEmbeddings(
#     dashscope_api_key=api_key,
#     model="text-embedding-v4" # 默认1024维
# )

COLLECTION_NAME = 'movies'
# todo 加载数据到内存！
client.load_collection(COLLECTION_NAME)
# 要修改的标题
target_title = "西虹市首富"
# 查询目标数据
query_res = client.query(
    collection_name=COLLECTION_NAME,
    filter=f'title == "{target_title}"'
)
old_record = query_res[0] # 完整数据
print(old_record['release_year']) # 要被修改的原数据

# todo 构造更新后的全量数据
update_data = [{
    "id": old_record["id"],  # 必须带上主键 ID
    # 复用原向量（若标题或剧情变了，就需重新 embed）
    "vector": old_record["vector"],
    "genre": old_record["genre"],
    "title": old_record["title"],
    "duration_minutes": old_record["duration_minutes"],
    "release_year": 2015,  # todo 将年份从 2018 修改为 2015
    "plot": old_record["plot"]
}]

# todo 执行 Upsert (存在则更新，不存在则插入)
upsert_res = client.upsert(collection_name=COLLECTION_NAME, data=update_data)
print(f"✏️ 成功修改数据，影响记录数: {upsert_res.get('upsert_count', 0)}")

# todo 刷新并重新加载，使修改对搜索立即生效
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(COLLECTION_NAME)

# 查看修改后的数据
query_res = client.query(
    collection_name=COLLECTION_NAME,
    filter=f'title == "{target_title}"'
)
new_record = query_res[0] # 完整数据
print(new_record['release_year'])

"""
================== 运行输出 ==================
2018
✏️ 成功修改数据，影响记录数: 1
2015
===============================================
"""