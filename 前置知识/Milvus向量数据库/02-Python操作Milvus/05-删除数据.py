"""
=================================================================
Python操作Milvus —— 删除数据
=================================================================
按 ID / 过滤条件删除数据。
"""
from pymilvus import MilvusClient
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'
# todo 加载数据到内存！
client.load_collection(COLLECTION_NAME)
# 查看删除前数据条数
print(client.get_collection_stats(COLLECTION_NAME))

# 单条删除 基于id
client.delete(collection_name=COLLECTION_NAME,
              filter="id == 469269879804198124")
# 范围删除
client.delete(collection_name=COLLECTION_NAME,
              filter="id in [2, 3, 4]")
# 精确匹配字符串删除
client.delete(collection_name=COLLECTION_NAME,
              filter='genre == 1')
# 数值比较与组合条件删除，2010年之前且时长大于 150 分钟
client.delete(collection_name=COLLECTION_NAME,
              filter="release_year < 2010 && duration_minutes > 150")

# todo 刷新并重新加载，使修改对搜索立即生效
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(COLLECTION_NAME)

# todo 【查看删除后的数据情况可以去Attu页面查看】
# get_collection_stats 是查看集合中写操作的总数！
print(client.get_collection_stats(COLLECTION_NAME))

"""
================== 运行输出 ==================
{'row_count': 30}
{'row_count': 30}
===============================================
"""