"""
=================================================================
Python操作Milvus —— 索引操作
=================================================================
创建/查看/删除向量索引和标量索引。
"""
from pymilvus import MilvusClient, DataType # 导包
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
# 集合名字
COLLECTION_NAME = 'movies'

# todo 删除索引
try:
    client.drop_index(
        collection_name=COLLECTION_NAME, # 集合名
        index_name='vector_index' # 要删除的索引名
    )
    client.drop_index(
        collection_name=COLLECTION_NAME, # 集合名
        index_name='title_index' # 要删除的索引名
    )
except Exception as e:
    print(e)

# todo 1. 创建索引对象
index_params = client.prepare_index_params()
# todo 2. 为向量字段 'vector' 添加索引
# 参数metric_type相似度的度量方式：
# 默认为L2为欧式距离
# COSINE为余弦相似度
# IP为内积（归一化后与COSINE结果一致，效率高）
index_params.add_index(
    field_name="vector", # 字段名
    index_name="vector_index", # 可选，为索引起个名字
    index_type="IVF_FLAT", # 索引类型
    metric_type="IP", # 度量方式
    params={"nlist": 128} # 索引参数nlist：聚类数量
)

# todo 为普通标量字段 'title' 添加索引
# 标量索引无需指定复杂的 index_type，Milvus 会自动适配，加速基于电影名称的过滤
index_params.add_index(
    field_name="title", # 字段名
    index_name="title_index", # 可选，为索引起个名字
    index_type="AUTOINDEX", # 标量索引类型
)

# todo 3. 创建索引：将定义好的索引参数应用到集合中
client.create_index(
    collection_name=COLLECTION_NAME, # 集合名
    index_params=index_params, # 索引参数对象
)
print("向量及标量字段的索引均已创建成功！")

# todo 查看集合中的索引
# 查看该集合下所有的索引名称列表
index_list = client.list_indexes(COLLECTION_NAME)
print("集合中的索引名称列表：", index_list)
# 预期输出示例：['id', 'vector', 'title_index', 'genre_index']

for index in index_list:
    print('*'*30)
    print("索引名称：", index)
    # 查看具体某个索引的详细配置
    index_detail = client.describe_index(
        collection_name=COLLECTION_NAME,
        index_name=index
    )
    print(f"\n【{index}】索引的详细配置：", index_detail)

"""
================== 运行输出 ==================
向量及标量字段的索引均已创建成功！
集合中的索引名称列表： ['vector_index', 'title_index']
******************************
索引名称： vector_index

【vector_index】索引的详细配置： {'nlist': '128', 'metric_type': 'IP', 'index_type': 'IVF_FLAT', 'field_name': 'vector', 'index_name': 'vector_index', 'total_rows': 0, 'indexed_rows': 0, 'pending_index_rows': 0, 'state': 'Finished'}
******************************
索引名称： title_index

【title_index】索引的详细配置： {'index_type': 'AUTOINDEX', 'field_name': 'title', 'index_name': 'title_index', 'total_rows': 0, 'indexed_rows': 0, 'pending_index_rows': 0, 'state': 'Finished'}
===============================================
"""