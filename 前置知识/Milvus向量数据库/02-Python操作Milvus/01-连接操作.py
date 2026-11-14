"""
=================================================================
Python操作Milvus —— 连接操作
=================================================================
连接 Milvus 服务，数据库的增删查用。
"""
from pymilvus import MilvusClient # 导包

# 连接对象
client = MilvusClient(uri="http://127.0.0.1:19530")

# 所有数据库的名字
dbnames_list = client.list_databases()
if "edurag" not in dbnames_list: # 判断数据库是否存在
    # 创建数据库 最多能建64个，Milvus Lite不支持
    client.create_database('edurag')

# 使用数据库
client.use_database('edurag')
print(client.list_databases())

# 删除数据库（演示，实际使用时注释掉）
# client.drop_database('edurag')
print(client.list_databases())

# 查看当前库所有集合名字
print(client.list_collections())

"""
================== 运行输出 ==================
['default', 'itcast', 'edurag']
['default', 'itcast', 'edurag']
['movies']
===============================================
"""