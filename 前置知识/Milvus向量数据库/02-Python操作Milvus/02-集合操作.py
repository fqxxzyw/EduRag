"""
=================================================================
Python操作Milvus —— 集合操作
=================================================================
定义 Schema、创建集合、查看集合信息。
集合名：movies（电影数据）
"""
from pymilvus import MilvusClient, DataType # 导包
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
# 集合名字
COLLECTION_NAME = 'movies'

# todo 1. 定义schema。
schema = client.create_schema(enable_dynamic_field=True)
"""参数enable_dynamic_field=True：
    可以插入未提前定义的动态字段。
    动态字段一般以 JSON 格式存储，通常命名为 $meta。
    在插入数据时，所有未定义的字段及其值将被保存为键值对。
"""

# todo 2. schema添加并定义【字段】：主键、向量、分区标量、其他标量
# 所有字段都有description自定义备注参数，注意milvus lite不支持
# todo 2.1 主键字段
"""主键字段，必须有，只能有一个字段，值非空唯一
1. 主键类型只能是int64或varchar类型，varchar类型需规定长度1–65535
2. 主键如果自增，必须int64，且插入时不能带主键值
3. is_primary=True 主键声明
4. auto_id=False 是否自增
"""
schema.add_field(
    field_name="id", # 字段名
    datatype=DataType.INT64, # 类型
    is_primary=True, # 主键
    auto_id=True, # 自动id！
    description="主键，电影ID"
)
# todo 2.2 向量字段
"""向量字段，必须有，最多可以有4个字段，非空
1. 向量字段的类型：
密集向量：
    FLOAT_VECTOR：，4字节/维(32位)，精度最高、最通用
    FLOAT16_VECTOR：2字节/维(16位)，节省一半空间且小数精度高
    BFLOAT16_VECTOR：2字节/维(16位)，节省一半空间且数值范围极大，小数范围少
    INT8_VECTOR：1字节/维(8位)，正负整数数（-128~127）向量
稀疏向量：SPARSE_FLOAT_VECTOR，仅存非零元素和非零元素的索引，维度极高
二进制向量：BINARY_VECTOR，1比特/维(0或1)，存储极度紧凑。
2. dim 维度数量，必须和向量（嵌入）模型的维度一致！
    text-embedding-v1 1536
    text-embedding-v4 默认1024
    mxbai-embed-large 1024
    m3e-base 768
"""
schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=1024,
    description="向量字段，嵌入向量"
)
# todo 2.3 分区标量（分区键）
# 标量：原始数据本身、非向量
"""分区标量（分区键），可选，只能有一个字段，非空
1. 分区标量的类型只有2个选择：INT64或VARCHAR
    VARCHAR必须加max_length参数规定字符最大数量
2. is_partition_key=True 分区标量声明
"""
schema.add_field(
    field_name="genre",
    datatype=DataType.INT64,
    is_partition_key=True, # 分区标量声明
    description="分区标量（分区键），电影类别"
)
# todo 2.4 普通标量
# 标量：原始数据本身、非向量
"""普通标量，可选，任意个，默认非空
1. 普通标量datatype类型：
    INT64 整数
    VARCHAR 需要max_length参数规定字符长度
    ARRAY 数组类型list，存同类型元素、最多4096个，元素类型不支持JSON、ARRAY
    FLOAT 小数
    BOOL True、False
    JSON 字典
2. nullable默认为False
"""
# 标量字段（文档内容）
schema.add_field(field_name="title",
                 datatype=DataType.VARCHAR,
                 max_length=200,
                 description="普通标量，电影名称，向量内容来源1")
schema.add_field(field_name="plot",
                 datatype=DataType.VARCHAR,
                 max_length=500,
                 description="普通标量，电影介绍，向量内容来源2")
schema.add_field(field_name="duration_minutes",
                 datatype=DataType.INT32,
                 nullable=True,
                 description="普通标量，电影时长")
schema.add_field(field_name="release_year",
                 datatype=DataType.INT32,
                 nullable=True,
                 description="普通标量，上映年份")

# todo 删除集合
if client.has_collection(COLLECTION_NAME): # 如果存在
    client.drop_collection(COLLECTION_NAME) # 就删除

# todo 3. 创建集合 必须先有schema结构定义！
client.create_collection(
    collection_name=COLLECTION_NAME, # 集合名字
    schema=schema,          # 传入上面定义的完整结构定义
    num_partitions=7,      # 物理分区数，配合分区键使用
    description="自定义描述信息"
)
"""参数num_partitions分区数，配合分区键使用：
    分区键（分区标量）= 按什么规则进行分区
    分区参数num_partitions = 分区数，总共创建多少个区
    哈希计算（取余）：分区编号 = 分区标量  %  分区数
比如：
    分区参数num_partitions=7 表示分7个区
    对分区键（tenant_id=1001,1002,1003...1008,1009...）做哈希计算
        1001 % 7 = 2 ，在2号分区
        1008 % 7 = 0 ，在0号分区
        1009 % 7 = 1 ，在1号分区
"""

# todo 4. 查看集合
# 查看所有集合名字
print(client.list_collections())
# 集合是否存在
print(client.has_collection(COLLECTION_NAME))
# 看集合的完整结构：schema、字段、维度、分区信息
print(client.describe_collection(COLLECTION_NAME))
# 查看增改操作的数据总数
print(client.get_collection_stats(COLLECTION_NAME))
# 查看集合的字段信息
for field in client.describe_collection(COLLECTION_NAME)['fields']:
    print("-" * 30)
    print(f"字段名: {field['name']}")
    print(f"  类型: {DataType(field['type']).name}")
    print(f"  描述: {field['description']}")
    print(f"  是否为主键: {field.get('is_primary', False)}")
    print(f"  是否为分区标量: {field.get('is_partition_key', False)}")
    # 如果是向量字段，提取并打印它的维度(dim)
    if 'dim' in field.get('params', {}):
        print(f"  向量维度: {field['params']['dim']}")

"""
================== 运行输出 ==================
['movies']
True
{'collection_name': 'movies', 'auto_id': True, 'num_shards': 1, ...}
{'row_count': 0}
------------------------------
字段名: id
  类型: INT64
  描述: 主键，电影ID
  是否为主键: True
  是否为分区标量: False
------------------------------
字段名: vector
  类型: FLOAT_VECTOR
  描述: 向量字段，嵌入向量
  是否为主键: False
  是否为分区标量: False
  向量维度: 1024
------------------------------
字段名: genre
  类型: INT64
  描述: 分区标量（分区键），电影类别
  是否为主键: False
  是否为分区标量: True
------------------------------
字段名: title
  类型: VARCHAR
  描述: 普通标量，电影名称，向量内容来源1
  是否为主键: False
  是否为分区标量: False
------------------------------
字段名: plot
  类型: VARCHAR
  描述: 普通标量，电影介绍，向量内容来源2
  是否为主键: False
  是否为分区标量: False
------------------------------
字段名: duration_minutes
  类型: INT32
  描述: 普通标量，电影时长
  是否为主键: False
  是否为分区标量: False
------------------------------
字段名: release_year
  类型: INT32
  描述: 普通标量，上映年份
  是否为主键: False
  是否为分区标量: False
===============================================
"""