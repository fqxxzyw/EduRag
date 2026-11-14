"""
=================================================================
Python操作Milvus —— 向量检索
=================================================================
基于语义相似度的向量搜索。
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

# ===================== 2. 向量搜索 =====================
# 2.1 构造查询文本（语义：人类穿越虫洞寻找新家园）
query_text = "人类穿越虫洞寻找适合居住的新星球"
# 生成查询向量
query_vector = model.embed_documents([query_text])[0]

# 2.2 执行向量搜索
res = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],  # 必须传列表格式
    limit=3,  # 返回最相似的3条
    metric_type="IP",  # 与索引一致：内积相似度
    output_fields=["title", "genre", "plot", "release_year"]  # 返回字段
)

print("\n🔍 向量语义搜索结果：")
# 解析结果（外层是批次，内层是单条数据）
for hits in res:
    for hit in hits:
        print(f"相似度分数：{hit['distance']:.4f}")
        print(f"电影名称：{hit['entity']['title']}")
        print(f"电影分类：{hit['entity']['genre']}")
        print(f"剧情简介：{hit['entity']['plot']}\n")

"""
================== 运行输出 ==================
🔍 向量语义搜索结果：
相似度分数：0.5132
电影名称：独行月球
电影分类：2
剧情简介：人类为抵御小行星撞击地球部署了月盾计划，不料陨石提前来袭全员紧急撤离，维修工独孤月因为意外错过了撤退通知，成为了唯一被遗落在月球上的人类。

相似度分数：0.4704
电影名称：复仇者联盟4
电影分类：3
剧情简介：灭霸的一个响指让宇宙一半的生命灰飞烟灭，幸存的超级英雄们集结在一起，利用量子领域进行时间旅行，试图收集无限宝石逆转这场毁灭性的灾难。

相似度分数：0.4640
电影名称：疯狂的外星人
电影分类：2
剧情简介：游乐园耍猴人耿浩意外救下了一个从天而降的外星人，面对这个拥有超能力的天外来客，他和好兄弟大飞将其当成猴子驯养，引发了一系列鸡飞狗跳的闹剧。
"""