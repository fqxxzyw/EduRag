"""
=================================================================
Python操作Milvus —— 分区操作
=================================================================
创建分区、向分区插入数据、分区内搜索、删除分区。
"""
import os
api_key=os.environ.get("AliApiKey")
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, WeightedRanker
# 阿里云的text-embedding-v4模型，自动进行归一化
model = DashScopeEmbeddings(
    dashscope_api_key=api_key,
    model="text-embedding-v4" # 默认1024维
)
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies' # 集合名称
client.load_collection(COLLECTION_NAME) # 加载集合
NEW_PARTITION_NAME = 'partition_new' # 新建的分区名称

# todo 1. 查看分区列表
partition_list = client.list_partitions(COLLECTION_NAME)
print("当前分区列表:", partition_list)

"""建表的时候可以通过指定分区字段+分区数量的方式来创建分区
1. 没有指定分区数量（num_partitions=16），会自动建立16个分区
2. 但有分区标量（is_partition_key=True），依旧会依旧分区键把数据放到不同的分区中
3. 既不设分区数（num_partitions=16），也没分区键（is_partition_key=True），就只有一个分区_default，所有数据都到其中。
4. 注意：
4.1 设置分区键（is_partition_key=True）的集合中，禁止手动创建或删除物理分区。
4.2 融合检索时不能指定分区！只能用expr参数接收带分区键的过滤条件，从而代替分区功能（前提是已经指定分区键）
"""

# todo 2. 在指定的集合中创建一个新的分区
if not client.has_partition(COLLECTION_NAME, NEW_PARTITION_NAME):
    client.create_partition(
        collection_name=COLLECTION_NAME,
        partition_name=NEW_PARTITION_NAME
    )

# 构造数据
data_list = [
    {"title": "哪吒之魔童降世", "genre": 4, "duration_minutes": 110, "release_year": 2019,
     "plot": "灵珠与魔丸被申公豹暗中调换，魔丸转世的哪吒自幼受尽陈塘关百姓排挤与偏见，在父母与太乙真人呵护下坚守本心，以我命由我不由天的信念逆天抗劫，最终以身献祭救下全城百姓。"},
    {"title": "西游记之大圣归来", "genre": 4, "duration_minutes": 89, "release_year": 2015,
     "plot": "被封印五行山五百年的孙悟空意外被小和尚江流儿解除禁锢，消沉落魄的齐天大圣在结伴同行、对抗山妖的冒险途中，重拾斗志与一身神通，完成自我救赎。"},
    {"title": "深海", "genre": 4, "duration_minutes": 112, "release_year": 2023,
     "plot": "深陷原生家庭痛苦的少女参宿误入奇幻深海幻境，偶遇深海大饭店老板南河，二人结伴奔赴神秘深海之眼，在光怪陆离的海底冒险中彼此治愈，揭开藏在梦境背后关于成长与救赎的真相。"},
    {"title": "白蛇：缘起", "genre": 4, "duration_minutes": 99, "release_year": 2019,
     "plot": "白蛇小白意外失忆偶遇善良的捕蛇少年阿宣，一人一妖跨越种族相恋，为躲避道士追杀结伴闯荡，在一次次生死磨难中相守，谱写白蛇传前世凄美浪漫的爱恋故事。"},
    {"title": "姜子牙", "genre": 4, "duration_minutes": 110, "release_year": 2020,
     "plot": "封神大战后姜子牙因不忍斩杀无辜狐妖被贬下凡，为找寻真相踏上幽都旅途，一路探寻宿命与天道的真相，最终打破固有规则，寻得属于自己的苍生大道。"},
    {"title": "大鱼海棠", "genre": 4, "duration_minutes": 105, "release_year": 2016,
     "plot": "居住在神之围楼的少女椿为报恩复活人类少年鲲，不惜逆天而行损耗自身灵力，接连引发天灾祸患，在湫的默默牺牲守护下，三人历经磨难改写命运轮回。"},
    {"title": "长安三万里", "genre": 4, "duration_minutes": 168, "release_year": 2023,
     "plot": "晚年高适回忆半生过往，串联起与李白、杜甫等一众盛唐诗人的半生羁绊，以漫漫人生路铺陈大唐兴衰，数十首经典唐诗穿插其间，再现恢弘浪漫的盛唐风华。"},
    {"title": "雄狮少年", "genre": 4, "duration_minutes": 104, "release_year": 2021,
     "plot": "瘦小怯懦的留守少年阿娟偶然迷上舞狮，拜师落魄的退役狮王咸鱼强，与两名伙伴日复一日艰苦训练，克服家境贫寒与先天短板，最终站上全国舞狮大赛的赛场追逐梦想。"},
    {"title": "罗小黑战记", "genre": 4, "duration_minutes": 105, "release_year": 2019,
     "plot": "流落人间的小黑猫罗小黑因家园被毁四处漂泊，机缘巧合结识无限，一人一妖穿梭人类与妖精世界，在旅途之中化解种族矛盾，揭开妖精世界尘封多年的纷争往事。"},
    {"title": "天书奇谭", "genre": 4, "duration_minutes": 89, "release_year": 1983,
     "plot": "天宫袁公盗取天书下凡刻印，天书被三只狐狸盗走祸乱人间，孤儿蛋生在袁公指引下勤学天书法术，一路和妖狐斗智斗勇，最终夺回天书、惩治作恶妖邪。"}
]
# 将标题和剧情拼接后进行向量化
titles = [movie["title"] for movie in data_list]
plots = [movie["plot"] for movie in data_list]
# 使用空格拼接，例如："阿凡达 战斗中负伤而下身瘫痪..."
combined_texts = [f"{title} {plot}" for title, plot in zip(titles, plots)]
print("正在基于【标题+剧情】生成向量...")
vectors = model.embed_documents(combined_texts)
print(f"✅ 成功生成 {len(vectors)} 条向量，维度为: {len(vectors[0])}")
# 4. 组装数据
insert_data = []
for i, movie in enumerate(data_list):
    insert_data.append({
        "dense_vector": vectors[i],
        "genre": movie["genre"],
        "title": movie["title"],
        "plot": movie["plot"],
        # 按照顺序i，从拼接标题和剧情的合并文本中拿出title_plot字段的标量值
        "title_plot": combined_texts[i],
        "duration_minutes": movie["duration_minutes"],
        "release_year": movie["release_year"],
    })

# todo 3. 指定分区添加数据
res = client.insert(
    collection_name=COLLECTION_NAME,
    partition_name=NEW_PARTITION_NAME, # todo 指定分区
    data=insert_data
)
print(f"🎉 成功插入 {res['insert_count']} 条带剧情简介的电影数据！")

# todo 刷盘加载
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(COLLECTION_NAME)

# todo 4. 指定分区进行搜索
query_text = "孙悟空大战阿凡达"
query_vector = model.embed_documents([query_text])[0]
print(f"\n🔍 {query_text}")
search_res = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    anns_field='dense_vector', # 有多个向量的时候要指定向量字段
    limit=3,
    metric_type="IP",
    output_fields=["title", "genre", "plot", "duration_minutes"],
    filter="release_year <= 2021",
    partition_names=[NEW_PARTITION_NAME]  # 👈 核心：仅在此分区内搜索
)

print("\n🔍 向量语义分区搜索结果：")
# 解析结果（外层是批次，内层是单条数据）
for hits in search_res:
    for hit in hits:
        print(f"相似度分数：{hit['distance']:.4f}")
        print(f"电影名称：{hit['entity']['title']}")
        print(f"电影分类：{hit['entity']['genre']}")
        print(f"剧情简介：{hit['entity']['plot']}\n")

input('sssss')
print(f'即将删除分区 {NEW_PARTITION_NAME}！分区中数据将全部删除！')
# 此时可以先去Attu页面查看分区情况

if client.has_partition(COLLECTION_NAME, NEW_PARTITION_NAME):
    # 释放分区
    client.release_partitions(COLLECTION_NAME, [NEW_PARTITION_NAME])
    # 删除分区
    client.drop_partition(COLLECTION_NAME, NEW_PARTITION_NAME)

# todo 释放集合内存
client.release_collection(COLLECTION_NAME)

"""
================== 运行输出 ==================
当前分区列表: ['_default']
正在基于【标题+剧情】生成向量...
✅ 成功生成 10 条向量，维度为: 1024
🎉 成功插入 10 条带剧情简介的电影数据！

🔍 孙悟空大战阿凡达

🔍 向量语义分区搜索结果：
相似度分数：0.5050
电影名称：西游记之大圣归来
电影分类：4
剧情简介：被封印五行山五百年的孙悟空...

相似度分数：0.4341
电影名称：大鱼海棠
电影分类：4
剧情简介：居住在神之围楼的少女椿为报恩...

相似度分数：0.4276
电影名称：姜子牙
电影分类：4
剧情简介：封神大战后姜子牙因不忍斩杀无辜狐妖...

即将删除分区 partition_new！分区中数据将全部删除！
===============================================
"""