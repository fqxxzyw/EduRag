"""
=================================================================
Python操作Milvus —— 融合检索
=================================================================
BM25 稀疏向量 + 密集向量 的混合检索。
包含：重设字段索引、重新插入数据、RRF/Weighted 融合。
"""
from pymilvus import MilvusClient, DataType, FunctionType, Function # 导包
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
# 集合名字
COLLECTION_NAME = 'movies'

# todo schema规定字段
# 定义schema
schema = client.create_schema(enable_dynamic_field=True)
# 主键
schema.add_field(
    field_name="id", # 字段名
    datatype=DataType.INT64, # 类型
    is_primary=True, # 主键
    auto_id=True, # 自动id！
    description="主键，电影ID"
)
# 标量字段（文档内容）
schema.add_field(
    field_name="genre",
    datatype=DataType.INT64,
    # is_partition_key=True, # 分区标量声明
    # description="分区标量（分区键），电影类别"
    description="标量，电影类别，为了后边演示分区操作，取消了分区键的设置"
)
schema.add_field(field_name="title",
                 datatype=DataType.VARCHAR,
                 max_length=200,
                 description="普通标量，电影名称，向量内容1",
                 )
schema.add_field(field_name="plot",
                 datatype=DataType.VARCHAR,
                 max_length=500,
                 description="普通标量，电影介绍，向量内容2")
# todo 多了个标量字段，是2个向量字段的原内容
schema.add_field(field_name="title_plot",
                 datatype=DataType.VARCHAR,
                 max_length=800,
                 description="普通标量，电影名称+介绍，向量内容3",
                 # todo 启用分词器 BM25算法要求输入字段开启分词器
                 enable_analyzer=True,
                 # 分词器参数：中文分词
                 analyzer_params={"type": "chinese"}
                 )
schema.add_field(field_name="duration_minutes",
                 datatype=DataType.INT32,
                 nullable=True,
                 description="普通标量，电影时长")
schema.add_field(field_name="release_year",
                 datatype=DataType.INT32,
                 nullable=True,
                 description="普通标量，上映年份")
# 稠密向量字段
schema.add_field(
    field_name="dense_vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=1024, # 只有稠密向量才需要指定dim，要和向量模型产生维度一致
    description="稠密，【title电影名字+plot电影介绍】转换的向量"
)
# todo 多了个 稀疏向量字段
schema.add_field(
    field_name="sparse_vector",
    datatype=DataType.SPARSE_FLOAT_VECTOR,
    description="稀疏，【title电影名字+plot电影介绍】转换的向量"
)
# todo 稀疏向量生成函数
#  只要往title_plot字段写入文本，就自动调用BM25算法
#  将其转换为稀疏向量，并存入sparse_vector字段中。
bm25_func = Function(
    name="title_plot_bm25_func",
    input_field_names=["title_plot"],
    output_field_names=["sparse_vector"],
    function_type=FunctionType.BM25, # todo BM25算法
    description='只要往title_plot字段写入文本，就自动调用本函数，使用BM25生成稀疏向量'
)
# 稀疏向量生成函数添加到schema中
schema.add_function(bm25_func)

# todo 创建集合
client.drop_collection(COLLECTION_NAME) # 先删除再创建集合
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    # 没有指定分区数量，会自动建立16个分区
    # 但有分区标量，依旧会依旧分区键把数据放到不同的分区中
    # 如果不设分区数量，也没有分区键，就只有一个_default分区
    description='为了后边演示分区操作，取消了分区数量的设置',
)

# todo 索引
index_params = client.prepare_index_params()
# todo 为稀疏向量字段 'sparse_vector' 添加索引
index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_vector_index",
    index_type="AUTOINDEX", # 索引类型
    metric_type="BM25", # 度量方式
)
# 为向量字段 'dense_vector' 添加索引
index_params.add_index(
    field_name="dense_vector", # 字段名
    index_name="dense_vector_index", # 可选，为索引起个名字
    index_type="IVF_FLAT", # 索引类型
    metric_type="IP", # 度量方式
    params={"nlist": 128} # 索引参数nlist：IVF（聚类）簇的数量
)
# 标量字段 'title' 添加索引
index_params.add_index(
    field_name="title", # 字段名
    index_name="title_index", # 可选，为索引起个名字
    index_type="AUTOINDEX", # 标量索引类型
)
# 创建索引：将定义好的索引参数应用到集合中
client.create_index(
    collection_name=COLLECTION_NAME, # 集合名
    index_params=index_params, # 索引参数对象
)

# 查看该集合下所有的索引名称列表
index_list = client.list_indexes(COLLECTION_NAME)
print("集合中的索引名称列表：", index_list)

import os
api_key=os.environ.get("AliApiKey")
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

# 1. 初始化 Milvus 客户端和 Embedding 模型
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'

# 阿里云的text-embedding-v4模型，自动进行归一化
model = DashScopeEmbeddings(
    dashscope_api_key=api_key,
    model="text-embedding-v4" # 默认1024维
)

# 2. 30条真实电影数据
movies_data = [
    # --- 科幻类 (genre=1) ---
    {"title": "流浪地球", "genre": 1, "duration_minutes": 173, "release_year": 2019,
     "plot": "太阳即将毁灭，人类在地球表面建造出巨大的推进器，寻找新的家园。然而宇宙之路危机四伏，为了拯救地球，年轻人们挺身而出展开争分夺秒的生死之战。"},
    {"title": "星际穿越", "genre": 1, "duration_minutes": 169, "release_year": 2014,
     "plot": "在未来地球气候恶化、农作物相继灭绝的时代，一队探险家利用新发现的虫洞，穿越遥远的星系，在浩瀚宇宙中寻找适合人类移居的新星球。"},
    {"title": "盗梦空间", "genre": 1, "duration_minutes": 148, "release_year": 2010,
     "plot": "道姆·柯布是一名经验老道的窃贼，他能够潜入人们精神最为脆弱的梦境中窃取潜意识中有价值的秘密。一次看似不可能的植入思想任务就此展开。"},
    {"title": "黑客帝国", "genre": 1, "duration_minutes": 136, "release_year": 1999,
     "plot": "一名年轻的网络黑客尼奥发现看似正常的现实世界实际上是由名为矩阵的计算机人工智能系统控制的，他在神秘女郎崔妮蒂的引导下加入了反抗军。"},
    {"title": "银翼杀手2049", "genre": 1, "duration_minutes": 164, "release_year": 2017,
     "plot": "故事发生在前作三十年后，新任银翼杀手K发现了一个可能让社会陷入混乱的秘密，为了阻止灾难，他开始寻找失踪多年的前银翼杀手瑞克·戴克。"},
    {"title": "阿凡达", "genre": 1, "duration_minutes": 162, "release_year": 2009,
     "plot": "战斗中负伤而下身瘫痪的前海军战士杰克，决定替死去的同胞哥哥来到潘多拉星，操纵人类与当地纳美人基因结合创造的阿凡达，却在探索中爱上了外星公主并背叛了人类。"},
    {"title": "沙丘", "genre": 1, "duration_minutes": 156, "release_year": 2021,
     "plot": "天才少年保罗·厄崔迪被命运推向宇宙中最危险的星球，为了保护家人和人民，他必须前往全宇宙最危险的地方，对抗邪恶势力以夺取珍贵的香料资源。"},
    {"title": "火星救援", "genre": 1, "duration_minutes": 144, "release_year": 2015,
     "plot": "在一次火星登陆任务中，宇航员马克遭遇沙尘暴被队友误认为死亡而遗弃，他必须依靠有限的物资和科学知识，在荒凉的火星上独自生存等待救援。"},
    {"title": "降临", "genre": 1, "duration_minutes": 116, "release_year": 2016,
     "plot": "十二个神秘的飞行器降临地球各地，语言学家露易丝被军方招募去尝试与外星人沟通。在破译外星语言的过程中，她获得了预知未来的能力，从而改变了人类的命运。"},
    {"title": "机器人总动员", "genre": 1, "duration_minutes": 98, "release_year": 2008,
     "plot": "公元2700年，地球因垃圾成山而被人类废弃，只留下一个名叫瓦力的机器人在日复一日地清理垃圾，直到有一天，一艘飞船带来了一个高级机器人伊娃，打破了它的孤独生活。"},

    # --- 喜剧类 (genre=2) ---
    {"title": "夏洛特烦恼", "genre": 2, "duration_minutes": 104, "release_year": 2015,
     "plot": "昔日校花秋雅的婚礼上，一事无成的夏洛大闹现场，醉酒后意外穿越回高中时代，凭借未来的记忆成为流行乐坛巨星，最终却发现名利并非真正的幸福。"},
    {"title": "你好，李焕英", "genre": 2, "duration_minutes": 128, "release_year": 2021,
     "plot": "贾晓玲在经历了一次痛失母亲的悲痛后，意外穿越回1981年，与正值青春的母亲李焕英相遇，她试图通过自己的努力让母亲拥有一个更美好的未来。"},
    {"title": "唐人街探案", "genre": 2, "duration_minutes": 135, "release_year": 2015,
     "plot": "天赋异禀的结巴少年秦风警校落榜，被姥姥遣送泰国找远房表舅唐仁散心，不想一夜之间两人卷入一桩离奇的黄金失窃案，被迫开启了一段搞笑又惊险的逃亡破案之旅。"},
    {"title": "西虹市首富", "genre": 2, "duration_minutes": 118, "release_year": 2018,
     "plot": "丙级业余足球队守门员王多鱼突然得知自己是百亿富豪的唯一继承人，但继承条件是必须在一个月内合法花光十亿资产，一场令人啼笑皆非的挥霍之旅由此展开。"},
    {"title": "疯狂的外星人", "genre": 2, "duration_minutes": 116, "release_year": 2019,
     "plot": "游乐园耍猴人耿浩意外救下了一个从天而降的外星人，面对这个拥有超能力的天外来客，他和好兄弟大飞将其当成猴子驯养，引发了一系列鸡飞狗跳的闹剧。"},
    {"title": "人在囧途", "genre": 2, "duration_minutes": 97, "release_year": 2010,
     "plot": "玩具集团老板李成功和讨债的挤奶工牛耿在春运期间踏上返乡之路，因为各种阴差阳错的巧合，两个阶层迥异的男人被迫结伴同行，经历了种种倒霉与温情。"},
    {"title": "泰囧", "genre": 2, "duration_minutes": 105, "release_year": 2012,
     "plot": "商业成功人士徐朗为了争夺公司授权前往泰国寻找一位神秘人物，途中遇到了卖葱油饼的王宝，两人在异国他乡展开了一场充满笑料的公路冒险。"},
    {"title": "飞驰人生", "genre": 2, "duration_minutes": 98, "release_year": 2019,
     "plot": "曾经叱咤风云的拉力赛车手张驰被禁赛五年，如今只能经营大排档的他为了重拾尊严与梦想，决定重返车坛挑战年轻一代的天才车手，却面临着没钱没车的窘境。"},
    {"title": "独行月球", "genre": 2, "duration_minutes": 122, "release_year": 2022,
     "plot": "人类为抵御小行星撞击地球部署了月盾计划，不料陨石提前来袭全员紧急撤离，维修工独孤月因为意外错过了撤退通知，成为了唯一被遗落在月球上的人类。"},
    {"title": "满江红", "genre": 2, "duration_minutes": 159, "release_year": 2023,
     "plot": "南宋绍兴年间，岳飞死后四年，秦桧率兵与金国会谈。会谈前夜宰相府发生命案，小兵张大与亲兵营副统领孙均等小人物被裹挟进这局充满悬疑与反转的迷局之中。"},

    # --- 动作/犯罪类 (genre=3) ---
    {"title": "战狼2", "genre": 3, "duration_minutes": 123, "release_year": 2017,
     "plot": "脱下军装的冷锋本想在海上漂泊度过余生，却被卷入了一场非洲国家的叛乱。为了保护同胞和无辜平民，他孤身犯险冲回沦陷区，展开了惊心动魄的营救行动。"},
    {"title": "红海行动", "genre": 3, "duration_minutes": 138, "release_year": 2018,
     "plot": "索马里海域外，中国商船遭遇劫持，蛟龙突击队奉命执行营救任务。在恶劣的环境和凶残的恐怖分子面前，队员们展现了极高的战术素养和不畏牺牲的精神。"},
    {"title": "长津湖", "genre": 3, "duration_minutes": 176, "release_year": 2021,
     "plot": "抗美援朝战争期间，志愿军第七穿插连在极寒严酷的长津湖地区，面对装备精良的美军王牌部队，凭借着钢铁般的意志和英勇无畏的战斗精神，扭转了战场态势。"},
    {"title": "速度与激情9", "genre": 3, "duration_minutes": 142, "release_year": 2021,
     "plot": "多米尼克和他的团队迎来了最致命的对手——正是他的亲生弟弟雅各布。为了拯救世界和保护家人，他们不得不驾驶着改装战车跨越全球展开极速对决。"},
    {"title": "碟中谍6", "genre": 3, "duration_minutes": 147, "release_year": 2018,
     "plot": "伊森·亨特和他的IMF小组在执行任务时失败，导致三枚钚核落入极端分子手中。为了弥补过失并阻止全球核危机，伊森必须在倒计时结束前找回核武器。"},
    {"title": "复仇者联盟4", "genre": 3, "duration_minutes": 181, "release_year": 2019,
     "plot": "灭霸的一个响指让宇宙一半的生命灰飞烟灭，幸存的超级英雄们集结在一起，利用量子领域进行时间旅行，试图收集无限宝石逆转这场毁灭性的灾难。"},
    {"title": "蝙蝠侠：黑暗骑士", "genre": 3, "duration_minutes": 152, "release_year": 2008,
     "plot": "面对高谭市最狡猾残忍的罪犯小丑带来的巨大混乱与道德考验，蝙蝠侠不仅要打击犯罪，还要在正义与堕落的边缘坚守底线，甚至不惜背负所有的误解。"},
    {"title": "无间道", "genre": 3, "duration_minutes": 101, "release_year": 2002,
     "plot": "黑帮头目韩琛派刘建明进入警队当卧底，同时警方也派陈永仁渗入黑帮。十年后，两人的身份都面临暴露的危险，他们在黑白两道之间挣扎，展开了一场惊心动魄的心理博弈。"},
    {"title": "让子弹飞", "genre": 3, "duration_minutes": 132, "release_year": 2010,
     "plot": "悍匪张麻子劫了买官上任的马邦德，随后自己冒充县长赴鹅城就职。在那里，他与南国一霸黄四郎斗智斗勇，上演了一出充满黑色幽默与暴力美学的传奇故事。"},
    {"title": "怒火·重案", "genre": 3, "duration_minutes": 128, "release_year": 2021,
     "plot": "重案组督察张崇邦亲眼目睹战友被杀，追查线索时发现凶手竟是昔日的好兄弟邱刚敖。曾经的同袍如今站在了对立面，两人在残酷的枪战中展开了宿命般的对决。"}
]

# 3. 将标题和剧情拼接后进行向量化
# 这种组合既保留了精确匹配的锚点，又注入了丰富的语义信息
titles = [movie["title"] for movie in movies_data]
plots = [movie["plot"] for movie in movies_data]

# 使用空格或特殊符号拼接，例如："阿凡达 战斗中负伤而下身瘫痪..."
combined_texts = [f"{title} {plot}" for title, plot in zip(titles, plots)]

print("正在基于【标题+剧情】生成向量...")
vectors = model.embed_documents(combined_texts)
print(f"✅ 成功生成 {len(vectors)} 条向量，维度为: {len(vectors[0])}")

# 4. 组装数据
insert_data = []
for i, movie in enumerate(movies_data):
    insert_data.append({
        "dense_vector": vectors[i],
        "genre": movie["genre"],
        "title": movie["title"],
        "plot": movie["plot"],
        # 按照顺序i，从拼接标题和剧情的合并文本中拿出title_plot字段的标量值
        "title_plot": combined_texts[i], # todo
        "duration_minutes": movie["duration_minutes"],
        "release_year": movie["release_year"],
    })

# 5. 批量插入数据到 Milvus
res = client.insert(collection_name=COLLECTION_NAME, data=insert_data)
print(f"🎉 成功插入 {res['insert_count']} 条带剧情简介的电影数据！")

# 6. 验证集合状态
# todo 刷盘加载
client.flush(collection_name=COLLECTION_NAME)
client.load_collection(COLLECTION_NAME)
# 查看增改操作的数据总数
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"当前集合增改操作的数据总数: {stats}")

import os
api_key=os.environ.get("AliApiKey")
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

# 1. 初始化 Milvus 客户端和 Embedding 模型
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'

# 阿里云的text-embedding-v4模型，自动进行归一化
model = DashScopeEmbeddings(
    dashscope_api_key=api_key,
    model="text-embedding-v4" # 默认1024维
)

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
# ================= 融合检索 =================
query_text = "宇航员在外星球生存"
print(f"Query: {query_text}")
query_vector = model.embed_documents([query_text])[0]
expr='release_year != 2015' # 过滤条件

# 构建第一路：稠密向量语义检索请求
query_vector_req = AnnSearchRequest(
    data=[query_vector], # 传入查询向量
    anns_field="dense_vector", # 指定向量字段
    param={"metric_type": "IP"}, # 指定IP内积度量方式，与索引一致
    limit=10, # 这一路的结果数量
    expr=expr # 过滤条件
)

# 构建第二路：稀疏向量(BM25)关键词检索请求
query_text_req = AnnSearchRequest(
    data=[query_text], # 直接传入原文本，Milvus会自动调用内置 BM25 函数
    anns_field="sparse_vector", # 指定稀疏向量字段
    param={"metric_type": "BM25"}, # 指定度量方式，要与索引一致
    limit=10, # 这一路的结果数量
    expr=expr # 过滤条件
)
"""
0. BM25是基于稀疏向量进行搜索（计算）的相似度算法。
1. BM25的作用
    （稠密向量）相似度搜索：懂语义，但容易漏关键词
    （稀疏向量）BM25关键词搜索：不懂语义，只抓关键词
    混合检索 = 语义 + 关键词 ==> 最准！
2. BM25举例：
2.1 搜索 宇航员在外星球生存：宇航员、外星球、生存
2.2 文本：
    火星救援：宇航员、火星、生存
    星际穿越：宇航员、外星球、家园
    流浪地球：地球、太阳、宇宙 
2.3 BM25打分，谁关键词多，谁就更接近：
    火星救援，有宇航员、生存 -> 高分
    星际穿越，有宇航员、外星球 -> 高分
    流浪地球，一个关键词都没有 -> 0分
3. BM25计算相似度得分公式 【BM25分数=IDF词稀有度*TF词频*长度平滑】
IDF词稀有度，整体上、词越少见分越高：
    在sparse_vector稀疏向量字段的所有值中，出现的次数越少分数越高
TF词频，局部中、文档中出现越多分越高：
    当前sparse_vector稀疏向量字段值中，出现的次数越多分数越高
长度平滑，文档越短、分越高：
    稀疏向量字段对应的标量字段字符数越少，得分越高
4. 稀疏向量存在的意义：
4.1 和稠密向量一起共用相同的搜索或计算引擎
4.2 省资源、速度快，稀疏向量在计算机底层极其轻量
    极致压缩：
        假设词典有 10 万个词，那么向量就是 10 万维。
        但在任何一篇具体的文档中，最多也就出现几百个词。
        这意味着这个 10 万维的向量里，99% 以上的数字都是 0。
    性能飞跃：
        存储时，只记录“非零的词和它的分数（TF、IDF、BM25）”，省空间。
        搜索计算时，对极少的非零维度进行运算，速度极快。
        文档数量庞大的时候，比字符串硬匹配快多了！
"""

# --- 场景 A：使用 RRFRanker (排名融合) ---
print("\n=== RRFRanker 搜索结果 ===")
rrf_results = client.hybrid_search(
    collection_name=COLLECTION_NAME,
    reqs=[query_vector_req, query_text_req], # 传入两个请求
    ranker=RRFRanker(k=60), # 排名器最多只看前 60 名
    limit=1, # 最终结果数量
    output_fields=["title", "genre", "plot", "release_year"],
)
# print(rrf_results)
for hits in rrf_results:
    # print(hits)
    for hit in hits:
        print(hit['distance']) # 评分大于0的小数，越大越相似
        print(hit)
"""RRFRanker单路得分计算公式【 单路每个召回结果得分 = 1 / (k+名次) 】   
1. k 越小：第一名优势爆炸大，后面名次得分断崖下跌 
比如 k=5：
    # 第 1 名 = 1/(5+1)≈0.166
    # 第 2 名 = 1/(5+2)≈0.14，差距很大
2. k 越大：前后名次得分差距变小，各路结果更均衡 
比如 k=60：
    # 第 1=1/61≈0.0164
    # 第 2=1/62≈0.0161
    # 结果就是前后分数拉不开
    # 冷门文档只要另一路靠前也能逆袭上位
"""

# --- 场景 B：使用 WeightedRanker (加权平均) ---
print("\n=== WeightedRanker 搜索结果 ===")
weighted_results = client.hybrid_search(
    collection_name=COLLECTION_NAME,
    reqs=[query_vector_req, query_text_req], # 传入两个请求
    ranker=WeightedRanker(0.7, 0.3), # 语义占 70%，关键词占 30%
    limit=3, # 最终结果数量
    output_fields=["title", "genre", "plot", "release_year"],
)
for hits in weighted_results:
    for hit in hits:
        print(hit['distance'])  # 评分0-1，越大越相似
        print(hit)

# todo 释放集合
client.release_collection(COLLECTION_NAME)

"""
================== 运行输出 ==================
集合中的索引名称列表： ['sparse_vector_index', 'dense_vector_index', 'title_index']
正在基于【标题+剧情】生成向量...
✅ 成功生成 30 条向量，维度为: 1024
🎉 成功插入 30 条带剧情简介的电影数据！
当前集合增改操作的数据总数: {'row_count': 30}
Query: 宇航员在外星球生存

=== RRFRanker 搜索结果 ===
0.032786883413791656
{'id': ..., 'distance': 0.0328, 'entity': {'title': '星际穿越', 'genre': 1, 'plot': '在未来地球气候恶化...', 'release_year': 2014}}

=== WeightedRanker 搜索结果 ===
0.7019844651222229
{'id': ..., 'distance': 0.702, 'entity': {'title': '星际穿越', 'genre': 1, 'plot': '在未来地球气候恶化...', 'release_year': 2014}}
0.679426372051239
{'id': ..., 'distance': 0.679, 'entity': {'title': '沙丘', 'genre': 1, 'plot': '天才少年保罗·厄崔迪...', 'release_year': 2021}}
0.4631536304950714
{'id': ..., 'distance': 0.463, 'entity': {'title': '降临', 'genre': 1, 'plot': '十二个神秘的飞行器...', 'release_year': 2016}}
===============================================
"""