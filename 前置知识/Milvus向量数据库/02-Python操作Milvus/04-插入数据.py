"""
=================================================================
Python操作Milvus —— 插入数据
=================================================================
向集合中插入 30 条电影数据（含向量嵌入）。
"""
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

# 4. 组装成 Milvus 插入格式的数据
insert_data = []
for i, movie in enumerate(movies_data):
    insert_data.append({
        "vector": vectors[i],
        "genre": movie["genre"],
        "title": movie["title"],
        "duration_minutes": movie["duration_minutes"],
        "release_year": movie["release_year"],
        "plot": movie["plot"]
    })

# 5. 批量插入数据到 Milvus
try:
    res = client.insert(collection_name=COLLECTION_NAME, data=insert_data)
    print(f"🎉 成功插入 {res['insert_count']} 条带剧情简介的电影数据！")
except Exception as e:
    print(f"❌ 插入失败: {e}")

# 6. 验证集合状态
# todo 强制刷新数据到硬盘
client.flush(collection_name=COLLECTION_NAME)
# todo 加载集合到内存，加载后才能查询
client.load_collection(COLLECTION_NAME)
# 查看增改操作的数据总数
stats = client.get_collection_stats(COLLECTION_NAME)
print(f"当前集合增改操作的数据总数: {stats}")

"""
================== 运行输出 ==================
正在基于【标题+剧情】生成向量...
✅ 成功生成 30 条向量，维度为: 1024
🎉 成功插入 30 条带剧情简介的电影数据！
当前集合增改操作的数据总数: {'row_count': 30}
===============================================
"""