"""
=================================================================
Python操作Milvus —— 查询搜索（标量查询）
=================================================================
按主键/过滤条件/关键词进行标量查询。
"""
from pymilvus import MilvusClient
import os
from langchain_community.embeddings import DashScopeEmbeddings
api_key = os.environ.get("AliApiKey")

model = DashScopeEmbeddings(
    dashscope_api_key=api_key,
    model="text-embedding-v4"
)

# 初始化客户端
client = MilvusClient(uri="http://127.0.0.1:19530")
client.use_database("edurag")
COLLECTION_NAME = 'movies'

# 确保集合已加载（必须加载才能查询）
client.load_collection(COLLECTION_NAME)

# ===================== 标量查询 =====================
# 1.1 根据主键id查询（单条数据）
res = client.query(
    collection_name=COLLECTION_NAME,
    filter="id == 469068973928488139",  # 主键精确匹配
    output_fields=["id", "title", "genre", "release_year", "plot"]  # 返回字段
)
print("🔍 主键查询结果：")
for item in res:
    print(item)

# 1.2 根据分类（分区标量）查询（genre=1 科幻片）
res = client.query(
    collection_name=COLLECTION_NAME,
    filter="genre == 2",
    limit=5,  # 最多返回5条
    output_fields=["title", "genre", "release_year", "duration_minutes"]
)
print("\n🔍 喜剧片查询结果：")
for item in res:
    print(item)

# 1.3 范围查询（上映年份 >= 2020 年）
res = client.query(
    collection_name=COLLECTION_NAME,
    filter="release_year >= 2020",
    output_fields=["title", "release_year"]
)
print("\n🔍 2020年后上映电影：")
for item in res:
    print(item)

# 1.4 模糊匹配（标题包含"地球"）
res = client.query(
    collection_name=COLLECTION_NAME,
    filter='title like "%人%"',
    output_fields=["title", "plot"]
)
print("\n🔍 标题含'人'的电影：")
for item in res:
    print(item)

# 1.5 多条件组合查询（喜剧片 + 时长>150分钟 + 2015年后上映）
res = client.query(
    collection_name=COLLECTION_NAME,
    filter="genre == 2 && duration_minutes > 150 && release_year >= 2015",
    output_fields=["title", "duration_minutes", "release_year"]
)
print("\n🔍 组合条件查询：")
for item in res:
    print(item)

"""
================== 运行输出 ==================
🔍 主键查询结果：
{'id': 469068973928488139, 'title': '夏洛特烦恼', 'genre': 2, 'release_year': 2015, 'plot': '昔日校花秋雅的婚礼上，一事无成的夏洛大闹现场，醉酒后意外穿越回高中时代，凭借未来的记忆成为流行乐坛巨星，最终却发现名利并非真正的幸福。'}

🔍 喜剧片查询结果：
{'genre': 2, 'release_year': 2015, 'duration_minutes': 104, 'title': '夏洛特烦恼', 'id': 469068973928488139}
{'genre': 2, 'release_year': 2021, 'duration_minutes': 128, 'title': '你好，李焕英', 'id': 469068973928488140}
{'genre': 2, 'release_year': 2015, 'duration_minutes': 135, 'title': '唐人街探案', 'id': 469068973928488141}
{'genre': 2, 'release_year': 2019, 'duration_minutes': 116, 'title': '疯狂的外星人', 'id': 469068973928488143}
{'genre': 2, 'release_year': 2010, 'duration_minutes': 97, 'title': '人在囧途', 'id': 469068973928488144}

🔍 2020年后上映电影：
{'title': '你好，李焕英', 'release_year': 2021, 'id': 469068973928488140}
{'title': '独行月球', 'release_year': 2022, 'id': 469068973928488147}
{'title': '满江红', 'release_year': 2023, 'id': 469068973928488148}
{'title': '长津湖', 'release_year': 2021, 'id': 469068973928488151}
{'title': '速度与激情9', 'release_year': 2021, 'id': 469068973928488152}
{'title': '怒火·重案', 'release_year': 2021, 'id': 469068973928488158}

🔍 标题含'人'的电影：
{'title': '唐人街探案', 'plot': '天赋异禀的结巴少年秦风警校落榜，被姥姥遣送泰国找远房表舅唐仁散心，不想一夜之间两人卷入一桩离奇的黄金失窃案，被迫开启了一段搞笑又惊险的逃亡破案之旅。', 'id': 469068973928488141}
{'title': '疯狂的外星人', 'plot': '游乐园耍猴人耿浩意外救下了一个从天而降的外星人，面对这个拥有超能力的天外来客，他和好兄弟大飞将其当成猴子驯养，引发了一系列鸡飞狗跳的闹剧。', 'id': 469068973928488143}
{'title': '人在囧途', 'plot': '玩具集团老板李成功和讨债的挤奶工牛耿在春运期间踏上返乡之路，因为各种阴差阳错的巧合，两个阶层迥异的男人被迫结伴同行，经历了种种倒霉与温情。', 'id': 469068973928488144}
{'title': '飞驰人生', 'plot': '曾经叱咤风云的拉力赛车手张驰被禁赛五年，如今只能经营大排档的他为了重拾尊严与梦想，决定重返车坛挑战年轻一代的天才车手，却面临着没钱没车的窘境。', 'id': 469068973928488146}

🔍 组合条件查询：
{'title': '满江红', 'duration_minutes': 159, 'release_year': 2023, 'id': 469068973928488148}
"""