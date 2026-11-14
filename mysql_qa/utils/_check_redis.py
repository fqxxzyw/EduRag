import os, sys, redis
# # 把项目根目录加入模块搜索路径，保证能 import base（本文件在 mysql_qa/utils 下，向上两层即根目录）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print(sys.path)

from base import Config

# 1. 连接 redis 的 0 号库（db=0）
client = redis.Redis(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    password=Config.REDIS_PASSWORD,
    db=0,                 # 指定 0 号库
    decode_responses=True # 自动把 bytes 转成 str，拿到的直接是字符串
)

# 2. 存入：key 为 a:b:c，值为 xxxxx
client.set("a:b:c", "xxxxx")
print(f"✅ 存入成功：a:b:c = {client.get('a:b:c')}")

client.set("a:b:1", "xxxxx1")
client.set("a:b:2", "xxxxx2")
client.set("a:xx:3", "xxxxx3")
client.set("a:xxx:4", "xxxxx4")


# 3. 按前缀搜索
#    说明：redis 里的冒号 : 只是普通的 key 字符，没有层级结构，
#    所谓"按前缀查"就是用 pattern 匹配模式去扫 key：
#    - "a:*"    匹配 a: 开头的所有 key（能匹配到 a:b:c）
#    - "a:b:*"  匹配 a:b: 开头的所有 key（能匹配到 a:b:c）
#    - "a*"     匹配 a 开头的所有 key（范围更宽，abc 这种也会中）
def search_by_prefix(pattern):
    """按匹配模式搜索 key，并打印每个 key 的 value"""
    print(f"\n🔍 按模式搜索：{pattern}")
    # scan_iter 是增量扫描（推荐），不会一次性阻塞 redis
    # 对比 keys(pattern)：数据量大时会卡住 redis，生产环境慎用
    found = 0
    for key in client.scan_iter(match=pattern):
        value = client.get(key)
        print(f"   key: {key}  ->  value: {value}")
        found += 1
    if found == 0:
        print("   （没搜到任何 key）")

# 4. 分别按前缀 a 、前缀 a:b 查看
search_by_prefix("a:*")    # 前缀 a
search_by_prefix("a:b:*")  # 前缀 a:b:

# 补充：只想看 key 不看 value 的话，scan_iter 拿到的就是 key 本身
# 也可以用全局命令查看：
#   KEYS a:*
#   SCAN 0 MATCH a:* COUNT 100
