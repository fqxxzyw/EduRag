import redis, json, traceback
from base import Config, logger


class RedisClient:
    def __init__(self):
        self.logger = logger
        try: # redis连接对象
            self.client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                password=Config.REDIS_PASSWORD,
                db=Config.REDIS_DB, # 默认0号库
                decode_responses=True, # 自动bytes转str
                )
            self.logger.info(f"✅ Redis连接成功：{Config.REDIS_HOST}:{Config.REDIS_PORT}，password:******")
        except Exception as e:
            self.logger.error(f"Redis连接异常: {traceback.format_exc()}")
            raise e

    def set_data(self, key:str, value):
        """批量存储数据到redis
        1. Redis不能存Python稍微复杂点的数据类型→必须先json.dumps转字符串
        2. json.dumps(xxx, ensure_ascii=False) →
            2.1 字典dict ➔ JSON对象{}。字典的键必须是字符串
            2.2 列表list/元组tuple ➔ JSON数组[]
            2.3 字符串str ➔ JSON字符串
            2.4 数字int/float) ➔ JSON数字，支持正负数、小数等
            2.5 布尔值True/False ➔ JSON小写布尔值true/false
            2.6 空值None ➔ JSON空值null
        3. 参数ensure_ascii=False，让中文不乱码
            关闭默认的“非ASCII字符自动转义”
                json.dumps({"name": "张三"}) →
                    '{"name": "\u5f20\u4e09"}'
                json.dumps({"name": "张三"}, ensure_ascii=False) →
                    '{"name": "张三"}'
        4. 可以用 json.loads(xxx) 还原
        """
        try:
            self.client.set(
                name=key, # f'edu_rag:name2:{key}'
                value=json.dumps(value, ensure_ascii=False),
                ex=Config.REDIS_TTL # 缓存过期时间
            )
            self.logger.info(f"数据存储成功：{key}")
        except redis.RedisError as e:
            self.logger.error(f"数据存储失败：{traceback.format_exc()}")

    def get_data(self, key:str):
        """获取批量的数据"""
        try:
            value = self.client.get(key)
            if value:
                self.logger.info(f"数据获取成功：{key}")
                return json.loads(value)
            else:
                self.logger.info(f"数据获取失败：{key}，无数据")
                return None
        except redis.RedisError as e:
            self.logger.error(f"Redis异常：{traceback.format_exc()}")
            return None

    def set_answer(self, query:str, answer:str):
        """存储一个问题的答案到redis"""
        try: # f'edu_rag:answer:{query}' 完整key名
            self.client.set(
                name=f'edu_rag:answer:{query}', # key
                value=answer,
                ex=Config.REDIS_TTL # 缓存过期时间
            )
            self.logger.info(f"问题缓存成功：{query}")
        except redis.RedisError as e:
            self.logger.error(f"问题缓存失败：{traceback.format_exc()}")

    def get_answer(self, query:str):
        """从redis中获取一个问题的缓存答案"""
        try: # f'edu_rag:answer:{query}' 完整key名
            answer = self.client.get(f'edu_rag:answer:{query}')
            if answer:
                self.logger.info(f"缓存命中：{query}")
                return answer
            else:
                self.logger.info(f"缓存未命中：{query}，无数据")
                return None
        except redis.RedisError as e:
            self.logger.error(f"Redis异常：{traceback.format_exc()}")
            return None

    def clear_all_data(self):
        """清空所有数据"""
        try:
            self.client.flushdb()
            self.logger.info("数据已清空")
        except redis.RedisError as e:
            self.logger.error(f"数据清空失败：{traceback.format_exc()}")


if __name__ == '__main__':
    # 测试
    RedisClient().clear_all_data()
    RedisClient().client.ping()