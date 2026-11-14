import pymysql, traceback
import pandas as pd
# 自定义的base包在项目的根目录，直接导入
# base包的__init__.py中定义了包功能接口，直接导入
from base import Config, logger, get_abs_path

class MysqlClient:
    def __init__(self):
        self.logger = logger  # 日志对象
        try:  # 直接连接到指定数据库，一步到位
            self.conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,  # 直接指定库名
                charset="utf8mb4"  # 强制字符集，防止乱码
            )
            self.table_name = Config.MYSQL_TABLE
            self.cur = self.conn.cursor() # 创建游标对象
            self.logger.info(f"✅ 直接连接数据库成功：{Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DB}，username:{Config.MYSQL_USER}")
        except Exception as e:
            self.logger.error(
                f"❌ 连接数据库失败：{Config.MYSQL_HOST}:{Config.MYSQL_PORT}，ERROR: {traceback.format_exc()}")
            raise e

    def creat_table(self):
        """先删表 再重建表"""
        try:
            # 1. 删除表（如果存在）
            self.cur.execute(f"DROP TABLE IF EXISTS `{self.table_name}`")

            # 2. 创建新表 表名直接拼接，必须加`反引号
            create_sql = f"""
                CREATE TABLE `{self.table_name}` (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    subject_name VARCHAR(20),
                    question VARCHAR(1000),
                    answer VARCHAR(1000)
                ) ENGINE=InnoDB;
            """
            self.cur.execute(create_sql)
            self.conn.commit()  # InnoDB引擎，需提交！
            self.logger.info(f"表重建成功：{self.table_name}")

        except Exception as e:
            self.logger.error(f"表重建失败：{e}")
            raise e


    def check_and_reconnect(self):
        """mysql连接对象会断开链接（网络波动、超时）
        尝试链接如果失败就重建链接"""
        try:
            self.conn.ping()  # 尝试链接
        except:
            self.close() # 关闭连接
            self.__init__() # 重新初始化


    def insert_data(self, csv_path):
        """插入csv测试数据"""
        self.check_and_reconnect() # 重连
        try:
            # 1. 读取数据
            df = pd.read_csv(csv_path)
            # 2. dataframe类型转换成列表[list]
            # [[一条数据], ...]
            data = df.values.tolist()
            # 3. 插入数据
            insert_sql = f"INSERT INTO `{self.table_name}` (subject_name, question, answer) VALUES (%s, %s, %s)"
            self.cur.executemany(insert_sql, data)
            self.conn.commit()  # InnoDB引擎，需提交！
            self.logger.info(f"数据插入成功：{csv_path}")
        except Exception as e:
            self.logger.error(f"数据插入失败，ERROR: {traceback.format_exc()}")
            raise e

    def fetch_all_questions(self, table_name=Config.MYSQL_TABLE):
        self.check_and_reconnect() # 重连
        """获取所有问题"""
        try: # 获取查询结果
            sql = f"SELECT question FROM `{table_name}`"
            self.cur.execute(sql)
            results = self.cur.fetchall() # 返回 [(问题1,),...]
            self.logger.info(f"从mysql查询结果的数量为：{len(results)}")
            return results # 返回 [(问题1,),...]
        except Exception as e:
            self.logger.error(f"获取所有问题失败，ERROR: {traceback.format_exc()}")
            return []  # 返回空列表

    def fetch_answer(self, question: str):
        """获取指定问题的答案"""
        self.check_and_reconnect()  # 重连
        try:
            # 正确写法：表名拼接，WHERE 字段用 %s 占位
            sql = f"SELECT answer FROM `{Config.MYSQL_TABLE}` WHERE question = %s"
            self.cur.execute(sql, (question,))
            result = self.cur.fetchone()  # 获取查询结果 （答案,）
            self.logger.info(f"问题：{question}，查询结果数量：{len(result)}")
            # 查询结果可能为 None
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"获取答案失败，ERROR: {traceback.format_exc()}")
            return None  # 返回 None

    def close(self):
        try:
            self.cur.close()
            self.conn.close()
            self.logger.info("关闭数据库连接")
        except Exception as e:
            self.logger.error(f"关闭数据库连接失败，ERROR: {traceback.format_exc()}")


def database_init(
        csv_file_path='mysql_qa/data/JP学科知识问答.csv'
    ):
    """项目需要先建库建表插入数据"""
    # 构造数据文件的绝对路径
    csv_file_abs_path = get_abs_path(csv_file_path)
    # print(csv_file_abs_path)
    mysql_client = MysqlClient()
    # 💡需要先创建 数据库edurag
    create_db_query = "CREATE DATABASE IF NOT EXISTS edurag"
    mysql_client.cur.execute(create_db_query)
    mysql_client.conn.commit()
    # 💡再创建 数据表jpkb
    mysql_client.creat_table()
    # 💡插入项目测试数据
    mysql_client.insert_data(csv_file_abs_path)
    # 🔍查看数据
    ret_list = mysql_client.fetch_all_questions()
    # ret_list ➡️ [(问题1,),...]
    print(ret_list[0])
    print(len(ret_list)) # 467条测试数据
    # ⚠️关闭连接
    mysql_client.close()


if __name__ == '__main__':

    # 项目数据初始化
    database_init()
    pass