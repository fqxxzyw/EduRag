import os, sys
# 把项目根目录加入模块搜索路径，保证能 import base（本文件在 mysql_qa/utils 下，向上两层即根目录）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(sys.path)

from mysql_qa.core_search import Bm25SoftmaxSearch
from mysql_qa.utils.mysql_client import database_init
from base import logger
import traceback
import time

class MySQLQaSystem:
    def __init__(self):
        self.logger = logger
        self.core_search = Bm25SoftmaxSearch()

    def search(self, query):
        """ 搜索 并记录耗时 """
        start_time = time.time()
        self.logger.info(f"开始检索：{query}")
        # todo 执行Bm25SoftmaxSearch搜索
        answer, _ = self.core_search.search(query, threshold=0.85)
        if answer:
            self.logger.info(f"检索成功：答案长度{len(answer)}")
        else:
            self.logger.info(f"检索失败，需要调用RAG系统")
            answer = 'SQL未找到答案'
        process_time = time.time() - start_time
        self.logger.info(f"检索结束，耗时：{process_time:.3f}s")
        return answer # 返回检索结果

    def main(self):
        """ 程序启动入口 """
        try:
            print('\n【欢迎使用MySQL QA系统！（exit退出程序）】')

            while True:
                print('\n', '='*20)
                query = input("请输入问题：").strip()
                if query == 'exit':
                    print("程序已退出")
                    break
                answer = self.search(query)
                print('答案：', answer)
        except Exception as e:
            logger.error(f"程序异常：{traceback.format_exc()}")
            print("程序异常，请检查日志")
        finally:
            # 关闭数据库连接
            self.core_search.mysql_client.close()


if __name__ == '__main__':



    # database_init() # 初始化数据库 TODO

    q_list = [
        '关联子查询的执行顺序是什么', '归一化算法问题',
        'os.path.dirname() 方法有什么作用？',
        '函数和类区别', '什么是张量',
        '特殊符号如何切割'
    ]

    MySQLQaSystem().main() # 程序入口