
import time, traceback
import uuid # 导入 UUID 库，生成唯一会话 ID

# 导入 MySQL 和 Redis 客户端，管理数据库和缓存
from mysql_qa.core_search import Bm25SoftmaxSearch
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import VectorStore
# 使用新的rag_system代码
from rag_qa.core.rag_system import RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config


class IntegratedQASystem:
    def __init__(self):
        # 初始化BM25+softmax检索模块
        self.bm25_search = Bm25SoftmaxSearch()
        # 使用Bm25SoftmaxSearch的mysql_client
        self.mysql_client = self.bm25_search.mysql_client
        # 使用Bm25SoftmaxSearch的redis_client
        self.redis_client = self.bm25_search.redis_client
        # 初始化向量库检索模块
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag_system = RAGSystem()
        # 初始化大模型对象
        self.llm = self.rag_system.llm

        # 加载对话历史表，用于存储会话记录
        self._init_conversation_table()

    def _init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        self.mysql_client.check_and_reconnect() # 重连
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS conversations(
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                ) ENGINE=InnoDB
            """ # 执行SQL语句用游标对象
            self.mysql_client.cur.execute(create_table_sql)
            # 事务提交 （用连接对象）
            self.mysql_client.conn.commit()
            # 记录表初始化成功的日志
            logger.info("对话历史表初始化成功")
        except Exception:
            # 记录表初始化失败的错误日志
            logger.error(f"初始化对话历史表失败: {traceback.format_exc()}")
            raise # 抛异常，终止程序


    def call_llm(self, prompt):
        """调用llm生成答案，并流式输出
        后续FastAPI调用"""
        try:
            messages = [
                {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                {"role": "user", "content": prompt},  # 用户输入的提示
            ]
            # 遍历流式输出的每个chunk
            for chunk in self.llm.stream(messages):
                yield chunk.content
        except Exception as e:
            # 记录 API 调用失败的错误日志
            logger.error(f"LLM调用失败: {traceback.format_exc()}")
            # 返回错误信息
            yield f"错误：LLM调用失败 - {e}"

    def _get_session_history(self, session_id):
        """获取最近5轮对话历史"""
        self.mysql_client.check_and_reconnect()  # 重连
        try:
            # 执行 SQL 查询，获取最近 5 轮对话，按时间倒序（降序）排序
            self.mysql_client.cur.execute("""
                      SELECT question, answer
                      FROM conversations
                      WHERE session_id = %s
                      ORDER BY timestamp DESC
                      LIMIT %s
                  """, (session_id, 5))
            # 将查询结果转换为字典列表
            history = [
                {"question": row[0], "answer": row[1]}
                for row in self.mysql_client.cur.fetchall()
            ] # → [Q5, Q4, Q3, Q2, Q1]
            return history[::-1] # → [Q1, Q2, Q3, Q4, Q5]
        #  [起始位置:结束位置（不要）:方向符号、步长]

        except Exception:
            # 记录查询失败的错误日志
            logger.error(f"获取对话历史失败: {traceback.format_exc()}")
            # 返回空列表
            return []

    def _update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        self.mysql_client.check_and_reconnect()  # 重连
        try:
            # 插入新的对话记录
            self.mysql_client.cur.execute("""
                INSERT INTO conversations (
                    session_id, question, 
                    answer, timestamp
                )
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._get_session_history(session_id)
            # 删除超出 5 轮的旧记录
            self.mysql_client.cur.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))
            # 提交事务
            self.mysql_client.conn.commit()
            # 记录更新成功的日志
            logger.info(f"会话 {session_id} 历史更新成功")
            # 返回更新后的历史
            return history
        except Exception:
            # 记录数据库操作失败的错误日志
            logger.error(f"更新会话历史失败: {traceback.format_exc()}")
            # 回滚事务
            self.mysql_client.conn.rollback()
            raise # 抛出异常

    def _clear_session_history(self, session_id: str) -> bool:
        """清除指定会话历史"""
        self.mysql_client.check_and_reconnect()  # 重连
        try:
            # 删除指定 session_id 的所有对话记录
            self.mysql_client.cur.execute("""
                DELETE FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            # 提交事务
            self.mysql_client.conn.commit()
            # 记录清除成功的日志
            logger.info(f"会话 {session_id} 历史已清除")
            return True # 返回 True 表示成功
        except Exception:
            # 记录清除失败的错误日志
            logger.error(f"清除会话历史失败: {traceback.format_exc()}")
            # 回滚事务
            self.mysql_client.conn.rollback()
            return False # 返回 False 表示失败

    def query(self, query, source_filter=None, session_id=None):
        # print(f'你好')
        """查询集成系统，支持对话历史和流式输出"""
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        # 获取对话历史，若无 session_id 则返回空列表
        history = self._get_session_history(session_id) if session_id else []
        # print(f'history--->{history}')
        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if not need_rag:
            # 如果找到可靠答案，记录答案到日志
            logger.info(f"MySQL答案长度: {len(answer)}")
            if session_id:
                # 更新对话历史
                self._update_session_history(session_id, query, answer)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回答案，标记为完整答案
            yield answer, True

        else:
            logger.info("无可靠MySQL答案，使用RAG系统")
            # 初始化收集完整答案的字符串
            collected_answer = ""
            # 从 RAG 系统获取流式输出
            for token in self.rag_system.generate_answer_withhistory(
                    query,
                    source_filter=source_filter,
                    history=history
                ):
                collected_answer += token # 累积答案
                # 逐 token 返回，标记为部分答案
                yield token, False
            # 缓存答案
            self.bm25_search.redis_client.set_answer(
                query,
                collected_answer
            )
            if session_id:
                # 更新对话历史，存储完整答案
                self._update_session_history(session_id, query, collected_answer)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回空字符串，标记流结束
            yield "", True


def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 生成唯一会话 ID
    session_id = str(uuid.uuid4())
    # 打印欢迎信息
    print("\n欢迎使用集成问答系统！")
    # 打印会话 ID
    print(f"会话ID: {session_id}")

    try:
        while True:
            # 获取用户输入的查询
            # 提示用户输入查询或退出
            print("输入查询进行问答，输入 'exit' 退出。")
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！")
                # 退出循环
                break
            # 获取用户输入的学科过滤
            source_filter = input(f"请输入学科类别 ({'/'.join(Config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in Config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 迭代 query 方法的生成器
            for chunk, is_complete in qa_system.query(
                    query,
                    source_filter=source_filter,
                    session_id=session_id
                ):
                if chunk:
                    # 仅当 chunk 非空时打印
                    print(chunk, end="", flush=True)
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 打印对话历史
            history = qa_system._get_session_history(session_id)
            print("\n最近对话历史:") # idx序号从1开始
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n 答: {entry['answer']}")
    except Exception:
        # 记录系统错误日志
        logger.error(f"系统错误: {traceback.format_exc()}")
        # 打印错误信息
        print(f"发生错误: {traceback.format_exc()}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()


if __name__ == "__main__":
    # from mysql_qa.utils.mysql_client import database_init
    # database_init() # 初始化mysql数据库

    '''清除历史会话'''
    # qa_system = IntegratedQASystem()
    # session_id = '49466345-1e2a-4a6c-8641-72ed9aef3369'
    # ret = qa_system._clear_session_history(session_id)
    # print(ret)

    '''程序入口'''
    main()

