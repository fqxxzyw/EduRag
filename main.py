import os, traceback

# TODO
# '''将整个项目目录添加到python的导包路径list
# 实现效果：无所谓当前工作路径在哪，只要程序启动，导包不出错
# 注意：要在程序入口.py的最开始这么做
# '''
# import sys
# current_file = os.path.abspath(__file__) # 当前文件的绝对路径
# project_root = os.path.dirname(current_file) # 工程根目录
# sys.path.append(project_root)

from rag_qa.core.process_documents import process_documents # 文档加载分割
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from base import Config, logger

# TODO 新增 导入 MySQL QA 系统对象
from mysql_qa.core_search import Bm25SoftmaxSearch


def main(query_mode=True, data_path=Config.DATA_DIR):
    '''rag_qa系统入口：
        1. 数据处理 query_mode=False :
            文档加载分割→转向量存向量库
        2. 默认 交互式查询模式 query_mode=True :
            接收用户问题, 调用RAG流程生成答案
    :param query_mode: 运行模式标识, True: 查询模式, False: 数据处理模式
    :param directory_path: 数据处理模式下的文档根目录，相对项目根目录的路径
    :return: 无
    '''

    # TODO 一、数据处理模式(query_mode=False): 对文档进行加载、分割、向量化、存向量库
    if not query_mode:
        logger.info("进入数据处理模式...")
        # todo 1. 创建向量存储检索对象
        try:
            vector_store = VectorStore()
        except Exception as e:
            logger.error(f"初始化向量存储检索对象失败: {traceback.format_exc()}")
            logger.error("错误：无法初始化向量存储检索对象，无法进入数据处理模式。")
            return # 失败提前退出

        # todo 2. 遍历、拼接每个学科的数据目录绝对路径，逐一加载、分割、向量化、存向量库
        total_chunks_added = 0 # [Document]文档块计数器
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        for source in Config.VALID_SOURCES:
            # todo 2.1 拼接各个学科数据文档的绝对路径
            # print(f'==={Config.DATA_DIR}/{source}_data')
            data_abs_path = f"{data_path}/{source}_data"
            # todo 2.2 判断学科数据文档目录是否存在
            # 不是绝对路径 或 路径不存在 就跳过
            if not os.path.isabs(data_abs_path) or not os.path.exists(data_abs_path):
                logger.warning(f"目录 {data_abs_path} 不存在，跳过处理")
            else: # 学科数据文档目录 存在就处理
                logger.info(f"开始处理目录: {data_abs_path}")
                try: # todo 2.3 加载、分割文档，返回子块文档对象列表 [Document]
                    chunks = process_documents(
                        data_path,
                        Config.PARENT_CHUNK_SIZE, # 父块大小
                        Config.CHILD_CHUNK_SIZE, # 子块大小
                        Config.CHUNK_OVERLAP, # 子块重叠
                    )
                    if not chunks:
                        logger.info(f"目录 {data_abs_path} 未发现有效文档或处理结果为空")
                    else: # todo 2.4 [Document]转向量存入向量库
                        vector_store.add_documents(chunks)
                        total_chunks_added += len(chunks) # [Document]文档块计数器
                        logger.info(f"成功处理目录 {data_abs_path}，添加了 {len(chunks)} 个文档块")
                except Exception as e:
                    logger.error(f"处理目录 {data_abs_path} 时出错: {e}")
        # todo 3. 数据处理模式 流程结束
        logger.info(f"数据处理完成，共添加了 {total_chunks_added} 个文档块到向量存储")

    # TODO 二、交互式查询模式(默认query_mode=True):
    #  1. 接收用户问题, 调用MysqlQA系统
    #  2. 判断答案是否≥阈值0.85，
    #      - 如果≥0.85，直接返回答案
    #      - 如果＜0.85，调用RAG系统生成答案
    else:
        logger.info("进入交互式查询模式...")
        try: # todo 1. 创建 RAG 系统对象
            rag_system = RAGSystem()
        except Exception as e:
             logger.error(f"初始化 RAGSystem 失败: {e}")
             print("错误：无法初始化 RAG 系统，无法进入查询模式。")
             return
        # VALID_SOURCES → ["ai", "java", "test", "ops", "bigdata"]
        valid_sources = Config.VALID_SOURCES
        print("\n欢迎使用 EduRAG 交互式查询系统！")

        bm25_search = Bm25SoftmaxSearch() # 创建mysql检索对象

        while True:
            # todo 2. 获取用户问题
            query = input("\n请输入您的问题（输入 'exit' 退出）：")
            if query.lower() == "exit":
                logger.info("用户退出查询模式")
                print("再见！")
                return # 退出
            # todo 3. 获取用户选择的学科类别
            source_filter_input = input(f"请输入学科类别{valid_sources}，直接回车默认全部：").strip().lower() # 去空格、转小写
            source_filter = None # 学科类别，默认不过滤
            if source_filter_input in valid_sources:
                source_filter = source_filter_input
                logger.info(f"用户选择了学科: {source_filter}")
            else:
                logger.warning(
                    f"无效的学科类别 '{source_filter_input}'，将不过滤"
                )
                print(f"提示：输入的学科 '{source_filter_input}' 无效，将不过滤。")

            """ TODO 新增逻辑开始 调用MysqlQA系统 """
            try:
                # 先查缓存、再bm25检索，有结果自动缓存到redis
                logger.info('正在查询MysqlQA系统...')
                answer, need_rag = bm25_search.search(query)
                if not need_rag: # 如果有答案，且答案匹配度大于0.85
                    logger.info(f'已通过MysqlQA系统获取答案：{len(answer)}')
                    print("-" * 30)
                    print(f"问题: {query}")
                    print(f"回答: {answer}")
                    print("-" * 30)
                    continue
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时，MysqlQA系统异常 {traceback.format_exc()}")
            """ 新增逻辑结束 """

            # todo 4. 调用RAG系统生成答案
            try:
                print("正在生成答案，请稍候...")
                answer = rag_system.generate_answer(query, source_filter)
                # 缓存答案：RAG系统没有缓存的功能，所以这里将答案缓存到Redis中
                bm25_search.redis_client.set_answer(query, answer)
                print("-" * 30)
                print(f"问题: {query}")
                print(f"回答: {answer}")
                print("-" * 30)
            except Exception as e:
                logger.error(f"处理查询 '{query}' 时失败: {traceback.format_exc()}")
                print(f"抱歉，处理您的问题时遇到了错误，请稍后重试或联系管理员{Config.CUSTOMER_SERVICE_PHONE}。\n")


'''命令行模式：通过命令行参数控制'''
def cmd_run():
    import argparse  # 命令行参数解析库
    print('➡️ Edu RAG 命令行模式：')

    # todo 1. 自定义 程序介绍 和 额外说明
    description = """EduRAG v7.9 启动说明：
    1. 默认进入RAG问答模式：
        python ./main.py
    2. 查看命令菜单
        python ./main.py --h
        python ./main.py --help 
    3. 进入RAG数据处理模式，对默认目录中的数据文件进行处理 
        python ./main.py --data
    4. 进入RAG数据处理模式，对指定目录中的数据文件进行处理 
        python ./main.py --data --dir 数据文件的相对或绝对路径
    """
    epilog = f'感谢使用 EduRAG v7.9！如有问题请联系管理员: {Config.CUSTOMER_SERVICE_PHONE}'

    # todo 2. 创建参数解析器，实例化ArgumentParser对象，设置"描述信息"
    parser = argparse.ArgumentParser(
        description=description, # 程序介绍
        epilog=epilog, # 额外的说明
        # 让介绍和说明的内容能够换行
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # todo 3. 添加数据处理模式参数, --data 表示进入数据处理模式.
    parser.add_argument(
        '--data',
        action='store_true',
        help='数据处理模式（RAG离线流程）'
    )
    '''参数 action='store_true'
    1. 当命令行出现 --data → 自动将 args.data 设为 True
    2. 当命令行未出现 --data → 自动将 args.data 设为 False
    '''

    # todo 4. 添加数据目录参数, --dir 表示数据目录.
    parser.add_argument(
        '--dir',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=Config.DATA_DIR, # 默认数据目录的绝对路径
        help='在--data模式下 输入数据目录的路径 进行数据处理'
    )
    '''保障相对路径和绝对路径都支持（默认仅支持绝对路径）
    type=lambda p: os.path.abspath(os.path.expanduser(p))
    1. os.path.abspath() 
        先调用 os.path.isabs() 进行判断。
        如果已经是绝对路径，它直接返回；
        如果是相对路径，它会基于当前工作目录os.getcwd()自动拼接成绝对路径
    2. os.path.expanduser()
        将路径字符串中的波浪号（~）替换为当前用户家目录的绝对路径
    '''

    # todo 5. 解析命令行参数
    args = parser.parse_args()
    # print(args.data, args.dir)

    # todo 6. 执行main函数
    #  query_mode=True表示进入查询模式, False进入数据处理模式
    main(query_mode=(not args.data),  data_path=args.dir)


if __name__ == "__main__":
    # from mysql_qa.utils.mysql_client import database_init
    # database_init() # 初始化mysql数据库

    '''数据处理模式'''
    # print('➡️ Edu RAG 数据处理模式：')
    # main(query_mode=False)

    '''运行查询模式'''
    # print('➡️ Edu RAG QA查询模式：')
    main(query_mode=True)

    '''命令行模式'''
    # cmd_run()


