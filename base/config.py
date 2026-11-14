
import os
import configparser # .ini配置文件解析库
from base.path_tools import get_abs_path, get_project_root

class Config:
    # 配置文件相对（项目目录）路径
    CONFIG_REL_NAME = 'config.ini'

    def __init__(self, config_file=None):
        if config_file is None: # 默认使用项目根目录下的配置文件
            config_file = get_abs_path(Config.CONFIG_REL_NAME)
        # 创建一个配置文件对象，能够解析.ini文件
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read(config_file, encoding='utf8')

        """config.ini
        [log]
        # 日志文件目录 相对于项目根路径的位置
        log_dir = logs
        """
        # 日志配置
        self._LOG_REL_DIR = self.config.get('log', 'log_dir', fallback='logs') # 日志文件目录 相对于项目根路径的位置
        # 日志文件目录绝对路径
        self.LOG_DIR = get_abs_path(self._LOG_REL_DIR)
        # 日志控制台级别
        self.LOG_CONSOLE_LEVEL = self.config.getint('log', 'log_console_level', fallback=10)
        # 日志文件级别
        self.LOG_FILE_LEVEL = self.config.getint('log', 'log_file_level', fallback=10)


        # MySQL配置，fallback 为默认值
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='127.0.0.1')
        self.MYSQL_PORT = self.config.getint('mysql', 'port', fallback=3306)
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='123456')
        self.MYSQL_DB = self.config.get('mysql', 'db', fallback='edurag')
        self.MYSQL_TABLE = self.config.get('mysql', 'table', fallback='jpkb')

        # Redis配置
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='127.0.0.1')
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='123456')
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)
        self.REDIS_TTL = self.config.getint('redis', 'ttl', fallback=86400) # 缓存有效期（秒），配置中拿不到就默认1天


        # todo ======= RAG新增配置 =======
        # Milvus配置
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='127.0.0.1')
        self.MILVUS_PORT = self.config.getint('milvus', 'port', fallback=19530)
        self.MILVUS_DATABASE = self.config.get('milvus', 'database_name', fallback='edurag')
        self.MILVUS_COLLECTION = self.config.get('milvus', 'collection_name', fallback='edurag_doc')

        # 数据配置
        self.DATA_DIR = get_abs_path(self.config.get('data', 'data_dir', fallback='data'))

        # 模型配置
        self.LLM_MODEL = self.config.get('model', 'dashscope_model', fallback='qwen-plus')

        try: # 云上模型APIKey
            self.DASHSCOPE_API_KEY = self.config.get('model', 'dashscope_api_key')
        except: # 本地环境变量获取
            self.DASHSCOPE_API_KEY = os.getenv(self.config.get('model', 'env_api_key', fallback='API KEY ERROR!!'))

        self.DASHSCOPE_BASE_URL = self.config.get('model', 'dashscope_base_url', fallback='')
        # 本地向量模型目录绝对路径
        self.EMBEDDING_MODEL_PATH = get_abs_path(self.config.get('model', 'embedding_path', fallback=''))
        # 本地重排序模型目录绝对路径
        self.RERANK_MODEL_PATH = get_abs_path(self.config.get('model', 'rerank_path', fallback=''))

        # 本地预训练的BERT模型在项目路径中的相对路径
        self.BERT_PATH = get_abs_path(self.config.get('model', 'bert_base_path', fallback=''))
        # 针对BERT模型微调后 新模型在项目路径中的相对路径
        self.BERT_QUERY_PATH = get_abs_path(self.config.get('model', 'bert_query_classifier_path', fallback=''))
        # 训练的BERT模型的数据集在项目路径中的相对路径
        self.BERT_TRAIN_DATA = get_abs_path(self.config.get('model', 'bert_train_data', fallback=''))


        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        # 有效来源列表
        self.VALID_SOURCES = eval(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        # 客服电话
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='400-1234567')

        # 设备配置
        self.DEVICE = self.config.get('device', 'device', fallback='cpu')

# 进程内 全局单例
# 别的文件引用的时候：
# from base.config import Config
# Config不管是谁引用都是同一个对象！
Config = Config()


if __name__ == '__main__':
    # 测试 配置文件 读取 日志文件目录
    print(Config.LOG_DIR)
    print(type(Config.MYSQL_PORT))
    print(Config.VALID_SOURCES)
    print(Config.DATA_DIR)
    # print(Config.DASHSCOPE_API_KEY)