
import os, re, logging
from datetime import datetime
# python的类型包
from typing import Optional
from base.config import Config

# todo ============ 日志目录 ============
# 日志保存根目录
LOG_ROOT = Config.LOG_DIR
# 确保日志目录存在，存在就不创建
os.makedirs(LOG_ROOT, exist_ok=True)

# todo ============ 日志格式 ============
# DEFAULT_LOG_FORMAT = logging.Formatter(
#     "%(asctime)s - %(name)s - %(levelname)s "
#     "- %(module)s - %(classname)s - %(funcName)s "
#     "- %(filename)s:%(lineno)d - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
# 日志格式配置（包含时间、模块、行号、函数，便于调试）
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d:%(funcName)s - %(message)s"
DEFAULT_LOG_FORMAT = logging.Formatter(
    fmt=log_format, # 日志格式
    datefmt="%Y-%m-%d %H:%M:%S" # asctime具体时间格式
)
"""
%(asctime)s      时间
%(name)s         Logger 名称
%(levelname)s    日志级别 INFO/WARNING/ERROR
%(module)s       模块名（py 文件名，不带后缀）
%(classname)s    类名（没有类则显示 root）
%(funcName)s     函数名
%(filename)s     文件名
%(lineno)d       行号
%(message)s      日志内容
"""

# todo ============ 日志对象 ============
def get_logger(
        name: str = "Edu RAG",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: Optional[str] = None
) -> logging.Logger:
    """
    获取配置好的日志器（开箱即用）
    :param name: 日志对象名称
    :param console_level: 控制台日志级别（默认INFO，开发时可设为DEBUG）
    :param file_level: 文件日志级别（默认DEBUG，记录详细信息）
    :param log_file: 自定义日志文件名（默认按日期生成：agent_20240121.log）
    :return: 配置完成的Logger对象
    """
    # 1. 创建/获取日志对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 全局最低级别

    # todo ！避免重复添加Handler（多次导入时只配置一次）
    if logger.handlers:
        # 如果log对象存在就提前return结束函数
        # 避免后续的代码重复执行logger = get_logger(...)
        return logger

    # 2. 配置控制台Handler（开发调试用）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level) # 日志级别
    console_handler.setFormatter(DEFAULT_LOG_FORMAT) # 日志格式
    logger.addHandler(console_handler) # log对象添加专门负责控制台的日志对象

    # 3. 配置文件Handler（生产环境留存日志）
    if not log_file:
        log_file = os.path.join(
            LOG_ROOT, # 日志文件目录
            f"{name}_{datetime.now().strftime('%Y%m%d')}.log" # 日志文件名
        )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger

# todo ============ 全局日志对象 ============
# 全局日志对象：后续所有代码都导入并使用这个logger对象
logger = get_logger(
    name='Edu RAG',
    console_level=Config.LOG_CONSOLE_LEVEL, # 控制台中显示的日志级别
    file_level=Config.LOG_FILE_LEVEL, # 文件中记录的日志级别
)


if __name__ == '__main__':
    # 受到 .ini配置文件配置的日志级别限制！
    logger.debug('调试')
    logger.info('测试')
    logger.warning('警告')
    logger.error('错误')
    logger.critical('严重错误！')