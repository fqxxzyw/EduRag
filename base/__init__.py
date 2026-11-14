
# 在init.py中定义包的功能接口，使用相对导入
from .config import Config
from .logger import logger
from .path_tools import get_abs_path
from .path_tools import get_project_root

"""项目中别的py文件就可以直接导包使用 
from base import Config, logger ...
"""
# 定义包功能接口白名单
__all__ = [
    'Config',
    'logger',
    'get_abs_path',
    'get_project_root'
]