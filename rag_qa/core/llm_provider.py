"""
rag_qa 核对代码目录
采用单列模式，全局之创建一次LLM实例，候选直接返回缓存对象
"""
from langchain.chat_models import init_chat_model
from base import logger, Config

# 大模型单例对象，全局只创建一次
_llm_instance = None


def get_llm():
    """大模型工厂函数
    使用 LangChain 的 init_chat_model 统一入口，通过 OpenAI 兼容模式调用通义千问。
    单例模式：全局只创建一次 LLM 实例，后续调用直接返回缓存对象。
    :return: ChatModel 实例（invoke 返回 AIMessage，需要 .content 取文本）
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = init_chat_model(
            f"openai:{Config.LLM_MODEL}",
            api_key=Config.DASHSCOPE_API_KEY,
            base_url=Config.DASHSCOPE_BASE_URL,
            temperature=0.1,
        )
        logger.info(f"LLM工厂: 模型已创建 -> {Config.LLM_MODEL}")
    return _llm_instance
