"""
=================================================================
Models模型 —— 嵌入模型 Embeddings Models
=================================================================
将文本转换为向量（浮点数列表），本质是在高维空间坐标系中表达语义方向。
有了向量就可以做语义计算：语义搜索、相似文本匹配、文本相似度比对等。

文本嵌入模型排行榜（MTEB）：https://huggingface.co/spaces/mteb/leaderboard

注意：本项目（EduRAG）生产环境直接使用 BGEM3EmbeddingFunction（milvus_model），
不走 LangChain 的嵌入模型封装。以下代码仅用于学习和对比。
"""
import os

# ==============================
# 1. 本地 Ollama 嵌入模型
# ==============================
print('➡️ 本地 Ollama 嵌入模型 mxbai-embed-large:latest')
from langchain_ollama import OllamaEmbeddings
# ollama pull mxbai-embed-large:latest
model = OllamaEmbeddings(model="mxbai-embed-large:latest")
# 获取一段文本的向量
ret1 = model.embed_query("我喜欢你")
print(len(ret1), ret1[:5], "...")  # 只打印前5个元素
# 获取多段文本的向量
ret2 = model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"])
print(len(ret2), len(ret2[0]))


# ==============================
# 2. 本地 HuggingFace 嵌入模型
# ==============================
print('\n➡️ 本地 HuggingFace 嵌入模型 m3e-base')
from langchain_huggingface import HuggingFaceEmbeddings
model = HuggingFaceEmbeddings(
    # 模型文件夹路径（根据实际路径调整）
    model_name="../../../rag_qa/models/m3e-base"
)
ret1 = model.embed_query("我喜欢你")
print(len(ret1), ret1[:5], "...")
ret2 = model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"])
print(len(ret2), len(ret2[0]))


# ==============================
# 3. 云上嵌入模型（阿里云通义）
# ==============================
print('\n➡️ 云上嵌入模型 qwen3.7-text-embedding')
from langchain_community.embeddings import DashScopeEmbeddings
model = DashScopeEmbeddings(
    model='qwen3.7-text-embedding',
    # 注意 api_key 参数名称是 dashscope_api_key！
    dashscope_api_key=os.environ.get("AliApiKey")
)
ret1 = model.embed_query("我喜欢你")
print(len(ret1), ret1[:5], "...")
ret2 = model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"])
print(len(ret2), len(ret2[0]))
