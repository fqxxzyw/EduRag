"""
=================================================================
Models模型 —— Ollama 本地模型
=================================================================
先启动本地 Ollama 服务，再执行代码。
Ollama 官网：https://ollama.com/

本项目中 Ollama 用于 RAGAS 评估脚本（rag_qa/rag_assessment/rag_as.py），
使用本地 qwen3:4b 模型进行评估。
"""
from langchain_ollama import OllamaLLM

# 实例化模型（需要先 ollama pull qwen3:4b 拉取模型）
model = OllamaLLM(model="qwen3:4b")

# ==============================
# 1. invoke 同步调用
# ==============================
result = model.invoke("请给我讲个笑话吧")
print("【invoke 结果】")
print(result)
# 运行结果（示例）：
# 【invoke 结果】
# 哈哈，来个轻松的小笑话吧！
# 小明去面试，面试官问他："你有什么特长？"
# 小明自信地答："我能让事情变得简单！"
# 面试官点点头："那你能把这间办公室变得简单吗？"
# 小明一拍脑门："当然！一秒钟就变成一个点！"


# ==============================
# 2. stream 流式输出
# ==============================
print("\n【stream 结果】")
result = model.stream(input='你都会啥绝活？')
for chunk in result:
    # flush=True 刷新输出缓冲区，流式显示
    print(chunk, end='', flush=True)
print()
# 运行结果（示例）：
# 【stream 结果】
# 哈哈，我的"绝活"可多啦！最拿手的有这几个：
# 1. 3秒翻译10种语言
# 2. 10行代码写个小助手
# 3. 用大白话讲复杂知识
# ...（逐 token 实时输出）


# ==============================
# ChatOllama —— Ollama 的 Chat 模型版本
# ==============================
from langchain_ollama import ChatOllama

chat_model = ChatOllama(model="qwen3:4b")
result = chat_model.invoke("请给我讲个笑话吧")
print("\n【ChatOllama 结果】")
print(result.content)
# 运行结果（示例）：
# 【ChatOllama 结果】
# 为什么数学老师离婚了？
# 因为发现另一半是负数！


from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:qwen3:4b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama 本地服务不需要真实 API Key，填任意字符串即可
    temperature=0,
    max_tokens=1000,
)

result = model.invoke("请给我讲个笑话吧")
print("\n【本地ollama init_chat_model 结果】")
print(result.content)