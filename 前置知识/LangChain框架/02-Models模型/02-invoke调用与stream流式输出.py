"""
=================================================================
Models模型 —— invoke 同步调用 与 stream 流式输出
=================================================================
invoke：一次性返回完整结果
stream：逐 token 流式返回，适合实时显示
"""
import os
from langchain.chat_models import init_chat_model

api_key = os.environ.get("AliApiKey")

model = init_chat_model(
    "openai:qwen3.7-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0,
    max_tokens=1000,
)

# ==============================
# 1. invoke 同步调用 —— 等待模型生成完毕后一次性返回
# ==============================
result = model.invoke("帮我讲个笑话吧")
print("【invoke 结果】")
print(result.content)
print(type(result))  # <class 'langchain_core.messages.ai.AIMessage'>
# 运行结果（示例）：
# 【invoke 结果】
# 有一天，一块三分熟的牛排在街上走着，突然看到一块五分熟的牛排，可他俩谁也没跟对方打招呼。
# 你知道为什么吗？
# ……因为他们不熟！
# <class 'langchain_core.messages.ai.AIMessage'>


# ==============================
# 2. stream 流式输出 —— 逐 token 实时返回
# ==============================
print("\n【stream 结果】")
result_stream = model.stream("你都会啥绝活？")
for chunk in result_stream:
    # chunk.content 是当前 token 的文本内容
    # flush=True 刷新输出缓冲区，实现流式显示效果
    print(chunk.content, end='', flush=True)
print()  # 换行
# 运行结果（示例）：
# 【stream 结果】
# 哈哈，说到"绝活"，我虽然是个 AI，但肚子里的"墨水"和手里的"工具"可不少。
# 你可以把我当成一个全能型的数字助手，我的主要"绝活"包括：
# 1. 逻辑推理与解题
# 2. 编程与开发
# 3. 海量信息处理
# ...（逐 token 实时输出，体验打字机效果）
