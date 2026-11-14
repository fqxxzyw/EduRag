"""
=================================================================
Models模型 —— Chat模式（消息类型）
=================================================================
聊天消息包含以下几种类型：
    - AIMessage:    大模型输出的消息（assistant 角色）
    - HumanMessage: 用户消息（user 角色）
    - SystemMessage: 系统消息，设定环境/背景/角色扮演（system 角色）
    - ChatMessage:  可接受任意角色，但大多数时候用上面三种

简化消息列表写法：用元组 (角色, 内容) 代替消息对象，
好处是可以在内容字符串中使用 {变量} 占位，灵活填充。
"""
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

api_key = os.environ.get("AliApiKey")

model = init_chat_model(
    "openai:qwen3.7-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0,
)

# ==============================
# 方式1：使用消息对象列表
# ==============================
messages = [
    SystemMessage(content='你是一名边塞诗人'),
    HumanMessage(content='给我写一首七律'),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首七律。")
]

result = model.invoke(messages)
print("【消息对象方式】")
print(result.content)
# 运行结果（示例）：
# 【消息对象方式】
# 朔风卷雪暗边州，铁甲凝霜夜未休。
# 雁断衡阳云外路，笳吹大漠月中秋。
# 十年征战身如寄，万里思乡泪暗流。
# 莫道功名轻似叶，且将肝胆照金瓯。


# ==============================
# 方式2：简化消息列表（推荐）—— 用元组 (角色, 内容)
# ==============================
messages_simple = [
    # (角色，内容)  角色：system/human/ai
    ("system", "你是一名边塞诗人"),
    ("human", "写一首唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照你上一个回复的格式，再写一首唐诗")
]

result = model.invoke(messages_simple)
print("\n【简化元组方式】")
print(result.content)
# 运行结果（示例）：
# 【简化元组方式】
# 朔气满阴山，霜锋照玉关。
# 十年征战苦，白首未言还。
