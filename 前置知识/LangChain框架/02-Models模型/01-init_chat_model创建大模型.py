"""
=================================================================
Models模型 —— init_chat_model 创建大模型
=================================================================
项目实际用法：通过 init_chat_model + OpenAI兼容模式 调用阿里云通义千问

init_chat_model 是 LangChain 提供的统一模型创建函数，
支持通过 "openai:{model_name}" 格式对接任何 OpenAI 兼容接口。

阿里云百炼平台提供 OpenAI 兼容接口，所以可以用 init_chat_model 直接调用通义千问。
"""
import os
from langchain.chat_models import init_chat_model

# 从环境变量中获取 API 密钥
api_key = os.environ.get("AliApiKey")

# 实例化模型 = 创建一个可以调用 AI 的对象
model = init_chat_model(
    # 1. 模型标识：格式为 "openai:{模型名称}"，走 OpenAI 兼容模式
    "openai:qwen3.7-flash",
    # 2. 大模型的接口地址（阿里云百炼 OpenAI 兼容接口）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 3. API 密钥
    api_key=api_key,
    # 4. 温度系数（控制回答随机性）
    #    0：最严谨、固定、不随机、适合问答/总结/任务
    #    1.0：最发散、最有创意、适合写文案、故事、笑话
    temperature=0,
    # 5. 最大生成字数（1000 token ≈ 750 汉字/英文单词）
    max_tokens=1000,
)

# 调用 AI，发一句话过去
result = model.invoke("帮我讲个笑话吧")
# 打印 AI 回答的内容
print(result.content)
# 运行结果（示例）：
# 有一天，一块三分熟的牛排在街上溜达，迎面走来一块五分熟的牛排。两人擦肩而过，谁也没理谁。
# 你知道为什么吗？
# ……因为他们不熟！
# （中文里"熟"既指食物加热程度，也指人与人之间的熟悉程度～）


