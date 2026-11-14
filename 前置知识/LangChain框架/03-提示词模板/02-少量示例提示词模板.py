"""
=================================================================
提示词模板 —— 少量示例提示词模板（Few-shot）
=================================================================
FewShotPromptTemplate：支持基于模板注入任意数量的示例信息。
先给模型看几个"示范"，再让模型处理新内容。
"""
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

api_key = os.environ.get("AliApiKey")

# ==============================
# 1. 定义示例模板 + 示例数据
# ==============================
# 示例的模板
example_template = PromptTemplate.from_template("单词：{word}, 反义词：{antonym}")
# 示例的动态数据注入，要求是 list 内部套字典
examples_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"},
]

# ==============================
# 2. 构建少量示例提示词模板
# ==============================
few_shot_template = FewShotPromptTemplate(
    # 前缀：示例之前的提示词
    prefix="告知我单词的反义词，我提供如下的示例：",
    # 示例数据的模板
    example_prompt=example_template,
    # 示例的数据（用来注入动态数据的），list 内套字典
    examples=examples_data,
    # 后缀：示例之后的提示词
    suffix="基于前面的示例告知我，{input_word}的反义词是？",
    # 在前缀 prefix 或后缀 suffix 中所需要注入的变量名，可多个
    input_variables=['input_word']
)

# ==============================
# 3. 组装完整提示词
# ==============================
# 注意：这里用的是 invoke，不是 format
prompt_text = few_shot_template.invoke(input={"input_word": "左"}).to_string()
print("【完整提示词】")
print(prompt_text)
print('*' * 10)

# ==============================
# 4. 调用模型
# ==============================
model = init_chat_model(
    "openai:qwen3.7-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0,
)
res = model.invoke(input=prompt_text)
print("【模型回答】")
print(res.content)
# 运行结果（示例）：
# 【完整提示词】
# 告知我单词的反义词，我提供如下的示例：
# 单词：大, 反义词：小
# 单词：上, 反义词：下
# 基于前面的示例告知我，左的反义词是？
# **********
# 【模型回答】
# 单词：左, 反义词：右
