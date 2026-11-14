"""
=================================================================
提示词模板 —— 课后练习
=================================================================
练习需求：基于 2 个示例，构建 FewShotPromptTemplate，
对目标文本进行信息抽取：抽取产品名称和核心卖点 2 个字段，格式为 Json。

示例 1：
    MacBookPro 高效节能，性能强大，适合牛马工作使用
    会输出：{"产品名称": "MacBookPro", "产品卖点": "高效节能，性能强大"}

示例 2：
    联想笔记本拥有 RTX4060 独立显卡，畅玩游戏，丝滑流畅
    会输出：{"产品名称": "联想笔记本", "产品卖点": "畅玩游戏，丝滑流畅"}

目标文本：
    华为 MatepadPro，高清大屏，长效续航，你的好帮手。
    要求输出：{"产品名称": "华为MatepadPro", "产品卖点": "高清大屏，长效续航"}
"""
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

api_key = os.environ.get("AliApiKey")

# ==============================
# 1. 示例模板
# ==============================
example_template = PromptTemplate.from_template("输入：{input_demo}, 会输出：{out_demo}")

# todo 注意：提示词字符串中的大括号 {xxx} 要写成 {{xxx}}（转义）
# todo 注意：提示词字符串中，要注意单引号和双引号的使用
input_demo1 = 'MacBookPro 高效节能，性能强大，适合牛马工作使用'
out_demo1 = '{{"产品名称": "MacBookPro", "产品卖点": "高效节能，性能强大"}}'
input_demo2 = '联想笔记本拥有 RTX4060 独立显卡，畅玩游戏，丝滑流畅'
out_demo2 = '{{"产品名称": "联想笔记本", "产品卖点": "畅玩游戏，丝滑流畅"}}'

# ==============================
# 2. 示例数据
# ==============================
examples_data = [
    {"input_demo": input_demo1, "out_demo": out_demo1},
    {"input_demo": input_demo2, "out_demo": out_demo2},
]

# ==============================
# 3. 构建少量示例提示词模板
# ==============================
few_shot_template = FewShotPromptTemplate(
    prefix="抽取产品名称和核心卖点 2 个字段，格式为 Json，我提供如下的示例：",
    example_prompt=example_template,
    examples=examples_data,
    suffix="基于前面的示例告知我，{input_content}的抽取结果是？",
    input_variables=['input_content']
)

# ==============================
# 4. 组装提示词并调用模型
# ==============================
input_content = '华为 MatepadPro，高清大屏，长效续航，你的好帮手。'
prompt_text = few_shot_template.invoke(input={"input_content": input_content}).to_string()
print("【完整提示词】")
print(prompt_text)
print('*' * 10)

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
# 抽取产品名称和核心卖点 2 个字段，格式为 Json，我提供如下的示例：
# 输入：MacBookPro 高效节能... 会输出：{"产品名称": "MacBookPro", "产品卖点": "高效节能，性能强大"}
# 输入：联想笔记本拥有 RTX4060... 会输出：{"产品名称": "联想笔记本", "产品卖点": "畅玩游戏，丝滑流畅"}
# 基于前面的示例告知我，华为 MatepadPro...的抽取结果是？
# **********
# 【模型回答】
# {"产品名称": "华为 MatepadPro", "产品卖点": "高清大屏，长效续航"}
