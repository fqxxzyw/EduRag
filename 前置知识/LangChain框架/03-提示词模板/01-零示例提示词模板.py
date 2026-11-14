"""
=================================================================
提示词模板 —— 零示例提示词模板（Zero-shot）
=================================================================
PromptTemplate 是最基础的提示词模板，支持 {变量} 动态填入。

项目中所有 6 个提示词模板（rag_prompt / hyde_prompt / subquery_prompt /
backtracking_prompt / rag_prompt_withhistory / strategy_selector）
都使用 PromptTemplate + input_variables 方式。
"""
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

api_key = os.environ.get("AliApiKey")

# ==============================
# 1. 创建提示词模板对象
# ==============================
prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，简单回答。"
)

# 另外一种写法
template_str = '我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字，简单回答。'
prompt_template2 = PromptTemplate(
    input_variables=['lastname', 'gender'],
    template=template_str
)


# ==============================
# 2. 创建模型对象
# ==============================
model = init_chat_model(
    "openai:qwen3.7-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    temperature=0,
)


# ==============================
# 3. 方式一：先 format 再 invoke
# ==============================
# 获取完整提示词：提示词模板的 format 方法注入信息
prompt_text = prompt_template.format(lastname="张", gender="女儿")
print(f"【完整提示词】{prompt_text}")
# 通过模型对象，传入完整提示词，获取结果
res = model.invoke(input=prompt_text)
print(f"【模型回答】{res.content}")
# 运行结果（示例）：
# 【完整提示词】我的邻居姓张, 刚生了女儿, 你帮我起个名字，简单回答。
# 【模型回答】推荐：张悦宁（寓意喜乐安宁，音韵柔和）。祝小宝宝健康平安！


# ==============================
# 4. 方式二：模板对象的 format 和 invoke 区别
# ==============================
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("我的邻居是：{lastname}，最喜欢：{hobby}")

# format 接收关键字参数
res1 = template.format(lastname="张大明", hobby="钓鱼")
print(f"\n【format 结果】{res1}，类型：{type(res1)}")
# 运行结果：【format 结果】我的邻居是：张大明，最喜欢：钓鱼，类型：<class 'str'>

# invoke 接收字典
res2 = template.invoke({"lastname": "周杰轮", "hobby": "唱歌"})
print(f"【invoke 结果】{res2}，类型：{type(res2)}")
# 运行结果：【invoke 结果】text='我的邻居是：周杰轮，最喜欢：唱歌'，类型：<class 'langchain_core.prompt_values.StringPromptValue'>

print(f"【invoke to_string】{res2.to_string()}")  # 只获取内容
# 运行结果：【invoke to_string】我的邻居是：周杰轮，最喜欢：唱歌
