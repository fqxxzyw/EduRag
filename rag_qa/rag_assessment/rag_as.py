"""
评估代码
"""
import json, os
# todo 1. 核心导包
from ragas import evaluate# ragas.evaluate函数，用于执行RAG评估
from ragas.run_config import RunConfig # 配置大模型并发数、超时时间
# 导入ragas的评估指标，包括忠实度、答案相关性、上下文精确率、上下文召回率
from ragas.metrics import (
    faithfulness, # 忠实度: 答案是否完全基于提供的上下文(判断幻觉)
    answer_relevancy, # 答案相关性: 答案是否紧扣问题(答的准不准)
    context_precision, # 上下文精确率: 上下文是否为有效内容、排序是否合理(噪音情况)
    context_recall # 上下文召回率: 上下文是否包含回答问题所需全部信息(信息完整度)
)
from datasets import Dataset  # Dataset类构建RAGAS所需的数据格式


# todo 2.模型及相应配置:
#  选择用于评估的LLM(大语言模型)和嵌入模型(用于计算相似度, 用于语义计算)

''' ⚠️ 无论使用云上模型还是本地模型，都请耐心等待，都很慢 '''
# TODO 使用 Ollama 本地模型（qwen2.5:7b + mxbai-embed-large）
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 大语言模型：qwen2.5:7b
llm = ChatOllama(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",
    # 修改默认配置，可一定程度提升速度
    # temperature=0,          # todo 评估固定随机性，稳定+提速
    # num_ctx=1024,           # 强制上下文窗口1024，可以降低显存
    # num_predict=512,        # 如果评估不需要长文本，就限制最大输出
    # request_timeout=600,    # 接口超时 10分钟，防止中途断开
    # extra_body = {
    #     "format": "json" # todo 强制模型按json规范输出
    # }
)
# 向量嵌入模型：mxbai-embed-large:latest
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
    base_url="http://localhost:11434"
)
# 配置并发数、超时时间
run_config = RunConfig(
    # timeout=600,      # 单次请求超时，单位秒
    max_workers=1,      # 本地ollama强制为串行、不并发！
    max_retries=2       # 失败重试次数
)


# # TODO 使用 阿里云百炼 模型（qwen-plus + text-embedding-v4）
# from langchain_community.chat_models import ChatTongyi
# from langchain_community.embeddings import DashScopeEmbeddings
# llm = ChatTongyi(
#     model="qwen-plus",
#     api_key=os.getenv('AliApiKey')
# )
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v4",
#     dashscope_api_key=os.getenv('AliApiKey'),
# )
# # 配置并发数、超时时间
# run_config = RunConfig(
#     # timeout=120,      # 单次请求超时，单位秒
#     max_workers=4,      # 并发请求数，根据API限流调整（一般 4~16）
#     max_retries=5       # 失败重试次数
# )

# todo 3.加载评估数据集:
# 读取包含问题, 答案, 上下文, 真实答案的JSON文件
with open('rag_evaluate_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'加载的评估样本数量: {len(data)}')
if data:
    print(f'查看第一条评估样本: {data[0]}')

# todo 4.转换数据格式:
# 将原始数据转换为RAGAS所需的DataSet格式
eval_data = {
    'question': [item['question'] for item in data],         # 问题列表
    'answer': [item['answer'] for item in data],              # 生成答案列表
    'contexts': [item['context'] for item in data],          # 上下文列表
    'ground_truth': [item['ground_truth'] for item in data]   # 标准答案列表
}
dataset = Dataset.from_dict(eval_data)
print(f'转换后的数据集: {dataset}')

# todo 5. 执行评估:
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config # 加入配置
)

# todo 6. 输出并保存评估结果
print(f'RAGAS评估结果: {result}')
# 结果保存为CSV文件
result.to_pandas().to_csv('rag_as.csv', index=False, encoding='utf-8-sig')
print("✅ 评估结果已保存至 rag_as.csv")