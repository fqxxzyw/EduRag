"""
=================================================================
LangChain 介绍
=================================================================
大模型RAG应用开发框架
基于LangChain 1.3.1版本
官方文档 https://docs.langchain.com/

LangChain 是帮你轻松搭建 RAG 的开发框架。
没有 LangChain 也能手写 RAG，有了 LangChain 不用造轮子，几行代码就能做出 RAG 问答机器人。

LangChain 由 Harrison Chase 创建于 2022 年 10 月，
它围绕 LLMs（大语言模型）建立，核心理念是：
    为各种 LLMs 实现通用的接口，把 LLMs 相关的组件 "链接" 在一起，
    简化 LLMs 应用的开发难度，方便开发者快速地开发复杂的 LLMs 应用。

本项目（EduRAG）中实际用到的 LangChain 模块：
    1. Models模型 —— init_chat_model 创建大模型 + invoke/stream 调用
    2. 提示词模板 —— PromptTemplate 构建可复用的提示词
    3. 文档加载器 —— BaseLoader 继承自定义加载器 + TextLoader + MarkdownLoader
    4. 文档分割器 —— CharacterTextSplitter / RecursiveCharacterTextSplitter / MarkdownTextSplitter
    5. 向量存储器 —— 概念了解（项目直接使用 pymilvus 操作 Milvus，不走 LangChain 封装）
"""

# LangChain 提供的核心功能（本项目用到的部分已标注 ★）：
#   ★ 提示词模板构建功能
#   ★ 调用各类模型的功能
#     会话记忆的相关功能          —— 项目用 MySQL 手动管理，不用 LangChain Memory
#   ★ 各类文档处理的功能
#     构建各种功能链，并执行      —— 项目不用 Chain，直接 llm.invoke(prompt)
#     构建Agent智能体             —— 属于 SmartVoyage 项目，不属于 EduRAG
