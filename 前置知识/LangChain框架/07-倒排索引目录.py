"""
=================================================================
本章节倒排索引目录
=================================================================
前置学习 —— LangChain 框架（项目实际使用部分）
"""

# =====================================================================
# 核心概念与架构
# =====================================================================
# LangChain         围绕 LLMs 的开发框架，为各种大模型提供统一接口
# RAG               检索增强生成，本项目核心架构
# init_chat_model   LangChain 统一模型创建函数，项目用它对接通义千问
# invoke()          同步调用模型，一次性返回完整结果
# stream()          流式调用模型，逐 token 返回

# =====================================================================
# Python SDK 类与函数
# =====================================================================
# from langchain.chat_models import init_chat_model    # 创建大模型（项目唯一入口）
# from langchain_openai.chat_models import ChatOpenAI   # OpenAI 兼容模型
# from langchain_ollama import ChatOllama, OllamaLLM    # Ollama 本地模型
# from langchain_ollama import OllamaEmbeddings          # Ollama 嵌入模型
# from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace 嵌入模型
# from langchain_community.embeddings import DashScopeEmbeddings  # 阿里通义嵌入模型
#
# from langchain_core.prompts import PromptTemplate      # 提示词模板（项目使用）
# from langchain_core.prompts import FewShotPromptTemplate  # 少量示例模板
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  # 消息类型
#
# from langchain_core.documents import Document          # 文档对象（项目核心）
# from langchain_core.document_loaders import BaseLoader  # 加载器基类（项目继承）
# from langchain_community.document_loaders import TextLoader  # 文本加载器
# from langchain_community.document_loaders import UnstructuredMarkdownLoader  # MD加载器
#
# from langchain_text_splitters import CharacterTextSplitter          # 普通字符分割器
# from langchain_text_splitters import RecursiveCharacterTextSplitter  # 递归字符分割器
# from langchain_text_splitters import MarkdownTextSplitter            # MD分割器
# from langchain_text_splitters import MarkdownHeaderTextSplitter      # MD标题分割器

# =====================================================================
# 关键参数与配置项
# =====================================================================
# model           模型名称，如 "qwen3.7-flash-2026-07-15"
# base_url        API 接口地址
# api_key         API 密钥（从环境变量 AliApiKey 获取）
# temperature     温度系数（0=严谨，0.7=平衡，1.0=创意）
# max_tokens      最大生成 token 数
# chunk_size      分割块大小（字符数）
# chunk_overlap   分割块重叠字符数（建议 chunk_size 的 10%~20%）
# separators      递归分割器的分隔符列表

# =====================================================================
# 本项目未使用但原文档涉及的内容（已移除）
# =====================================================================
# Chain 链             —— LLMChain / SequentialChain / 并行链 / 分支链 / LCEL
# 输出解析器           —— StrOutputParser / JsonOutputParser / PydanticOutputParser
# History 会话记忆     —— RunnableWithMessageHistory / RedisChatMessageHistory
# Indexes 组件         —— RecordManager
# 检索器               —— Retriever / as_retriever / similarity_search
# Agent 智能体         —— create_react_agent / middleware
# ChatPromptTemplate   —— from_messages / MessagesPlaceholder
# CsvLoader / JsonLoader / PdfLoader / DirectoryLoader  # 项目不用的加载器
# 代码分割器           —— RecursiveCharacterTextSplitter.from_language

# =====================================================================
# 部署与工具
# =====================================================================
# Ollama           本地模型运行环境 https://ollama.com/
# 阿里百炼平台      通义千问 API 服务 https://bailian.console.aliyun.com/
# MTEB             文本嵌入模型排行榜 https://huggingface.co/spaces/mteb/leaderboard
