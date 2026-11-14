# LangChain学习笔记

## 什么是LangChain

LangChain是一个用于开发大语言模型应用的框架。

### 核心概念

- **Models**: 大语言模型接口
- **Prompts**: 提示词模板管理
- **Chains**: 链式调用组合

## 快速开始

### 安装

```bash
pip install langchain
```

### 基本用法

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4")
result = model.invoke("Hello!")
print(result.content)
```

## 进阶内容

### 文档处理

LangChain提供了丰富的文档加载器和分割器。

### 向量存储

支持多种向量数据库，如FAISS、Chroma、Milvus等。
