# EduRAG 智慧问答系统

EduRAG 是一个面向 IT 教育培训场景的 **RAG（Retrieval-Augmented Generation，检索增强生成）** 问答系统示例项目。它把「FAQ 精确命中」「本地知识库语义检索」「大模型生成」三类能力组合在一起：

- **FAQ 优先**：先用 MySQL 中的问答库做 BM25 + softmax 相似度匹配，命中阈值（默认 0.85）就直接返回标准答案。
- **RAG 兜底**：FAQ 未命中时，对本地文档（PDF / DOCX / PPTX / 图片 / Markdown / TXT）构建的向量库做混合检索与重排序，再交给大模型基于上下文生成答案。
- **意图与策略**：用微调后的 BERT 判断「通用知识 / 专业咨询」，并对专业咨询再选择「直接检索 / 回溯问题检索 / 子查询检索 / HyDE 假设问题检索」四种策略。
- **会话能力**：支持多轮对话历史（MySQL 持久化）、Redis 答案缓存、FastAPI WebSocket 流式输出和内置聊天前端。

> 说明：本项目是教学/课程配套工程，代码强调流程完整与可读性。用于生产前请重点阅读文末「鲁棒性与已知风险」并做加固。

---

## 1. 整体架构

### 1.1 离线数据处理（建库）

```text
rag_qa/data/<学科>_data/
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 文档加载器 rag_qa/edu_document_loaders                       │
│   .txt/.md → TextLoader / UnstructuredMarkdownLoader         │
│   .pdf     → OCRPDFLoader（PyMuPDF 提取文本 + 大图 OCR）      │
│   .docx    → OCRDOCLoader（python-docx 段落/表格/图片 OCR）   │
│   .pptx    → OCRPPTLoader（python-pptx + 图片 OCR）           │
│   .jpg/.png→ OCRIMGLoader（整图 OCR）                         │
└─────────────────────────────────────────────────────────────┘
        │  LangChain Document（metadata: source/file_path/timestamp）
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 父子分块 process_documents.py                                │
│   父块 1200 字（parent_content 存进 metadata）               │
│   子块 300 字 / 重叠 50（真正参与向量化）                     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ BGE-M3 双向量（milvus_model）                                │
│   dense  1024 维 → FLOAT_VECTOR / FLOAT16_VECTOR             │
│   sparse 稀疏词权重 → SPARSE_FLOAT_VECTOR                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Milvus：database=edurag，collection=edurag_doc
  稠密索引 IVF_FLAT(IP)，稀疏索引 SPARSE_INVERTED_INDEX(IP)
  写入方式 upsert（md5(正文) 作为主键，天然去重/增量更新）
```

### 1.2 在线问答（查询）

```text
用户输入
   │
   ├─ 日常问候正则 → 直接模板回复
   │
   ▼
MySQL FAQ：Redis 缓存 → 问题分词(jieba) → BM25 → softmax
   │
   ├─ 最高概率 ≥ 0.85 → 返回标准答案（写 Redis 缓存）
   │
   ▼
BERT 意图识别（bert-query-classifier-500）
   │
   ├─ 通用知识 → 直接调用 Qwen 生成（不检索）
   │
   ▼ 专业咨询
LLM 策略选择（直接 / 回溯 / 子查询 / HyDE）
   │
   ▼
Milvus 混合检索 Top-K 子块（可按 source 学科过滤）
   │
   ▼
按 parent_id 去重 → 取回父块 → BGE-Reranker-large 精排 → Top-M
   │
   ▼
Prompt（+ 历史对话）→ Qwen 流式生成
   │
   ▼
返回答案：同时写 Redis 缓存、写 MySQL 会话历史
```

---

## 2. 技术栈与版本

当前开发环境（`.venv`）为 **Python 3.10.22**，`requirements.txt` 已锁定主要依赖版本。

| 类别 | 组件 | 说明 |
|---|---|---|
| 应用框架 | FastAPI 0.136.3 + Uvicorn 0.49.0 | HTTP / WebSocket / 静态页面 |
| LLM 调用 | LangChain 1.x + `langchain-openai` + 阿里云百炼 | OpenAI 兼容模式调用通义千问 |
| 向量模型 | BGE-M3（`milvus-model`） | 同时产出稠密 + 稀疏向量 |
| 重排模型 | BGE-Reranker-large | CrossEncoder 精排 |
| 意图识别 | BERT-base-chinese + 微调分类器 | HuggingFace Transformers 5.x |
| 向量库 | Milvus 2.6.x（pymilvus 2.6.15） | 稠密 + 稀疏混合检索 |
| 关系库 | MySQL 8.0（PyMySQL） | FAQ 知识库、会话历史 |
| 缓存 | Redis 8.x（redis-py 8.0） | 答案缓存、问题列表缓存 |
| 文档解析 | PyMuPDF / python-docx / python-pptx / Pillow / OpenCV | PDF、Word、PPT、图片 |
| OCR | RapidOCR（ONNX Runtime） | `_edu_ocr.py` 自动选择 GPU 版或 CPU 版 |
| 分词 | jieba | FAQ 检索、BM25 |
| 评估 | RAGAS 0.4.0 | 离线评估四个指标 |
| 运行底座 | Docker Desktop / Docker Compose | MySQL、Redis、etcd、MinIO、Milvus、Attu |

---

## 3. 目录结构

```text
EduRag/
├── main.py                     # 集成入口：FAQ + RAG（命令行参数 --data / --dir）
├── main_plus.py                # 集成入口（带历史会话，流式生成）
├── rag_api.py                  # FastAPI 生产/容器入口（HTTP + WebSocket + 静态页面）
├── rag_api_bak.py              # FastAPI 本地开发版（可直接 python 运行）
├── config.ini                  # 全局配置：数据库/模型/检索/设备
├── requirements.txt            # Python 依赖
├── docker-compose.yml          # 基础设施：MySQL/Redis/etcd/MinIO/Milvus/Attu
├── Dockerfile.dev              # 开发镜像（含默认 CMD）
├── Dockerfile.prod             # 生产镜像（复制代码，不内置 CMD）
├── install-edurag.bat          # Windows 一键启动本地 Docker 基础设施
├── _get_version.py             # 辅助脚本：打印解释器/依赖并导出 requirements2.txt
├── base/                       # 公共基础能力
│   ├── __init__.py             #   统一导出 Config / logger / 路径工具
│   ├── config.py               #   读取 config.ini，进程内单例
│   ├── logger.py               #   控制台 + 文件日志
│   └── path_tools.py           #   项目根目录与绝对路径换算
├── mysql_qa/                   # FAQ（BM25 + softmax）子系统
│   ├── main.py                 #   命令行入口
│   ├── core_search.py          #   BM25 + softmax 检索核心
│   ├── data/JP学科知识问答.csv  #   FAQ 种子数据（约 467 条）
│   └── utils/
│       ├── mysql_client.py     #   MySQL 连接、建表、导入、查询
│       ├── redis_client.py     #   Redis 缓存（JSON / 纯文本）
│       ├── preprocess.py       #   jieba 分词预处理
│       └── _check_redis.py     #   Redis 连通性与 key 前缀演示脚本
├── rag_qa/                     # RAG 子系统
│   ├── main.py                 #   子系统命令行入口（数据处理 / 交互问答）
│   ├── core/
│   │   ├── process_documents.py#   遍历目录、选加载器、父子分块
│   │   ├── vector_store.py     #   Milvus 建集合、写入、混合检索 + 重排
│   │   ├── rag_system.py       #   RAG 主流程与 4 种检索策略
│   │   ├── query_classifier.py #   BERT 意图识别（含训练/评估/预测）
│   │   ├── strategy_selector.py#   LLM 选择检索策略
│   │   ├── llm_provider.py     #   LLM 单例工厂
│   │   └── prompts.py          #   各类 PromptTemplate
│   ├── edu_document_loaders/   #   自定义文档加载器
│   │   ├── __init__.py         #     sys.path 处理 + 统一导出
│   │   ├── _edu_ocr.py         #     RapidOCR 对象工厂
│   │   ├── edu_docloader.py    #     Word(.docx) + 图片 OCR
│   │   ├── edu_pdfloader.py    #     PDF 文本 + 大图 OCR + 旋转校正
│   │   ├── edu_pptloader.py    #     PPTX 文本/表格/图片 OCR
│   │   ├── edu_imgloader.py    #     单文件图片 OCR
│   │   └── samples/            #     四种格式的测试样例
│   ├── edu_text_spliter/       #   自定义文本分割器
│   │   ├── __init__.py         #     统一导出
│   │   ├── edu_chinese_recursive_text_splitter.py # 中文标点递归分割
│   │   └── edu_model_text_spliter.py              # BERT 语义分段（可选）
│   ├── classify_data/          # 意图识别训练数据与生成脚本
│   │   ├── generate_classify_data.py # 规则 + Qwen 生成标注数据
│   │   ├── model_generic_500.json    # 默认训练集（500 条）
│   │   └── model_generic_1830.json   # 备用训练集
│   ├── models/                 # 本地模型目录（被 .gitignore 忽略）
│   │   ├── bge-m3/             #   向量模型
│   │   ├── bge-reranker-large/ #   重排模型
│   │   ├── bert-base-chinese/  #   预训练基座
│   │   ├── bert-query-classifier-500/ # 微调后的意图分类器
│   │   ├── nlp_bert_document-segmentation_chinese-base/ # 语义分段（可选）
│   │   └── m3e-base/           #   当前主流程未使用
│   ├── data/                   # 知识文档（被 .gitignore 忽略）
│   │   ├── ai_data/            #   学科目录：ai
│   │   └── java_data/          #   学科目录：java
│   └── rag_assessment/         # RAGAS 离线评估
│       ├── rag_as.py           #   评估脚本
│       ├── rag_evaluate_data.json
│       └── rag_as*.csv         #   历史评估结果
├── static/index.html           # 内置聊天前端（单文件 HTML/CSS/JS）
├── 前置知识/                    # LangChain / Milvus 教学示例，不参与运行
├── logs/                       # 运行日志（按天生成）
├── docker-data/                # Docker 持久化数据（不要随意删除）
├── 项目程序启动说明.md          # 原始启动说明
├── 本机Docker使用说明.md        # Windows + Docker Desktop 说明
└── 项目支持虚拟机信息.md        # 教学虚拟机信息与基础设施 compose
```

> 注意：`rag_qa/models/`、`rag_qa/data/`、`logs/`、`docker-data/` 已被 `.gitignore` / `.dockerignore` 排除；**模型和知识文档需要单独准备**。

---

## 4. 快速开始（Windows 本机 + Docker Desktop）

### 4.0 前置条件

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（也支持 Linux；以下命令以 PowerShell 为例） |
| Python | 3.10（建议 3.10.22，与当前 `.venv` 一致） |
| Docker | Docker Desktop，含 Docker Compose v2，且已启动 |
| 内存/磁盘 | 建议 ≥ 8 GB 内存；模型 + 镜像 + 数据建议预留 30 GB 以上 |
| API Key | 阿里云百炼（DashScope）API Key，供通义千问调用 |
| 端口 | 3307(MySQL)、6379(Redis)、19530/9091(Milvus)、3000(Attu)、9000/9001(MinIO)、8000(API) |

### 4.1 创建 Python 环境并安装依赖

```powershell
cd D:\soft\python\EduRag

# 创建虚拟环境（若已有 .venv 可跳过）
python -m venv .venv

# 激活
.\.venv\Scripts\Activate.ps1
# 如提示禁止运行脚本：Set-ExecutionPolicy -Scope Process RemoteSigned

# 安装依赖（体积较大，建议配置国内镜像）
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

> - `requirements.txt` 中 `torch==2.12.1` 在 Windows / Linux 上默认安装 CPU 版本。需要 GPU 时，请先按 PyTorch 官方说明安装 CUDA 版本，再执行 `pip install -r requirements.txt`。
> - Dockerfile 中单独从 `https://download.pytorch.org/whl/cpu` 安装 CPU 版 torch，避免容器内拉取 CUDA 大包。

### 4.2 准备模型文件

将以下模型放入 `rag_qa/models/`（目录名必须与 `config.ini` 一致）：

| 目录 | 用途 | 必需 |
|---|---|---|
| `rag_qa/models/bge-m3` | 稠密 + 稀疏向量 | ✅ |
| `rag_qa/models/bge-reranker-large` | 检索结果精排 | ✅ |
| `rag_qa/models/bert-base-chinese` | 意图分类基座 | ✅ |
| `rag_qa/models/bert-query-classifier-500` | 微调后的意图分类器 | ✅ |
| `rag_qa/models/nlp_bert_document-segmentation_chinese-base` | `AliTextSplitter` 语义分段 | ❌（主流程未用） |
| `rag_qa/models/m3e-base` | 备用向量模型 | ❌（主流程未用） |

模型来源：项目配套网盘 / HuggingFace / ModelScope。由于模型目录被 `.gitignore` 排除，从 Git 仓库拉取后需要手动补齐。

### 4.3 准备知识文档

按学科把文档放到 `rag_qa/data/<学科>_data/`，学科名必须出现在 `config.ini` 的 `[app] valid_sources` 中：

```text
rag_qa/data/
├── ai_data/
│   ├── LLM基础知识.pdf
│   └── 人工智能就业课课程大纲.md
├── java_data/...
├── test_data/...
├── ops_data/...
└── bigdata_data/...
```

支持的扩展名：`.txt`、`.md`、`.pdf`、`.docx`、`.ppt`、`.pptx`、`.jpg`、`.jpeg`、`.png`。

### 4.4 配置 `config.ini` 与 API Key

```ini
[mysql]
host = 127.0.0.1
port = 3307          ; 本机 Docker 映射端口（容器内是 3306）
user = root
password = 123456
database = edurag
table = jpkb

[redis]
host = 127.0.0.1
port = 6379
password = 123456
db = 0
ttl = 180            ; 缓存 180 秒；要缓存一天可改为 86400

[milvus]
host = 127.0.0.1
port = 19530
database_name = edurag
collection_name = edurag_doc

[model]
dashscope_model = qwen3.7-flash
# dashscope_api_key = sk-xxx   ; 不推荐写明文
env_api_key = AliApiKey        ; 从环境变量 AliApiKey 读取
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
embedding_path = rag_qa/models/bge-m3
rerank_path = rag_qa/models/bge-reranker-large
bert_base_path = rag_qa/models/bert-base-chinese
bert_query_classifier_path = rag_qa/models/bert-query-classifier-500
bert_train_data = rag_qa/classify_data/model_generic_500.json

[device]
device = cpu         ; 无 GPU 时用 cpu；切到 cuda 前请阅读第 8 节
```

设置 API Key（只在当前 PowerShell 会话生效）：

```powershell
$env:AliApiKey = "sk-替换为你的API Key"
```

> `base/config.py` 读取逻辑：如果 `[model]` 中存在 `dashscope_api_key` 就用它；**取不到时会捕获异常**，改为读取环境变量 `env_api_key` 指定的变量名。当前配置使用环境变量方式。

### 4.5 启动基础设施（Docker）

```powershell
# 启动 Docker Desktop，确认引擎就绪后：
docker compose up -d
docker compose ps
```

首次启动会下载镜像，Milvus 可能需要 1～2 分钟才健康。查看日志：

```powershell
docker compose logs --tail 100 milvus
docker compose logs --tail 100 mysql
```

服务与端口：

| 服务 | 宿主端口 | 说明 |
|---|---|---|
| MySQL | 127.0.0.1:3307 | 自动创建库 `edurag` |
| Redis | 127.0.0.1:6379 | 密码 `123456` |
| Milvus | 127.0.0.1:19530 / 9091 | 9091 为健康检查 |
| Attu | http://127.0.0.1:3000 | Milvus 可视化管理 |
| MinIO | 127.0.0.1:9000（控制台 9001） | Milvus 对象存储 |
| etcd | 127.0.0.1:2379 | Milvus 元数据 |

数据持久化在 `./docker-data/`，停止服务用 `docker compose stop`（保留数据），不要执行 `down -v`。

> 也可以运行 `install-edurag.bat`：它会自动检测空闲端口、生成 `docker-compose.local.yml`、备份并改写 `config.ini`，然后启动容器。它**不会**导入数据、安装 Python 依赖或配置 API Key。

### 4.6 创建 Milvus 数据库

`VectorStore` 初始化时会连接 `database_name = edurag`，**该数据库必须先存在**。可通过 Attu 创建，或执行：

```powershell
python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530'); print(c.list_databases()); c.create_database('edurag') if 'edurag' not in c.list_databases() else None; c.close()"
```

### 4.7 初始化 MySQL FAQ 数据（首次）

```powershell
python -c "from mysql_qa.utils.mysql_client import database_init; database_init()"
```

该命令会：

1. 创建数据库 `edurag`（若不存在）；
2. **删除并重建** `jpkb` 表；
3. 导入 `mysql_qa/data/JP学科知识问答.csv`（约 467 条）。

> ⚠️ `creat_table()` 会 `DROP TABLE IF EXISTS`，**重复执行会清空表中已有数据**。只在首次初始化或确认可丢弃数据时执行。

### 4.8 离线构建向量库（首次 / 文档更新后）

```powershell
# 方式一：命令行参数（推荐）
python main.py --data

# 方式二：指定数据根目录（相对或绝对路径都支持）
python main.py --data --dir rag_qa/data

# 方式三：直接调用函数
python -c "from main import main; main(query_mode=False)"
```

执行内容：遍历 `valid_sources` 中每个学科的 `*_data` 目录 → 加载文档 → 父子分块 → BGE-M3 向量化 → `upsert` 到 Milvus。

### 4.9 启动问答

```powershell
# 1) 纯 RAG 交互式问答（不含 FAQ）
python -c "from rag_qa.main import main; main(query_mode=True)"

# 2) 集成问答（FAQ + RAG，命令行）
python main.py

# 3) 集成 + 历史会话（命令行，流式打印）
python main_plus.py

# 4) Web API + 内置前端
python -m uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

浏览器访问 <http://127.0.0.1:8000>。健康检查：<http://127.0.0.1:8000/health>。

---

## 5. 运行方式一览

| 形态 | 入口 | 需要启动的服务 | 特点 |
|---|---|---|---|
| 纯 FAQ | `python -m mysql_qa.main` / `MySQLQaSystem().main()` | MySQL、Redis | BM25 + softmax，无大模型 |
| 纯 RAG | `rag_qa/main.py` 的 `main(query_mode=True)` | Milvus + 模型 + API Key | 交互式问答，无 FAQ |
| RAG 建库 | `main(query_mode=False)` / `--data` | Milvus + 模型 | 离线向量化 |
| 集成问答 | `main.py` | MySQL、Redis、Milvus + 模型 | 先 FAQ 后 RAG |
| 集成 + 历史 | `main_plus.py` | 同上 | 流式输出、多轮历史 |
| Web API | `rag_api.py`（`uvicorn rag_api:app`） | 同上 | HTTP + WebSocket + 前端 |
| 本地 API 备用 | `rag_api_bak.py` | 同上 | 直接 `python rag_api_bak.py` |
| 容器部署 | `Dockerfile.prod` + `rag_api:app` | 外部基础设施 | 需挂载配置/模型/数据 |
| RAG 评估 | `rag_qa/rag_assessment/rag_as.py` | Ollama 或 DashScope | RAGAS 四指标 |

### 5.1 Docker 部署 API（可选）

```powershell
# 构建生产镜像（models/data 被 .dockerignore 排除，需运行时挂载）
docker build -f Dockerfile.prod -t rag-api:latest .

docker run -d `
  --name rag-api `
  -p 8000:8000 `
  -v ${PWD}/config.ini:/app/config.ini `
  -v ${PWD}/rag_qa/models:/app/rag_qa/models `
  -v ${PWD}/rag_qa/data:/app/rag_qa/data `
  -e AliApiKey="sk-替换为你的API Key" `
  rag-api:latest `
  uvicorn rag_api:app --host 0.0.0.0 --port 8000 --workers 2
```

> - `Dockerfile.prod` 不写 `CMD`，必须在 `docker run` 末尾给出启动命令。
> - 容器内访问宿主 Docker 服务时，`config.ini` 中的 `host` 需要改成宿主 IP / `host.docker.internal`，或改用自定义网络；当前 `config.ini` 的 `127.0.0.1` 只适用于**程序直接跑在宿主机**的场景。
> - `--workers 2` 会让每个进程各加载一套 Embedding / Reranker / BERT 模型，内存占用翻倍。

### 5.2 Web API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 内置聊天页面 `static/index.html` |
| GET | `/health` | 健康检查 |
| GET | `/api/sources` | 返回 `Config.VALID_SOURCES` |
| POST | `/api/create_session` | 创建 UUID 会话 |
| GET | `/api/history/{session_id}` | 最近 5 轮历史 |
| DELETE | `/api/history/{session_id}` | 清空会话历史 |
| POST | `/api/query` | 非流式查询；FAQ 命中返回答案，否则提示改用 WebSocket |
| WS | `/api/stream` | 流式查询，消息体 `{query, source_filter, session_id}` |

WebSocket 消息类型：`start` / `token` / `end` / `error`。

---

## 6. 代码逐文件说明

### 6.1 根目录

| 文件 | 作用 | 关键点 |
|---|---|---|
| `main.py` | 集成入口：FAQ 优先 + RAG 兜底，支持 `--data/--dir` 命令行 | 主逻辑 `main(query_mode, data_path)`；`cmd_run()` 解析参数 |
| `main_plus.py` | 集成入口的历史会话版 | `IntegratedQASystem` 全局单例；`query()` 生成器逐 token 产出；会话历史存 MySQL |
| `rag_api.py` | FastAPI 应用（Web/容器入口） | 全局 `qa_system = IntegratedQASystem()`；HTTP + WebSocket + 静态页面 + CORS |
| `rag_api_bak.py` | API 的本地开发版 | 与 `rag_api.py` 基本相同，区别是可 `python` 直接运行、FAQ 流式标志不同 |
| `config.ini` | 全局配置 | 数据库、模型路径、检索参数、设备、业务学科列表 |
| `requirements.txt` | 依赖锁定 | 见第 2 节；含 torch（CPU）与 RAGAS |
| `docker-compose.yml` | 基础设施编排 | MySQL/Redis/etcd/MinIO/Milvus/Attu，均含健康检查与资源限制 |
| `Dockerfile.dev` | 开发镜像 | 装系统依赖、CPU torch、requirements，CMD 启动 uvicorn |
| `Dockerfile.prod` | 生产镜像 | 与 dev 类似，但 `COPY . .` 且不写 CMD |
| `install-edurag.bat` | Windows 一键基础设施 | 自检 Docker、选空闲端口、备份并改写 config、生成 local compose、启动服务 |
| `_get_version.py` | 环境信息辅助脚本 | 打印 Python/pip/包版本并导出 `requirements2.txt` |
| `static/index.html` | 单文件聊天前端 | 会话列表、学科标签、WebSocket 流式渲染、历史恢复 |
| `本机Docker使用说明.md`/`项目程序启动说明.md`/`项目支持虚拟机信息.md` | 原始说明 | 与本 README 有重叠，可作为补充 |

### 6.2 `base/` 公共层

| 文件 | 作用 |
|---|---|
| `base/__init__.py` | 统一导出 `Config`、`logger`、`get_abs_path`、`get_project_root`，全项目统一 `from base import ...` |
| `base/config.py` | `configparser` 读取 `config.ini`，把各段配置转成大写属性；模块级实例 `Config = Config()` 是进程内单例 |
| `base/logger.py` | 创建 `Edu RAG` 日志器：控制台 + `logs/Edu RAG_YYYYMMDD.log`；用 `if logger.handlers: return` 防止重复挂载 Handler |
| `base/path_tools.py` | 基于 `__file__` 推导项目根目录，`get_abs_path()` 统一把配置中的相对路径转绝对路径，避免依赖当前工作目录 |

### 6.3 `mysql_qa/` FAQ 子系统

| 文件 | 作用 |
|---|---|
| `mysql_qa/main.py` | `MySQLQaSystem`：命令行问答、记录耗时、退出时关闭连接 |
| `mysql_qa/core_search.py` | `Bm25SoftmaxSearch`：加载问题列表（Redis 优先、MySQL 回源）→ 构建 `BM25Okapi` → 查询时 jieba 分词 → BM25 分数 → softmax 归一化 → 取最高分与阈值比较 → 命中则从 Redis/MySQL 取答案并回填缓存 |
| `mysql_qa/utils/mysql_client.py` | `MysqlClient`：连接、建表、删表重建、CSV 导入、取全部问题、按问题取答案、断线重连；`database_init()` 完成首次建库建表导数据 |
| `mysql_qa/utils/redis_client.py` | `RedisClient`：`set_data/get_data`（JSON，用于问题列表）、`set_answer/get_answer`（纯文本，用于答案）、`clear_all_data()` |
| `mysql_qa/utils/preprocess.py` | `preprocess_text()`：英文转小写 + jieba 精准/全模式分词，可返回生成器或列表 |
| `mysql_qa/utils/_check_redis.py` | 演示 Redis 连接、写入、`scan_iter` 前缀扫描（`keys` 在数据量大时阻塞，生产慎用） |

### 6.4 `rag_qa/core/` 核心层

| 文件 | 作用 |
|---|---|
| `process_documents.py` | 按扩展名选择加载器；`os.walk` 遍历目录；给每个 `Document` 补 `source/file_path/timestamp`；父块切分后把 `parent_content/parent_id` 写入子块 metadata；最终返回全部子块 |
| `vector_store.py` | 初始化 BGE-M3 + BGE-Reranker；创建/加载 Milvus 集合与索引；`add_documents()` 用 md5(正文) 做主键并 `upsert`；`hybrid_search_with_rerank()` 做稠密+稀疏混合检索、按父块去重、CrossEncoder 重排、截取 Top-M |
| `rag_system.py` | `RAGSystem`：组合向量库、分类器、策略选择器、LLM；实现直接检索、回溯检索、子查询检索、HyDE 检索；`generate_answer()`（非流式）与 `generate_answer_withhistory()`（流式、带历史） |
| `query_classifier.py` | `QueryClassifier`：加载 BERT 分类器；训练数据划分、编码、Dataset 构建、TrainingArguments、Trainer 训练/评估；`predict_category()` 输出「通用知识/专业咨询」 |
| `strategy_selector.py` | `StrategySelector`：用提示词让 LLM 在四种策略中选一个；异常时安全回退到「直接检索」 |
| `llm_provider.py` | `get_llm()` 单例工厂：`init_chat_model("openai:<model>", api_key, base_url, temperature=0.1)` |
| `prompts.py` | 五套 `PromptTemplate`：基础 RAG、HyDE、子查询、回溯、带历史 RAG |

### 6.5 `rag_qa/edu_document_loaders/` 文档加载器

| 文件 | 作用 | 备注 |
|---|---|---|
| `__init__.py` | 把本目录插入 `sys.path`，再导出各 Loader | 让 Loader 内部可以用 `from _edu_ocr import get_ocr` 的绝对导入 |
| `_edu_ocr.py` | `get_ocr(use_cuda=True)`：优先 `rapidocr_paddle`，失败再 `rapidocr_onnxruntime` | 只捕获 `ImportError`；GPU 版装了就默认走 GPU |
| `edu_docloader.py` | `.docx`：按文档原始顺序遍历段落/表格；段落内图片 OCR；表格逐行逐单元格取文本 | 图片 ID 通过 `a:blip/@r:embed` 取 |
| `edu_pdfloader.py` | `.pdf`：`page.get_text("text")` + 大图 OCR；处理页面旋转 | 只 OCR 宽/高超过页面 60% 的图片（`PDF_OCR_THRESHOLD`） |
| `edu_pptloader.py` | `.pptx`：按 `(top,left)` 排序形状；文本框/表格/图片；组合形状递归 | python-pptx 不支持旧版 `.ppt` |
| `edu_imgloader.py` | 单张图片整图 OCR | 直接把路径交给 RapidOCR |
| `samples/` | `ocr_01.pptx`、`ocr_02.docx`、`ocr_03.pdf`、`ocr_04.png` | 可单独运行各 loader 的 `__main__` 自测 |

### 6.6 `rag_qa/edu_text_spliter/` 文本分割器

| 文件 | 作用 |
|---|---|
| `__init__.py` | 把本目录加入 `sys.path`，导出 `ChineseRecursiveTextSplitter`、`MarkdownTextSplitter` 及模型分割器 |
| `edu_chinese_recursive_text_splitter.py` | 继承 `RecursiveCharacterTextSplitter`，自定义中文分隔符层级（段落→换行→句号→英文句号→分号→逗号），重写 `_split_text` 支持正则分隔符，并把分隔符保留在片段末尾 |
| `edu_model_text_spliter.py` | `AliTextSplitter`：BERT 语义分段模型按句子判断段落边界；`SimpleAliTextSplitter` 为其简化封装。当前主流程未使用 |

### 6.7 数据、训练与评估

| 文件 | 作用 |
|---|---|
| `rag_qa/classify_data/generate_classify_data.py` | 用规则模板 + 同义词替换 + 通义千问生成两类意图标注数据，去重、打乱后写 JSON；默认 6000 条，入口实际只生成 60 条用于演示 |
| `rag_qa/classify_data/model_generic_500.json` | 默认训练集（`Config.BERT_TRAIN_DATA`） |
| `rag_qa/rag_assessment/rag_as.py` | RAGAS 评估：faithfulness / answer_relevancy / context_precision / context_recall；默认 Ollama 本地模型，注释中保留了 DashScope 方案；结果写 CSV |
| `rag_qa/rag_assessment/rag_evaluate_data.json` | 评估样本（question/answer/context/ground_truth） |

### 6.8 `前置知识/`

LangChain 与 Milvus 的教学示例（模型调用、Prompt、文档加载、分割、向量库、Milvus CRUD/检索）。**不参与 EduRAG 运行**，可作为学习资料；其中部分示例依赖 Docker 中的 Milvus。

---

## 7. 核心机制说明

### 7.1 父子分块（Parent-Child Chunking）

- 父块大小 `parent_chunk_size=1200`：保留较大上下文，最终交给 LLM。
- 子块大小 `child_chunk_size=300`、`chunk_overlap=50`：只对子块做向量化，提高检索精度。
- 每个子块的 metadata 中保存 `parent_id` 与 `parent_content`；检索命中子块后按 `parent_id` 去重，取父块内容作为上下文。
- 好处：检索粒度小、生成上下文完整；代价：`parent_content` 在每个子块中重复存储，占用 Milvus 标量空间。

### 7.2 混合检索与重排

- 稠密向量：BGE-M3 dense（1024 维），索引 `IVF_FLAT`，度量 `IP`（内积），查询 `nprobe=10`。
- 稀疏向量：BGE-M3 sparse，索引 `SPARSE_INVERTED_INDEX`，度量 `IP`，建索引时 `drop_ratio_build=0.2`。
- 融合：`WeightedRanker(1.0, 0.7)`，语义权重 1.0、关键词权重 0.7。
- 精排：取回父块后用 `bge-reranker-large` 计算 (query, parent) 相关性，按分数降序取前 `candidate_m=2`。
- 学科过滤：Milvus 表达式 `source == '<source>'`，仅对「专业咨询」且用户选择学科时生效。

### 7.3 意图识别

- 模型：`bert-base-chinese` + 500 条标注数据微调（`rag_qa/models/bert-query-classifier-500`）。
- 标签：`{"通用知识": 0, "专业咨询": 1}`。
- 推理：分词（`max_length=128`, `padding=max_length`, `truncation=True`）→ 前向 → `argmax(logits)`。
- 通用知识不检索，直接让 LLM 回答；专业咨询才进入检索流程。
- 训练入口：`python rag_qa/core/query_classifier.py`（注意 `__main__` 中默认只跑预测，训练相关调用是注释状态）。

### 7.4 检索策略

| 策略 | 触发场景 | 实现 |
|---|---|---|
| 直接检索 | 事实明确、实体具体、无需推理 | 原 query 直接混合检索 |
| 回溯问题检索 | 场景复杂、口语化、有隐含前提 | LLM 把 query 简化成标准问法再检索 |
| 子查询检索 | 多实体对比、多维度并列 | LLM 拆成多行子查询，分别检索（`k // 2`）后按正文去重合并 |
| HyDE 假设问题检索 | 语义抽象、概念宽泛、开放性问题 | LLM 先生成假设答案，用假设答案去检索 |

策略选择完全依赖 `StrategySelector` 的提示词；如果 LLM 输出不在四个枚举值内，`_retrieve_entry()` 会走 `else` 分支，安全回退为「直接检索」。

### 7.5 FAQ + 缓存

- 启动 `Bm25SoftmaxSearch` 时：先读 Redis 中的「原始问题列表」和「分词问题列表」；任一缺失则从 MySQL 全量读取、分词，并写回 Redis。
- 查询时：先查 `edu_rag:answer:<query>` 缓存；未命中则 BM25 打分、softmax 归一化、取最高概率。
- 命中阈值 0.85：查最相似问题的答案（先 Redis 后 MySQL），并以**用户原始 query** 为 key 回填缓存。
- 缓存 TTL 由 `[redis] ttl` 控制，当前为 180 秒。

### 7.6 会话历史

- `main_plus.IntegratedQASystem` 在 MySQL 中维护 `conversations` 表（`session_id/question/answer/timestamp`）。
- 每次问答后插入记录，并删除该会话第 5 轮之前的数据；读取时按时间倒序取 5 条再反转。
- `rag_system.generate_answer_withhistory()` 会：
  - 用最近 5 轮历史拼成 `Q:...\nA:...` 放入提示词；
  - 把历史问题拼到当前 query 前面，用于检索增强；
  - 用 `llm.stream()` 逐 token 产出。

---

## 8. 鲁棒性与已知风险

以下按「环境 / 模型 / 数据 / 存储 / API / 安全 / 运维」分类。带 ⚠️ 的是建议优先处理的问题。

### 8.1 环境与路径

1. **工作目录依赖**
   - `rag_api.py` 的 `StaticFiles(directory="static")` 和 `FileResponse("static/index.html")` 依赖当前工作目录；请在项目根目录启动，否则静态页面 404。
   - `rag_qa/edu_text_spliter/edu_model_text_spliter.py` 中 `model_path = '../models/...'` 是相对 CWD 的路径，只适合在该文件目录下运行。
   - `rag_qa/rag_assessment/rag_as.py` 使用相对路径读取 `rag_evaluate_data.json`，需在 `rag_assessment` 目录下运行。
2. **`sys.path` 注入**
   - `rag_qa/edu_document_loaders/__init__.py` 与 `rag_qa/edu_text_spliter/__init__.py` 会把自身目录插入 `sys.path`，并让子模块使用 `from _edu_ocr import ...` 这类绝对导入。直接以脚本方式跨目录运行时可能找不到模块；建议统一通过包导入（`from rag_qa...`）。
3. **Windows 编码**
   - 控制台默认 GBK 时，`print` 中文可能乱码（日志文件是 UTF-8，不受影响）。可在启动前执行 `chcp 65001` 或设置 `$env:PYTHONIOENCODING="utf-8"`。
   - `mysql_client.py` 已强制 `charset="utf8mb4"`，CSV 导入仍建议统一 UTF-8。
4. **配置读取的隐式回退**
   - `base/config.py` 读取 MySQL 库名用的是 `[mysql] db`，而 `config.ini` 写的是 `database`，目前靠 `fallback='edurag'` 恰好一致。修改库名时务必同时改代码或配置键，否则会连到错误的库。
   - `VALID_SOURCES` 用 `eval()` 解析，配置必须是合法 Python 列表字面量；不要让外部输入进入该配置。
5. **端口与地址**
   - 本机 Docker 的 MySQL 映射为 **3307**；教学虚拟机是 3306。切换环境时别忘了改 `config.ini`。
   - 容器内 API 访问宿主机服务时，`127.0.0.1` 指向容器自身，需要改为宿主 IP 或 `host.docker.internal`（并确保 Docker 网络可达）。

### 8.2 模型与设备

1. **模型文件缺失或目录不完整**
   - 模型目录被 `.gitignore` 忽略，克隆后必须手动放置；`from_pretrained` 对不完整目录会直接抛异常。
   - `QueryClassifier._load_model()` 只判断路径是否存在：如果目录存在但文件缺失，会走「加载微调模型」分支并失败，不会回退到基座模型。
2. **CPU / GPU 切换 ⚠️**
   - `[device] device` 同时影响 Embedding、Reranker、BERT 分类器和 `fp16`。
   - Milvus 集合在创建时就把稠密字段类型固定为 `FLOAT_VECTOR`（CPU）或 `FLOAT16_VECTOR`（GPU）。**已有集合后切换设备，字段类型不会自动改变**；建议切换设备时重建 collection，或用同一设备重跑建库。
   - `BGEM3EmbeddingFunction(use_fp16=...)` 与 Milvus 字段类型必须一致，代码注释中已强调。
3. **显存 / 内存**
   - `VectorStore` 初始化会同时加载 BGE-M3（约数 GB）与 Reranker-large；`IntegratedQASystem` 先创建一个 `VectorStore`，又创建 `RAGSystem`（内部再创建一个 `VectorStore`），等于**重复加载两套模型**。内存紧张时优先消除这处重复。
   - `uvicorn --workers N` 会按进程复制模型，内存按 N 倍增长。
4. **OCR 设备选择**
   - `_edu_ocr.get_ocr(use_cuda=True)` 默认开启 CUDA；如果安装了 `rapidocr_paddle` 但机器无 GPU，运行时可能报错。只有 `ImportError` 会触发 ONNX 回退，其他异常不会。
   - 建议无 GPU 环境显式传 `use_cuda=False`，或卸载 paddle 版只保留 `rapidocr-onnxruntime`。

### 8.3 数据与文档处理

1. **支持范围**
   - 只支持 `.txt/.md/.pdf/.docx/.ppt/.pptx/.jpg/.jpeg/.png`；`.doc`、`.xls` 等会记录「不支持的文件类型」并被跳过。
   - `.ppt` 虽在映射表中，但 `python-pptx` 只支持 `.pptx`，旧格式会在加载时抛异常（被 `try/except` 捕获并记录，但文件内容会丢失）。
2. **`source` 元数据推导**
   - `load_documents_from_directory()` 用 `os.path.basename(root).replace("_data","")` 作为学科来源。只有文件**直接位于** `ai_data/` 下才得到 `ai`；如果放在 `ai_data/chapter1/`，`source` 会变成 `chapter1`，导致按学科过滤时检索不到。
3. **重复处理与计数**
   - `main.py` 数据处理分支中调用的是 `process_documents(data_path, ...)`（整个数据根目录），而不是循环里的 `data_abs_path`。结果会对同一批文档重复处理 `len(VALID_SOURCES)` 次，日志和计数被放大；`upsert` 能避免重复入库，但会浪费大量时间。建议改为 `process_documents(data_abs_path, ...)`（`rag_qa/main.py` 中已是正确写法）。
4. **MD5 主键冲突**
   - `vector_store.add_documents()` 用 `md5(page_content)` 作为主键。**不同学科、不同文件中内容完全相同的子块会互相覆盖**，后写入的 `source/timestamp` 会覆盖先写入的，可能导致学科过滤结果偏移。需要保留来源时，应把 `source`、`parent_id` 或文件路径纳入哈希。
5. **父块元数据体积**
   - 每个子块都携带完整 `parent_content`，Milvus 标量字段上限 65535；父块越大、子块越多，存储放大越明显。调整 `parent_chunk_size` 时需同步评估容量。
6. **文档加载的异常处理**
   - 单个文件加载失败只会记录错误并继续，整批任务不会中断，这是优点；但也意味着**坏文件会被静默跳过**，建库后应检查日志中的「加载失败」「不支持的类型」。
7. **OCR 的局限**
   - 扫描版 PDF / 图片识别质量依赖 RapidOCR；公式、手写体、复杂表格可能识别错误。
   - PDF 只对宽高都超过页面 60% 的图片做 OCR，小图（如小图标、局部截图）会被跳过；如果扫描页被拆成多张小图，可能漏识别。
   - Word 表格中的图片、页眉页脚内容不会被 `OCRDOCLoader` 处理；合并单元格在 `row.cells` 中会重复出现，导致文本重复。
8. **Markdown / TXT**
   - `.md` 使用 `UnstructuredMarkdownLoader`，依赖 `unstructured` 及其解析后端；环境缺依赖时会加载失败。
   - `.txt` 固定用 UTF-8 读取，GBK 文件会报错并被跳过。
9. **增量更新**
   - 当前建库是全量遍历 + `upsert`，没有删除机制。源文档删除或内容改名后，旧向量仍留在 Milvus 中；需要手动清理或重建 collection。

### 8.4 向量库（Milvus）

1. **数据库必须先存在**：`MilvusClient(db_name='edurag')` 在库不存在时初始化失败；首次部署务必先创建。
2. **集合 schema 固定**：字段、维度、索引在建集合时确定；修改 `embedding` 模型或向量维度后必须重建 collection，否则写入/检索会报错。
3. **`load_collection` 的必要性**：写入后代码会再次 `load_collection`；如果手动在 Milvus 中 release 了集合，查询会失败。
4. **过滤表达式拼接**
   - `filter_expr = f"source == '{source_filter}'"` 是字符串拼接。当前前端只传学科标签，但 HTTP/WS 请求可被伪造，`source_filter` 未做白名单校验，存在 **Milvus 表达式注入**风险。应在 API 层校验 `source_filter in Config.VALID_SOURCES`。
5. **检索参数边界**
   - `_retrieve_subqueries()` 使用 `k = Config.RETRIEVAL_K // 2`；若把 `retrieval_k` 配成 1，则 `k=0`，Milvus 会因 `limit=0` 报错。保持 `retrieval_k >= 2`。
   - `candidate_m` 大于 `retrieval_k` 时，最终上下文数量会被检索结果限制，不会报错但配置语义不一致。

### 8.5 MySQL / Redis

1. **默认参数在导入时建连 ⚠️**
   - `Bm25SoftmaxSearch.__init__` 的参数默认值是 `MysqlClient()` / `RedisClient()`，Python 在**函数定义时**求值，也就是模块被导入时就会连接数据库。服务没启动时，光 `import main` / `import main_plus` 就会抛异常。建议改成 `None`，在 `__init__` 内部创建。
   - 这也会让多个实例共享同一组连接对象，不利于隔离与关闭。
2. **`fetch_answer` 的空结果处理**
   - `result = cur.fetchone()` 后先执行 `len(result)` 再判断 `None`；无匹配行时会抛 `TypeError`，被外层捕获后记录成「获取答案失败」，并返回 `None`。功能上仍能兜底，但日志会误导排查。
3. **建表是破坏性操作**
   - `database_init()` → `creat_table()` 会 `DROP TABLE IF EXISTS jpkb`。重复执行会清空 FAQ 数据；部署脚本中不要无条件调用。
4. **SQL 拼接**
   - 表名使用 f-string 拼接（来自配置，可接受），值统一用 `%s` 参数化，方向正确；新增查询务必保持参数化。
5. **Redis TTL 与两套 key**
   - `set_data` 用 JSON（问题列表/分词列表），`set_answer` 用纯文本（答案），互不冲突。
   - 当前 `ttl=180`，意味着问题列表和答案缓存每 3 分钟失效一次；build 阶段每次失效都会从 MySQL 全量重载问题并重新分词。生产建议按数据更新频率调大，或把「问题列表缓存」与「答案缓存」拆成不同 TTL。
   - `clear_all_data()` 会 `flushdb()`，只应在专用 Redis 实例上使用。
6. **连接管理**
   - `MysqlClient.check_and_reconnect()` 通过 `ping()` 做被动重连；长时间运行、网络抖动后仍可能出现半开连接，需要在上层捕获异常并重试。
   - 进程退出时 `close()` 是必要的；`main_plus` 已在 `finally` 中关闭，`rag_api` 没有统一的关闭钩子。

### 8.6 API、并发与安全

1. **全局初始化副作用**
   - `rag_api.py` 在模块顶层创建 `qa_system = IntegratedQASystem()`：导入模块即加载模型、连接数据库。好处是启动即预热；坏处是任何依赖该模块的工具（测试、文档生成）都会要求全套服务在线。
2. **同步代码阻塞事件循环**
   - FastAPI 的 `async def` 处理函数中直接调用同步的 `bm25_search.search()` 和 `qa_system.query()`，会阻塞事件循环；WebSocket 里的 `asyncio.sleep(0.01)` 只能缓解 token 发送节奏，不能解决检索/LLM 调用的阻塞。生产建议改用 `run_in_threadpool` 或线程池/异步客户端。
3. **连接与并发**
   - 每个进程只有一个 `pymysql` 连接对象；`uvicorn --workers 2` 是两个进程，各自一份，问题不大；但在同一进程内若引入线程并发，多个请求共享一个连接会出错。
4. **WebSocket 细节**
   - 问候语分支直接 `break` 退出循环，连接会在 `finally` 中关闭；前端需要对下一次提问重新建连。
   - 客户端状态用 `websocket.client_state == websocket.client_state.CONNECTED` 判断，可工作但比较脆弱；更稳妥的是直接 `try/except` 发送。
   - 客户端断开时 `WebSocketDisconnect` 已被捕获；其他异常会尝试向已断开连接发送错误消息，仍可能抛二次异常，已被内层 `try` 吞掉。
5. **会话与鉴权**
   - 没有登录/鉴权；`/api/history/{session_id}` 只要知道 session_id 就能读取/清空历史（IDOR）。
   - `session_id` 由客户端传入，服务端未校验是否为自己创建的会话。
6. **CORS**
   - `allow_origins=["*"]` 与 `allow_credentials=True` 同时使用在浏览器规范下是无效组合；需要携带 Cookie 时应配置明确白名单。
7. **注入与输入校验**
   - `source_filter` 未做白名单校验，会进入 Milvus 表达式；`query` 未限制长度，超长输入会放大 embedding / LLM 成本。
   - 日常问候正则用 `re.match`，只匹配开头，可接受；注释中「去除 # 前缀」与实现不符。
8. **密钥与默认口令**
   - `config.ini` 当前未被 `.gitignore` 排除，且包含 `password = 123456`；`git add .` 会把明文口令提交。请确认不要提交真实密钥，必要时把 `config.ini` 加入 `.gitignore` 并提供 `config.example.ini`。
   - API Key 建议只用环境变量 / 容器 Secret；不要把 `sk-...` 写进代码或提交历史。
   - 默认口令（MySQL/Redis 123456）、无认证的 Attu/MinIO 都不适合暴露到公网。

### 8.7 运行与运维

1. **日志**
   - 日志按天写 `logs/Edu RAG_YYYYMMDD.log`，但没有轮转/清理策略；长期运行需自行归档或加 `RotatingFileHandler`。
   - `logger` 是全局单例，重复导入不会重复挂 Handler；但修改日志级别需要改 `config.ini` 后重启进程。
2. **模型与服务启动顺序**
   - 正确顺序：Docker 基础设施 → 创建 Milvus 数据库 → 初始化 MySQL → 建向量库 → 启动 API。
   - 顺序错误时，异常往往出现在 `VectorStore()` 初始化或 `Bm25SoftmaxSearch()` 导入阶段，先看 `docker compose ps` 与日志。
3. **数据备份**
   - `docker-data/` 是 MySQL/Redis/Milvus/MinIO/etcd 的持久化目录，删除即丢数据；备份时应整体停服或使用数据库自带工具导出。
   - `install-edurag.bat` 在 `docker-data` 非空时会拒绝重建 compose，避免与其他部署共享数据目录；这是保护措施，不要绕过。
4. **镜像与网络**
   - `docker-compose.yml` 的 MinIO 使用 `milvusdb/minio`（因官方镜像许可变更），需要能访问对应仓库。
   - 国内网络可给 Docker 配置镜像加速；镜像拉取失败时 `docker compose up -d` 会中断，重试前先看具体服务日志。
5. **Dockerfile**
   - `.dockerignore` 排除了 `rag_qa/models`、`rag_qa/data`、`docker-data`、`logs`，因此镜像内没有模型和知识文档，必须挂载。
   - `Dockerfile.prod` 不设置 `CMD`，忘记在 `docker run` 末尾给命令会直接启动失败。
   - 容器内 `WORKDIR /app`；挂载 `config.ini` 时注意宿主路径与容器路径一致。
6. **RAGAS 评估**
   - 默认依赖本机 Ollama（`qwen2.5:7b`、`mxbai-embed-large`）且 `max_workers=1`，耗时较长；没有 Ollama 时请切到注释中的 DashScope 方案。
   - 评估结果、上下文质量、LLM 评判都会影响指标，指标波动不等于线上效果波动，建议固定数据集与模型版本后再比较。

### 8.8 已知代码层面的小问题（不影响跑通，但建议修）

| 位置 | 现象 | 建议 |
|---|---|---|
| `main.py` 数据处理循环 | 每轮都处理整个 `data_path`，重复 N 次 | 改为 `process_documents(data_abs_path, ...)` |
| `base/config.py` | 读 `mysql.db`，配置键是 `database` | 统一键名，去掉对 fallback 的依赖 |
| `main_plus.py` | `VectorStore` 与 `RAGSystem` 各建一个向量库，模型重复加载 | 让 `RAGSystem` 接收外部 `vector_store`，或复用同一实例 |
| `rag_system.py` 历史处理 | `history` 被赋值为生成器并在拼接时耗尽，后续 `h['question']` 迭代为空，检索 query 丢失历史问题 | 先保留原始 list，再单独生成 `history_context` |
| `mysql_client.fetch_answer` | 先 `len(result)` 后判空 | 先判 `None` 再取长度 |
| `edu_pdfloader` / `edu_docloader` | 字符串 `+=` 累积、进度条不 `close()` | 用列表 `append` + `"".join`，并在 `finally` 关闭进度条 |
| `Bm25SoftmaxSearch.__init__` | 默认参数在导入时建立 DB/Redis 连接 | 改为 `None` 后在方法内延迟创建 |
| `rag_api.py` | `allow_origins=["*"]` + `allow_credentials=True` | 配置明确来源白名单 |
| `rag_api.py` | `source_filter` 未校验 | 进入检索前校验 `in Config.VALID_SOURCES` |
| `static/index.html` + WS | 问候语后连接被关闭 | 前端在 `end` 后按需重建连接，或后端不 `break` |

---

## 9. 首次部署检查清单

- [ ] Python 3.10 虚拟环境已激活，`pip install -r requirements.txt` 成功
- [ ] `rag_qa/models/` 下 4 个必需模型目录完整
- [ ] `rag_qa/data/<学科>_data/` 中有文档，学科名在 `valid_sources` 中
- [ ] `$env:AliApiKey` 已设置且可调用通义千问
- [ ] `docker compose ps` 显示 MySQL/Redis/Milvus 均为 healthy（或至少 running）
- [ ] Milvus 中已创建数据库 `edurag`
- [ ] MySQL 已执行 `database_init()`，`jpkb` 表有数据
- [ ] 已执行 `python main.py --data`，Milvus 集合 `edurag_doc` 有向量
- [ ] `python main.py` 可以回答 FAQ 与专业问题
- [ ] `uvicorn rag_api:app` 启动成功，`/health` 返回 healthy，页面可访问
- [ ] 日志无「加载失败」「连接失败」「模型缺失」等关键错误

### 9.1 快速自检命令

```powershell
# 配置与路径
python base/config.py

# MySQL 连通性与 FAQ 数据
python -c "from mysql_qa.utils.mysql_client import MysqlClient; c=MysqlClient(); print(len(c.fetch_all_questions())); c.close()"

# Redis 连通性
python -c "from mysql_qa.utils.redis_client import RedisClient; print(RedisClient().client.ping())"

# Milvus 连通性与集合
python -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://127.0.0.1:19530', db_name='edurag'); print(c.list_collections()); c.close()"

# 文档加载器（样例路径是相对 CWD 的，必须进入该目录运行）
cd rag_qa/edu_document_loaders
python edu_docloader.py
cd ../..

# 建库（小范围验证）
python main.py --data --dir rag_qa/data/ai_data

# API 健康检查
curl http://127.0.0.1:8000/health
```

---

## 10. 常见问题

**Q1：启动报 `Connection refused` / `Can't connect to MySQL`**
先确认 Docker Desktop 已启动且 `docker compose ps` 正常；再核对 `config.ini` 端口（本机 Docker 是 3307，虚拟机是 3306）。Milvus 首次启动较慢，可 `docker compose logs -f milvus` 等待健康。

**Q2：报 `database not found: edurag`（Milvus）**
Milvus 不会自动创建数据库，需按 4.6 节创建 `edurag`。

**Q3：报模型路径不存在 / `from_pretrained` 失败**
检查 `rag_qa/models/` 下目录是否完整；模型文件被 `.gitignore` 排除，需单独下载。路径由 `base.path_tools.get_abs_path()` 基于项目根目录解析，与启动目录无关（但 `AliTextSplitter` 例外）。

**Q4：FAQ 总是走 RAG，标准答案不生效**
- `jpkb` 表是否为空（`database_init()` 是否执行）；
- 问题相似度是否真的 ≥ 0.85（看日志中的 `最高分`）；
- Redis 中旧缓存是否干扰（可调小 TTL 或手动删 key）。

**Q5：文档建库成功但检索不到某学科**
检查该文件是否直接位于 `<学科>_data/` 下；放在子目录中会导致 `source` 变成子目录名，按学科过滤就查不到。另确认 `valid_sources` 与目录名一致。

**Q6：GPU 环境启动失败 / 向量维度不匹配**
确认 `[device] device = cuda`、torch 为 CUDA 版本、显存足够；如果之前用 CPU 建过集合，稠密字段类型是 `FLOAT_VECTOR`，切到 GPU 的 `FLOAT16_VECTOR` 前需要重建 collection。

**Q7：`python main.py` 卡在「正在生成答案」**
首次调用大模型需要网络与 API Key；检查 `AliApiKey` 是否在当前终端设置（环境变量不会跨终端继承）。同时观察日志中是否出现 LLM 调用异常。

**Q8：Web 页面能打开但提问无响应**
打开浏览器开发者工具看 WebSocket 是否连上 `/api/stream`；后端日志是否出现检索/模型异常。`/api/query` 在 FAQ 未命中时只返回提示文案，真正生成走 WebSocket。

**Q9：中文日志/控制台乱码**
日志文件为 UTF-8；控制台可执行 `chcp 65001` 或设置 `$env:PYTHONIOENCODING="utf-8"`。

**Q10：想重新开始（清空知识库）**
- MySQL FAQ：重新执行 `database_init()`（会重建 `jpkb`）；
- Milvus：在 Attu 删除 collection `edurag_doc`，或重建数据库后重新建库；
- Redis：`RedisClient().clear_all_data()`（仅限专用实例）；
- 模型与文档：无需重下，重新执行 `python main.py --data` 即可。

---

## 11. 开发与扩展建议

- **新增文档格式**：在 `process_documents.document_loaders` 中注册扩展名 → 实现一个 `BaseLoader` 子类，`lazy_load()` 返回 `Document` → 在 `__init__.py` 导出。
- **新增检索策略**：在 `prompts.py` 加提示词 → 在 `rag_system.py` 加 `_retrieve_xxx()` → 在 `_retrieve_entry()` 加分支 → 在 `strategy_selector.py` 的策略说明和输出枚举中同步。
- **更换大模型**：改 `config.ini` 的 `dashscope_model` / `dashscope_base_url` / Key；若换非 OpenAI 兼容协议，需要在 `llm_provider.py` 换 `init_chat_model` 的 provider。
- **更换向量模型**：改 `embedding_path` 后必须重建 collection（维度/字段类型可能变化），并同步调整 `dense_dim` 与索引参数。
- **加鉴权**：在 FastAPI 层加中间件/依赖注入，校验 session 归属；`source_filter` 做白名单；生产环境收窄 CORS、关闭无认证的 Attu/MinIO 公网访问。
- **加监控**：对 `/health` 增加 MySQL/Redis/Milvus 深度探活；为检索耗时、LLM 首 token 延迟、缓存命中率加指标。

---

## 12. 相关文档

- `项目程序启动说明.md`：分模块的启动顺序（EduFQA / EduRAG / EduRagQA / 历史会话版 / RagAPI / Docker）
- `本机Docker使用说明.md`：Windows + Docker Desktop 的详细操作
- `项目支持虚拟机信息.md`：教学虚拟机 IP、账号、端口与基础设施 compose
- `前置知识/`：LangChain 与 Milvus 的学习示例

---

> 最后提醒：本项目的默认口令、开放 CORS、无鉴权 API、明文配置等设置仅适合教学与内网演示。对外部署前，请完成第 8.6 节的安全加固，并确保 `config.ini`、API Key、`docker-data/` 不进入版本库。
