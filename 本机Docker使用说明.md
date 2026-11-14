# Windows 本机 Docker 使用说明

本项目使用电脑上的 Docker Desktop 运行 MySQL、Redis、Milvus 及其依赖服务；Python 程序在 Windows 本机运行。以下命令均在项目根目录的 PowerShell 中执行。

## 启动与停止

先启动 Docker Desktop，等待 Docker 引擎就绪，然后执行：

```powershell
docker compose up -d
docker compose ps
```

首次启动需要下载镜像，服务可能需要几分钟才能就绪。查看日志：

MinIO 使用 Milvus 项目维护的 `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` 镜像；Docker Desktop 需要能够访问对应镜像仓库。

```powershell
docker compose logs --tail 100 milvus
docker compose logs --tail 100 mysql
```

停止容器（保留数据）：

```powershell
docker compose stop
```

再次启动可执行 `docker compose up -d`。服务数据保存在项目内的 `docker-data` 目录，请勿随意删除。

MySQL 使用 `127.0.0.1:3307`（容器内仍为 3306），Redis 为 `127.0.0.1:6379`，Milvus 为 `127.0.0.1:19530`，对应配置已写在 `config.ini`。Milvus 管理界面 Attu 地址为 http://127.0.0.1:3000。

## 首次使用

确保服务已就绪，并在运行程序的 Python 环境安装 `requirements.txt` 中的依赖。项目需使用 `rag_qa/models` 中的模型文件以及 `rag_qa/data` 中的知识文档。默认使用 CPU，模型加载和处理文档可能较慢，且需要足够内存和磁盘空间。

MySQL 首次启动会自动创建 `edurag` 数据库。**仅在确认该数据库中的 `jpkb` 表没有需要保留的数据时**，可运行下列命令导入项目 FAQ 数据：该初始化函数会先删除并重建 `jpkb` 表，重复执行会清除表中已有数据。

```powershell
python -c "from mysql_qa.utils.mysql_client import database_init; database_init()"
```

Milvus 首次使用时需要创建 `edurag` 数据库。可通过 Attu 创建，或在已安装项目依赖的 Python 环境中运行：

```powershell
python -c "from pymilvus import MilvusClient; c = MilvusClient(uri='http://127.0.0.1:19530'); c.create_database('edurag') if 'edurag' not in c.list_databases() else None; c.close()"
```

接着首次导入知识文档：

```powershell
python -c "from main import main; main(query_mode=False)"
```

## 阿里云密钥与运行程序

需要可用的阿里云百炼 API Key。将其设置为当前 PowerShell 会话的环境变量（不要将真实密钥写进项目文件）：

```powershell
$env:AliApiKey = '替换为你的API Key'
```

环境变量只在当前 PowerShell 会话生效，重新打开终端后需重新设置。完成以上依赖和数据初始化后，可运行：

```powershell
python -m uvicorn rag_api:app --host 127.0.0.1 --port 8000
```

浏览器访问 http://127.0.0.1:8000。API 启动时需要连接上述服务并加载本地模型；如果连接失败，先查看 `docker compose ps` 和相应服务日志。
