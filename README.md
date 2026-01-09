# Hasu RAG: 链接！ 喜欢！ 爱生活！ 剧情智能分层问答系统

这是一个基于 **LangChain v1.x**、**Qdrant** 和 **Xinference** 构建的高级 RAG（检索增强生成）系统。
本项目专门针对长篇剧情/剧本数据进行了深度优化，实现了**混合检索 (Hybrid Search)**、**带约束的时间序聚类 (Time-Constrained Clustering)** 和 **双路上下文融合 (Dual-Path Context Fusion)**，能够同时应对“微观细节追问”、“宏观剧情概括”以及“深度剧情分析”三类用户查询。

---

## 系统架构与实现思路

本系统并不是简单的 RAG 套壳，而是针对剧情类数据的特性设计了独特的分层路由与索引架构。

### 1. 双层知识库设计 (Dual-Layer Knowledge Base)

为了兼顾细节与宏观，我们构建了两个独立的 Qdrant 集合：

| 集合名称 | 数据层级 | 存储内容 | 检索模式 | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **`story_knowledge_base`** | **微观层 (Raw Fragments)** | 原始剧情切片，包含完整的台词、场景、时间元数据。 | **Hybrid (Dense + Sparse)** | “梢在哪一话哭了？“<br>“花帆最喜欢的食物是什么？” |
| **`story_summary_store`** | **宏观层 (Cluster Summaries)** | 经过聚类生成的剧情摘要，涵盖不同时间跨度的事件概括。 | **Dense Only** | “第10话讲了什么故事？”<br>“吟子对花帆的感情变化是怎样的？” |

### 2. 核心工作流

#### A. 数据入库 (Ingestion)
* **脚本**: `app/ingest.py` (全量) / `app/ingest_append.py` (增量)
* **逻辑**:
    1.  读取剧情数据，支持增量追加（基于 UUID5 去重）。
    2.  利用 **Xinference (bge-m3)** 生成稠密向量 (Dense Embeddings)。
    3.  利用 **FastEmbed (BM25)** 生成稀疏向量 (Sparse Embeddings) 用于关键词精确匹配。
    4.  存入 `story_knowledge_base`。

#### B. 分层索引构建 (Hierarchy Building)
* **脚本**: `app/build_hierarchy.py`
* **算法思路 (Time-Constrained Agglomerative Clustering)**:
    1.  **加载与排序**: 加载所有碎片并严格按剧情时间线排序。
    2.  **连接性约束聚类**: 使用 `AgglomerativeClustering` 配合 `kneighbors_graph`，强制只聚合在时间上相邻的剧情块，**杜绝跨时间线的错误归纳**。
    3.  **LLM 摘要生成与智能增量更新**: 调用 LLM 生成剧情摘要。脚本具备**智能变更检测**功能，通过对比新旧聚类的“成员指纹”，自动识别因新数据加入导致的聚类边界变化。仅对真正内容变动的簇重新生成摘要，完美解决增量数据带来的“蝴蝶效应”，既保证准确性又最大化节省 Token。
    4.  **存入**: 将摘要存入 `story_summary_store`，并标记层级。

#### C. 智能问答管道 (Query Pipeline)
* **脚本**: `app/query.py`
* **流程**:
    1.  **深度意图识别 (Intent Classification)**: 基于 LLM 的三元分类：
        * `fact`: 事实追问（时间、地点、物品）。
        * `overview`: 纯宏观概括。
        * `analysis`: **深度分析**（如角色关系、情感动机、性格评价）。
    2.  **动态路由与融合 (Routing & Fusion)**:
        * **Fact 模式**: 单路检索原始库。利用 **Query Rewrite** 自动扩展同义词（如“哭” -> “泣く, 涙, 号泣”），并大幅提升 Top-K 窗口以优化计数类问题召回率。
        * **Overview 模式**: 单路检索摘要库。
        * **Analysis 模式 (Dual-Path)**: **同时**检索摘要库（获取背景与氛围）和原始库（获取具体台词证据），通过 `Context Fusion` 模板将宏观与微观信息拼接，生成有理有据的深度回答。

---

## 项目结构说明

```text
.
├── app/
│   ├── api_server.py          # OpenAI 兼容 API 服务器 (FastAPI)
│   ├── auth.py                # 鉴权与并发控制逻辑
│   ├── api_keys.yaml          # API Key 配置文件
│   ├── build_hierarchy.py     # 分层索引构建主脚本 (支持断点续传/覆盖)
│   ├── ingest.py              # 全量数据入库脚本 (会清空旧数据)
│   ├── ingest_append.py       # 增量数据入库脚本 (幂等追加)
│   ├── query.py               # [入口] 核心智能问答处理逻辑
│   ├── rag_wrapper.py         # API 服务与核心逻辑的适配层
│   ├── run_api_server.sh      # API 服务器启动脚本
│   ├── reranker.py            # 自定义 Rerank 组件封装
│   ├── visualize_clusters.py  # 生成静态时间线聚类图 (PNG)
│   └── visualize_interactive.py # 生成交互式剧情地图 (HTML)
├── data/
│   ├── optimized_final.json   # 已清洗的主数据集（默认全量入库使用）
│   ├── stories.json           # 旧版原始数据源（可选）
│   └── new_stories.json       # 增量数据源示例（可自定义）
├── qdrant_storage/            # Qdrant 本地数据持久化目录
└── requirements.txt           # Python 依赖
```

---

## 🚀 快速开始

### 1. 环境准备
使用 Python 3.11，并安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
# Qdrant & Xinference
export QDRANT_URL="http://localhost:6333"
export XINFERENCE_SERVER_URL="http://your-server:9997"
# LLM 配置 (OpenAI 协议兼容)
export LLM_BASE_URL="http://your-server:9001/v1"
export LLM_API_KEY=""
export LLM_MODEL_NAME=""
```

### 3. 数据文件
默认全量入库读取 `data/optimized_final.json`。如需自定义路径，可设置：

```bash
export DATA_FILE="/path/to/your_full_data.json"      # 全量入库使用
export APPEND_DATA_FILE="/path/to/your_new_data.json" # 增量入库使用
```

### 4. 数据流程
```bash
# 1. 初始化原始知识库
python app/ingest.py

# 2. 构建分层摘要索引
python app/build_hierarchy.py
```

### 5. 启动服务 (二选一)

*   **交互式命令行模式**:
    ```bash
    python app/query.py
    ```
*   **OpenAI 兼容 API 模式**:
    ```bash
    ./app/run_api_server.sh
    ```

---

## 📡 OpenAI 兼容 API

本系统提供标准的 OpenAI 兼容接口，方便接入 Koishi, ChatLuna 等第三方框架。

### 1. 配置 API Keys
编辑 `app/api_keys.yaml`：
```yaml
api_keys:
  - key: "sk-your-api-key"
    name: "Admin"
    rate_limit_minutes: 3
```

### 2. API 端点
*   **获取模型**: `GET /v1/models`
*   **聊天完成**: `POST /v1/chat/completions` (支持 `stream: true/false`)

### 3. 核心特性
*   **多线程并发控制**: 采用全局互斥锁，确保 RAG 检索时系统负载稳定，并发请求将立即收到“系统繁忙”提醒。
*   **消息清洗**: 自动识别并剔除机器人框架常见的 `<at>` 标签和昵称前缀，确保 RAG 检索的纯净度。
*   **静默请求过滤**: 忽略 `system` 角色消息和 `temperature` 等参数，始终以最优生产配置运行。

---

## 🐳 Docker 部署

```bash
# 端口映射: 8000 为 API 服务端口
docker run -d \
  --name hasu-rag-api \
  -p 8000:8000 \
  -e QDRANT_URL="http://ip:6333" \
  -e XINFERENCE_SERVER_URL="http://ip:9997" \
  ... 镜像名
```

---

## 关键技术栈

*   **LangChain**: 编排 RAG 流程与 Prompt 管理。
*   **Qdrant**: 混合检索 (Dense + Sparse) 核心存储。
*   **Xinference**: 本地化高性能模型推理 (兼容 OpenAI 协议)。
*   **FastAPI**: 高性能异步 API 框架。
