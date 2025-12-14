# Hasu RAG: 链接！ 喜欢！ 爱生活！ 剧情智能分层问答系统

这是一个基于 **LangChain**、**Qdrant** 和 **Xinference** 构建的高级 RAG（检索增强生成）系统。
本项目专门针对长篇剧情/剧本数据进行了深度优化，实现了**混合检索 (Hybrid Search)**、**时间序聚类 (Time-Aware Clustering)** 和 **分层摘要索引 (RAPTOR-like Hierarchy)**，能够同时应对“微观细节追问”和“宏观剧情概括”两类用户查询。

---

## 系统架构与实现思路

本系统并不是简单的 RAG 套壳，而是针对剧情类数据的特性设计了独特的分层路由与索引架构。

### 1. 双层知识库设计 (Dual-Layer Knowledge Base)

为了兼顾细节与宏观，我们构建了两个独立的 Qdrant 集合：

| 集合名称 | 数据层级 | 存储内容 | 检索模式 | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **`story_knowledge_base`** | **微观层 (Raw Fragments)** | 原始剧情切片，包含完整的台词、场景、时间元数据。 | **Hybrid (Dense + Sparse)** | “梢在哪一话哭了？“<br>“花帆最喜欢的食物是什么？” |
| **`story_summary_store`** | **宏观层 (Cluster Summaries)** | 经过聚类生成的剧情摘要，涵盖不同时间跨度的事件概括。 | **Dense Only** | “第10话讲了什么故事？”<br>“103期故事中有什么重大转折？” |

### 2. 核心工作流

#### A. 数据入库 (Ingestion)
*   **脚本**: `app/ingest.py`
*   **逻辑**:
    1.  读取 `data/stories.json` 中的剧情数据。
    2.  利用 **Xinference (bge-m3)** 生成稠密向量 (Dense Embeddings) 用于语义匹配。
    3.  利用 **Qdrant (BM25)** 生成稀疏向量 (Sparse Embeddings) 用于关键词精确匹配（如人名、特定术语）。
    4.  存入 `story_knowledge_base`。

#### B. 分层索引构建 (Hierarchy Building)
*   **脚本**: `app/build_hierarchy.py`
*   **算法思路 (RAPTOR over Time)**:
    1.  **加载数据**: 从底层知识库加载所有剧情碎片。
    2.  **时间序约束聚类**: 使用 `AgglomerativeClustering`，但引入时间/话号约束，强制只聚合相邻或相近的剧情块，防止跨时间线的错误归纳。
    3.  **LLM 摘要生成**: 对聚类后的文本块调用 LLM 生成日文剧情摘要。
    4.  **递归构建**: 将生成的摘要视为新的节点，重复聚类过程，形成树状结构。
    5.  **存入**: 将所有摘要存入 `story_summary_store`。

#### C. 智能问答管道 (Query Pipeline)
*   **脚本**: `app/query_smart.py`
*   **流程**:
    1.  **意图识别 (Intent Classification)**: 分析用户问题是 `overview` (概括) 还是 `specific` (细节)。
    2.  **智能路由 (Routing)**:
        *   **Overview**: 直接检索 `story_summary_store`，获取宏观上下文，生成回答。
        *   **Specific**:
            *   **Query Rewrite**: 将中文问题翻译为日文，并自动扩展同义词（如“哭” -> “泣く, 涙, 号泣”），修正角色称呼。
            *   **Hybrid Retrieval**: 并行执行 Dense 和 Sparse 检索，在 Qdrant 内部进行互惠排序融合 (RRF)。
            *   **Rerank**: 使用 Xinference Rerank 模型对前 150 条结果进行精排，选出 Top N。
            *   **Generation**: 基于精选片段生成中文回答。

---

## 项目结构说明

```text
.
├── app/
│   ├── build_hierarchy.py     # 分层索引构建主脚本 (RAPTOR)
│   ├── diagnostic.py          # Qdrant 连接诊断工具
│   ├── evaluate.py            # 检索效果评估工具 (Review mode)
│   ├── ingest.py              # 全量数据入库脚本 (会清空旧数据)
│   ├── ingest_append.py       # 增量数据入库脚本
│   ├── query_smart.py         # 核心问答启动脚本
│   ├── reranker.py            # 自定义 Rerank 组件封装
│   ├── visualize_clusters.py  # 聚类效果可视化工具
│   └── visualize_interactive.py # 生成交互式剧情地图
├── data/
│   ├── stories.json           # 原始数据源 (需自行准备)
│   └── new_stories.json       # 增量数据源 (需自行准备)
├── qdrant_storage/            # Qdrant 本地数据持久化目录
├── docker-compose.yml         # Qdrant 服务编排 (可选)
└── requirements.txt           # Python 依赖
```

### 可视化分析

项目包含用于验证聚类效果的可视化脚本。下图展示了剧情在时间线上的聚类分布 (`app/build_hierarchy.py` 的产出效验)：

![Cluster Timeline Analysis](app/cluster_timeline.png)

---

## 快速开始

### 1. 环境准备
如果你使用 Docker 或 Conda 创建环境，请明确指定 Python 3.11，并配置好以下服务：

*   **Qdrant**: 向量数据库 (默认 `localhost:6333`)
*   **Xinference**: 模型推理服务 (提供 embedding, rerank, llm)

不支持: Python 3.8 及以下，Python 3.12+ (部分库可能尚未提供预编译的 Wheel 包)

安装 Python 依赖：

```bash
pip install -r app/requirements.txt
```

### 2. 配置环境变量

你可以通过环境变量覆盖默认配置（通常在 `.env` 或直接 export）：

```bash
export QDRANT_URL="http://localhost:6333"
export XINFERENCE_SERVER_URL="http://your-xinference-server:9997"
# LLM 配置 (适配 OpenAI 接口)
export LLM_BASE_URL="..."
export LLM_API_KEY="..."
```

### 3. 数据流程

**Step 1. 初始化原始知识库**

```bash
python app/ingest.py
```

**Step 2. 构建分层摘要索引** (此步耗时较长，视数据量而定)

```bash
python app/build_hierarchy.py
```

**Step 3. 启动问答系统**

```bash
python app/query_smart.py
```

---

## 关键技术栈

*   **LangChain**: 编排 RAG 流程与 Prompt 管理。
*   **Qdrant**: 混合检索 (Dense + Sparse) 核心存储。
*   **Xinference**: 本地化高性能模型推理 (兼容 OpenAI 协议)。
*   **FastEmbed**: 提供轻量级的 Sparse Vector 生成 (BM25)。