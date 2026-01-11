# Hasu RAG: 链接！ 喜欢！ 爱生活！ 剧情智能分层问答系统

基于 **LangChain v1.x**、**Qdrant** 和 **Xinference** 的长篇剧情 RAG。特点：混合检索 (Dense + BM25)、时间序约束聚类、双路上下文融合，既能回答细节追问也能做宏观/分析类总结。

---

## 最新变更与支持范围
- 仅支持 **`data/timeline_flow_optimized.json`** 新格式；旧版 stories/optimized_final/new_stories 路径已废弃。
- 入库生成 ID：`scene/id` -> UUID5（稳定可覆盖），按文件顺序写入 `metadata.order` 保障时间线。
- Metadata 完整保留 `ctx/stats/timeline/act/emo/state_*`、台词 `dialogues` 与来源 `src`，所有聚类与可视化均按 `metadata.order` 排序。
- 入库时会将 `ctx` 中的角色/表情等关键词追加到正文末尾 `[meta]`，用于 BM25 稀疏检索；摘要生成会自动剥离该段。

---

## 数据格式（timeline_flow_optimized.json）
- **顶层字段**：`id`、`scene`、`src`、`text`（已扁平化）、`script`（对话列表）、`ctx`、`stats`、`timeline`、`merged_from`（可选）。
- **ctx**：`chars`、`voices`、`loc`、`time`、`bgm`、`type`、`act`、`emo`、`state_act`、`state_emo` 等。
- **stats**：`tok`（token 数）、`dlg`（对话轮次）。
- **timeline**：按角色记录动作/情感演变。
- 入库内容来源：优先 `text`，缺失则拼接 `script` 的 `c/t` 字段。
- `ctx.chars/loc/time/emo/state_emo` 会被提炼为关键词，追加到正文尾部 `[meta]` 用于 BM25。

---

## 系统架构
- **双层知识库**
  - `story_knowledge_base`：原始剧情切片，Hybrid 检索 (Dense + Sparse)。
  - `story_summary_store`：时间序聚类摘要，Dense 检索。
- **查询路由**
  - 意图分类：`fact` / `overview` / `analysis`。
  - `fact` → 细节库；`overview` → 摘要库；`analysis` → 摘要 + 细节双路融合。
- **检索细节**
  - 语义通道与关键词通道分别重写，LLM 提供动态 alpha 以调节两路候选规模。
- **时间序聚类**：Agglomerative + 邻接约束，仅合并相邻片段；LLM 生成摘要并按指纹增量更新。

---

## 工作流
1) **安装依赖**  
   ```bash
   pip install -r requirements.txt  # Python 3.11
   ```
2) **配置环境变量（示例）**
   - 推荐：复制 `.env.example` 为 `.env`，脚本会自动加载该文件；`docker-compose.yml` 也会挂载并注入环境变量。
   ```bash
   export QDRANT_URL="http://localhost:6333"
   export XINFERENCE_SERVER_URL="http://your-server:9997"
   export LLM_BASE_URL=""
   export LLM_API_KEY=""
   export LLM_MODEL_NAME=""
   # 可选：自定义数据路径
   export DATA_FILE="/path/to/timeline_flow_optimized.json"
   export APPEND_DATA_FILE="/path/to/timeline_flow_optimized.json"
   ```
3) **全量入库（会重建集合）**  
   ```bash
   python app/ingest.py
   ```
4) **增量/幂等追加**  
   ```bash
   python app/ingest_append.py
   ```
   - 仅支持新格式，需设置 `APPEND_DATA_FILE` 指向可用文件。
5) **构建/刷新摘要索引**  
   ```bash
   python app/build_hierarchy.py
   ```
6) **检索体验**
   - 交互 CLI：`python app/query.py`
   - API 服务：`./app/run_api_server.sh`（OpenAI 兼容）

可选：可视化时间线  
```bash
python app/visualize_clusters.py          # 静态 PNG
python app/visualize_interactive.py       # 交互式 HTML
```

---

## 目录速览
```text
.
├── app/
│   ├── ingest.py / ingest_append.py     # 全量/增量入库（仅新格式）
│   ├── build_hierarchy.py               # 时间序聚类 + 摘要
│   ├── query.py                         # 交互式问答入口
│   ├── api_server.py / run_api_server.sh# OpenAI 兼容 API
│   ├── reranker.py                      # 精排组件
│   ├── visualize_clusters.py / visualize_interactive.py
│   └── ...                              # 其他适配层/鉴权等
├── data/
│   └── timeline_flow_optimized.json     # 默认数据集
├── qdrant_storage/                      # Qdrant 本地持久化
└── requirements.txt
```

---

## Docker 部署（参考）
```bash
docker run -d \
  --name hasu-rag-api \
  -p 8000:8000 \
  -e QDRANT_URL="http://ip:6333" \
  -e XINFERENCE_SERVER_URL="http://ip:9997" \
  your-image-name
```

---

## OneBot MCP 语音发送（可选）
用于在 **用户明确要求“发送/播放/发到/发给”** 时，通过 MCP 工具把语音发送到 OneBot v11。
默认只匹配 `vo_adv_...`（可带 `.mp3`）的语音文件名。

### 容器挂载示例
```bash
-v /Users/nickel/Developer/hasu-rag/voices:/data/voices:ro
```

### 环境变量示范（路径映射 + 仅 mp3 + 自动发送）
```bash
# MCP 开关
export ONEBOT_MCP_ENABLED=1
export ONEBOT_MCP_AUTOSTART=1

# OneBot 连接
export ONEBOT_BASE_URL="http://127.0.0.1:5700"
export ONEBOT_ACCESS_TOKEN=""

# 语音目录（RAG 侧）
export ONEBOT_VOICE_DIR="/Users/nickel/Developer/hasu-rag/voices"
export ONEBOT_VOICE_EXTS=".mp3"

# 路径映射：RAG 路径 -> OneBot 容器路径
export ONEBOT_PATH_MAP_FROM="/Users/nickel/Developer/hasu-rag/voices"
export ONEBOT_PATH_MAP_TO="/data/voices"

# 目标默认值（群）
export ONEBOT_DEFAULT_TARGET=""

# 发送模式（auto/voice/file）
export ONEBOT_SEND_MODE="auto"

# 多文件发送与间隔
export ONEBOT_MULTI_SEND=1
export ONEBOT_SEND_INTERVAL_SECONDS=1.5
```

### 行为说明
- 仅当用户问题中出现“发送/播放/发到/发给”等词触发发送。
- 语音文件名优先从 **回答** 中提取，其次是 **用户问题**，最后从 **检索文档 metadata** 中兜底。
- 默认仅发送 **第一条** 语音；设置 `ONEBOT_MULTI_SEND=1` 则按出现顺序逐条发送，并按 `ONEBOT_SEND_INTERVAL_SECONDS` 间隔。

### 进阶配置
```bash
# MCP 服务器路径（默认 app/mcp_onebot_server.py）
export ONEBOT_MCP_SERVER_PATH="/Users/nickel/Developer/hasu-rag/app/mcp_onebot_server.py"

# 语音查找策略
export ONEBOT_ALLOW_SUBDIRS=0
export ONEBOT_VOICE_RECURSIVE=0

# 发送 record 的方式：path 或 base64
export ONEBOT_RECORD_MODE="path"
export ONEBOT_FILE_PREFIX="file://"
export ONEBOT_BASE64_PREFIX="base64://"

# OneBot 超时（秒）
export ONEBOT_TIMEOUT="10"

# 默认目标的另一种写法
export ONEBOT_DEFAULT_TARGET_TYPE=""
export ONEBOT_DEFAULT_TARGET_ID=""
```
备注：`ONEBOT_RECORD_MODE=base64` 仅影响语音消息段；`upload_*_file` 仍要求 OneBot 能访问文件路径。
