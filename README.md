# Hasu RAG: 链接！ 喜欢！ 爱生活！ 剧情智能分层问答系统

基于 **LangChain v1.x**、**Qdrant** 和 **Xinference** 的长篇剧情 RAG。特点：混合检索 (Dense + BM25)、时间序约束聚类、双路上下文融合，既能回答细节追问也能做宏观/分析类总结。

---

## 最新变更与支持范围
- 仅支持 **`data/timeline_flow_optimized.json`** 新格式；旧版 stories/optimized_final/new_stories 路径已废弃。
- 入库生成 ID：`scene/id` -> UUID5（稳定可覆盖），按文件顺序写入 `metadata.order` 保障时间线。
- Metadata 完整保留 `ctx/stats/timeline/act/emo/state_*`、台词 `dialogues` 与来源 `src`，所有聚类与可视化均按 `metadata.order` 排序。

---

## 数据格式（timeline_flow_optimized.json）
- **顶层字段**：`id`、`scene`、`src`、`text`（已扁平化）、`script`（对话列表）、`ctx`、`stats`、`timeline`、`merged_from`（可选）。
- **ctx**：`chars`、`voices`、`loc`、`time`、`bgm`、`type`、`act`、`emo`、`state_act`、`state_emo` 等。
- **stats**：`tok`（token 数）、`dlg`（对话轮次）。
- **timeline**：按角色记录动作/情感演变。
- 入库内容来源：优先 `text`，缺失则拼接 `script` 的 `c/t` 字段。

---

## 系统架构
- **双层知识库**
  - `story_knowledge_base`：原始剧情切片，Hybrid 检索 (Dense + Sparse)。
  - `story_summary_store`：时间序聚类摘要，Dense 检索。
- **查询路由**
  - 意图分类：`fact` / `overview` / `analysis`。
  - `fact` → 细节库；`overview` → 摘要库；`analysis` → 摘要 + 细节双路融合。
- **时间序聚类**：Agglomerative + 邻接约束，仅合并相邻片段；LLM 生成摘要并按指纹增量更新。

---

## 工作流
1) **安装依赖**  
   ```bash
   pip install -r requirements.txt  # Python 3.11
   ```
2) **配置环境变量（示例）**
   ```bash
   export QDRANT_URL="http://localhost:6333"
   export XINFERENCE_SERVER_URL="http://your-server:9997"
   export LLM_BASE_URL="http://your-server:9001/v1"
   export LLM_API_KEY=""
   export LLM_MODEL_NAME=""
   # 可选：自定义数据路径
   export DATA_FILE="/path/to/timeline_flow_optimized.json"
   export APPEND_DATA_FILE="/path/to/new_timeline_data.json"
   ```
3) **全量入库（会重建集合）**  
   ```bash
   python app/ingest.py
   ```
4) **增量/幂等追加**  
   ```bash
   python app/ingest_append.py
   ```
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
