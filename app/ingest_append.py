import json
import os
import uuid
import warnings
from typing import List, Tuple

import env_loader  # load .env if present

# 屏蔽警告
warnings.filterwarnings("ignore")

# 核心组件
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from langchain_community.embeddings import XinferenceEmbeddings
from langchain_core.documents import Document

# --- 配置 (必须与原 ingest.py 保持完全一致) ---
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_UID", "bge-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story_knowledge_base")
SPARSE_VECTOR_NAME = "langchain-sparse" # 必须与建表时一致

def resolve_default_append_path() -> str:
    """
    选择一个实际存在的默认数据路径，避免用户直接回车后指向不存在的文件。
    仅支持 timeline_flow_optimized.json 新格式。
    """
    candidates = [
        os.getenv("APPEND_DATA_FILE", "").strip(),
        "/data/timeline_flow_optimized.json",
        "data/timeline_flow_optimized.json",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("未找到可用的增量数据文件，请设置 APPEND_DATA_FILE。")

def _normalize_text(item: dict) -> str:
    """仅支持 timeline_flow_optimized.json 的 text/script。"""
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        return text
    script = item.get("script", [])
    if isinstance(script, list) and script:
        lines = []
        for turn in script:
            speaker = turn.get("c")
            text = turn.get("t", "")
            prefix = f"{speaker}: " if speaker else ""
            lines.append(f"{prefix}{text}")
        return "\n".join(lines)
    raise ValueError("新格式数据缺少 text 或 script")


def _collect_strings(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_collect_strings(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_collect_strings(item))
        return out
    return []


def _extract_meta_tokens(item: dict) -> List[str]:
    ctx = item.get("ctx") or {}
    tokens = []
    tokens.extend(_collect_strings(ctx.get("chars")))
    tokens.extend(_collect_strings(ctx.get("loc")))
    tokens.extend(_collect_strings(ctx.get("time")))
    tokens.extend(_collect_strings(ctx.get("emo")))
    tokens.extend(_collect_strings(ctx.get("state_emo")))
    seen = set()
    deduped = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped[:120]


def load_data_with_ids(file_path: str, order_offset: int = 0) -> Tuple[List[Document], List[str]]:
    """
    仅支持 timeline_flow_optimized.json，新格式缺失关键字段将直接报错。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("数据文件格式错误，需为列表或单条对象。")
    
    docs = []
    ids = []
    
    print(f"📊 正在解析 {len(data)} 条新数据...")

    for order_idx, item in enumerate(data):
        ctx = item.get("ctx") or {}
        stats = item.get("stats") or {}
        timeline = item.get("timeline") or {}

        raw_id = item.get("id") or item.get("scene")
        if raw_id:
            # 只要 ID 相同，生成的 UUID 就相同 -> 覆盖旧数据
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))
        else:
            point_id = str(uuid.uuid4())
            raw_id = ""

        processed_meta = {
            "scene": item.get("scene") or raw_id or "unknown",
            "id": raw_id,
            "source": item.get("src", ""),
            "order": order_offset + order_idx,
            "chars": ctx.get("chars") or [],
            "voices": ctx.get("voices") or [],
            "loc": ctx.get("loc"),
            "time": ctx.get("time"),
            "bgm": ctx.get("bgm", ""),
            "type": ctx.get("type", ""),
            "stats": stats,
            "timeline": timeline,
            "act": ctx.get("act") or {},
            "emo": ctx.get("emo") or {},
            "state_act": ctx.get("state_act") or {},
            "state_emo": ctx.get("state_emo") or {},
            "state": ctx.get("state"),
            "weather": ctx.get("weather"),
            "merged_from": item.get("merged_from") or [],
            "script": item.get("script", []),
        }

        content = _normalize_text(item)
        meta_tokens = _extract_meta_tokens(item)
        if meta_tokens:
            content = f"{content}\n\n[meta] " + " ".join(meta_tokens)
        ids.append(point_id)
        docs.append(Document(page_content=content, metadata=processed_meta))
        
    return docs, ids

def main():
    print(f"\n🚀 启动增量入库模式 (Incremental Ingestion)...")
    print(f"🎯 目标集合: {COLLECTION_NAME}")

    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 安全检查：确保集合存在
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ 错误: 集合 '{COLLECTION_NAME}' 不存在！")
        print("   请先运行 ingest.py 进行初始化建表。")
        return

    # 2. 输入新数据路径
    try:
        default_path = resolve_default_append_path()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    file_path = input(f"📂 请输入新数据文件路径 [默认: {default_path}]: ").strip()
    if not file_path:
        file_path = default_path
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        print("   请确认路径，或设置 APPEND_DATA_FILE 指向正确的数据文件。")
        return

    # 3. 初始化模型 (Dense + Sparse)
    print("🔌 初始化 Embeddings (Xinference + FastEmbed)...")
    dense_embeddings = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 4. 加载新数据
    try:
        existing_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
        docs, ids = load_data_with_ids(file_path, order_offset=existing_count)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    # 5. 连接 VectorStore (注意：这里不创建集合，只连接)
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    print(f"🌊 正在追加/更新 {len(docs)} 条数据到 Qdrant...")
    
    # 6. 执行追加 (Add Documents)
    # Qdrant 默认行为：如果 ID 存在则 Update，不存在则 Insert
    vector_store.add_documents(documents=docs, ids=ids)
    
    print(f"✅ 增量入库完成！")
    print(f"💡 提示: 新数据已立即可被检索。")

if __name__ == "__main__":
    main()
