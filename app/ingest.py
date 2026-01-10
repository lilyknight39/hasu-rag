import json
import os
import uuid
import warnings
from typing import List, Tuple

import env_loader  # load .env if present

# 屏蔽恼人的警告
warnings.filterwarnings("ignore")

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
# [New] 引入稀疏向量配置类
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain_community.embeddings import XinferenceEmbeddings
from langchain_core.documents import Document

# 配置
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_UID", "bge-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story_knowledge_base")

# 定义稀疏向量的默认名称 (LangChain 默认使用这个名字)
SPARSE_VECTOR_NAME = "langchain-sparse"

def resolve_data_file() -> str:
    """
    尝试解析数据文件位置，优先使用环境变量，其次使用容器与本地的默认路径。
    这样无论在 Docker（/data）还是本地直接运行（data/）都能找到文件。
    """
    candidates = [
        os.getenv("DATA_FILE", "").strip(),
        "/data/timeline_flow_optimized.json",
        "data/timeline_flow_optimized.json",
    ]
    candidates = [p for p in candidates if p]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"未找到可用的数据文件，请检查是否存在: {', '.join(candidates)}")

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
    # 去重并裁剪，避免元信息过长污染正文
    seen = set()
    deduped = []
    for tok in tokens:
        tok = tok.strip()
        if not tok or tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped[:120]


def load_data_with_ids(file_path: str) -> Tuple[List[Document], List[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("数据文件格式错误，需为列表或单条对象。")
    
    docs = []
    ids = []
    
    print(f"📊 正在解析 {len(data)} 条数据...")

    for order_idx, item in enumerate(data):
        ctx = item.get("ctx") or {}
        stats = item.get("stats") or {}
        timeline = item.get("timeline") or {}

        raw_id = item.get("id") or item.get("scene")
        if raw_id:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))
        else:
            point_id = str(uuid.uuid4())
            raw_id = ""

        processed_meta = {
            "scene": item.get("scene") or raw_id or "unknown",
            "id": raw_id,
            "source": item.get("src", ""),
            "order": order_idx,  # 保留原始时间顺序
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
            # 保留原始脚本以避免信息丢失（含 t/c/v/kf 等）
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
    print("🚀 开始 v1.x 混合检索入库 (Hybrid Ingestion)...")

    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 彻底重建集合 (手动定义 Hybrid Schema)
    if client.collection_exists(COLLECTION_NAME):
        print(f"🗑️ 正在清理旧集合...")
        client.delete_collection(COLLECTION_NAME)

    print(f"🛠️ 正在创建混合检索集合: {COLLECTION_NAME}")
    # [关键修复] 在初始化 VectorStore 前，手动创建好集合结构
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # A. 密集向量配置 (Dense - BGE-M3)
        # 使用空字符串 "" 作为默认向量名，这是 Qdrant 标准
        vectors_config={
            "": VectorParams(
                size=1024, 
                distance=Distance.COSINE
            )
        },
        # B. 稀疏向量配置 (Sparse - BM25)
        # 必须显式定义 sparse_vectors_config
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False, # 放在内存里更快
                )
            )
        }
    )

    # 2. 初始化模型
    print("🔌 连接 Xinference (Dense)...")
    dense_embeddings = XinferenceEmbeddings(
        server_url=XINFERENCE_URL, 
        model_uid=EMBED_MODEL
    )
    
    print("🔌 初始化 FastEmbed (Sparse)...")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 3. 加载数据
    try:
        data_path = resolve_data_file()
        print(f"📂 数据文件: {data_path}")
        docs, ids = load_data_with_ids(data_path)
    except Exception as e:
        print(f"❌ 读取或解析数据失败: {e}")
        return

    print(f"📄 准备写入 {len(docs)} 条数据...")
    
    # 4. 实例化 VectorStore
    # 现在集合已经存在了，校验可以通过
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name=SPARSE_VECTOR_NAME, # 显式指定稀疏向量名
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    print("🌊 正在生成向量并上传 (Dense + Sparse)...")
    vector_store.add_documents(documents=docs, ids=ids)
    
    print(f"✅ 混合检索库构建完成！")

if __name__ == "__main__":
    main()
