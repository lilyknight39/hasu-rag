import json
import os
import uuid
import warnings
from typing import List, Tuple

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
    优先使用环境变量，其次尝试增量示例文件，再回退到当前可用的全量文件。
    """
    candidates = [
        os.getenv("APPEND_DATA_FILE", "").strip(),
        "/data/new_stories.json",
        "data/new_stories.json",
        "/data/optimized_final.json",
        "data/optimized_final.json",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    # 如果都不存在，保留旧默认值，后续会有明确报错
    return "/data/new_stories.json"

def load_data_with_ids(file_path: str) -> Tuple[List[Document], List[str]]:
    """
    加载数据逻辑保持不变，确保 ID 生成算法一致 (UUID5)，
    这样如果数据重复，Qdrant 会执行更新而不是插入重复项。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict): data = [data]
    
    docs = []
    ids = []
    
    print(f"📊 正在解析 {len(data)} 条新数据...")

    for item in data:
        # 1. 处理 Metadata
        raw_meta = item.get("metadata", {}).copy()
        processed_meta = {
            "scene": raw_meta.get("scene_id", "unknown"),
            "chars": raw_meta.get("characters", []),
            "time": raw_meta.get("time_period", ""),
            "loc": raw_meta.get("location", ""),
            "source": raw_meta.get("source_file", ""),
            "dialogues": json.dumps(raw_meta.get("dialogues", []), ensure_ascii=False)
        }
        
        # 2. 生成 ID (关键：使用相同的种子生成 UUID)
        raw_id = item.get("chunk_id")
        if raw_id:
            # 只要 chunk_id 相同，生成的 UUID 就相同 -> 覆盖旧数据
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            processed_meta["id"] = raw_id
        else:
            point_id = str(uuid.uuid4())
            
        ids.append(point_id)
        docs.append(Document(page_content=item.get("content"), metadata=processed_meta))
        
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
    default_path = resolve_default_append_path()
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
        docs, ids = load_data_with_ids(file_path)
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
