import os
import time
import re

# [New] 引入混合检索组件
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import XinferenceEmbeddings
from qdrant_client import QdrantClient
from reranker import XinferenceRerank

# 智能导入 Retriever
try:
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever

# 配置
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_UID", "bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL_UID", "bge-reranker-v2-m3")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story_knowledge_base")

def get_snippet(text, query, window=100):
    clean_query = re.sub(r'[?？!！,，.。]', '', query)
    clean_text = text.replace('\n', ' ')
    if clean_query in clean_text:
        idx = clean_text.find(clean_query)
        start = max(0, idx - window)
        end = min(len(clean_text), idx + len(clean_query) + window)
        return f"...{clean_text[start:end]}...", True
    else:
        return f"{clean_text[:400]}...", False

def main():
    print(f"\n⚔️ 启动混合检索评估 (Hybrid Search Mode)...")
    
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 初始化双路 Embedding (必须与 Ingest 时一致)
    dense_embeddings = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # 2. 初始化混合 VectorStore
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME, 
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings, # 关键：注入稀疏模型
        retrieval_mode=RetrievalMode.HYBRID # 关键：开启混合模式
    )
    
    # 3. 粗排配置 (Hybrid Recall)
    # 混合检索会同时跑向量和关键词，然后在 Qdrant 内部做 RRF 融合
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 50})

    # 4. 精排配置 (Rerank)
    reranker = XinferenceRerank(
        url=f"{XINFERENCE_URL.rstrip('/')}/v1/rerank",
        model_uid=RERANK_MODEL,
        top_n=5
    )

    # 5. 管道组装
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    while True:
        print("\n" + "="*60)
        query = input("问题 (输入 'q' 退出): ")
        if query.lower() in ['q', 'exit']: break
        
        start_t = time.time()
        print(f"🔍 [Hybrid] Qdrant(Dense+Sparse) -> [Rerank] Xinference...")
        
        try:
            results = compression_retriever.invoke(query)
            cost_t = time.time() - start_t
            
            print(f"\n耗时 {cost_t:.2f}s | 召回 {len(results)} 条结果:")
            print("-" * 60)
            
            for i, doc in enumerate(results):
                score = doc.metadata.get("relevance_score", 0)
                scene = doc.metadata.get("scene", "未知")
                
                snippet, is_hit = get_snippet(doc.page_content, query)
                hit_mark = "精确命中" if is_hit else "语义/RRF相关"
                
                print(f"Rank #{i+1} | Score: {score:.4f} | {hit_mark}")
                print(f"   场景: {scene}")
                print(f"   内容: {snippet}")
                print("-" * 60)
                
        except Exception as e:
            print(f"检索失败: {e}")

if __name__ == "__main__":
    main()