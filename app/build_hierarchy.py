import os
import json
import time
import numpy as np
import warnings
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import XinferenceEmbeddings

# 屏蔽警告
warnings.filterwarnings("ignore")

# --- 配置 ---
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
SOURCE_COLLECTION = "story_knowledge_base" 
SUMMARY_COLLECTION = "story_summary_store" 

# LLM 配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.123.113:9001/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-anonanon")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

def fetch_all_vectors(client, collection_name):
    """从 Qdrant 拉取所有向量"""
    print("📥 原始データチャンクを取得中 (Fetching chunks)...")
    points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=True,
            offset=offset
        )
        batch, next_offset = result
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    
    # [关键] 必须按 ID 或时间排序，否则聚类会乱
    # 假设 ID 是有序的 UUIDv5 或者入库时已有序
    # 如果 metadata 里有 timestamp，最好用 metadata 排序
    points.sort(key=lambda p: p.id)
    
    print(f"✅ 合計 {len(points)} 個の断片を取得しました。")
    return points

def perform_clustering(points, n_clusters=None):
    """使用带时间约束的层次聚类"""
    print(f"🧮 時間制約付きクラスタリングを実行中 (Time-Constrained Agglomerative)...")
    
    if not points: return {}

    # 1. 提取向量
    vectors_list = []
    for p in points:
        vec = p.vector
        if isinstance(vec, dict):
            dense_vec = vec.get("", vec.get("default"))
            if dense_vec is None:
                for v in vec.values():
                    if isinstance(v, list):
                        dense_vec = v
                        break
            vectors_list.append(dense_vec)
        elif isinstance(vec, list):
            vectors_list.append(vec)
    
    X = np.array(vectors_list)
    
    # 2. 连接性约束 (Time Constraints)
    connectivity = kneighbors_graph(X, n_neighbors=1, include_self=False)

    # 3. 动态确定簇数
    if n_clusters is None:
        target_n_clusters = max(1, len(X) // 8)
    else:
        target_n_clusters = n_clusters
    
    print(f"   🎯 目標クラスタ数: {target_n_clusters}")

    # 4. 聚类
    model = AgglomerativeClustering(
        n_clusters=target_n_clusters, 
        connectivity=connectivity,
        linkage='ward' 
    )
    model.fit(X)
    
    # 5. 整理结果并按时间重排序
    raw_clusters = {}
    for idx, label in enumerate(model.labels_):
        if label not in raw_clusters: raw_clusters[label] = []
        raw_clusters[label].append(points[idx])
    
    # 根据每个簇中第一个片段的ID进行排序，保证 Cluster 0 是最早的剧情
    sorted_keys = sorted(raw_clusters.keys(), key=lambda k: raw_clusters[k][0].id)
    
    ordered_clusters = {}
    for new_id, old_id in enumerate(sorted_keys):
        ordered_clusters[new_id] = raw_clusters[old_id]
        
    return ordered_clusters

def generate_summary(llm, cluster_points):
    """生成日文摘要"""
    texts = [p.payload.get('page_content', '') for p in cluster_points]
    combined_text = "\n---\n".join(texts)[:20000] # 略微增大 Context Limit
    
    template = """あなたは『蓮ノ空女学院スクールアイドルクラブ』のシナリオ編集の専門家です。
以下のテキストは、時系列順に並んだ一連のストーリー断片です。

【タスク】:
これらの断片を統合し、**何が起きたか、誰がどう感じたか、重要な変化は何か**を要約してください。

【制約事項（厳守）】:
1. **出力は要約本文のみ**（挨拶不要）。
2. 日本語で記述。
3. 200〜500文字程度。
4. 抽象的な表現だけでなく、具体的な出来事（例：誰がどこに行った、何の曲を練習した）を含めること。

【テキスト断片】:
{context}

【要約出力】:
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({"context": combined_text}).strip()
    except Exception as e:
        print(f"⚠️ Summary Error: {e}")
        return "要約生成エラー"

def main():
    client = QdrantClient(url=QDRANT_URL)
    
    print("\n" + "="*50)
    print("🛠️  Build Hierarchy Index (Smart Incremental Mode)")
    print("="*50)
    
    # 1. 拉取所有数据 (包括刚刚 Append 进去的)
    all_points = fetch_all_vectors(client, SOURCE_COLLECTION)
    if not all_points: return

    # 2. 重新进行全局聚类
    # 注意：即便只加了1条数据，为了保证时间切分的全局最优，我们也重新跑一次聚类。
    # 因为聚类很快（几秒钟），而且不花钱。真正花钱的是后面的 LLM。
    clusters = perform_clustering(all_points)
    
    # 3. 准备摘要集合
    if not client.collection_exists(SUMMARY_COLLECTION):
        print(f"🆕 创建新摘要集合...")
        client.create_collection(
            collection_name=SUMMARY_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    
    # 初始化模型
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL_NAME, temperature=0.1
    )
    embed_model = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid="bge-m3")

    print("\n📝 智能同步摘要 (Smart Syncing)...")
    
    stats_skipped = 0
    stats_updated = 0
    
    total_clusters = len(clusters)

    for cluster_id, points in clusters.items():
        # A. 构建当前簇的“指纹” (使用包含的子片段 ID 列表)
        # 如果子片段列表完全一致，说明这个簇的内容没有变
        current_child_ids = [p.id for p in points]
        
        # B. 检查数据库里是否已有这个 Cluster ID
        existing_points = client.retrieve(
            collection_name=SUMMARY_COLLECTION,
            ids=[cluster_id],
            with_payload=True
        )
        
        should_generate = True
        
        if existing_points:
            existing_payload = existing_points[0].payload
            existing_child_ids = existing_payload.get('child_ids', [])
            
            # [核心逻辑] 对比指纹
            # 注意：列表比较要求顺序一致，我们假设 points 已经按时间排好序
            if current_child_ids == existing_child_ids:
                should_generate = False
                print(f"   ⏩ Cluster {cluster_id+1}/{total_clusters} 无变化 (Skipped).")
                stats_skipped += 1
            else:
                print(f"   🔄 Cluster {cluster_id+1}/{total_clusters} 内容变更/偏移，需更新 (Updating)...")
                # 可能是因为新数据插入导致切分点变了，或者追加了新数据
        else:
            print(f"   🆕 Cluster {cluster_id+1}/{total_clusters} 是新簇 (New)...")

        # C. 如果需要生成 (新簇 或 内容变更)
        if should_generate:
            try:
                # 1. 生成摘要
                summary_text = generate_summary(llm, points)
                
                # 2. 向量化
                vector = embed_model.embed_query(summary_text)
                
                # 3. 存入 (Upsert 会覆盖旧 ID)
                point = PointStruct(
                    id=cluster_id, 
                    vector=vector,
                    payload={
                        "page_content": summary_text,
                        "child_ids": current_child_ids, # 存入新指纹
                        "level": "summary",
                        "count": len(points)
                    }
                )
                client.upsert(collection_name=SUMMARY_COLLECTION, points=[point])
                stats_updated += 1
                
                # 稍微预览一下
                print(f"      -> 摘要生成完毕: {summary_text[:30]}...")
                
            except Exception as e:
                print(f"❌ Error on cluster {cluster_id}: {e}")
                time.sleep(2)

    print(f"\n🎉 同步完成！")
    print(f"   - 跳过 (无变化): {stats_skipped}")
    print(f"   - 更新/新增: {stats_updated}")

if __name__ == "__main__":
    main()