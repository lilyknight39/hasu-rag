import os
import json
import time
import numpy as np
import warnings
from sklearn.cluster import KMeans
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import XinferenceEmbeddings
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# 屏蔽警告
warnings.filterwarnings("ignore")

# --- 配置 ---
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
SOURCE_COLLECTION = "story_knowledge_base" 
SUMMARY_COLLECTION = "story_summary_store" 

# LLM 配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.example.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "your-model-name")

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
    print(f"✅ 合計 {len(points)} 個の断片を取得しました。")
    return points

def perform_clustering(points, n_clusters=None):
    """
    [Advanced] 使用 '带连接性约束的层次聚类' (Agglomerative with Connectivity)
    结合了 K-Means 的全局优化能力和 Sequential 的时间约束。
    """
    print(f"🧮 実行中: 時間制約付き階層的クラスタリング (Constrained Agglomerative Clustering)...")
    
    if not points:
        return {}

    # 1. 提取向量 (复用之前的健壮逻辑)
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
    
    # 2. 关键步骤：构建连接性矩阵 (Connectivity Matrix)
    # 这告诉算法：X[i] 只能和 X[i-1] 或 X[i+1] 合并，不能跨时间跳跃
    # n_neighbors=1 表示只连接最近的 1 个邻居（即时间上的前一个/后一个）
    connectivity = kneighbors_graph(X, n_neighbors=1, include_self=False)

    # 3. 动态确定簇的数量
    # 如果没有指定 n_clusters，我们可以基于压缩比自动计算
    # 或者使用 distance_threshold (距离阈值) 让数据自己决定
    if n_clusters is None:
        target_n_clusters = max(1, len(X) // 8) # 默认每8个合并为一个
    else:
        target_n_clusters = n_clusters

    print(f"   🎯 目標クラスタ数: {target_n_clusters}")

    # 4. 执行聚类
    # linkage='ward': 最小化簇内的方差（最常用的合并策略）
    # connectivity=connectivity: 加上时间锁链
    model = AgglomerativeClustering(
        n_clusters=target_n_clusters, 
        connectivity=connectivity,
        linkage='ward' 
    )
    
    model.fit(X)
    
    # 5. 整理结果
    clusters = {}
    for idx, label in enumerate(model.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[idx])
    
    # [重要优化] 层次聚类的 label ID 是乱序的，我们需要按时间重新排序 Cluster ID
    # 否则生成的摘要顺序会乱
    sorted_cluster_ids = sorted(clusters.keys(), key=lambda k: clusters[k][0].id)
    
    ordered_clusters = {}
    for new_id, old_id in enumerate(sorted_cluster_ids):
        ordered_clusters[new_id] = clusters[old_id]
        
    print(f"✅ クラスタリング完了: {len(ordered_clusters)} 個のシーケンスに分割されました。")
    return ordered_clusters

def generate_summary(llm, cluster_points):
    """
    [Upgrade] 使用纯日文 Prompt，并强制格式化输出
    """
    texts = [p.payload['page_content'] for p in cluster_points]
    combined_text = "\n---\n".join(texts)[:15000]
    
    # 纯日文 Prompt，强调剧情概括和格式控制
    template = """あなたは『蓮ノ空女学院スクールアイドルクラブ』のシナリオ編集の専門家です。
以下のテキストは、意味的に類似した一連のストーリー断片（同じ時間帯やテーマ）です。

【タスク】:
これらの断片を読み、**核心的な出来事、主な対立、関与するキャラクター**を要約した、簡潔なあらすじを作成してください。

【制約事項（厳守）】:
1. **出力は要約本文のみ**としてください。「はい」「以下は要約です」などの挨拶や前置きは**一切禁止**です。
2. 日本語で記述してください。
3. 200〜400文字程度の、客観的な三人称視点で書いてください。
4. 具体的なキャラクター名（花帆、梢など）を正確に使用してください。

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
        return "要約の生成に失敗しました。"

def main():
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 询问模式
    print("\n" + "="*50)
    print("🛠️  Build Hierarchy Index (Japanese Mode)")
    print("="*50)
    
    mode = input("请选择模式 / モード選択:\n [1] 断点续传 (Resume / 推奨)\n [2] 覆盖重写 (Overwrite / 全部削除して再作成)\n输入 1 or 2 [默认 1]: ").strip()
    
    is_overwrite = (mode == '2')
    
    # 2. 拉取数据
    all_points = fetch_all_vectors(client, SOURCE_COLLECTION)
    if not all_points: return
    
    # 按 ID 排序确保 K-Means 结果一致性
    all_points.sort(key=lambda p: p.id)

    # 3. 聚类
    n_clusters = max(1, len(all_points) // 8)
    clusters = perform_clustering(all_points, n_clusters)
    
    # 4. 集合管理
    if is_overwrite:
        if client.collection_exists(SUMMARY_COLLECTION):
            print(f"🗑️  旧摘要集合已删除 (Deleting collection)...")
            client.delete_collection(SUMMARY_COLLECTION)
        print(f"🆕 创建新集合 (Creating new collection)...")
        client.create_collection(
            collection_name=SUMMARY_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    else:
        # Resume 模式：如果集合不存在，也得创建
        if not client.collection_exists(SUMMARY_COLLECTION):
            print(f"🆕 集合不存在，正在创建 (Creating collection)...")
            client.create_collection(
                collection_name=SUMMARY_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
        else:
            print(f"🔄 正在断点续传 (Resuming)...")

    # 初始化模型
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0.1 # 摘要生成需要低温度
    )
    embed_model = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid="bge-m3")

    print("\n📝 要約生成と入庫を開始します (Generating summaries)...")
    
    existing_count = client.count(SUMMARY_COLLECTION).count
    print(f"   現在の進捗 (Current count): {existing_count}/{n_clusters}")

    for cluster_id, points in clusters.items():
        # 如果不是覆盖模式，且 ID 已存在，则跳过
        if not is_overwrite:
            check = client.retrieve(collection_name=SUMMARY_COLLECTION, ids=[cluster_id])
            if check:
                print(f"   ⏩ Cluster {cluster_id+1}/{n_clusters} は既に存在します (Skipping).")
                continue

        print(f"   ▶️ Processing Cluster {cluster_id+1}/{n_clusters} (Fragments: {len(points)})...")
        
        try:
            # A. 生成日文摘要
            summary_text = generate_summary(llm, points)
            
            # B. 计算向量
            vector = embed_model.embed_query(summary_text)
            
            # C. 上传
            child_ids = [p.id for p in points]
            point = PointStruct(
                id=cluster_id, 
                vector=vector,
                payload={
                    "page_content": summary_text,
                    "child_ids": child_ids,
                    "level": "summary",
                    "count": len(points)
                }
            )
            client.upsert(collection_name=SUMMARY_COLLECTION, points=[point])
            
            # 这里的摘要打印出来看看，确认是否纯净
            print(f"      [摘要预览]: {summary_text[:50]}...")
            
        except Exception as e:
            print(f"❌ Error processing cluster {cluster_id}: {e}")
            time.sleep(2)

    print(f"\n🎉 構築完了！ (Build Complete)")

if __name__ == "__main__":
    main()