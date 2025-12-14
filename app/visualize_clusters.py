import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from qdrant_client import QdrantClient

# --- 配置 ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "story_knowledge_base"

def fetch_vectors(client):
    """从 Qdrant 拉取所有向量 (需包含 Metadata 以便排序)"""
    print("📥 正在拉取向量数据...")
    points = []
    
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_vectors=True,
            with_payload=True, # 需要 Payload 里的 ID/Time 来排序
            offset=offset
        )
        batch, next_offset = result
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
        
    # [关键] 必须按 ID/时间排序，否则时间约束聚类会失效
    # 假设你的 ID 是有序的，或者是 UUID 但入库顺序是对的
    # 如果 metadata 里有 scene_id 或 time，最好用那个排序
    # points.sort(key=lambda p: p.payload.get('scene', '')) 
    points.sort(key=lambda p: p.id) 

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
            
    return np.array(vectors_list)

def main():
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 获取数据
    X = fetch_vectors(client)
    print(f"✅ 获取到 {len(X)} 个向量 (已按时间排序)")

    # 2. 执行时间约束聚类 (与 build_hierarchy.py 保持一致)
    n_clusters = max(1, len(X) // 8)
    print(f"🧮 执行 Agglomerative Clustering (目标簇数: {n_clusters})...")
    
    # 构建连接性矩阵 (只能连接前后邻居)
    connectivity = kneighbors_graph(X, n_neighbors=1, include_self=False)
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        connectivity=connectivity,
        linkage='ward'
    )
    labels = model.fit_predict(X)

    # [视觉优化] 重新映射 Label ID 以便颜色渐变
    # Agglomerative 返回的 label 可能是乱的 (例如第1段是 label 5，第2段是 label 100)
    # 我们将其重置为 0, 1, 2... 按出现顺序，这样画出来的颜色就是彩虹色
    unique_labels = {}
    new_labels = []
    counter = 0
    for label in labels:
        if label not in unique_labels:
            unique_labels[label] = counter
            counter += 1
        new_labels.append(unique_labels[label])
    labels = np.array(new_labels)

    # 3. 执行降维 (1024D -> 2D)
    perp = min(30, len(X) - 1)
    print(f"📉 执行 t-SNE 降维 (Perplexity: {perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 4. 绘图
    print("🎨 正在生成“剧情时间线”散点图...")
    plt.figure(figsize=(14, 10))
    
    # A. 绘制轨迹线 (连接点，显示时间流动)
    # alpha=0.1 让线很淡，不要喧宾夺主
    plt.plot(X_2d[:, 0], X_2d[:, 1], c='gray', alpha=0.15, linewidth=0.5)

    # B. 绘制散点
    # c=labels: 现在的 labels 是按时间递增的 (0 -> N)
    # cmap='turbo': 这是一个非常适合显示序列数据的彩虹色谱
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='turbo', s=15, alpha=0.8)
    
    plt.title(f'Story Timeline Visualization\n(Time-Constrained Clustering, {n_clusters} scenes)', fontsize=16)
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    
    # Colorbar 显示 "Scene Index"
    cbar = plt.colorbar(scatter)
    cbar.set_label('Timeline Progression (Early -> Late)')
    
    # 5. 保存
    output_file = "cluster_timeline.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 图片已保存为: {output_file}")
    print(f"   (请将此文件复制出来查看)")

if __name__ == "__main__":
    main()