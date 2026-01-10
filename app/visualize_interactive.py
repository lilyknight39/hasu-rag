import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from qdrant_client import QdrantClient

# --- 配置 ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "story_knowledge_base"

def fetch_data(client):
    """从 Qdrant 拉取向量 + 智能解析元数据"""
    print("📥 正在拉取数据 (Vectors + Metadata)...")
    points = []
    
    offset = None
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_vectors=True,
            with_payload=True,
            offset=offset
        )
        batch, next_offset = result
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
        
    # 按时间顺序排序：优先 metadata.order，其次 scene/id
    def sort_key(p):
        payload = p.payload or {}
        meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
        order = meta.get("order")
        if order is not None:
            try:
                return (0, int(order))
            except Exception:
                return (0, str(order))
        scene = meta.get("scene") or meta.get("scene_id") or meta.get("id")
        if scene:
            return (1, str(scene))
        return (2, str(p.id))

    points.sort(key=sort_key) 
    
    vectors_list = []
    meta_list = []

    print(f"🔍 正在解析 {len(points)} 条数据的元数据...")

    for p in points:
        # 1. 提取向量 (保持不变)
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
            
        # 2. [修复核心] 智能提取元数据
        payload = p.payload
        
        # 辅助函数：尝试从 payload 或 payload['metadata'] 中查找多个可能的键
        def get_value(possible_keys, default_val):
            # A. 检查 payload 根目录
            for k in possible_keys:
                if k in payload and payload[k]:
                    return payload[k]
            
            # B. 检查 payload['metadata'] (LangChain 常见嵌套)
            if 'metadata' in payload and isinstance(payload['metadata'], dict):
                sub_meta = payload['metadata']
                for k in possible_keys:
                    if k in sub_meta and sub_meta[k]:
                        return sub_meta[k]
            
            return default_val

        # 查找 Scene (兼容 scene, scene_id)
        scene = get_value(['scene', 'scene_id', 'id'], 'Unknown Scene')
        
        # 查找 Location (兼容 loc, location, place)
        loc = get_value(['loc', 'location', 'place', 'spot'], 'Unknown Loc')
        
        # 查找 Characters (兼容 chars, characters, roles)
        # 注意：这里拿到的可能是 List，也可能是 JSON String
        chars_raw = get_value(['chars', 'characters', 'roles'], [])
        
        # 3. 数据清洗与格式化
        # 处理 characters: 如果是字符串形式的 JSON，尝试解析
        if isinstance(chars_raw, str):
            if chars_raw.startswith('[') or chars_raw.startswith('{'):
                try:
                    chars_obj = json.loads(chars_raw)
                    if isinstance(chars_obj, list):
                        chars_str = ", ".join(chars_obj)
                    else:
                        chars_str = str(chars_obj)
                except:
                    chars_str = chars_raw
            else:
                chars_str = chars_raw
        elif isinstance(chars_raw, list):
            chars_str = ", ".join([str(c) for c in chars_raw])
        else:
            chars_str = str(chars_raw)

        # 截取内容预览
        content = payload.get('page_content', '')
        # 如果根目录没有 content，去 metadata 找
        if not content and 'metadata' in payload:
             content = payload['metadata'].get('page_content', '')
        
        content_preview = content[:100].replace("\n", " ") + "..."

        meta_list.append({
            'id': p.id,
            'scene': scene,
            'characters': chars_str,
            'content': content_preview,
            'location': loc
        })
            
    return np.array(vectors_list), meta_list

def main():
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. 获取数据
    X, metadata = fetch_data(client)
    print(f"✅ 获取到 {len(X)} 条数据")

    # 2. 执行时间约束聚类
    n_clusters = max(1, len(X) // 8)
    print(f"🧮 执行 Agglomerative Clustering (目标簇数: {n_clusters})...")
    
    connectivity = kneighbors_graph(X, n_neighbors=1, include_self=False)
    model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
    labels = model.fit_predict(X)

    # 重新映射 Labels 颜色 (0->N)
    unique_labels = {}
    new_labels = []
    counter = 0
    for label in labels:
        if label not in unique_labels:
            unique_labels[label] = counter
            counter += 1
        new_labels.append(unique_labels[label])
    
    # 3. 降维
    perp = min(30, len(X) - 1)
    print(f"📉 执行 t-SNE 降维 (Perplexity: {perp})...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 4. 构建 Pandas DataFrame (这是 Plotly 的核心)
    print("📊 正在构建交互式图表...")
    df = pd.DataFrame(metadata)
    df['x'] = X_2d[:, 0]
    df['y'] = X_2d[:, 1]
    df['Cluster_ID'] = new_labels
    df['Time_Flow'] = df['Cluster_ID'] # 用于颜色条

    # 5. 生成 Plotly 图表
    fig = px.scatter(
        df, 
        x='x', 
        y='y',
        color='Time_Flow', # 颜色依据
        color_continuous_scale=px.colors.sequential.Turbo, # 彩虹配色
        hover_data={
            'x': False, # 隐藏坐标值
            'y': False,
            'Time_Flow': False,
            'Cluster_ID': True,
            'scene': True,       # 显示场景ID
            'location': True,    # 显示地点
            'characters': True,  # 显示角色
            'content': True      # 显示内容预览
        },
        title=f"Story Timeline Interactive Map ({n_clusters} Scenes)",
        width=1200,
        height=800
    )

    # 优化显示：把点稍微画大一点，半透明
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    
    # 增加连线 (这是一个稍微复杂的操作，Plotly 默认不方便给 Scatter 加线)
    # 我们添加一个 Line Trace 来模拟轨迹
    fig.add_scatter(
        x=df['x'], y=df['y'], 
        mode='lines', 
        line=dict(color='gray', width=0.5), 
        opacity=0.2, 
        name='Timeline Path',
        hoverinfo='skip'
    )

    # 6. 保存为 HTML
    output_file = "interactive_timeline.html"
    fig.write_html(output_file)
    print(f"✅ 交互式图表已保存为: {output_file}")
    print(f"   (请将此文件复制到宿主机，用 Chrome/Edge/Safari 打开)")

if __name__ == "__main__":
    main()
