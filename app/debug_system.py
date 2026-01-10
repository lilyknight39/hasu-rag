import os
import json
import requests
from qdrant_client import QdrantClient

import env_loader  # load .env if present

# 配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "story_knowledge_base")
RERANK_MODEL = os.getenv("RERANK_MODEL_UID", "bge-reranker-v2-m3")

def check_qdrant_data():
    print(f"\n🔍 诊断 1: 检查 Qdrant 数据完整性...")
    try:
        client = QdrantClient(url=QDRANT_URL)
        # 随机取一条数据看看 Payload 长什么样
        res = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True
        )
        points, _ = res
        
        if not points:
            print("❌ Qdrant 集合为空！请重新运行 ingest.py。")
            return
            
        payload = points[0].payload
        print("✅ 成功读取一条数据。Payload 结构如下:")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        
        # 检查关键字段
        if "scene" not in payload:
            print("⚠️ 警告: Payload 中缺失 'scene' 字段！这就是显示'未知场景'的原因。")
            print("   -> 请检查 stories.json 的结构，确保 meta 字段下有 scene。")
        else:
            print("✅ 'scene' 字段存在。")
            
    except Exception as e:
        print(f"❌ 连接 Qdrant 失败: {e}")

def check_xinference_rerank():
    print(f"\n🔍 诊断 2: 测试 Xinference Rerank 服务...")
    url = f"{XINFERENCE_URL.rstrip('/')}/v1/rerank"
    
    # 构造一个极简单的请求，排除文本过长的问题
    payload = {
        "model": RERANK_MODEL,
        "query": "你好",
        "documents": ["你好，我是测试文档。", "今天天气不错。"]
    }
    
    print(f"   -> 发送请求到: {url}")
    print(f"   -> 模型 UID: {RERANK_MODEL}")
    
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        
        if response.status_code == 200:
            print("✅ Rerank 服务响应正常！")
            print("   结果:", response.json())
        else:
            print(f"❌ Rerank 服务报错: Status {response.status_code}")
            print("   错误详情:", response.text)
            print("   -> 建议: 登录 Xinference 后台检查模型日志，可能是显存不足(OOM)。")
            
    except Exception as e:
        print(f"❌ 请求发送失败: {e}")

if __name__ == "__main__":
    check_qdrant_data()
    check_xinference_rerank()
