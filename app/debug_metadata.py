from qdrant_client import QdrantClient
import os
import json

import env_loader  # load .env if present

client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
COLLECTION_NAME = "story_knowledge_base"

# 拉取 1 条数据看看
res = client.scroll(collection_name=COLLECTION_NAME, limit=1, with_payload=True)
if res[0]:
    print("🔎 第一条数据的完整 Payload 结构：")
    print(json.dumps(res[0][0].payload, indent=2, ensure_ascii=False))
else:
    print("❌ 集合为空")
