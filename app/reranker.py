import requests
import logging
from typing import Sequence, Optional
from pydantic import Field
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_core.callbacks.manager import Callbacks

logger = logging.getLogger(__name__)

class XinferenceRerank(BaseDocumentCompressor):
    url: str = Field(..., description="Xinference Rerank URL")
    model_uid: str = Field(..., description="模型 UID")
    top_n: int = Field(default=3, description="保留文档数")
    score_threshold: Optional[float] = Field(default=None)
    request_timeout: int = Field(default=30) # 增加超时时间
    max_chars: int = Field(default=1000, description="单文档最大字符数限制，防止OOM")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        # [Fix] 防爆截断：只取前 max_chars 个字符发给模型
        # Rerank 只需要开头和关键部分就能判断相关性，不需要全文
        doc_contents = [doc.page_content[:self.max_chars] for doc in documents]
        
        payload = {
            "model": self.model_uid,
            "query": query,
            "documents": doc_contents,
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout
            )
            response.raise_for_status()
            api_result = response.json()
        except Exception as e:
            logger.error(f"Xinference Rerank 失败: {e}")
            # 降级：如果失败，返回前 top_n 个原始文档，保证流程不中断
            return documents[:self.top_n]

        ranked_results = api_result.get("results", [])
        final_results = []

        for res in ranked_results:
            index = res.get("index")
            score = res.get("relevance_score")

            if index is None or index >= len(documents):
                continue
            
            if self.score_threshold is not None and score < self.score_threshold:
                continue

            doc = documents[index]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)

        return final_results[:self.top_n]