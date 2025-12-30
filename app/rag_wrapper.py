"""
简化的 RAG 包装器 - 直接调用 query.py 中的函数
"""

from query import process_single_query, get_rag_components
from typing import Generator


class RAGWrapper:
    """RAG 系统包装器"""
    
    def __init__(self):
        # 触发组件初始化
        get_rag_components()
    
    def query(self, question: str) -> Generator[str, None, None]:
        """调用 query.py 中的处理函数"""
        return process_single_query(question)


def get_rag_system() -> RAGWrapper:
    """获取 RAG 系统实例"""
    return RAGWrapper()
