import os
import json
import time
import warnings

# 屏蔽烦人的警告
warnings.filterwarnings("ignore")

# --- 核心组件 ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 检索组件 ---
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import XinferenceEmbeddings
from qdrant_client import QdrantClient

# --- 自定义组件 ---
try:
    from reranker import XinferenceRerank
except ImportError:
    print("❌ 错误: 找不到 reranker.py")
    exit(1)

try:
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever

# ==============================================================================
# 🛠️ 配置区域 (请确保与 build_hierarchy.py 一致)
# ==============================================================================

# 1. 检索后端 (Xinference)
XINFERENCE_URL = os.getenv("XINFERENCE_SERVER_URL", "http://192.168.123.113:9997")
EMBED_MODEL = "bge-m3"
RERANK_MODEL = "bge-reranker-v2-m3"

# 2. 向量数据库 (Qdrant)
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
RAW_COLLECTION_NAME = "story_knowledge_base"  # 细节碎片
SUMMARY_COLLECTION_NAME = "story_summary_store" # 宏观摘要
SPARSE_VECTOR_NAME = "langchain-sparse"

# 3. 生成后端 (Gemini Proxy)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.example.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "your-model-name")

# ==============================================================================

def format_docs(docs, is_summary=False):
    """
    格式化检索结果，兼容细节碎片 (包含JSON metadata) 和宏观摘要。
    [Fix] 优化 format_docs 以适应两种数据格式。
    """
    formatted = []
    for i, doc in enumerate(docs):
        meta = doc.metadata.copy()
        score = meta.pop("relevance_score", 0)
        
        # 1. 处理细节碎片 (包含复杂的 JSON metadata)
        if not is_summary:
            # 智能解析嵌套 JSON
            for key, value in meta.items():
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        meta[key] = json.loads(value)
                    except:
                        pass
            
            meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
            content = (
                f"<fragment id='{i+1}' score='{score:.4f}'>\n"
                f"  <content>\n{doc.page_content}\n  </content>\n"
                f"  <metadata>\n{meta_json}\n  </metadata>\n"
                f"</fragment>"
            )
        
        # 2. 处理宏观摘要 (page_content 即摘要本身)
        else:
            # 摘要的 metadata 较简单，我们只保留关键信息
            level = meta.get("level", "summary")
            child_count = meta.get("count", 0)
            
            content = (
                f"<summary_topic id='{i+1}' score='{score:.4f}' level='{level}' child_count='{child_count}'>\n"
                f"  <topic_summary>\n{doc.page_content}\n  </topic_summary>\n"
                f"</summary_topic>"
            )

        formatted.append(content)
        
    return "\n\n".join(formatted)

def main():
    print(f"\n启动分层智能问答系统 (Hierarchical RAG)...")
    
    # 1. 初始化连接
    client = QdrantClient(url=QDRANT_URL)
    dense_emb = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL)
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # --- 初始化两个 Vector Store ---
    raw_store = QdrantVectorStore(
        client=client,
        collection_name=RAW_COLLECTION_NAME,
        embedding=dense_emb,
        sparse_embedding=sparse_emb,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        retrieval_mode=RetrievalMode.HYBRID # 细节：Hybrid Search
    )
    
    # 检查摘要集合是否存在
    if not client.collection_exists(SUMMARY_COLLECTION_NAME):
        print(f"⚠️ 注意：宏观摘要集合 '{SUMMARY_COLLECTION_NAME}' 不存在！")
        print("    请先运行 'build_hierarchy.py' 来生成摘要索引。")
        summary_store = None
    else:
        summary_store = QdrantVectorStore(
            client=client,
            collection_name=SUMMARY_COLLECTION_NAME,
            embedding=dense_emb,
            retrieval_mode=RetrievalMode.DENSE # 摘要：Dense Search 即可
        )


    # 2. 初始化 LLM
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0.7, # 用于生成回答
        streaming=True,
        max_tokens=10240
    )

    rewrite_llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=0.0, # 用于翻译和分类，保证稳定性
        streaming=False
    )
    
    # =========================================================
    # 1. 意图分类链 (Intent Classification Chain)
    # =========================================================
    intent_template = """分析用户的查询，判断它是对剧情的“宏观总结”还是“细节追问”。

【用户问题】：{query}

【判断规则】：
1. 宏观总结 (overview)：包含“概括”、“总结”、“大意”、“讲了什么”、“背景”等词，或询问主题/篇章内容。
2. 细节追问 (specific)：询问具体角色、具体台词、具体时间、具体动作、或寻找语音文件ID。

请只输出 'overview' 或 'specific'，不要包含任何其他解释。"""
    intent_prompt = ChatPromptTemplate.from_template(intent_template)
    intent_chain = intent_prompt | rewrite_llm | StrOutputParser()

    # =========================================================
    # 2. 细节检索路径 (Specific Path)
    # =========================================================
    
    # 查询重写 Prompt (与你提供的最新版一致)
    rewrite_template = """你是一个专为 **RAG 混合检索系统 (Hybrid Search)** 设计的查询优化专家。... [此处省略，逻辑与你提供的最新版本一致]"""
    # ... (将最新的 rewrite_template 放到这里)
    rewrite_template = """你是一个专为 **RAG 混合检索系统 (Hybrid Search)** 设计的查询优化专家。
该系统的下游包含两个检索引擎，你需要构造一个能同时满足它们需求的日文查询语句：

1. **语义检索引擎 (Dense Vector / BGE-M3)**: 
   - 偏好：完整的自然语言句子，包含主语、谓语、宾语。
   - 目标：理解“谁做了什么”、“某种氛围”等抽象语义。
   - 要求：**不要**破坏句子的语法结构。

2. **关键词检索引擎 (Sparse Vector / BM25)**:
   - 偏好：精确的专有名词（人名、地名、曲名）和稀有词汇。
   - 目标：通过字面匹配通过 IDF 机制过滤无关文档。
   - 要求：必须包含准确的角色名（如“乙宗梢”而非“梢”）。

3. **术语修正 (Term Correction)**: 用户可能会使用错误的汉字或不准确的称呼。请利用你的知识库进行修正。
    - 示例: "三剑士" -> 应该修正为 "三銃士" (因为莲之空剧情中常用的是“お気楽三銃士”)
    【角色名映射表 (必须严格遵守)】：
    - 梢 / 梢前辈 -> 乙宗梢
    - 花帆 / 花帆桑 -> 日野下花帆
    - 慈 / 慈前辈 -> 藤島慈
    - 瑠璃乃 -> 大沢瑠璃乃
    - 吟子 / 百生 / 小吟子 -> 百生吟子
    - 小铃 / 徒町 -> 徒町小鈴
    - 姬芽 / 安養寺 -> 安養寺姫芽
    - 塞拉斯 -> セラス (或昵称 セっちゃん)

【执行步骤】：
1. **意图理解**: ...
2. **自然翻译**: ...
3. **同义词泛化 (关键)**: 
   - 针对动作或情感描述，**必须**扩展相关的日文同义词，以确保关键词检索 (BM25) 能命中不同描述风格的片段。
   - 示例: "哭" -> 必须包含 `泣く 涙 号泣 涙声 泣き顔`
   - 示例: "笑" -> 必须包含 `笑う 笑顔 微笑む`
4. **话题锁定**: ...

【输出格式】：
[修正后的自然日文问句] [同义词扩展] [类型关键词] [核心话题词]

【示例】：
用户: 梢哭了几次？
输出: 乙宗梢は何回泣きましたか？ 泣く 涙 号泣 涙声 泣き顔 シーン (包含所有"哭"相关的词)

用户: 谁在练习室生气了？
输出: 練習室で誰が怒りましたか？ 怒る 激怒 不機嫌 喧嘩 場所

用户问题: {question}
"""
    rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)
    # 重写链使用温度为 0 的 LLM
    rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()
    
    # Reranker 配置
    reranker = XinferenceRerank(
        url=f"{XINFERENCE_URL.rstrip('/')}/v1/rerank",
        model_uid=RERANK_MODEL,
        top_n=25,
        request_timeout=240
    )
    
    # 细节检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=raw_store.as_retriever(search_kwargs={"k": 150})
    )

    # 细节生成 Prompt
    specific_answer_template = """你是一个精通《莲之空女学院》剧情的 AI 助手。
请仔细阅读以下检索到的【日文剧情片段】，并用**中文**回答用户的问题。

【剧情片段】：
{context}

【用户原问题】：
{original_question}

【回答要求】：
1. 基于日文片段的内容进行推理，但**必须用中文回答**。
2. 沉浸式回答：请直接以“剧情专家”的身份回答，**严禁**在回答中提及“片段ID”、“XML结构”或“根据提供的JSON数据”。
3. 准确引用：如果需要指出具体出处，请使用元数据中的 `scene` (场景ID) 或 时间/地点。
4. 如果检索结果中没有相关内容，请直接说“在当前检索到的剧情中未找到”，不要编造。
"""
    specific_answer_prompt = ChatPromptTemplate.from_template(specific_answer_template)

    # =========================================================
    # 3. 宏观摘要路径 (Overview Path)
    # =========================================================
    summary_answer_template = """你是一个高阶剧情分析师。
用户正在寻求对剧情的宏观概括。请基于以下检索到的【主题摘要片段】，用流畅的中文回答用户的问题。

【主题摘要片段】：
{context}

【用户原问题】：
{original_question}

【回答要求】：
1. 侧重于宏观叙事、主要角色关系和核心事件，不要陷入细节。
2. 严禁提及“片段”、“摘要”、“ID”等系统术语。
"""
    summary_answer_prompt = ChatPromptTemplate.from_template(summary_answer_template)


    # --- 交互循环 ---
    while True:
        print("\n" + "="*50)
        user_query = input("请提问 [q退出]: ")
        if user_query.lower() in ['q', 'exit']: break
        
        try:
            # 1. 意图分类
            print(f"正在分析意图...")
            intent = intent_chain.invoke({"query": user_query}).strip().lower()
            print(f"   分类结果: {intent}")
            
            # --- 宏观概括路径 ---
            if intent == 'overview' and summary_store:
                
                print("路由: 宏观摘要检索...")
                # 在摘要集合中进行搜索
                docs = summary_store.similarity_search(user_query, k=3)
                context_str = format_docs(docs, is_summary=True)
                
                print("正在生成宏观回答...")
                print("-" * 30)
                final_chain = summary_answer_prompt | llm | StrOutputParser()

            # --- 细节追问路径 (默认路径) ---
            else:
                if intent == 'overview' and not summary_store:
                    print("宏观摘要未就绪，强制转为细节检索。")
                
                print(f"正在提取关键词...")
                # 1. 执行重写
                jp_query = rewrite_chain.invoke({"question": user_query})
                print(f"\r生成搜索词: {jp_query}")
                
                # 2. 执行检索 (用日文搜)
                print(f"正在检索 (Hybrid + Rerank)...")
                docs = compression_retriever.invoke(jp_query)
                
                if not docs:
                    print("未找到相关剧情。")
                    continue
                    
                # 3. 格式化上下文
                context_str = format_docs(docs, is_summary=False)
                
                # 4. 生成回答 (用中文回)
                print(f"正在生成回答...")
                print("-" * 30)
                final_chain = specific_answer_prompt | llm | StrOutputParser()

            # 统一输出
            for chunk in final_chain.stream({
                "context": context_str,
                "original_question": user_query
            }):
                print(chunk, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"\n❌ 流程出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()