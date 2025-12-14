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

def format_docs(docs):
    """
    智能格式化：混合处理 '摘要(Summary)' 和 '原始片段(Fragment)'
    """
    formatted = []
    for i, doc in enumerate(docs):
        meta = doc.metadata.copy()
        score = meta.pop("relevance_score", 0)
        
        # 判断是摘要还是原始片段 (通过 metadata 中的 level 字段)
        # build_hierarchy.py 生成的摘要带有 "level": "summary"
        is_summary = meta.get("level") == "summary"
        
        if is_summary:
            # --- 格式 A: 摘要 ---
            content = (
                f"<summary_context id='S-{i+1}' score='{score:.4f}'>\n"
                f"  <content>{doc.page_content}</content>\n"
                f"  <note>This is a high-level summary of a story arc.</note>\n"
                f"</summary_context>"
            )
        else:
            # --- 格式 B: 原始片段 ---
            # 尝试解析嵌套的 json 字符串
            for key, value in meta.items():
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        meta[key] = json.loads(value)
                    except:
                        pass
            
            meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
            content = (
                f"<evidence_fragment id='F-{i+1}' score='{score:.4f}'>\n"
                f"  <content>\n{doc.page_content}\n  </content>\n"
                f"  <metadata>\n{meta_json}\n  </metadata>\n"
                f"</evidence_fragment>"
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
    intent_template = """你是一个查询意图分析专家。请分析用户的关于《莲之空女学院》剧情的提问。

【用户问题】：{query}

【分类定义】：
1. **analysis (深度分析)**: 
   - 询问角色之间的**关系、感情、态度**（如“吟子怎么看花帆”、“两人的关系变化”）。
   - 询问**原因、动机、背景**（如“为什么要这么做”、“背后的含义”）。
   - 询问**性格、评价、成长**。
   - **绝大多数非纯事实检索的问题都应归为此类。**

2. **overview (宏观概括)**: 
   - 仅当用户明确要求“总结全文”、“概括某章大意”、“讲了什么故事”时。

3. **fact (事实追问)**: 
   - 询问极其具体的**时间、地点、次数、物品**（如“第几话哭了”、“吃的什么”、“ID是多少”）。
   - 寻找具体的某句台词出处。

请只输出其中一个标签：'analysis', 'overview', 或 'fact'。
"""
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
    # 3. 融合生成 Prompt (Context Fusion)
    # =========================================================
    fusion_answer_template = """你是一个精通《莲之空女学院》剧情的专家。
为了回答用户的问题，我们为你提供了两个层面的信息：
1. **<summary_context>**: 剧情的宏观摘要（提供背景、氛围和事件脉络）。
2. **<evidence_fragment>**: 具体的对话和场景细节（提供直接证据和台词）。

【用户问题】：
{original_question}

【上下文信息】：
{context}

【回答要求】：
1. **结构化回答**：先用宏观摘要定下基调，再引用具体片段的细节作为论据。
2. **深度分析**：结合角色的具体台词（fragment）和当时的情境（summary），分析角色的心理活动。
3. **引用规范**：如果引用了片段，请提及场景或时间；如果引用了摘要，可作为背景描述。
4. 必须用**中文**回答。
"""
    fusion_answer_prompt = ChatPromptTemplate.from_template(fusion_answer_template)

    # --- 交互循环 ---
    while True:
        print("\n" + "="*50)
        user_query = input("🙋 请提问 (中文) [q退出]: ")
        if user_query.lower() in ['q', 'exit']: break
        
        try:
            # 1. 意图分类
            print(f"🤖 正在分析意图...", end="", flush=True)
            intent = intent_chain.invoke({"query": user_query}).strip().lower()
            print(f"\r✅ 意图识别: 【{intent}】           ")
            
            combined_docs = []

            # =================================================
            # 🚀 策略 A: 深度分析 (Analysis) -> 双路检索融合
            # =================================================
            if 'analysis' in intent:
                print("🔄 启动双路检索 (Dual-Path Retrieval)...")
                
                # Path 1: 查摘要 (获取背景)
                if summary_store:
                    print("   └── 正在提取宏观背景 (Summary)...")
                    # 摘要不需要重写，直接用中文搜语义即可，取 Top 5
                    summary_docs = summary_store.similarity_search(user_query, k=5)
                    combined_docs.extend(summary_docs)
                
                # Path 2: 查细节 (获取证据)
                print("   └── 正在挖掘细节证据 (Details)...")
                jp_query = rewrite_chain.invoke({"question": user_query})
                # 细节检索需要重写为日文
                detail_docs = compression_retriever.invoke(jp_query)
                # 我们取前 15 个细节，避免冲淡摘要的权重
                combined_docs.extend(detail_docs[:15])

            # =================================================
            # 📖 策略 B: 宏观概括 (Overview) -> 只查摘要
            # =================================================
            elif 'overview' in intent and summary_store:
                print("🔍 检索宏观摘要...")
                combined_docs = summary_store.similarity_search(user_query, k=10)

            # =================================================
            # 🔍 策略 C: 事实追问 (Fact) -> 只查细节
            # =================================================
            else: # fact 或 fallback
                print("🔍 检索具体细节...")
                jp_query = rewrite_chain.invoke({"question": user_query})
                combined_docs = compression_retriever.invoke(jp_query)

            # --- 统一生成环节 ---
            if not combined_docs:
                print("⚠️ 未找到相关信息。")
                continue

            # 格式化所有文档（自动处理混合类型）
            context_str = format_docs(combined_docs)
            
            print(f"🤖 正在生成深度回答 (Context Size: {len(combined_docs)} chunks)...")
            print("-" * 30)
            
            # 使用融合 Prompt
            final_chain = fusion_answer_prompt | llm | StrOutputParser()
            
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