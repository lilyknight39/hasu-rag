import os
import json
import warnings
import re
import sys
import asyncio
import threading
from pathlib import Path
from typing import Optional, Tuple

# 屏蔽烦人的警告
warnings.filterwarnings("ignore")

# --- 核心组件 ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# 3. 生成后端 (LLM)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")

# ==============================================================================
# 🔧 OneBot MCP (optional)
# ==============================================================================

ONEBOT_MCP_ENABLED = os.getenv("ONEBOT_MCP_ENABLED", "0") == "1"
ONEBOT_DEFAULT_TARGET = os.getenv("ONEBOT_DEFAULT_TARGET", "").strip()
ONEBOT_DEFAULT_TARGET_TYPE = os.getenv("ONEBOT_DEFAULT_TARGET_TYPE", "").strip().lower()
ONEBOT_DEFAULT_TARGET_ID = os.getenv("ONEBOT_DEFAULT_TARGET_ID", "").strip()
ONEBOT_MCP_SERVER_PATH = os.getenv("ONEBOT_MCP_SERVER_PATH", "").strip()
ONEBOT_SEND_MODE = os.getenv("ONEBOT_SEND_MODE", "auto").strip().lower()
ONEBOT_MULTI_SEND = os.getenv("ONEBOT_MULTI_SEND", "0") == "1"
ONEBOT_SEND_INTERVAL_SECONDS = float(os.getenv("ONEBOT_SEND_INTERVAL_SECONDS", "1.0"))

_ONEBOT_TRIGGER_RE = re.compile(r"(发送|播放|发到|发送到|发给|发送给)")
_ONEBOT_FILE_MODE_RE = re.compile(r"(文件|file|上传)", re.IGNORECASE)
_ONEBOT_VOICE_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vo_adv_[A-Za-z0-9_@-]+(?:\.mp3)?)(?![A-Za-z0-9_])",
    re.IGNORECASE
)
_ONEBOT_FILE_EXT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vo_adv_[A-Za-z0-9_@-]+\.mp3)(?![A-Za-z0-9_])",
    re.IGNORECASE
)
_ONEBOT_TARGET_RE = re.compile(r"(?:发到|发送到|发给|发送给)\s*(群|群聊|群里|私聊|私信|好友)?\s*([0-9]{5,})")

# ==============================================================================
# 📝 Prompt Templates (供 API 和交互模式共用)
# ==============================================================================

INTENT_TEMPLATE = """你是一个查询意图分析专家。请分析用户的关于《莲之空女学院》剧情的提问。

【用户问题】：{query}

【分类定义】：
1. **analysis (深度分析)**: 
   - 询问角色之间的**关系、感情、态度**（如"吟子怎么看花帆"、"两人的关系变化"）。
   - 询问**原因、动机、背景**（如"为什么要这么做"、"背后的含义"）。
   - 询问**性格、评价、成长**。
   - **绝大多数非纯事实检索的问题都应归为此类。**

2. **overview (宏观概括)**: 
   - 仅当用户明确要求"总结全文"、"概括某章大意"、"讲了什么故事"时。

3. **fact (事实追问)**: 
   - 询问极其具体的**时间、地点、次数、物品**（如"第几话哭了"、"吃的什么"、"ID是多少"）。
   - 寻找具体的某句台词出处。

请只输出其中一个标签：'analysis', 'overview', 或 'fact'。
若用户在问题中明确指定 analysis、overview、fact，优先使用该标签。
"""

DENSE_REWRITE_TEMPLATE = """你是为 **BGE-M3 语义检索** 服务的查询优化器。
将用户问题改写为一条自然、完整的日文问句，保持主谓宾和语境，不要拆成关键词。
纠正常见角色名，使用下方映射表的全名；如用户已有正确日文名，保持不变。
若用户只给出短语/关键词，请补全成通顺的问句，但不要添加无关信息。
【角色名映射表】梢/梢前辈->乙宗梢；花帆/花帆桑->日野下花帆；慈/慈前辈->藤島慈；瑠璃乃->大沢瑠璃乃；吟子/百生/小吟子->百生吟子；小铃/徒町->徒町小鈴；姬芽/安養寺->安養寺姫芽；塞拉斯->セラス。
【输出示例】输入: 梢哭了几次？ 输出: 乙宗梢は何回泣きましたか？
只输出改写后的日文问句，不要解释，不要追加其他字段。

用户问题: {question}
"""

SPARSE_KEYWORD_TEMPLATE = """你是为 **BM25 关键词检索** 生成查询的专家。
输出一串日文/假名关键词，偏重人名、地名、道具名、曲名、稀有词，使用空格分隔。
原则：
- 角色名用全名；目标名词用具体表记（含重要词形变体，如 动词/表情/别称）。
- 计数类词最多保留 1 个（如「何回」或「回数」），不要堆叠。
- 避免泛泛的「シーン/登場/場面」等低 IDF 词，除非用户要求。
- 若用户明确禁止某词表记，遵从用户指示。
要求：
- 纠正常见角色名，使用下方映射表的全名。
- 可以加入同义词/形态变化，但保持关键词形式，避免完整句子。
- 若用户问题包含中文或编号，请保留。
【角色名映射表】梢/梢前辈->乙宗梢；花帆/花帆桑->日野下花帆；慈/慈前辈->藤島慈；瑠璃乃->大沢瑠璃乃；吟子/百生/小吟子->百生吟子；小铃/徒町->徒町小鈴；姬芽/安養寺->安養寺姫芽；塞拉斯->セラス。
【输出示例】输入: 梢哭了几次？ 输出: 乙宗梢 泣く 涙 回数
格式：仅输出关键词串，使用空格分隔，不要添加解释或其他内容。

用户问题: {question}
"""

ALPHA_TEMPLATE = """你是混合检索参数顾问，需要为 Dense+BM25 混合检索输出一个 alpha 值 (0.15~0.65)：
- alpha 小 -> 更依赖 BM25；alpha 大 -> 更依赖语义。
- 若问题含明确编号/ID/数字或短关键词，偏 BM25 (0.2~0.35)；
- 若问题是长句、关系/因果分析，偏语义 (0.4~0.55)。
【输出示例】示例1: 0.3  示例2: 0.45
请结合原始中文和两种重写，输出一个数字（例如 0.3 或 0.45），禁止输出其他字符。

【原始问题】：{original}
【语义重写】：{dense}
【关键词重写】：{sparse}
"""

ANSWER_TEMPLATE = """你是一个精通《莲之空女学院》剧情的专家级 AI 剧情分析师。
为了回答用户的问题，我们为你提供了检索到的剧情信息，其中可能包含：
1. **<summary_section>**: 剧情的宏观摘要（概括背景、大纲）。
2. **<story_fragment>**: 具体的对话、独白和场景细节（精确证据）。

【剧情片段】：
{context}

【用户问题】：
{original_question}

【回答要求】：
1. **深度融合与证据展示 (Evidence-Based)**：
   - 请综合 **宏观背景** (Summary) 与 **微观细节** (Fragment) 进行作答。
   - 既然用户看不到原文，你需**复述**关键台词、动作描写或心理活动作为论据，而不仅仅是给出一个结论。
   - 优先引用具体的对话和动作细节，摘要仅作为背景补充。

2. **自然的隐形引用 (Natural Citation)**：
   - **绝对禁止**使用 "F-X", "S-X", "片段1", "ID:xxx" 这种机械索引。
   - ✅ **正确示范**：
     - "在练习室的冲突场景中 (story_main_10500701_scene_005)，花帆哭着说..."
     - "正如之前提到的姬芽失去挚友的经历..."
     - "当两人在钓鱼场独处时，姬芽提到..."
   - ❌ **错误示范**：
     - "根据 F-1，她们吵架了。"
     - "参考摘要 section 2..."
   - *注：若提及片段，**可以**保留具体的 Scene ID (如 story_main_... ) 以便溯源，但要嵌入在自然语句中。*

3. **结构化回答**：
   - 逻辑清晰，观点明确。
   - 每一条论点都应当有具体的剧情细节（台词/动作）支撑。

4. **兜底策略**：
   - 如果检索到的信息中没有任何与问题相关的内容，请直接回答："在当前检索到的剧情中未找到相关信息。"，不要编造。

5. **语言要求**：必须用**中文**回答。
"""

# ==============================================================================

def _get_mcp_server_path() -> Path:
    if ONEBOT_MCP_SERVER_PATH:
        return Path(ONEBOT_MCP_SERVER_PATH).expanduser().resolve()
    return (Path(__file__).parent / "mcp_onebot_server.py").resolve()


def _parse_target_spec(raw: str) -> Optional[Tuple[str, int]]:
    raw = raw.strip()
    if not raw:
        return None
    if ":" in raw:
        prefix, id_str = raw.split(":", 1)
        prefix = prefix.strip().lower()
        id_str = id_str.strip()
    else:
        prefix = ""
        id_str = raw

    try:
        target_id = int(id_str)
    except ValueError:
        return None

    if prefix in ("group", "g", "grp"):
        target_type = "group"
    elif prefix in ("private", "p", "user", "u"):
        target_type = "private"
    elif prefix in ("onebot", "qq"):
        target_type = "group"
    elif prefix == "":
        target_type = "group"
    else:
        return None

    return target_type, target_id


def _get_default_target() -> Optional[Tuple[str, int]]:
    if ONEBOT_DEFAULT_TARGET:
        return _parse_target_spec(ONEBOT_DEFAULT_TARGET)
    if ONEBOT_DEFAULT_TARGET_TYPE and ONEBOT_DEFAULT_TARGET_ID:
        try:
            target_id = int(ONEBOT_DEFAULT_TARGET_ID)
        except ValueError:
            return None
        if ONEBOT_DEFAULT_TARGET_TYPE in ("group", "private"):
            return ONEBOT_DEFAULT_TARGET_TYPE, target_id
    return None


def _extract_target_from_query(query: str) -> Optional[Tuple[str, int]]:
    match = _ONEBOT_TARGET_RE.search(query)
    if not match:
        return None
    hint = (match.group(1) or "").strip()
    target_id = int(match.group(2))
    if hint in ("私聊", "私信", "好友"):
        return "private", target_id
    return "group", target_id


def _should_trigger_onebot(query: str) -> bool:
    return bool(_ONEBOT_TRIGGER_RE.search(query))


def _select_send_mode(query: str) -> str:
    if ONEBOT_SEND_MODE in ("voice", "file"):
        return ONEBOT_SEND_MODE
    return "file" if _ONEBOT_FILE_MODE_RE.search(query) else "voice"


def _dedupe_preserve(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _extract_all_voice_names_from_text(text: str) -> list[str]:
    if not text:
        return []
    matches = [m.group(1) for m in _ONEBOT_VOICE_NAME_RE.finditer(text)]
    if not matches:
        matches = [m.group(1) for m in _ONEBOT_FILE_EXT_RE.finditer(text)]
    return _dedupe_preserve(matches)


def _extract_voice_name_from_text(text: str) -> Optional[str]:
    names = _extract_all_voice_names_from_text(text)
    return names[0] if names else None


def _extract_voice_names_from_docs(docs) -> list[str]:
    results = []
    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        voices = meta.get("voices") or meta.get("voice")
        ctx = meta.get("ctx")

        if not voices and ctx:
            if isinstance(ctx, str) and (ctx.startswith("{") or ctx.startswith("[")):
                try:
                    ctx = json.loads(ctx)
                except Exception:
                    ctx = None
            if isinstance(ctx, dict):
                voices = ctx.get("voices") or ctx.get("voice")

        if isinstance(voices, str):
            parsed = _extract_all_voice_names_from_text(voices)
            if parsed:
                results.extend(parsed)
                continue
            try:
                voices = json.loads(voices)
            except Exception:
                voices = [voices]

        if isinstance(voices, list):
            for voice in voices:
                if isinstance(voice, str):
                    parsed = _extract_all_voice_names_from_text(voice)
                    if parsed:
                        results.extend(parsed)

        content = getattr(doc, "page_content", "")
        parsed = _extract_all_voice_names_from_text(content)
        if parsed:
            results.extend(parsed)

    return _dedupe_preserve(results)


def _onebot_log(message: str) -> None:
    print(f"[OneBot] {message}")


def _snippet(text: Optional[str], limit: int = 120) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) > limit:
        return f"{cleaned[:limit - 3]}..."
    return cleaned


async def _call_onebot_mcp_tool(
    file_names: list[str],
    target_type: str,
    target_id: int,
    mode: str,
    interval_seconds: float
) -> None:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception:
        try:
            from mcp.client import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except Exception as exc:
            _onebot_log(f"MCP client not available: {exc}")
            return

    server_path = _get_mcp_server_path()
    if not server_path.exists():
        _onebot_log(f"MCP server script not found: {server_path}")
        return

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tool_name = "send_file" if mode == "file" else "send_voice"
            total = len(file_names)
            for idx, file_name in enumerate(file_names, start=1):
                payload = {
                    "file_name": file_name,
                    "target_type": target_type,
                    "target_id": target_id
                }
                _onebot_log(
                    f"calling MCP tool={tool_name} target={target_type}:{target_id} "
                    f"file={file_name} ({idx}/{total})"
                )
                await session.call_tool(tool_name, payload)
                if idx < total and interval_seconds > 0:
                    await asyncio.sleep(interval_seconds)
            _onebot_log("MCP tool call finished")


def _call_onebot_mcp_tool_safe(
    file_names: list[str],
    target_type: str,
    target_id: int,
    mode: str,
    interval_seconds: float
) -> None:
    try:
        asyncio.run(_call_onebot_mcp_tool(file_names, target_type, target_id, mode, interval_seconds))
    except Exception as exc:
        _onebot_log(f"MCP call failed: {exc}")


def _maybe_trigger_onebot_tool(user_query: str, answer_text: str, docs=None) -> None:
    trigger_match = _ONEBOT_TRIGGER_RE.search(user_query)
    if not trigger_match:
        return

    if not ONEBOT_MCP_ENABLED:
        _onebot_log(f"trigger found '{trigger_match.group(0)}' but MCP disabled; skip")
        return

    _onebot_log(f"trigger matched: '{trigger_match.group(0)}'")

    target_from_query = _extract_target_from_query(user_query)
    target = target_from_query or _get_default_target()
    if not target:
        _onebot_log("no target configured; set ONEBOT_DEFAULT_TARGET or ONEBOT_DEFAULT_TARGET_TYPE/ID")
        return

    target_source = "query" if target_from_query else "default"
    target_type, target_id = target
    _onebot_log(f"target: {target_type}:{target_id} (source={target_source})")

    file_names = _extract_all_voice_names_from_text(answer_text)
    file_source = "answer"
    if not file_names:
        file_names = _extract_all_voice_names_from_text(user_query)
        file_source = "query"
    if not file_names:
        file_names = _extract_voice_names_from_docs(docs)
        file_source = "docs"
    if not file_names:
        _onebot_log(
            "voice file name not found; "
            f"answer_snippet='{_snippet(answer_text)}'; "
            f"query_snippet='{_snippet(user_query)}'; "
            f"docs={len(docs) if docs else 0}"
        )
        return

    mode = _select_send_mode(user_query)
    if ONEBOT_SEND_MODE in ("voice", "file"):
        _onebot_log(f"send mode forced by env: {mode}")
    if not ONEBOT_MULTI_SEND and len(file_names) > 1:
        _onebot_log(f"multi send disabled; {len(file_names)} matches, using first only")
        file_names = [file_names[0]]
    else:
        _onebot_log(f"multi send enabled; sending {len(file_names)} file(s)")
    _onebot_log(f"voice file(s): {', '.join(file_names)} (source={file_source})")
    _onebot_log(f"send mode: {mode}")
    _onebot_log(f"dispatching MCP tool call (interval={ONEBOT_SEND_INTERVAL_SECONDS}s)")
    thread = threading.Thread(
        target=_call_onebot_mcp_tool_safe,
        args=(file_names, target_type, target_id, mode, ONEBOT_SEND_INTERVAL_SECONDS),
        daemon=True
    )
    thread.start()


def format_docs(docs):
    """
    智能格式化：混合处理 '摘要(Summary)' 和 '原始片段(Fragment)'
    [优化]: 移除对 LLM 诱导性强的 F-x ID，改用语义化标签
    """
    formatted = []
    for i, doc in enumerate(docs):
        meta = doc.metadata.copy()
        score = meta.pop("relevance_score", 0)
        
        # 判断是摘要还是原始片段
        is_summary = meta.get("level") == "summary"
        
        if is_summary:
            # --- 格式 A: 摘要 ---
            content = (
                f"<summary_section index='{i+1}'>\n" # 移除 id='S-x'，改用 index
                f"  <content>{doc.page_content}</content>\n"
                f"</summary_section>"
            )
        else:
            # --- 格式 B: 原始片段 ---
            # 1. 解析 Metadata
            for key, value in meta.items():
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try: meta[key] = json.loads(value)
                    except: pass
            
            # 2. 提取更可读的场景信息，替代冷冰冰的 ID
            # 尝试获取场景名、时间或地点，组合成一个 readable_source
            scene_id = meta.get('scene', 'Unknown_Scene')
            location = meta.get('loc', '') or meta.get('location', '')
            
            # 构造一个给 LLM 看的“来源标签”，例如：[场景: story_main_... | 地点: 练习室]
            # 这样 LLM 就算引用，也会引用成 "在练习室的场景中..."
            source_tag = f"Scene: {scene_id}"
            if location:
                source_tag += f", Location: {location}"

            meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
            
            content = (
                f"<story_fragment sequence='{i+1}'>\n" # 移除 id='F-x'
                f"  <source_info>{source_tag}</source_info>\n" # 显式告诉 LLM 这是什么场景
                f"  <content>\n{doc.page_content}\n  </content>\n"
                f"  <metadata>\n{meta_json}\n  </metadata>\n"
                f"</story_fragment>"
            )
            
        formatted.append(content)
        
    return "\n\n".join(formatted)


def _dedupe_documents(docs):
    """
    通过文档顺序/point id 去重，避免多路检索后重复的上下文干扰 rerank。
    """
    seen = set()
    unique = []
    for doc in docs:
        meta = doc.metadata or {}
        key = None
        if meta.get("order") is not None:
            key = f"order:{meta.get('order')}"
        elif meta.get("id"):
            key = f"id:{meta.get('id')}"
        elif meta.get("scene"):
            key = f"scene:{meta.get('scene')}"
        else:
            key = doc.page_content[:120]
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _rerank_with_fallback(docs, query, reranker, limit):
    if not docs:
        return []
    try:
        reranked = reranker.compress_documents(docs, query=query)
        if reranked:
            return reranked[:limit]
    except Exception:
        pass
    return docs[:limit]


def _retrieve_detail_docs(raw_store, reranker, dense_query, sparse_query, raw_query, k_dense=180, k_sparse=120, limit=20, alpha=0.35):
    """
    双通道细节检索：
    - dense_query: 适配语义通道的日文完整问句。
    - sparse_query: 关键词串，偏向 BM25。
    - raw_query: 用户原文，保留编号/未修正的关键词兜底。
    """
    doc_pool = []
    dense_k = max(20, int(k_dense * (0.6 + alpha)))   # alpha 越大，dense 越多
    sparse_k = max(20, int(k_sparse * (1.4 - alpha))) # alpha 越小，sparse 越多
    print(f"🔧 [Internal] 检索参数: dense_k={dense_k}, sparse_k={sparse_k}, alpha={alpha}")

    def _run(retriever, query):
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        return []

    # 语义优先的主检索
    semantic_retriever = raw_store.as_retriever(search_kwargs={"k": dense_k})
    doc_pool.extend(_run(semantic_retriever, dense_query))

    # 保留关键词的回落检索
    lexical_retriever = raw_store.as_retriever(search_kwargs={"k": sparse_k})
    doc_pool.extend(_run(lexical_retriever, sparse_query or raw_query))

    # 用户原文再兜底一次，兼顾原始语言/编号
    if raw_query and raw_query != sparse_query:
        doc_pool.extend(_run(lexical_retriever, raw_query))

    doc_pool = _dedupe_documents(doc_pool)
    return _rerank_with_fallback(doc_pool, dense_query or sparse_query or raw_query, reranker, limit)


def _parse_alpha(alpha_str: str, default: float = 0.35) -> float:
    try:
        val = float(alpha_str.strip())
        return max(0.15, min(0.65, val))
    except Exception:
        return default

# ==============================================================================
# 🌐 API 接口 (供 api_server.py 调用)
# ==============================================================================

_rag_components = None  # 全局缓存

def get_rag_components():
    """获取/初始化 RAG 组件（单例）"""
    global _rag_components
    if _rag_components:
        return _rag_components
    
    print("🔧 初始化 RAG 组件...")
    
    client = QdrantClient(url=QDRANT_URL)
    dense_emb = XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL)
    sparse_emb = FastEmbedSparse(model_name="Qdrant/bm25")
    
    raw_store = QdrantVectorStore(
        client=client, collection_name=RAW_COLLECTION_NAME,
        embedding=dense_emb, sparse_embedding=sparse_emb,
        sparse_vector_name=SPARSE_VECTOR_NAME, retrieval_mode=RetrievalMode.HYBRID
    )
    
    summary_store = None
    if client.collection_exists(SUMMARY_COLLECTION_NAME):
        summary_store = QdrantVectorStore(
            client=client, collection_name=SUMMARY_COLLECTION_NAME,
            embedding=dense_emb, retrieval_mode=RetrievalMode.DENSE
        )
    
    llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL_NAME,
                     temperature=0.7, streaming=False, max_tokens=20480)
    rewrite_llm = ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL_NAME,
                             temperature=0.0, streaming=False)
    
    reranker = XinferenceRerank(url=f"{XINFERENCE_URL.rstrip('/')}/v1/rerank",
                                model_uid=RERANK_MODEL, top_n=20, request_timeout=240)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=raw_store.as_retriever(search_kwargs={"k": 150})
    )

    intent_chain = ChatPromptTemplate.from_template(INTENT_TEMPLATE) | rewrite_llm | StrOutputParser()
    dense_rewrite_chain = ChatPromptTemplate.from_template(DENSE_REWRITE_TEMPLATE) | rewrite_llm | StrOutputParser()
    sparse_rewrite_chain = ChatPromptTemplate.from_template(SPARSE_KEYWORD_TEMPLATE) | rewrite_llm | StrOutputParser()
    alpha_chain = ChatPromptTemplate.from_template(ALPHA_TEMPLATE) | rewrite_llm | StrOutputParser()
    answer_chain = ChatPromptTemplate.from_template(ANSWER_TEMPLATE) | llm | StrOutputParser()
    
    _rag_components = {
        'summary_store': summary_store,
        'compression_retriever': compression_retriever,
        'raw_store': raw_store,
        'reranker': reranker,
        'rewrite_llm': rewrite_llm,
        'llm': llm,
        'intent_chain': intent_chain,
        'dense_rewrite_chain': dense_rewrite_chain,
        'sparse_rewrite_chain': sparse_rewrite_chain,
        'alpha_chain': alpha_chain,
        'answer_chain': answer_chain
    }
    print("✅ RAG 组件初始化完成")
    return _rag_components


def process_single_query(user_query: str):
    """
    处理单个查询（供 API 调用，流式返回）
    
    使用模块级常量 INTENT_TEMPLATE, DENSE_REWRITE_TEMPLATE, SPARSE_KEYWORD_TEMPLATE, ANSWER_TEMPLATE
    """
    c = get_rag_components()
    
    # 执行查询流程
    print(f"\n🔍 [Internal] 开始处理查询: {user_query}")
    intent = c['intent_chain'].invoke({"query": user_query}).strip().lower()
    print(f"💡 [Internal] 识别意图: {intent}")
    
    combined_docs = []
    dense_query = c['dense_rewrite_chain'].invoke({"question": user_query}).strip()
    sparse_query = c['sparse_rewrite_chain'].invoke({"question": user_query}).strip()
    alpha_raw = c['alpha_chain'].invoke({
        "original": user_query,
        "dense": dense_query,
        "sparse": sparse_query
    }).strip()
    alpha = _parse_alpha(alpha_raw, default=0.35)
    print(f"🔄 [Internal] 语义重写 (JP): {dense_query}")
    print(f"🧩 [Internal] 关键词重写 (BM25): {sparse_query}")
    print(f"⚖️ [Internal] Alpha 建议: {alpha_raw} -> 采用 {alpha}")
    
    if 'analysis' in intent:
        if c['summary_store']:
            print("📅 [Internal] 正在检索宏观背景 (Summary)...")
            combined_docs.extend(c['summary_store'].similarity_search(user_query, k=5))
        
        print("🧪 [Internal] 双通道检索细节 (JP rewrite + 原文关键词)...")
        detail_docs = _retrieve_detail_docs(
            raw_store=c['raw_store'],
            reranker=c['reranker'],
            dense_query=dense_query,
            sparse_query=sparse_query,
            raw_query=user_query,
            k_dense=200,
            k_sparse=140,
            limit=18,
            alpha=alpha
        )
        combined_docs.extend(detail_docs)
        
    elif 'overview' in intent and c['summary_store']:
        print("📖 [Internal] 正在检索宏观摘要...")
        combined_docs = c['summary_store'].similarity_search(user_query, k=10)
        
    else:
        print("🕵️ [Internal] 双通道检索事实 (JP rewrite + 原文关键词)...")
        combined_docs = _retrieve_detail_docs(
            raw_store=c['raw_store'],
            reranker=c['reranker'],
            dense_query=dense_query,
            sparse_query=sparse_query,
            raw_query=user_query,
            k_dense=220,
            k_sparse=160,
            limit=20,
            alpha=alpha
        )
    
    print(f"📚 [Internal] 检索完成，共获取 {len(combined_docs)} 个上下文片段")
    
    if not combined_docs:
        yield "未找到相关信息。"
        return
    
    context_str = format_docs(combined_docs)
    answer_chunks = []
    for chunk in c['answer_chain'].stream({"context": context_str, "original_question": user_query}):
        answer_chunks.append(chunk)
        yield chunk
    full_answer = "".join(answer_chunks)
    _maybe_trigger_onebot_tool(user_query, full_answer, combined_docs)


def main():
    print(f"\n启动分层智能问答系统 (Hierarchical RAG)...")
    
    # 触发组件初始化，复用 API 同款管线
    get_rag_components()

    # --- 交互循环 ---
    while True:
        print("\n" + "="*50)
        user_query = input("🙋 请提问 (中文) [q退出]: ")
        if user_query.lower() in ['q', 'exit']: break
        
        try:
            for chunk in process_single_query(user_query):
                print(chunk, end="", flush=True)
            print()
            
        except Exception as e:
            print(f"\n❌ 流程出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
