"""
OpenAI 兼容的 API 服务器

提供 /v1/chat/completions 端点，支持流式和非流式响应
集成 RAG 系统，并提供 API key 鉴权、并发控制和频率限制
"""

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union
import time
import uuid
import logging
import subprocess
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from auth import AuthManager
from rag_wrapper import get_rag_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# OpenAI 兼容数据模型
# ============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="hasu-rag", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    stream: Optional[bool] = Field(default=False, description="是否流式响应")
    # 以下参数仅为兼容性存在，将被忽略
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "hasu-rag"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# ============================================================================
# FastAPI 应用
# ============================================================================

# 使用 lifespan 管理应用生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的钩子"""
    logger.info("🚀 启动 OpenAI 兼容 API 服务器...")
    logger.info("📦 预加载 RAG 系统...")
    
    # 预加载 RAG 系统（避免第一次请求时初始化太慢）
    try:
        get_rag_system()
        logger.info("✅ RAG 系统加载成功")
    except Exception as e:
        logger.error(f"❌ RAG 系统加载失败: {e}")

    mcp_process = None
    if os.getenv("ONEBOT_MCP_AUTOSTART", "0") == "1":
        script_path = (Path(__file__).parent / "mcp_onebot_server.py").resolve()
        if script_path.exists():
            try:
                mcp_process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    env=os.environ.copy()
                )
                logger.info(f"✅ MCP OneBot server started (pid={mcp_process.pid})")
            except Exception as e:
                logger.warning(f"⚠️ MCP OneBot server failed to start: {e}")
        else:
            logger.warning(f"⚠️ MCP OneBot server script not found: {script_path}")
    
    yield

    if mcp_process:
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mcp_process.kill()
    
    logger.info("🛑 关闭 API 服务器...")

app = FastAPI(
    title="Hasu-RAG OpenAI Compatible API",
    description="OpenAI 兼容的 RAG API 服务器，支持流式和非流式响应",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"🕒 {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"🕒 Result: {response.status_code}")
    return response

import os
import re

# 初始化认证管理器 (auth.py 内部已默认处理相对路径)
auth_manager = AuthManager()

# ============================================================================
# 辅助函数
# ============================================================================

def clean_query(text: str) -> str:
    """
    剔除机器人框架产生的多余信息（如 QQ 昵称、AT 标签等）
    """
    # 1. 剔除 <at ... /> 标签
    text = re.sub(r'<at\b[^>]*\/?>', '', text)
    
    # 2. 剔除开头的昵称前缀（支持中文和英文冒号），例如 "昵称：" 或 "昵称: "
    # 注意：只匹配开头的一段，避免误删正文内容
    text = re.sub(r'^[^：:\n]+[：:]\s*', '', text)
    
    return text.strip()

def extract_user_message(messages: List[ChatMessage]) -> str:
    """从消息列表中提取用户问题"""
    # 获取最后一条用户消息
    for msg in reversed(messages):
        if msg.role == "user":
            return clean_query(msg.content)
    
    raise HTTPException(
        status_code=400,
        detail="No user message found in the conversation"
    )

def create_sse_chunk(content: str, finish_reason: Optional[str] = None) -> str:
    """创建 SSE (Server-Sent Events) 格式的数据块"""
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "hasu-rag",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason
        }]
    }
    
    import json
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

def sync_stream_generator(question: str, api_key: str):
    """同步流式生成器 (在线程池中运行)"""
    try:
        rag = get_rag_system()
        
        # 使用 RAG 系统流式生成答案
        for chunk in rag.query(question):
            yield create_sse_chunk(chunk)
        
        # 发送结束标记
        yield create_sse_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"❌ 流式生成错误: {e}")
        error_chunk = {
            "error": {
                "message": f"生成过程中发生错误: {str(e)}",
                "type": "internal_error"
            }
        }
        import json
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
    
    finally:
        # 释放请求槽位
        auth_manager.release_request_slot(api_key)

def fake_stream_generator(message: str):
    """模拟流式生成器，用于返回友好的错误提示"""
    yield create_sse_chunk(message)
    yield create_sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"

# ============================================================================
# API 端点
# ============================================================================

@app.get("/")
def root():
    """根端点"""
    return {
        "message": "Hasu-RAG OpenAI Compatible API",
        "version": "1.0.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": int(time.time())}

@app.get("/v1/models")
def list_models():
    """
    列出可用模型 (OpenAI 兼容)
    """
    logger.info("🔍 Received request for /v1/models")
    return ModelList(data=[ModelCard(id="hasu-rag")])

@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    OpenAI 兼容的聊天完成端点
    
    支持流式和非流式响应
    """
    # 1. 验证并获取请求槽位（包括 API key 验证、并发检查、频率限制）
    try:
        api_key, config = auth_manager.acquire_request_slot(authorization)
    except HTTPException as e:
        # 将 429 错误伪装成友好的机器人回复
        if e.status_code == 429 and isinstance(e.detail, dict):
            error_info = e.detail.get("error", {})
            raw_msg = error_info.get("message", "系统繁忙")
            friendly_msg = f"【系统提醒】{raw_msg}"
            
            if request.stream:
                return StreamingResponse(
                    fake_stream_generator(friendly_msg),
                    media_type="text/event-stream"
                )
            else:
                return ChatCompletionResponse(
                    id=f"chatcmpl-system-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=friendly_msg),
                            finish_reason="stop"
                        )
                    ],
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
        raise
    
    # 2. 提取用户问题
    try:
        # 打印原始请求信息（调试用）
        logger.info(f"📥 Received Chat Request: model={request.model}, stream={request.stream}")
        question = extract_user_message(request.messages)
        logger.info(f"📝 最终提取问题 ({config.name}): {question}")
    except Exception as e:
        auth_manager.release_request_slot(api_key)
        raise HTTPException(status_code=400, detail=str(e))
    
    # 3. 根据 stream 参数选择响应模式
    if request.stream:
        # 流式响应
        logger.info("🌊 使用流式响应")
        return StreamingResponse(
            sync_stream_generator(question, api_key),
            media_type="text/event-stream"
        )
    else:
        # 非流式响应
        logger.info("📄 使用非流式响应")
        try:
            rag = get_rag_system()
            
            # 收集所有生成的内容
            full_response = ""
            for chunk in rag.query(question):
                full_response += chunk
            
            # 构造 OpenAI 格式的响应
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=full_response),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": len(question),
                    "completion_tokens": len(full_response),
                    "total_tokens": len(question) + len(full_response)
                }
            )
            
            return response
        
        except Exception as e:
            logger.error(f"❌ 生成答案时发生错误: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"生成答案时发生错误: {str(e)}"
            )
        
        finally:
            # 释放请求槽位
            auth_manager.release_request_slot(api_key)

# ============================================================================
# 错误处理
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """自定义 HTTP 异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "invalid_request_error"}}
        if not isinstance(exc.detail, dict)
        else exc.detail
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"❌ 未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error"
            }
        }
    )

# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
