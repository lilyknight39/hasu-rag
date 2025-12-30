#!/bin/bash
# OpenAI 兼容 API 服务器启动脚本

echo "🚀 启动 Hasu-RAG OpenAI 兼容 API 服务器..."
echo ""
echo "📡 服务器地址: http://localhost:8000"
echo "📖 API 文档: http://localhost:8000/docs"
echo "🔑 请确保 api_keys.yaml 已正确配置"
echo ""

cd "$(dirname "$0")"
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
