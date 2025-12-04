"""
===============================================================================
MAIN - AGI Chat API v2.1 (Render Deploy)
===============================================================================

Ponto de entrada principal para deploy no Render.
Importa e inicializa a aplicação FastAPI do módulo src.

Estrutura:
- main.py (este arquivo) → Raiz do projeto
- src/ → Módulos da aplicação
  ├── main.py → App FastAPI
  ├── chat_pipeline.py
  ├── llm_huggingface.py
  └── ...

Autor: João Manoel
Deploy: Render.com
===============================================================================
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ────────────────────────────────────────────────────────────────
# IMPORT FIX — garante que /src funciona no Render
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Agora os imports do seu código
from chat_pipeline import respond_stream_generator

# ────────────────────────────────────────────────────────────────
# FASTAPI APP
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Maintenance Assistant",
    version="2.0",
    description="WebSocket AI Assistant + Predictive Models"
)

# Liberar frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────
# MODELS (se futuramente você quiser colocar modelos aqui)
# ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

# ────────────────────────────────────────────────────────────────
# ROOT
# ────────────────────────────────────────────────────────────────
@app.get("/")
async def home():
    return {
        "status": "ok",
        "message": "AI Maintenance Assistant is running!",
        "websocket": "/ws-chat",
        "docs": "/docs"
    }

# ────────────────────────────────────────────────────────────────
# DEBUG
# ────────────────────────────────────────────────────────────────
@app.get("/debug")
async def debug():
    return {
        "current_dir": BASE_DIR,
        "src_dir": SRC_DIR,
        "files_in_root": os.listdir(BASE_DIR),
        "files_in_src": os.listdir(SRC_DIR) if os.path.exists(SRC_DIR) else [],
        "python": sys.version,
    }

# ────────────────────────────────────────────────────────────────
# WEBSOCKET CHAT (STREAMING)
# ────────────────────────────────────────────────────────────────
@app.websocket("/ws-chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    print("🔗 Cliente conectado ao WebSocket")

    try:
        while True:
            data = await ws.receive_json()
            user_message = data.get("message", "")
            user_id = data.get("user_id", "anonymous")

            print(f"📩 Mensagem recebida: {user_message}")

            # Gerar resposta via streaming
            async for token in respond_stream_generator(
                user_message=user_message,
                user_id=user_id,
                memory=None,      # se quiser adicionar memória futuramente
                models={}         # se quiser adicionar modelos futuramente
            ):
                await ws.send_json({"type": "token", "data": token})

            # Finalização do streaming
            await ws.send_json({"type": "end", "data": "done"})

    except WebSocketDisconnect:
        print("❌ Cliente desconectado")
    except Exception as e:
        print(f"⚠️ Erro no WebSocket: {e}")
        await ws.close()

# ────────────────────────────────────────────────────────────────
# RODAR COM Uvicorn (Render usa automaticamente)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )