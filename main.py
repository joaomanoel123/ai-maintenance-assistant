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
# =============================================================================
# ROTAS REQUERIDAS PELO FRONT-END
# =============================================================================

# ✔ HEALTH CHECK (frontend usa a cada refresh)
@app.get("/health")
async def health():
    return {"status": "healthy"}


# ✔ STATS (usado no dashboard e na home)
@app.get("/stats")
async def stats():
    return {
        "predictions": {"total": 156},
        "feedback": {"total_received": 89, "average_score": 4.3},
        "memory": {"short_term_size": 45, "long_term_size": 1203},
        "system": {"models_loaded": 2}
    }


# ✔ PREDICT (usado quando o usuário clica em “Executar Análise”)
@app.post("/predict")
async def predict(payload: dict):
    """
    IMPORTANTE:
    Substituir por IA real no futuro.
    Aqui está só um modelo mock, compatível com o frontend.
    """
    return {
        "result_id": "res_" + str(int(datetime.now().timestamp())),
        "prediction": {
            "type": "failure",       # ou "rul"
            "value": 0.23,           # probabilidade ou ciclos
            "confidence": 0.87
        },
        "reasoning": {
            "severity": "MEDIUM"
        }
    }


# ✔ CHAT HTTP (fallback, não streaming)
@app.post("/chat")
async def chat_http(payload: dict):
    user_message = payload.get("message", "")
    user_id = payload.get("user_id", "anonymous")

    final_text = ""

    # usa seu pipeline real (stream → acumula tokens)
    async for token in respond_stream_generator(
        user_message=user_message,
        user_id=user_id,
        memory=None,
        models={}
    ):
        final_text += token

    return {"response": final_text}


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
