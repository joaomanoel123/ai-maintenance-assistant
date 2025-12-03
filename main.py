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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Import correto do chat_pipeline dentro de src/
from src.chat_pipeline import respond_stream_generator

app = FastAPI(title="AI Maintenance Assistant")

# CORS liberado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rota simples
@app.get("/")
def home():
    return {"status": "ok", "message": "API is running!", "version": "1.0"}


# ==========================
#  WEBSOCKET DO CHAT (frontend usa este!)
# ==========================

@app.websocket("/ws-chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            user_id = data.get("user_id", "anonymous")

            # streaming
            async for chunk in respond_stream_generator(
                user_message=user_message,
                user_id=user_id,
                memory=None,   # sem memória ainda
                models={}
            ):
                await websocket.send_json({
                    "type": "token",
                    "data": chunk
                })

            await websocket.send_json({
                "type": "end",
                "data": "done"
            })

    except WebSocketDisconnect:
        print("Cliente desconectado")



# Execução local
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )