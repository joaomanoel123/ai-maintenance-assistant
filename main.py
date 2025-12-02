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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# importa o pipeline do chat dentro da pasta src
from src.chat_pipeline import respond_stream_generator

app = FastAPI(title="AI Maintenance Assistant")

# liberar chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "API is running!"}


@app.post("/chat")
async def chat(payload: dict):
    user_message = payload.get("message", "")
    user_id = payload.get("user_id", "")

    response_text = ""

    async for chunk in respond_stream_generator(
        user_message=user_message,
        user_id=user_id,
        memory=None,
        models={}
    ):
        response_text += chunk

    return {"response": response_text}
