"""
===============================================================================
MAIN CHAT API - Pipeline AGI v2.0 com Memória Vetorial
===============================================================================

Backend completo com:
- WebSocket para chat streaming
- Memória vetorial (ChromaDB)
- Modelos preditivos integrados
- Ingestão de datasets Kaggle
- Persistência no Neon PostgreSQL

Autor: João Manoel
Deploy: Render.com
===============================================================================
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Imports dos módulos customizados
from model_loader import load_models, MODELS, model_predict
from embeddings import VectorMemory
from chat_pipeline import respond_stream_generator, extract_prediction_intent
from db import save_experience_record, init_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# INICIALIZAR APP
# ============================================

app = FastAPI(
    title="AGI Chat + Predictive API",
    version="2.1",
    description="Sistema AGI com chat responsivo e memória vetorial"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memória vetorial global
memory: Optional[VectorMemory] = None

# ============================================
# MODELS
# ============================================

class PredictRequest(BaseModel):
    type: str  # 'cmapss' ou 'ai4i'
    features: list

class ChatMessage(BaseModel):
    message: str
    user_id: str = "anonymous"

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inicializar modelos e memória vetorial"""
    logger.info("🚀 Iniciando AGI Chat API...")
    
    try:
        # 1. Carregar modelos preditivos
        logger.info("📦 Carregando modelos preditivos...")
        load_models()
        logger.info(f"✅ Modelos carregados: {list(MODELS.keys())}")
        
        # 2. Inicializar banco de dados
        logger.info("🗄️ Inicializando banco de dados...")
        await init_database()
        logger.info("✅ Database pronto")
        
        # 3. Inicializar memória vetorial
        logger.info("🧠 Inicializando memória vetorial...")
        global memory
        memory = VectorMemory(collection_name="agi_memory")
        await memory.start()
        logger.info("✅ Memória vetorial inicializada")
        
        # 4. Verificar se precisa ingestão inicial
        doc_count = memory.get_collection_size()
        if doc_count == 0:
            logger.info("📚 Memória vazia, executando ingestão inicial...")
            await run_initial_ingestion()
        else:
            logger.info(f"✅ Memória já possui {doc_count} documentos")
        
        logger.info("🎉 Startup concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro no startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup"""
    logger.info("👋 Encerrando AGI Chat API...")
    if memory:
        await memory.close()

# ============================================
# ENDPOINTS REST
# ============================================

@app.get("/")
def read_root():
    """Endpoint raiz"""
    return {
        "status": "ok",
        "message": "AGI Chat + Predictive API",
        "version": "2.1",
        "endpoints": {
            "predict": "/predict",
            "chat_ws": "/ws-chat",
            "health": "/health",
            "memory_stats": "/memory/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    models_status = {k: v is not None for k, v in MODELS.items()}
    memory_status = memory is not None and memory.collection is not None
    
    return {
        "status": "healthy",
        "models": models_status,
        "memory": memory_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(payload: PredictRequest):
    """
    Endpoint de predição (REST)
    """
    model_key = payload.type
    
    if model_key not in MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Modelo '{model_key}' não disponível. Opções: {list(MODELS.keys())}"
        )
    
    model = MODELS[model_key]
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail=f"Modelo '{model_key}' não carregado"
        )
    
    try:
        prediction = model_predict(model_key, payload.features)
        
        return {
            "model": model_key,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/stats")
async def get_memory_stats():
    """Estatísticas da memória vetorial"""
    if not memory:
        raise HTTPException(status_code=503, detail="Memória não inicializada")
    
    return {
        "collection_name": memory.collection_name,
        "total_documents": memory.get_collection_size(),
        "embedding_model": "all-MiniLM-L6-v2",
        "status": "active"
    }

@app.post("/memory/add")
async def add_to_memory(data: Dict[str, Any]):
    """Adicionar documento à memória"""
    if not memory:
        raise HTTPException(status_code=503, detail="Memória não inicializada")
    
    try:
        doc_id = data.get("id", f"doc_{datetime.now().timestamp()}")
        document = data.get("document")
        metadata = data.get("metadata", {})
        
        if not document:
            raise HTTPException(status_code=400, detail="Campo 'document' obrigatório")
        
        memory.add_documents(
            ids=[doc_id],
            documents=[document],
            metadatas=[metadata]
        )
        
        return {
            "status": "success",
            "id": doc_id,
            "message": "Documento adicionado à memória"
        }
    
    except Exception as e:
        logger.error(f"Erro ao adicionar à memória: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# WEBSOCKET CHAT
# ============================================

@app.websocket("/ws-chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket para chat com streaming
    """
    await websocket.accept()
    logger.info("✅ Cliente conectado ao chat")
    
    try:
        while True:
            # Receber mensagem
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            user_id = data.get("user_id", "anonymous")
            
            if not user_message:
                await websocket.send_json({
                    "type": "error",
                    "data": "Mensagem vazia"
                })
                continue
            
            logger.info(f"📥 Mensagem recebida de {user_id}: {user_message[:50]}...")
            
            # Verificar memória
            if not memory:
                await websocket.send_json({
                    "type": "error",
                    "data": "Memória não disponível"
                })
                continue
            
            # Gerar resposta com streaming
            try:
                async for chunk in respond_stream_generator(
                    user_message, 
                    user_id, 
                    memory,
                    MODELS
                ):
                    await websocket.send_json({
                        "type": "token",
                        "data": chunk
                    })
                
                # Sinal de fim
                await websocket.send_json({
                    "type": "end",
                    "data": "done"
                })
                
                logger.info(f"✅ Resposta enviada para {user_id}")
            
            except Exception as e:
                logger.error(f"Erro ao gerar resposta: {e}")
                await websocket.send_json({
                    "type": "error",
                    "data": f"Erro ao processar: {str(e)}"
                })
    
    except WebSocketDisconnect:
        logger.info("👋 Cliente desconectado")
    
    except Exception as e:
        logger.error(f"❌ Erro no WebSocket: {e}")

# ============================================
# INGESTÃO INICIAL
# ============================================

async def run_initial_ingestion():
    """
    Ingestão inicial de conhecimento base
    """
    logger.info("📚 Executando ingestão inicial...")
    
    # Conhecimento base sobre o sistema
    base_knowledge = [
        {
            "id": "system_intro",
            "document": "Sistema AGI Generativa v2.0 com capacidades de predição (CMAPSS e AI4I), raciocínio cognitivo, tomada de decisão e aprendizado por feedback (RLHF).",
            "metadata": {"category": "system", "priority": "high"}
        },
        {
            "id": "cmapss_info",
            "document": "CMAPSS é o modelo de predição de RUL (Remaining Useful Life) que estima a vida útil restante de motores turbofan usando dados de 21 sensores ao longo do tempo.",
            "metadata": {"category": "models", "type": "cmapss"}
        },
        {
            "id": "ai4i_info",
            "document": "AI4I é o modelo de predição de falhas em máquinas industriais que analisa temperatura, rotação, torque e desgaste para prever probabilidade de falha.",
            "metadata": {"category": "models", "type": "ai4i"}
        },
        {
            "id": "rul_definition",
            "document": "RUL (Remaining Useful Life) é a estimativa de quanto tempo ou ciclos operacionais restam antes de uma manutenção ser necessária.",
            "metadata": {"category": "concepts", "term": "rul"}
        },
        {
            "id": "prediction_process",
            "document": "O processo de predição envolve: 1) Coleta de dados dos sensores, 2) Normalização, 3) Predição usando modelo CNN-RNN ou Random Forest, 4) Análise de raciocínio, 5) Recomendação de ação.",
            "metadata": {"category": "process"}
        },
        {
            "id": "rlhf_info",
            "document": "RLHF (Reinforcement Learning from Human Feedback) é o mecanismo de aprendizado contínuo onde o sistema melhora suas respostas baseado no feedback dos usuários.",
            "metadata": {"category": "features", "type": "rlhf"}
        },
        {
            "id": "modules_info",
            "document": "Os módulos AGI incluem: Memória (curto e longo prazo), Raciocínio (causal, temporal, indutivo), Decisão (orientada a metas), Geração (explicações textuais) e Metacognição.",
            "metadata": {"category": "architecture"}
        }
    ]
    
    ids = [item["id"] for item in base_knowledge]
    documents = [item["document"] for item in base_knowledge]
    metadatas = [item["metadata"] for item in base_knowledge]
    
    memory.add_documents(ids=ids, documents=documents, metadatas=metadatas)
    
    logger.info(f"✅ Ingestão concluída: {len(base_knowledge)} documentos adicionados")

# ============================================
# INGESTÃO DE DATASETS KAGGLE (OPCIONAL)
# ============================================

@app.post("/ingest/kaggle")
async def ingest_kaggle_datasets():
    """
    Endpoint para ingerir datasets do Kaggle
    (Executar manualmente ou via cron job)
    """
    if not memory:
        raise HTTPException(status_code=503, detail="Memória não inicializada")
    
    try:
        from ingest_datasets import ingest_ai4i_sample, ingest_cmapss_sample
        
        logger.info("📊 Iniciando ingestão de datasets Kaggle...")
        
        # Ingerir AI4I
        await ingest_ai4i_sample(memory)
        
        # Ingerir CMAPSS
        await ingest_cmapss_sample(memory)
        
        total_docs = memory.get_collection_size()
        
        return {
            "status": "success",
            "message": "Datasets ingeridos com sucesso",
            "total_documents": total_docs
        }
    
    except Exception as e:
        logger.error(f"Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# EXECUÇÃO
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info("="*80)
    logger.info("🚀 AGI CHAT API v2.1")
    logger.info("="*80)
    logger.info(f"📡 Porta: {port}")
    logger.info(f"🔗 WebSocket: ws://localhost:{port}/ws-chat")
    logger.info(f"📚 Docs: http://localhost:{port}/docs")
    logger.info("="*80)
    
    uvicorn.run(
        "main_chat:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
