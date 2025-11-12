"""
===============================================================================
MAIN CHAT API - Pipeline AGI v2.0 com Memória Vetorial (REFATORADO)
===============================================================================

Backend completo com:
- WebSocket para chat streaming
- Memória vetorial (ChromaDB)
- Modelos preditivos integrados
- Pipeline modular e organizado
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
from pydantic import BaseModel

# Imports dos módulos customizados
from model_loader import load_models, MODELS, model_predict
from embeddings import VectorMemory
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
# PIPELINE CORE
# ============================================

def search_context(question: str, n_results: int = 5) -> str:
    """
    Busca contexto relevante na memória vetorial
    """
    if not memory or not memory.collection:
        logger.warning("Memória não disponível para busca")
        return ""
    
    try:
        results = memory.query(question, n_results=n_results)
        
        if not results or not results.get("documents"):
            return ""
        
        # Extrai documentos relevantes
        context_docs = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        for doc, meta in zip(documents, metadatas):
            priority = meta.get("priority", "normal")
            category = meta.get("category", "general")
            context_docs.append(f"[{category.upper()}] {doc}")
        
        context = "\n".join(context_docs)
        logger.info(f"📚 Contexto recuperado: {len(documents)} documentos")
        
        return context
    
    except Exception as e:
        logger.error(f"Erro ao buscar contexto: {e}")
        return ""


def build_system_prompt(context: str) -> str:
    """
    Constrói o prompt do sistema com contexto
    """
    base_prompt = """Você é uma AGI (Inteligência Artificial Geral) avançada especializada em análise preditiva e manutenção industrial.

CAPACIDADES:
- Predição de RUL (Remaining Useful Life) com modelo CMAPSS
- Predição de falhas industriais com modelo AI4I
- Raciocínio causal e temporal
- Análise de dados de sensores
- Recomendações de manutenção

CONTEXTO RELEVANTE:
{context}

INSTRUÇÕES:
1. Use o contexto acima para fundamentar suas respostas
2. Seja técnico mas acessível
3. Para predições, explique o raciocínio
4. Sugira ações preventivas quando apropriado
5. Se não tiver certeza, seja honesto sobre limitações"""

    return base_prompt.format(context=context)


async def generate_answer_stream(question: str, context: str, user_id: str):
    """
    Gera resposta com streaming usando HuggingFace Inference API
    
    Yields:
        str: Tokens da resposta
    """
    try:
        from huggingface_hub import AsyncInferenceClient
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        model = os.getenv("MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        
        client = AsyncInferenceClient(token=hf_token)
        
        system_prompt = build_system_prompt(context)
        
        # Prompt completo
        full_prompt = f"""{system_prompt}

---

Pergunta do usuário: {question}

Resposta:"""
        
        # Streaming token por token
        stream = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            stream=True
        )
        
        async for token in stream:
            yield token
    
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {e}")
        yield f"[ERRO] Não foi possível gerar resposta: {str(e)}"


async def pipeline(question: str, user_id: str = "anonymous"):
    """
    Pipeline principal de processamento
    
    1. Busca contexto na memória vetorial
    2. Gera resposta com LLM usando contexto
    3. Salva experiência no banco
    4. Retorna resposta em streaming
    """
    logger.info(f"🔄 Pipeline iniciado para: {question[:50]}...")
    
    # 1. Buscar contexto relevante
    context = search_context(question)
    
    # 2. Detectar intenção de predição
    prediction_result = None
    if any(keyword in question.lower() for keyword in ['prever', 'predição', 'rul', 'falha', 'cmapss', 'ai4i']):
        logger.info("🎯 Intenção de predição detectada")
        # Aqui você pode chamar extract_prediction_intent do chat_pipeline.py
        # prediction_result = extract_prediction_intent(question, MODELS)
    
    # 3. Gerar resposta com streaming
    full_response = ""
    async for chunk in generate_answer_stream(question, context, user_id):
        full_response += chunk
        yield chunk
    
    # 4. Salvar experiência no banco (após completar resposta)
    try:
        await save_experience_record(
            user_id=user_id,
            user_input=question,
            assistant_output=full_response,
            context_used=context,
            prediction_data=prediction_result
        )
        logger.info("💾 Experiência salva no banco")
    except Exception as e:
        logger.error(f"Erro ao salvar experiência: {e}")


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
        "message": "AGI Chat + Predictive API (Pipeline v2.1)",
        "version": "2.1",
        "endpoints": {
            "predict": "/predict",
            "ask": "/ask",
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

@app.post("/ask")
async def ask_endpoint(data: ChatMessage):
    """
    Endpoint REST para perguntas (sem streaming)
    Similar ao exemplo do pipeline.py
    """
    if not memory:
        raise HTTPException(status_code=503, detail="Memória não inicializada")
    
    try:
        # Coletar toda a resposta
        full_answer = ""
        async for chunk in pipeline(data.message, data.user_id):
            full_answer += chunk
        
        return {
            "question": data.message,
            "answer": full_answer,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro no endpoint /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    WebSocket para chat com streaming usando pipeline
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
            
            # Usar pipeline para gerar resposta com streaming
            try:
                async for chunk in pipeline(user_message, user_id):
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
    logger.info("🚀 AGI CHAT API v2.1 (PIPELINE)")
    logger.info("="*80)
    logger.info(f"📡 Porta: {port}")
    logger.info(f"🔗 WebSocket: ws://localhost:{port}/ws-chat")
    logger.info(f"📝 REST Ask: http://localhost:{port}/ask")
    logger.info(f"📚 Docs: http://localhost:{port}/docs")
    logger.info("="*80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
