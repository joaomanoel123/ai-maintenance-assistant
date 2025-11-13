"""
===============================================================================
CHAT PIPELINE - Lógica de Chat com Streaming (Hugging Face)
===============================================================================

Implementa:
- Recuperação de contexto da memória vetorial
- Detecção de intenção de predição
- Geração com Hugging Face (local ou API)
- Persistência de experiências

Autor: João Manoel
===============================================================================
"""

import asyncio
import logging
import re
from typing import AsyncGenerator, Dict, Optional, Any
from datetime import datetime

from embeddings import VectorMemory
from model_loader import model_predict
from db import save_experience_record
from llm_huggingface import generate_response_stream
from utils_llm import format_prompt_for_chat

logger = logging.getLogger(__name__)

# ============================================
# DETECÇÃO DE INTENÇÃO
# ============================================

def extract_prediction_intent(message: str) -> Optional[Dict[str, Any]]:
    """
    Detectar se usuário quer fazer uma predição
    
    Returns:
        Dict com tipo e dados se detectado, None caso contrário
    """
    message_lower = message.lower()
    
    # Palavras-chave para RUL
    rul_keywords = ['rul', 'vida útil', 'vida util', 'quanto tempo', 'ciclos', 'cmapss']
    
    # Palavras-chave para Falha
    failure_keywords = ['falha', 'failure', 'quebra', 'defeito', 'ai4i', 'probabilidade']
    
    # Verificar RUL
    if any(kw in message_lower for kw in rul_keywords):
        return {"type": "rul", "model": "cmapss"}
    
    # Verificar Falha
    if any(kw in message_lower for kw in failure_keywords):
        return {"type": "failure", "model": "ai4i"}
    
    return None

# ============================================
# CONSTRUÇÃO DE PROMPT
# ============================================

def build_llm_prompt(
    user_message: str,
    contexts: list,
    prediction_info: Optional[Dict] = None
) -> str:
    """
    Construir prompt para Hugging Face
    
    Args:
        user_message: Mensagem do usuário
        contexts: Contextos recuperados da memória
        prediction_info: Informações de predição (opcional)
    
    Returns:
        Prompt formatado
    """
    # System prompt
    system_prompt = """Você é um assistente especializado em AGI (Inteligência Artificial Geral) e manutenção preditiva.
Seu papel é ajudar usuários a entender conceitos de RUL (Remaining Useful Life), predição de falhas, 
e análise de dados de sensores industriais. Seja claro, técnico quando necessário, mas acessível."""
    
    # Contexto da memória
    context_text = ""
    if contexts:
        context_text = "Informações relevantes:\n"
        for i, ctx in enumerate(contexts[:3], 1):
            context_text += f"{i}. {ctx}\n"
    
    # Informação de predição
    if prediction_info:
        context_text += f"\nPredição realizada:\n{prediction_info}\n"
    
    # Formatar prompt
    prompt = format_prompt_for_chat(
        system_prompt=system_prompt,
        user_message=user_message,
        context=context_text if context_text else None
    )
    
    return prompt

# ============================================
# GERAÇÃO DE RESPOSTA COM HUGGING FACE
# ============================================

async def respond_stream_generator(
    user_message: str,
    user_id: str,
    memory: VectorMemory,
    models: Dict
) -> AsyncGenerator[str, None]:
    """
    Gerar resposta usando Hugging Face com streaming
    
    Yields:
        Chunks da resposta
    """
    try:
        logger.info(f"🤖 Processando mensagem: {user_message[:50]}...")
        
        # 1. RECUPERAR CONTEXTO DA MEMÓRIA
        results = memory.query(user_message, n_results=5)
        contexts = results['documents'][0] if results and results['documents'] else []
        
        logger.info(f"📚 Contextos recuperados: {len(contexts)}")
        
        # 2. DETECTAR INTENÇÃO DE PREDIÇÃO
        prediction_intent = extract_prediction_intent(user_message)
        prediction_info = None
        
        if prediction_intent and models.get(prediction_intent['model']):
            logger.info(f"🎯 Intenção de predição detectada: {prediction_intent['type']}")
            
            # Fazer predição
            try:
                if prediction_intent['model'] == 'cmapss':
                    features = [520.0] * 21
                    pred_result = model_predict('cmapss', features)
                    rul_value = pred_result.get('rul', 'N/A')
                    prediction_info = f"RUL estimado: {rul_value} ciclos"
                
                elif prediction_intent['model'] == 'ai4i':
                    features = [1, 300, 310, 1500, 40, 100]
                    pred_result = model_predict('ai4i', features)
                    prob = pred_result.get('probability', 0)
                    prediction_info = f"Probabilidade de falha: {prob*100:.1f}%"
                
                logger.info(f"✅ Predição: {prediction_info}")
            
            except Exception as e:
                logger.error(f"❌ Erro na predição: {e}")
                prediction_info = "Erro ao realizar predição"
        
        # 3. CONSTRUIR PROMPT PARA LLM
        prompt = build_llm_prompt(user_message, contexts, prediction_info)
        
        logger.info("🤗 Gerando resposta com Hugging Face...")
        
        # 4. GERAR RESPOSTA COM STREAMING
        full_response = ""
        
        async for chunk in generate_response_stream(prompt, max_length=512):
            full_response += chunk
            yield chunk
        
        # 5. SALVAR EXPERIÊNCIA
        try:
            await save_experience_record(
                user_id=user_id,
                user_message=user_message,
                assistant_response=full_response,
                contexts=contexts,
                prediction_info=prediction_info
            )
            logger.info("💾 Experiência salva")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar experiência: {e}")
    
    except Exception as e:
        logger.error(f"❌ Erro na geração de resposta: {e}")
        yield f"Desculpe, ocorreu um erro: {str(e)}"
