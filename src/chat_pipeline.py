"""
===============================================================================
CHAT PIPELINE - Pipeline de Processamento de Chat
===============================================================================

Pipeline completo para processamento de mensagens:
1. Busca contexto na memória vetorial
2. Detecta intenção (predição ou chat)
3. Executa predição se necessário
4. Gera resposta via LLM com streaming
5. Salva experiência no banco

Autor: João Manoel
===============================================================================
"""

import logging
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================
# IMPORTS LOCAIS
# ============================================

try:
    from utils_llm import (
        auto_format_prompt,
        build_context_from_memory,
        clean_llm_response,
        validate_llm_response,
        truncate_context
    )
    from shared import (
        extract_prediction_intent,
        detect_model_type,
        extract_sensor_data,
        validate_message,
        format_prediction_response,
        log_user_interaction,
        log_prediction
    )
    from db import save_experience_record
except ImportError as e:
    logger.warning(f"Alguns módulos opcionais não encontrados: {e}")
    # Fallbacks caso os módulos não existam
    def auto_format_prompt(model_name, system_prompt, user_message, context=None):
        return f"{system_prompt}\n\n{context or ''}\n\nUser: {user_message}\nAssistant:"
    
    def build_context_from_memory(results):
        if not results or not results.get("documents"):
            return ""
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs[:5])
    
    def clean_llm_response(text):
        return text.strip()
    
    def validate_llm_response(text, min_length=10):
        return len(text.strip()) >= min_length
    
    def truncate_context(text, max_tokens=2000, tokens_per_word=1.3):
        words = text.split()
        max_words = int(max_tokens / tokens_per_word)
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "\n[... truncado ...]"
    
    def extract_prediction_intent(message):
        msg_lower = message.lower()
        if "rul" in msg_lower or "vida útil" in msg_lower:
            return "rul"
        if "falha" in msg_lower or "failure" in msg_lower:
            return "failure"
        if "prever" in msg_lower or "predição" in msg_lower:
            return "predict"
        return "chat"
    
    def detect_model_type(message):
        msg_lower = message.lower()
        if "cmapss" in msg_lower or "turbofan" in msg_lower:
            return "cmapss"
        if "ai4i" in msg_lower or "industrial" in msg_lower:
            return "ai4i"
        return None
    
    def extract_sensor_data(message):
        import re
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, message)
        return [float(m) for m in matches] if matches else None
    
    def validate_message(message, min_length=3, max_length=2000):
        return message and isinstance(message, str) and min_length <= len(message.strip()) <= max_length
    
    def format_prediction_response(prediction, model_type, confidence=None):
        return f"Predição ({model_type}): {prediction}"
    
    def log_user_interaction(user_id, message, intent, response_length=0):
        logger.info(f"USER: {user_id} | INTENT: {intent} | MSG_LEN: {len(message)}")
    
    def log_prediction(user_id, model_type, prediction, features_count):
        logger.info(f"PREDICTION: {user_id} | MODEL: {model_type} | RESULT: {prediction}")
    
    async def save_experience_record(*args, **kwargs):
        logger.info("Experiência registrada (fallback)")

# ============================================
# SISTEMA PROMPT
# ============================================

BASE_SYSTEM_PROMPT = """Você é uma AGI (Inteligência Artificial Geral) especializada em manutenção preditiva industrial.

CAPACIDADES:
- Predição de RUL (Remaining Useful Life) usando modelo CMAPSS
- Predição de falhas industriais usando modelo AI4I
- Análise de dados de sensores em tempo real
- Raciocínio causal e temporal sobre equipamentos
- Recomendações técnicas de manutenção

PERSONALIDADE:
- Técnico mas acessível
- Direto e objetivo
- Proativo em sugerir ações preventivas
- Honesto sobre limitações

FORMATO DE RESPOSTA:
- Use markdown para estruturar
- Inclua emojis técnicos quando apropriado (🔧⚠️📊✅)
- Seja conciso mas completo
- Sempre explique o raciocínio por trás de predições"""

# ============================================
# EXTRAÇÃO DE INTENÇÃO DE PREDIÇÃO
# ============================================

def extract_prediction_intent_detailed(
    user_message: str,
    models: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Detecta se a mensagem pede uma predição e extrai detalhes
    
    Args:
        user_message: Mensagem do usuário
        models: Dicionário de modelos disponíveis
    
    Returns:
        Dicionário com detalhes da predição ou None
    """
    intent = extract_prediction_intent(user_message)
    
    if intent not in ["predict", "rul", "failure"]:
        return None
    
    # Detectar tipo de modelo
    model_type = detect_model_type(user_message)
    
    # Tentar extrair dados de sensores
    sensor_data = extract_sensor_data(user_message)
    
    # Verificar se o modelo está disponível
    if model_type and model_type not in models:
        return {
            "intent": intent,
            "model_type": model_type,
            "available": False,
            "error": f"Modelo '{model_type}' não disponível"
        }
    
    return {
        "intent": intent,
        "model_type": model_type,
        "sensor_data": sensor_data,
        "available": model_type in models if model_type else False
    }

# ============================================
# EXECUÇÃO DE PREDIÇÃO
# ============================================

async def execute_prediction(
    prediction_info: Dict[str, Any],
    models: Dict[str, Any],
    user_id: str
) -> Optional[Dict[str, Any]]:
    """
    Executa predição com base nas informações extraídas
    
    Args:
        prediction_info: Informações da predição
        models: Modelos disponíveis
        user_id: ID do usuário
    
    Returns:
        Resultado da predição ou None
    """
    if not prediction_info or not prediction_info.get("available"):
        return None
    
    model_type = prediction_info.get("model_type")
    sensor_data = prediction_info.get("sensor_data")
    
    if not model_type or not sensor_data:
        logger.warning("Dados insuficientes para predição")
        return None
    
    try:
        from model_loader import model_predict
        
        # Executar predição
        result = model_predict(model_type, sensor_data)
        
        # Log
        log_prediction(user_id, model_type, result, len(sensor_data))
        
        return {
            "model_type": model_type,
            "prediction": result,
            "sensor_data": sensor_data,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro ao executar predição: {e}")
        return None

# ============================================
# GERADOR DE RESPOSTA COM STREAMING
# ============================================

async def respond_stream_generator(
    user_message: str,
    user_id: str,
    memory,
    models: Dict[str, Any],
    llm_client=None
) -> AsyncGenerator[str, None]:
    """
    Pipeline completo de geração de resposta com streaming
    
    Args:
        user_message: Mensagem do usuário
        user_id: ID do usuário
        memory: Instância de VectorMemory
        models: Dicionário de modelos preditivos
        llm_client: Cliente LLM (HuggingFace)
    
    Yields:
        Tokens da resposta
    """
    
    # 1. Validar mensagem
    if not validate_message(user_message):
        yield "❌ Por favor, envie uma mensagem válida."
        return
    
    logger.info(f"📥 Pipeline iniciado: {user_message[:50]}...")
    
    try:
        # 2. Buscar contexto na memória vetorial
        context = ""
        if memory and memory.collection:
            try:
                results = memory.query(user_message, n_results=5)
                context = build_context_from_memory(results)
                context = truncate_context(context, max_tokens=2000)
                logger.info(f"📚 Contexto recuperado: {len(context)} chars")
            except Exception as e:
                logger.warning(f"Erro ao buscar contexto: {e}")
        
        # 3. Detectar intenção de predição
        prediction_info = extract_prediction_intent_detailed(user_message, models)
        prediction_result = None
        
        if prediction_info and prediction_info.get("available"):
            logger.info(f"🎯 Predição detectada: {prediction_info.get('model_type')}")
            prediction_result = await execute_prediction(prediction_info, models, user_id)
            
            if prediction_result:
                # Adicionar resultado da predição ao contexto
                pred_text = format_prediction_response(
                    prediction_result.get("prediction"),
                    prediction_result.get("model_type")
                )
                context = f"{pred_text}\n\n{context}"
        
        # 4. Construir prompt do sistema
        system_prompt = BASE_SYSTEM_PROMPT
        if context:
            system_prompt += f"\n\nCONTEXTO RELEVANTE:\n{context}"
        
        # 5. Gerar resposta via LLM com streaming
        if not llm_client:
            yield "⚠️ LLM não disponível. Por favor, configure o modelo."
            return
        
        # Formatar prompt
        model_name = getattr(llm_client, 'model_name', 'gpt2')
        prompt = auto_format_prompt(
            model_name=model_name,
            system_prompt=system_prompt,
            user_message=user_message,
            context=None  # Já incluído no system_prompt
        )
        
        logger.info("🤖 Gerando resposta com LLM...")
        
        # Stream da resposta
        full_response = ""
        async for token in llm_client.generate_stream(
            prompt=prompt,
            max_length=800,
            temperature=0.7,
            top_p=0.9
        ):
            clean_token = token
            full_response += clean_token
            yield clean_token
        
        # 6. Limpar e validar resposta
        full_response = clean_llm_response(full_response)
        
        if not validate_llm_response(full_response):
            logger.warning("Resposta inválida gerada")
            yield "\n\n⚠️ Desculpe, não consegui gerar uma resposta adequada."
            return
        
        # 7. Salvar experiência no banco
        try:
            intent = prediction
