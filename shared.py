"""
===============================================================================
SHARED - Funções Compartilhadas do Sistema
===============================================================================

Módulo com funções utilitárias compartilhadas entre diferentes componentes:
- Formatação de prompts
- Detecção de intenções
- Validação de dados
- Helpers comuns

Autor: João Manoel
===============================================================================
"""

import re
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================
# FORMATAÇÃO DE PROMPTS
# ============================================

def format_prompt_for_chat(message: str) -> str:
    """
    Formata mensagem para chat simples
    
    Args:
        message: Mensagem do usuário
    
    Returns:
        Prompt formatado
    """
    return f"User: {message}\nAssistant:"


def format_prompt_with_context(
    message: str, 
    context: str = None,
    system_prompt: str = None
) -> str:
    """
    Formata prompt com contexto e instruções do sistema
    
    Args:
        message: Mensagem do usuário
        context: Contexto relevante
        system_prompt: Prompt do sistema
    
    Returns:
        Prompt completo formatado
    """
    prompt_parts = []
    
    # Sistema
    if system_prompt:
        prompt_parts.append(f"System: {system_prompt}")
    
    # Contexto
    if context:
        prompt_parts.append(f"Context: {context}")
    
    # Mensagem do usuário
    prompt_parts.append(f"User: {message}")
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


# ============================================
# DETECÇÃO DE INTENÇÕES
# ============================================

def extract_prediction_intent(message: str) -> str:
    """
    Detecta a intenção da mensagem do usuário
    
    Args:
        message: Mensagem do usuário
    
    Returns:
        Tipo de intenção: "predict", "rul", "failure", "chat"
    """
    message_lower = message.lower()
    
    # Keywords de predição geral
    if "prever" in message_lower or "predição" in message_lower or "predict" in message_lower:
        return "predict"
    
    # Keywords específicas de RUL
    rul_keywords = [
        "rul", "vida útil", "vida util", "remaining useful life",
        "quanto tempo", "duração", "ciclos restantes", "cmapss",
        "tempo restante", "vida restante"
    ]
    if any(keyword in message_lower for keyword in rul_keywords):
        return "rul"
    
    # Keywords específicas de falha
    failure_keywords = [
        "falha", "failure", "defeito", "quebra", "problema",
        "vai falhar", "probabilidade de falha", "ai4i",
        "breakdown", "malfunction"
    ]
    if any(keyword in message_lower for keyword in failure_keywords):
        return "failure"
    
    # Padrão: conversa normal
    return "chat"


def detect_model_type(message: str) -> Optional[str]:
    """
    Detecta qual modelo preditivo usar
    
    Args:
        message: Mensagem do usuário
    
    Returns:
        Tipo do modelo: "cmapss", "ai4i" ou None
    """
    message_lower = message.lower()
    
    # CMAPSS (RUL de motores turbofan)
    cmapss_keywords = ["cmapss", "turbofan", "motor", "engine", "rul"]
    if any(keyword in message_lower for keyword in cmapss_keywords):
        return "cmapss"
    
    # AI4I (Falhas industriais)
    ai4i_keywords = ["ai4i", "industrial", "máquina", "machine", "temperatura", "torque"]
    if any(keyword in message_lower for keyword in ai4i_keywords):
        return "ai4i"
    
    return None


# ============================================
# EXTRAÇÃO DE DADOS
# ============================================

def extract_sensor_data(message: str) -> Optional[List[float]]:
    """
    Extrai valores numéricos da mensagem (possíveis dados de sensores)
    
    Args:
        message: Mensagem contendo valores
    
    Returns:
        Lista de floats ou None
    """
    # Padrão para números (inteiros ou decimais, positivos ou negativos)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, message)
    
    if not matches:
        return None
    
    try:
        values = [float(m) for m in matches]
        return values if len(values) > 0 else None
    except ValueError:
        logger.warning(f"Erro ao converter valores: {matches}")
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Tenta extrair JSON de um texto
    
    Args:
        text: Texto contendo possível JSON
    
    Returns:
        Dicionário ou None
    """
    import json
    
    # Procurar por blocos JSON
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            data = json.loads(match)
            return data
        except json.JSONDecodeError:
            continue
    
    return None


# ============================================
# VALIDAÇÃO
# ============================================

def validate_message(message: str, min_length: int = 3, max_length: int = 2000) -> bool:
    """
    Valida mensagem do usuário
    
    Args:
        message: Mensagem a validar
        min_length: Tamanho mínimo
        max_length: Tamanho máximo
    
    Returns:
        True se válida
    """
    if not message or not isinstance(message, str):
        return False
    
    message = message.strip()
    
    if len(message) < min_length:
        logger.warning(f"Mensagem muito curta: {len(message)} caracteres")
        return False
    
    if len(message) > max_length:
        logger.warning(f"Mensagem muito longa: {len(message)} caracteres")
        return False
    
    return True


def validate_sensor_features(features: List[float], expected_count: int) -> bool:
    """
    Valida features de sensores
    
    Args:
        features: Lista de valores
        expected_count: Número esperado de features
    
    Returns:
        True se válidas
    """
    if not features or not isinstance(features, list):
        return False
    
    if len(features) != expected_count:
        logger.warning(f"Features incorretas: esperado {expected_count}, recebido {len(features)}")
        return False
    
    # Verificar se todos são numéricos
    try:
        [float(f) for f in features]
        return True
    except (ValueError, TypeError):
        logger.warning("Features contêm valores não numéricos")
        return False


def is_safe_input(text: str) -> bool:
    """
    Verifica se o input é seguro (sem injeção)
    
    Args:
        text: Texto a validar
    
    Returns:
        True se seguro
    """
    # Padrões suspeitos
    unsafe_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'eval\(',
        r'exec\(',
        r'__import__',
        r'os\.',
        r'system\(',
    ]
    
    text_lower = text.lower()
    
    for pattern in unsafe_patterns:
        if re.search(pattern, text_lower):
            logger.warning(f"Input suspeito detectado: {pattern}")
            return False
    
    return True


# ============================================
# FORMATAÇÃO DE SAÍDA
# ============================================

def format_prediction_response(
    prediction: Any,
    model_type: str,
    confidence: float = None
) -> str:
    """
    Formata resposta de predição de forma legível
    
    Args:
        prediction: Resultado da predição
        model_type: Tipo do modelo ("cmapss" ou "ai4i")
        confidence: Confiança da predição (opcional)
    
    Returns:
        Resposta formatada
    """
    if model_type == "cmapss":
        # RUL - Remaining Useful Life
        rul_value = prediction if isinstance(prediction, (int, float)) else prediction.get("rul", 0)
        
        response = f"📊 **Predição de RUL (Vida Útil Restante)**\n\n"
        response += f"🔧 Ciclos restantes estimados: **{rul_value:.0f}**\n"
        
        if confidence:
            response += f"📈 Confiança: {confidence*100:.1f}%\n"
        
        # Recomendação baseada no RUL
        if rul_value < 50:
            response += "\n⚠️ **ATENÇÃO:** Manutenção urgente recomendada!"
        elif rul_value < 100:
            response += "\n⚡ **ALERTA:** Agendar manutenção em breve."
        else:
            response += "\n✅ **STATUS:** Equipamento em condições normais."
        
        return response
    
    elif model_type == "ai4i":
        # Falha - Failure Prediction
        failure_prob = prediction if isinstance(prediction, (int, float)) else prediction.get("failure_probability", 0)
        
        response = f"📊 **Predição de Falha Industrial**\n\n"
        response += f"⚠️ Probabilidade de falha: **{failure_prob*100:.1f}%**\n"
        
        if confidence:
            response += f"📈 Confiança: {confidence*100:.1f}%\n"
        
        # Recomendação baseada na probabilidade
        if failure_prob > 0.7:
            response += "\n🚨 **CRÍTICO:** Risco alto de falha! Intervenção imediata necessária."
        elif failure_prob > 0.4:
            response += "\n⚠️ **ALERTA:** Risco moderado. Monitorar de perto."
        else:
            response += "\n✅ **STATUS:** Risco baixo. Operação normal."
        
        return response
    
    else:
        return f"Predição: {prediction}"


def format_error_message(error: Exception, user_friendly: bool = True) -> str:
    """
    Formata mensagem de erro
    
    Args:
        error: Exceção
        user_friendly: Se True, retorna mensagem amigável
    
    Returns:
        Mensagem formatada
    """
    if user_friendly:
        return "❌ Desculpe, ocorreu um erro ao processar sua solicitação. Por favor, tente novamente."
    else:
        return f"❌ Erro: {str(error)}"


# ============================================
# HELPERS DE TIMESTAMP
# ============================================

def get_timestamp() -> str:
    """Retorna timestamp atual formatado"""
    return datetime.now().isoformat()


def format_timestamp(timestamp: datetime) -> str:
    """Formata timestamp para exibição"""
    return timestamp.strftime("%d/%m/%Y %H:%M:%S")


# ============================================
# HELPERS DE LOGGING
# ============================================

def log_user_interaction(
    user_id: str,
    message: str,
    intent: str,
    response_length: int = 0
):
    """
    Log de interação do usuário
    
    Args:
        user_id: ID do usuário
        message: Mensagem enviada
        intent: Intenção detectada
        response_length: Tamanho da resposta
    """
    logger.info(
        f"USER_INTERACTION | "
        f"user_id={user_id} | "
        f"intent={intent} | "
        f"msg_len={len(message)} | "
        f"resp_len={response_length}"
    )


def log_prediction(
    user_id: str,
    model_type: str,
    prediction: Any,
    features_count: int
):
    """
    Log de predição realizada
    
    Args:
        user_id: ID do usuário
        model_type: Tipo do modelo
        prediction: Resultado da predição
        features_count: Número de features usadas
    """
    logger.info(
        f"PREDICTION | "
        f"user_id={user_id} | "
        f"model={model_type} | "
        f"result={prediction} | "
        f"features={features_count}"
    )


# ============================================
# CONSTANTES
# ============================================

# Mensagens padrão
DEFAULT_WELCOME_MESSAGE = """
👋 Olá! Sou a AGI Preditiva, sua assistente especializada em análise e predição industrial.

Posso ajudar você com:
🔧 Predição de RUL (Vida Útil Restante) de equipamentos
⚠️ Predição de Falhas em máquinas industriais
📊 Análise de dados de sensores
💡 Recomendações de manutenção

Como posso ajudá-lo hoje?
"""

DEFAULT_ERROR_MESSAGE = "❌ Desculpe, não consegui processar sua solicitação. Por favor, tente novamente."

DEFAULT_EMPTY_MESSAGE = "Por favor, envie uma mensagem válida."

# Limites
MAX_MESSAGE_LENGTH = 2000
MIN_MESSAGE_LENGTH = 3
MAX_CONTEXT_LENGTH = 5000

# Features esperadas por modelo
CMAPSS_FEATURES_COUNT = 21  # 21 sensores
AI4I_FEATURES_COUNT = 5     # 5 features (temp, rotação, torque, etc)


# ============================================
# TESTE
# ============================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTANDO SHARED.PY")
    print("="*80)
    
    # Teste 1: Formatação de prompt
    print("\n📝 Teste 1: Formatação de prompt")
    prompt = format_prompt_for_chat("Como funciona o RUL?")
    print(prompt)
    
    # Teste 2: Detecção de intenção
    print("\n🎯 Teste 2: Detecção de intenção")
    tests = [
        "Prever o RUL do motor",
        "A máquina vai falhar?",
        "Olá, como você está?",
    ]
    for test in tests:
        intent = extract_prediction_intent(test)
        model = detect_model_type(test)
        print(f"  '{test}' -> intent={intent}, model={model}")
    
    # Teste 3: Extração de dados
    print("\n🔢 Teste 3: Extração de dados")
    message = "Temp: 85.5, Pressão: 120.0, RPM: 3500"
    values = extract_sensor_data(message)
    print(f"  Valores extraídos: {values}")
    
    # Teste 4: Validação
    print("\n✅ Teste 4: Validação")
    valid = validate_message("Mensagem válida com tamanho adequado")
    invalid = validate_message("ab")
    print(f"  Mensagem válida: {valid}")
    print(f"  Mensagem inválida: {invalid}")
    
    # Teste 5: Formatação de resposta
    print("\n📊 Teste 5: Formatação de resposta")
    response = format_prediction_response(75.5, "cmapss", 0.92)
    print(response)
    
    # Teste 6: Segurança
    print("\n🔒 Teste 6: Verificação de segurança")
    safe = is_safe_input("Qual o RUL do motor?")
    unsafe = is_safe_input("<script>alert('test')</script>")
    print(f"  Input seguro: {safe}")
    print(f"  Input inseguro: {unsafe}")
    
    print("\n✅ Todos os testes concluídos!")
    print("="*80)
