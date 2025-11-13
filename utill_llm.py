"""
===============================================================================
UTILS LLM - Funções Utilitárias para LLM
===============================================================================

Funções auxiliares para:
- Formatação de prompts (diferentes modelos)
- Extração de informações
- Validação de respostas
- Truncamento de contexto
- Limpeza de texto

Autor: João Manoel
===============================================================================
"""

import os
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ============================================
# FORMATAÇÃO DE PROMPTS
# ============================================

def format_prompt_for_chat(
    system_prompt: str, 
    user_message: str, 
    context: str = None
) -> str:
    """
    Formata o prompt de entrada para a LLM (formato genérico).
    
    Args:
        system_prompt: Instruções do sistema
        user_message: Mensagem do usuário
        context: Contexto adicional (opcional)
    
    Returns:
        Prompt formatado
    """
    prompt = f"{system_prompt}\n\nUsuário: {user_message}\n"
    
    if context:
        prompt += f"\nContexto adicional:\n{context}\n"
    
    prompt += "\nAssistente:"
    
    return prompt


def format_prompt_for_mistral(
    system_prompt: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Formato específico para Mistral/Zephyr
    
    Template: <|system|>\n...\n<|user|>\n...\n<|assistant|>
    """
    prompt = f"<|system|>\n{system_prompt}</s>\n"
    
    if context:
        prompt += f"<|context|>\n{context}</s>\n"
    
    prompt += f"<|user|>\n{user_message}</s>\n<|assistant|>"
    
    return prompt


def format_prompt_for_llama(
    system_prompt: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Formato específico para Llama-2 Chat
    
    Template: <s>[INST] <<SYS>>\n...\n<</SYS>>\n\n... [/INST]
    """
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    
    if context:
        prompt += f"Contexto:\n{context}\n\n"
    
    prompt += f"{user_message} [/INST]"
    
    return prompt


def format_prompt_for_gpt(
    system_prompt: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Formato para GPT-2 e modelos simples
    """
    prompt = f"{system_prompt}\n\n"
    
    if context:
        prompt += f"Context: {context}\n\n"
    
    prompt += f"User: {user_message}\nAssistant:"
    
    return prompt


def format_prompt_for_flan(
    system_prompt: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Formato para FLAN-T5 (encoder-decoder)
    """
    prompt = f"{system_prompt}\n\n"
    
    if context:
        prompt += f"Context: {context}\n\n"
    
    prompt += f"Question: {user_message}\nAnswer:"
    
    return prompt


def auto_format_prompt(
    model_name: str,
    system_prompt: str,
    user_message: str,
    context: str = None
) -> str:
    """
    Detecta automaticamente o formato correto baseado no nome do modelo
    
    Args:
        model_name: Nome do modelo (ex: "mistralai/Mistral-7B")
        system_prompt: Prompt do sistema
        user_message: Mensagem do usuário
        context: Contexto opcional
    
    Returns:
        Prompt formatado adequadamente
    """
    model_lower = model_name.lower()
    
    # Mistral / Zephyr
    if "mistral" in model_lower or "zephyr" in model_lower:
        return format_prompt_for_mistral(system_prompt, user_message, context)
    
    # Llama Chat
    elif "llama" in model_lower and "chat" in model_lower:
        return format_prompt_for_llama(system_prompt, user_message, context)
    
    # FLAN-T5
    elif "flan" in model_lower:
        return format_prompt_for_flan(system_prompt, user_message, context)
    
    # GPT-2 / Modelos simples
    elif "gpt" in model_lower:
        return format_prompt_for_gpt(system_prompt, user_message, context)
    
    # Fallback genérico
    else:
        logger.warning(f"Modelo desconhecido: {model_name}, usando formato genérico")
        return format_prompt_for_chat(system_prompt, user_message, context)


# ============================================
# TRUNCAMENTO E LIMPEZA
# ============================================

def truncate_context(
    context: str,
    max_tokens: int = 2000,
    tokens_per_word: float = 1.3
) -> str:
    """
    Trunca o contexto para caber no limite de tokens
    
    Args:
        context: Contexto a truncar
        max_tokens: Máximo de tokens permitidos
        tokens_per_word: Estimativa de tokens por palavra
    
    Returns:
        Contexto truncado
    """
    words = context.split()
    estimated_tokens = len(words) * tokens_per_word
    
    if estimated_tokens <= max_tokens:
        return context
    
    # Calcular quantas palavras manter
    max_words = int(max_tokens / tokens_per_word)
    
    # Truncar e adicionar indicador
    truncated = " ".join(words[:max_words])
    truncated += "\n\n[... contexto truncado ...]"
    
    logger.info(f"Contexto truncado: {len(words)} → {max_words} palavras")
    
    return truncated


def clean_llm_response(response: str) -> str:
    """
    Limpa a resposta da LLM removendo artefatos comuns
    
    Args:
        response: Resposta bruta da LLM
    
    Returns:
        Resposta limpa
    """
    if not response:
        return ""
    
    # Remover tags especiais que alguns modelos geram
    response = re.sub(r'<\|.*?\|>', '', response)
    response = re.sub(r'</s>', '', response)
    response = re.sub(r'<s>', '', response)
    
    # Remover prompt vazado (quando o modelo repete o input)
    response = re.sub(r'^(User:|Usuário:|Question:|Pergunta:).*?\n', '', response, flags=re.MULTILINE)
    response = re.sub(r'^(Assistant:|Assistente:|Answer:|Resposta:)\s*', '', response)
    
    # Remover múltiplas linhas vazias
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Trim
    response = response.strip()
    
    return response


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extrai blocos de código do texto
    
    Args:
        text: Texto contendo código
    
    Returns:
        Lista de dicionários com {language, code}
    """
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_blocks = []
    for lang, code in matches:
        code_blocks.append({
            "language": lang or "text",
            "code": code.strip()
        })
    
    return code_blocks


# ============================================
# DETECÇÃO DE INTENÇÕES
# ============================================

def detect_prediction_intent(message: str) -> Optional[str]:
    """
    Detecta se a mensagem pede uma predição
    
    Args:
        message: Mensagem do usuário
    
    Returns:
        Tipo de predição ('rul', 'falha') ou None
    """
    message_lower = message.lower()
    
    # Keywords para RUL
    rul_keywords = [
        'rul', 'vida útil', 'vida util', 'remaining useful life',
        'quanto tempo', 'duração', 'ciclos restantes', 'cmapss'
    ]
    
    # Keywords para falha
    falha_keywords = [
        'falha', 'failure', 'defeito', 'quebra', 'problema',
        'vai falhar', 'probabilidade', 'ai4i', 'predição de falha'
    ]
    
    # Verificar RUL
    if any(keyword in message_lower for keyword in rul_keywords):
        return 'rul'
    
    # Verificar falha
    if any(keyword in message_lower for keyword in falha_keywords):
        return 'falha'
    
    return None


def extract_sensor_values(message: str) -> Optional[List[float]]:
    """
    Extrai valores numéricos da mensagem (possíveis leituras de sensores)
    
    Args:
        message: Mensagem do usuário
    
    Returns:
        Lista de valores float ou None
    """
    # Padrão para números (inteiros ou decimais)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, message)
    
    if not matches:
        return None
    
    try:
        values = [float(m) for m in matches]
        return values if len(values) > 0 else None
    except ValueError:
        return None


# ============================================
# VALIDAÇÃO
# ============================================

def validate_llm_response(response: str, min_length: int = 10) -> bool:
    """
    Valida se a resposta da LLM é aceitável
    
    Args:
        response: Resposta da LLM
        min_length: Tamanho mínimo em caracteres
    
    Returns:
        True se válida
    """
    if not response or len(response.strip()) < min_length:
        return False
    
    # Verificar se não é apenas repetição
    words = response.split()
    if len(set(words)) < len(words) * 0.3:  # Menos de 30% palavras únicas
        logger.warning("Resposta com muita repetição")
        return False
    
    return True


def is_safe_response(response: str) -> bool:
    """
    Verifica se a resposta é segura (sem conteúdo problemático)
    
    Args:
        response: Resposta a validar
    
    Returns:
        True se segura
    """
    unsafe_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        # Adicione mais padrões conforme necessário
    ]
    
    response_lower = response.lower()
    
    for pattern in unsafe_patterns:
        if re.search(pattern, response_lower):
            logger.warning(f"Resposta contém padrão inseguro: {pattern}")
            return False
    
    return True


# ============================================
# CONSTRUÇÃO DE CONTEXTO
# ============================================

def build_context_from_memory(
    query_results: Dict[str, Any],
    max_docs: int = 5
) -> str:
    """
    Constrói contexto formatado a partir dos resultados da memória vetorial
    
    Args:
        query_results: Resultados do memory.query()
        max_docs: Máximo de documentos a incluir
    
    Returns:
        Contexto formatado
    """
    if not query_results or not query_results.get("documents"):
        return ""
    
    documents = query_results.get("documents", [[]])[0][:max_docs]
    metadatas = query_results.get("metadatas", [[]])[0][:max_docs]
    distances = query_results.get("distances", [[]])[0][:max_docs]
    
    context_parts = []
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        category = meta.get("category", "geral")
        priority = meta.get("priority", "normal")
        
        # Adicionar marcador de relevância
        relevance = "🔥" if dist < 0.5 else "📌" if dist < 0.7 else "📄"
        
        context_parts.append(
            f"{relevance} [{category.upper()}] {doc}"
        )
    
    context = "\n\n".join(context_parts)
    
    return context


def build_system_prompt_with_context(
    base_prompt: str,
    context: str,
    prediction_context: Optional[str] = None
) -> str:
    """
    Constrói prompt do sistema completo com contexto
    
    Args:
        base_prompt: Prompt base do sistema
        context: Contexto da memória vetorial
        prediction_context: Contexto de predição (opcional)
    
    Returns:
        System prompt completo
    """
    prompt = base_prompt + "\n\n"
    
    if context:
        prompt += f"CONTEXTO RELEVANTE:\n{context}\n\n"
    
    if prediction_context:
        prompt += f"DADOS DE PREDIÇÃO:\n{prediction_context}\n\n"
    
    prompt += """INSTRUÇÕES:
1. Use o contexto acima para fundamentar suas respostas
2. Seja técnico mas acessível
3. Para predições, explique o raciocínio
4. Sugira ações preventivas quando apropriado
5. Se não tiver certeza, seja honesto sobre limitações
6. Mantenha respostas concisas e diretas"""
    
    return prompt


# ============================================
# HELPERS
# ============================================

def estimate_tokens(text: str) -> int:
    """
    Estima número de tokens em um texto
    
    Args:
        text: Texto a analisar
    
    Returns:
        Estimativa de tokens
    """
    # Estimativa simples: ~1.3 tokens por palavra
    words = len(text.split())
    chars = len(text)
    
    # Usar média entre contagem de palavras e caracteres/4
    estimated = int((words * 1.3 + chars / 4) / 2)
    
    return estimated


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Divide texto em chunks para processamento
    
    Args:
        text: Texto a dividir
        chunk_size: Tamanho do chunk em tokens
        overlap: Sobreposição entre chunks
    
    Returns:
        Lista de chunks
    """
    words = text.split()
    
    # Converter token size para palavras (aprox)
    words_per_chunk = int(chunk_size / 1.3)
    words_overlap = int(overlap / 1.3)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - words_overlap
    
    return chunks


def create_conversation_history(
    messages: List[Dict[str, str]],
    max_messages: int = 10
) -> str:
    """
    Formata histórico de conversa
    
    Args:
        messages: Lista de mensagens [{role, content}]
        max_messages: Máximo de mensagens a incluir
    
    Returns:
        Histórico formatado
    """
    recent_messages = messages[-max_messages:]
    
    history_parts = []
    for msg in recent_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        prefix = "👤 Usuário:" if role == "user" else "🤖 Assistente:"
        history_parts.append(f"{prefix} {content}")
    
    return "\n\n".join(history_parts)


# ============================================
# TESTE
# ============================================

if __name__ == "__main__":
    print("="*80)
    print("🧪 TESTANDO UTILS_LLM")
    print("="*80)
    
    # Teste 1: Formatação de prompt
    print("\n📝 Teste 1: Formatação de prompt")
    prompt = format_prompt_for_chat(
        system_prompt="Você é um assistente técnico",
        user_message="Como funciona um motor?",
        context="Motores convertem energia em movimento"
    )
    print(prompt)
    
    # Teste 2: Detecção de intenção
    print("\n🎯 Teste 2: Detecção de intenção")
    intent1 = detect_prediction_intent("Qual o RUL deste motor?")
    intent2 = detect_prediction_intent("A máquina vai falhar?")
    print(f"Intent 1: {intent1}")
    print(f"Intent 2: {intent2}")
    
    # Teste 3: Extração de valores
    print("\n🔢 Teste 3: Extração de valores")
    values = extract_sensor_values("Temp: 85.5, Pressão: 120, RPM: 3500")
    print(f"Valores: {values}")
    
    # Teste 4: Limpeza de resposta
    print("\n🧹 Teste 4: Limpeza de resposta")
    dirty = "<|system|>User: test</s>Assistant: A resposta é clara.</s>"
    clean = clean_llm_response(dirty)
    print(f"Limpo: {clean}")
    
    # Teste 5: Estimativa de tokens
    print("\n📊 Teste 5: Estimativa de tokens")
    text = "Esta é uma frase de exemplo para testar a contagem de tokens"
    tokens = estimate_tokens(text)
    print(f"Texto: {text}")
    print(f"Tokens estimados: {tokens}")
    
    print("\n✅ Todos os testes concluídos!")
    print("="*80)
