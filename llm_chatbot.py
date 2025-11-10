"""
===============================================================================
LLM HUGGING FACE - Integração com Transformers
===============================================================================

Suporta:
- Modelos locais (GPT-2, GPT-J, Mistral, etc)
- Hugging Face Inference API
- Pipeline de geração com streaming
- Cache de modelos

Autor: João Manoel
===============================================================================
"""

import os
import logging
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)

# Variáveis globais
_model = None
_tokenizer = None
_device = None
_use_api = False

# ============================================
# CONFIGURAÇÃO
# ============================================

class HuggingFaceConfig:
    """Configuração do Hugging Face"""
    
    # Modo: 'local' ou 'api'
    MODE = os.getenv("HF_MODE", "local")
    
    # Modelo a usar (local)
    MODEL_NAME = os.getenv("HF_MODEL", "gpt2")  # Opções: gpt2, gpt2-medium, mistralai/Mistral-7B-Instruct-v0.2
    
    # API Token (se usar API)
    API_TOKEN = os.getenv("HF_API_TOKEN", None)
    
    # API Endpoint (se usar API)
    API_ENDPOINT = os.getenv("HF_API_ENDPOINT", None)
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parâmetros de geração
    MAX_LENGTH = int(os.getenv("HF_MAX_LENGTH", "512"))
    TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("HF_TOP_P", "0.9"))
    TOP_K = int(os.getenv("HF_TOP_K", "50"))

# ============================================
# INICIALIZAÇÃO
# ============================================

async def initialize_huggingface():
    """
    Inicializar modelo Hugging Face
    """
    global _model, _tokenizer, _device, _use_api
    
    logger.info("🤗 Inicializando Hugging Face...")
    logger.info(f"   Modo: {HuggingFaceConfig.MODE}")
    logger.info(f"   Modelo: {HuggingFaceConfig.MODEL_NAME}")
    logger.info(f"   Device: {HuggingFaceConfig.DEVICE}")
    
    if HuggingFaceConfig.MODE == "api":
        # Modo API
        if not HuggingFaceConfig.API_TOKEN:
            logger.warning("⚠️ HF_API_TOKEN não configurado, usando modo local")
            HuggingFaceConfig.MODE = "local"
        else:
            _use_api = True
            logger.info("✅ Modo API configurado")
            return
    
    # Modo Local
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"📦 Carregando modelo: {HuggingFaceConfig.MODEL_NAME}")
        
        # Carregar tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            HuggingFaceConfig.MODEL_NAME,
            trust_remote_code=True
        )
        
        # Configurar pad_token se necessário
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        logger.info("✅ Tokenizer carregado")
        
        # Carregar modelo
        _device = HuggingFaceConfig.DEVICE
        
        # Usar quantização se CPU ou memória limitada
        if _device == "cpu" or "gpt2" not in HuggingFaceConfig.MODEL_NAME.lower():
            logger.info("💾 Carregando modelo com 8-bit quantization...")
            _model = AutoModelForCausalLM.from_pretrained(
                HuggingFaceConfig.MODEL_NAME,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                HuggingFaceConfig.MODEL_NAME,
                trust_remote_code=True
            )
            _model = _model.to(_device)
        
        _model.eval()
        
        logger.info(f"✅ Modelo carregado no device: {_device}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo Hugging Face: {e}")
        logger.warning("⚠️ Sistema continuará com respostas template")

# ============================================
# GERAÇÃO (LOCAL)
# ============================================

async def generate_local_stream(
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """
    Gerar resposta usando modelo local com streaming
    """
    if not _model or not _tokenizer:
        logger.error("Modelo não inicializado")
        yield "Modelo não disponível. "
        return
    
    try:
        # Tokenizar
        inputs = _tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(_device)
        
        # Configurar geração
        generation_config = {
            "max_new_tokens": 200,
            "temperature": temperature,
            "top_p": HuggingFaceConfig.TOP_P,
            "top_k": HuggingFaceConfig.TOP_K,
            "do_sample": True,
            "pad_token_id": _tokenizer.eos_token_id,
            "eos_token_id": _tokenizer.eos_token_id
        }
        
        # Gerar
        with torch.no_grad():
            # Streaming manual (token por token)
            input_length = inputs['input_ids'].shape[1]
            
            for _ in range(generation_config["max_new_tokens"]):
                # Gerar próximo token
                outputs = _model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Logits do último token
                next_token_logits = outputs.logits[:, -1, :]
                
                # Aplicar temperatura
                next_token_logits = next_token_logits / temperature
                
                # Sampling
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Verificar EOS
                if next_token.item() == _tokenizer.eos_token_id:
                    break
                
                # Decodificar token
                token_text = _tokenizer.decode(next_token[0], skip_special_tokens=True)
                
                # Yield token
                yield token_text + " "
                
                # Adicionar token à sequência
                inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)
                inputs['attention_mask'] = torch.cat([
                    inputs['attention_mask'], 
                    torch.ones((1, 1), dtype=torch.long).to(_device)
                ], dim=-1)
                
                # Pequeno delay para simular streaming
                await asyncio.sleep(0.05)
    
    except Exception as e:
        logger.error(f"Erro na geração local: {e}")
        yield f"Erro ao gerar resposta: {str(e)}"

# ============================================
# GERAÇÃO (API)
# ============================================

async def generate_api_stream(
    prompt: str,
    max_length: int = 512
) -> AsyncGenerator[str, None]:
    """
    Gerar resposta usando Hugging Face Inference API
    """
    if not HuggingFaceConfig.API_TOKEN:
        logger.error("API Token não configurado")
        yield "API não configurada. "
        return
    
    try:
        import httpx
        
        # Endpoint
        if HuggingFaceConfig.API_ENDPOINT:
            api_url = HuggingFaceConfig.API_ENDPOINT
        else:
            api_url = f"https://api-inference.huggingface.co/models/{HuggingFaceConfig.MODEL_NAME}"
        
        headers = {"Authorization": f"Bearer {HuggingFaceConfig.API_TOKEN}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": HuggingFaceConfig.TEMPERATURE,
                "top_p": HuggingFaceConfig.TOP_P,
                "return_full_text": False,
                "do_sample": True
            },
            "options": {
                "use_cache": False
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extrair texto gerado
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    generated_text = result.get("generated_text", "")
                else:
                    generated_text = str(result)
                
                # Simular streaming (dividir em palavras)
                words = generated_text.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.05)
            
            else:
                logger.error(f"API error: {response.status_code}")
                yield f"Erro na API: {response.status_code}"
    
    except Exception as e:
        logger.error(f"Erro na geração via API: {e}")
        yield f"Erro: {str(e)}"

# ============================================
# INTERFACE UNIFICADA
# ============================================

async def generate_response_stream(
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7
) -> AsyncGenerator[str, None]:
    """
    Gerar resposta (detecta automaticamente modo local ou API)
    """
    if _use_api:
        logger.info("🌐 Gerando via API")
        async for chunk in generate_api_stream(prompt, max_length):
            yield chunk
    elif _model and _tokenizer:
        logger.info("💻 Gerando localmente")
        async for chunk in generate_local_stream(prompt, max_length, temperature):
            yield chunk
    else:
        logger.warning("⚠️ Nenhum modelo disponível, usando fallback")
        yield "Modelo não disponível. Por favor, configure HF_MODEL ou HF_API_TOKEN."

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def format_prompt_for_chat(
    system_prompt: str,
    user_message: str,
    context: Optional[str] = None
) -> str:
    """
    Formatar prompt para chat (estilo instruction-following)
    """
    prompt = f"### System:\n{system_prompt}\n\n"
    
    if context:
        prompt += f"### Context:\n{context}\n\n"
    
    prompt += f"### User:\n{user_message}\n\n### Assistant:\n"
    
    return prompt

def get_model_info() -> Dict[str, Any]:
    """
    Obter informações sobre o modelo carregado
    """
    info = {
        "mode": HuggingFaceConfig.MODE,
        "model_name": HuggingFaceConfig.MODEL_NAME,
        "device": HuggingFaceConfig.DEVICE if _device else "not_loaded",
        "loaded": _model is not None,
        "api_configured": _use_api
    }
    
    if _model:
        try:
            info["num_parameters"] = sum(p.numel() for p in _model.parameters())
        except:
            pass
    
    return info

async def test_generation():
    """
    Testar geração (útil para debug)
    """
    logger.info("🧪 Testando geração...")
    
    test_prompt = "Hello, how are you?"
    
    response = ""
    async for chunk in generate_response_stream(test_prompt):
        response += chunk
    
    logger.info(f"✅ Teste concluído: {response[:100]}...")
    
    return response

# ============================================
# LIMPEZA
# ============================================

async def cleanup():
    """
    Liberar memória
    """
    global _model, _tokenizer
    
    if _model:
        del _model
        _model = None
    
    if _tokenizer:
        del _tokenizer
        _tokenizer = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("🧹 Memória liberada")
