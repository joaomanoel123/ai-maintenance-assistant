"""
===============================================================================
HUGGING FACE LLM - Integração com Modelos Hugging Face
===============================================================================

Implementa integração com modelos do Hugging Face:
- Modelos de texto (GPT-2, Llama, Mistral, etc)
- Streaming de tokens
- Inferência local ou via API

Autor: João Manoel
===============================================================================
"""

import os
import logging
from typing import AsyncGenerator, Optional, Dict
import asyncio

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURAÇÃO
# ============================================

class HuggingFaceConfig:
    """Configuração de modelos Hugging Face"""
    
    # Modelos recomendados (do menor ao maior)
    MODELS = {
        # Modelos pequenos (rodam local em CPU)
        "gpt2": "gpt2",                                    # 124M params
        "gpt2-medium": "gpt2-medium",                      # 355M params
        "distilgpt2": "distilgpt2",                        # 82M params (mais rápido)
        
        # Modelos médios (necessitam GPU ou API)
        "flan-t5-base": "google/flan-t5-base",            # 250M params (bom para QA)
        "flan-t5-large": "google/flan-t5-large",          # 780M params
        
        # Modelos grandes (usar via Inference API)
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # 7B params
        "llama-7b": "meta-llama/Llama-2-7b-chat-hf",         # 7B params (requer aprovação)
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",         # 7B params (aberto)
    }
    
    # Configuração padrão
    DEFAULT_MODEL = os.getenv("HF_MODEL", "gpt2")  # Pequeno para começar
    USE_API = os.getenv("HF_USE_API", "false").lower() == "true"
    API_TOKEN = os.getenv("HF_API_TOKEN", None)
    
    # Parâmetros de geração
    MAX_LENGTH = int(os.getenv("HF_MAX_LENGTH", "200"))
    TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("HF_TOP_P", "0.9"))
    TOP_K = int(os.getenv("HF_TOP_K", "50"))

# ============================================
# MODO 1: INFERÊNCIA LOCAL
# ============================================

class LocalHuggingFaceLLM:
    """
    LLM local usando transformers
    """
    
    def __init__(self, model_name: str = HuggingFaceConfig.DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"  # Pode ser "cuda" se tiver GPU
        
    def load(self):
        """Carregar modelo e tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"🔄 Carregando modelo local: {self.model_name}")
            
            # Verificar GPU
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("✅ GPU detectada, usando CUDA")
            else:
                logger.info("⚠️ GPU não disponível, usando CPU")
            
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Carregar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Configurar pad token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✅ Modelo {self.model_name} carregado no {self.device}")
            return True
            
        except ImportError:
            logger.error("❌ transformers ou torch não instalado!")
            logger.error("Instale com: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    async def generate_stream(
        self, 
        prompt: str,
        max_length: int = HuggingFaceConfig.MAX_LENGTH,
        temperature: float = HuggingFaceConfig.TEMPERATURE,
        top_p: float = HuggingFaceConfig.TOP_P,
        top_k: int = HuggingFaceConfig.TOP_K
    ) -> AsyncGenerator[str, None]:
        """
        Gerar resposta com streaming
        """
        if not self.model or not self.tokenizer:
            yield "Erro: Modelo não carregado"
            return
        
        try:
            # Tokenizar input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Gerar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decodificar
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remover prompt da resposta
            response = generated_text[len(prompt):].strip()
            
            # Simular streaming (quebrar em palavras)
            words = response.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Simular latência
        
        except Exception as e:
            logger.error(f"❌ Erro na geração: {e}")
            yield f"Erro ao gerar resposta: {str(e)}"

# ============================================
# MODO 2: INFERENCE API (RECOMENDADO)
# ============================================

class HuggingFaceInferenceAPI:
    """
    LLM via Hugging Face Inference API (gratuito com rate limit)
    """
    
    def __init__(
        self, 
        model_name: str = HuggingFaceConfig.DEFAULT_MODEL,
        api_token: Optional[str] = None
    ):
        self.model_name = model_name
        self.api_token = api_token or HuggingFaceConfig.API_TOKEN
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        if not self.api_token:
            logger.warning("⚠️ HF_API_TOKEN não configurado, usando modo público (rate limit)")
    
    async def generate_stream(
        self,
        prompt: str,
        max_length: int = HuggingFaceConfig.MAX_LENGTH,
        temperature: float = HuggingFaceConfig.TEMPERATURE,
        top_p: float = HuggingFaceConfig.TOP_P
    ) -> AsyncGenerator[str, None]:
        """
        Gerar resposta usando Inference API
        """
        try:
            import httpx
            
            headers = {}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extrair texto gerado
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                    elif isinstance(result, dict):
                        generated_text = result.get("generated_text", "")
                    else:
                        generated_text = str(result)
                    
                    # Simular streaming
                    words = generated_text.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.05)
                
                elif response.status_code == 503:
                    yield "Modelo está carregando, tente novamente em alguns segundos..."
                    logger.warning("⚠️ Modelo ainda carregando")
                
                else:
                    error_msg = f"Erro API: {response.status_code}"
                    logger.error(f"❌ {error_msg}: {response.text}")
                    yield error_msg
        
        except ImportError:
            yield "Erro: httpx não instalado. Execute: pip install httpx"
        except Exception as e:
            logger.error(f"❌ Erro na API: {e}")
            yield f"Erro: {str(e)}"

# ============================================
# FACTORY - CRIAR LLM
# ============================================

def create_llm(
    model_name: Optional[str] = None,
    use_api: Optional[bool] = None
) -> Optional[object]:
    """
    Factory para criar instância de LLM
    
    Args:
        model_name: Nome do modelo (padrão: config)
        use_api: Usar API ou local (padrão: config)
    
    Returns:
        Instância de LLM ou None
    """
    model_name = model_name or HuggingFaceConfig.DEFAULT_MODEL
    use_api = use_api if use_api is not None else HuggingFaceConfig.USE_API
    
    logger.info(f"🤗 Criando LLM: {model_name} (API: {use_api})")
    
    if use_api:
        # Usar Inference API
        llm = HuggingFaceInferenceAPI(model_name)
        logger.info("✅ LLM criado (Inference API)")
        return llm
    else:
        # Usar modelo local
        llm = LocalHuggingFaceLLM(model_name)
        if llm.load():
            logger.info("✅ LLM criado (Local)")
            return llm
        else:
            logger.error("❌ Falha ao carregar modelo local")
            return None

# ============================================
# HELPER - FORMATAR PROMPT
# ============================================

def format_prompt_for_model(
    model_name: str,
    system_prompt: str,
    user_message: str,
    context: Optional[str] = None
) -> str:
    """
    Formatar prompt conforme o modelo
    
    Diferentes modelos têm formatos diferentes:
    - GPT-2: texto simples
    - Llama-2: [INST] ... [/INST]
    - Mistral/Zephyr: <|system|> ... <|user|> ...
    """
    
    # GPT-2 e modelos simples
    if "gpt2" in model_name.lower():
        prompt = f"{system_prompt}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"User: {user_message}\nAssistant:"
        return prompt
    
    # Llama-2 Chat
    elif "llama" in model_name.lower() and "chat" in model_name.lower():
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        if context:
            prompt += f"{context}\n\n"
        prompt += f"{user_message} [/INST]"
        return prompt
    
    # Mistral/Zephyr
    elif "mistral" in model_name.lower() or "zephyr" in model_name.lower():
        prompt = f"<|system|>\n{system_prompt}</s>\n"
        if context:
            prompt += f"<|assistant|>\n{context}</s>\n"
        prompt += f"<|user|>\n{user_message}</s>\n<|assistant|>"
        return prompt
    
    # FLAN-T5 (diferente - encoder-decoder)
    elif "flan" in model_name.lower():
        prompt = f"{system_prompt}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Question: {user_message}\nAnswer:"
        return prompt
    
    # Fallback genérico
    else:
        prompt = f"{system_prompt}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"User: {user_message}\nAssistant:"
        return prompt

# ============================================
# TESTE
# ============================================

async def test_llm():
    """Função de teste"""
    print("="*80)
    print("🧪 TESTANDO HUGGING FACE LLM")
    print("="*80)
    
    # Criar LLM (API por padrão)
    llm = create_llm(model_name="gpt2", use_api=True)
    
    if not llm:
        print("❌ Falha ao criar LLM")
        return
    
    # Testar geração
    prompt = "Explain what is machine learning in simple terms:"
    print(f"\n📝 Prompt: {prompt}\n")
    print("🤖 Resposta: ", end="", flush=True)
    
    async for token in llm.generate_stream(prompt, max_length=100):
        print(token, end="", flush=True)
    
    print("\n\n✅ Teste concluído!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm())
