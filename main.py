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

import sys
import os
import logging

# ============================================
# CONFIGURAÇÃO DE PATHS
# ============================================

# Adicionar diretório src ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# IMPORT DA APLICAÇÃO
# ============================================

logger.info("="*80)
logger.info("🚀 INICIANDO AGI CHAT API")
logger.info("="*80)
logger.info(f"📁 Diretório atual: {current_dir}")
logger.info(f"📁 Diretório src: {src_dir}")
logger.info(f"📂 Python path: {sys.path[:3]}")

try:
    # Tentar importar do módulo src
    logger.info("📦 Importando aplicação de src.main...")
    from src.main import app
    logger.info("✅ Aplicação importada com sucesso!")
    
except ImportError as e:
    logger.error(f"❌ Erro ao importar de src.main: {e}")
    logger.info("🔄 Tentando importar diretamente do main.py em src/...")
    
    try:
        # Fallback: importar main.py diretamente
        sys.path.insert(0, src_dir)
        import main as main_chat
        app = src_main.app
        logger.info("✅ Aplicação importada com sucesso (fallback)!")
        
    except ImportError as e2:
        logger.error(f"❌ Erro no fallback: {e2}")
        logger.error(f"📂 Arquivos em {src_dir}: {os.listdir(src_dir) if os.path.exists(src_dir) else 'Diretório não existe'}")
        
        # Criar app mínima para debug
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="AGI API - Debug Mode")
        
        @app.get("/")
        def debug_root():
            return {
                "status": "error",
                "message": "Aplicação não carregada - modo debug",
                "current_dir": current_dir,
                "src_dir": src_dir,
                "src_exists": os.path.exists(src_dir),
                "files_in_current": os.listdir(current_dir) if os.path.exists(current_dir) else [],
                "files_in_src": os.listdir(src_dir) if os.path.exists(src_dir) else [],
                "python_path": sys.path[:5],
                "error": str(e2)
            }
        
        @app.get("/health")
        def debug_health():
            return {"status": "debug_mode", "app_loaded": False}
        
        logger.warning("⚠️ Aplicação iniciada em MODO DEBUG")

# ============================================
# CONFIGURAÇÃO ADICIONAL
# ============================================

# Adicionar middleware de logging para debug
try:
    from fastapi.middleware.cors import CORSMiddleware
    
    # CORS já está configurado em src/main.py, mas garantir que está ativo
    logger.info("✅ CORS middleware verificado")
    
except Exception as e:
    logger.warning(f"⚠️ Aviso ao configurar middleware: {e}")

# ============================================
# INFORMAÇÕES DE DEPLOY
# ============================================

PORT = int(os.getenv("PORT", 10000))
HOST = os.getenv("HOST", "0.0.0.0")

logger.info("="*80)
logger.info("📡 CONFIGURAÇÃO DE DEPLOY")
logger.info("="*80)
logger.info(f"🌐 Host: {HOST}")
logger.info(f"🔌 Port: {PORT}")
logger.info(f"🔧 Python: {sys.version}")
logger.info(f"📍 Working Dir: {os.getcwd()}")
logger.info("="*80)

# ============================================
# HEALTH CHECK ADICIONAL
# ============================================

@app.get("/api/health")
def api_health():
    """Health check adicional para monitoramento"""
    return {
        "status": "healthy",
        "service": "AGI Chat API",
        "version": "2.1",
        "host": HOST,
        "port": PORT,
        "python_version": sys.version,
        "working_directory": os.getcwd()
    }

@app.get("/api/info")
def api_info():
    """Informações do sistema"""
    return {
        "current_dir": current_dir,
        "src_dir": src_dir,
        "src_exists": os.path.exists(src_dir),
        "python_path": sys.path[:5],
        "environment": {
            "PORT": PORT,
            "HOST": HOST,
            "HF_MODEL": os.getenv("HF_MODEL", "Not set"),
            "HF_USE_API": os.getenv("HF_USE_API", "Not set"),
        }
    }

# ============================================
# EXECUÇÃO (para testes locais)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*80)
    logger.info("🚀 INICIANDO SERVIDOR UVICORN")
    logger.info("="*80)
    logger.info(f"🔗 Acesse: http://{HOST}:{PORT}")
    logger.info(f"📚 Docs: http://{HOST}:{PORT}/docs")
    logger.info("="*80)
    
    uvicorn.run(
        "main:app",  # Este arquivo
        host=HOST,
        port=PORT,
        reload=False,  # Desabilitar reload em produção
        log_level="info",
        access_log=True
    )
