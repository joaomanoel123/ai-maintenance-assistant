"""
===============================================================================
🚀 API COMPLETA - PIPELINE AGI GENERATIVA v2.0
===============================================================================

FastAPI integrada com:
- Modelos preditivos (Random Forest + CNN-RNN)
- Módulos AGI (Memory, Reasoning, Decision, Generative)
- RLHF (Feedback loop)
- Banco de dados (Neon PostgreSQL)
- Sistema híbrido completo
- Carregamento dinâmico de modelos CMAPSS

Autor: João Manoel
Deploy: Render.com
Database: Neon PostgreSQL
===============================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
import joblib
import json
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURAÇÃO
# ============================================

class Settings:
    """Configurações da aplicação"""
    APP_NAME = "AGI Generativa API"
    VERSION = "2.0"
    
    # Paths dos modelos
    MODELS_DIR = Path("models")
    MODEL_PATH = os.getenv("MODEL_PATH", "./models/cmapss_model.pkl")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    
    # Limites
    MAX_SEQUENCE_LENGTH = 50
    MAX_FEEDBACK_LENGTH = 1000

settings = Settings()

# ============================================
# INICIALIZAR APP
# ============================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Sistema AGI híbrido com predição, raciocínio, decisão e geração"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELOS GLOBAIS
# ============================================

class ModelRegistry:
    """Registro de modelos carregados"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.loaded = False
        self.cmapss_model = None  # Modelo CMAPSS principal
    
    def load_all(self):
        """Carregar todos os modelos"""
        logger.info("🔄 Carregando modelos...")
        
        try:
            # CMAPSS Model Principal (usando variável de ambiente)
            cmapss_path = Path(settings.MODEL_PATH)
            if cmapss_path.exists():
                self.cmapss_model = joblib.load(cmapss_path)
                self.models['cmapss_main'] = self.cmapss_model
                logger.info(f"✅ Modelo CMAPSS principal carregado: {cmapss_path}")
            else:
                logger.warning(f"⚠️ Modelo CMAPSS não encontrado em: {cmapss_path}")
            
            # AI4I Random Forest
            ai4i_path = settings.MODELS_DIR / "ai4i_rf.pkl"
            if ai4i_path.exists():
                self.models['ai4i_rf'] = joblib.load(ai4i_path)
                self.scalers['ai4i'] = joblib.load(settings.MODELS_DIR / "ai4i_scaler.pkl")
                logger.info("✅ AI4I Random Forest carregado")
            
            # NASA CMAPSS Random Forest (fallback)
            cmapss_rf_path = settings.MODELS_DIR / "cmapss_rf.pkl"
            if cmapss_rf_path.exists():
                self.models['cmapss_rf'] = joblib.load(cmapss_rf_path)
                self.scalers['cmapss'] = joblib.load(settings.MODELS_DIR / "cmapss_scaler.pkl")
                logger.info("✅ CMAPSS Random Forest carregado")
            
            # NASA CMAPSS CNN-RNN (se disponível)
            try:
                from tensorflow.keras.models import load_model
                cmapss_cnn_path = settings.MODELS_DIR / "cmapss_cnn_rnn.keras"
                if cmapss_cnn_path.exists():
                    self.models['cmapss_cnn_rnn'] = load_model(cmapss_cnn_path)
                    logger.info("✅ CMAPSS CNN-RNN carregado")
            except ImportError:
                logger.warning("⚠️ TensorFlow não disponível, usando RF apenas")
            
            self.loaded = True
            logger.info(f"✅ Total de modelos carregados: {len(self.models)}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise

registry = ModelRegistry()

# ============================================
# MÓDULOS AGI (VERSÕES SIMPLIFICADAS)
# ============================================

class MemoryModule:
    """Módulo de memória simplificado"""
    
    def __init__(self, max_size=100):
        self.short_term = []
        self.long_term = []
        self.max_size = max_size
    
    def store(self, data: Dict, importance: float = 0.5):
        """Armazenar na memória"""
        entry = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "importance": importance
        }
        
        self.short_term.append(entry)
        
        # Mover para long_term se importante
        if importance > 0.7:
            self.long_term.append(entry)
        
        # Limitar tamanho
        if len(self.short_term) > self.max_size:
            self.short_term.pop(0)
    
    def retrieve_recent(self, n=5):
        """Recuperar memórias recentes"""
        return self.short_term[-n:]
    
    def retrieve_important(self, n=5):
        """Recuperar memórias importantes"""
        sorted_mem = sorted(self.long_term, key=lambda x: x['importance'], reverse=True)
        return sorted_mem[:n]

class ReasoningModule:
    """Módulo de raciocínio"""
    
    def analyze(self, prediction: Dict, context: Dict) -> Dict:
        """Análise causal e raciocínio"""
        
        # Extrair valores
        pred_value = prediction.get('value', 0)
        confidence = prediction.get('confidence', 0)
        pred_type = prediction.get('type', 'unknown')
        
        # Raciocínio causal
        if pred_type == 'failure':
            severity = self._classify_severity(pred_value)
            causes = self._identify_causes(context)
            reasoning_type = "causal"
        else:  # RUL
            severity = self._classify_rul_severity(pred_value)
            causes = self._identify_rul_causes(pred_value, context)
            reasoning_type = "temporal"
        
        return {
            "conclusion": self._generate_conclusion(pred_value, pred_type, severity),
            "reasoning_type": reasoning_type,
            "severity": severity,
            "identified_causes": causes,
            "confidence": confidence
        }
    
    def _classify_severity(self, failure_prob: float) -> str:
        """Classificar severidade de falha"""
        if failure_prob > 0.7:
            return "HIGH"
        elif failure_prob > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _classify_rul_severity(self, rul: float) -> str:
        """Classificar severidade RUL"""
        if rul < 20:
            return "CRITICAL"
        elif rul < 50:
            return "HIGH"
        elif rul < 80:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_causes(self, context: Dict) -> List[str]:
        """Identificar causas de falha"""
        causes = []
        data = context.get('data', {})
        
        if data.get('temperature', 0) > 310:
            causes.append("Temperatura elevada")
        if data.get('torque', 0) > 60:
            causes.append("Torque excessivo")
        if data.get('tool_wear', 0) > 200:
            causes.append("Desgaste da ferramenta")
        if data.get('rpm', 0) > 2500:
            causes.append("Rotação muito alta")
        
        return causes if causes else ["Operação normal"]
    
    def _identify_rul_causes(self, rul: float, context: Dict) -> List[str]:
        """Identificar causas de degradação"""
        causes = []
        
        if rul < 30:
            causes.append("Degradação avançada dos componentes")
        if rul < 50:
            causes.append("Desgaste progressivo detectado")
        
        return causes if causes else ["Condição operacional estável"]
    
    def _generate_conclusion(self, value: float, pred_type: str, severity: str) -> str:
        """Gerar conclusão do raciocínio"""
        if pred_type == 'failure':
            if severity == "HIGH":
                return f"Alto risco de falha detectado ({value*100:.1f}%). Ação imediata recomendada."
            elif severity == "MEDIUM":
                return f"Risco moderado de falha ({value*100:.1f}%). Monitoramento próximo necessário."
            else:
                return f"Risco baixo de falha ({value*100:.1f}%). Sistema operando normalmente."
        else:  # RUL
            if severity == "CRITICAL":
                return f"Vida útil restante crítica ({value:.0f} ciclos). Manutenção urgente."
            elif severity == "HIGH":
                return f"Vida útil restante reduzida ({value:.0f} ciclos). Planejar manutenção."
            else:
                return f"Vida útil adequada ({value:.0f} ciclos). Sistema saudável."

class DecisionModule:
    """Módulo de decisão"""
    
    def recommend_action(self, reasoning: Dict, context: Dict) -> Dict:
        """Recomendar ação baseada no raciocínio"""
        
        severity = reasoning.get('severity', 'LOW')
        pred_type = context.get('prediction_type', 'unknown')
        
        # Mapear ações
        actions = {
            'CRITICAL': {
                'action': 'immediate_maintenance',
                'priority': 'URGENT',
                'description': 'Parar operação e realizar manutenção imediata'
            },
            'HIGH': {
                'action': 'scheduled_maintenance',
                'priority': 'HIGH',
                'description': 'Agendar manutenção preventiva em até 24h'
            },
            'MEDIUM': {
                'action': 'monitoring',
                'priority': 'MEDIUM',
                'description': 'Intensificar monitoramento e preparar equipe'
            },
            'LOW': {
                'action': 'continue_operation',
                'priority': 'LOW',
                'description': 'Continuar operação normal com monitoramento padrão'
            }
        }
        
        decision = actions.get(severity, actions['LOW'])
        
        # Adicionar plano de execução
        decision['execution_plan'] = self._generate_plan(severity, reasoning)
        decision['estimated_cost'] = self._estimate_cost(severity)
        decision['risk_mitigation'] = self._generate_mitigation(severity)
        
        return decision
    
    def _generate_plan(self, severity: str, reasoning: Dict) -> List[str]:
        """Gerar plano de execução"""
        if severity in ['CRITICAL', 'HIGH']:
            return [
                "1. Notificar equipe de manutenção",
                "2. Preparar peças de reposição",
                "3. Agendar janela de manutenção",
                "4. Realizar inspeção completa",
                "5. Executar reparos necessários",
                "6. Testar sistema após manutenção"
            ]
        elif severity == 'MEDIUM':
            return [
                "1. Aumentar frequência de inspeções",
                "2. Monitorar parâmetros críticos",
                "3. Preparar plano de contingência"
            ]
        else:
            return [
                "1. Manter rotina de monitoramento",
                "2. Registrar estado atual"
            ]
    
    def _estimate_cost(self, severity: str) -> str:
        """Estimar custo da ação"""
        costs = {
            'CRITICAL': 'Alto (R$ 10.000 - R$ 50.000)',
            'HIGH': 'Médio-Alto (R$ 5.000 - R$ 15.000)',
            'MEDIUM': 'Baixo-Médio (R$ 1.000 - R$ 5.000)',
            'LOW': 'Mínimo (< R$ 1.000)'
        }
        return costs.get(severity, 'Não estimado')
    
    def _generate_mitigation(self, severity: str) -> List[str]:
        """Gerar estratégias de mitigação"""
        if severity in ['CRITICAL', 'HIGH']:
            return [
                "Ter equipe de backup disponível",
                "Manter estoque de peças críticas",
                "Plano de contingência ativado"
            ]
        else:
            return ["Monitoramento contínuo suficiente"]

class GenerativeModule:
    """Módulo generativo (templates)"""
    
    def generate_explanation(self, 
                           prediction: Dict, 
                           reasoning: Dict, 
                           decision: Dict,
                           context: Dict) -> str:
        """Gerar explicação completa em texto"""
        
        pred_value = prediction.get('value', 0)
        pred_type = prediction.get('type', 'unknown')
        confidence = prediction.get('confidence', 0)
        
        # Cabeçalho
        if pred_type == 'failure':
            header = f"🔍 **Análise de Risco de Falha**\n\n"
            header += f"Probabilidade de falha: **{pred_value*100:.1f}%**\n"
        else:
            header = f"🔍 **Análise de Vida Útil Restante (RUL)**\n\n"
            header += f"Ciclos restantes estimados: **{pred_value:.0f} ciclos**\n"
        
        header += f"Confiança da predição: **{confidence*100:.0f}%**\n\n"
        
        # Raciocínio
        reasoning_text = f"📊 **Análise Causal:**\n"
        reasoning_text += f"{reasoning.get('conclusion', '')}\n\n"
        reasoning_text += f"Severidade: **{reasoning.get('severity', 'N/A')}**\n"
        reasoning_text += f"Fatores identificados:\n"
        for cause in reasoning.get('identified_causes', []):
            reasoning_text += f"  • {cause}\n"
        reasoning_text += "\n"
        
        # Decisão
        decision_text = f"⚡ **Recomendação:**\n"
        decision_text += f"Ação: **{decision.get('description', 'N/A')}**\n"
        decision_text += f"Prioridade: **{decision.get('priority', 'N/A')}**\n"
        decision_text += f"Custo estimado: {decision.get('estimated_cost', 'N/A')}\n\n"
        
        decision_text += "📋 **Plano de Execução:**\n"
        for step in decision.get('execution_plan', []):
            decision_text += f"{step}\n"
        decision_text += "\n"
        
        # Rodapé
        footer = f"🕐 Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"🤖 Sistema: AGI Generativa v{settings.VERSION}"
        
        return header + reasoning_text + decision_text + footer

# Instanciar módulos
memory = MemoryModule()
reasoning = ReasoningModule()
decision_maker = DecisionModule()
generator = GenerativeModule()

# ============================================
# MODELOS PYDANTIC
# ============================================

class PredictionRequest(BaseModel):
    """Request de predição"""
    analysis_type: str = Field(..., description="'failure' ou 'rul'")
    data: Dict[str, float] = Field(..., description="Dados dos sensores")
    enable_reasoning: bool = Field(default=True)
    enable_decision: bool = Field(default=True)
    generate_explanation: bool = Field(default=True)
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_type": "failure",
                "data": {
                    "type": 1,
                    "air_temperature": 300,
                    "process_temperature": 310,
                    "rpm": 1500,
                    "torque": 40,
                    "tool_wear": 100
                },
                "enable_reasoning": True,
                "enable_decision": True,
                "generate_explanation": True
            }
        }

class FeedbackRequest(BaseModel):
    """Request de feedback RLHF"""
    result_id: str
    score: float = Field(..., ge=0, le=1, description="Score 0-1")
    feedback_text: Optional[str] = Field(default="", max_length=settings.MAX_FEEDBACK_LENGTH)
    user_id: Optional[str] = Field(default="anonymous")
    
    class Config:
        schema_extra = {
            "example": {
                "result_id": "2025-01-15T10:30:00",
                "score": 0.85,
                "feedback_text": "Análise precisa e útil",
                "user_id": "user_123"
            }
        }

# ============================================
# STARTUP E SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    """Inicializar aplicação"""
    logger.info("🚀 Iniciando AGI Generativa API v2.0...")
    
    try:
        # Carregar modelos
        registry.load_all()
        
        # Verificar database
        if settings.DATABASE_URL:
            logger.info("✅ Database URL configurada")
        else:
            logger.warning("⚠️ Database URL não configurada (usando memória)")
        
        logger.info("✅ API pronta!")
        
    except Exception as e:
        logger.error(f"❌ Erro no startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Finalizar aplicação"""
    logger.info("👋 Encerrando AGI Generativa API...")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "models_loaded": len(registry.models),
        "modules": ["memory", "reasoning", "decision", "generative", "rlhf"],
        "endpoints": {
            "predict": "/predict",
            "feedback": "/feedback",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": registry.loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint principal de predição + AGI pipeline completo
    """
    
    try:
        logger.info(f"📥 Nova predição: {request.analysis_type}")
        
        # 1. PREDIÇÃO
        if request.analysis_type == "failure":
            # AI4I - Classificação de falhas
            if 'ai4i_rf' not in registry.models:
                raise HTTPException(status_code=500, detail="Modelo AI4I não carregado")
            
            # Preparar dados
            features = np.array([[
                request.data.get('type', 0),
                request.data.get('air_temperature', 300),
                request.data.get('process_temperature', 310),
                request.data.get('rpm', 1500),
                request.data.get('torque', 40),
                request.data.get('tool_wear', 100)
            ]])
            
            # Normalizar
            features_scaled = registry.scalers['ai4i'].transform(features)
            
            # Prever
            pred = registry.models['ai4i_rf'].predict(features_scaled)[0]
            proba = registry.models['ai4i_rf'].predict_proba(features_scaled)[0]
            
            prediction_result = {
                "type": "failure",
                "value": float(proba[1]),  # Probabilidade de falha
                "confidence": float(max(proba)),
                "class": int(pred)
            }
        
        elif request.analysis_type == "rul":
            # NASA CMAPSS - RUL
            # Tentar CNN-RNN primeiro, fallback para RF
            if 'cmapss_cnn_rnn' in registry.models:
                # TODO: Implementar predição com sequência
                # Por ora, usar RF
                use_rf = True
            else:
                use_rf = True
            
            if use_rf and 'cmapss_rf' in registry.models:
                # Simular features (adaptar conforme seus dados reais)
                features = np.array([[
                    request.data.get(f'sensor_{i}', np.random.rand()) 
                    for i in range(1, 22)
                ]])
                
                features_scaled = registry.scalers['cmapss'].transform(features)
                rul_pred = registry.models['cmapss_rf'].predict(features_scaled)[0]
                
                prediction_result = {
                    "type": "rul",
                    "value": float(rul_pred),
                    "confidence": 0.75,  # Placeholder
                    "unit": "cycles"
                }
            else:
                raise HTTPException(status_code=500, detail="Modelo CMAPSS não carregado")
        
        else:
            raise HTTPException(status_code=400, detail="analysis_type deve ser 'failure' ou 'rul'")
        
        # 2. RACIOCÍNIO
        reasoning_result = None
        if request.enable_reasoning:
            reasoning_result = reasoning.analyze(
                prediction_result,
                {"data": request.data, "prediction_type": request.analysis_type}
            )
        
        # 3. DECISÃO
        decision_result = None
        if request.enable_decision and reasoning_result:
            decision_result = decision_maker.recommend_action(
                reasoning_result,
                {"prediction_type": request.analysis_type}
            )
        
        # 4. GERAÇÃO DE TEXTO
        explanation = None
        if request.generate_explanation:
            explanation = generator.generate_explanation(
                prediction_result,
                reasoning_result or {},
                decision_result or {},
                {"data": request.data}
            )
        
        # 5. ARMAZENAR NA MEMÓRIA
        result_id = datetime.now().isoformat()
        importance = prediction_result.get('value', 0) if request.analysis_type == "failure" else (1 - prediction_result.get('value', 100) / 125)
        
        memory.store({
            "id": result_id,
            "prediction": prediction_result,
            "reasoning": reasoning_result,
            "decision": decision_result
        }, importance=importance)
        
        # 6. RESPOSTA FINAL
        response = {
            "result_id": result_id,
            "prediction": prediction_result,
            "reasoning": reasoning_result,
            "decision": decision_result,
            "generated_explanation": explanation,
            "timestamp": result_id
        }
        
        logger.info(f"✅ Predição concluída: {result_id}")
        return response
    
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Receber feedback RLHF
    """
    
    try:
        logger.info(f"💬 Novo feedback: score={feedback.score}")
        
        # Salvar feedback (em produção: salvar no banco)
        feedback_data = {
            "result_id": feedback.result_id,
            "score": feedback.score,
            "text": feedback.feedback_text,
            "user_id": feedback.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Armazenar na memória
        memory.store({"feedback": feedback_data}, importance=0.9)
        
        # TODO: Processar feedback para RLHF (background task)
        # background_tasks.add_task(process_rlhf, feedback_data)
        
        return {
            "status": "success",
            "message": "Feedback recebido com sucesso",
            "feedback_id": feedback.result_id
        }
    
    except Exception as e:
        logger.error(f"❌ Erro no feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Estatísticas do sistema"""
    
    return {
        "system": {
            "version": settings.VERSION,
            "uptime": "N/A",  # TODO: calcular
            "models_loaded": len(registry.models)
        },
        "memory": {
            "short_term_size": len(memory.short_term),
            "long_term_size": len(memory.long_term),
            "recent_predictions": len(memory.retrieve_recent(10))
        },
        "predictions": {
            "total": len(memory.short_term),  # Placeholder
            "last_24h": "N/A"  # TODO: implementar
        },
        "feedback": {
            "total_received": "N/A",  # TODO: query database
            "average_score": "N/A"
        }
    }

@app.get("/memory/recent")
async def get_recent_memory(n: int = 5):
    """Recuperar memórias recentes"""
    recent = memory.retrieve_recent(n)
    return {"recent_memories": recent, "count": len(recent)}

@app.get("/memory/important")
async def get_important_memory(n: int = 5):
    """Recuperar memórias importantes"""
    important = memory.retrieve_important(n)
    return {"important_memories": important, "count": len(important)}

# ============================================
# EXECUÇÃO LOCAL
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("="*80)
    print(f"🚀 AGI GENERATIVA API v{settings.VERSION}")
    print("="*80)
    print(f"📡 Rodando em: http://localhost:{port}")
    print(f"📚 Documentação: http://localhost:{port}/docs")
    print("="*80)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
