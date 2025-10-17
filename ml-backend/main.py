"""
Backend API para Manutenção Preditiva
NASA CMAPSS + AI4I Datasets
Deploy: Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import joblib
import os

app = FastAPI(
    title="Predictive Maintenance API",
    description="API para previsão de falhas e RUL (Remaining Useful Life)",
    version="2.0.0"
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
# MODELOS DE DADOS
# ============================================

class RULPredictionRequest(BaseModel):
    """Request para predição de RUL (NASA CMAPSS)"""
    sensor_data: List[float]  # 21 sensores + 3 settings
    unit_id: Optional[int] = None
    cycle: Optional[int] = None

class FailurePredictionRequest(BaseModel):
    """Request para predição de falha (AI4I)"""
    air_temperature: float  # [K]
    process_temperature: float  # [K]
    rotational_speed: float  # [rpm]
    torque: float  # [Nm]
    tool_wear: float  # [min]
    product_type: str  # L, M, H

class PredictionResponse(BaseModel):
    """Response padrão"""
    prediction: float
    confidence: float
    classification: str
    recommendation: str
    timestamp: str
    model_type: str

# ============================================
# CLASSE DO MODELO
# ============================================

class PredictiveMaintenanceModel:
    def __init__(self):
        self.cmapss_model = None
        self.cmapss_scaler = None
        self.ai4i_model = None
        self.ai4i_scaler = None
        self.load_models()
    
    def load_models(self):
        """Carrega modelos treinados"""
        print("🔄 Carregando modelos...")
        
        # NASA CMAPSS
        try:
            if os.path.exists('cmapss_model.keras'):
                import tensorflow as tf
                self.cmapss_model = tf.keras.models.load_model('cmapss_model.keras')
                print("✅ Modelo NASA CMAPSS carregado")
            if os.path.exists('cmapss_scaler.pkl'):
                self.cmapss_scaler = joblib.load('cmapss_scaler.pkl')
                print("✅ Scaler CMAPSS carregado")
        except Exception as e:
            print(f"⚠️ CMAPSS não carregado: {e}")
        
        # AI4I
        try:
            if os.path.exists('ai4i_model.pkl'):
                self.ai4i_model = joblib.load('ai4i_model.pkl')
                print("✅ Modelo AI4I carregado")
            if os.path.exists('ai4i_scaler.pkl'):
                self.ai4i_scaler = joblib.load('ai4i_scaler.pkl')
                print("✅ Scaler AI4I carregado")
        except Exception as e:
            print(f"⚠️ AI4I não carregado: {e}")
    
    def predict_rul(self, sensor_data: List[float]) -> Dict:
        """
        Prediz RUL (Remaining Useful Life)
        NASA CMAPSS Dataset
        """
        if self.cmapss_model is None:
            # Simulação se modelo não carregado
            return self._simulate_rul(sensor_data)
        
        try:
            # Preprocessar
            data = np.array(sensor_data).reshape(1, -1)
            
            if self.cmapss_scaler:
                data = self.cmapss_scaler.transform(data)
            
            # Predizer
            # Nota: CNN-RNN precisa de sequências, aqui simplificamos
            rul = float(self.cmapss_model.predict(data, verbose=0)[0][0])
            rul = max(0, rul)  # RUL não pode ser negativo
            
            # Classificação de risco
            if rul < 30:
                classification = "🔴 CRÍTICO"
                recommendation = "Manutenção urgente necessária! Motor próximo da falha."
            elif rul < 60:
                classification = "🟠 ATENÇÃO"
                recommendation = "Agendar manutenção preventiva em breve."
            elif rul < 100:
                classification = "🟡 MONITORAR"
                recommendation = "Continuar monitoramento regular."
            else:
                classification = "🟢 NORMAL"
                recommendation = "Motor operando dentro do esperado."
            
            # Confiança baseada em RUL
            confidence = min(0.95, max(0.60, 1 - (rul / 200)))
            
            return {
                "prediction": round(rul, 2),
                "confidence": round(confidence, 4),
                "classification": classification,
                "recommendation": recommendation,
                "unit": "ciclos restantes"
            }
        
        except Exception as e:
            raise Exception(f"Erro na predição RUL: {str(e)}")
    
    def predict_failure(self, features: Dict) -> Dict:
        """
        Prediz falha de máquina
        AI4I Dataset
        """
        if self.ai4i_model is None:
            # Simulação se modelo não carregado
            return self._simulate_failure(features)
        
        try:
            # Converter Type
            type_map = {'L': 0, 'M': 1, 'H': 2}
            product_type = type_map.get(features['product_type'], 1)
            
            # Criar array de features
            data = np.array([[
                product_type,
                features['air_temperature'],
                features['process_temperature'],
                features['rotational_speed'],
                features['torque'],
                features['tool_wear']
            ]])
            
            # Normalizar
            if self.ai4i_scaler:
                data = self.ai4i_scaler.transform(data)
            
            # Predizer
            prediction = self.ai4i_model.predict(data)[0]
            probability = self.ai4i_model.predict_proba(data)[0][1]
            
            # Classificação
            if probability >= 0.7:
                classification = "🔴 ALTO RISCO DE FALHA"
                recommendation = "Interromper operação e realizar manutenção imediata!"
            elif probability >= 0.4:
                classification = "🟠 RISCO MODERADO"
                recommendation = "Reduzir carga e agendar inspeção."
            elif probability >= 0.2:
                classification = "🟡 RISCO BAIXO"
                recommendation = "Monitorar parâmetros operacionais."
            else:
                classification = "🟢 OPERAÇÃO NORMAL"
                recommendation = "Máquina operando dentro dos padrões."
            
            return {
                "prediction": int(prediction),
                "confidence": round(float(probability), 4),
                "classification": classification,
                "recommendation": recommendation,
                "failure_probability": f"{probability*100:.2f}%"
            }
        
        except Exception as e:
            raise Exception(f"Erro na predição de falha: {str(e)}")
    
    def _simulate_rul(self, sensor_data: List[float]) -> Dict:
        """Simulação de RUL para desenvolvimento"""
        avg_value = np.mean(sensor_data)
        rul = max(0, 150 - (avg_value * 50))
        
        if rul < 30:
            classification = "🔴 CRÍTICO"
            recommendation = "Manutenção urgente necessária!"
        elif rul < 60:
            classification = "🟠 ATENÇÃO"
            recommendation = "Agendar manutenção preventiva."
        else:
            classification = "🟢 NORMAL"
            recommendation = "Motor operando normalmente."
        
        return {
            "prediction": round(rul, 2),
            "confidence": 0.75,
            "classification": classification,
            "recommendation": recommendation,
            "unit": "ciclos restantes",
            "mode": "simulation"
        }
    
    def _simulate_failure(self, features: Dict) -> Dict:
        """Simulação de falha para desenvolvimento"""
        # Simular baseado em thresholds
        risk_score = 0
        
        if features['torque'] > 50:
            risk_score += 0.3
        if features['tool_wear'] > 150:
            risk_score += 0.3
        if features['rotational_speed'] < 1500:
            risk_score += 0.2
        if features['process_temperature'] > 320:
            risk_score += 0.2
        
        probability = min(0.95, risk_score)
        
        if probability >= 0.6:
            classification = "🔴 ALTO RISCO"
            recommendation = "Manutenção imediata!"
        elif probability >= 0.3:
            classification = "🟠 RISCO MODERADO"
            recommendation = "Agendar inspeção."
        else:
            classification = "🟢 OPERAÇÃO NORMAL"
            recommendation = "Máquina OK."
        
        return {
            "prediction": 1 if probability > 0.5 else 0,
            "confidence": probability,
            "classification": classification,
            "recommendation": recommendation,
            "failure_probability": f"{probability*100:.2f}%",
            "mode": "simulation"
        }

# Inicializar modelo
ml_model = PredictiveMaintenanceModel()

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Endpoint raiz"""
    return {
        "message": "🏭 Predictive Maintenance API",
        "version": "2.0.0",
        "datasets": ["NASA CMAPSS", "AI4I 2020"],
        "models": {
            "cmapss": "loaded" if ml_model.cmapss_model else "simulation",
            "ai4i": "loaded" if ml_model.ai4i_model else "simulation"
        },
        "endpoints": {
            "rul_prediction": "/predict/rul",
            "failure_prediction": "/predict/failure",
            "health": "/health",
            "info": "/model/info"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "cmapss": "loaded" if ml_model.cmapss_model else "simulation",
            "ai4i": "loaded" if ml_model.ai4i_model else "simulation"
        }
    }

@app.get("/model/info")
def model_info():
    """Informações dos modelos"""
    return {
        "nasa_cmapss": {
            "description": "Predição de RUL (Remaining Useful Life) para motores turbofan",
            "input": "21 sensores + 3 configurações operacionais",
            "output": "Ciclos restantes até falha",
            "status": "loaded" if ml_model.cmapss_model else "simulation",
            "architecture": "CNN-RNN Hybrid"
        },
        "ai4i": {
            "description": "Predição de falhas em máquinas industriais",
            "input": "Temperatura, velocidade, torque, desgaste",
            "output": "Probabilidade de falha (0-1)",
            "status": "loaded" if ml_model.ai4i_model else "simulation",
            "architecture": "Random Forest Classifier"
        },
        "deployment": "Render",
        "last_updated": "2025-10-16"
    }

@app.post("/predict/rul", response_model=PredictionResponse)
def predict_rul(request: RULPredictionRequest):
    """
    Predição de RUL (Remaining Useful Life)
    NASA CMAPSS Dataset
    
    Exemplo:
    {
        "sensor_data": [25.5, 100.2, 60.1, ...],  // 21-24 valores
        "unit_id": 1,
        "cycle": 150
    }
    """
    try:
        if not request.sensor_data:
            raise HTTPException(
                status_code=400,
                detail="sensor_data não pode estar vazio"
            )
        
        if len(request.sensor_data) < 20:
            raise HTTPException(
                status_code=400,
                detail="Mínimo de 20 sensores esperados"
            )
        
        # Predizer
        result = ml_model.predict_rul(request.sensor_data)
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            classification=result["classification"],
            recommendation=result["recommendation"],
            timestamp=datetime.now().isoformat(),
            model_type="NASA CMAPSS RUL"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {str(e)}"
        )

@app.post("/predict/failure", response_model=PredictionResponse)
def predict_failure(request: FailurePredictionRequest):
    """
    Predição de Falha de Máquina
    AI4I Dataset
    
    Exemplo:
    {
        "air_temperature": 298.5,
        "process_temperature": 308.2,
        "rotational_speed": 1500,
        "torque": 45.3,
        "tool_wear": 120,
        "product_type": "M"
    }
    """
    try:
        # Validações
        if request.product_type not in ['L', 'M', 'H']:
            raise HTTPException(
                status_code=400,
                detail="product_type deve ser L, M ou H"
            )
        
        if request.air_temperature < 250 or request.air_temperature > 350:
            raise HTTPException(
                status_code=400,
                detail="air_temperature fora do range válido (250-350K)"
            )
        
        # Predizer
        features = {
            "air_temperature": request.air_temperature,
            "process_temperature": request.process_temperature,
            "rotational_speed": request.rotational_speed,
            "torque": request.torque,
            "tool_wear": request.tool_wear,
            "product_type": request.product_type
        }
        
        result = ml_model.predict_failure(features)
        
        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            classification=result["classification"],
            recommendation=result["recommendation"],
            timestamp=datetime.now().isoformat(),
            model_type="AI4I Failure Detection"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {str(e)}"
        )

@app.get("/stats")
def get_statistics():
    """Estatísticas da API"""
    return {
        "total_predictions": "N/A",
        "models_loaded": {
            "cmapss": ml_model.cmapss_model is not None,
            "ai4i": ml_model.ai4i_model is not None
        },
        "uptime": "online",
        "deployment": "Render"
    }

@app.get("/examples")
def get_examples():
    """Exemplos de uso"""
    return {
        "rul_prediction": {
            "description": "Predizer vida útil restante de motor turbofan",
            "endpoint": "/predict/rul",
            "method": "POST",
            "example": {
                "sensor_data": [
                    518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61,
                    554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66,
                    2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0,
                    38.86, 23.4190
                ],
                "unit_id": 1,
                "cycle": 100
            },
            "response": {
                "prediction": 85.5,
                "confidence": 0.87,
                "classification": "🟡 MONITORAR",
                "recommendation": "Continuar monitoramento regular.",
                "model_type": "NASA CMAPSS RUL"
            }
        },
        "failure_prediction": {
            "description": "Predizer falha em máquina industrial",
            "endpoint": "/predict/failure",
            "method": "POST",
            "example": {
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0,
                "product_type": "M"
            },
            "response": {
                "prediction": 0,
                "confidence": 0.12,
                "classification": "🟢 OPERAÇÃO NORMAL",
                "recommendation": "Máquina operando dentro dos padrões.",
                "model_type": "AI4I Failure Detection"
            }
        }
    }

# ============================================
# PARA DEPLOY NO RENDER
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)