# main.py - API Principal com Carregamento de Modelos
# Autor: João Manoel | Versão: 2.0

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forçar CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import jwt
from passlib.context import CryptContext
import numpy as np
import pandas as pd
import pickle
import joblib

# Tentar importar TensorFlow (pode não estar instalado)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow não disponível - usando modelos sklearn apenas")

# ==================== CONFIGURAÇÕES ====================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Caminhos dos modelos
MODEL_PATH_PKL = os.getenv("MODEL_PATH", "./ml-models/cmapss_model.pkl")
MODEL_PATH_H5 = os.getenv("MODEL_PATH_H5", "./ml-models/predictive_maintenance.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "./ml-models/scaler.pkl")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Sistema AGI Preditivo",
    version="2.0.0",
    description="Sistema de IA com múltiplos modelos de Machine Learning"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== VARIÁVEIS GLOBAIS DE MODELOS ====================
models_loaded = {
    'sklearn': None,
    'tensorflow': None,
    'scaler': None
}

model_metadata = {
    'sklearn': {'name': 'CMAPSS Model', 'version': 'v1.0', 'loaded': False},
    'tensorflow': {'name': 'CNN-RNN Hybrid', 'version': 'v2.0', 'loaded': False},
    'scaler': {'name': 'StandardScaler', 'loaded': False}
}

# ==================== SCHEMAS ====================
class PredictRequest(BaseModel):
    """Request para predição"""
    temperature: float
    pressure: float
    vibration: float
    rotational_speed: Optional[float] = 1500
    torque: Optional[float] = 40
    tool_wear: Optional[float] = 50

class PredictResponse(BaseModel):
    """Response da predição"""
    prediction: str
    confidence: float
    model_used: str
    timestamp: datetime
    details: Optional[Dict] = None

class ModelInfo(BaseModel):
    """Informações do modelo"""
    name: str
    version: str
    loaded: bool
    type: str

# ==================== CARREGAMENTO DE MODELOS ====================
@app.on_event("startup")
async def load_models():
    """Carrega todos os modelos disponíveis no startup"""
    global models_loaded, model_metadata
    
    print("\n" + "="*70)
    print("🚀 CARREGANDO MODELOS DE MACHINE LEARNING")
    print("="*70 + "\n")
    
    # 1. Carregar Modelo Sklearn/Joblib
    try:
        if os.path.exists(MODEL_PATH_PKL):
            models_loaded['sklearn'] = joblib.load(MODEL_PATH_PKL)
            model_metadata['sklearn']['loaded'] = True
            print(f"✅ Modelo Sklearn carregado: {MODEL_PATH_PKL}")
        else:
            print(f"⚠️  Modelo Sklearn não encontrado: {MODEL_PATH_PKL}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo Sklearn: {e}")
    
    # 2. Carregar Modelo TensorFlow/Keras
    if TF_AVAILABLE:
        try:
            if os.path.exists(MODEL_PATH_H5):
                models_loaded['tensorflow'] = tf.keras.models.load_model(MODEL_PATH_H5)
                model_metadata['tensorflow']['loaded'] = True
                print(f"✅ Modelo TensorFlow carregado: {MODEL_PATH_H5}")
            else:
                print(f"⚠️  Modelo TensorFlow não encontrado: {MODEL_PATH_H5}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo TensorFlow: {e}")
    else:
        print("ℹ️  TensorFlow não disponível - pulando")
    
    # 3. Carregar Scaler
    try:
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                models_loaded['scaler'] = pickle.load(f)
            model_metadata['scaler']['loaded'] = True
            print(f"✅ Scaler carregado: {SCALER_PATH}")
        else:
            print(f"⚠️  Scaler não encontrado: {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Erro ao carregar scaler: {e}")
    
    # Resumo
    print("\n" + "="*70)
    print("📊 RESUMO DO CARREGAMENTO")
    print("="*70)
    models_loaded_count = sum(1 for v in models_loaded.values() if v is not None)
    print(f"✅ Total de modelos carregados: {models_loaded_count}/3")
    
    for key, meta in model_metadata.items():
        status_icon = "✅" if meta['loaded'] else "❌"
        print(f"{status_icon} {meta['name']}: {'Carregado' if meta['loaded'] else 'Não carregado'}")
    
    print("="*70 + "\n")
    
    if models_loaded_count == 0:
        print("⚠️  ATENÇÃO: Nenhum modelo carregado! API funcionará em modo limitado.")
    else:
        print("🎉 API pronta para realizar predições!\n")

# ==================== FUNÇÕES DE PREDIÇÃO ====================
def predict_with_sklearn(features: np.ndarray) -> Dict:
    """Predição usando modelo Sklearn"""
    if models_loaded['sklearn'] is None:
        raise HTTPException(status_code=503, detail="Modelo Sklearn não disponível")
    
    try:
        prediction = models_loaded['sklearn'].predict(features)[0]
        
        # Tentar obter probabilidades se disponível
        confidence = 0.85  # Default
        if hasattr(models_loaded['sklearn'], 'predict_proba'):
            proba = models_loaded['sklearn'].predict_proba(features)[0]
            confidence = float(max(proba))
        
        return {
            'prediction': 'anomaly' if prediction == 1 else 'normal',
            'confidence': confidence,
            'model_used': 'sklearn',
            'raw_prediction': int(prediction)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição Sklearn: {str(e)}")

def predict_with_tensorflow(features: np.ndarray) -> Dict:
    """Predição usando modelo TensorFlow"""
    if models_loaded['tensorflow'] is None:
        raise HTTPException(status_code=503, detail="Modelo TensorFlow não disponível")
    
    try:
        # Reshape para CNN-RNN (batch, timesteps, features)
        features_reshaped = features.reshape(1, -1, 1)
        
        prediction_prob = models_loaded['tensorflow'].predict(features_reshaped, verbose=0)[0][0]
        prediction = 'anomaly' if prediction_prob > 0.5 else 'normal'
        confidence = float(prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'model_used': 'tensorflow',
            'probability_anomaly': float(prediction_prob),
            'probability_normal': float(1 - prediction_prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição TensorFlow: {str(e)}")

def preprocess_input(data: PredictRequest) -> np.ndarray:
    """Pré-processa os dados de entrada"""
    features = np.array([[
        data.temperature,
        data.pressure,
        data.vibration,
        data.rotational_speed,
        data.torque,
        data.tool_wear
    ]])
    
    # Aplicar scaler se disponível
    if models_loaded['scaler'] is not None:
        features = models_loaded['scaler'].transform(features)
    
    return features

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint raiz com status da API"""
    models_status = {
        key: meta['loaded'] 
        for key, meta in model_metadata.items()
    }
    
    return {
        "message": "Sistema AGI Preditivo v2.0",
        "status": "operational",
        "author": "João Manoel",
        "models_loaded": models_status,
        "total_models": sum(models_status.values()),
        "endpoints": {
            "predict": "/predict",
            "models": "/models",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.head("/")
async def head_root():
    """Health check para Render"""
    return {}

@app.get("/health")
async def health_check():
    """Verifica saúde da API e modelos"""
    models_status = []
    
    for key, meta in model_metadata.items():
        models_status.append({
            "name": meta['name'],
            "version": meta.get('version', 'N/A'),
            "loaded": meta['loaded'],
            "type": key
        })
    
    all_loaded = all(meta['loaded'] for meta in model_metadata.values())
    
    return {
        "status": "healthy" if any(meta['loaded'] for meta in model_metadata.values()) else "degraded",
        "timestamp": datetime.utcnow(),
        "models": models_status,
        "all_models_loaded": all_loaded
    }

@app.get("/models", response_model=List[ModelInfo])
async def get_models_info():
    """Lista todos os modelos disponíveis"""
    models = []
    
    for key, meta in model_metadata.items():
        models.append(ModelInfo(
            name=meta['name'],
            version=meta.get('version', 'N/A'),
            loaded=meta['loaded'],
            type=key
        ))
    
    return models

@app.post("/predict", response_model=PredictResponse)
async def predict(data: PredictRequest):
    """
    Realiza predição usando o melhor modelo disponível
    
    Priority:
    1. TensorFlow (mais preciso)
    2. Sklearn (mais rápido)
    """
    
    # Verificar se há algum modelo carregado
    if not any(models_loaded.values()):
        raise HTTPException(
            status_code=503,
            detail="Nenhum modelo disponível. Verifique os logs de startup."
        )
    
    try:
        # Pré-processar entrada
        features = preprocess_input(data)
        
        # Tentar usar TensorFlow primeiro (mais preciso)
        if models_loaded['tensorflow'] is not None:
            result = predict_with_tensorflow(features)
        
        # Fallback para Sklearn
        elif models_loaded['sklearn'] is not None:
            result = predict_with_sklearn(features)
        
        else:
            raise HTTPException(
                status_code=503,
                detail="Nenhum modelo de predição disponível"
            )
        
        # Adicionar recomendação
        recommendation = "Continue operação normal"
        if result['prediction'] == 'anomaly':
            if result['confidence'] > 0.9:
                recommendation = "⚠️ CRÍTICO: Parar equipamento e inspecionar imediatamente"
            elif result['confidence'] > 0.7:
                recommendation = "⚠️ ALERTA: Agendar manutenção preventiva urgente"
            else:
                recommendation = "⚠️ Monitorar de perto e verificar em 24h"
        
        return PredictResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            model_used=result['model_used'],
            timestamp=datetime.utcnow(),
            details={
                **result,
                'recommendation': recommendation,
                'input_data': {
                    'temperature': data.temperature,
                    'pressure': data.pressure,
                    'vibration': data.vibration,
                    'rpm': data.rotational_speed,
                    'torque': data.torque,
                    'tool_wear': data.tool_wear
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante predição: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(data: List[PredictRequest]):
    """Predição em lote para múltiplas amostras"""
    if len(data) > 100:
        raise HTTPException(
            status_code=400,
            detail="Máximo de 100 amostras por requisição"
        )
    
    results = []
    for item in data:
        try:
            result = await predict(item)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "input": item.dict()
            })
    
    return {
        "total": len(data),
        "success": len([r for r in results if not isinstance(r, dict) or 'error' not in r]),
        "failed": len([r for r in results if isinstance(r, dict) and 'error' in r]),
        "results": results
    }

@app.get("/models/reload")
async def reload_models():
    """Recarrega todos os modelos (útil após upload)"""
    await load_models()
    return {
        "message": "Modelos recarregados",
        "status": [
            {key: meta['loaded']} 
            for key, meta in model_metadata.items()
        ]
    }

# ==================== EXECUTAR ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
