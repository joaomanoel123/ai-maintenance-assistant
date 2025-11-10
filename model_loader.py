"""
===============================================================================
MODEL LOADER - Carregador de Modelos Preditivos
===============================================================================

Carrega modelos CMAPSS e AI4I no startup

Autor: João Manoel
===============================================================================
"""

import os
import logging
from pathlib import Path
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Dicionário global de modelos
MODELS = {
    'cmapss': None,
    'cmapss_scaler': None,
    'ai4i': None,
    'ai4i_scaler': None
}

def load_models():
    """
    Carregar todos os modelos preditivos
    """
    logger.info("🔄 Carregando modelos preditivos...")
    
    # Diretório de modelos
    models_dir = Path(os.getenv("MODELS_DIR", "./models"))
    
    # ========================================
    # 1. CMAPSS MODEL
    # ========================================
    cmapss_path = os.getenv("CMAPSS_PATH", str(models_dir / "cmapss_model.pkl"))
    
    if os.path.exists(cmapss_path):
        try:
            # Tentar carregar como joblib
            MODELS['cmapss'] = joblib.load(cmapss_path)
            logger.info(f"✅ CMAPSS (joblib) carregado: {cmapss_path}")
        except Exception:
            try:
                # Tentar carregar como Keras/TensorFlow
                import tensorflow as tf
                MODELS['cmapss'] = tf.keras.models.load_model(cmapss_path)
                logger.info(f"✅ CMAPSS (keras) carregado: {cmapss_path}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar CMAPSS: {e}")
    else:
        logger.warning(f"⚠️ CMAPSS não encontrado: {cmapss_path}")
    
    # CMAPSS Scaler
    cmapss_scaler_path = str(models_dir / "cmapss_scaler.pkl")
    if os.path.exists(cmapss_scaler_path):
        try:
            MODELS['cmapss_scaler'] = joblib.load(cmapss_scaler_path)
            logger.info(f"✅ CMAPSS Scaler carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar CMAPSS Scaler: {e}")
    
    # ========================================
    # 2. AI4I MODEL
    # ========================================
    ai4i_path = os.getenv("AI4I_PATH", str(models_dir / "ai4i_rf.pkl"))
    
    if os.path.exists(ai4i_path):
        try:
            MODELS['ai4i'] = joblib.load(ai4i_path)
            logger.info(f"✅ AI4I carregado: {ai4i_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar AI4I: {e}")
    else:
        logger.warning(f"⚠️ AI4I não encontrado: {ai4i_path}")
    
    # AI4I Scaler
    ai4i_scaler_path = str(models_dir / "ai4i_scaler.pkl")
    if os.path.exists(ai4i_scaler_path):
        try:
            MODELS['ai4i_scaler'] = joblib.load(ai4i_scaler_path)
            logger.info(f"✅ AI4I Scaler carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar AI4I Scaler: {e}")
    
    # ========================================
    # 3. VERIFICAR CARREGAMENTO
    # ========================================
    loaded_models = [k for k, v in MODELS.items() if v is not None]
    logger.info(f"📦 Modelos carregados: {loaded_models}")
    
    if not any([MODELS['cmapss'], MODELS['ai4i']]):
        logger.warning("⚠️ Nenhum modelo preditivo foi carregado!")

def model_predict(model_key: str, features: list):
    """
    Fazer predição com modelo específico
    
    Args:
        model_key: 'cmapss' ou 'ai4i'
        features: lista de features
    
    Returns:
        Predição do modelo
    """
    model = MODELS.get(model_key)
    scaler = MODELS.get(f"{model_key}_scaler")
    
    if model is None:
        raise RuntimeError(f"Modelo '{model_key}' não carregado")
    
    try:
        # Converter para numpy array
        features_array = np.array([features])
        
        # Normalizar se scaler disponível
        if scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Predizer
        if hasattr(model, 'predict'):
            prediction = model.predict(features_array)
            
            # Se for probabilidade (AI4I), pegar classe 1
            if hasattr(model, 'predict_proba') and model_key == 'ai4i':
                proba = model.predict_proba(features_array)
                return {
                    "class": int(prediction[0]),
                    "probability": float(proba[0][1]),
                    "confidence": float(max(proba[0]))
                }
            else:
                # RUL (CMAPSS)
                return {
                    "rul": float(prediction[0]) if len(prediction.shape) == 1 else float(prediction[0][0]),
                    "unit": "cycles"
                }
        else:
            raise RuntimeError(f"Modelo '{model_key}' não possui método predict")
    
    except Exception as e:
        logger.error(f"Erro na predição com {model_key}: {e}")
        raise

def get_model_info(model_key: str) -> dict:
    """
    Obter informações sobre um modelo
    """
    model = MODELS.get(model_key)
    
    if model is None:
        return {"loaded": False}
    
    info = {"loaded": True, "type": type(model).__name__}
    
    # Adicionar informações específicas
    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_
    
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    
    return info
