"""
===============================================================================
INGEST DATASETS - Ingestão de Datasets do Kaggle
===============================================================================

Processa datasets e adiciona à memória vetorial

Autor: João Manoel
===============================================================================
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from embeddings import VectorMemory

logger = logging.getLogger(__name__)

# ============================================
# INGESTÃO AI4I
# ============================================

async def ingest_ai4i_sample(
    memory: VectorMemory,
    filepath: Optional[str] = None,
    sample_size: int = 1000
):
    """
    Ingerir dataset AI4I na memória vetorial
    
    Args:
        memory: Instância de VectorMemory
        filepath: Caminho do arquivo CSV (opcional)
        sample_size: Número de amostras a ingerir
    """
    logger.info("📊 Iniciando ingestão AI4I...")
    
    # Tentar localizar arquivo
    if not filepath:
        possible_paths = [
            "./kaggle_inputs/ai4i2020.csv",
            "./data/ai4i2020.csv",
            "../kaggle_inputs/ai4i2020.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
    
    if not filepath or not os.path.exists(filepath):
        logger.warning("⚠️ Arquivo AI4I não encontrado, criando dados sintéticos...")
        df = create_synthetic_ai4i(sample_size)
    else:
        try:
            df = pd.read_csv(filepath)
            logger.info(f"✅ AI4I carregado: {df.shape}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar AI4I: {e}")
            return
    
    # Amostrar se dataset muito grande
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        logger.info(f"📉 Amostragem reduzida para {sample_size} linhas")
    
    # Criar documentos
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        # Criar documento textual
        doc = f"""Máquina tipo {row.get('Type', 'N/A')}: 
Temperatura ar: {row.get('Air temperature [K]', 0):.1f}K, 
Temperatura processo: {row.get('Process temperature [K]', 0):.1f}K,
Rotação: {row.get('Rotational speed [rpm]', 0):.0f} RPM,
Torque: {row.get('Torque [Nm]', 0):.1f} Nm,
Desgaste ferramenta: {row.get('Tool wear [min]', 0):.0f} min,
Falha: {'Sim' if row.get('Machine failure', 0) == 1 else 'Não'}"""
        
        documents.append(doc)
        ids.append(f"ai4i_{idx}")
        metadatas.append({
            "dataset": "ai4i",
            "type": str(row.get('Type', '')),
            "failure": int(row.get('Machine failure', 0))
        })
    
    # Adicionar à memória
    try:
        memory.add_documents(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"✅ AI4I: {len(documents)} documentos adicionados à memória")
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar AI4I à memória: {e}")

# ============================================
# INGESTÃO CMAPSS
# ============================================

async def ingest_cmapss_sample(
    memory: VectorMemory,
    filepath: Optional[str] = None,
    sample_size: int = 500
):
    """
    Ingerir dataset CMAPSS na memória vetorial
    """
    logger.info("📊 Iniciando ingestão CMAPSS...")
    
    # Tentar localizar arquivo
    if not filepath:
        possible_paths = [
            "./kaggle_inputs/cmapss_train.txt",
            "./data/train_FD001.txt",
            "../kaggle_inputs/train_FD001.txt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
    
    if not filepath or not os.path.exists(filepath):
        logger.warning("⚠️ Arquivo CMAPSS não encontrado, criando dados sintéticos...")
        df = create_synthetic_cmapss(sample_size)
    else:
        try:
            # Carregar CMAPSS (formato texto sem header)
            sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
            setting_cols = ['setting_1', 'setting_2', 'setting_3']
            columns = ['unit', 'cycle'] + setting_cols + sensor_cols
            
            df = pd.read_csv(filepath, sep='\s+', header=None, names=columns)
            logger.info(f"✅ CMAPSS carregado: {df.shape}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar CMAPSS: {e}")
            return
    
    # Amostrar
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        logger.info(f"📉 Amostragem reduzida para {sample_size} linhas")
    
    # Criar documentos
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        # Pegar alguns sensores principais
        doc = f"""Motor unidade {row.get('unit', 0)}, ciclo {row.get('cycle', 0)}: 
Sensor 1: {row.get('sensor_1', 0):.2f}, 
Sensor 2: {row.get('sensor_2', 0):.2f},
Sensor 7: {row.get('sensor_7', 0):.2f},
Sensor 11: {row.get('sensor_11', 0):.2f},
Sensor 15: {row.get('sensor_15', 0):.2f}"""
        
        documents.append(doc)
        ids.append(f"cmapss_{idx}")
        metadatas.append({
            "dataset": "cmapss",
            "unit": int(row.get('unit', 0)),
            "cycle": int(row.get('cycle', 0))
        })
    
    # Adicionar à memória
    try:
        memory.add_documents(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"✅ CMAPSS: {len(documents)} documentos adicionados à memória")
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar CMAPSS à memória: {e}")

# ============================================
# DADOS SINTÉTICOS
# ============================================

def create_synthetic_ai4i(n_samples: int = 1000) -> pd.DataFrame:
    """Criar dados sintéticos AI4I"""
    import numpy as np
    
    logger.info(f"🔧 Gerando {n_samples} amostras sintéticas AI4I...")
    
    types = np.random.choice(['L', 'M', 'H'], n_samples, p=[0.6, 0.3, 0.1])
    air_temp = np.random.uniform(295, 305, n_samples)
    process_temp = np.random.uniform(305, 315, n_samples)
    rpm = np.random.uniform(1200, 2800, n_samples)
    torque = np.random.uniform(20, 70, n_samples)
    tool_wear = np.random.uniform(0, 250, n_samples)
    failures = np.random.binomial(1, 0.1, n_samples)
    
    df = pd.DataFrame({
        'Type': types,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Machine failure': failures
    })
    
    return df

def create_synthetic_cmapss(n_samples: int = 500) -> pd.DataFrame:
    """Criar dados sintéticos CMAPSS"""
    import numpy as np
    
    logger.info(f"🔧 Gerando {n_samples} amostras sintéticas CMAPSS...")
    
    data = []
    for i in range(n_samples):
        row = {
            'unit': np.random.randint(1, 20),
            'cycle': np.random.randint(1, 200),
            'setting_1': np.random.uniform(-0.0007, 0.0007),
            'setting_2': np.random.uniform(0.0001, 0.0005),
            'setting_3': np.random.uniform(95, 105)
        }
        
        # Sensores
        for j in range(1, 22):
            row[f'sensor_{j}'] = np.random.uniform(500, 650)
        
        data.append(row)
    
    return pd.DataFrame(data)

# ============================================
# INGESTÃO COMPLETA
# ============================================

async def run_full_ingestion(memory: VectorMemory):
    """
    Executar ingestão completa de todos os datasets
    """
    logger.info("🚀 Iniciando ingestão completa...")
    
    await ingest_ai4i_sample(memory, sample_size=1000)
    await ingest_cmapss_sample(memory, sample_size=500)
    
    total_docs = memory.get_collection_size()
    logger.info(f"✅ Ingestão completa concluída: {total_docs} documentos totais")
