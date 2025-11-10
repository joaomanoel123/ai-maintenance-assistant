"""
===============================================================================
DB MODULE - Database e Persistência
===============================================================================

Gerencia conexão com PostgreSQL/Neon e persistência de experiências

Autor: João Manoel
===============================================================================
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict
import json

logger = logging.getLogger(__name__)

# Variável global para conexão
_db_pool = None

async def init_database():
    """
    Inicializar conexão com database
    """
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.warning("⚠️ DATABASE_URL não configurada, usando modo sem persistência")
        return
    
    try:
        import asyncpg
        
        # Criar pool de conexões
        global _db_pool
        _db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=10)
        
        logger.info("✅ Pool de conexões criado")
        
        # Criar tabelas se não existirem
        await create_tables()
        
        logger.info("✅ Database inicializado")
    
    except ImportError:
        logger.warning("⚠️ asyncpg não instalado, usando modo sem persistência")
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar database: {e}")

async def create_tables():
    """
    Criar tabelas necessárias
    """
    if not _db_pool:
        return
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS chat_experiences (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(100),
        user_message TEXT NOT NULL,
        assistant_response TEXT NOT NULL,
        contexts JSONB,
        prediction_info TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_user_id ON chat_experiences(user_id);
    CREATE INDEX IF NOT EXISTS idx_created_at ON chat_experiences(created_at);
    """
    
    try:
        async with _db_pool.acquire() as conn:
            await conn.execute(create_table_sql)
        logger.info("✅ Tabelas criadas/verificadas")
    except Exception as e:
        logger.error(f"❌ Erro ao criar tabelas: {e}")

async def save_experience_record(
    user_id: str,
    user_message: str,
    assistant_response: str,
    contexts: Optional[List[str]] = None,
    prediction_info: Optional[str] = None
):
    """
    Salvar experiência de chat no database
    """
    if not _db_pool:
        logger.debug("Database não disponível, experiência não salva")
        return
    
    try:
        async with _db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_experiences 
                (user_id, user_message, assistant_response, contexts, prediction_info)
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id,
                user_message,
                assistant_response,
                json.dumps(contexts) if contexts else None,
                prediction_info
            )
        
        logger.debug(f"💾 Experiência salva para user {user_id}")
    
    except Exception as e:
        logger.error(f"❌ Erro ao salvar experiência: {e}")

async def get_user_history(user_id: str, limit: int = 10) -> List[Dict]:
    """
    Recuperar histórico de um usuário
    """
    if not _db_pool:
        return []
    
    try:
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_message, assistant_response, created_at
                FROM chat_experiences
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                user_id,
                limit
            )
            
            return [dict(row) for row in rows]
    
    except Exception as e:
        logger.error(f"❌ Erro ao buscar histórico: {e}")
        return []

async def close_database():
    """
    Fechar conexões do database
    """
    global _db_pool
    if _db_pool:
        await _db_pool.close()
        _db_pool = None
        logger.info("👋 Pool de conexões fechado")
