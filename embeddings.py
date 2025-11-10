"""
===============================================================================
EMBEDDINGS - Memória Vetorial com ChromaDB
===============================================================================

Implementa memória vetorial usando ChromaDB + SentenceTransformers

Autor: João Manoel
===============================================================================
"""

import os
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class VectorMemory:
    """
    Memória vetorial com ChromaDB
    """
    
    def __init__(self, collection_name: str = "agi_memory"):
        self.collection_name = collection_name
        
        # Configurar persistência
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"📂 ChromaDB persist dir: {persist_dir}")
        
        # Inicializar cliente ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            logger.info("✅ ChromaDB client inicializado")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar ChromaDB: {e}")
            raise
        
        # Modelo de embeddings
        logger.info("🔄 Carregando modelo de embeddings...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Modelo de embeddings carregado")
        
        self.collection = None
    
    async def start(self):
        """
        Inicializar ou recuperar collection
        """
        try:
            # Verificar se collection existe
            collections = self.client.list_collections()
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                # Recuperar collection existente
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"✅ Collection '{self.collection_name}' recuperada")
            else:
                # Criar nova collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "AGI Memory Collection"}
                )
                logger.info(f"✅ Collection '{self.collection_name}' criada")
            
            # Verificar tamanho
            count = self.collection.count()
            logger.info(f"📊 Collection possui {count} documentos")
        
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar collection: {e}")
            raise
    
    def add_documents(
        self, 
        ids: List[str], 
        documents: List[str], 
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Adicionar documentos à memória
        
        Args:
            ids: Lista de IDs únicos
            documents: Lista de textos
            metadatas: Lista de metadados (opcional)
        """
        if not self.collection:
            raise RuntimeError("Collection não inicializada. Execute start() primeiro.")
        
        try:
            # Gerar embeddings
            embeddings = self.embed_model.encode(
                documents, 
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()
            
            # Preparar metadatas
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            # Adicionar à collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✅ {len(documents)} documentos adicionados à memória")
        
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar documentos: {e}")
            raise
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Buscar documentos similares
        
        Args:
            query_text: Texto da consulta
            n_results: Número de resultados
            where: Filtros de metadata (opcional)
        
        Returns:
            Dicionário com resultados
        """
        if not self.collection:
            raise RuntimeError("Collection não inicializada")
        
        try:
            # Gerar embedding da query
            query_embedding = self.embed_model.encode(
                query_text,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()
            
            # Buscar
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Erro na query: {e}")
            return {"documents": [[]], "distances": [[]], "metadatas": [[]]}
    
    def get_collection_size(self) -> int:
        """
        Obter número de documentos na collection
        """
        if not self.collection:
            return 0
        
        try:
            return self.collection.count()
        except:
            return 0
    
    def delete_collection(self):
        """
        Deletar collection (cuidado!)
        """
        if self.collection:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"⚠️ Collection '{self.collection_name}' deletada")
            self.collection = None
    
    async def close(self):
        """
        Fechar conexões
        """
        logger.info("👋 Fechando memória vetorial")
        # ChromaDB não precisa de close explícito
    
    def search_by_metadata(
        self, 
        where: Dict, 
        n_results: int = 10
    ) -> Dict:
        """
        Buscar por metadata
        
        Args:
            where: Filtro de metadata (ex: {"category": "models"})
            n_results: Número de resultados
        """
        if not self.collection:
            raise RuntimeError("Collection não inicializada")
        
        try:
            results = self.collection.get(
                where=where,
                limit=n_results
            )
            return results
        except Exception as e:
            logger.error(f"❌ Erro na busca por metadata: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
