"""
Vector Store Service with Hybrid search (Dense + Sparse)
Production Ready: Uses FastEmbed for SPLADE/BGE-M3 generation.
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import uuid
import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding

# Backend Clients
from pinecone import Pinecone, ServerlessSpec, PineconeException
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, Distance, VectorParams,
    PointStruct, SparseVectorParams, SparseVector,
    HnswConfigDiff, SparseIndexParams
)

from app.core.config import settings
import structlog

logger = structlog.get_logger()


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends"""
    
    @abstractmethod
    async def add_document_batch(
        self,
        documents: List[Dict[str, Any]],
        collection: str
    ) -> List[str]:
        """Add a batch of documents to the vetor store (More efficient)"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        query_sparse: Optional[Dict[str, float]] = None, # Added for native hybrid
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]] :
        """Perform similarity or hybrid search natively in DB"""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str, collection: str):
        """Delete a document"""
        pass
    
class PineconeBackend(VectorStoreBackend):
    """Pinecone vector store backend"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # Check if index exists (Cached lookup in production usually)
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info("creating_pinecone_index", name=self.index_name)
            self.pc.create_index(
                name = self.index_name,
                dimension= settings.DIMENSION,
                metric = "dotproduct", # Recommended for Hybrid
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        
    @retry(
        stop= stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry= retry_if_exception_type(PineconeException)
    )   
    async def add_document_batch(
        self,
        documents: List[Dict[str, Any]],
        collection: str
    )-> List[str]:
        """Batch upsert ith retries"""
        vectors_to_upsert = []
        doc_ids = []
        
        for doc in documents:
            doc_id = doc.get("id", str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            # Prepare metadata
            metadata = doc.get("metadata", {})
            metadata["text"] = doc.get("text", "")
            metadata["collection"] = collection
            
            # Prepare vector structure
            vector_data = {
                "id" : doc_id,
                "values" : doc["dense_embedding"],
                "metadata" : metadata
            }
            
            # Map FastEmbed output to Pinecone Sparse format
            if "sparse_embedding" in doc and doc["sparse_embedding"]:
                vector_data["sparse_values"] = {
                    "indices" : [int(i) for i in doc["sparse_embedding"].indices],
                    "values" : [float(v) for v in doc["sparse_embedding"].values]
                }
            vectors_to_upsert(vector_data)
        
        # Pinecone upsert (Namespace acts as Collection)
        self.index.upsert(vectors=vectors_to_upsert, namespace=collection)
        return doc_ids
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(main=1, max=5))
    async def similarity_search(self, query_dense, collection, query_sparse=None, top_k=10, filter=None):
        
        # Prepare arguments
        search_kwargs = {
            "vector" : query_dense,
            "top_k" : top_k,
            "include_metadata" : True,
            "namespace" : collection,
            "filter" : filter
        }
        
        # Add sparse vector if provided
        if query_sparse:
            search_kwargs["sparse_vector"] = {
                "indices" : [int(i) for i in query_sparse.indices],
                "values" :  [float(v) for v in query_sparse.values]
            }
        
        results = self.index.query(**search_kwargs)
        
        return [
            {
                "id" : match.id,
                "score": match.score,
                "metadata": match.metadata,
                "text": match.metadata.get("text", "")
            } 
            for match in results.matches
        ]
        
    async def delete_document(self, doc_id: str, collection: str):
        self.index.delete(ids=[doc_id], namespace=collection)

    
class QdrantBackend(VectorStoreBackend):
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    
    async def _ensure_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=settings.DIMENSION, distance=Distance.COSINE, hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10000))},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseVectorParams(on_disk=True))}
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def add_documents_batch(self, documents: List[Dict[str, Any]], collection: str) -> List[str]:
        await self._ensure_collection(collection)
        points = []
        doc_ids = []
        
        for doc in documents:
            doc_id = doc.get("id", str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            vector_struct = {"dense": doc["dense_embedding"]}
            
            # Map FastEmbed output to Qdrant Sparse format
            if "sparse_embedding" in doc and doc["sparse_embedding"]:
                vector_struct["sparse"] = SparseVector(
                    indices=doc["sparse_embedding"].indices.tolist(),
                    values=doc["sparse_embedding"].values.tolist()
                )

            points.append(PointStruct(
                id=doc_id, 
                vector=vector_struct, 
                payload={**doc.get("metadata", {}), "text": doc.get("text", "")}
            ))
            
        self.client.upsert(collection_name=collection, points=points)
        return doc_ids

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def similarity_search(self, query_dense, collection, query_sparse=None, top_k=10, filter=None):
        qdrant_filter = None
        if filter:
            conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
            qdrant_filter = Filter(must=conditions)
            
        # Qdrant Hybrid Query (Simplest implementation using Fusion)
        # Note: True Hybrid in Qdrant 1.10+ uses Prefetch, simplified here to dense search
        results = self.client.search(
            collection_name=collection,
            query_vector=("dense", query_dense),
            limit=top_k,
            query_filter=qdrant_filter
        )
        return [{"id": str(h.id), "score": h.score, "metadata": h.payload, "text": h.payload.get("text", "")} for h in results]

    async def delete_document(self, doc_id: str, collection: str):
        self.client.delete(collection_name=collection, points_selector=[doc_id])

class VectorStore:
    """
    Production Vector Store with SPLADE/BGE-M3 integration.
    """
    
    def __init__(
        self,
        backend: VectorStoreBackend,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        sparse_model_name : str = "prithvida/Splade_PP_En_V1" # <--- Default SPLADE model
    ):
        self.backend = backend
        
        # 1. initialize Dense Model
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name = settings.EMBEDDING_MODEL,
            model_kwargs = {"device" : "cpu"},
            encode_kwargs = {'normalize_embeddings': True}
        )
        
        # 2. initialize Sparse Model (SPLADE)
        # FastEmbed handels downloading and caching the model automatically.
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection: str = "default"
    ) -> List[str]:
        """
        Generates both Dense and Sparse embeddings and uploads them.
        """
        if metadatas is None:
            metadatas= [{} for _ in texts]
            
        # 1. Generate Dense Embeddings (Async Batch)
        dense_embeddings = await self.embeddings.aembed_documents(texts)
        
        # 2. Generate Sparse Embeddings (Sync Batch Via FastEmbed)
        # fastembed return a generator, so we cast to list.
        # It handles tokenization and weight generation (SPLADE logic) internally.
        
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        # 3. Combine Data
        documents_to_add = []
        for i, text in enumerate(texts):
            documents_to_add.append({
                "text" : text,
                "dense_embedding": dense_embeddings[i],
                "sparse_embedding" : sparse_embeddings[i], # Contains .indices and values
                "metadata": metadatas[i]
            })
        
        # 4. Upload to Backend
        return await self.backend.add_document_batch(documents_to_add, collection)
    
    async def search(
        self,
        query: str ,
        collection: str = "default",
        top_k : int = 10,
        enable_hybrid: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using generated embeddings.
        """
        # 1. Generate Dense Vector
        query_dense = await self.embeddings.aembed_query(query)
        
        # 2. generate sparse Vector (if enabled)
        query_sparse = None
        if enable_hybrid:
            query_sparse = list(self.sparse_model.embed([query]))[0]
            
        # 3. Search Backend
        return await self.backend.similarity_search(
            query_dense = query_dense,
            collection = collection,
            query_sparse= query_sparse,
            top_k = top_k,
            filter= filter
        )
        
# Factory function
def create_vector_store(
    backend_type: str = "pinecone",
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> VectorStore:
    """
    Create a vector store with specified backend.
    
    Args:
        backend_type: "pinecone", "qdrant", or "opensearch"
        embeddings: Optional embeddings model
    
    Returns:
        VectorStore instance
    """
    if backend_type == "pinecone":
        backend = PineconeBackend()
    elif backend_type == "qdrant":
        backend = QdrantBackend()
    else:
        raise ValueError(f"Unsupported backend: {backend_type}")
    
    return VectorStore(backend=backend, embeddings=embeddings)