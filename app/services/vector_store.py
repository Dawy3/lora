"""
Vector Store Service with Hybrid search (Dense + Sparse)
Supports multiple backends: Pinecone, Qdrant, OpenSearch
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import structlog
import uuid

from pinecone import Pinecone, ServerlessSpec
from qdrant_client.models import Filter, FieldCondition, MatchValue


from app.core.config import settings

logger = structlog.get_logger()


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends"""
    
    @abstractmethod
    async def add_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection: str
    ) -> str:
        """Add a document to the vetor store"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]] :
        """Perform similarity search"""
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
        
        # Create index if it doesn't exist
        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name = self.index_name,
                dimension= settings.DIMENSION,
                metric = "cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.PINECONE_ENVIRONMENT
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        
        
    async def add_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection: str
    )-> str:
        """Add document to Pinecone"""
        doc_id = str(uuid.uuid4())
        
        # Add Collection to metadata
        full_metadata = {**metadata, "collection": collection, "text":text}
        
        # Upsert to Pinecone
        self.index.upsert(
            vectors=[(doc_id, embedding, full_metadata)],
            namespace=collection
        )
        return doc_id
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        top_k : int = 10,
        filter: Optional[Dict[str, Any]]= None
    )-> List[Dict[str, Any]]:
        """Search in Pinecone"""
        results= self.index.query(
            vector= query_embedding,
            top_k = top_k,
            include_metadata=True,
            namespace=collection,
            filter= filter  
        )
        
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
        """Delte from Pinecone"""
        self.index.delete(ids=[doc_id], namespace=collection)
        
    async def update_document(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection: str
    ):
        """Update document in Pinecone"""
        full_metadta= {**metadata, "collection": collection}
        self.index.upsert(
            vectors=[(doc_id, embedding, full_metadta)],
            namespace= collection
        )
    
    async def get_document(
        self,
        doc_id: str,
        collection: str
    ) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        result = self.index.fetch(ids=[doc_id], namespace=collection)
        if result.vectors and doc_id in result.vectors:
            vector_data = result.vectors[doc_id]
            return{
                "id" : doc_id,
                "metadata": vector_data.metadata,
                "text" : vector_data.metadata.get("text")
            }
        return None
    
class QdrantBackend(VectorStoreBackend):
    """Qdrant vector store backend"""
    
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
    
    async def add_document(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection: str
    ) -> str:
        """Add document to Qdrant"""
        from qdrant_client.models import PointStruct
        import uuid
        
        # Ensure collection exists
        try:
            self.client.get_collection(collection)
        except:
            from qdrant_client.models import Distance, VectorParams
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        doc_id = str(uuid.uuid4())
        point = PointStruct(
            id=doc_id,
            vector=embedding,
            payload={**metadata, "text": text}
        )
        
        self.client.upsert(collection_name=collection, points=[point])
        return doc_id
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search in Qdrant"""
        
        # Build filter if provided
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter
        )
        
        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "metadata": hit.payload,
                "text": hit.payload.get("text", "")
            }
            for hit in results
        ]
    
    async def delete_document(self, doc_id: str, collection: str):
        """Delete from Qdrant"""
        self.client.delete(collection_name=collection, points_selector=[doc_id])
        

class VectorStore:
    """
    Main vector store interface with hybrid search capabilities.
    
    Combines dense vector search with sparse MB25 search for better retrieval.
    """
    
    def __init__(
        self,
        backend: VectorStoreBackend,
        embeddings: Optional[HuggingFaceEmbeddings] = None
    ):
        self.backend = backend
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name = settings.EMBEDDING_MODEL,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embedding': True}
        )
        
        # BM25 index for sparse retrieval (in-memory for demo) 
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.bm25_docs: Dict[str, List[Dict[str, Any]]] = {}
        
        
    async def add_document(
        self,
        text: str,
        embedding : Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection: str = "default"
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            text: Document text
            embedding: Pre-computed embedding (optional)
            metadata: Document metadata
            collection : Collection name
        """
        # Generate embedding if not provided
        if embedding in None:
            embedding = await self.embeddings.aembed_query(text)
            
        # Add to backend
        doc_id = await self.backend.add_document(
            text=text,
            embedding= embedding,
            metadata= metadata or {},
            collection = collection
        )
        
        # Update BM25 index
        await self._update_bm25_index(collection, text, doc_id, metadata or {})
        
        return doc_id
    
    async def _update_bm25_index(
        self,
        collection:str,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ):
        """Update BM25 index for sparse retrieval"""
        # Tokenize text
        tokens = text.lower().split()
        
        # Add to collection's BM25 index
        if collection not in self.bm25_docs:
            self.bm25_docs[collection] = []
            
        self.bm25_docs[collection].append({
            "id" : doc_id,
            "text" : text,
            "tokens": tokens,
            "metadata" : metadata
        })
        
        # Rebuild BM25 index
        all_tokens = [doc["tokens"] for doc in self.bm25_docs[collection]]
        self.bm25_indices[collection] = BM25Okapi(all_tokens)
        
        
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str = "default",
        top_k: int = 10,
        filter : Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Dense Vector Search"""
        return await self.backend.similarity_search(
            query_embedding= query_embedding,
            collection= collection,
            top_k= top_k,
            filter= filter
        )
        
    async def bm25_search(
        self,
        query: str,
        collection: str = "default",
        top_k : int = 10
    )-> List[Dict[str, Any]]:
        """Sparse BM25 search"""
        if collection not in self.bm25_indices:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # GEt BM25 scores
        scores = self.bm25_indices[collection].get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.bm25_docs[collection]):
                doc = self.bm25_docs[collection][idx]
                results.append({
                    "id" : doc["id"],
                    "score" : float(scores[idx]),
                    "text" : doc["text"],
                    "metadata" : doc["metadata"]
                })
        
        return results
    
    async def hybrid_search(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10,
        dense_weight: float= 0.7,       # Semantic
        sparse_weight: float = 0.3      # Keyword
    ) -> Dict[str, Any]:
        """
        Hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            collection: Collection name
            top_k: Number of results
            dense_weight: Weight for dense search (0-1)
            sparse_weight: Weight for sparse search (0-1)
            
        Returns:
            Dict with dense, sparse, and reranked results
        """
        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Dense Search
        dense_results = await self.similarity_search(
            query_embedding= query_embedding,
            collection = collection,
            top_k = top_k * 2  # Get more for reranking
        )
        
        # Sparse Search
        sparse_results = await self.bm25_search(
            query= query,
            collection= collection,
            top_k= top_k *2
        )
        
        # Combine and rerank
        reranked = self._rerank_results(
            dense_results,
            sparse_results,
            dense_weight, #  0.7
            sparse_weight, # 0.3
            top_k
        )
        
        return{
            "dense_results" : dense_results[:top_k],
            "sparse_results" : sparse_results[:top_k],
            "reranked_results": reranked
        }
        
    
    def _rerank_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight : float,
        sparse_weight : float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using reciprocal rank fusion"""    
        # Normalize Weights
        total_weight = dense_weight + sparse_weight
        dense_weight = dense_weight / total_weight
        sparse_weight = sparse_weight / total_weight
        
        # Create combined scores
        scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        
        # Add dense scores ("Meaning")
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            scores[doc_id] = dense_weight / (rank + 60) # Reciprocal Rank Fusion (RRF) with k=60 
            doc_data[doc_id] = result
        
        # Add sparse scores ("Keyword")
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result["id"]
            if doc_id in scores:
                scores[doc_id] += sparse_weight / (rank + 60)
            else:
                scores[doc_id] = sparse_weight / (rank + 60)
                doc_data[doc_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top-k
        return [
            {**doc_data[doc_id], "hybrid_score": scores[doc_id]}
            for doc_id in sorted_ids[:top_k]
        ]
    
    async def delete_document(self, doc_id: str, collection: str = "default"):
        """Delete a document"""
        await self.backend.delete_document(doc_id, collection)
    
    async def update_document(
        self,
        doc_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        collection: str = "default"
    ):
        """Update a document"""
        await self.backend.update_document(doc_id, embedding, metadata, collection)
    
    async def get_document(
        self,
        doc_id: str,
        collection: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        return await self.backend.get_document(doc_id, collection)
    

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