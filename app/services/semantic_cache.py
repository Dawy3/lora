"""
Production Semantic Caching Service
Features: Redis Vector Search (RedisSearh), Two-Tier Caching (Exact + Semantic)
Tenant Isolation, and Circuit Breaking.
"""
import json
import hashlib
import asyncio
import time
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import redis.asyncio as redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import structlog

from app.config import settings

# Configure structed logging
logger = structlog.get_logger()

class SemanticCache:
    """
    High preformance semantic Cache using Redis Vector search.
    
    Architecture:
    1. L1 Cache (Exact) : SHA256 hash lookup. Ultra-fast, zero embedding Cost.
    2. L2 Cache (Semantic): HNSW Vector Index lookup using RedisSearch.
    """
    
    INDEX_NAME = "semantic_cache_idx"
    VECTOR_DIM = settings.DIMENSION
    DISTANCE_METRIC = "COSINE"
    
    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        embedding_model: Optional[HuggingFaceEmbeddings] = None,
        similarity_threshold: float = 0.95,   # 0.95 -> is saver for production than lower value
        ttl_seconds: int = 86400,  # 24 Hours
        connect_timeout: int = 5
    ):
        # connection pool for scalability
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connection = 100,
            decode_response= False, # Must be False for vector binary data
            socket_connect_timeout= connect_timeout,
            socket_keepalive = True         
        )
        
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        self.embeddings = embedding_model or HuggingFaceEmbeddings(
            model_name = settings.EMBEDDING_MODEL,
            model_kwargs = {"device" : "cpu"},
            encode_kwargs = {'normalize_embeddings': True}
        )
        
        self.similarity_threshold = similarity_threshold    # 0.95
        self.ttl_seconds = ttl_seconds      # 24 h
        
        # Prefix 
        self.exact_prefix = "cache:exact"
        self.semantic_prefix = "cache:semantic"
        
    
    async def initialize(self):
        """
        Idemponent initialization of Redis vector Index.
        Call this on startup.
        """
        try:
            # Check if index exists
            await self.redis_client.ft(self.INDEX_NAME).info()
            logger.info("redis_index_exists", index=self.INDEX_NAME)
        except redis.ResponseError:
            # Create index if not exists
            logger.info("Create redis index", index=self.INDEX_NAME)
            
            # Define schema
            schema = (
                TagField("user_id"),            # For strict tenant isolation
                TextField("query"),             # For reference/debugging
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.VECTOR_DIM,
                        "DISTANCE_METRIC" : self.DISTANCE_METRIC
                    }
                )
            )
            
            definition = IndexDefinition(
                prefix=[self.semantic_prefix],
                index_type=IndexType.HASH
            )
            
            await self.redis_client.ft(self.INDEX_NAME).create_index(
                fields=schema,
                definition= definition
            )
            
    def _hash_query(self, query:str, user_id:str = "global") -> str:
        """Create deterministic hash for L1 Exact Match."""
        # Normalize: strip whitespace, lowercase (optional depending on use case)
        normalize = f"{user_id}:{query.strip().lower()}"
        return hashlib.sha256(normalize.encode()).hexdigest() # 1. encode -> convert it to bytes | 2.hashlib.sha256-> ex(1001001110101)| 3.hexidigest 'make it redable -> ex(e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855)
    
    async def _safe_embed(self, text:str) -> Optional[List[float]]:
        """Circuit embedding wrapper for embedding generation."""
        try:
            # Adding a timeout constraint specifically for the embedding call
            return await asyncio.wait_for(
                self.embeddings.aembed_query(text),
                timeout= 3.0
            )
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            return None
        
    
    async def get(
        self,
        query:str,
        user_id: str = "global",
        min_similarity: Optional[float] = None
    )-> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        """
        Retrieve response from cache.
        Algorithm: L1 Exact Match -> L2 Semantic Search
        """
        threshold = min_similarity or self.similarity_threshold
        start_time = time.time()
        
        try:
            # --- Tier 1 : Exact Match (L1) ---
            # Extremely cheap check. Handles "user ask same exact things twice".
            exact_key = f"{self.exact_prefix}:{self._hash_query(query, user_id)}"
            exact_data = await self.redis_client.get(exact_key)
            
            if exact_data:
                entry = json.loads(exact_data)
                logger.info("cache_hit_l1_exact", latency=time.time() - start_time)
                return {
                    "response" : entry["response"],
                    "similarity" : 1.0,
                    "source" : "l1_exact"
                }, None         # We return None for the vector because we didn't need to generate one
                
            
            # --- Tier 2: Semantic Match (L2) ---
            # Only generate embedding if L1 failed
            query_vector =  await self._safe_embed(query)
            if not query_vector:
                return None, None # Fallback to LLm if embedding fails
            
            # Convert python list to binary blob for Redis
            vector_blob = np.array(query_vector, dtype=np.float32).tobytes()
            
            # Construct Vector Search Query
            # KNN 1 means "find the single nearest nighbor"
            q = Query(f"(@user_id:{{{user_id}}})=>[KNN 1 @embedding $vec_blob AS score]")\
                .sort_by("score")\
                .return_fields("response", "query", "score")\
                .dialect(2)             # Use RediSearch’s modern query language (vector search, KNN syntax, Advanced expression)
                
            params = {"vec_blob": vector_blob}
            
            # Execute Search
            results = await self.redis_client.ft(self.INDEX_NAME).search(q, params)
            
            if results.docs:
                top_hit = results.docs[0]
                # Redis returns distance (0 to 1), we want similarity (1 - distance) for Cosine
                # Note: Exact math depends on distance metric. For Cosine in Redis: distance = 1 - similarity
                # So similarity = 1 - distance (approx)
                
                # Check how Redis returns score based on your version. 
                # Usually it returns `score` as the distance.
                distance = float(top_hit.score)
                similarity = 1 - (distance / 2) # Approximation for normalized vectors in Redis
                
                # More robust threshold check
                if similarity >= threshold:
                    logger.info("cache_hit_l2_semantic", similarity=similarity, latency=time.time()-start_time)
                    return{
                        "response" : top_hit.response,
                        "similarity" : similarity,
                        "source" : "l2_semantic"
                    }, query_vector
                    
            logger.info("cache_miss", latency=time.time()-start_time)
            return None
        
        except Exception as e:
            # FAIL OPEN: If cache errors, log it and return None so user still gets an answer
            logger.error("cache_get_error", error=str(e), exc_info=True)
            return None

    async def  set(
        self,
        query:str,
        response: str,
        user_id: str = "global",
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Write to cache (L1 + L2).
        Use background task or fire-and-forget logic in production usually.
        but here we await for safety.
        """
        try:
            # 1. Store L1 Exact Match
            exact_key = f"{self.exact_prefix}:{self._hash_query(query, user_id)}"
            l1_payload = {
                "query" : query,
                "response" : response,
                "metadata" : metadata,
                "cached_at" : datetime.utcnow().isoformat()
            }
            # Fire L1 write (fast)
            await self.redis_client.setex(
                exact_key,
                self.ttl_seconds,
                json.dumps(l1_payload)
            )
            
            # 2. Store L2 Semantic Entry
            # We need the vector again.
            # In a real app, you might pass the vector from 'get' if you calculate it there to save $$
            vector = embedding
            if vector is None:
                # If we hit L1 Cache previosly, or called set() manually without get()
                logger.debug("Calculating_embedding_in_set", reason = "vector_not_provided")
                vector = await self._safe_embed(query)
                
            if not vector:
                return
            
            vector_blob = np.array(vector, dtype=np.float32).tobytes()
            semantic_key = f"{self.semantic_prefix}:{self._hash_query(query, user_id)}"
            
            mapping = {
                "user_id": user_id,
                "query" : query,
                "response": response,
                "embedding" : vector_blob,
                "created_at" : str(time.time())
            }
            
            if metadata:
                mapping["metadata"] = json.dumps(metadata)
                
            async with self.redis_client.pipeline(transaction=True) as pipe:
                await pipe.hset(semantic_key, mapping=mapping)
                await pipe.expire(semantic_key, self.ttl_seconds)
                await pipe.execute()
            
            logger.info("cache_set_success", user_id=user_id)
            
        except Exception as e:    
            logger.error("cache_set_error", error= str(e))
    
