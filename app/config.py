"""
Application Configuration
Uses Pydantic Settings for environment-based oconfiguration
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file= ".env",
        env_file_encoding= "utf-8",
        case_sensitive=True
    )
    
    # Application
    APP_NAME: str = "CRM AI Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="CROS allowed origins")
    
    # LLM
    OPENROUTER_API_KEY :str = Field(..., description="Openrrouter API key")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM : str = Field(default="gpt-4o-mini", description="The generator model")
    EMBEDDING_MODEL : str = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION: int = 384
    
    # LangSmith (Monitoring)
    LANGSMITH_API_KEY : Optional[str] = Field(default=None, description="LangSmith API key")
    LANGSMITH_PROJECT : str = Field(default="crm-ai-assistant", description="LangSmith API key for tracing")
    LANGSMITH_ENDPOINT: str= Field(default="https://api.smith.langchain.com", description="LangSmith endpoint")
    
    # Database (PostgreSQL for checkpointing)
    POSTGRES_HOST: str = Field(default="localhost", description="PostgerSQL host")
    POSTGRS_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_USER: str = Field(default="postgres", description="PostgreSQL user")
    POSTGRES_PASSWORD: str = Field(default="postgres", description="PostgreSQL Password")
    POSTGRES_DB: str = Field(default="crm_ai", description="PostgreSQL database name")
    
    @property
    def POSTGRES_URL(self) -> str:
        """Build PostgreSQL connection URL"""
        return f"posgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRS_PORT}/{self.POSTGRES_DB}"
    
    #Redis (semantic caching and session management)
    REDIS_HOST: str = Field(default="localhost", description="Redis host")
    REDIS_PORT: int = Field(default=6379, description="Redis port")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis Password")
    
    @property
    def REDIS_URL(self) -> str:
        """Build Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # vector Store (Pinecone/Qdrant)
    VECTOR_STORE_TYPE: str = Field(default="pinecone", description="vector store type: pinecone, qdrant")
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", description="Pinecone environment")
    PINECONE_INDEX_NAME: str = Field(default="crm-ai-knowledge", description="Pinecone index name")
    
    # Qdrant
    QDRANT_URL: Optional[str] = Field(default=None, description="Qdrant server URL")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant API key")
    
    # CRM Integration
    CRM_API_URL: str = Field(..., description="CRM API base URL")
    CRM_API_KEY: str = Field(..., description="CRM API key")
    CRM_TYPE: str = Field(default="generic", description="CRM type: salesforce, hubspot, pipedrive, generic")
    
    # Webhooks
    WEBHOOK_SECRET: Optional[str] = Field(default=None, description="Webhook signature secret")
    
    # Semantic Cache
    SEMANTIC_CACHE_ENABLED: bool = Field(default=True, description="Enable semantic caching")
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = Field(default=0.95, description="Similarity threshold for cache hits")
    SEMANTIC_CACHE_TTL: int = Field(default=86400, description="Cache TTL in seconds (default: 24 hours)")

    # Hybrid Search
    HYBRID_SEARCH_DENSE_WEIGHT: float = Field(default=0.7, description="Weight for dense vector search")
    HYBRID_SEARCH_SPARSE_WEIGHT: float = Field(default=0.3, description="Weight for sparse BM25 search")
    HYBRID_SEARCH_TOP_K: int = Field(default=10, description="Number of results to retrieve")
    
    # Workflow
    MAX_WORKFLOW_ITERATIONS: int = Field(default=10, description="Maximum workflow iterations")
    WORKFLOW_TIMEOUT: int = Field(default=300, description="Workflow timeout in seconds")
    
    # AWS (if deploying to AWS)
    AWS_REGION: str = Field(default="us-east-1", description="AWS region")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS access key")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS secret key")
    S3_BUCKET: Optional[str] = Field(default=None, description="S3 bucket for data storage")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Requests per minute per user")

    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics")
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    
    # Feature Flags
    ENABLE_RAGAS_EVALUATION: bool = Field(default=False, description="Enable Ragas evaluation in production")
    ENABLE_COT_LOGGING: bool = Field(default=True, description="Log chain of thought reasoning")


# Global settings instance
settings = Settings()