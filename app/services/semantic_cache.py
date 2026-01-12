"""
Semantic Cache for reducing LLM costs
Caches responses based on semantic similarity rather than exact matches.
"""
import hashlib
import json
from typing import Optional, Dict, Any
from redis.asyncio import Redis
from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import logging 

logger = logging.getLogger(__name__)


