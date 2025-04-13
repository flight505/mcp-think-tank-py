#!/usr/bin/env python3
"""
Embeddings module for vector-based retrieval of semantic memory
"""
import logging
import hashlib
from typing import Dict, List, Optional, Any
import time

import numpy as np

from src.tools.cache import LRUCache

logger = logging.getLogger("mcp-think-tank.embeddings")

class EmbeddingProvider:
    """
    Provides embedding generation and similarity calculation for semantic search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_capacity: int = 10000, cache_ttl: Optional[int] = None):
        """
        Initialize the embedding provider
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_capacity: Maximum number of embeddings to cache (default: 10000)
            cache_ttl: Optional time-to-live in seconds for cache entries (default: None)
        """
        self.model_name = model_name
        self.model = None
        # Initialize embedding cache
        self.embedding_cache = LRUCache[str, List[float]](capacity=cache_capacity, ttl=cache_ttl)
        self.batch_size = 64  # Batch size for bulk embedding operations
        self._init_model()
    
    def _init_model(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except ImportError:
            logger.warning("Failed to import sentence-transformers. Semantic search will be unavailable.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Generate a hash for the text to use as cache key
        
        Args:
            text: Text to hash
            
        Returns:
            SHA-256 hash of the text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for the given text
        
        Args:
            text: The text to encode
            
        Returns:
            List of floats representing the embedding, or None if generation fails
        """
        if not self.model:
            return None
            
        # Generate cache key
        cache_key = self._get_text_hash(text)
        
        # Check cache first
        cached_embedding = self.embedding_cache.get(cache_key)
        if cached_embedding is not None:
            return cached_embedding
            
        try:
            # Generate new embedding
            start_time = time.time()
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()  # Convert numpy array to list for JSON serialization
            
            # Add to cache
            self.embedding_cache.put(cache_key, embedding_list)
            
            # Log generation time for performance monitoring
            generation_time = time.time() - start_time
            logger.debug(f"Generated embedding in {generation_time:.4f}s")
            
            return embedding_list
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Dictionary mapping text hashes to embeddings
        """
        if not self.model:
            return {self._get_text_hash(text): None for text in texts}
            
        results = {}
        texts_to_encode = []
        cache_keys = []
        
        # Check cache first for each text
        for text in texts:
            cache_key = self._get_text_hash(text)
            cached_embedding = self.embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                # Use cached embedding
                results[cache_key] = cached_embedding
            else:
                # Add to list for batch encoding
                texts_to_encode.append(text)
                cache_keys.append(cache_key)
        
        # If there are texts to encode
        if texts_to_encode:
            try:
                # Process in batches
                all_embeddings = []
                for i in range(0, len(texts_to_encode), self.batch_size):
                    batch_texts = texts_to_encode[i:i+self.batch_size]
                    start_time = time.time()
                    batch_embeddings = self.model.encode(batch_texts)
                    batch_time = time.time() - start_time
                    logger.debug(f"Generated {len(batch_texts)} embeddings in batch in {batch_time:.4f}s")
                    all_embeddings.extend(batch_embeddings)
                
                # Store results and update cache
                for i, (cache_key, embedding) in enumerate(zip(cache_keys, all_embeddings)):
                    embedding_list = embedding.tolist()
                    self.embedding_cache.put(cache_key, embedding_list)
                    results[cache_key] = embedding_list
                    
            except Exception as e:
                logger.error(f"Failed to generate embeddings batch: {e}")
                # Set None for all uncached texts
                for cache_key in cache_keys:
                    if cache_key not in results:
                        results[cache_key] = None
        
        return results
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity as a float between -1 and 1
        """
        try:
            # Convert lists to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_closest(self, query_embedding: List[float], 
                    candidates: Dict[str, List[float]], 
                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find the closest candidates to the query embedding
        
        Args:
            query_embedding: The query embedding
            candidates: Dictionary mapping identifiers to embeddings
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with id and similarity sorted by similarity (descending)
        """
        results = []
        
        for id, embedding in candidates.items():
            similarity = self.compute_similarity(query_embedding, embedding)
            results.append({"id": id, "similarity": similarity})
        
        # Sort by similarity (descending)
        # Using a separate function to handle mypy type checking
        def get_similarity(item: Dict[str, Any]) -> float:
            return item["similarity"]
            
        results.sort(key=get_similarity, reverse=True)
        
        # Return top matches
        return results[:limit]
    
    def search_by_text(self, query: str, 
                      candidates: Dict[str, Dict[str, Any]], 
                      embedding_key: str = "embedding",
                      limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for candidates using a text query
        
        Args:
            query: The text query to search for
            candidates: Dictionary mapping identifiers to objects with embeddings
            embedding_key: Key in the candidate objects that contains the embedding
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with id, data, and similarity sorted by similarity (descending)
        """
        if not self.model:
            logger.warning("Embedding model not available, unable to perform semantic search")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
            results = []
            
            for id, data in candidates.items():
                if embedding_key not in data or not data[embedding_key]:
                    continue
                    
                similarity = self.compute_similarity(query_embedding, data[embedding_key])
                results.append({"id": id, "data": data, "similarity": similarity})
            
            # Sort by similarity (descending)
            def get_similarity(item: Dict[str, Any]) -> float:
                return item["similarity"]
                
            results.sort(key=get_similarity, reverse=True)
            
            # Return top matches
            return results[:limit]
        except Exception as e:
            logger.error(f"Search by text failed: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding cache
        
        Returns:
            Dictionary with cache statistics
        """
        return self.embedding_cache.get_stats()
    
    def clear_cache(self) -> None:
        """
        Clear the embedding cache
        """
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing unused resources
        """
        import gc
        gc.collect()
        logger.info("Memory optimization performed") 