#!/usr/bin/env python3
"""
Embedding utilities for MCP Think Tank
Provides text embedding functionality for semantic search and similarity
"""
import os
import logging
from typing import List, Union, Optional, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .cache import LRUCache

logger = logging.getLogger("mcp-think-tank.embedding")

# Cache for embedding models to avoid reloading
MODEL_CACHE = LRUCache[str, SentenceTransformer](capacity=3)

# Default model for embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: Optional[str] = None,
    cache: bool = True
) -> SentenceTransformer:
    """
    Get an embedding model by name

    Args:
        model_name: Name of the model to use (default: all-MiniLM-L6-v2)
        cache: Whether to cache the model

    Returns:
        SentenceTransformer model
    """
    model_name = model_name or DEFAULT_MODEL

    # Check if model is in cache
    if cache and MODEL_CACHE.contains(model_name):
        return MODEL_CACHE.get(model_name)  # type: ignore

    # Load the model
    try:
        model = SentenceTransformer(model_name)
        
        # Cache the model if requested
        if cache:
            MODEL_CACHE.put(model_name, model)
            
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}")
        raise


def get_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts

    Args:
        texts: List of text strings to embed
        model_name: Name of the model to use (default: all-MiniLM-L6-v2)
        normalize: Whether to normalize the embeddings

    Returns:
        Numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])

    model = get_embedding_model(model_name)
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize
    )
    
    return embeddings


def compute_similarity(
    query: Union[str, np.ndarray],
    corpus: Union[List[str], np.ndarray],
    model_name: Optional[str] = None,
    top_k: Optional[int] = None
) -> List[Tuple[int, float]]:
    """
    Compute similarity between a query and a corpus

    Args:
        query: Query text or embedding
        corpus: List of text strings or embeddings
        model_name: Name of the model to use
        top_k: Number of top results to return (None for all)

    Returns:
        List of (index, score) tuples sorted by descending score
    """
    model = get_embedding_model(model_name)
    
    # Convert query to embedding if it's a string
    if isinstance(query, str):
        query_embedding = model.encode(query, normalize_embeddings=True)
    else:
        query_embedding = query
        
    # Convert corpus to embeddings if it's a list of strings
    if isinstance(corpus, list) and all(isinstance(item, str) for item in corpus):
        corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
    else:
        corpus_embeddings = corpus
        
    # Compute cosine similarities
    similarities = corpus_embeddings @ query_embedding.T
    
    # Create list of (index, score) tuples
    scored_results = [(i, float(score)) for i, score in enumerate(similarities)]
    
    # Sort by descending score
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k results if specified
    if top_k is not None:
        return scored_results[:top_k]
        
    return scored_results


def semantic_search(
    query: str,
    texts: List[str],
    model_name: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on a list of texts

    Args:
        query: Search query
        texts: List of texts to search
        model_name: Name of the model to use
        top_k: Number of top results to return

    Returns:
        List of dictionaries with 'index', 'text', and 'score' keys
    """
    if not texts:
        return []
        
    # Compute similarities
    results = compute_similarity(query, texts, model_name, top_k)
    
    # Format results
    return [
        {
            "index": idx,
            "text": texts[idx],
            "score": score
        }
        for idx, score in results
    ] 