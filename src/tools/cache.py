#!/usr/bin/env python3
"""
Cache implementation for MCP Think Tank
Provides LRU (Least Recently Used) caching to improve performance
"""
import time
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, List, Callable, TypeVar, Generic

logger = logging.getLogger("mcp-think-tank.cache")

T = TypeVar('T')  # Generic type for cache values

class LRUCache(Generic[T]):
    """
    Least Recently Used (LRU) cache implementation
    
    Provides fast O(1) lookups with automatic eviction of least recently used items
    when capacity is reached. Supports optional Time-To-Live (TTL) for entries.
    """
    
    def __init__(self, capacity: int = 1000, ttl: Optional[int] = None):
        """
        Initialize the LRU Cache
        
        Args:
            capacity: Maximum number of items to store (default: 1000)
            ttl: Optional time-to-live in seconds for cache entries (default: None)
        """
        self.capacity = max(capacity, 1)  # Ensure capacity is at least 1
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
            
        value, timestamp = self.cache[key]
        
        # Check TTL if enabled
        if self.ttl is not None and time.time() - timestamp > self.ttl:
            # Remove expired item
            self.cache.pop(key)
            self.misses += 1
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    def put(self, key: str, value: T) -> None:
        """
        Put item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if already exists to update position
        if key in self.cache:
            self.cache.pop(key)
            
        # Add new item with current timestamp
        self.cache[key] = (value, time.time())
        
        # Remove oldest item if capacity exceeded
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
            
    def remove(self, key: str) -> bool:
        """
        Remove item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed, False if not found
        """
        if key in self.cache:
            self.cache.pop(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all items from cache"""
        self.cache.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "utilization": len(self.cache) / self.capacity if self.capacity > 0 else 0,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

    def contains(self, key: str) -> bool:
        """
        Check if key exists in cache and is not expired
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is not expired, False otherwise
        """
        if key not in self.cache:
            return False
            
        if self.ttl is not None:
            _, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                # Remove expired item
                self.cache.pop(key)
                return False
                
        return True
        
    def get_keys(self) -> List[str]:
        """
        Get all valid (non-expired) keys in the cache
        
        Returns:
            List of cache keys
        """
        if self.ttl is None:
            return list(self.cache.keys())
            
        # Filter out expired keys
        valid_keys = []
        current_time = time.time()
        
        for key, (_, timestamp) in list(self.cache.items()):
            if current_time - timestamp <= self.ttl:
                valid_keys.append(key)
            else:
                # Remove expired item
                self.cache.pop(key)
                
        return valid_keys
        
    def get_or_compute(self, key: str, compute_fn: Callable[[], T]) -> T:
        """
        Get value from cache or compute it if not found
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not in cache
            
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value
            
        # Compute value and cache it
        computed_value = compute_fn()
        self.put(key, computed_value)
        return computed_value 