#!/usr/bin/env python3
"""
Caching utilities for MCP Think Tank
Provides various caching mechanisms to optimize performance
"""
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, TypeVar, Generic, List, Callable, Tuple, OrderedDict as OrderedDictType
from collections import OrderedDict


K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Least Recently Used (LRU) Cache implementation
    Efficiently caches items, evicting least recently used items when capacity is reached
    """

    def __init__(self, capacity: int = 100, ttl: Optional[int] = None):
        """
        Initialize LRU Cache

        Args:
            capacity: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds (optional)
        """
        self.capacity = max(1, capacity)
        self.ttl = ttl
        self.cache: OrderedDictType[K, Tuple[V, float]] = OrderedDict()
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: K) -> Optional[V]:
        """
        Get an item from the cache

        Args:
            key: The key to look up

        Returns:
            The cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None

            value, timestamp = self.cache[key]

            # Check if expired
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                self.cache.pop(key)
                self.miss_count += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            return value

    def put(self, key: K, value: V) -> None:
        """
        Put an item in the cache

        Args:
            key: The key to store
            value: The value to cache
        """
        with self.lock:
            # Update or insert
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)

            # Evict oldest item if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def remove(self, key: K) -> bool:
        """
        Remove an item from the cache

        Args:
            key: The key to remove

        Returns:
            True if the key was removed, False otherwise
        """
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache"""
        with self.lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0

    def contains(self, key: K) -> bool:
        """
        Check if a key is in the cache (and not expired)

        Args:
            key: The key to check

        Returns:
            True if the key exists and is not expired, False otherwise
        """
        with self.lock:
            if key not in self.cache:
                return False

            # Check if expired
            if self.ttl is not None:
                _, timestamp = self.cache[key]
                if time.time() - timestamp > self.ttl:
                    self.cache.pop(key)
                    return False

            return True

    def get_or_compute(self, key: K, compute_func: Callable[[], V]) -> V:
        """
        Get a value from the cache or compute and store it if not present

        Args:
            key: The key to look up
            compute_func: Function to compute the value if not in cache

        Returns:
            The cached or computed value
        """
        with self.lock:
            value = self.get(key)
            if value is None:
                value = compute_func()
                self.put(key, value)
            return value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_access = self.hit_count + self.miss_count
            hit_ratio = (
                self.hit_count / total_access if total_access > 0 else 0
            )
            return {
                "capacity": self.capacity,
                "size": len(self.cache),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": hit_ratio,
                "ttl": self.ttl,
            }

    def get_keys(self) -> List[K]:
        """
        Get all keys in the cache

        Returns:
            List of keys
        """
        with self.lock:
            return list(self.cache.keys())

    def __len__(self) -> int:
        """Get the number of items in the cache"""
        with self.lock:
            return len(self.cache)

    def __contains__(self, key: object) -> bool:
        """Check if a key is in the cache"""
        try:
            return self.contains(key)  # type: ignore
        except Exception:
            return False 