import pandas as pd
import numpy as np
from datetime import datetime

class DataCache:
    """In-memory cache for data and models"""

    def __init__(self):
        self._cache = {}
        self._timestamps = {}

    def set(self, key, value, ttl=None):
        """Store value in cache"""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()

    def get(self, key):
        """Retrieve value from cache"""
        if key in self._cache:
            return self._cache[key]
        return None

    def exists(self, key):
        """Check if key exists in cache"""
        return key in self._cache

    def clear(self, key=None):
        """Clear cache"""
        if key:
            if key in self._cache:
                del self._cache[key]
                del self._timestamps[key]
        else:
            self._cache.clear()
            self._timestamps.clear()

    def get_all_keys(self):
        """Get all cached keys"""
        return list(self._cache.keys())

    def size(self):
        """Get cache size"""
        return len(self._cache)

# Global cache instance
cache = DataCache()
