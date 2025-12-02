"""
API Utilities with Best Practices.

Includes retry logic, caching, rate limiting, and logging.
"""
import time
import functools
import logging
from typing import Callable, Optional, Any
from datetime import datetime, timedelta
import hashlib
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APICache:
    """Simple in-memory cache for API responses."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time to live in seconds (default: 1 hour)
        """
        self.cache: dict = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return value
            else:
                # Expired, remove it
                del self.cache[key]
                logger.debug(f"Cache expired for key: {key[:16]}...")
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = (value, datetime.now())
        logger.debug(f"Cached value for key: {key[:16]}...")
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        logger.info("Cache cleared")


# Global cache instance
api_cache = APICache(ttl_seconds=3600)


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0,
                       backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def cached_api_call(ttl_seconds: int = 3600):
    """
    Decorator for caching API call results.
    
    Args:
        ttl_seconds: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = api_cache._generate_key(func.__name__, *args, **kwargs)
            
            # Check cache
            cached_result = api_cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            api_cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Need to wait
            oldest_call = self.calls[0]
            wait_time = self.time_window - (now - oldest_call) + 0.1  # Small buffer
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                # Clean up again after waiting
                now = time.time()
                self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        # Record this call
        self.calls.append(now)


def log_api_call(func: Callable) -> Callable:
    """Decorator for logging API calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Calling {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    return wrapper

