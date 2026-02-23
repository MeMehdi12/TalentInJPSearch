"""
API Enhancements Module
========================

Production-ready API enhancements:
- Rate limiting per API key / IP
- Response caching for common queries
- Search analytics tracking
- OpenAI daily spending cap
- Retry logic with exponential backoff
- Comprehensive error logging

Author: Talentin AI Search Team
Version: 1.0.0
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Optional, Any, Callable
from functools import wraps
from collections import defaultdict
from datetime import datetime, date
from dataclasses import dataclass, field
from threading import Lock

from fastapi import Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# RATE LIMITING
# =============================================================================

@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting"""
    tokens: float
    last_update: float
    max_tokens: int
    refill_rate: float  # tokens per second


class RateLimiter:
    """
    Token bucket rate limiter with per-key limits.
    
    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        if not limiter.allow("api_key_123"):
            raise HTTPException(429, "Rate limit exceeded")
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300  # Clean old buckets every 5 min
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.refill_rate = requests_per_minute / 60.0
        self.buckets: Dict[str, RateLimitBucket] = {}
        self.lock = Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
    
    def _cleanup_old_buckets(self) -> None:
        """Remove buckets that haven't been used in a while"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            # Remove buckets inactive for > 10 minutes
            stale_keys = [
                key for key, bucket in self.buckets.items()
                if now - bucket.last_update > 600
            ]
            for key in stale_keys:
                del self.buckets[key]
            
            self.last_cleanup = now
            logger.info(f"Rate limiter cleanup: removed {len(stale_keys)} stale buckets")
    
    def allow(self, key: str) -> bool:
        """
        Check if request is allowed for given key.
        
        Args:
            key: Client identifier (API key, IP, etc.)
        
        Returns:
            True if request is allowed, False if rate limited
        """
        self._cleanup_old_buckets()
        
        now = time.time()
        
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = RateLimitBucket(
                    tokens=self.burst_size,
                    last_update=now,
                    max_tokens=self.burst_size,
                    refill_rate=self.refill_rate
                )
            
            bucket = self.buckets[key]
            
            # Refill tokens based on time elapsed
            elapsed = now - bucket.last_update
            bucket.tokens = min(
                bucket.max_tokens,
                bucket.tokens + (elapsed * bucket.refill_rate)
            )
            bucket.last_update = now
            
            # Check if request can be served
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True
            
            return False
    
    def get_wait_time(self, key: str) -> float:
        """Get seconds until next request is allowed"""
        with self.lock:
            if key not in self.buckets:
                return 0.0
            
            bucket = self.buckets[key]
            if bucket.tokens >= 1:
                return 0.0
            
            needed = 1 - bucket.tokens
            return needed / bucket.refill_rate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                "active_buckets": len(self.buckets),
                "requests_per_minute": self.requests_per_minute,
                "burst_size": self.burst_size
            }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        rpm = int(os.getenv("RATE_LIMIT_RPM", "60"))
        burst = int(os.getenv("RATE_LIMIT_BURST", "10"))
        _rate_limiter = RateLimiter(requests_per_minute=rpm, burst_size=burst)
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract client identifier
        api_key = request.headers.get("X-API-Key")
        client_ip = request.client.host if request.client else "unknown"
        client_id = api_key or client_ip
        
        limiter = get_rate_limiter()
        
        if not limiter.allow(client_id):
            wait_time = limiter.get_wait_time(client_id)
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": round(wait_time, 2),
                    "message": f"Please wait {wait_time:.1f} seconds before retrying"
                },
                headers={"Retry-After": str(int(wait_time) + 1)}
            )
        
        response = await call_next(request)
        return response


# =============================================================================
# RESPONSE CACHING
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    expires_at: float
    hits: int = 0


class ResponseCache:
    """
    In-memory response cache with TTL.
    
    Usage:
        cache = ResponseCache(ttl_seconds=60)
        
        # Check cache
        cached = cache.get(key)
        if cached:
            return cached
        
        # Compute result
        result = expensive_operation()
        
        # Store in cache
        cache.set(key, result)
        return result
    """
    
    def __init__(
        self,
        ttl_seconds: int = 60,
        max_entries: int = 1000,
        cleanup_interval: int = 300
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
        
        # Stats
        self.total_hits = 0
        self.total_misses = 0
    
    def _make_key(self, data: Any) -> str:
        """Generate cache key from data"""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _cleanup(self) -> None:
        """Remove expired entries"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            expired = [
                key for key, entry in self.cache.items()
                if entry.expires_at < now
            ]
            for key in expired:
                del self.cache[key]
            
            self.last_cleanup = now
            
            if expired:
                logger.info(f"Cache cleanup: removed {len(expired)} expired entries")
    
    def _evict_if_needed(self) -> None:
        """Evict least-used entries if cache is full"""
        if len(self.cache) < self.max_entries:
            return
        
        # Remove least-hit entries
        with self.lock:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].hits
            )
            # Remove bottom 10%
            to_remove = len(self.cache) // 10
            for key, _ in sorted_entries[:to_remove]:
                del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired"""
        with self.lock:
            # Cleanup inside lock to prevent race condition
            now = time.time()
            if now - self.last_cleanup >= self.cleanup_interval:
                expired = [
                    k for k, entry in self.cache.items()
                    if entry.expires_at < now
                ]
                for k in expired:
                    del self.cache[k]
                self.last_cleanup = now
            
            if key not in self.cache:
                self.total_misses += 1
                return None
            
            entry = self.cache[key]
            
            if entry.expires_at < time.time():
                del self.cache[key]
                self.total_misses += 1
                return None
            
            entry.hits += 1
            self.total_hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        self._evict_if_needed()
        
        ttl = ttl or self.ttl_seconds
        expires_at = time.time() + ttl
        
        with self.lock:
            self.cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
                hits=0
            )
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.total_hits + self.total_misses
            hit_rate = (self.total_hits / total * 100) if total > 0 else 0
            
            return {
                "entries": len(self.cache),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "hit_rate_percent": round(hit_rate, 2)
            }


# Global cache instances
_search_cache: Optional[ResponseCache] = None
_facet_cache: Optional[ResponseCache] = None


def get_search_cache() -> ResponseCache:
    """Get search response cache"""
    global _search_cache
    if _search_cache is None:
        ttl = int(os.getenv("SEARCH_CACHE_TTL", "60"))
        max_entries = int(os.getenv("SEARCH_CACHE_MAX", "1000"))
        _search_cache = ResponseCache(ttl_seconds=ttl, max_entries=max_entries)
    return _search_cache


def get_facet_cache() -> ResponseCache:
    """Get facet response cache"""
    global _facet_cache
    if _facet_cache is None:
        ttl = int(os.getenv("FACET_CACHE_TTL", "300"))  # 5 min default
        max_entries = int(os.getenv("FACET_CACHE_MAX", "100"))
        _facet_cache = ResponseCache(ttl_seconds=ttl, max_entries=max_entries)
    return _facet_cache


# =============================================================================
# SEARCH ANALYTICS
# =============================================================================

@dataclass
class SearchAnalytics:
    """Track search analytics"""
    query: str
    filters: Dict[str, Any]
    result_count: int
    took_ms: int
    timestamp: datetime
    client_id: Optional[str] = None
    source: str = "api"


class AnalyticsTracker:
    """
    Track and analyze search patterns.
    
    Useful for:
    - Understanding popular queries
    - Identifying slow queries
    - Monitoring search quality
    """
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.searches: list = []
        self.lock = Lock()
        
        # Aggregated stats
        self.total_searches = 0
        self.total_latency_ms = 0
        self.query_counts: Dict[str, int] = defaultdict(int)
        self.skill_counts: Dict[str, int] = defaultdict(int)
        self.location_counts: Dict[str, int] = defaultdict(int)
    
    def track(self, analytics: SearchAnalytics) -> None:
        """Track a search event"""
        with self.lock:
            # Keep bounded history
            if len(self.searches) >= self.max_entries:
                self.searches.pop(0)
            
            self.searches.append(analytics)
            
            # Update aggregates
            self.total_searches += 1
            self.total_latency_ms += analytics.took_ms
            
            # Track query patterns
            query_words = analytics.query.lower().split()
            for word in query_words:
                if len(word) > 2:
                    self.query_counts[word] += 1
            
            # Track filter usage
            skills = analytics.filters.get("skills", {}).get("must_have", [])
            for skill in skills:
                self.skill_counts[skill.lower()] += 1
            
            location = analytics.filters.get("location", {})
            if location.get("city"):
                self.location_counts[location["city"].lower()] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics summary"""
        with self.lock:
            avg_latency = (
                self.total_latency_ms / self.total_searches
                if self.total_searches > 0 else 0
            )
            
            # Top items
            top_skills = sorted(
                self.skill_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            top_locations = sorted(
                self.location_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            top_query_words = sorted(
                self.query_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            
            return {
                "total_searches": self.total_searches,
                "avg_latency_ms": round(avg_latency, 2),
                "top_skills": dict(top_skills),
                "top_locations": dict(top_locations),
                "top_query_words": dict(top_query_words),
                "recent_searches": len(self.searches)
            }
    
    def get_slow_queries(self, threshold_ms: int = 500) -> list:
        """Get queries that took longer than threshold"""
        with self.lock:
            return [
                {
                    "query": s.query,
                    "took_ms": s.took_ms,
                    "result_count": s.result_count,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in self.searches
                if s.took_ms > threshold_ms
            ][-50:]  # Last 50 slow queries


# Global analytics tracker
_analytics_tracker: Optional[AnalyticsTracker] = None


def get_analytics_tracker() -> AnalyticsTracker:
    """Get global analytics tracker"""
    global _analytics_tracker
    if _analytics_tracker is None:
        _analytics_tracker = AnalyticsTracker()
    return _analytics_tracker


# =============================================================================
# OPENAI SPENDING CAP
# =============================================================================

class SpendingCap:
    """
    Track and enforce OpenAI API spending limits.
    
    Default: $10/day
    """
    
    def __init__(self, daily_limit_usd: float = 10.0):
        self.daily_limit_usd = daily_limit_usd
        self.daily_spend: Dict[str, float] = {}  # date -> spend
        self.lock = Lock()
    
    def _today_key(self) -> str:
        """Get today's date key"""
        return date.today().isoformat()
    
    def add_spend(self, amount_usd: float) -> None:
        """Record spending"""
        with self.lock:
            key = self._today_key()
            self.daily_spend[key] = self.daily_spend.get(key, 0.0) + amount_usd
            
            # Clean old entries
            old_keys = [k for k in self.daily_spend.keys() if k < key]
            for old_key in old_keys:
                del self.daily_spend[old_key]
    
    def get_today_spend(self) -> float:
        """Get today's total spend"""
        with self.lock:
            return self.daily_spend.get(self._today_key(), 0.0)
    
    def can_spend(self, estimated_cost: float = 0.001) -> bool:
        """Check if we can make another API call"""
        today_spend = self.get_today_spend()
        return (today_spend + estimated_cost) <= self.daily_limit_usd
    
    def get_remaining(self) -> float:
        """Get remaining budget for today"""
        return max(0, self.daily_limit_usd - self.get_today_spend())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get spending statistics"""
        with self.lock:
            today_spend = self.get_today_spend()
            return {
                "daily_limit_usd": self.daily_limit_usd,
                "today_spend_usd": round(today_spend, 4),
                "remaining_usd": round(self.daily_limit_usd - today_spend, 4),
                "percent_used": round((today_spend / self.daily_limit_usd) * 100, 1)
            }


# Global spending cap
_spending_cap: Optional[SpendingCap] = None


def get_spending_cap() -> SpendingCap:
    """Get global spending cap"""
    global _spending_cap
    if _spending_cap is None:
        limit = float(os.getenv("OPENAI_DAILY_LIMIT_USD", "10.0"))
        _spending_cap = SpendingCap(daily_limit_usd=limit)
    return _spending_cap


def check_spending_cap(estimated_cost: float = 0.001) -> None:
    """
    Check spending cap and raise exception if exceeded.
    
    Raises:
        HTTPException: If daily spending limit reached
    """
    cap = get_spending_cap()
    if not cap.can_spend(estimated_cost):
        stats = cap.get_stats()
        logger.warning(f"OpenAI daily spending cap reached: ${stats['today_spend_usd']:.4f}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Daily OpenAI spending limit reached",
                "limit_usd": stats["daily_limit_usd"],
                "spent_usd": stats["today_spend_usd"],
                "resets_at": "00:00 UTC"
            }
        )


# =============================================================================
# RETRY LOGIC
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_retries=3)
        def unreliable_operation():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        raise
                    
                    # Exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


async def async_retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """
    Async retry with exponential backoff.
    
    Usage:
        result = await async_retry_with_backoff(unreliable_async_func)
    """
    import asyncio
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(f"Async function failed after {max_retries + 1} attempts: {e}")
                raise
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
    
    raise last_exception


# =============================================================================
# ERROR LOGGING
# =============================================================================

class ErrorLogger:
    """
    Comprehensive error logging with context.
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recent_errors: list = []
        self.max_recent = 100
        self.lock = Lock()
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ) -> str:
        """
        Log an error with full context.
        
        Returns:
            Error ID for tracking
        """
        import traceback
        import uuid
        
        error_id = str(uuid.uuid4())[:8]
        error_type = type(error).__name__
        
        with self.lock:
            self.error_counts[error_type] += 1
            
            error_entry = {
                "id": error_id,
                "type": error_type,
                "message": str(error),
                "timestamp": datetime.now().isoformat(),
                "context": context or {},
                "traceback": traceback.format_exc()
            }
            
            if request:
                error_entry["request"] = {
                    "method": request.method,
                    "url": str(request.url),
                    "client": request.client.host if request.client else None
                }
            
            # Keep bounded history
            if len(self.recent_errors) >= self.max_recent:
                self.recent_errors.pop(0)
            
            self.recent_errors.append(error_entry)
        
        # Log to standard logging
        logger.error(
            f"[{error_id}] {error_type}: {error}",
            extra={"error_id": error_id, "context": context}
        )
        
        return error_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        with self.lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_counts_by_type": dict(self.error_counts),
                "recent_error_count": len(self.recent_errors)
            }
    
    def get_recent_errors(self, limit: int = 20) -> list:
        """Get recent errors"""
        with self.lock:
            return self.recent_errors[-limit:]


# Global error logger
_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get global error logger"""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger


# =============================================================================
# REQUEST LOGGING MIDDLEWARE
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Generate request ID
        request_id = hashlib.md5(
            f"{time.time()}{request.url}".encode()
        ).hexdigest()[:8]
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - Started"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"{response.status_code} ({duration_ms}ms)"
            )
            
            # Add timing header
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_logger = get_error_logger()
            error_id = error_logger.log_error(e, request=request)
            
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Error [{error_id}] ({duration_ms}ms)"
            )
            
            raise


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cache_key_from_query(query: Dict[str, Any]) -> str:
    """Generate cache key from search query"""
    # Normalize query for consistent caching
    normalized = json.dumps(query, sort_keys=True, default=str)
    return hashlib.md5(normalized.encode()).hexdigest()


def get_client_id(request: Request) -> str:
    """Extract client identifier from request"""
    # Prefer API key, fall back to IP
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Rate limiting
    "RateLimiter",
    "get_rate_limiter",
    "RateLimitMiddleware",
    
    # Caching
    "ResponseCache",
    "get_search_cache",
    "get_facet_cache",
    "cache_key_from_query",
    
    # Analytics
    "SearchAnalytics",
    "AnalyticsTracker",
    "get_analytics_tracker",
    
    # Spending cap
    "SpendingCap",
    "get_spending_cap",
    "check_spending_cap",
    
    # Retry
    "retry_with_backoff",
    "async_retry_with_backoff",
    
    # Error logging
    "ErrorLogger",
    "get_error_logger",
    
    # Middleware
    "RequestLoggingMiddleware",
    
    # Helpers
    "get_client_id",
]
