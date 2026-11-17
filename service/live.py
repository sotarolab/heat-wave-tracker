"""An in-memory cache for values that are fetched or rendered on request.

A route reads a public feed or draws an image at most once per `ttl_s`. A failed refetch
serves the last good value and is retried a minute later, so a slow feed degrades the page
to slightly old data instead of an error.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

_cache: dict[str, tuple[float, Any]] = {}
# One cycle is about 130 field images of a few tens of kilobytes each, plus the feeds.
_CACHE_MAX = 512
FOREVER = float("inf")
RETRY_S = 60.0
_failed: dict[str, float] = {}


class Unavailable(RuntimeError):
    """The value could not be fetched a moment ago and is not being asked for again yet."""


def cached(key: str, ttl_s: float, fetch: Callable[[], Any]) -> Any:
    """fetch() at most once per ttl_s; raises only when there is nothing to serve.

    A fetch that fails with nothing held is not tried again for RETRY_S: the feeds answer a
    burst of requests with 429, and asking again at once only extends the refusal.
    """
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < ttl_s:
        return hit[1]
    if not hit and now - _failed.get(key, 0.0) < RETRY_S:
        raise Unavailable(key)
    try:
        value = fetch()
    except Exception:
        if not hit:
            _failed[key] = now
            raise
        _cache[key] = (now - ttl_s + RETRY_S, hit[1])
        return hit[1]
    _failed.pop(key, None)
    _cache[key] = (now, value)
    if len(_cache) > _CACHE_MAX:
        del _cache[min(_cache, key=lambda k: _cache[k][0])]
    return value


def clear() -> None:
    """Empty the response cache and the failure record; the tests use it."""
    _cache.clear()
    _failed.clear()
