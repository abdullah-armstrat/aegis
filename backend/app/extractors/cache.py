"""A tiny on-disk cache keyed by a hash of the input (dev-speed helper).

Slow calls — the captioner, the LLM reasoner, and (later) a live reverse-image API — are
cached by a SHA-256 of their input so that iterating on fusion rules does not re-run a
multi-second model call every time. This is purely a development-speed device; it has no
effect on the architecture and is safe to delete. The reverse-image *history index* is a
separate, intentional, committed artefact — this one is a transient dev cache
under ``backend/.cache`` (git-ignored).

Usage::

    cache = JsonCache("llm")
    key = cache.key(prompt, model)
    hit = cache.get(key)
    if hit is None:
        hit = expensive_call(...)
        cache.set(key, hit)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Transient dev cache root (git-ignored via backend/.cache/).
_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"


class JsonCache:
    """A namespaced JSON file cache. Each entry is one file named by its key hash."""

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        self._dir = (root or _CACHE_ROOT) / namespace
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        """Stable SHA-256 over the given parts (order-sensitive)."""
        h = hashlib.sha256()
        for part in parts:
            h.update(repr(part).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt cache entry is a miss, never an error

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(json.dumps(value), encoding="utf-8")
        except (TypeError, OSError):
            pass  # caching is best-effort; never break the caller
