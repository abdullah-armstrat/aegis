"""A small on-disk JSON cache keyed by a hash of the input.

Slow calls such as the captioner and the LLM reasoner are cached by the SHA-256 of their input,
so re-running the fusion rules during development does not repeat multi-second model calls.
It lives in ``backend/.cache`` (git-ignored) and is safe to delete.

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

# cache root, backend/.cache/ (git-ignored)
_CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache"


class JsonCache:
    """A JSON file cache in its own folder. Each entry is one file named by its key."""

    def __init__(self, namespace: str, root: Path | None = None) -> None:
        self._dir = (root or _CACHE_ROOT) / namespace
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        """SHA-256 hex digest of the given parts, in order."""
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
            return None  # a corrupt entry just counts as a miss

    def set(self, key: str, value: Any) -> None:
        try:
            self._path(key).write_text(json.dumps(value), encoding="utf-8")
        except (TypeError, OSError):
            pass  # caching is best-effort, so a failed write is ignored
