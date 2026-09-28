"""LLM reasoner: asks the local LLM (through Ollama) whether the extracted texts agree.

The model only sees the bundle's text fields and is only asked whether they describe the same
thing and carry the same tone, never whether anything is true, which avoids its knowledge cutoff.
The whole prompt is kept in ``SUPPLIED_TEXT_ONLY_PROMPT`` so that rule is not lost in later edits.
This only adds to the rules. When ``use_llm`` is off or the model is unreachable, its check is
NOT_ASSESSED.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

import httpx

from app.config import get_settings
from app.extractors.cache import JsonCache

# --- The prompt. Keep its supplied-text-only rules if it is ever edited. ---
SUPPLIED_TEXT_ONLY_PROMPT = """\
You are comparing pieces of text that were automatically extracted from a single social
media post. Your ONLY job is to judge their relationship to each other.

STRICT RULES:
- Reason ONLY over the text supplied below. Do NOT use outside knowledge.
- NEVER decide whether any statement is true, false, real, or fake.
- Only judge: do these texts describe the same thing, and do they carry the same tone?

Supplied text:
- Caption: {caption}
- Scene description(s): {scene}
- On-screen text: {on_screen}

Answer with a single JSON object, no other text:
{{"same_subject": true|false, "same_tone": true|false, "explanation": "<one plain sentence>"}}
"""

_TIMEOUT_SECONDS = 30.0
# Loading the model is the slow part of the first call, so at start-up the server asks Ollama to
# load it (an empty prompt loads it without generating) and keep it loaded. The warm-up can wait
# longer than a normal request.
_WARM_UP_TIMEOUT_SECONDS = 180.0
_WARM_UP_KEEP_ALIVE = "30m"


@dataclass
class ReasonerVerdict:
    """The LLM's judgement of how the supplied texts relate (never a truth verdict)."""

    same_subject: bool | None
    same_tone: bool | None
    explanation: str
    available: bool  # False if the model is off or unreachable; the caller marks NOT_ASSESSED


# --------------------------------------------------------------------------- instrumentation
#
# Measurement only: nothing below does anything unless a script enters ``record_calls``, and the
# /analyze path never does. It keeps the raw model response, which is otherwise parsed and thrown
# away, so the evaluation scripts can measure output validity, latency and run-to-run stability.


@dataclass
class CallRecord:
    """One recorded reasoner call. Only the evaluation scripts read these."""

    validity: str  # clean | salvaged | coerced | unparseable | cache_hit | not_called
    available: bool
    latency_s: float | None
    raw_response: str | None
    error: str | None = None
    error_type: str | None = None
    is_timeout: bool = False
    same_subject: bool | None = None
    same_tone: bool | None = None
    cache_hit: bool = False
    detail: dict = field(default_factory=dict)


_probe: Callable[[CallRecord], None] | None = None
_bypass_cache = False


@contextmanager
def record_calls(sink: list[CallRecord], *, bypass_cache: bool = False) -> Iterator[list[CallRecord]]:
    """Record a :class:`CallRecord` for every reasoner call made inside the block.

    ``bypass_cache`` skips the cache for reads and writes, since a cache hit has no raw response
    or latency to measure. The previous state is restored on exit, even after an exception.
    """
    global _probe, _bypass_cache
    prev_probe, prev_bypass = _probe, _bypass_cache
    _probe, _bypass_cache = sink.append, bypass_cache
    try:
        yield sink
    finally:
        _probe, _bypass_cache = prev_probe, prev_bypass


def _emit(record: CallRecord) -> None:
    """Pass a record to the active probe, if any. Never raises, so measuring cannot break a call."""
    if _probe is None:
        return
    try:
        _probe(record)
    except Exception:  # noqa: BLE001 - a broken probe must not affect the reasoner
        pass


def _build_prompt(caption: str | None, scene: list[str], on_screen: list[str]) -> str:
    """Fill in the prompt, writing "(none)" for any empty field."""
    return SUPPLIED_TEXT_ONLY_PROMPT.format(
        caption=caption or "(none)",
        scene="; ".join(scene) if scene else "(none)",
        on_screen="; ".join(on_screen) if on_screen else "(none)",
    )


def _coerce_bool_traced(value: object) -> tuple[bool | None, str]:
    """Like ``_coerce_bool``, but also says how: "bool", "coerced" or "unreadable"."""
    if isinstance(value, bool):
        return value, "bool"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True, "coerced"
        if v in {"false", "no", "n", "0"}:
            return False, "coerced"
    return None, "unreadable"


def _coerce_bool(value: object) -> bool | None:
    """Turn a loose model value into a bool, or None ('uncertain') if it cannot be read.

    In a test run on 2026-05-31 Phi-3-mini sometimes gave real booleans and sometimes strings
    like "yes" or "false". Anything unclear becomes None, so fusion treats it as uncertain.
    """
    return _coerce_bool_traced(value)[0]


# The three fields the reasoner asks for, so the stem fallback below never fills one field
# from another field's value.
_REASONER_FIELDS = frozenset({"same_subject", "same_tone", "explanation"})


def _find_key_traced(payload: dict, *candidates: str) -> tuple[object, str]:
    """Like ``_find_key``, but also says how: "exact", "stem" or "missing"."""
    for c in candidates:
        if c in payload:
            return payload[c], "exact"
    # A key that is the exact name of another field is never a typo of this one. Without this,
    # `same_subject` and `same_tone` (both starting "same") matched each other, so a reply missing
    # one got the other's value. Found by the output validity script on 2026-08-17.
    siblings = {f for f in _REASONER_FIELDS if f not in candidates}
    for c in candidates:
        stem = c[:4]
        for k in payload:
            if not isinstance(k, str) or k.lower() in siblings:
                continue
            if k.lower().startswith(stem):
                return payload[k], "stem"
    return None, "missing"


def _find_key(payload: dict, *candidates: str) -> object:
    """Return the value of the first candidate key, else a near-miss key, else None.

    The model sometimes mangles keys (it once wrote ``"explanrance"`` for ``explanation``), so
    after the exact names we accept any key starting with the same four letters.
    """
    return _find_key_traced(payload, *candidates)[0]


def _parse_payload_traced(payload: dict) -> tuple[ReasonerVerdict, dict]:
    """Like ``_parse_payload``, but also returns a trace of what the parser had to fix.

    ``validity`` is the worst case over the fields: ``salvaged`` (a key matched by stem), then
    ``coerced`` (a bool sent as a string), then ``clean``. Missing keys are listed separately in
    ``missing_keys``.
    """
    subject_raw, subject_key = _find_key_traced(payload, "same_subject")
    tone_raw, tone_key = _find_key_traced(payload, "same_tone")
    explanation_raw, explanation_key = _find_key_traced(payload, "explanation")

    subject, subject_bool = _coerce_bool_traced(subject_raw)
    tone, tone_bool = _coerce_bool_traced(tone_raw)

    key_resolution = {
        "same_subject": subject_key,
        "same_tone": tone_key,
        "explanation": explanation_key,
    }
    bool_resolution = {"same_subject": subject_bool, "same_tone": tone_bool}

    if "stem" in key_resolution.values():
        validity = "salvaged"
    elif "coerced" in bool_resolution.values():
        validity = "coerced"
    else:
        validity = "clean"

    verdict = ReasonerVerdict(
        same_subject=subject,
        same_tone=tone,
        explanation=str(explanation_raw).strip() if explanation_raw is not None else "",
        available=True,
    )
    trace = {
        "validity": validity,
        "key_resolution": key_resolution,
        "bool_resolution": bool_resolution,
        "missing_keys": [k for k, how in key_resolution.items() if how == "missing"],
        "extra_keys": [k for k in payload if k not in key_resolution],
    }
    return verdict, trace


def _parse_payload(payload: dict) -> ReasonerVerdict:
    """Turn the model's JSON object into a ReasonerVerdict, allowing for small mistakes.

    Missing or unreadable booleans become None ('uncertain'), and near-miss keys are accepted.
    """
    return _parse_payload_traced(payload)[0]


def warm_up() -> str:
    """Load the model into Ollama's memory before the first request needs it.

    Returns a one-line status for /health. Never raises: if Ollama is not running, the first
    request finds out and its check is reported as NOT_ASSESSED.
    """
    settings = get_settings()
    started = perf_counter()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={"model": settings.ollama_model, "prompt": "", "stream": False,
                  "keep_alive": _WARM_UP_KEEP_ALIVE},
            timeout=_WARM_UP_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        return f"not loaded: {type(exc).__name__}"
    return f"loaded in {perf_counter() - started:.1f} s"


def reason_over_text(
    caption: str | None,
    scene_descriptions: list[str],
    on_screen_text: list[str],
) -> ReasonerVerdict:
    """Ask the local LLM how the supplied text fields relate. Results are cached by input hash.

    Returns ``available=False`` instead of raising when the LLM is off or unreachable, so fusion
    can report NOT_ASSESSED.
    """
    settings = get_settings()
    if not settings.use_llm:
        _emit(CallRecord(validity="not_called", available=False, latency_s=None, raw_response=None,
                         error="LLM reasoning disabled."))
        return ReasonerVerdict(None, None, "LLM reasoning disabled.", available=False)

    prompt = _build_prompt(caption, scene_descriptions, on_screen_text)
    cache = JsonCache("llm")
    cache_key = cache.key(prompt, settings.ollama_model)
    if not _bypass_cache:
        cached = cache.get(cache_key)
        if cached is not None:
            _emit(CallRecord(validity="cache_hit", available=bool(cached.get("available")),
                             latency_s=None, raw_response=None, cache_hit=True,
                             same_subject=cached.get("same_subject"),
                             same_tone=cached.get("same_tone")))
            return ReasonerVerdict(**cached)

    raw_text: str | None = None
    started = perf_counter()
    try:
        resp = httpx.post(
            f"{settings.ollama_host}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw_text = resp.json()["response"]
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("model response was not a JSON object")
        verdict, trace = _parse_payload_traced(payload)
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
        _emit(CallRecord(
            validity="unparseable", available=False, latency_s=perf_counter() - started,
            raw_response=raw_text, error=str(exc), error_type=type(exc).__name__,
            is_timeout=isinstance(exc, httpx.TimeoutException),
        ))
        if isinstance(exc, httpx.TimeoutException):
            reason = "The language model took too long to answer."
        elif isinstance(exc, httpx.HTTPError):
            reason = "The language model could not be reached."
        else:
            reason = "The language model's answer could not be read."
        return ReasonerVerdict(None, None, reason, available=False)

    _emit(CallRecord(
        validity=trace["validity"], available=True, latency_s=perf_counter() - started,
        raw_response=raw_text, same_subject=verdict.same_subject, same_tone=verdict.same_tone,
        detail=trace,
    ))
    if not _bypass_cache:
        cache.set(cache_key, verdict.__dict__)
    return verdict
