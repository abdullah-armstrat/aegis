"""LLM reasoner — phrases cross-modal mismatches in plain language.

The LLM is a *reasoner over supplied text only*. It is given the bundle's extracted text
fields and asked solely relational questions — do these pieces of text describe the same
thing, do they carry the same tone — never "is this true?". This sidesteps the knowledge-
cutoff and hallucination problems entirely, which is why the rule is strict.

To stop that guardrail eroding through prompt drift over many edits, the
entire prompt lives in the single constant ``SUPPLIED_TEXT_ONLY_PROMPT`` below. Change the
prompt *only* here, and keep the guardrail clause intact.

The deterministic rule layer (``rules.py``) runs independently and first. This layer only
adds to it, so the reproducible rule results are never overridden, and it is skipped entirely
when ``settings.use_llm`` is false or the model is unreachable — in which case the affected
checks are reported as NOT_ASSESSED, never silently dropped.
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

# --- The one place the prompt lives, with its supplied-text-only guardrail. Edit here only. ---
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


@dataclass
class ReasonerVerdict:
    """The LLM's relational judgement over supplied text (never a truth verdict)."""

    same_subject: bool | None
    same_tone: bool | None
    explanation: str
    available: bool  # False => model unreachable / disabled; caller marks NOT_ASSESSED


# --------------------------------------------------------------------------- instrumentation
#
# MEASUREMENT ONLY. Everything below is inert unless a script enters ``record_calls``; the
# /analyze path never does, so production behaviour is byte-for-byte unchanged. It exists
# because the raw model response is parsed and discarded, which makes output validity, latency
# and run-to-run stability impossible to reconstruct after the fact — the three LLM-specific
# metrics the examiner asked for.


@dataclass
class CallRecord:
    """One instrumented reasoner call. Never read by production code."""

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
    """Capture a :class:`CallRecord` per reasoner call for the duration of the block.

    ``bypass_cache`` skips the input-hash cache for both read and write, which is required to
    measure anything at all: a cache hit returns a stored verdict with no raw response, no
    network call and no latency. Restores the previous state on exit, including on exception.
    """
    global _probe, _bypass_cache
    prev_probe, prev_bypass = _probe, _bypass_cache
    _probe, _bypass_cache = sink.append, bypass_cache
    try:
        yield sink
    finally:
        _probe, _bypass_cache = prev_probe, prev_bypass


def _emit(record: CallRecord) -> None:
    """Hand a record to the active probe. Never raises — measurement must not break the caller."""
    if _probe is None:
        return
    try:
        _probe(record)
    except Exception:  # noqa: BLE001 - a broken probe must not affect the reasoner
        pass


def _build_prompt(caption: str | None, scene: list[str], on_screen: list[str]) -> str:
    return SUPPLIED_TEXT_ONLY_PROMPT.format(
        caption=caption or "(none)",
        scene="; ".join(scene) if scene else "(none)",
        on_screen="; ".join(on_screen) if on_screen else "(none)",
    )


def _coerce_bool_traced(value: object) -> tuple[bool | None, str]:
    """``_coerce_bool`` plus how it resolved: ``bool`` | ``coerced`` | ``unreadable``."""
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
    """Coerce a loose model value to bool, or None ('uncertain') if unrecognised.

    In a test run on 2026-05-31, Phi-3-mini returned booleans inconsistently — sometimes a
    real bool, sometimes the strings "true"/"false"/"yes"/"no". Anything we can't read with
    confidence becomes None so fusion treats it as uncertain rather than a false negative.
    """
    return _coerce_bool_traced(value)[0]


# The three fields the reasoner asks for. Used to stop the stem fallback below salvaging one
# field from another field's value.
_REASONER_FIELDS = frozenset({"same_subject", "same_tone", "explanation"})


def _find_key_traced(payload: dict, *candidates: str) -> tuple[object, str]:
    """``_find_key`` plus how it resolved: ``exact`` | ``stem`` | ``missing``."""
    for c in candidates:
        if c in payload:
            return payload[c], "exact"
    # A payload key that is the canonical name of a DIFFERENT reasoner field is never a typo of
    # this one, so it must not be stem-matched. Without this guard `same_subject` and `same_tone`
    # (both starting "same") resolve to each other: a model that omits one silently receives its
    # sibling's value, fabricating a judgement it never made. `same_subject` drives the flag, so
    # that reached production. Defect found by the validity harness, 2026-08-17.
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
    """Return the first candidate key present, else a fuzzy near-miss match, else None.

    The spike showed the model can mangle keys (it emitted ``"explanrance"`` for
    ``explanation``). We first try exact candidates, then fall back to any key whose first
    four letters match a candidate's — enough to salvage typos without matching unrelated
    keys. Defensive parsing: trust a small local model's output structure as little as possible.
    """
    return _find_key_traced(payload, *candidates)[0]


def _parse_payload_traced(payload: dict) -> tuple[ReasonerVerdict, dict]:
    """``_parse_payload`` plus a trace of what the tolerant parser had to do.

    ``validity`` is the worst outcome across the three fields, worst-first:
    ``salvaged`` (a key only matched by stem) > ``coerced`` (a bool arrived as a string) >
    ``clean``. Keys that were absent entirely are reported separately in ``missing_keys``:
    they are neither salvaged nor coerced, and the four-way taxonomy has no bucket for them.
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
    """Tolerantly turn a raw model JSON object into a ReasonerVerdict.

    Missing/unreadable booleans become None ('uncertain'); the explanation is salvaged from a
    near-miss key if the exact one is absent.
    """
    return _parse_payload_traced(payload)[0]


def reason_over_text(
    caption: str | None,
    scene_descriptions: list[str],
    on_screen_text: list[str],
) -> ReasonerVerdict:
    """Ask the local LLM to relate the supplied text fields. Cached by input hash.

    Returns ``available=False`` (rather than raising) when the LLM is disabled or the server
    is unreachable, so fusion can record NOT_ASSESSED instead of a misleading silence.
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
        return ReasonerVerdict(None, None, f"LLM unavailable: {exc}", available=False)

    _emit(CallRecord(
        validity=trace["validity"], available=True, latency_s=perf_counter() - started,
        raw_response=raw_text, same_subject=verdict.same_subject, same_tone=verdict.same_tone,
        detail=trace,
    ))
    if not _bypass_cache:
        cache.set(cache_key, verdict.__dict__)
    return verdict
