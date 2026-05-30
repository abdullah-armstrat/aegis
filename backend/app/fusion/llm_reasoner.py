"""LLM reasoner — phrases cross-modal mismatches in plain language.

The LLM is a *reasoner over supplied text only*. It is given the bundle's extracted text
fields and asked solely relational questions — do these pieces of text describe the same
thing, do they carry the same tone — never "is this true?". This sidesteps the knowledge-
cutoff and hallucination problems entirely and is the strict rule of ADR-005.

To stop that guardrail eroding through prompt drift over many edits (review note #6), the
entire prompt lives in the single constant ``SUPPLIED_TEXT_ONLY_PROMPT`` below. Change the
prompt *only* here, and keep the guardrail clause intact.

The deterministic rule layer (``rules.py``) runs independently and first; this layer is
additive (ADR-004), and is skipped entirely when ``settings.use_llm`` is false or the model
is unreachable — in which case the affected checks are reported as NOT_ASSESSED, never
silently dropped (ADR-009).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.extractors.cache import JsonCache

# --- The one place the prompt lives. Guardrail = ADR-005. Edit here only. ---
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


def _build_prompt(caption: str | None, scene: list[str], on_screen: list[str]) -> str:
    return SUPPLIED_TEXT_ONLY_PROMPT.format(
        caption=caption or "(none)",
        scene="; ".join(scene) if scene else "(none)",
        on_screen="; ".join(on_screen) if on_screen else "(none)",
    )


def _coerce_bool(value: object) -> bool | None:
    """Coerce a loose model value to bool, or None ('uncertain') if unrecognised.

    The Phi-3-mini spike (DEVLOG 2026-05-31) returned booleans inconsistently — sometimes a
    real bool, sometimes the strings "true"/"false"/"yes"/"no". Anything we can't read with
    confidence becomes None so fusion treats it as uncertain rather than a false negative.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    return None


def _find_key(payload: dict, *candidates: str) -> object:
    """Return the first candidate key present, else a fuzzy near-miss match, else None.

    The spike showed the model can mangle keys (it emitted ``"explanrance"`` for
    ``explanation``). We first try exact candidates, then fall back to any key whose first
    four letters match a candidate's — enough to salvage typos without matching unrelated
    keys. Defensive parsing per ADR-012: trust the model's structure as little as possible.
    """
    for c in candidates:
        if c in payload:
            return payload[c]
    for c in candidates:
        stem = c[:4]
        for k in payload:
            if isinstance(k, str) and k.lower().startswith(stem):
                return payload[k]
    return None


def _parse_payload(payload: dict) -> ReasonerVerdict:
    """Tolerantly turn a raw model JSON object into a ReasonerVerdict (ADR-012).

    Missing/unreadable booleans become None ('uncertain'); the explanation is salvaged from a
    near-miss key if the exact one is absent.
    """
    explanation = _find_key(payload, "explanation")
    return ReasonerVerdict(
        same_subject=_coerce_bool(_find_key(payload, "same_subject")),
        same_tone=_coerce_bool(_find_key(payload, "same_tone")),
        explanation=str(explanation).strip() if explanation is not None else "",
        available=True,
    )


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
        return ReasonerVerdict(None, None, "LLM reasoning disabled.", available=False)

    prompt = _build_prompt(caption, scene_descriptions, on_screen_text)
    cache = JsonCache("llm")
    cache_key = cache.key(prompt, settings.ollama_model)
    cached = cache.get(cache_key)
    if cached is not None:
        return ReasonerVerdict(**cached)

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
        payload = json.loads(resp.json()["response"])
        if not isinstance(payload, dict):
            raise ValueError("model response was not a JSON object")
        verdict = _parse_payload(payload)
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return ReasonerVerdict(None, None, f"LLM unavailable: {exc}", available=False)

    cache.set(cache_key, verdict.__dict__)
    return verdict
