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
        verdict = ReasonerVerdict(
            same_subject=payload.get("same_subject"),
            same_tone=payload.get("same_tone"),
            explanation=str(payload.get("explanation", "")).strip(),
            available=True,
        )
    except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
        return ReasonerVerdict(None, None, f"LLM unavailable: {exc}", available=False)

    cache.set(cache_key, verdict.__dict__)
    return verdict
