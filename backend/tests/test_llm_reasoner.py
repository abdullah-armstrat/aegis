"""Tests for the LLM reasoner's contract and guardrail.

These do not call a live model — they assert the behaviour fusion relies on: the
supplied-text-only guardrail is present in the prompt (ADR-005), and an unavailable/disabled
LLM degrades to available=False (so fusion can mark NOT_ASSESSED, ADR-009) rather than
raising or fabricating a verdict.
"""

from app.config import get_settings
from app.extractors.cache import JsonCache
from app.fusion import llm_reasoner
from app.fusion.llm_reasoner import (
    SUPPLIED_TEXT_ONLY_PROMPT,
    _coerce_bool,
    _parse_payload,
    reason_over_text,
)


def test_parser_handles_clean_payload():
    v = _parse_payload(
        {"same_subject": False, "same_tone": False, "explanation": "They differ."}
    )
    assert v.available is True
    assert v.same_subject is False
    assert v.same_tone is False
    assert v.explanation == "They differ."


def test_parser_salvages_the_real_malformed_spike_payload():
    """The exact payload phi3:mini returned in the 2026-05-31 spike: a typo'd key
    'explanrance' instead of 'explanation' (DEVLOG / ADR-012). The explanation must still
    be recovered rather than silently lost."""
    raw = {
        "same_subject": True,
        "same_tone": False,
        "explanrance": "The scene description does not match the subject of a massive flood.",
    }
    v = _parse_payload(raw)
    assert v.same_tone is False
    assert "does not match" in v.explanation  # recovered from the near-miss key


def test_parser_coerces_string_booleans_and_degrades_unknowns():
    # Small models sometimes emit string booleans.
    v = _parse_payload({"same_subject": "true", "same_tone": "no", "explanation": "x"})
    assert v.same_subject is True
    assert v.same_tone is False
    # An unreadable/missing boolean becomes None ('uncertain'), never a silent False.
    v2 = _parse_payload({"explanation": "only prose, no booleans"})
    assert v2.same_subject is None
    assert v2.same_tone is None


def test_coerce_bool_units():
    assert _coerce_bool(True) is True
    assert _coerce_bool("YES") is True
    assert _coerce_bool("false") is False
    assert _coerce_bool("maybe") is None
    assert _coerce_bool(None) is None


def test_prompt_keeps_the_supplied_text_only_guardrail():
    p = SUPPLIED_TEXT_ONLY_PROMPT.lower()
    # The ADR-005 guardrail clauses must be present.
    assert "only over the text supplied" in p
    assert "never decide whether" in p
    assert "true, false, real, or fake" in p


def test_disabled_llm_returns_unavailable(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "false")
    get_settings.cache_clear()
    verdict = reason_over_text("a caption", ["a scene"], ["text"])
    assert verdict.available is False
    assert verdict.same_subject is None
    get_settings.cache_clear()


def test_unreachable_server_degrades_gracefully(monkeypatch, tmp_path):
    """With the LLM enabled but the server unreachable, the reasoner must return
    available=False (so fusion marks NOT_ASSESSED) rather than raise or fabricate."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    monkeypatch.setenv("AEGIS_OLLAMA_HOST", "http://127.0.0.1:1")  # dead port, fails fast
    get_settings.cache_clear()
    # Isolate the cache to a temp dir so we exercise the network path, not a prior hit.
    monkeypatch.setattr(llm_reasoner, "JsonCache", lambda ns: JsonCache(ns, root=tmp_path))

    verdict = reason_over_text("a caption", ["a scene"], ["some text"])

    assert verdict.available is False
    assert verdict.same_subject is None
    assert "unavailable" in verdict.explanation.lower()
    get_settings.cache_clear()
