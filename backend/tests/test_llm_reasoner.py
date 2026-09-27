"""Tests for the LLM reasoner's contract and guardrail.

These do not call a live model — they assert the behaviour fusion relies on: the
supplied-text-only guardrail is present in the prompt, and an unavailable/disabled
LLM degrades to available=False (so fusion can mark NOT_ASSESSED) rather than
raising or fabricating a verdict.
"""

from app.config import get_settings
from app.extractors.cache import JsonCache
from app.fusion import llm_reasoner
from app.fusion.llm_reasoner import (
    SUPPLIED_TEXT_ONLY_PROMPT,
    _coerce_bool,
    _parse_payload,
    _parse_payload_traced,
    reason_over_text,
    record_calls,
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
    'explanrance' instead of 'explanation'. The explanation must still
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
    # The supplied-text-only guardrail clauses must be present.
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
    assert verdict.explanation == "The language model could not be reached."
    get_settings.cache_clear()


# --- measurement instrumentation (LLM-specific eval; must never affect production) ---


def test_probe_is_inert_by_default():
    """No probe is installed unless a measurement script enters record_calls, and the cache is
    never bypassed in production. Guards the 'measurement only' promise."""
    assert llm_reasoner._probe is None
    assert llm_reasoner._bypass_cache is False


def test_parse_payload_traced_classifies_clean_salvaged_and_coerced():
    """The four-way validity taxonomy is derived from how the tolerant parser resolved each
    field, not guessed. Pure function — no model call."""
    _, clean = _parse_payload_traced(
        {"same_subject": True, "same_tone": False, "explanation": "x"}
    )
    assert clean["validity"] == "clean"

    # The real 2026-05-31 malformed payload: 'explanrance' matches only by 4-letter stem.
    _, salvaged = _parse_payload_traced(
        {"same_subject": True, "same_tone": False, "explanrance": "recovered"}
    )
    assert salvaged["validity"] == "salvaged"
    assert salvaged["key_resolution"]["explanation"] == "stem"

    _, coerced = _parse_payload_traced(
        {"same_subject": "true", "same_tone": "no", "explanation": "x"}
    )
    assert coerced["validity"] == "coerced"
    assert coerced["bool_resolution"]["same_subject"] == "coerced"


def test_find_key_does_not_salvage_one_field_from_a_sibling():
    """Regression for the stem-collision defect found by the validity harness, 2026-08-17.

    ``_find_key``'s salvage fallback matches a candidate's first four letters, and both
    ``same_subject`` and ``same_tone`` begin "same". Before the fix, a model that omitted one
    of them silently received the OTHER one's value — a fabricated judgement, and since
    ``same_subject`` drives the flag, one that reached production. An omitted key must now
    resolve to 'missing' and leave the verdict uncertain (None).
    """
    # same_tone omitted -> must NOT be filled from same_subject.
    verdict, trace = _parse_payload_traced({"same_subject": True, "explanation": "x"})
    assert trace["key_resolution"]["same_tone"] == "missing"
    assert trace["missing_keys"] == ["same_tone"]
    assert verdict.same_tone is None
    assert verdict.same_subject is True  # the field that WAS supplied is unaffected

    # Symmetric: same_subject omitted must not be filled from same_tone. This is the direction
    # that could reach the scorecard, so it is asserted explicitly.
    verdict2, trace2 = _parse_payload_traced({"same_tone": False, "explanation": "x"})
    assert trace2["key_resolution"]["same_subject"] == "missing"
    assert verdict2.same_subject is None

    # The genuine typo salvage this fallback exists for still works ('explanrance').
    verdict3, trace3 = _parse_payload_traced(
        {"same_subject": True, "same_tone": False, "explanrance": "recovered"}
    )
    assert trace3["key_resolution"]["explanation"] == "stem"
    assert verdict3.explanation == "recovered"


def test_record_calls_captures_a_failed_call_and_restores_state(monkeypatch, tmp_path):
    """With the LLM enabled but unreachable, the probe must record one 'unparseable' call and
    the reasoner must still degrade to available=False exactly as before."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    monkeypatch.setenv("AEGIS_OLLAMA_HOST", "http://127.0.0.1:1")  # dead port, fails fast
    get_settings.cache_clear()
    monkeypatch.setattr(llm_reasoner, "JsonCache", lambda ns: JsonCache(ns, root=tmp_path))

    sink = []
    with record_calls(sink, bypass_cache=True):
        assert llm_reasoner._bypass_cache is True
        verdict = reason_over_text("a caption", ["a scene"], [])

    assert verdict.available is False          # production behaviour unchanged
    assert len(sink) == 1
    assert sink[0].validity == "unparseable"
    assert sink[0].available is False
    assert sink[0].latency_s is not None       # a failed call still has a wall-clock cost
    # State is restored on exit, so nothing leaks into a later request.
    assert llm_reasoner._probe is None
    assert llm_reasoner._bypass_cache is False
    get_settings.cache_clear()


def test_warm_up_asks_ollama_to_load_the_model_and_never_raises(monkeypatch):
    import httpx

    sent = []

    def fake_post(url, json, timeout):
        sent.append((url, json, timeout))
        return httpx.Response(200, json={"done": True, "done_reason": "load"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(llm_reasoner.httpx, "post", fake_post)
    assert llm_reasoner.warm_up().startswith("loaded in ")
    url, body, timeout = sent[0]
    assert url.endswith("/api/generate") and body["prompt"] == "" and body["model"] == get_settings().ollama_model
    assert timeout > llm_reasoner._TIMEOUT_SECONDS

    def refused(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_reasoner.httpx, "post", refused)
    assert llm_reasoner.warm_up() == "not loaded: ConnectError"


def test_the_server_warms_the_llm_at_start_up_only_when_it_is_on(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    calls = []
    monkeypatch.setattr(llm_reasoner, "warm_up", lambda: calls.append(1) or "loaded in 0.1 s")
    for on, expected in (("false", []), ("true", [1])):
        calls.clear()
        monkeypatch.setenv("AEGIS_USE_LLM", on)
        get_settings.cache_clear()
        with TestClient(main.app) as client:
            assert calls == expected
            warm = client.get("/health").json()["config"]["llm_warm_up"]
            assert warm == ("loaded in 0.1 s" if expected else None)
    get_settings.cache_clear()
