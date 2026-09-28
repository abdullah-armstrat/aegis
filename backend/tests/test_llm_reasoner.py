"""Tests for the LLM reasoner (no live model is called).

Checks the tolerant JSON parsing, that the prompt keeps its "supplied text only" rule, and
that a disabled or unreachable LLM returns available=False instead of raising.
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
    """A real phi3:mini reply used the misspelt key 'explanrance'; the text is still recovered."""
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
    # A missing or unreadable boolean becomes None (uncertain), not False.
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
    """LLM on but server unreachable: returns available=False rather than raising."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    monkeypatch.setenv("AEGIS_OLLAMA_HOST", "http://127.0.0.1:1")  # dead port, fails fast
    get_settings.cache_clear()
    # Use an empty temp cache so the network path runs instead of an old cached answer.
    monkeypatch.setattr(llm_reasoner, "JsonCache", lambda ns: JsonCache(ns, root=tmp_path))

    verdict = reason_over_text("a caption", ["a scene"], ["some text"])

    assert verdict.available is False
    assert verdict.same_subject is None
    assert verdict.explanation == "The language model could not be reached."
    get_settings.cache_clear()


# --- measurement hooks used by the LLM evaluation scripts ---


def test_probe_is_inert_by_default():
    """Outside record_calls there is no probe and the cache is not bypassed."""
    assert llm_reasoner._probe is None
    assert llm_reasoner._bypass_cache is False


def test_parse_payload_traced_classifies_clean_salvaged_and_coerced():
    """Each reply is labelled clean, salvaged or coerced from how the parser read its fields."""
    _, clean = _parse_payload_traced(
        {"same_subject": True, "same_tone": False, "explanation": "x"}
    )
    assert clean["validity"] == "clean"

    # The real misspelt reply: 'explanrance' only matches on its first four letters.
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
    """A missing key is not filled in from another key starting with the same four letters.

    Both ``same_subject`` and ``same_tone`` start with "same", so the typo fallback used to copy
    one into the other. A missing key should now be 'missing' and give None.
    """
    # same_tone missing -> not filled from same_subject.
    verdict, trace = _parse_payload_traced({"same_subject": True, "explanation": "x"})
    assert trace["key_resolution"]["same_tone"] == "missing"
    assert trace["missing_keys"] == ["same_tone"]
    assert verdict.same_tone is None
    assert verdict.same_subject is True  # the key that was given is unchanged

    # The other way round matters more, since same_subject drives the flag.
    verdict2, trace2 = _parse_payload_traced({"same_tone": False, "explanation": "x"})
    assert trace2["key_resolution"]["same_subject"] == "missing"
    assert verdict2.same_subject is None

    # The real typo case ('explanrance') is still recovered.
    verdict3, trace3 = _parse_payload_traced(
        {"same_subject": True, "same_tone": False, "explanrance": "recovered"}
    )
    assert trace3["key_resolution"]["explanation"] == "stem"
    assert verdict3.explanation == "recovered"


def test_record_calls_captures_a_failed_call_and_restores_state(monkeypatch, tmp_path):
    """An unreachable LLM is recorded as one 'unparseable' call and state is reset afterwards."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_USE_LLM", "true")
    monkeypatch.setenv("AEGIS_OLLAMA_HOST", "http://127.0.0.1:1")  # dead port, fails fast
    get_settings.cache_clear()
    monkeypatch.setattr(llm_reasoner, "JsonCache", lambda ns: JsonCache(ns, root=tmp_path))

    sink = []
    with record_calls(sink, bypass_cache=True):
        assert llm_reasoner._bypass_cache is True
        verdict = reason_over_text("a caption", ["a scene"], [])

    assert verdict.available is False          # same result as without recording
    assert len(sink) == 1
    assert sink[0].validity == "unparseable"
    assert sink[0].available is False
    assert sink[0].latency_s is not None       # a failed call still takes time
    # Reset on exit so nothing carries over to later requests.
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
