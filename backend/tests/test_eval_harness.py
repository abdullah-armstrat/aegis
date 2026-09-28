"""Tests for the evaluation harness, checked through its results rather than its printout.

Covers the manifest format, that counts add up, the exact rules-only confusion matrix, and a
few single cases. All examples use one blank image, which the picture-based caption methods
report as not assessed (there is nothing in it for CLIP to compare).
"""

import pytest

from app.config import get_settings
from tests.eval.run_eval import _load_examples, _predict, evaluate


def test_manifest_loads_and_is_well_formed():
    examples = _load_examples()
    assert len(examples) >= 10  # the set began with 10-15 examples
    ids = [e["id"] for e in examples]
    assert len(ids) == len(set(ids)), "example ids must be unique"
    valid = {"fire", "clear"}
    for e in examples:
        for flag, label in e.get("expected", {}).items():
            assert label in valid, f"{e['id']}: bad label {label!r} for {flag}"


def test_report_counts_are_internally_consistent():
    """Each flag's tp+fp+fn+tn+not_assessed equals its number of labels, so nothing is dropped."""
    examples = _load_examples()
    label_counts: dict[str, int] = {}
    for e in examples:
        for flag in e.get("expected", {}):
            label_counts[flag] = label_counts.get(flag, 0) + 1

    report, _records = evaluate(use_llm=False)
    for flag, m in report.per_flag.items():
        c = m.confusion
        assert c.total == label_counts[flag], (
            f"{flag}: confusion total {c.total} != {label_counts[flag]} labels"
        )


def test_rules_only_confusion_is_exactly_as_expected(monkeypatch):
    """Pin the exact rules-only confusion matrix on the 19 examples, caption check on word overlap.

    If the rules or the examples change, these numbers need working out again.
    """
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", "overlap")
    get_settings.cache_clear()
    report, _ = evaluate(use_llm=False)

    def conf(flag):
        c = report.per_flag[flag].confusion
        return (c.tp, c.fp, c.fn, c.tn, c.not_assessed)

    # emotional_framing: em01, em03, combo01 have markers and fire (tp=3); em02 and the two mild
    #   hard_em cases have none (tn=3); ec01 has no caption, so not assessed (na=1).
    assert conf("emotional_framing") == (3, 0, 0, 3, 1)
    # recycled_context: rc01, rc02, combo01 fire; rc03, rc04 clear; ec01 not assessed.
    assert conf("recycled_context") == (3, 0, 0, 2, 1)
    # caption_content_mismatch: mm01, mm02, hard_mm01 fire (tp=3); kc01, kc02 clear (tn=2);
    #   the 3 hard_kc synonym cases also fire (fp=3) because word overlap can't see synonyms.
    assert conf("caption_content_mismatch") == (3, 3, 0, 2, 0)

    o = report.overall.confusion
    assert (o.tp, o.fp, o.fn, o.tn, o.not_assessed) == (9, 3, 0, 7, 2)

    # The only false positives left are the word-overlap caption ones.
    for flag in ("emotional_framing", "recycled_context"):
        m = report.per_flag[flag]
        assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert report.per_flag["caption_content_mismatch"].precision == 0.5
    # Recall is 1.0 everywhere: the rules over-fire rather than miss.
    for flag in ("emotional_framing", "recycled_context", "caption_content_mismatch"):
        assert report.per_flag[flag].recall == 1.0
    assert report.overall.precision == 0.75
    assert report.overall.recall == 1.0
    assert round(report.overall.f1, 3) == 0.857


def test_known_deterministic_outcomes():
    """A few clear-cut single cases, so a rule regression shows up here."""
    by_id = {e["id"]: e for e in _load_examples()}

    combo = _predict(by_id["combo01"], use_llm=False)
    assert combo.get("recycled_context") == "fired"
    assert combo.get("emotional_framing") == "fired"

    assert _predict(by_id["mm01"], use_llm=False).get("caption_content_mismatch") == "fired"
    assert _predict(by_id["rc03"], use_llm=False).get("recycled_context") == "clear"

    ec = _predict(by_id["ec01"], use_llm=False)
    assert ec.get("emotional_framing") == "not_assessed"
    assert ec.get("recycled_context") == "not_assessed"


def test_regression_lookups_are_injected_not_hashed():
    """The injected lookup gives FIRED, CLEAR or NOT_ASSESSED depending on the fixture entry."""
    from app.models import FlagStatus
    from tests.eval.legacy_lookup import legacy_lookup

    matches, status = legacy_lookup("flood_recycled_2019.jpg")
    assert status == FlagStatus.FIRED and len(matches) == 2
    assert legacy_lookup("protest_recycled.jpg")[1] == FlagStatus.FIRED
    assert legacy_lookup("consistent_sunset.jpg") == ([], FlagStatus.CLEAR)
    assert legacy_lookup("studio_cat.jpg") == ([], FlagStatus.CLEAR)
    assert legacy_lookup("unknown_to_cache.png") == ([], FlagStatus.NOT_ASSESSED)


# (caption_content_mismatch, overall) as (tp, fp, fn, tn, not_assessed), rules-only with the
# captioner off. The other two flags don't depend on the caption method.
CAPTION_PINS = {
    "off": ((0, 0, 0, 0, 8), (6, 0, 0, 5, 10)),
    "image": ((0, 0, 0, 0, 8), (6, 0, 0, 5, 10)),    # blank picture, so not assessed
    "meaning": ((0, 0, 0, 0, 8), (6, 0, 0, 5, 10)),
}


@pytest.mark.parametrize("method", [
    "off",
    pytest.param("image", marks=pytest.mark.slow),
    pytest.param("meaning", marks=pytest.mark.slow),
])
def test_caption_check_pinned_per_method(monkeypatch, method):
    """Each caption method gives the pinned counts; picture-based ones can't assess a blank image."""
    get_settings.cache_clear()
    monkeypatch.setenv("AEGIS_CAPTION_MATCH_METHOD", method)
    get_settings.cache_clear()
    report, _ = evaluate(use_llm=False)

    def conf(m):
        return (m.confusion.tp, m.confusion.fp, m.confusion.fn, m.confusion.tn, m.confusion.not_assessed)

    caption, overall = CAPTION_PINS[method]
    assert conf(report.per_flag["caption_content_mismatch"]) == caption
    assert conf(report.per_flag["emotional_framing"]) == (3, 0, 0, 3, 1)
    assert conf(report.per_flag["recycled_context"]) == (3, 0, 0, 2, 1)
    assert conf(report.overall) == overall
    get_settings.cache_clear()
