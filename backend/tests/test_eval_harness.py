"""In-process verification of the evaluation harness.

These tests certify the harness end-to-end *without relying on reading its printed output*
(which is itself the point — a number confirmed by a passing assertion is execution-verified,
not transcribed from a screen). They assert: the manifest is well-formed; every flag's
confusion counts sum to its label total (nothing dropped); the EXACT rules-only confusion
matrix; and individual deterministic outcomes.
"""

from tests.eval.run_eval import _load_examples, _predict, evaluate


def test_manifest_loads_and_is_well_formed():
    examples = _load_examples()
    assert len(examples) >= 10  # SSOT: start with 10–15
    ids = [e["id"] for e in examples]
    assert len(ids) == len(set(ids)), "example ids must be unique"
    valid = {"fire", "clear"}
    for e in examples:
        for flag, label in e.get("expected", {}).items():
            assert label in valid, f"{e['id']}: bad label {label!r} for {flag}"


def test_report_counts_are_internally_consistent():
    """Every flag's (tp+fp+fn+tn+not_assessed) must equal the number of labels for that flag,
    so the headline P/R/F1 are computed over exactly the labelled decisions."""
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


def test_rules_only_confusion_is_exactly_as_expected():
    """Pin the EXACT rules-only confusion matrix. These values are execution-verified: if the
    rules or manifest change, this fails loudly and the expected values must be re-derived.

    Hand derivation on the 19-example set (rules-only, as of ADR-014 Option B):
      emotional_framing:  em01/em03/combo01 fire->fired; em02/hard_em01/hard_em02
                          clear->clear; ec01 (no caption text) clear->not_assessed
                          => tp=3 fp=0 fn=0 tn=3 na=1
      recycled_context:   rc01/rc02/combo01 fire->fired; rc03/rc04 clear->clear;
                          ec01 clear->not_assessed
                          => tp=3 fp=0 fn=0 tn=2 na=1
      caption_content_mismatch (injected scenes): mm01/mm02/hard_mm01 fire->fired;
                          kc01/kc02 clear->clear; hard_kc01/02/03 clear->FIRED (synonyms)
                          => tp=3 fp=3 fn=0 tn=2 na=0
    """
    report, _ = evaluate(use_llm=False)

    def conf(flag):
        c = report.per_flag[flag].confusion
        return (c.tp, c.fp, c.fn, c.tn, c.not_assessed)

    # 19-example set (13 clear-cut + 6 hard). REAL measured values, read from run_eval.py
    # --json (re-derived 2026-08-17 after ADR-014 Option B replaced the emotional-framing
    # sentiment proxy with deterministic manipulation markers).
    # emotional_framing: em01 (4 markers), em03 (4), combo01 (3) fire->fired (tp=3);
    #   em02 (0 markers), hard_em01 (0), hard_em02 (0) clear->clear (tn=3); ec01 has no caption
    #   text so markers cannot be computed -> not_assessed (na=1). fp=0: the last remaining
    #   false positive (hard_em01, mild negativity) is gone because mild negative prose carries
    #   no markers. precision 0.75 -> 1.00.
    assert conf("emotional_framing") == (3, 0, 0, 3, 1)
    # recycled_context: exact fixture lookup, untouched (tp=3, tn=2, na=1).
    assert conf("recycled_context") == (3, 0, 0, 2, 1)
    # caption_content_mismatch: mm01/mm02/hard_mm01 fire->fired (tp=3); kc01/kc02 clear (tn=2);
    #   hard_kc01/hard_kc02/hard_kc03 (synonyms/paraphrase) clear->FIRED (fp=3 — literal
    #   word-overlap can't see synonyms). precision 0.50 (untouched by the EF change).
    assert conf("caption_content_mismatch") == (3, 3, 0, 2, 0)

    o = report.overall.confusion
    assert (o.tp, o.fp, o.fn, o.tn, o.not_assessed) == (9, 3, 0, 7, 2)

    # emotional_framing and recycled_context are now both perfect on this set; every remaining
    # false positive in the system belongs to caption↔scene literal word-overlap.
    for flag in ("emotional_framing", "recycled_context"):
        m = report.per_flag[flag]
        assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    assert report.per_flag["caption_content_mismatch"].precision == 0.5
    # Recall is 1.0 everywhere: the rules catch every true positive; they over-fire, not miss.
    for flag in ("emotional_framing", "recycled_context", "caption_content_mismatch"):
        assert report.per_flag[flag].recall == 1.0
    assert report.overall.precision == 0.75
    assert report.overall.recall == 1.0
    assert round(report.overall.f1, 3) == 0.857


def test_known_deterministic_outcomes():
    """Pin a few unambiguous cases so a regression in the rules is caught here."""
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
    """ADR-018: every example shares one blank image, so the recycled-context lookup is injected
    from the pre-WP-1 fixture by source_ref, reproducing the old extractor's three outcomes."""
    from app.models import FlagStatus
    from tests.eval.legacy_lookup import legacy_lookup

    matches, status = legacy_lookup("flood_recycled_2019.jpg")
    assert status == FlagStatus.FIRED and len(matches) == 2
    assert legacy_lookup("protest_recycled.jpg")[1] == FlagStatus.FIRED
    assert legacy_lookup("consistent_sunset.jpg") == ([], FlagStatus.CLEAR)
    assert legacy_lookup("studio_cat.jpg") == ([], FlagStatus.CLEAR)
    assert legacy_lookup("unknown_to_cache.png") == ([], FlagStatus.NOT_ASSESSED)
