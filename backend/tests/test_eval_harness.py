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

    Hand derivation (rules-only):
      emotional_framing:  em01 fire->fired, em02 clear->clear, em03 fire->fired,
                          combo01 fire->fired, ec01 clear->not_assessed
                          => tp=3 fp=0 fn=0 tn=1 na=1
      recycled_context:   rc01 fire->fired, rc02 fire->fired, rc03 clear->clear,
                          rc04 clear->clear, combo01 fire->fired, ec01 clear->not_assessed
                          => tp=3 fp=0 fn=0 tn=2 na=1
      caption_content_mismatch (injected scenes): mm01 fire, mm02 fire, kc01 clear, kc02 clear
                          => tp=2 fp=0 fn=0 tn=2 na=0
    """
    report, _ = evaluate(use_llm=False)

    def conf(flag):
        c = report.per_flag[flag].confusion
        return (c.tp, c.fp, c.fn, c.tn, c.not_assessed)

    # 19-example set (13 clear-cut + 6 hard). REAL measured values, read from run_eval.py
    # --json. The hard set deliberately breaks two flags:
    # emotional_framing: em01/em03/combo01 fire->fired (tp=3); em02 clear->clear (tn=1);
    #   hard_em01/hard_em02 (mild negativity) clear->FIRED (fp=2 — the sentiment model IS
    #   over-confident on mild text); ec01 -> not_assessed (na=1). precision 0.60.
    assert conf("emotional_framing") == (3, 2, 0, 1, 1)
    # recycled_context: exact fixture lookup, untouched by the hard set (tp=3, tn=2, na=1).
    assert conf("recycled_context") == (3, 0, 0, 2, 1)
    # caption_content_mismatch: mm01/mm02/hard_mm01 fire->fired (tp=3); kc01/kc02 clear (tn=2);
    #   hard_kc01/hard_kc02/hard_kc03 (synonyms/paraphrase) clear->FIRED (fp=3 — literal
    #   word-overlap can't see synonyms). precision 0.50.
    assert conf("caption_content_mismatch") == (3, 3, 0, 2, 0)

    o = report.overall.confusion
    assert (o.tp, o.fp, o.fn, o.tn, o.not_assessed) == (9, 5, 0, 5, 2)

    # recycled_context is perfect; the other two are intentionally degraded by the hard set.
    rc = report.per_flag["recycled_context"]
    assert rc.precision == 1.0 and rc.recall == 1.0 and rc.f1 == 1.0
    assert report.per_flag["emotional_framing"].precision == 0.6
    assert report.per_flag["caption_content_mismatch"].precision == 0.5
    # Recall is 1.0 everywhere: the rules catch every true positive; they over-fire, not miss.
    for flag in ("emotional_framing", "recycled_context", "caption_content_mismatch"):
        assert report.per_flag[flag].recall == 1.0
    assert round(report.overall.precision, 3) == 0.643
    assert report.overall.recall == 1.0
    assert round(report.overall.f1, 3) == 0.783


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
