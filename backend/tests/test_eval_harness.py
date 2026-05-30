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

    assert conf("emotional_framing") == (3, 0, 0, 1, 1)
    assert conf("recycled_context") == (3, 0, 0, 2, 1)
    assert conf("caption_content_mismatch") == (2, 0, 0, 2, 0)

    o = report.overall.confusion
    assert (o.tp, o.fp, o.fn, o.tn, o.not_assessed) == (8, 0, 0, 5, 2)

    for flag in ("emotional_framing", "recycled_context", "caption_content_mismatch"):
        m = report.per_flag[flag]
        assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


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
