"""Unit tests for the evaluation metrics (deterministic, no models).

These pin the precision/recall/F1 maths and — most importantly — the rule that NOT_ASSESSED
predictions are excluded from P/R and counted as coverage (ADR-009 applied to evaluation).
"""

import math

from tests.eval.metrics import (
    Confusion,
    build_report,
    metrics_for,
    update_confusion,
)


def test_confusion_classification():
    c = Confusion()
    update_confusion(c, "fire", "fired")   # tp
    update_confusion(c, "clear", "fired")  # fp
    update_confusion(c, "fire", "clear")   # fn
    update_confusion(c, "clear", "clear")  # tn
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert c.not_assessed == 0


def test_not_assessed_excluded_from_pr_counted_as_coverage():
    c = Confusion()
    update_confusion(c, "fire", "not_assessed")
    update_confusion(c, "clear", "not_assessed")
    update_confusion(c, "fire", "fired")  # the only assessed pair
    m = metrics_for("x", c)
    assert c.not_assessed == 2
    assert c.assessed == 1
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert math.isclose(m.coverage, 1 / 3)


def test_perfect_and_undefined_metrics():
    c = Confusion(tp=3, tn=2)
    m = metrics_for("x", c)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0
    # No positives predicted or labelled -> precision/recall undefined (None), not 0.
    c2 = Confusion(tn=4)
    m2 = metrics_for("x", c2)
    assert m2.precision is None and m2.recall is None and m2.f1 is None


def test_f1_is_harmonic_mean():
    c = Confusion(tp=1, fp=1, fn=1)  # P=0.5, R=0.5 -> F1=0.5
    m = metrics_for("x", c)
    assert math.isclose(m.precision, 0.5)
    assert math.isclose(m.recall, 0.5)
    assert math.isclose(m.f1, 0.5)


def test_build_report_micro_average():
    pairs = {
        "emotional_framing": [("fire", "fired"), ("clear", "clear")],
        "recycled_context": [("fire", "fired"), ("fire", "not_assessed")],
    }
    report = build_report("rules-only", pairs)
    assert report.config == "rules-only"
    assert report.per_flag["recycled_context"].confusion.not_assessed == 1
    # Overall micro: tp=2, tn=1, not_assessed=1 -> precision 1.0, recall 1.0.
    assert report.overall.precision == 1.0
    assert report.overall.recall == 1.0
    assert report.overall.confusion.not_assessed == 1
