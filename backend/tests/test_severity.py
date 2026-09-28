"""Tests for severity levels: set by the kind of evidence and the upper end of the 95% interval
of the check's false-alarm rate. Live web search results are medium at most, and no level
uses a verdict word."""

import re
import sys
from pathlib import Path

import pytest

from app.fusion import severity as sev
from app.fusion.rules import recycled_context_rule
from app.fusion.severity import CAPS, FALSE_ALARMS, Evidence, FalseAlarms, severity
from app.models import EvidenceBundle, FlagStatus, Meta, Modality, Severity, WebMatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def rates(monkeypatch):
    def set_rate(check, raised, cases):
        monkeypatch.setitem(FALSE_ALARMS, check, None if cases is None else FalseAlarms(raised, cases, "test"))
    return set_rate


def test_the_rate_used_is_the_upper_end_of_the_95_percent_interval():
    assert FalseAlarms(0, 15, "x").upper == pytest.approx(0.2039, abs=1e-4)
    assert FalseAlarms(3, 48, "x").upper == pytest.approx(0.1684, abs=1e-4)
    assert FalseAlarms(0, 9977, "x").upper < 0.0004


def test_high_is_complete_direct_evidence_from_a_check_whose_upper_rate_is_under_5_percent(rates):
    rates("x", 0, 100)  # upper 3.7%
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.HIGH
    rates("x", 0, 72)   # upper 5.1%: too few cases for high
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.LOW
    rates("x", 1, 100)  # measured 1%, upper 5.4%
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.LOW


def test_medium_is_incomplete_direct_evidence_or_an_indirect_signal_whose_upper_rate_is_at_most_10_percent(rates):
    rates("x", 0, 100)
    assert severity("x", Evidence.DIRECT_INCOMPLETE) == Severity.MEDIUM
    rates("y", 2, 200)  # upper 3.6%
    assert severity("y", Evidence.INDIRECT) == Severity.MEDIUM
    rates("y", 3, 48)   # measured 6.3%, upper 16.8%
    assert severity("y", Evidence.INDIRECT) == Severity.LOW


def test_low_is_any_other_finding_including_an_unmeasured_check(rates):
    rates("z", None, None)
    assert severity("z", Evidence.INDIRECT) == Severity.LOW
    assert severity("z", Evidence.DIRECT_COMPLETE) == Severity.LOW
    assert severity("never-listed", Evidence.INDIRECT) == Severity.LOW


def test_info_is_every_check_that_raised_no_finding(rates):
    rates("x", 0, 100)
    for status in (FlagStatus.CLEAR, FlagStatus.NOT_ASSESSED):
        for evidence in Evidence:
            assert severity("x", evidence, status) == Severity.INFO


def test_indirect_evidence_never_reaches_high(rates):
    rates("x", 0, 100000)
    assert severity("x", Evidence.INDIRECT) == Severity.MEDIUM


def test_a_live_web_page_date_is_medium_at_most(rates):
    assert CAPS["recycled_web"] == Severity.MEDIUM
    rates("recycled_web", 0, 1000)  # even with a rate low enough for high
    assert severity("recycled_web", Evidence.DIRECT_COMPLETE) == Severity.MEDIUM
    rates("recycled_web", 0, 15)    # as measured: 0 of 15, upper 20.4%
    assert severity("recycled_web", Evidence.DIRECT_COMPLETE) == Severity.LOW


def _recycled(found_by, dates, posted):
    return recycled_context_rule(EvidenceBundle(
        web_matches=[WebMatch(url=f"https://e.com/{i}", published_date=d, found_by=f)
                     for i, (d, f) in enumerate(zip(dates, found_by))],
        extractor_status={"reverse_image": FlagStatus.FIRED},
        meta=Meta(modality=Modality.IMAGE, posted_date=posted)))


@pytest.mark.parametrize("found_by, dates, posted, level", [
    (["index"], ["2019-03-04"], "2024-01-01", Severity.HIGH),            # index page dated earlier
    (["index"], ["2019-03-04"], None, Severity.MEDIUM),                  # index, no posting date
    (["index", "index"], ["2025-01-01", None], "2024-01-01", Severity.MEDIUM),  # an undated page
    (["web"], ["2019-03-04"], "2024-01-01", Severity.LOW),               # web only: 0/15, upper 20.4%
    (["web"], ["2019-03-04"], None, Severity.LOW),
    (["web", "index"], ["2019-03-04", "2025-01-01"], "2024-01-01", Severity.LOW),  # only the web page is earlier
    (["web", "index"], ["2019-03-04", "2020-01-01"], "2024-01-01", Severity.HIGH),  # an index page is earlier too
])
def test_recycled_context_levels_follow_the_lookup_the_finding_rests_on(found_by, dates, posted, level):
    flag = _recycled(found_by, dates, posted)
    assert flag.status == FlagStatus.FIRED
    assert flag.severity == level


def test_every_rule_result_carries_a_level_the_rule_allows():
    """Every branch of every check (built by the interface text audit script)."""
    from interface_text_audit import all_flags

    allowed_when_fired = {
        "recycled_context": {Severity.HIGH, Severity.MEDIUM},  # the audit only uses index pages
        "caption_content_mismatch": {Severity.LOW},
        "emotional_framing": {Severity.LOW},
        "audio_visual_mismatch": {Severity.LOW},
    }
    for flag in all_flags():
        if flag.status != FlagStatus.FIRED:
            assert flag.severity == Severity.INFO, flag
        else:
            assert flag.severity in allowed_when_fired[flag.type.value], flag
            if flag.type.value == "recycled_context":
                complete = "before the stated posting date" in flag.evidence
                assert flag.severity == (Severity.HIGH if complete else Severity.MEDIUM)


def test_the_measured_rates_come_from_the_named_evaluations():
    for check, measured in FALSE_ALARMS.items():
        if measured is not None:
            assert 0 < measured.cases and 0 <= measured.raised <= measured.cases
            assert measured.measured_on


def test_no_level_uses_a_verdict_word():
    words = re.compile(r"\b(true|false|fake|real|safe|verified|genuine|trust\w*|misleading|lie)\b", re.I)
    assert not any(words.search(level.value) for level in Severity)
    assert not words.search(sev.__doc__.split("Direct evidence")[1])
