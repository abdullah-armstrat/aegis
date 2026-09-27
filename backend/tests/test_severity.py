"""Severity follows one stated rule per level (ADR-045): the kind of evidence and the check's
measured false-alarm rate decide it, and no level uses a verdict word."""

import re
import sys
from pathlib import Path

import pytest

from app.fusion import severity as sev
from app.fusion.severity import FALSE_ALARMS, Evidence, FalseAlarms, severity
from app.models import FlagStatus, Severity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def rates(monkeypatch):
    def set_rate(check, raised, cases):
        monkeypatch.setitem(FALSE_ALARMS, check, None if cases is None else FalseAlarms(raised, cases, "test"))
    return set_rate


def test_high_is_complete_direct_evidence_from_a_check_under_5_percent(rates):
    rates("x", 0, 100)
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.HIGH
    rates("x", 4, 100)
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.HIGH
    rates("x", 5, 100)
    assert severity("x", Evidence.DIRECT_COMPLETE) == Severity.LOW  # 5% is not under 5%


def test_medium_is_incomplete_direct_evidence_or_an_indirect_signal_at_most_10_percent(rates):
    rates("x", 0, 100)
    assert severity("x", Evidence.DIRECT_INCOMPLETE) == Severity.MEDIUM
    rates("y", 10, 100)
    assert severity("y", Evidence.INDIRECT) == Severity.MEDIUM
    rates("y", 11, 100)
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
    rates("x", 0, 1000)
    assert severity("x", Evidence.INDIRECT) == Severity.MEDIUM


def test_every_rule_result_carries_the_level_the_rule_gives_it():
    """Every branch of every check (the interface review's text audit builds them all)."""
    from interface_text_audit import all_flags

    expected_fired = {
        "recycled_context": {Severity.HIGH, Severity.MEDIUM},
        "caption_content_mismatch": {Severity.MEDIUM, Severity.LOW},
        "emotional_framing": {Severity.LOW},
        "audio_visual_mismatch": {Severity.LOW},
    }
    flags = all_flags()
    for flag in flags:
        if flag.status != FlagStatus.FIRED:
            assert flag.severity == Severity.INFO, flag
        else:
            assert flag.severity in expected_fired[flag.type.value], flag
    # Complete direct evidence (a page dated before the stated posting date) is high; the rest medium.
    recycled = [f for f in flags if f.type.value == "recycled_context" and f.status == FlagStatus.FIRED]
    assert recycled
    for f in recycled:
        assert f.severity == (Severity.HIGH if "before the stated posting date" in f.evidence else Severity.MEDIUM)


def test_the_measured_rates_come_from_the_named_evaluations():
    for check, measured in FALSE_ALARMS.items():
        if measured is not None:
            assert 0 < measured.cases and 0 <= measured.raised <= measured.cases
            assert measured.measured_on


def test_no_level_uses_a_verdict_word():
    words = re.compile(r"\b(true|false|fake|real|safe|verified|genuine|trust\w*|misleading|lie)\b", re.I)
    assert not any(words.search(level.value) for level in Severity)
    assert not words.search(sev.__doc__.split("Direct evidence")[1])
