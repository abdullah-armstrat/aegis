"""Precision, recall and F1 for each flag.

Each flag is treated as a yes/no detector. A ``not_assessed`` prediction (the check couldn't
run) is left out of precision and recall and reported as coverage instead, so a missing
extractor doesn't move the scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Labels say what should happen ("fire"/"clear"); predictions are the system's status words.
LABEL_FIRE = "fire"
LABEL_CLEAR = "clear"

PRED_FIRED = "fired"
PRED_CLEAR = "clear"
PRED_NOT_ASSESSED = "not_assessed"


@dataclass
class Confusion:
    """Counts for one flag type across the test set."""

    tp: int = 0  # labelled fire, predicted fired
    fp: int = 0  # labelled clear, predicted fired
    fn: int = 0  # labelled fire, predicted clear
    tn: int = 0  # labelled clear, predicted clear
    not_assessed: int = 0  # prediction could not be made (excluded from P/R)

    @property
    def assessed(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def total(self) -> int:
        return self.assessed + self.not_assessed


@dataclass
class FlagMetrics:
    """Precision / recall / F1 for one flag type (None where undefined)."""

    flag: str
    confusion: Confusion
    precision: float | None
    recall: float | None
    f1: float | None
    coverage: float | None  # assessed / total, i.e. how often the check could run


def _safe_div(num: int, denom: int) -> float | None:
    return num / denom if denom else None


def _prf(c: Confusion) -> tuple[float | None, float | None, float | None]:
    precision = _safe_div(c.tp, c.tp + c.fp)
    recall = _safe_div(c.tp, c.tp + c.fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def update_confusion(c: Confusion, label: str, predicted: str) -> None:
    """Add one (label, prediction) pair to the counts."""
    if predicted == PRED_NOT_ASSESSED:
        c.not_assessed += 1
        return
    fired = predicted == PRED_FIRED
    should_fire = label == LABEL_FIRE
    if should_fire and fired:
        c.tp += 1
    elif not should_fire and fired:
        c.fp += 1
    elif should_fire and not fired:
        c.fn += 1
    else:
        c.tn += 1


def metrics_for(flag: str, confusion: Confusion) -> FlagMetrics:
    precision, recall, f1 = _prf(confusion)
    coverage = _safe_div(confusion.assessed, confusion.total)
    return FlagMetrics(
        flag=flag,
        confusion=confusion,
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=coverage,
    )


@dataclass
class EvalReport:
    """Per-flag metrics plus a micro-averaged overall, for one configuration."""

    config: str  # e.g. "rules-only" or "rules+llm"
    per_flag: dict[str, FlagMetrics] = field(default_factory=dict)
    overall: FlagMetrics | None = None


def build_report(config: str, pairs_by_flag: dict[str, list[tuple[str, str]]]) -> EvalReport:
    """Build a report from {flag_type: [(label, predicted), ...]}.

    The overall row adds up every flag's counts (micro-average) rather than averaging the
    per-flag scores.
    """
    report = EvalReport(config=config)
    micro = Confusion()
    for flag, pairs in pairs_by_flag.items():
        c = Confusion()
        for label, predicted in pairs:
            update_confusion(c, label, predicted)
        report.per_flag[flag] = metrics_for(flag, c)
        micro.tp += c.tp
        micro.fp += c.fp
        micro.fn += c.fn
        micro.tn += c.tn
        micro.not_assessed += c.not_assessed
    report.overall = metrics_for("__overall__", micro)
    return report
