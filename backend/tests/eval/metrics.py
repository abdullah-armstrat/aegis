"""Flag-level evaluation metrics (SSOT §5.3).

Each flag behaves as a binary detector, so we score it with precision / recall / F1 — the
appropriate, area-standard metric for this kind of mixed AI/SWE system. The one wrinkle that
matters: a prediction can be ``not_assessed`` (the check could not run — ADR-009), which is
neither a true positive nor a false one. Those are **excluded from precision/recall** and
reported separately as a coverage figure, so an extractor that is simply absent never inflates
or deflates the quality numbers. This keeps the metric honest and makes the captioner's arrival
a clean before/after (caption↔scene moves from "not assessed" to scored).

Pure functions over plain data — no I/O, no model calls — so this module is deterministic and
unit-tested. Nothing here invents a number; it only counts what the pipeline actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Ground-truth labels and predicted statuses use distinct vocabularies on purpose:
# a label says what *should* happen ("fire"/"clear"); a prediction is the system's status.
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
    coverage: float | None  # assessed / total — how often the check could run at all


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
    """Fold one (label, prediction) pair into a confusion accumulator."""
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
    """Build a full report from {flag_type: [(label, predicted), ...]}.

    The overall row micro-averages by summing every flag's confusion counts — so it reflects
    per-decision performance across the whole test set, not an average of averages.
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
