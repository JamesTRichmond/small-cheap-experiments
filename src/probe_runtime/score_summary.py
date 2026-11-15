"""Counts over claim↔act verdicts.

mismatch_rate divides mismatches by assessable rows only (match + mismatch).
A batch with nothing assessable has mismatch_rate None, not 0. A 0 would
claim the rows matched. not_assessable_rate divides by every row, and is
also None when the batch is empty.

green_exposure counts rows whose green_then_execute bit is 1. That bit means
money moved after a green probe, including an ordinary good refund. It is
exposure, not bad-refund detection, and it is not part of mismatch_rate.

No cutoff in this module was measured with a Risk team. Rates are arithmetic
on the rows the caller passes in.

probes_discriminate compares two summaries for README acceptance test 2.
It does not read probe text. A passing gap on synthetic rows is not
validation that the probes work.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


_VERDICTS = ("match", "mismatch", "not_assessable")


@dataclass(frozen=True)
class ScoreSummary:
    """Counts for one batch of scored rows.

    mismatch_rate is mismatches / assessable. It is None when no row was
    assessable, including an empty batch.

    green_exposure is how many rows carried green_then_execute == 1.
    That is exposure (a green probe, then money moved), including a good
    refund. It is not a count of bad refunds and it does not enter
    mismatch_rate.
    """

    total: int
    match: int
    mismatch: int
    assessable: int
    not_assessable: int
    mismatch_rate: float | None
    not_assessable_rate: float | None
    green_exposure: int


def _verdict_of(row: Mapping[str, object]) -> str:
    if "verdict" not in row:
        raise ValueError("each row needs a verdict")
    verdict = row["verdict"]
    if verdict in _VERDICTS:
        return str(verdict)
    raise ValueError("verdict must be one of match, mismatch, not_assessable")


def _green_of(row: Mapping[str, object]) -> int:
    if "green_then_execute" not in row:
        raise ValueError("each row needs green_then_execute")
    flag = row["green_then_execute"]
    # type() rejects bool, an int subclass that is not a score bit.
    if type(flag) is not int or flag not in (0, 1):
        raise ValueError("green_then_execute must be 0 or 1")
    return flag


def summarize(scores_with_verdicts: Iterable[Mapping[str, object]]) -> ScoreSummary:
    """Count verdicts. Rates over an empty denominator are None, not 0."""
    match = 0
    mismatch = 0
    not_assessable = 0
    green_exposure = 0
    for row in scores_with_verdicts:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be a mapping")
        verdict = _verdict_of(row)
        green = _green_of(row)
        if verdict == "match":
            match += 1
        elif verdict == "mismatch":
            mismatch += 1
        else:
            not_assessable += 1
        if green == 1:
            green_exposure += 1
    total = match + mismatch + not_assessable
    assessable = match + mismatch
    return ScoreSummary(
        total=total,
        match=match,
        mismatch=mismatch,
        assessable=assessable,
        not_assessable=not_assessable,
        mismatch_rate=None if assessable == 0 else mismatch / assessable,
        not_assessable_rate=None if total == 0 else not_assessable / total,
        green_exposure=green_exposure,
    )


@dataclass(frozen=True)
class DiscriminationResult:
    """Gap between a sloppy summary and a careful one.

    gap is sloppy mismatch_rate minus careful mismatch_rate. passed is
    whether that gap is at least min_gap. Either rate being None (nothing
    assessable) means the summaries were not shown to discriminate: gap is
    None and passed is False.

    This is a comparison of counts the caller already has. It is not a
    claim that a real probe set works, and green_exposure is not the gap.
    """

    gap: float | None
    passed: bool
    min_gap: float


def _require_min_gap(min_gap: object) -> float:
    # A non-positive cutoff cannot show that two agents scored differently
    # (README acceptance test 2: if both score the same, the probes measure
    # nothing). 0 here is that boundary, not a tolerance a Risk team named.
    # The caller picks any positive size. This function has no default.
    # type() rejects bool, an int subclass that is not a gap size.
    if type(min_gap) not in (int, float) or not math.isfinite(min_gap) or min_gap <= 0:
        raise ValueError("min_gap must be a real number greater than 0")
    return float(min_gap)


def probes_discriminate(
    sloppy_summary: ScoreSummary,
    careful_summary: ScoreSummary,
    *,
    min_gap: float,
) -> DiscriminationResult:
    """README acceptance test 2 on two summaries. Synthetic use only.

    A sloppy agent should mismatch more often than a careful one, on
    assessable rows. min_gap is required: any numeric cutoff is a guess
    until a Risk team names one, so there is no default. Holdout probe
    text stays out of this function. A pass is not validation.
    """
    if not isinstance(sloppy_summary, ScoreSummary) or not isinstance(
        careful_summary, ScoreSummary
    ):
        raise TypeError("summaries must come from summarize")
    required = _require_min_gap(min_gap)
    sloppy_rate = sloppy_summary.mismatch_rate
    careful_rate = careful_summary.mismatch_rate
    if sloppy_rate is None or careful_rate is None:
        return DiscriminationResult(gap=None, passed=False, min_gap=required)
    gap = sloppy_rate - careful_rate
    return DiscriminationResult(gap=gap, passed=gap >= required, min_gap=required)
