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
"""

from __future__ import annotations

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
