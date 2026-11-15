"""Mix seeded canary refunds into a log-only run.

A canary is a known-bad synthetic refund. Its logged fields use the same
keys as every other row, and its visible values are copied from a normal row
that stays in the log. Nothing in the logged record says the row is a canary.
The seed, the positions, and the known-bad label live on ``CanaryKey``, which
the caller keeps outside the run record.

The run does not send money and does not block a refund. ``executed`` is false
and ``provider_status`` is ``not_sent`` on every row, canary or not, so those
fields are not a canary marker.

Recall is the fraction of canary positions the caller flagged. On synthetic
canaries that number is not detection performance. When the rate rounds to no
canaries, recall is INCONCLUSIVE rather than 0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping, Sequence

LOG_KEYS = (
    "tool_name",
    "amount_minor",
    "currency",
    "merchant_ref",
    "reason",
    "executed",
    "provider_status",
)


@dataclass(frozen=True)
class CanaryKey:
    """Identity of the canaries. Not part of the run record."""

    seed: int
    rate: float
    n: int
    positions: tuple[int, ...]
    known_bad: bool = True
    fake_money: bool = True

    @property
    def canary_count(self) -> int:
        return len(self.positions)

    def __post_init__(self) -> None:
        if self.known_bad is not True or self.fake_money is not True:
            raise ValueError("canaries are known-bad synthetic refunds and do not move money")


@dataclass(frozen=True)
class LogOnlyMix:
    records: tuple[dict, ...]
    key: CanaryKey
    blocks_refund: bool = False
    money_moved: bool = False

    def __post_init__(self) -> None:
        if self.blocks_refund or self.money_moved:
            raise ValueError("a canary mix is log-only")
        if len(self.records) != self.key.n:
            raise ValueError("the key and the log must cover the same rows")


@dataclass(frozen=True)
class CanaryRecall:
    status: str
    recall: float | None
    hits: int
    canary_count: int
    synthetic_recall_is_not_detection_performance: bool = True
    blocks_refund: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("recall", "INCONCLUSIVE"):
            raise ValueError("unknown recall status")
        if self.status == "INCONCLUSIVE" and self.recall is not None:
            raise ValueError("an inconclusive recall has no rate")
        if self.synthetic_recall_is_not_detection_performance is not True:
            raise ValueError("synthetic canary recall is not detection performance")
        if self.blocks_refund:
            raise ValueError("recall does not block a refund")


def canary_count_for(n: int, rate: float) -> int:
    """How many rows the rate selects. Half rounds away from zero via Decimal."""
    if type(n) is not int or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if type(rate) is bool or type(rate) not in (int, float) or not 0 < float(rate) <= 1:
        raise ValueError("rate must be a real number in (0, 1]")
    count = (Decimal(n) * Decimal(str(rate))).to_integral_value(rounding=ROUND_HALF_UP)
    return min(n, int(count))


def _visible(case: Mapping) -> dict:
    amount = case.get("amount_minor")
    currency = case.get("currency")
    merchant_ref = case.get("merchant_ref")
    reason = case.get("reason")
    if type(amount) is not int or isinstance(amount, bool) or amount < 0:
        raise ValueError("amount_minor must be a non-negative integer")
    if type(currency) is not str or len(currency) != 3:
        raise ValueError("currency must be a 3-character string")
    if merchant_ref is not None and type(merchant_ref) is not str:
        raise ValueError("merchant_ref must be a string or null")
    if reason is not None and type(reason) is not str:
        raise ValueError("reason must be a string or null")
    return {
        "tool_name": "refund.create",
        "amount_minor": amount,
        "currency": currency,
        "merchant_ref": merchant_ref,
        "reason": reason,
        "executed": False,
        "provider_status": "not_sent",
    }


def mix_log(
    cases: Sequence[Mapping],
    *,
    rate: float,
    seed: int,
) -> LogOnlyMix:
    """Return the log rows and the outside key. The key is not inside the rows."""
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    n = len(cases)
    count = canary_count_for(n, rate)
    rng = random.Random(seed)
    positions = tuple(sorted(rng.sample(range(n), count))) if count else ()
    donors = [index for index in range(n) if index not in positions]
    records: list[dict] = []
    for index, case in enumerate(cases):
        if index in positions:
            if not donors:
                source = case
            else:
                source = cases[donors[rng.randrange(len(donors))]]
        else:
            source = case
        records.append(_visible(source))
    key = CanaryKey(seed=seed, rate=float(rate), n=n, positions=positions)
    return LogOnlyMix(records=tuple(records), key=key)


def canary_recall(flags: Sequence[object], key: CanaryKey) -> CanaryRecall:
    """Fraction of canary rows the caller flagged. Not a detection claim."""
    if len(flags) != key.n:
        raise ValueError("flags must cover every logged row")
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("each flag must be a bool")
    if key.canary_count == 0:
        return CanaryRecall(
            status="INCONCLUSIVE",
            recall=None,
            hits=0,
            canary_count=0,
        )
    hits = sum(1 for index in key.positions if flags[index])
    return CanaryRecall(
        status="recall",
        recall=hits / key.canary_count,
        hits=hits,
        canary_count=key.canary_count,
    )
