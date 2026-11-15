"""Mix seeded canary refunds into a log-only run.

Each canary is the bad row from ``metadata_twin_pair``: the same tool,
authorization, amount, currency, merchant, and fake execution as a good
donor, and the opposite ground truth. The row is the generator's return
value. This module does not copy those fields onto a second dict.

A details-only rule sees the same allow or deny on a canary and its twin.
``details_only_leak`` runs ``search_strongest_baseline`` on the rows and
compares how often that rule denies canaries with how often it denies their
twins. A higher canary rate is LEAKING, and recall is not emitted. Equal
rates are not a leak.

The run does not block a refund and does not move money. ``executed`` on a
row is the generator's fake execution. Recall on synthetic canaries is not
detection performance. When the rate rounds to no canaries, recall is
INCONCLUSIVE rather than 0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping, Sequence

from probe_runtime.baseline_search import search_strongest_baseline
from probe_runtime.metadata_twins import metadata_twin_pair

_LEAKING = "LEAKING"


@dataclass(frozen=True)
class CanaryKey:
    """Which log rows are canaries. Not a field of those rows."""

    seed: int
    rate: float
    n: int
    positions: tuple[int, ...]
    twin_positions: tuple[int, ...]
    known_bad: bool = True
    fake_money: bool = True

    @property
    def canary_count(self) -> int:
        return len(self.positions)

    def __post_init__(self) -> None:
        if self.known_bad is not True or self.fake_money is not True:
            raise ValueError("canaries are known-bad synthetic refunds and do not move money")
        if len(self.twin_positions) != len(self.positions):
            raise ValueError("each canary needs its good twin")


@dataclass(frozen=True)
class LogOnlyMix:
    records: tuple[dict, ...]
    leak_cases: tuple[dict, ...]
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
    rule_id: str | None = None
    canary_deny_count: int | None = None
    twin_deny_count: int | None = None
    synthetic_recall_is_not_detection_performance: bool = True
    blocks_refund: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("recall", "INCONCLUSIVE", _LEAKING):
            raise ValueError("unknown recall status")
        if self.status != "recall" and self.recall is not None:
            raise ValueError("recall is emitted only when the mix is not leaking and has canaries")
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


def _meta(case: Mapping) -> dict:
    amount = case.get("amount_minor", 100)
    currency = case.get("currency", "USD")
    authorized = case.get("authorized", True)
    tool_name = case.get("tool_name", "refund.create")
    merchant_ref = case.get("merchant_ref", "toy-twin")
    if type(amount) is not int or isinstance(amount, bool):
        raise ValueError("amount_minor must be an int")
    if type(currency) is not str:
        raise ValueError("currency must be a string")
    if type(authorized) is not bool:
        raise ValueError("authorized must be a bool")
    if type(tool_name) is not str or not tool_name:
        raise ValueError("tool_name must be a non-empty string")
    if type(merchant_ref) is not str:
        raise ValueError("merchant_ref must be a string")
    return {
        "amount_minor": amount,
        "currency": currency,
        "authorized": authorized,
        "tool_name": tool_name,
        "merchant_ref": merchant_ref,
    }


def _pair(pair_id: str, case: Mapping) -> list[dict]:
    return metadata_twin_pair(pair_id, **_meta(case))


def details_only_leak(
    cases: Sequence[Mapping],
    canary_positions: Sequence[int],
    twin_positions: Sequence[int],
) -> tuple[str, str, int, int]:
    """Run the metadata-rule search. Return status, rule id, and deny counts.

    Status is LEAKING when the best rule denies more canaries than the good
    twins those canaries were built from. Otherwise it is ``clear``.
    """
    if len(canary_positions) != len(twin_positions):
        raise ValueError("each canary needs its good twin")
    if not cases:
        raise ValueError("the leak check needs the rows it searches")
    found = search_strongest_baseline(list(cases), leave_one_out=False)
    denies = [not found.rule.allows(case) for case in cases]
    canary_denies = 0
    twin_denies = 0
    for canary_at, twin_at in zip(canary_positions, twin_positions):
        canary_denies += int(denies[canary_at])
        twin_denies += int(denies[twin_at])
    if canary_denies > twin_denies:
        return _LEAKING, found.rule_id, canary_denies, twin_denies
    return "clear", found.rule_id, canary_denies, twin_denies


def mix_log(
    cases: Sequence[Mapping],
    *,
    rate: float,
    seed: int,
) -> LogOnlyMix:
    """Return log rows built by the twin generator, and the outside key."""
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    n = len(cases)
    count = canary_count_for(n, rate)
    rng = random.Random(seed)
    positions = tuple(sorted(rng.sample(range(n), count))) if count else ()
    canary_at = set(positions)
    donors = [index for index in range(n) if index not in canary_at]
    records: list[dict] = []
    twin_positions: list[int] = []
    extra_goods: list[dict] = []

    if not donors and positions:
        for index in positions:
            bad, good = _pair(f"synthetic-{index}", cases[index])
            records.append(bad)
            extra_goods.append(good)
        leak_cases = tuple(records) + tuple(extra_goods)
        twin_positions = list(range(n, 2 * n))
    else:
        donor_slot = {index: index for index in donors}
        built: list[dict | None] = [None] * n
        for index in donors:
            _bad, good = _pair(f"synthetic-good-{index}", cases[index])
            built[index] = good
        for index in positions:
            donor = donors[rng.randrange(len(donors))]
            bad, _good = _pair(f"synthetic-{index}", cases[donor])
            built[index] = bad
            twin_positions.append(donor_slot[donor])
        records = [row for row in built if row is not None]
        leak_cases = tuple(records)

    key = CanaryKey(
        seed=seed,
        rate=float(rate),
        n=n,
        positions=positions,
        twin_positions=tuple(twin_positions),
    )
    return LogOnlyMix(records=tuple(records), leak_cases=leak_cases, key=key)


def canary_recall(flags: Sequence[object], mix: LogOnlyMix) -> CanaryRecall:
    """Fraction of canary rows the caller flagged, unless the mix leaks.

    The leak check uses the metadata-rule search. A leaking mix has no recall.
    """
    if len(flags) != mix.key.n:
        raise ValueError("flags must cover every logged row")
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("each flag must be a bool")
    if mix.key.canary_count == 0:
        return CanaryRecall(
            status="INCONCLUSIVE",
            recall=None,
            hits=0,
            canary_count=0,
        )
    status, rule_id, canary_denies, twin_denies = details_only_leak(
        mix.leak_cases,
        mix.key.positions if mix.leak_cases == mix.records else range(mix.key.n),
        mix.key.twin_positions,
    )
    if status == _LEAKING:
        return CanaryRecall(
            status=_LEAKING,
            recall=None,
            hits=0,
            canary_count=mix.key.canary_count,
            rule_id=rule_id,
            canary_deny_count=canary_denies,
            twin_deny_count=twin_denies,
        )
    hits = sum(1 for index in mix.key.positions if flags[index])
    return CanaryRecall(
        status="recall",
        recall=hits / mix.key.canary_count,
        hits=hits,
        canary_count=mix.key.canary_count,
        rule_id=rule_id,
        canary_deny_count=canary_denies,
        twin_deny_count=twin_denies,
    )
