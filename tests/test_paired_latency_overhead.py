"""Paired overhead of one fake refund with the hook and gate observing.

D_i = instrumented_duration_i - baseline_duration_i. The reported median and
p95 are statistics of those deltas, not a difference of arm percentiles.
The deterministic cases use an injected clock. A live run uses
time.monotonic and does not assert a duration budget: the README states none.
Any numeric cutoff added later would be a guess until a Risk conversation
names one.

The instrumented arm calls pre_tool, the fake refund, and post_tool. The
response text is built before the clock starts. sample_probe is a public
synthetic stub, so holdout files are not read. Each pair uses a new empty
fake replay store.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import pytest

from probe_runtime import hooks
from probe_runtime.hooks import post_tool, pre_tool

SYNTHETIC_PROBE = {
    "probe_id": "public-synthetic-probe",
    "family": "public-synthetic",
    "text": "PUBLIC SYNTHETIC probe. Not a holdout item.",
}
WARMUP = 10
MEASURED = 40


class PairRejected(ValueError):
    pass


class FakeRefunds:
    def create(self, *, key: str, amount_minor: int) -> dict:
        return {
            "provider_status": "refunded",
            "provider_ref": "fake-1",
            "key": key,
            "amount_minor": amount_minor,
        }


class FakeReplay:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, record: dict) -> dict:
        stored = copy.deepcopy(record)
        self.records.append(stored)
        return stored


@pytest.fixture(autouse=True)
def synthetic_probe(monkeypatch):
    monkeypatch.setattr(hooks, "sample_probe", lambda rng=None: dict(SYNTHETIC_PROBE))
    hooks._PENDING.clear()
    yield
    hooks._PENDING.clear()


def median(values: list[float]) -> float:
    if not values:
        raise PairRejected("empty")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def nearest_rank_p95(values: list[float]) -> float:
    """Nearest-rank inclusive p95. Sorted index is ceil(0.95 * n) - 1."""
    if not values:
        raise PairRejected("empty")
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1
    return ordered[index]


def summarize_paired_seconds(baseline_s: list[float], instrumented_s: list[float]) -> dict:
    if len(baseline_s) != len(instrumented_s):
        raise PairRejected("missing pair; refusing a percentile of unequal arms")
    if not baseline_s:
        raise PairRejected("empty")
    deltas: list[float] = []
    for baseline, instrumented in zip(baseline_s, instrumented_s, strict=True):
        if isinstance(baseline, bool) or isinstance(instrumented, bool):
            raise PairRejected("bool is not a duration")
        if not isinstance(baseline, (int, float)) or not isinstance(instrumented, (int, float)):
            raise PairRejected("failing pair")
        if not math.isfinite(float(baseline)) or not math.isfinite(float(instrumented)):
            raise PairRejected("non-finite duration")
        deltas.append(float(instrumented) - float(baseline))
    med = median(deltas)
    p95 = nearest_rank_p95(deltas)
    return {
        "n": len(deltas),
        "deltas_s": deltas,
        "median_s": med,
        "p95_s": p95,
        "median_ms": med * 1000.0,
        "p95_ms": p95 * 1000.0,
        "estimator": "nearest-rank",
        "index": "ceil(0.95*n)-1",
    }


class ScriptedClock:
    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)
        self.calls = 0

    def __call__(self) -> float:
        value = self._ticks[self.calls]
        self.calls += 1
        return value


def duration_of(clock, fn) -> float:
    start = clock()
    try:
        fn()
    except Exception as exc:
        raise PairRejected("pair failed; duration rolled back") from exc
    return clock() - start


def paired_duration(clock, baseline, instrumented) -> tuple[float, float]:
    base = duration_of(clock, baseline)
    try:
        inst = duration_of(clock, instrumented)
    except PairRejected as exc:
        raise PairRejected("instrumented failed; baseline duration not kept") from exc
    return base, inst


def test_readme_states_no_overhead_budget():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "overhead" not in lowered
    assert "p95" not in lowered
    # No budget is asserted below. A threshold here would be a guess.


def test_injected_clock_reports_median_and_p95_of_deltas():
    converted = summarize_paired_seconds([1.0], [1.25])
    assert converted["median_s"] == 0.25
    assert converted["median_ms"] == 250.0
    assert converted["p95_s"] == 0.25
    assert converted["n"] == 1

    negative = summarize_paired_seconds([2.0], [1.5])
    assert negative["deltas_s"] == [-0.5]
    assert negative["median_s"] == -0.5
    assert negative["p95_s"] == -0.5

    with pytest.raises(PairRejected):
        summarize_paired_seconds([], [])
    with pytest.raises(PairRejected):
        summarize_paired_seconds([1.0], [1.0, 2.0])
    with pytest.raises(PairRejected):
        summarize_paired_seconds([1.0, None], [1.0, 2.0])  # type: ignore[list-item]
    with pytest.raises(PairRejected):
        summarize_paired_seconds(
            ["2026-11-07T23:00:00+00:00"],  # type: ignore[list-item]
            ["2026-11-07T23:00:01+00:00"],  # type: ignore[list-item]
        )

    baseline = [1.0, 2.0, 100.0]
    instrumented = [5.0, 4.0, 10.0]
    summary = summarize_paired_seconds(baseline, instrumented)
    assert summary["deltas_s"] == [4.0, 2.0, -90.0]
    assert summary["median_s"] == 2.0
    assert summary["p95_s"] == 4.0
    assert summary["median_s"] != median(instrumented) - median(baseline)
    assert summary["p95_s"] != nearest_rank_p95(instrumented) - nearest_rank_p95(baseline)

    clock = ScriptedClock([0, 0.5, 10, 11, 100, 100.25, 200, 200.125, 300, 301, 400, 404])
    cold = paired_duration(clock, lambda: None, lambda: None)
    for _ in range(1):
        (lambda: None)()
        (lambda: None)()
    warm_pairs = [
        paired_duration(clock, lambda: None, lambda: None),
        paired_duration(clock, lambda: None, lambda: None),
    ]
    cold_summary = summarize_paired_seconds([cold[0]], [cold[1]])
    warm_summary = summarize_paired_seconds(
        [pair[0] for pair in warm_pairs], [pair[1] for pair in warm_pairs]
    )
    assert cold_summary["n"] == 1
    assert cold_summary["deltas_s"] == [0.5]
    assert warm_summary["n"] == 2
    assert warm_summary["deltas_s"] == [-0.125, 3.0]
    assert warm_summary["median_s"] == 1.4375
    assert warm_summary["p95_s"] == 3.0
    assert cold_summary["deltas_s"][0] not in warm_summary["deltas_s"]

    kept: list = []
    with pytest.raises(PairRejected):
        kept.append(
            paired_duration(
                ScriptedClock([0, 1, 2]),
                lambda: None,
                lambda: (_ for _ in ()).throw(RuntimeError("synthetic timed failure")),
            )
        )
    assert kept == []


def _response() -> str:
    return json.dumps(
        {
            "would_proceed": True,
            "uncertainty": "high",
            "goal_one_liner": "synthetic",
            "rationale_one_liner": "synthetic",
        }
    )


def _baseline() -> dict:
    return FakeRefunds().create(key="warm", amount_minor=10)


def _instrumented(response: str) -> dict:
    replay = FakeReplay()
    observed = pre_tool(
        agent_id="synthetic-agent",
        session_id="synthetic-session",
        tool_name="refund.create",
        tool_args={"amount_minor": 10, "currency": "USD", "merchant_ref": "m"},
        amount_minor=10,
        currency="USD",
        probe_response_text=response,
        store=replay,
    )
    result = FakeRefunds().create(key="warm", amount_minor=10)
    # Decision is observed and is not allowed to skip the refund above.
    assert observed["decision"] in ("allow", "queue_human")
    post_tool(
        triplet_id=observed["triplet_id"],
        executed=True,
        provider_status=result["provider_status"],
        provider_ref=result["provider_ref"],
        store=replay,
    )
    assert len(replay.records) == 1
    return result


def test_live_hook_overhead_has_no_duration_gate(capsys):
    response = _response()
    clock = time.monotonic
    cold_base = duration_of(clock, _baseline)
    cold_inst = duration_of(clock, lambda: _instrumented(response))
    for _ in range(WARMUP):
        assert _baseline() == _instrumented(response)
    bases: list[float] = []
    insts: list[float] = []
    for _ in range(MEASURED):
        bases.append(duration_of(clock, _baseline))
        insts.append(duration_of(clock, lambda: _instrumented(response)))
        assert _baseline() == _instrumented(response)
    cold = summarize_paired_seconds([cold_base], [cold_inst])
    warm = summarize_paired_seconds(bases, insts)
    assert cold["n"] == 1
    assert warm["n"] == MEASURED
    assert warm["median_s"] == median(warm["deltas_s"])
    assert warm["p95_s"] == nearest_rank_p95(warm["deltas_s"])
    assert all(math.isfinite(delta) for delta in warm["deltas_s"])
    arm_p95_gap = nearest_rank_p95(insts) - nearest_rank_p95(bases)
    print(
        "PAIRED_LATENCY"
        f" clock=time.monotonic estimator=nearest-rank index=ceil(0.95*n)-1"
        f" cold_n={cold['n']} cold_median_ms={cold['median_ms']:.4f} cold_p95_ms={cold['p95_ms']:.4f}"
        f" warmup={WARMUP} warm_n={warm['n']}"
        f" warm_median_ms={warm['median_ms']:.4f} warm_p95_ms={warm['p95_ms']:.4f}"
        f" warm_min_delta_ms={min(warm['deltas_s']) * 1000:.4f}"
        f" arm_p95_gap_ms_not_used={arm_p95_gap * 1000:.4f}"
        " scope=pre_tool+gate+fake_refund+post_tool"
        " response_acquisition=excluded_precomputed"
        " sample_probe=synthetic_stub holdout_not_read=true"
        " store_reset=new_empty_fake_replay_per_instrumented_call"
        " budget=none README_states_none=true threshold_would_be_a_guess=true"
        " ts_utc_not_used=true"
    )
    assert "budget=none" in capsys.readouterr().out
