"""Log-only fake refunds around the existing hook API.

pre_tool and post_tool keep their legacy exceptions. call_refund_log_only
catches telemetry failures and still runs the fake refund once, so a store,
sampler, or hash failure does not replace the refund result or the refund's
own exception.

Gate queue_human is recorded and is not an execution input. The legacy hook
receives a pre-supplied response and may sample a probe afterwards. That is
not matched pairing: trace["pairing"] is "unassessable", and the sealed
response_text stays empty. raw_parse_ok is not provenance.

Overhead is median and nearest-rank p95 of per-pair D_i =
instrumented_duration - baseline_duration. Arm percentiles are not subtracted.
No duration budget is asserted.
"""

from __future__ import annotations

import copy
import json
import math
import time

import pytest

from probe_runtime import hooks
from probe_runtime.hooks import (
    PAIRING_UNASSESSABLE,
    call_refund_log_only,
    parse_probe_response,
    post_tool,
    pre_tool,
)
from probe_runtime.models import ProbeParsed
from probe_runtime.score import resolve_constraint_action, score_frames

SYNTHETIC_PROBE = {
    "probe_id": "public-synthetic-probe",
    "family": "public-synthetic",
    "text": "PUBLIC SYNTHETIC probe. Not a holdout item.",
}

WARMUP = 10
MEASURED = 40


class SyntheticProviderDecline(Exception):
    pass


class PairRejected(ValueError):
    pass


class FakeRefunds:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self.applied: dict[str, dict] = {}
        self.duplicates = 0

    def create(
        self,
        *,
        key: str,
        amount_minor: int,
        currency: str,
        merchant_ref: str | None,
        decline: bool = False,
    ) -> dict:
        self.attempts.append(key)
        if decline:
            raise SyntheticProviderDecline(key)
        if key in self.applied:
            self.duplicates += 1
            return dict(self.applied[key])
        record = {
            "provider_status": "refunded",
            "provider_ref": f"fake-{len(self.applied) + 1}",
            "key": key,
            "amount_minor": amount_minor,
            "currency": currency,
            "merchant_ref": merchant_ref,
        }
        self.applied[key] = record
        return dict(record)

    def snapshot(self) -> tuple:
        return (list(self.attempts), copy.deepcopy(self.applied), self.duplicates)


class FakeReplay:
    """In-memory stand-in. Not ReplayStore. One successful append stores one record."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[dict] = []
        self.append_attempts = 0

    def append(self, record: dict) -> dict:
        self.append_attempts += 1
        if self.fail:
            raise OSError("synthetic unwritable replay store")
        stored = copy.deepcopy(record)
        self.records.append(stored)
        return stored


@pytest.fixture(autouse=True)
def synthetic_probe_and_clean_pending(monkeypatch):
    """Do not call the holdout sampler. Pending intents are process-local."""
    monkeypatch.setattr(hooks, "sample_probe", lambda rng=None: dict(SYNTHETIC_PROBE))
    hooks._PENDING.clear()
    yield
    hooks._PENDING.clear()


def _response(would_proceed: object, uncertainty: object) -> str:
    return json.dumps(
        {
            "would_proceed": would_proceed,
            "uncertainty": uncertainty,
            "goal_one_liner": "synthetic",
            "rationale_one_liner": "synthetic",
        }
    )


def _outcome(fn):
    try:
        return ("return", fn())
    except Exception as exc:
        return ("raise", type(exc), exc.args)


def _compare(
    response_text: str,
    *,
    decline: bool = False,
    replay_fail: bool = False,
    tool_args_extra: dict | None = None,
    before_refund_returns=None,
):
    key = "synthetic-key"
    tool_args = {
        "amount_minor": 250,
        "currency": "USD",
        "merchant_ref": "merchant-synthetic",
        "reason": "synthetic",
        "idempotency_key": key,
    }
    if tool_args_extra:
        tool_args.update(tool_args_extra)
    base = FakeRefunds()
    observed = FakeRefunds()
    replay = FakeReplay(fail=replay_fail)
    trace: dict = {}

    def apply(ledger: FakeRefunds):
        return ledger.create(
            key=key,
            amount_minor=250,
            currency="USD",
            merchant_ref="merchant-synthetic",
            decline=decline,
        )

    def refund():
        if before_refund_returns is not None:
            before_refund_returns()
        return apply(observed)

    base_out = _outcome(lambda: apply(base))
    inst_out = _outcome(
        lambda: call_refund_log_only(
            refund,
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_args=tool_args,
            amount_minor=250,
            currency="USD",
            probe_response_text=response_text,
            store=replay,
            provider_status="declined" if decline else "refunded",
            provider_ref=None,
            trace=trace,
        )
    )
    assert inst_out == base_out
    assert observed.snapshot() == base.snapshot()
    assert observed.attempts == [key]
    assert observed.duplicates == 0
    assert trace["pairing"] == PAIRING_UNASSESSABLE
    return trace, replay, observed


def test_benign_refund_matches_baseline_and_is_not_paired_evidence():
    text = _response(True, "low")
    trace, replay, ledger = _compare(text)
    assert trace["decision"] == "allow"
    assert trace["pre_error"] is None
    assert trace["post_error"] is None
    assert replay.append_attempts == 1
    record = trace["record"]
    assert record["frame_claim"]["response_text"] == ""
    assert record["frame_claim"]["probe_id"] == SYNTHETIC_PROBE["probe_id"]
    assert record["frame_act"]["executed"] is True
    assert record["frame_act"]["blocked_by_gate"] is False
    parsed = parse_probe_response(text)
    assert parsed == ProbeParsed(True, "low", True, "synthetic", "synthetic")
    _, action = resolve_constraint_action(
        decision=trace["decision"], executed=True, parsed=parsed
    )
    assert record["scores"] == score_frames(parsed, True, action)
    assert ledger.applied["synthetic-key"]["provider_ref"] == "fake-1"


def test_flagged_gate_output_does_not_change_the_fake_refund():
    text = _response(True, "high")
    trace, replay, ledger = _compare(text)
    assert trace["decision"] == "queue_human"
    assert "synthetic-key" in ledger.applied
    record = trace["record"]
    assert record["frame_act"]["executed"] is True
    assert record["frame_act"]["blocked_by_gate"] is False
    assert record["frame_claim"]["response_text"] == ""
    assert trace["pairing"] == "unassessable"
    assert replay.append_attempts == 1

    refused = _response(False, "low")
    refused_trace, _, refused_ledger = _compare(refused)
    assert refused_trace["decision"] == "queue_human"
    assert refused_ledger.attempts == ["synthetic-key"]
    assert "synthetic-key" in refused_ledger.applied


def test_invalid_string_claim_refunds_once_and_stays_unassessable():
    text = _response("false", "low")
    trace, _, _ = _compare(text)
    parsed = parse_probe_response(text)
    assert parsed.raw_parse_ok is False
    assert parsed.would_proceed is None
    assert trace["decision"] == "queue_human"
    assert trace["decision"] != "allow"
    assert trace["record"]["frame_claim"]["parsed"]["raw_parse_ok"] is False
    assert trace["record"]["frame_act"]["executed"] is True
    assert trace["pairing"] == "unassessable"


def test_empty_synthetic_pool_preserves_legacy_error_and_refund(monkeypatch):
    def empty_pool(rng=None):
        raise RuntimeError("synthetic empty pool")

    monkeypatch.setattr(hooks, "sample_probe", empty_pool)
    with pytest.raises(RuntimeError, match="synthetic empty pool"):
        pre_tool(
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_name="refund.create",
            tool_args={"amount_minor": 1, "currency": "USD"},
            amount_minor=1,
            currency="USD",
            probe_response_text=_response(True, "low"),
            store=FakeReplay(),
        )
    trace, replay, ledger = _compare(_response(True, "low"))
    assert isinstance(trace["pre_error"], RuntimeError)
    assert trace["record"] is None
    assert replay.append_attempts == 0
    assert ledger.attempts == ["synthetic-key"]
    assert trace["pairing"] == "unassessable"


def test_unknown_triplet_and_lost_pending_keep_keyerror():
    with pytest.raises(KeyError, match="unknown triplet_id"):
        post_tool(
            triplet_id="missing-triplet",
            executed=True,
            provider_status="refunded",
            provider_ref=None,
            store=FakeReplay(),
        )
    observed = pre_tool(
        agent_id="synthetic-agent",
        session_id="synthetic-session",
        tool_name="refund.create",
        tool_args={"amount_minor": 1, "currency": "USD"},
        amount_minor=1,
        currency="USD",
        probe_response_text=_response(True, "none"),
        store=FakeReplay(),
    )
    hooks._PENDING.clear()
    with pytest.raises(KeyError, match="unknown triplet_id"):
        post_tool(
            triplet_id=observed["triplet_id"],
            executed=True,
            provider_status="refunded",
            provider_ref=None,
            store=FakeReplay(),
        )
    trace, _, ledger = _compare(
        _response(True, "high"),
        before_refund_returns=hooks._PENDING.clear,
    )
    assert isinstance(trace["post_error"], KeyError)
    assert ledger.attempts == ["synthetic-key"]
    assert trace["record"] is None


def test_unwritable_fake_store_does_not_replace_refund_outcome():
    text = _response(True, "low")
    pending_store = FakeReplay()
    observed = pre_tool(
        agent_id="synthetic-agent",
        session_id="synthetic-session",
        tool_name="refund.create",
        tool_args={"amount_minor": 250, "currency": "USD"},
        amount_minor=250,
        currency="USD",
        probe_response_text=text,
        store=pending_store,
    )
    with pytest.raises(OSError, match="unwritable"):
        post_tool(
            triplet_id=observed["triplet_id"],
            executed=True,
            provider_status="refunded",
            provider_ref=None,
            store=FakeReplay(fail=True),
        )
    trace, replay, ledger = _compare(text, replay_fail=True)
    assert isinstance(trace["post_error"], OSError)
    assert replay.append_attempts == 1
    assert ledger.applied["synthetic-key"]["amount_minor"] == 250


def test_malformed_tool_args_do_not_block_or_duplicate_the_refund():
    bad = {"blob": {1, 2}}
    with pytest.raises(TypeError):
        pre_tool(
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_name="refund.create",
            tool_args={"amount_minor": 250, "currency": "USD", **bad},
            amount_minor=250,
            currency="USD",
            probe_response_text=_response(True, "low"),
            store=FakeReplay(),
        )
    trace, replay, ledger = _compare(_response(False, "high"), tool_args_extra=bad)
    assert isinstance(trace["pre_error"], TypeError)
    assert replay.append_attempts == 0
    assert ledger.attempts == ["synthetic-key"]
    assert len(ledger.applied) == 1


def test_nonrefund_tool_still_raises_value_error():
    with pytest.raises(ValueError, match="refund.create"):
        pre_tool(
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_name="transfer.create",
            tool_args={"amount_minor": 1, "currency": "USD"},
            amount_minor=1,
            currency="USD",
            probe_response_text=_response(True, "low"),
            store=FakeReplay(),
        )


def test_provider_decline_is_not_replaced_by_telemetry_errors(monkeypatch):
    def empty_pool(rng=None):
        raise RuntimeError("synthetic empty pool")

    monkeypatch.setattr(hooks, "sample_probe", empty_pool)

    def decline():
        raise SyntheticProviderDecline("synthetic-key")

    with pytest.raises(SyntheticProviderDecline, match="synthetic-key"):
        call_refund_log_only(
            decline,
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_args={"amount_minor": 250, "currency": "USD"},
            amount_minor=250,
            currency="USD",
            probe_response_text=_response(True, "high"),
            store=FakeReplay(fail=True),
            provider_status="declined",
            trace={},
        )
    trace, _, _ = _compare(_response(True, "high"), decline=True, replay_fail=True)
    assert trace["post_error"] is None or isinstance(trace["post_error"], OSError)
    assert isinstance(trace["pre_error"], RuntimeError)


def test_each_call_uses_a_fresh_fake_store():
    text = _response(True, "none")
    _, first, _ = _compare(text)
    _, second, _ = _compare(text)
    assert first.append_attempts == 1
    assert second.append_attempts == 1
    assert len(first.records) == 1
    assert len(second.records) == 1


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


def summarize_paired_seconds(
    baseline_s: list[float], instrumented_s: list[float]
) -> dict:
    """Median and p95 of D_i, not a difference of arm percentiles."""
    if len(baseline_s) != len(instrumented_s):
        raise PairRejected("missing pair; refusing a percentile of unequal arms")
    if not baseline_s:
        raise PairRejected("empty")
    deltas: list[float] = []
    for baseline, instrumented in zip(baseline_s, instrumented_s, strict=True):
        if isinstance(baseline, bool) or isinstance(instrumented, bool):
            raise PairRejected("bool is not a duration")
        if not isinstance(baseline, (int, float)) or not isinstance(
            instrumented, (int, float)
        ):
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
        "p": 0.95,
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


def collect_cold_warm(clock, baseline, instrumented, warmup: int, measured: int):
    cold_pair = paired_duration(clock, baseline, instrumented)
    for _ in range(warmup):
        baseline()
        instrumented()
    bases: list[float] = []
    insts: list[float] = []
    for _ in range(measured):
        base, inst = paired_duration(clock, baseline, instrumented)
        bases.append(base)
        insts.append(inst)
    cold = summarize_paired_seconds([cold_pair[0]], [cold_pair[1]])
    warm = summarize_paired_seconds(bases, insts)
    return cold, warm


def test_paired_delta_unit_empty_singleton_and_rejection():
    converted = summarize_paired_seconds([1.0], [1.25])
    assert converted["median_s"] == 0.25
    assert converted["median_ms"] == 250.0
    assert converted["p95_ms"] == 250.0
    assert converted["estimator"] == "nearest-rank"
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
    assert summary["p95_s"] != nearest_rank_p95(instrumented) - nearest_rank_p95(
        baseline
    )


def test_injected_clock_splits_cold_warm_and_rolls_back_a_failed_pair():
    clock = ScriptedClock(
        [0, 0.5, 10, 11, 100, 100.25, 200, 200.125, 300, 301, 400, 404]
    )
    cold, warm = collect_cold_warm(clock, lambda: None, lambda: None, warmup=1, measured=2)
    assert cold["n"] == 1
    assert cold["deltas_s"] == [0.5]
    assert warm["n"] == 2
    assert warm["deltas_s"] == [-0.125, 3.0]
    assert warm["median_s"] == 1.4375
    assert warm["p95_s"] == 3.0
    assert cold["deltas_s"][0] not in warm["deltas_s"]
    assert clock.calls == 12

    kept: list = []
    with pytest.raises(PairRejected):
        kept.append(paired_duration(ScriptedClock([0, 1, 2]), lambda: None, _boom))
    assert kept == []


def _boom():
    raise RuntimeError("synthetic timed failure")


def test_live_overhead_uses_monotonic_pair_deltas_without_a_budget(capsys):
    response = _response(True, "high")
    assert parse_probe_response(response).raw_parse_ok is True

    def baseline_call():
        return FakeRefunds().create(
            key="warm",
            amount_minor=10,
            currency="USD",
            merchant_ref="m",
            decline=False,
        )

    def instrumented_call():
        return call_refund_log_only(
            lambda: FakeRefunds().create(
                key="warm",
                amount_minor=10,
                currency="USD",
                merchant_ref="m",
                decline=False,
            ),
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_args={"amount_minor": 10, "currency": "USD", "merchant_ref": "m"},
            amount_minor=10,
            currency="USD",
            probe_response_text=response,
            store=FakeReplay(),
            provider_status="refunded",
            provider_ref="fake-1",
            trace=None,
        )

    clock = time.monotonic
    cold_base = duration_of(clock, baseline_call)
    cold_inst = duration_of(clock, instrumented_call)
    for _ in range(WARMUP):
        assert baseline_call() == instrumented_call()
    bases: list[float] = []
    insts: list[float] = []
    for _ in range(MEASURED):
        base_value_box = {}
        inst_value_box = {}

        def timed_base(box=base_value_box):
            box["value"] = baseline_call()

        def timed_inst(box=inst_value_box):
            box["value"] = instrumented_call()

        bases.append(duration_of(clock, timed_base))
        insts.append(duration_of(clock, timed_inst))
        assert base_value_box["value"] == inst_value_box["value"]

    cold = summarize_paired_seconds([cold_base], [cold_inst])
    warm = summarize_paired_seconds(bases, insts)
    assert cold["n"] == 1
    assert warm["n"] == MEASURED
    assert warm["median_s"] == median(warm["deltas_s"])
    assert warm["p95_s"] == nearest_rank_p95(warm["deltas_s"])
    assert all(math.isfinite(delta) for delta in warm["deltas_s"])
    arm_p95_gap = nearest_rank_p95(insts) - nearest_rank_p95(bases)
    print(
        "LOG_ONLY_LATENCY"
        f" clock=time.monotonic estimator=nearest-rank index=ceil(0.95*n)-1"
        f" cold_n={cold['n']} cold_median_ms={cold['median_ms']:.4f}"
        f" cold_p95_ms={cold['p95_ms']:.4f}"
        f" warmup={WARMUP} warm_n={warm['n']}"
        f" warm_median_ms={warm['median_ms']:.4f} warm_p95_ms={warm['p95_ms']:.4f}"
        f" warm_min_delta_ms={min(warm['deltas_s']) * 1000:.4f}"
        f" arm_p95_gap_ms_not_used={arm_p95_gap * 1000:.4f}"
        " scope=pre_tool+fake_refund+post_tool"
        " response_acquisition=excluded_precomputed"
        " sample_probe=synthetic_stub holdout_not_read=true"
        " store_reset=new_empty_fake_replay_and_refund_ledger_per_call"
        " successful_appends_per_call=1"
        " ts_utc_not_used=true"
    )
    captured = capsys.readouterr().out
    assert "LOG_ONLY_LATENCY" in captured
    assert "arm_p95_gap_ms_not_used" in captured
