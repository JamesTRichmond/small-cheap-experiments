"""Acceptance test 1: the hook and gate only log a fake refund.

README: it never blocks a refund until a Risk person says what a false alarm
costs. PR #20 marks that test not covered. queue_human is not a veto.
pre_tool and post_tool still raise on their own. call_refund_log_only catches
a raise or a TimeoutError and still runs the refund once.

Hinton's question 1 is open: a queued refund that executes may still be stored
with humility_failed_to_block set by the current scorer. These tests do not
decide that bit.
"""

from __future__ import annotations

import copy
import json

import pytest

from probe_runtime import hooks
from probe_runtime.hooks import call_refund_log_only, post_tool, pre_tool

SYNTHETIC_PROBE = {
    "probe_id": "public-synthetic-probe",
    "family": "public-synthetic",
    "text": "PUBLIC SYNTHETIC probe. Not a holdout item.",
}


class SyntheticProviderDecline(Exception):
    pass


class FakeRefunds:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self.applied: dict[str, dict] = {}

    def create(self, *, key: str, amount_minor: int, decline: bool = False) -> dict:
        self.attempts.append(key)
        if decline:
            raise SyntheticProviderDecline(key)
        record = {
            "provider_status": "refunded",
            "provider_ref": f"fake-{len(self.applied) + 1}",
            "key": key,
            "amount_minor": amount_minor,
        }
        self.applied[key] = record
        return dict(record)


class FakeReplay:
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
def synthetic_probe(monkeypatch):
    monkeypatch.setattr(hooks, "sample_probe", lambda rng=None: dict(SYNTHETIC_PROBE))
    hooks._PENDING.clear()
    yield
    hooks._PENDING.clear()


def _response(would_proceed: bool, uncertainty: str) -> str:
    return json.dumps(
        {
            "would_proceed": would_proceed,
            "uncertainty": uncertainty,
            "goal_one_liner": "synthetic",
            "rationale_one_liner": "synthetic",
        }
    )


def _run(response: str, *, decline: bool = False, replay_fail: bool = False, before=None):
    base = FakeRefunds()
    observed = FakeRefunds()
    replay = FakeReplay(fail=replay_fail)
    trace: dict = {}

    def apply(ledger: FakeRefunds):
        return ledger.create(key="synthetic-key", amount_minor=250, decline=decline)

    def refund():
        if before is not None:
            before()
        return apply(observed)

    def baseline():
        return apply(base)

    try:
        base_result = ("return", baseline())
    except Exception as exc:
        base_result = ("raise", type(exc), exc.args)
    try:
        inst_result = (
            "return",
            call_refund_log_only(
                refund,
                agent_id="synthetic-agent",
                session_id="synthetic-session",
                tool_args={"amount_minor": 250, "currency": "USD", "merchant_ref": "m"},
                amount_minor=250,
                currency="USD",
                probe_response_text=response,
                store=replay,
                provider_status="declined" if decline else "refunded",
                trace=trace,
            ),
        )
    except Exception as exc:
        inst_result = ("raise", type(exc), exc.args)

    assert inst_result == base_result
    assert observed.attempts == base.attempts == ["synthetic-key"]
    assert observed.applied.keys() == base.applied.keys()
    return trace, replay


def test_allow_path_logs_one_refund_and_does_not_block():
    trace, replay = _run(_response(True, "low"))
    assert trace["decision"] == "allow"
    assert trace["pre_error"] is None and trace["post_error"] is None
    assert replay.append_attempts == 1
    assert len(replay.records) == 1
    assert trace["record"]["frame_act"]["executed"] is True
    assert trace["record"]["frame_act"]["blocked_by_gate"] is False
    assert trace["record"]["frame_act"]["tool_name"] == "refund.create"


def test_queue_human_still_refunds_once_and_is_logged():
    trace, replay = _run(_response(True, "high"))
    assert trace["decision"] == "queue_human"
    assert trace["record"]["frame_act"]["executed"] is True
    assert trace["record"]["frame_act"]["blocked_by_gate"] is False
    assert replay.append_attempts == 1
    # TODO: do not read humility_failed_to_block as a pass or a fail.
    # Whether queue_human plus executed is a humility failure is open.


def test_refused_claim_still_executes_the_fake_refund():
    trace, replay = _run(_response(False, "none"))
    assert trace["decision"] == "queue_human"
    assert trace["record"]["frame_act"]["executed"] is True
    assert replay.records[0]["frame_act"]["executed"] is True


def test_hook_timeout_and_raise_do_not_block_or_duplicate(monkeypatch):
    def time_out(rng=None):
        raise TimeoutError("synthetic hook timeout")

    monkeypatch.setattr(hooks, "sample_probe", time_out)
    with pytest.raises(TimeoutError, match="synthetic hook timeout"):
        pre_tool(
            agent_id="synthetic-agent",
            session_id="synthetic-session",
            tool_name="refund.create",
            tool_args={"amount_minor": 250, "currency": "USD"},
            amount_minor=250,
            currency="USD",
            probe_response_text=_response(True, "low"),
            store=FakeReplay(),
        )
    trace, replay = _run(_response(True, "low"))
    assert isinstance(trace["pre_error"], TimeoutError)
    assert trace["record"] is None
    assert replay.append_attempts == 0
    assert trace["decision"] is None

    def explode(rng=None):
        raise RuntimeError("synthetic empty pool")

    monkeypatch.setattr(hooks, "sample_probe", explode)
    trace, replay = _run(_response(False, "high"))
    assert isinstance(trace["pre_error"], RuntimeError)
    assert replay.append_attempts == 0


def test_post_tool_timeout_after_a_lost_pending_keeps_the_refund():
    with pytest.raises(KeyError, match="unknown triplet_id"):
        post_tool(
            triplet_id="missing",
            executed=True,
            provider_status="refunded",
            provider_ref=None,
            store=FakeReplay(),
        )

    # The refund callback runs after pre_tool. Clearing pending makes the
    # following post_tool raise KeyError, which the wrapper swallows.
    trace, replay = _run(_response(True, "high"), before=hooks._PENDING.clear)
    assert isinstance(trace["post_error"], KeyError)
    assert replay.append_attempts == 0 or trace["record"] is None

    def post_times_out(**kwargs):
        raise TimeoutError("synthetic post timeout")

    original = hooks.post_tool
    hooks.post_tool = post_times_out
    try:
        trace, _ = _run(_response(True, "none"))
    finally:
        hooks.post_tool = original
    assert isinstance(trace["post_error"], TimeoutError)
    assert trace["decision"] == "allow"


def test_unwritable_log_does_not_replace_the_refund_or_its_error():
    trace, replay = _run(_response(True, "low"), replay_fail=True)
    assert isinstance(trace["post_error"], OSError)
    assert replay.append_attempts == 1
    assert trace["record"] is None

    decline_trace, _ = _run(_response(True, "high"), decline=True, replay_fail=True)
    assert decline_trace["pre_error"] is None
    assert isinstance(decline_trace["post_error"], OSError)


def test_direct_nonrefund_tool_still_raises():
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
