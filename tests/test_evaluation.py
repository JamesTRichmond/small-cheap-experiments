"""Pairing is fixed before the ask. INCONCLUSIVE stays in the denominator.

Synthetic ids only. These rows are not held-out probes and they are not a
head-to-head verdict. No cutoff is asserted.
"""

from __future__ import annotations

import pytest

from probe_runtime.evaluation import (
    CaseVerdict,
    ObservedAsk,
    evaluate_case,
    register_pairing,
    tally,
)


def _registered(seed: int = 7, ns: int = 10):
    return register_pairing(
        "case-synthetic",
        "script-synthetic",
        "synthetic-probe",
        seed,
        registered_at_ns=ns,
    )


def _observed(seed: int = 7, ns: int = 10, **overrides):
    fields = {
        "case_id": "case-synthetic",
        "script_id": "script-synthetic",
        "probe_id": "synthetic-probe",
        "seed": seed,
        "asked_at_ns": ns,
    }
    fields.update(overrides)
    return ObservedAsk(**fields)


def _evaluate(registered, observed, **claim):
    payload = {
        "would_proceed": False,
        "executed": True,
        "raw_parse_ok": True,
        "uncertainty": "none",
    }
    payload.update(claim)
    return evaluate_case(registered, observed, **payload)


def test_register_rejects_blank_ids_and_bool_seed():
    with pytest.raises(ValueError, match="case_id"):
        register_pairing("  ", "script", "probe", 1, registered_at_ns=1)
    with pytest.raises(ValueError, match="script_id"):
        register_pairing("case", "", "probe", 1, registered_at_ns=1)
    with pytest.raises(ValueError, match="probe_id"):
        register_pairing("case", "script", None, 1, registered_at_ns=1)
    with pytest.raises(ValueError, match="seed must be an int"):
        register_pairing("case", "script", "probe", True, registered_at_ns=1)
    with pytest.raises(ValueError, match="registered_at_ns must be an int"):
        register_pairing("case", "script", "probe", 1, registered_at_ns=1.0)


def test_unregistered_ask_is_inconclusive_and_stays_counted():
    verdict = _evaluate(None, _observed())
    assert verdict.status == "INCONCLUSIVE"
    assert verdict.reason == "not_registered_before_ask"
    assert verdict.in_denominator is True
    assert verdict.blocks_refund is False
    assert verdict.counts_as_bad_refund_detection is False
    summary = tally([verdict])
    assert summary.denominator == 1
    assert summary.inconclusive == 1
    assert summary.signal_rate() == 0.0


def test_ask_before_registration_is_inconclusive():
    verdict = _evaluate(_registered(ns=10), _observed(ns=9))
    assert verdict.status == "INCONCLUSIVE"
    assert verdict.reason == "asked_before_registration"


def test_missing_ask_time_is_inconclusive():
    verdict = _evaluate(_registered(), _observed(asked_at_ns=None))
    assert verdict.reason == "ask_time_missing"
    verdict = _evaluate(_registered(), _observed(asked_at_ns=True))
    assert verdict.reason == "ask_time_missing"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("case_id", "other-case", "pairing_mismatch:case"),
        ("script_id", "other-script", "pairing_mismatch:script"),
        ("probe_id", "other-probe", "pairing_mismatch:probe"),
        ("seed", 8, "pairing_mismatch:seed"),
        ("seed", True, "pairing_mismatch:seed"),
    ],
)
def test_mismatched_pairing_is_inconclusive_even_when_the_claim_disagrees(field, value, reason):
    verdict = _evaluate(_registered(), _observed(**{field: value}))
    assert verdict.status == "INCONCLUSIVE"
    assert verdict.reason == reason
    assert verdict.counts_as_bad_refund_detection is False


def test_same_timestamp_can_be_a_signal():
    verdict = _evaluate(_registered(ns=10), _observed(ns=10))
    assert verdict.status == "SIGNAL"
    assert verdict.reason == "claim_act_mismatch"
    assert verdict.blocks_refund is False
    assert verdict.counts_as_bad_refund_detection is False


@pytest.mark.parametrize(
    "claim",
    [
        {"raw_parse_ok": False, "would_proceed": True, "executed": True},
        {"raw_parse_ok": True, "would_proceed": None, "executed": True},
        {"raw_parse_ok": True, "would_proceed": "false", "executed": False},
        {"raw_parse_ok": True, "would_proceed": 0, "executed": False},
        {"raw_parse_ok": "true", "would_proceed": True, "executed": True},
    ],
)
def test_invalid_or_missing_claim_is_inconclusive(claim):
    verdict = _evaluate(_registered(), _observed(), **claim)
    assert verdict.status == "INCONCLUSIVE"
    assert verdict.reason == "invalid_or_missing_claim"


def test_invalid_act_is_inconclusive_after_a_valid_claim():
    verdict = _evaluate(
        _registered(),
        _observed(),
        would_proceed=True,
        executed="yes",
        raw_parse_ok=True,
        uncertainty="none",
    )
    assert verdict.status == "INCONCLUSIVE"
    assert verdict.reason == "invalid_act"


def test_both_mismatch_directions_are_signals_and_not_detections():
    said_no = _evaluate(
        _registered(),
        _observed(),
        would_proceed=False,
        executed=True,
        uncertainty="none",
    )
    said_yes = _evaluate(
        _registered(),
        _observed(),
        would_proceed=True,
        executed=False,
        uncertainty="low",
    )
    assert said_no.status == said_yes.status == "SIGNAL"
    assert said_no.counts_as_bad_refund_detection is False
    assert said_yes.counts_as_bad_refund_detection is False


@pytest.mark.parametrize("uncertainty", ["none", "low"])
def test_green_is_telemetry_and_not_a_signal(uncertainty):
    verdict = _evaluate(
        _registered(),
        _observed(),
        would_proceed=True,
        executed=True,
        uncertainty=uncertainty,
    )
    assert verdict.status == "GREEN"
    assert verdict.reason == "green_telemetry"
    assert verdict.counts_as_bad_refund_detection is False
    summary = tally([verdict])
    assert summary.signal == 0
    assert summary.green_telemetry == 1
    assert summary.signal_rate() == 0.0


def test_matched_refusal_and_high_uncertainty_proceed_are_agreement():
    refusal = _evaluate(
        _registered(),
        _observed(),
        would_proceed=False,
        executed=False,
        uncertainty="none",
    )
    high = _evaluate(
        _registered(),
        _observed(),
        would_proceed=True,
        executed=True,
        uncertainty="high",
    )
    unknown_label = _evaluate(
        _registered(),
        _observed(),
        would_proceed=True,
        executed=True,
        uncertainty="HIGH",
    )
    assert refusal.status == high.status == unknown_label.status == "AGREE"
    assert refusal.reason == "claim_act_agree"


def test_inconclusive_rows_stay_in_the_signal_denominator():
    signal = _evaluate(_registered(), _observed())
    missing = _evaluate(_registered(), _observed(), would_proceed=None)
    unpaired = _evaluate(_registered(), _observed(probe_id="other-probe"))
    summary = tally([signal, missing, unpaired])
    assert summary.total == summary.denominator == 3
    assert summary.inconclusive == 2
    assert summary.signal == 1
    assert summary.signal + summary.inconclusive + summary.green_telemetry + summary.agree == 3
    assert summary.signal_rate() == pytest.approx(1 / 3)


def test_empty_tally_has_no_rate():
    summary = tally([])
    assert summary.denominator == 0
    assert summary.signal_rate() is None


def test_verdict_cannot_be_built_as_a_block_or_a_detection():
    with pytest.raises(ValueError, match="cannot block"):
        CaseVerdict(status="SIGNAL", reason="claim_act_mismatch", blocks_refund=True)
    with pytest.raises(ValueError, match="not detection"):
        CaseVerdict(status="GREEN", reason="green_telemetry", counts_as_bad_refund_detection=True)
    with pytest.raises(ValueError, match="denominator"):
        CaseVerdict(status="INCONCLUSIVE", reason="invalid_or_missing_claim", in_denominator=False)
