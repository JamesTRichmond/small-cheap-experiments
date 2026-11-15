"""Per-case wiring. The verdicts come from compare_batch and nowhere else.

The refunds below are synthetic. They are not market evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe_runtime.head_to_head import compare_batch
from probe_runtime.per_case_stop import per_case_comparisons


def _case(case_id: str, *, ground_truth: str, would_proceed: bool) -> dict:
    return {
        "case_id": case_id,
        "ground_truth": ground_truth,
        "tool_name": "refund.create",
        "authorized": True,
        "amount_minor": 100,
        "currency": "USD",
        "merchant_ref": "synthetic-merchant",
        "executed": True,
        "probe_response_text": json.dumps(
            {"would_proceed": would_proceed, "uncertainty": "low"}
        ),
    }


def test_each_case_is_passed_to_compare_batch_alone():
    bad = _case("synthetic-bad", ground_truth="bad", would_proceed=False)
    good = _case("synthetic-good", ground_truth="good", would_proceed=True)
    got = per_case_comparisons((bad, good))
    assert len(got) == 2
    assert got[0].verdict == compare_batch((bad,)).verdict
    assert got[0].reason == compare_batch((bad,)).reason
    assert got[1].verdict == compare_batch((good,)).verdict
    assert got[1].reason == compare_batch((good,)).reason
    for item in got:
        assert item.blocked is False
        assert item.money_moved is False
        assert item.market_evidence is False
        assert item.mode == "log_only"
        assert item.attempted == 1
    assert per_case_comparisons(()) == ()


def test_this_module_does_not_carry_a_second_verdict_table():
    source = Path(__file__).resolve().parents[1] / "src" / "probe_runtime" / "per_case_stop.py"
    text = source.read_text(encoding="utf-8")
    assert "def compare_batch" not in text
    assert "def _decide" not in text
    assert "REASON_EQUAL" not in text
    assert "compare_batch" in text
