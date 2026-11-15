"""Canaries match the refund schema and are named only by the outside key.

Fixtures are synthetic. Recall on them is not detection performance.
"""

from __future__ import annotations

import json

import pytest

from probe_runtime.canary_log import (
    LOG_KEYS,
    canary_count_for,
    canary_recall,
    mix_log,
)


def _case(index: int) -> dict:
    return {
        "amount_minor": 100 + index,
        "currency": "USD",
        "merchant_ref": f"merchant-{index}",
        "reason": f"reason-{index}",
        "executed": True,
        "is_canary": True,
        "note": "SYNTHETIC normal traffic",
    }


def _mix(n: int = 8, rate: float = 0.25, seed: int = 11):
    return mix_log([_case(i) for i in range(n)], rate=rate, seed=seed)


def test_logged_rows_share_a_schema_and_carry_no_canary_marker():
    mixed = _mix()
    assert mixed.blocks_refund is False
    assert mixed.money_moved is False
    assert mixed.key.known_bad is True
    assert mixed.key.fake_money is True
    assert mixed.key.positions
    blob = json.dumps(mixed.records)
    assert "canary" not in blob.lower()
    assert "known_bad" not in blob
    assert "fake_money" not in blob
    assert str(mixed.key.seed) not in blob
    keys = {tuple(record) for record in mixed.records}
    assert keys == {LOG_KEYS}
    for record in mixed.records:
        assert record["tool_name"] == "refund.create"
        assert type(record["amount_minor"]) is int and record["amount_minor"] >= 0
        assert type(record["currency"]) is str and len(record["currency"]) == 3
        assert record["merchant_ref"] is None or type(record["merchant_ref"]) is str
        assert record["reason"] is None or type(record["reason"]) is str
        assert record["executed"] is False
        assert record["provider_status"] == "not_sent"


def test_canary_field_values_also_appear_on_a_normal_row():
    mixed = _mix()
    canary_at = set(mixed.key.positions)
    normal = [row for i, row in enumerate(mixed.records) if i not in canary_at]
    canaries = [row for i, row in enumerate(mixed.records) if i in canary_at]
    assert normal and canaries
    for field in LOG_KEYS:
        normal_values = {row[field] for row in normal}
        canary_values = {row[field] for row in canaries}
        assert canary_values <= normal_values


def test_the_same_seed_repeats_and_a_different_seed_can_move():
    assert _mix(seed=11).key.positions == _mix(seed=11).key.positions
    assert canary_count_for(8, 0.25) == 2
    assert len(_mix().key.positions) == 2


def test_recall_uses_the_outside_key_and_is_not_a_detection_claim():
    mixed = _mix()
    caught = [index in mixed.key.positions for index in range(mixed.key.n)]
    report = canary_recall(caught, mixed.key)
    assert report.status == "recall"
    assert report.recall == 1.0
    assert report.synthetic_recall_is_not_detection_performance is True
    assert report.blocks_refund is False
    missed = canary_recall([False] * mixed.key.n, mixed.key)
    assert missed.recall == 0.0
    assert missed.synthetic_recall_is_not_detection_performance is True


def test_no_canaries_makes_recall_inconclusive():
    mixed = mix_log([_case(0)], rate=0.2, seed=1)
    assert mixed.key.positions == ()
    report = canary_recall([False], mixed.key)
    assert report.status == "INCONCLUSIVE"
    assert report.recall is None


def test_log_only_flags_cannot_be_turned_into_a_block():
    with pytest.raises(ValueError, match="log-only"):
        from probe_runtime.canary_log import LogOnlyMix

        mixed = _mix()
        LogOnlyMix(records=mixed.records, key=mixed.key, blocks_refund=True)
