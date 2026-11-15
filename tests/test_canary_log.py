"""Canaries are metadata twins from the twin generator.

Recall is withheld when a details-only rule denies canaries more often than
their good twins. Fixtures are synthetic. Recall is not detection performance.
"""

from __future__ import annotations

import json

import pytest

from probe_runtime.canary_log import (
    canary_count_for,
    canary_recall,
    details_only_leak,
    mix_log,
)
from probe_runtime.metadata_twins import metadata_projection, metadata_twin_pair


def _case(index: int) -> dict:
    return {
        "amount_minor": 100 + index,
        "currency": "USD",
        "merchant_ref": f"merchant-{index}",
        "reason": f"reason-{index}",
        "authorized": True,
        "tool_name": "refund.create",
    }


def _mix(n: int = 8, rate: float = 0.25, seed: int = 11):
    return mix_log([_case(i) for i in range(n)], rate=rate, seed=seed)


def test_each_canary_is_the_generator_bad_row_of_a_good_twin():
    mixed = _mix()
    assert mixed.blocks_refund is False
    assert mixed.money_moved is False
    assert mixed.key.known_bad is True
    assert mixed.key.fake_money is True
    assert "canary" not in json.dumps(mixed.records).lower()
    assert len(mixed.key.positions) == len(mixed.key.twin_positions) == 2
    for canary_at, twin_at in zip(mixed.key.positions, mixed.key.twin_positions):
        canary = mixed.records[canary_at]
        twin = mixed.records[twin_at]
        assert metadata_projection(canary) == metadata_projection(twin)
        assert canary["ground_truth"] == "bad"
        assert twin["ground_truth"] == "good"
        pair_id = canary["case_id"].removesuffix("-bad")
        rebuilt_bad, _rebuilt_good = metadata_twin_pair(
            pair_id,
            amount_minor=twin["amount_minor"],
            currency=twin["currency"],
            authorized=twin["authorized"],
            tool_name=twin["tool_name"],
            merchant_ref=twin["merchant_ref"],
        )
        assert canary == rebuilt_bad


def test_twin_mix_is_not_leaking_and_recall_is_not_a_detection_claim():
    mixed = _mix()
    caught = [index in mixed.key.positions for index in range(mixed.key.n)]
    report = canary_recall(caught, mixed)
    assert report.status == "recall"
    assert report.recall == 1.0
    assert report.canary_deny_count == report.twin_deny_count
    assert report.synthetic_recall_is_not_detection_performance is True
    assert report.blocks_refund is False
    missed = canary_recall([False] * mixed.key.n, mixed)
    assert missed.recall == 0.0
    assert missed.status == "recall"


def test_a_details_rule_that_separates_canaries_emits_no_recall():
    good = metadata_twin_pair("synthetic-good", amount_minor=100)[1]
    canary = metadata_twin_pair("synthetic-other", amount_minor=50000)[0]
    status, rule_id, canary_denies, twin_denies = details_only_leak(
        (canary, good),
        (0,),
        (1,),
    )
    assert status == "LEAKING"
    assert rule_id
    assert canary_denies > twin_denies
    mixed = _mix()
    leaking = type(mixed)(
        records=(canary, good),
        leak_cases=(canary, good),
        key=type(mixed.key)(
            seed=1,
            rate=0.5,
            n=2,
            positions=(0,),
            twin_positions=(1,),
        ),
    )
    report = canary_recall([True, False], leaking)
    assert report.status == "LEAKING"
    assert report.recall is None


def test_same_seed_repeats():
    assert _mix(seed=11).key.positions == _mix(seed=11).key.positions
    assert canary_count_for(8, 0.25) == 2


def test_no_canaries_makes_recall_inconclusive():
    mixed = mix_log([_case(0)], rate=0.2, seed=1)
    assert mixed.key.positions == ()
    report = canary_recall([False], mixed)
    assert report.status == "INCONCLUSIVE"
    assert report.recall is None


def test_log_only_flags_cannot_be_turned_into_a_block():
    mixed = _mix()
    with pytest.raises(ValueError, match="log-only"):
        type(mixed)(records=mixed.records, leak_cases=mixed.leak_cases, key=mixed.key, blocks_refund=True)
