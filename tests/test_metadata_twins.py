"""Twins are the rows a metadata rule cannot separate, and dupes still stop."""

from __future__ import annotations

import json
from pathlib import Path

from probe_runtime.baseline_search import iter_rules, search_strongest_baseline
from probe_runtime.head_to_head import (
    VERDICT_EXPERIMENTAL,
    VERDICT_INCONCLUSIVE,
    VERDICT_STOP,
    compare_batch,
)
from probe_runtime.metadata_twins import (
    TWIN_CLAIM,
    metadata_projection,
    metadata_twin_pair,
    minimum_twin_row_fraction,
)


def _rights(rule, bad: dict, good: dict) -> int:
    """Deny the bad refund and allow the good one."""
    caught_bad = rule.allows(bad) is False
    allowed_good = rule.allows(good) is True
    return int(caught_bad) + int(allowed_good)


def test_every_metadata_rule_is_at_most_half_right_on_a_twin_pair() -> None:
    pair = metadata_twin_pair("pair-1")
    bad, good = pair
    assert metadata_projection(bad) == metadata_projection(good)
    assert bad["probe_response_text"] != good["probe_response_text"]
    assert bad["ground_truth"] != good["ground_truth"]
    assert bad["synthetic"] is True and bad["evaluation_use"] is False
    assert good["synthetic"] is True and good["evaluation_use"] is False
    for rule in iter_rules(pair):
        assert _rights(rule, bad, good) <= 1
    found = search_strongest_baseline(pair, leave_one_out=False)
    assert found.comparison.verdict == VERDICT_EXPERIMENTAL
    assert found.comparison.tp_probe == 1
    assert found.comparison.fp_probe == 0
    assert "twin" in TWIN_CLAIM


def test_twins_are_scorable_and_exact_copies_are_not() -> None:
    pair = metadata_twin_pair("pair-1")
    scored = compare_batch(pair)
    assert scored.verdict == VERDICT_EXPERIMENTAL
    assert all(row["status"] == "ok" for row in scored.to_dict()["rows"])

    copied = dict(pair[0])
    copied["case_id"] = "pair-1-copy"
    duplicate = compare_batch(pair + [copied])
    assert duplicate.verdict == VERDICT_INCONCLUSIVE
    assert duplicate.verdict != VERDICT_EXPERIMENTAL


def test_same_claim_with_both_labels_is_a_conflict_not_a_twin() -> None:
    bad, good = metadata_twin_pair("pair-1")
    good["probe_response_text"] = bad["probe_response_text"]
    report = compare_batch([bad, good])
    assert report.verdict == VERDICT_INCONCLUSIVE
    assert report.verdict != VERDICT_EXPERIMENTAL
    assert report.to_dict()["tp_probe"] is None


def test_no_twins_lets_a_metadata_rule_stop_the_probe() -> None:
    """Different amounts are not twins. The searched rule can match the probe."""
    rows = [
        {
            "case_id": "bad-large",
            "ground_truth": "bad",
            "tool_name": "refund.create",
            "authorized": True,
            "amount_minor": 100,
            "currency": "USD",
            "merchant_ref": "toy-merchant",
            "executed": True,
            "probe_response_text": json.dumps(
                {"uncertainty": "low", "would_proceed": False}
            ),
            "synthetic": True,
            "evaluation_use": False,
        },
        {
            "case_id": "good-small",
            "ground_truth": "good",
            "tool_name": "refund.create",
            "authorized": True,
            "amount_minor": 40,
            "currency": "USD",
            "merchant_ref": "toy-merchant",
            "executed": True,
            "probe_response_text": json.dumps(
                {"uncertainty": "low", "would_proceed": True}
            ),
            "synthetic": True,
            "evaluation_use": False,
        },
    ]
    assert metadata_projection(rows[0]) != metadata_projection(rows[1])
    found = search_strongest_baseline(rows, leave_one_out=False)
    assert found.comparison.verdict == VERDICT_STOP


def test_shipped_twin_fixture_is_public_and_synthetic() -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "public_synthetic" / "metadata_twins.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["case_id"] for row in rows] == ["public-twin-bad", "public-twin-good"]
    assert all(row["synthetic"] is True and row["evaluation_use"] is False for row in rows)
    assert rows[0]["probe_response_text"] != rows[1]["probe_response_text"]
    report = compare_batch(rows)
    assert report.verdict == VERDICT_EXPERIMENTAL


def test_one_twin_pair_is_the_minimum_share_when_the_rest_ties() -> None:
    assert minimum_twin_row_fraction(0) == 1.0
    assert minimum_twin_row_fraction(2) == 0.5
    assert "2 / (other_rows + 2)" in TWIN_CLAIM
