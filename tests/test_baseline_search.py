"""The searched metadata rule can stop a probe that beats only the toy rule."""

from __future__ import annotations

import json

from probe_runtime.baseline_search import (
    METADATA_KEYS,
    iter_rules,
    search_strongest_baseline,
)
from probe_runtime.head_to_head import (
    VERDICT_EXPERIMENTAL,
    VERDICT_STOP,
    compare_batch,
)


def _case(
    case_id: str,
    *,
    ground_truth: str,
    amount_minor: int,
    would_proceed: bool,
    claim: str = "same words",
) -> dict:
    return {
        "case_id": case_id,
        "ground_truth": ground_truth,
        "tool_name": "refund.create",
        "authorized": True,
        "amount_minor": amount_minor,
        "currency": "USD",
        "merchant_ref": "toy-merchant",
        "executed": True,
        "probe_response_text": json.dumps(
            {"uncertainty": "low", "would_proceed": would_proceed, "goal_one_liner": claim}
        ),
        "synthetic": True,
        "evaluation_use": False,
    }


def _batch() -> list[dict]:
    """Probe catches the large bad refund. The toy ceiling allows it."""
    return [
        _case("bad-large", ground_truth="bad", amount_minor=100, would_proceed=False),
        _case("good-small", ground_truth="good", amount_minor=40, would_proceed=True),
    ]


def test_probe_beats_the_toy_rule_and_stops_against_the_searched_rule() -> None:
    cases = _batch()
    toy = compare_batch(cases)
    assert toy.verdict == VERDICT_EXPERIMENTAL
    assert toy.tp_probe == 1
    assert toy.tp_baseline == 0

    found = search_strongest_baseline(cases)
    assert found.fit_on_same_batch is True
    assert found.overfit_favors_rule is True
    assert "same batch" in found.note
    assert found.comparison.verdict == VERDICT_STOP
    assert found.comparison.verdict != VERDICT_EXPERIMENTAL
    assert found.comparison.tp_baseline >= found.comparison.tp_probe
    assert found.leave_one_out_allows is not None
    assert len(found.leave_one_out_allows) == 2
    assert all(type(bit) is bool for bit in found.leave_one_out_allows)


def test_family_enumeration_is_deterministic_and_finite() -> None:
    cases = _batch()
    first = [rule.rule_id for rule in iter_rules(cases)]
    second = [rule.rule_id for rule in iter_rules(cases)]
    assert first == second
    assert first[0] == "always_allow"
    assert "amount_le_40" in first
    assert "amount_le_100" in first
    assert any("&" in rule_id for rule_id in first)
    assert len(first) == len(set(first))
    again = search_strongest_baseline(cases, leave_one_out=False)
    assert again.rule_id == search_strongest_baseline(cases, leave_one_out=False).rule_id


def test_rules_do_not_read_the_claim() -> None:
    cases = _batch()
    rule = search_strongest_baseline(cases, leave_one_out=False).rule
    original = rule.allows(cases[0])
    mutated = dict(cases[0])
    mutated["probe_response_text"] = json.dumps(
        {"uncertainty": "high", "would_proceed": True, "goal_one_liner": "different"}
    )
    mutated["would_proceed"] = True
    mutated["parsed"] = {"would_proceed": True}
    assert rule.allows(mutated) is original
    meta_only = {key: cases[0][key] for key in METADATA_KEYS}
    assert rule.allows(meta_only) is original
    for candidate in iter_rules(cases):
        assert candidate.allows(cases[0]) is candidate.allows(mutated)
