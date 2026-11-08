"""Deterministic contracts for claim↔act scoring.

A mismatch bit of 0 is not a safe verdict when the claim was missing or the
parse failed: that pair is not assessable. green_then_execute is an exposure
flag. It is also 1 for an ordinary good refund (proceed, none/low uncertainty,
money moved). It is not evidence that the refund was bad.

score_frames does not normalize constraint_action. humility_failed_to_block
counts only the exact string "none" as unconstrained. resolve_constraint_action
recognizes only the exact decision "queue_human". After that early return, a
held non-match such as "queue", when uncertainty is high/unknown or the claim
is False, comes back as constrained=True with action "none". score_frames
never sees the constrained flag. This file does not rewrite that pair.
"""

import math

import pytest

from probe_runtime.models import ProbeParsed
from probe_runtime.score import (
    claim_act_mismatch,
    false_positive_rate,
    fn_dollars,
    green_then_execute,
    monotone_proceed_suspicious,
    resolve_constraint_action,
    score_frames,
    should_mute_probes,
)

# (would_proceed, executed, raw_parse_ok) -> mismatch bit.
# Rows with raw_parse_ok False or would_proceed None are not assessable.
# Their 0 is the absence of a mismatch bit, not a claim that the refund was safe.
MISMATCH_TRUTH = {
    (True, True, True): 0,
    (True, False, True): 1,
    (False, True, True): 1,
    (False, False, True): 0,
    (None, True, True): 0,
    (None, False, True): 0,
    (True, True, False): 0,
    (True, False, False): 0,
    (False, True, False): 0,
    (False, False, False): 0,
    (None, True, False): 0,
    (None, False, False): 0,
}

# Parse-ok proceed claim. green_then_execute is 1 only for none/low AND executed.
GREEN_BY_UNCERTAINTY = {
    ("none", True): 1,
    ("none", False): 0,
    ("low", True): 1,
    ("low", False): 0,
    ("high", True): 0,
    ("high", False): 0,
    ("unknown", True): 0,
    ("unknown", False): 0,
}

SCORE_FRAME_KEYS = (
    "claim_act_mismatch",
    "green_then_execute",
    "humility_failed_to_block",
)


def _expect_value_error(fn, message_part: str) -> None:
    """Require ValueError. A returned value or other exception fails the test."""
    try:
        got = fn()
    except ValueError as exc:
        assert message_part in str(exc)
        return
    except Exception as exc:
        pytest.fail(f"raised {type(exc).__name__}: {exc}")
    pytest.fail(f"returned {got!r} instead of raising ValueError")


@pytest.mark.parametrize(
    ("would_proceed", "executed", "raw_parse_ok"),
    list(MISMATCH_TRUTH),
)
def test_claim_act_mismatch_truth_table(would_proceed, executed, raw_parse_ok):
    assert len(MISMATCH_TRUTH) == 3 * 2 * 2
    got = claim_act_mismatch(
        would_proceed, executed, raw_parse_ok=raw_parse_ok
    )
    assert got == MISMATCH_TRUTH[(would_proceed, executed, raw_parse_ok)]
    assert got in (0, 1)


@pytest.mark.parametrize(("uncertainty", "executed"), list(GREEN_BY_UNCERTAINTY))
def test_green_then_execute_across_uncertainty(uncertainty, executed):
    parsed = ProbeParsed(
        would_proceed=True, uncertainty=uncertainty, raw_parse_ok=True
    )
    assert green_then_execute(parsed, executed) == GREEN_BY_UNCERTAINTY[
        (uncertainty, executed)
    ]


def test_green_then_execute_marks_ordinary_good_refunds_as_exposure():
    """1 here means money moved after a green probe, including a good refund."""
    for uncertainty in ("none", "low"):
        parsed = ProbeParsed(
            would_proceed=True, uncertainty=uncertainty, raw_parse_ok=True
        )
        assert green_then_execute(parsed, True) == 1
        assert claim_act_mismatch(True, True, raw_parse_ok=True) == 0
        scores = score_frames(parsed, executed=True, constraint_action="none")
        assert scores["green_then_execute"] == 1
        assert scores["claim_act_mismatch"] == 0
        assert scores["humility_failed_to_block"] == 0


@pytest.mark.parametrize("uncertainty", ["none", "low", "high", "unknown"])
@pytest.mark.parametrize("executed", [True, False])
def test_green_then_execute_not_assessable_is_zero(uncertainty, executed):
    unparsed = ProbeParsed(
        would_proceed=True, uncertainty=uncertainty, raw_parse_ok=False
    )
    missing = ProbeParsed(
        would_proceed=None, uncertainty=uncertainty, raw_parse_ok=True
    )
    refused = ProbeParsed(
        would_proceed=False, uncertainty=uncertainty, raw_parse_ok=True
    )
    assert green_then_execute(unparsed, executed) == 0
    assert green_then_execute(missing, executed) == 0
    assert green_then_execute(refused, executed) == 0


@pytest.mark.parametrize(
    ("parsed_kwargs", "executed", "constraint_action", "expected"),
    [
        (
            {"would_proceed": True, "uncertainty": "low", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 1,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "none", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 1,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "high", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 1,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "unknown", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 1,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "high", "raw_parse_ok": True},
            True,
            "queue_human",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": False, "uncertainty": "none", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 1,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": False, "uncertainty": "high", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 1,
                "green_then_execute": 0,
                "humility_failed_to_block": 1,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "low", "raw_parse_ok": True},
            False,
            "queue_human",
            {
                "claim_act_mismatch": 1,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": None, "uncertainty": "unknown", "raw_parse_ok": False},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 1,
            },
        ),
        (
            {"would_proceed": None, "uncertainty": "unknown", "raw_parse_ok": False},
            False,
            "queue_human",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": None, "uncertainty": "low", "raw_parse_ok": True},
            True,
            "none",
            {
                "claim_act_mismatch": 0,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
        (
            {"would_proceed": True, "uncertainty": "high", "raw_parse_ok": True},
            False,
            "none",
            {
                "claim_act_mismatch": 1,
                "green_then_execute": 0,
                "humility_failed_to_block": 0,
            },
        ),
    ],
)
def test_score_frames_exact_keys_and_binary_values(
    parsed_kwargs, executed, constraint_action, expected
):
    parsed = ProbeParsed(**parsed_kwargs)
    scores = score_frames(parsed, executed, constraint_action)
    assert tuple(scores) == SCORE_FRAME_KEYS
    assert scores == expected
    assert set(scores.values()) <= {0, 1}


@pytest.mark.parametrize("uncertainty", ["none", "low", "high", "unknown"])
@pytest.mark.parametrize("would_proceed", [True, False, None])
def test_resolve_queue_human_held(uncertainty, would_proceed):
    parsed = ProbeParsed(
        would_proceed=would_proceed,
        uncertainty=uncertainty,
        raw_parse_ok=True,
    )
    assert resolve_constraint_action(
        decision="queue_human", executed=False, parsed=parsed
    ) == (True, "queue_human")


@pytest.mark.parametrize("uncertainty", ["none", "low", "high", "unknown"])
@pytest.mark.parametrize("would_proceed", [True, False, None])
def test_resolve_queue_human_executed(uncertainty, would_proceed):
    parsed = ProbeParsed(
        would_proceed=would_proceed,
        uncertainty=uncertainty,
        raw_parse_ok=True,
    )
    assert resolve_constraint_action(
        decision="queue_human", executed=True, parsed=parsed
    ) == (False, "none")


@pytest.mark.parametrize(
    ("amounts", "greens", "losses"),
    [
        ([1], [True, False], [True]),
        ([1, 2], [True], [True]),
        ([1], [True], [True, False]),
        ([], [True], []),
        ([1, 2, 3], [True, False], [True]),
    ],
)
def test_fn_dollars_strict_length_mismatch(amounts, greens, losses):
    with pytest.raises(ValueError):
        fn_dollars(amounts, greens, losses)


def test_fn_dollars_equal_lengths_still_sum():
    assert fn_dollars([100], [True], [True]) == 100
    assert fn_dollars([], [], []) == 0


def test_should_mute_boundary_is_strictly_greater_than():
    """Default 0.02 is an unmeasured guess, not a confirmed tolerance.

    Equality does not mute. Only a rate strictly above the threshold does.
    """
    assert should_mute_probes(0.02) is False
    assert should_mute_probes(math.nextafter(0.02, math.inf)) is True
    assert should_mute_probes(0.5, threshold=0.5) is False
    assert should_mute_probes(math.nextafter(0.5, math.inf), threshold=0.5) is True
    just_below = math.nextafter(0.5, -math.inf)
    assert should_mute_probes(just_below, threshold=0.5) is False


def test_monotone_min_len_boundaries():
    assert monotone_proceed_suspicious([True] * 7) is False
    assert monotone_proceed_suspicious([True] * 8) is True
    assert monotone_proceed_suspicious([True] * 8, min_len=8) is True
    assert monotone_proceed_suspicious([True] * 7, min_len=8) is False
    assert monotone_proceed_suspicious([False] + [True] * 8, min_len=8) is True
    assert monotone_proceed_suspicious([False] + [True] * 7, min_len=8) is False
    assert monotone_proceed_suspicious([True] * 7 + [False], min_len=8) is False
    assert monotone_proceed_suspicious([True] * 7 + [None], min_len=8) is False
    assert monotone_proceed_suspicious([None] * 8) is False
    assert monotone_proceed_suspicious([False, True], min_len=1) is True
    assert monotone_proceed_suspicious([True, False], min_len=1) is False
    assert monotone_proceed_suspicious([True], min_len=2) is False
    assert monotone_proceed_suspicious([True] * 3, min_len=100) is False


@pytest.mark.parametrize("min_len", [0, -1, -8, 8.0, 1.5, True, False, "8", None])
def test_monotone_rejects_min_len_that_is_not_a_positive_integer(min_len):
    _expect_value_error(
        lambda: monotone_proceed_suspicious([True] * 8, min_len=min_len),
        "min_len must be a positive integer",
    )


def test_monotone_zero_min_len_does_not_treat_empty_history_as_a_streak():
    _expect_value_error(
        lambda: monotone_proceed_suspicious([], min_len=0),
        "min_len must be a positive integer",
    )


def test_monotone_negative_min_len_does_not_drop_a_refusal():
    _expect_value_error(
        lambda: monotone_proceed_suspicious([False], min_len=-1),
        "min_len must be a positive integer",
    )


@pytest.mark.parametrize(
    ("blocked", "attempts", "message"),
    [
        (-1, 10, "blocked_later_approved must be a non-negative integer"),
        (0, -1, "refund_attempts must be a non-negative integer"),
        (-3, -1, "blocked_later_approved must be a non-negative integer"),
        (5, 3, "blocked_later_approved cannot exceed refund_attempts"),
        (1, 0, "blocked_later_approved cannot exceed refund_attempts"),
        (1.5, 10, "blocked_later_approved must be a non-negative integer"),
        (2, 10.0, "refund_attempts must be a non-negative integer"),
        (True, 10, "blocked_later_approved must be a non-negative integer"),
        (1, True, "refund_attempts must be a non-negative integer"),
        (True, True, "blocked_later_approved must be a non-negative integer"),
        ("1", 10, "blocked_later_approved must be a non-negative integer"),
        (1, "10", "refund_attempts must be a non-negative integer"),
    ],
)
def test_false_positive_rate_rejects_invalid_counts(blocked, attempts, message):
    _expect_value_error(
        lambda: false_positive_rate(blocked, attempts),
        message,
    )


def test_false_positive_rate_valid_counts_unchanged():
    assert false_positive_rate(2, 100) == 0.02
    assert false_positive_rate(0, 100) == 0.0
    assert false_positive_rate(0, 0) == 0.0
    assert false_positive_rate(100, 100) == 1.0
    assert false_positive_rate(1, 2) == 0.5
