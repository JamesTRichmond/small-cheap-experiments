"""Wald boundaries with hand-computed values. Fixture guesses only.

These numbers are not a false-alarm tolerance and not a Risk decision.
The helper is not the head-to-head. Arithmetic that decides each
boundary is written next to the assertion.

No default exists for p0, p1, alpha, or beta. GUESS: every rate below
was chosen so the boundary is easy to check by hand.
"""

import inspect
import math
from pathlib import Path

import pytest

from probe_runtime.sequential import sequential_stop

# One-step fixture. Every value below is a dyadic rational, so the
# ratios used in the bounds are exact in binary floating point.
# p0 = 1/4, p1 = 1/2, alpha = 1/4, beta = 1/2.
# Upper = ln((1 - beta) / alpha) = ln((1/2) / (1/4)) = ln(2)
# Flag step = ln(p1 / p0) = ln((1/2) / (1/4)) = ln(2)
# One flag lands on the upper bound. Wald accepts H1 at >= upper.
# Lower = ln(beta / (1 - alpha)) = ln((1/2) / (3/4)) = ln(2/3)
# Unflagged step = ln((1 - p1) / (1 - p0)) = ln((1/2) / (3/4)) = ln(2/3)
# One unflagged case lands on the lower bound. Wald accepts H0 at <= lower.
_P0 = 0.25
_P1 = 0.5
_ALPHA = 0.25
_BETA = 0.5

# Two-step fixture, also dyadic.
# p0 = 1/4, p1 = 3/4, alpha = 1/8, beta = 1/8.
# Flag step = ln(3/4 / 1/4) = ln(3)
# Upper = ln((1 - 1/8) / (1/8)) = ln(7/8 / 1/8) = ln(7)
# Unflagged step = ln((1 - 3/4) / (1 - 1/4)) = ln((1/4) / (3/4)) = ln(1/3)
# Lower = ln((1/8) / (7/8)) = ln(1/7)
_FAR_P0 = 0.25
_FAR_P1 = 0.75
_FAR_ALPHA = 0.125
_FAR_BETA = 0.125


def _stop(outcomes, **kwargs):
    params = {"p0": _P0, "p1": _P1, "alpha": _ALPHA, "beta": _BETA}
    params.update(kwargs)
    return sequential_stop(outcomes, **params)


def _far(outcomes):
    return sequential_stop(
        outcomes,
        p0=_FAR_P0,
        p1=_FAR_P1,
        alpha=_FAR_ALPHA,
        beta=_FAR_BETA,
    )


def test_parameters_are_required_and_have_no_defaults():
    signature = inspect.signature(sequential_stop)
    for name in ("p0", "p1", "alpha", "beta"):
        parameter = signature.parameters[name]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        sequential_stop([])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sequential_stop([], _P0, _P1, _ALPHA, _BETA)  # type: ignore[misc]


def test_one_flag_lands_on_the_upper_bound():
    # ln(2) >= ln(2), so the first flag accepts H1.
    assert (1 - _BETA) / _ALPHA == _P1 / _P0 == 2
    result = _stop([True])
    assert result.decision == "accept_h1"
    assert result.log_likelihood_ratio == result.upper_bound
    assert result.log_likelihood_ratio == math.log(2)
    assert result.n_assessable == 1
    assert result.n_not_assessable == 0
    assert result.lower_bound < 0 < result.upper_bound


def test_one_unflagged_case_lands_on_the_lower_bound():
    # ln(2/3) <= ln(2/3), so the first unflagged case accepts H0.
    assert _BETA / (1 - _ALPHA) == (1 - _P1) / (1 - _P0)
    result = _stop([False])
    assert result.decision == "accept_h0"
    assert result.log_likelihood_ratio == result.lower_bound
    assert result.n_assessable == 1
    assert result.n_not_assessable == 0


def test_empty_and_not_assessable_streams_continue_at_zero():
    # No assessable case has been seen, so the ratio stays ln(1) = 0,
    # strictly between the bounds.
    empty = _stop([])
    assert empty.decision == "continue"
    assert empty.log_likelihood_ratio == 0.0
    assert empty.n_assessable == 0
    assert empty.n_not_assessable == 0
    assert empty.lower_bound < empty.log_likelihood_ratio < empty.upper_bound

    skipped = _stop([None, None, None])
    assert skipped.decision == "continue"
    assert skipped.log_likelihood_ratio == 0.0
    assert skipped.n_assessable == 0
    assert skipped.n_not_assessable == 3


def test_not_assessable_cases_are_skipped_and_counted():
    flagged = _stop([None, True])
    assert flagged.decision == "accept_h1"
    assert flagged.n_assessable == 1
    assert flagged.n_not_assessable == 1
    assert flagged.log_likelihood_ratio == flagged.upper_bound

    unflagged = _stop([None, False, None])
    # The walk stops on the unflagged case, before the trailing None.
    assert unflagged.decision == "accept_h0"
    assert unflagged.n_assessable == 1
    assert unflagged.n_not_assessable == 1
    assert unflagged.log_likelihood_ratio == unflagged.lower_bound


def test_one_step_does_not_cross_when_three_is_below_seven():
    # Flag step = ln(3). Upper = ln(7). 3 < 7, so ln(3) < ln(7).
    assert 3 < 7
    one = _far([True])
    assert one.decision == "continue"
    assert one.log_likelihood_ratio == math.log(3)
    assert one.upper_bound == math.log(7)
    assert one.log_likelihood_ratio < one.upper_bound
    assert one.n_assessable == 1

    # Unflagged step = ln(1/3). Lower = ln(1/7).
    # 1/3 > 1/7 because 7 > 3, so the ratio has not reached the lower bound.
    assert 7 > 3
    assert (1 / 3) > (1 / 7)
    unflagged = _far([False])
    assert unflagged.decision == "continue"
    assert unflagged.log_likelihood_ratio > unflagged.lower_bound
    assert unflagged.n_assessable == 1


def test_two_steps_cross_because_nine_passes_seven():
    # Two flags: 2*ln(3) = ln(9). 9 > 7, so ln(9) > ln(7). accept_h1.
    assert 9 > 7
    flagged = _far([True, True])
    assert flagged.decision == "accept_h1"
    assert flagged.n_assessable == 2
    assert flagged.log_likelihood_ratio == 2 * math.log(3)
    assert flagged.log_likelihood_ratio >= flagged.upper_bound

    # Two unflagged cases: 2*ln(1/3) = ln(1/9).
    # 1/9 < 1/7 because 7 < 9, so the ratio is below the lower bound.
    assert 7 < 9
    assert (1 / 9) < (1 / 7)
    unflagged = _far([False, False])
    assert unflagged.decision == "accept_h0"
    assert unflagged.n_assessable == 2
    assert unflagged.log_likelihood_ratio <= unflagged.lower_bound


def test_a_flag_and_its_opposite_cancel_and_the_walk_continues():
    # ln(3) + ln(1/3) = ln(1) = 0, which sits between ln(1/7) and ln(7).
    mixed = _far([True, False])
    assert mixed.decision == "continue"
    assert mixed.log_likelihood_ratio == 0.0
    assert mixed.n_assessable == 2
    assert mixed.lower_bound < 0 < mixed.upper_bound

    swapped = _far([False, True])
    assert swapped.decision == "continue"
    assert swapped.log_likelihood_ratio == 0.0
    assert swapped.n_assessable == 2


def test_the_walk_stops_and_does_not_consume_the_tail():
    # Two flags already give ln(9) > ln(7), so the decision is accept_h1.
    # A following False would pull the sum back to ln(3), and ln(3) < ln(7),
    # which would be continue if the tail were included. It must not be read.
    assert 9 > 7
    assert 3 < 7

    def _tail():
        yield True
        yield True
        raise AssertionError("outcome after the bound was consumed")

    stopped = sequential_stop(
        _tail(),
        p0=_FAR_P0,
        p1=_FAR_P1,
        alpha=_FAR_ALPHA,
        beta=_FAR_BETA,
    )
    assert stopped.decision == "accept_h1"
    assert stopped.n_assessable == 2
    assert stopped.n_not_assessable == 0
    if_tail_were_included = stopped.log_likelihood_ratio + math.log(
        (1 - _FAR_P1) / (1 - _FAR_P0)
    )
    assert if_tail_were_included == math.log(3)
    assert if_tail_were_included < stopped.upper_bound


def test_a_skipped_case_between_flags_is_counted_and_does_not_move_the_ratio():
    with_gap = _far([True, None, True])
    without_gap = _far([True, True])
    assert with_gap.decision == without_gap.decision == "accept_h1"
    assert with_gap.log_likelihood_ratio == without_gap.log_likelihood_ratio
    assert with_gap.n_assessable == without_gap.n_assessable == 2
    assert with_gap.n_not_assessable == 1
    assert without_gap.n_not_assessable == 0


def test_a_bad_outcome_after_the_bound_is_not_read():
    # The first flag accepts H1, so the following 1 is never validated.
    stopped = _stop([True, 1])
    assert stopped.decision == "accept_h1"
    assert stopped.n_assessable == 1

    with pytest.raises(ValueError, match="True, False, or None"):
        _stop([1])
    with pytest.raises(ValueError, match="True, False, or None"):
        _stop([0])
    with pytest.raises(ValueError, match="True, False, or None"):
        _far([False, "no"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p0": 0, "p1": 0.5, "alpha": 0.25, "beta": 0.5}, "p0"),
        ({"p0": 1, "p1": 0.5, "alpha": 0.25, "beta": 0.5}, "p0"),
        ({"p0": -0.1, "p1": 0.5, "alpha": 0.25, "beta": 0.5}, "p0"),
        ({"p0": True, "p1": 0.5, "alpha": 0.25, "beta": 0.5}, "p0"),
        ({"p0": 0.25, "p1": 0.25, "alpha": 0.25, "beta": 0.5}, "p0 and p1 must differ"),
        ({"p0": 0.25, "p1": 0.5, "alpha": 0, "beta": 0.5}, "alpha"),
        ({"p0": 0.25, "p1": 0.5, "alpha": 1, "beta": 0.5}, "alpha"),
        ({"p0": 0.25, "p1": 0.5, "alpha": 0.25, "beta": 0}, "beta"),
        ({"p0": 0.25, "p1": 0.5, "alpha": 0.25, "beta": True}, "beta"),
        (
            {"p0": 0.25, "p1": 0.5, "alpha": 0.5, "beta": 0.5},
            "alpha \\+ beta must be < 1",
        ),
        ({"p0": math.nan, "p1": 0.5, "alpha": 0.25, "beta": 0.25}, "p0"),
        ({"p0": 0.25, "p1": math.inf, "alpha": 0.25, "beta": 0.25}, "p1"),
        ({"p0": "0.25", "p1": 0.5, "alpha": 0.25, "beta": 0.25}, "p0"),
    ],
)
def test_parameters_outside_the_open_unit_interval_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sequential_stop([], **kwargs)


def test_module_is_a_library_helper_and_does_not_import_numpy():
    source = Path(sequential_stop.__code__.co_filename).read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "head_to_head" not in source
    text = (sequential_stop.__doc__ or "").lower()
    assert "guess" in text
    assert "no default" in text
