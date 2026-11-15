"""Wald sequential probability ratio test for paired case outcomes.

Library helper. It is not wired into the head-to-head comparison, and
it does not score a refund. The caller maps each paired case to a flag
and chooses the four parameters. There are no defaults.

H0 says the flag probability is p0. H1 says it is p1. alpha is the
bound used as P(accept H1 | H0). beta is the bound used as P(accept H0
| H1). A returned decision is the Wald boundary on those numbers.

GUESS: p0, p1, alpha, and beta are not measured false-alarm or miss
rates. accept_h1 is not evidence that probes discriminate, and it is
not a Risk decision.

Not-assessable outcomes (None) are skipped. They do not move the
log-likelihood ratio, and they are counted. The walk stops at the first
crossing, so later outcomes are not consumed.

Bounds, in natural log:
  upper = ln((1 - beta) / alpha)   accept H1 when the ratio >= upper
  lower = ln(beta / (1 - alpha))   accept H0 when the ratio <= lower
A flagged case adds ln(p1 / p0). An unflagged case adds
ln((1 - p1) / (1 - p0)). The standard-library math module is enough.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Decision = Literal["accept_h1", "accept_h0", "continue"]


@dataclass(frozen=True)
class SequentialStop:
    """Wald decision for the outcomes consumed so far.

    log_likelihood_ratio is the sum at the stopping outcome, or the
    current sum when the decision is continue. n_not_assessable counts
    skipped None outcomes among the outcomes that were consumed.
    upper_bound and lower_bound are the Wald thresholds for the
    caller's parameters.
    """

    decision: Decision
    log_likelihood_ratio: float
    n_assessable: int
    n_not_assessable: int
    upper_bound: float
    lower_bound: float


def sequential_stop(
    outcomes: Iterable[bool | None],
    *,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
) -> SequentialStop:
    """Stop when the paired flags cross a Wald bound. No default rates.

    ``p0``, ``p1``, ``alpha``, and ``beta`` are required. GUESS: whichever
    numbers the caller passes were not measured here. None is
    not-assessable: skipped, and counted. True is a flag. False is not.
    """
    p0_value = _open_unit("p0", p0)
    p1_value = _open_unit("p1", p1)
    if p0_value == p1_value:
        raise ValueError("p0 and p1 must differ")
    alpha_value = _open_unit("alpha", alpha)
    beta_value = _open_unit("beta", beta)
    # alpha + beta < 1 puts the lower bound below 0 and the upper above it.
    # That is a requirement of these bounds, not a measured tolerance.
    if alpha_value + beta_value >= 1:
        raise ValueError("alpha + beta must be < 1 so the bounds have opposite signs")

    upper = math.log((1.0 - beta_value) / alpha_value)
    lower = math.log(beta_value / (1.0 - alpha_value))
    log_likelihood_ratio = 0.0
    n_assessable = 0
    n_not_assessable = 0
    for outcome in outcomes:
        if outcome is None:
            n_not_assessable += 1
            continue
        if type(outcome) is not bool:
            raise ValueError("each outcome must be True, False, or None")
        log_likelihood_ratio += _step(outcome, p0_value, p1_value)
        n_assessable += 1
        if log_likelihood_ratio >= upper:
            return _result(
                "accept_h1",
                log_likelihood_ratio,
                n_assessable,
                n_not_assessable,
                upper,
                lower,
            )
        if log_likelihood_ratio <= lower:
            return _result(
                "accept_h0",
                log_likelihood_ratio,
                n_assessable,
                n_not_assessable,
                upper,
                lower,
            )
    return _result(
        "continue",
        log_likelihood_ratio,
        n_assessable,
        n_not_assessable,
        upper,
        lower,
    )


def _result(
    decision: Decision,
    log_likelihood_ratio: float,
    n_assessable: int,
    n_not_assessable: int,
    upper: float,
    lower: float,
) -> SequentialStop:
    return SequentialStop(
        decision=decision,
        log_likelihood_ratio=log_likelihood_ratio,
        n_assessable=n_assessable,
        n_not_assessable=n_not_assessable,
        upper_bound=upper,
        lower_bound=lower,
    )


def _step(flagged: bool, p0: float, p1: float) -> float:
    if flagged:
        return math.log(p1 / p0)
    return math.log((1.0 - p1) / (1.0 - p0))


def _open_unit(name: str, value: object) -> float:
    if (
        type(value) is bool
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be a real number in (0, 1)")
    return float(value)
