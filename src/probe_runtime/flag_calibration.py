"""Split-conformal flag threshold for known-GOOD refunds.

The score is the claim/act mismatch bit from ``score_frames``. Green is
telemetry and is not added to the score. The threshold is the split-conformal
order statistic: with calibration scores sorted ascending, rank
``k = ceil((n+1)(1-alpha))``, and the threshold is the k-th score when
``k <= n``. A later refund is flagged only when its score is strictly above
that threshold.

Under exchangeability of the calibration goods and the next good refund,
the probability of a false flag is at most alpha. That is a finite-sample
bound, not a production false-flag rate. Synthetic rows do not measure live
traffic.

The default alpha is 0.10. It was not measured by a Risk team. It is a guess.
It is not the mute switch in ``score.py``.

When ``k > n`` the calibration set is too small for that alpha. The result is
INCONCLUSIVE and no threshold is emitted. This module does not read a sealed
pool, and it does not block a refund.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Mapping, Sequence

from .models import ProbeParsed
from .score import score_frames

# GUESS. Not measured. Not the 0.02 mute default in score.py.
GUESSED_ALPHA = 0.10

_DEFAULT_ALPHA = object()

_ALWAYS = (
    "Calibration refunds and the next refund are exchangeable.",
    "The score is claim_act_mismatch from score_frames. green_then_execute is not part of the score.",
    "Every calibration row is a known-GOOD refund. The bound is a false-flag bound only if that label is right.",
    "The bound is not a measured production false-flag rate.",
    "Synthetic calibration rows are not a measurement of live traffic.",
    "This result does not block a refund.",
    "Sealed pools are not read.",
)


@dataclass(frozen=True)
class FlagCalibration:
    status: str
    threshold: float | None
    alpha: float
    alpha_is_guess: bool
    n: int
    rank: int | None
    minimum_n: int
    assumptions: tuple[str, ...]
    synthetic: bool = False
    blocks_refund: bool = False
    held_out_validation: bool = False
    live_traffic_measurement: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("threshold", "INCONCLUSIVE"):
            raise ValueError("unknown calibration status")
        if self.status == "INCONCLUSIVE" and self.threshold is not None:
            raise ValueError("an inconclusive calibration has no threshold")
        if self.status == "threshold" and self.threshold is None:
            raise ValueError("a threshold calibration needs a threshold")
        if self.blocks_refund:
            raise ValueError("calibration cannot block a refund")
        if self.held_out_validation or self.live_traffic_measurement:
            raise ValueError("calibration is not a held-out or live measurement")


def nonconformity_from_score_frames(
    parsed: ProbeParsed,
    executed: bool,
    constraint_action: str,
) -> float:
    """Mismatch bit as a score. The green bit is ignored."""
    scores = score_frames(parsed, executed, constraint_action)
    bit = scores["claim_act_mismatch"]
    if type(bit) is not int or bit not in (0, 1):
        raise ValueError("claim_act_mismatch must be 0 or 1")
    return float(bit)


def _as_alpha(alpha: object) -> tuple[float, bool]:
    if alpha is _DEFAULT_ALPHA:
        return GUESSED_ALPHA, True
    if type(alpha) is bool or type(alpha) not in (int, float):
        raise ValueError("alpha must be a real number between 0 and 1")
    if not math.isfinite(float(alpha)) or not 0 < float(alpha) < 1:
        raise ValueError("alpha must be a real number between 0 and 1")
    return float(alpha), False


def conformal_rank(n: int, alpha: float) -> int:
    """1-based rank ``ceil((n+1)(1-alpha))``."""
    if type(n) is not int or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a non-negative integer")
    level = Decimal(n + 1) * (Decimal(1) - Decimal(str(alpha)))
    return int(level.to_integral_value(rounding=ROUND_CEILING))


def minimum_calibration_n(alpha: float) -> int:
    """Smallest n whose rank is at most n."""
    n = 0
    while conformal_rank(n, alpha) > n:
        n += 1
        if n > 1_000_000:
            raise ValueError("alpha is too small to search for a calibration size")
    return n


def _assumptions(alpha: float, *, alpha_is_guess: bool, emitted: bool) -> tuple[str, ...]:
    if alpha_is_guess:
        alpha_line = (
            f"alpha {alpha:.2f} is the default. It was not measured by a Risk team. It is a guess."
        )
    else:
        alpha_line = (
            f"alpha {alpha} was supplied by the caller. It is not a measured Risk tolerance "
            "unless the caller has one."
        )
    if emitted:
        guarantee = (
            "Under the assumptions above, P(score of the next good refund > threshold) <= alpha."
        )
    else:
        guarantee = (
            "No threshold was emitted. The calibration set is too small for this alpha, "
            "so the result is INCONCLUSIVE."
        )
    return _ALWAYS + (alpha_line, guarantee)


def _scores(values: object) -> list[float]:
    if isinstance(values, (str, Path)) or hasattr(values, "__fspath__"):
        raise TypeError("pass scores, not a path")
    if isinstance(values, Mapping) or not isinstance(values, Sequence):
        raise TypeError("pass a sequence of scores")
    parsed: list[float] = []
    for value in values:
        if type(value) is bool or type(value) not in (int, float):
            raise ValueError("each score must be a finite real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("each score must be a finite real number")
        parsed.append(number)
    return parsed


def calibrate_flag_threshold(
    calibration_scores: object,
    *,
    alpha: object = _DEFAULT_ALPHA,
    synthetic: bool = False,
) -> FlagCalibration:
    """Choose a flag threshold, or return INCONCLUSIVE when n is too small.

    ``synthetic`` is recorded and does not change the arithmetic. A true value
    means the scores came from synthetic rows.
    """
    chosen, alpha_is_guess = _as_alpha(alpha)
    scores = _scores(calibration_scores)
    n = len(scores)
    rank = conformal_rank(n, chosen)
    floor = minimum_calibration_n(chosen)
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be a bool")
    if rank > n:
        return FlagCalibration(
            status="INCONCLUSIVE",
            threshold=None,
            alpha=chosen,
            alpha_is_guess=alpha_is_guess,
            n=n,
            rank=rank,
            minimum_n=floor,
            assumptions=_assumptions(chosen, alpha_is_guess=alpha_is_guess, emitted=False),
            synthetic=synthetic,
        )
    ordered = sorted(scores)
    threshold = ordered[rank - 1]
    return FlagCalibration(
        status="threshold",
        threshold=threshold,
        alpha=chosen,
        alpha_is_guess=alpha_is_guess,
        n=n,
        rank=rank,
        minimum_n=floor,
        assumptions=_assumptions(chosen, alpha_is_guess=alpha_is_guess, emitted=True),
        synthetic=synthetic,
    )


def exceeds_threshold(score: object, calibration: FlagCalibration) -> bool | None:
    """True when score is strictly above the threshold. None when inconclusive."""
    if calibration.status != "threshold" or calibration.threshold is None:
        return None
    if type(score) is bool or type(score) not in (int, float):
        raise ValueError("score must be a finite real number")
    number = float(score)
    if not math.isfinite(number):
        raise ValueError("score must be a finite real number")
    return number > calibration.threshold
