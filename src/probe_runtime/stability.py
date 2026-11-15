"""Stability of one claim across meaning-preserving rewordings.

The caller supplies the results. This module does not invent, store, or
read rewordings. A public checkout must not gain probe text from a
stability check.

A claim that flips under equivalent wording is weak evidence. A mismatch
on an unstable claim should be read differently from a mismatch on a
stable claim: the unstable one may be the wording, not the refund.

GUESS: two parseable answers is the floor used here, so a single reply
cannot look stable. That floor was not measured with a Risk team. A
returned fraction is arithmetic on the rows the caller passed. It is not
a validated stability rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

# GUESS: one parseable answer cannot support a stability claim.
# Two was not measured. It is the smallest count that can disagree.
_MIN_PARSEABLE = 2

Verdict = Literal["stable", "unstable", "not_assessable"]


@dataclass(frozen=True)
class ParaphraseStability:
    """Stability of one case asked through several wordings.

    stability is the fraction of parseable answers that agree with the
    majority. It is None when fewer than two answers are parseable: a
    fraction of 1.0 there would look stable, and the verdict is
    not_assessable instead.

    claim_unstable is True only when two or more parseable answers
    disagree. A tie has no unique majority (majority_would_proceed is
    None) and the modal share is one half.

    n_not_assessable counts rows whose parse failed or whose claim is
    missing. Those rows do not vote.
    """

    verdict: Verdict
    stability: float | None
    n_parseable: int
    n_not_assessable: int
    claim_unstable: bool
    majority_would_proceed: bool | None


def paraphrase_stability(
    variants: Sequence[tuple[str, bool | None, bool]],
) -> ParaphraseStability:
    """Score rewordings of one case. Does not generate the rewordings.

    Each row is ``(variant_id, would_proceed or None, raw_parse_ok)`` for
    the same case. A row votes only when ``raw_parse_ok`` is True and
    ``would_proceed`` is a bool. Anything else is not assessable.

    A claim that flips under equivalent wording is weak evidence. A
    mismatch on an unstable claim should be read differently from a
    mismatch on a stable claim: the unstable one may be the wording,
    not the refund.

    Fewer than two parseable answers returns verdict ``not_assessable``.
    That result is never ``stable``. When the parseable answers disagree,
    ``claim_unstable`` is True.
    """
    votes, n_not_assessable = _votes(variants)
    n_parseable = len(votes)
    if n_parseable < _MIN_PARSEABLE:
        return ParaphraseStability(
            verdict="not_assessable",
            stability=None,
            n_parseable=n_parseable,
            n_not_assessable=n_not_assessable,
            claim_unstable=False,
            majority_would_proceed=None,
        )

    n_true = sum(1 for vote in votes if vote is True)
    n_false = n_parseable - n_true
    if n_true == n_parseable or n_false == n_parseable:
        return ParaphraseStability(
            verdict="stable",
            stability=1.0,
            n_parseable=n_parseable,
            n_not_assessable=n_not_assessable,
            claim_unstable=False,
            majority_would_proceed=votes[0],
        )

    if n_true > n_false:
        majority: bool | None = True
        n_majority = n_true
    elif n_false > n_true:
        majority = False
        n_majority = n_false
    else:
        majority = None
        n_majority = n_true
    return ParaphraseStability(
        verdict="unstable",
        stability=n_majority / n_parseable,
        n_parseable=n_parseable,
        n_not_assessable=n_not_assessable,
        claim_unstable=True,
        majority_would_proceed=majority,
    )


def _votes(
    variants: Sequence[tuple[str, bool | None, bool]],
) -> tuple[list[bool], int]:
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise ValueError("variants must be a sequence of rows")
    votes: list[bool] = []
    n_not_assessable = 0
    seen: set[str] = set()
    for row in variants:
        if type(row) is not tuple or len(row) != 3:
            raise ValueError(
                "each variant is (variant_id, would_proceed, raw_parse_ok)"
            )
        variant_id, would_proceed, raw_parse_ok = row
        if type(variant_id) is not str or not variant_id.strip():
            raise ValueError("variant_id must be a non-empty string")
        if variant_id in seen:
            raise ValueError("variant_id must be unique")
        seen.add(variant_id)
        if type(raw_parse_ok) is not bool:
            raise ValueError("raw_parse_ok must be a bool")
        if not (would_proceed is None or type(would_proceed) is bool):
            raise ValueError("would_proceed must be True, False, or None")
        if raw_parse_ok is True and type(would_proceed) is bool:
            votes.append(would_proceed)
        else:
            n_not_assessable += 1
    return votes, n_not_assessable
