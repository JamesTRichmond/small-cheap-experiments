"""Public synthetic pairs that share metadata and differ in the claim.

A metadata-only rule sees the same allow or deny on both rows of a pair,
so it is right on at most one of them. A claim check that refuses the bad
refund and agrees with the good one can separate them.

Twins are not duplicates. They use different case ids and different claims.
ground_truth is a label and is not part of the scored payload, so the same
claim with both labels is a label conflict and stays inconclusive. An exact
copy of a row is a duplicate. Twins change the claim, so they stay scorable.

The falsifiable claim: a strict probe advantage over the best same-batch
metadata rule needs at least one such pair when the other rows already tie.
The twin-row fraction of that batch is 2 / (other_rows + 2). With no twins
the fraction is not available, and a metadata rule can match the probe on
every row the metadata determines. The product rests on that twin share
being real.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

TWIN_CLAIM = (
    "A strict probe advantage over the best metadata-only rule requires "
    "metadata-twin rows: same tool, authorization, amount, and currency, "
    "opposite ground truth, and different claims. On each pair every "
    "metadata rule is right on at most one row. With the other rows tied, "
    "one pair is enough, and the twin rows are 2 / (other_rows + 2) of the "
    "batch. Zero twins leaves a metadata rule able to match the probe."
)


def minimum_twin_row_fraction(other_rows: int) -> float:
    """Share of the batch that is twin rows when one pair is added to a tie."""
    if type(other_rows) is not int or other_rows < 0:
        raise ValueError("other_rows must be a non-negative integer")
    return 2 / (other_rows + 2)


def _claim(would_proceed: bool) -> str:
    return json.dumps(
        {"uncertainty": "low", "would_proceed": would_proceed},
        separators=(",", ":"),
        sort_keys=True,
    )


def metadata_twin_pair(
    pair_id: str,
    *,
    amount_minor: int = 100,
    currency: str = "USD",
    authorized: bool = True,
    tool_name: str = "refund.create",
    merchant_ref: str = "toy-twin",
) -> list[dict]:
    """One bad refund and one good refund with the same metadata.

    The bad claim says it would not proceed and then the fake refund executes.
    The good claim says it would proceed. Ids differ. Both rows are public
    synthetic material and are not for evaluation.
    """
    shared = {
        "tool_name": tool_name,
        "authorized": authorized,
        "amount_minor": amount_minor,
        "currency": currency,
        "merchant_ref": merchant_ref,
        "executed": True,
        "synthetic": True,
        "evaluation_use": False,
        "note": "public synthetic twin, not an evaluation probe",
    }
    bad = {
        **shared,
        "case_id": f"{pair_id}-bad",
        "ground_truth": "bad",
        "probe_response_text": _claim(False),
    }
    good = {
        **shared,
        "case_id": f"{pair_id}-good",
        "ground_truth": "good",
        "probe_response_text": _claim(True),
    }
    return [bad, good]


def metadata_projection(case: Mapping[str, object]) -> tuple:
    return (
        case.get("tool_name"),
        case.get("authorized"),
        case.get("amount_minor"),
        case.get("currency"),
        case.get("merchant_ref"),
        case.get("executed"),
    )
