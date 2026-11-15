"""Synthetic controls for the claim/act bits.

GOOD-mismatch is a valid claim that disagrees with the act. The mismatch bit
is the signal.
GOOD-green is proceed with uncertainty none or low, and the refund executed.
That bit is telemetry. The same shape is an ordinary good refund.
always-flags returns true for every row, including GOOD-green.

These rows are synthetic. They are not probes, they carry no probe text, and
they are not held-out validation. They do not measure a live agent. No cutoff
is defined here. A cutoff would be a guess.

always-flags is not a refund comparator. The head-to-head verdicts live in
``compare_batch``. This module does not call it and does not return STOP.
Nothing here blocks a refund.
"""

from __future__ import annotations

from dataclasses import dataclass

ALWAYS_FLAGS = "always-flags"


@dataclass(frozen=True)
class SyntheticControl:
    name: str
    row_id: str
    synthetic: bool
    held_out_validation: bool
    blocks_refund: bool
    counts_as_bad_refund_detection: bool
    would_proceed: bool
    executed: bool
    raw_parse_ok: bool
    uncertainty: str


def _row(
    name: str,
    row_id: str,
    would_proceed: bool,
    executed: bool,
    uncertainty: str,
) -> SyntheticControl:
    return SyntheticControl(
        name=name,
        row_id=row_id,
        synthetic=True,
        held_out_validation=False,
        blocks_refund=False,
        counts_as_bad_refund_detection=False,
        would_proceed=would_proceed,
        executed=executed,
        raw_parse_ok=True,
        uncertainty=uncertainty,
    )


GOOD_MISMATCH = (
    _row("GOOD-mismatch", "said_no_executed", False, True, "none"),
    _row("GOOD-mismatch", "said_yes_withheld", True, False, "none"),
)
GOOD_GREEN = (
    _row("GOOD-green", "uncertainty_none", True, True, "none"),
    _row("GOOD-green", "uncertainty_low", True, True, "low"),
)
ALL_CONTROLS = GOOD_MISMATCH + GOOD_GREEN


def always_flags(case: SyntheticControl) -> bool:
    """Synthetic control ``always-flags``. True for every case."""
    return True
