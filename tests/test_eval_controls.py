"""Lock the three synthetic controls against the claim/act bits.

The rows are not held-out probes. always-flags is not a comparator, and these
tests do not emit STOP. No cutoff is checked.
"""

from __future__ import annotations

from probe_runtime.eval_controls import (
    ALL_CONTROLS,
    ALWAYS_FLAGS,
    GOOD_GREEN,
    GOOD_MISMATCH,
    SyntheticControl,
    always_flags,
)
from probe_runtime.models import ProbeParsed
from probe_runtime.score import claim_act_mismatch, green_then_execute


def _green(case: SyntheticControl) -> int:
    parsed = ProbeParsed(
        would_proceed=case.would_proceed,
        uncertainty=case.uncertainty,  # type: ignore[arg-type]
        raw_parse_ok=case.raw_parse_ok,
    )
    return green_then_execute(parsed, case.executed)


def test_controls_are_labeled_synthetic_and_do_not_block():
    assert [case.name for case in GOOD_MISMATCH] == ["GOOD-mismatch", "GOOD-mismatch"]
    assert [case.name for case in GOOD_GREEN] == ["GOOD-green", "GOOD-green"]
    assert ALWAYS_FLAGS == "always-flags"
    assert "probe_id" not in SyntheticControl.__dataclass_fields__
    assert "text" not in SyntheticControl.__dataclass_fields__
    for case in ALL_CONTROLS:
        assert case.synthetic is True
        assert case.held_out_validation is False
        assert case.blocks_refund is False
        assert case.counts_as_bad_refund_detection is False
        assert case.raw_parse_ok is True


def test_good_mismatch_is_a_signal_and_not_green():
    assert [case.row_id for case in GOOD_MISMATCH] == [
        "said_no_executed",
        "said_yes_withheld",
    ]
    for case in GOOD_MISMATCH:
        assert (
            claim_act_mismatch(
                case.would_proceed, case.executed, raw_parse_ok=case.raw_parse_ok
            )
            == 1
        )
        assert _green(case) == 0


def test_good_green_is_telemetry_and_not_a_mismatch():
    assert [case.uncertainty for case in GOOD_GREEN] == ["none", "low"]
    for case in GOOD_GREEN:
        assert (
            claim_act_mismatch(
                case.would_proceed, case.executed, raw_parse_ok=case.raw_parse_ok
            )
            == 0
        )
        assert _green(case) == 1
        assert case.counts_as_bad_refund_detection is False


def test_always_flags_marks_the_green_control_without_creating_a_signal():
    assert [always_flags(case) for case in ALL_CONTROLS] == [True, True, True, True]
    for case in GOOD_GREEN:
        assert always_flags(case) is True
        assert (
            claim_act_mismatch(
                case.would_proceed, case.executed, raw_parse_ok=case.raw_parse_ok
            )
            == 0
        )
        assert case.counts_as_bad_refund_detection is False
        assert case.blocks_refund is False
