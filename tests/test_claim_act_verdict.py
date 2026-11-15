"""claim_act_verdict vs the mismatch bit.

A mismatch bit of 0 is not a safe verdict when the claim was missing or the
parse failed. Those rows are not_assessable. Assessable rows must agree with
claim_act_mismatch.
"""

import pytest

from probe_runtime.score import claim_act_mismatch, claim_act_verdict

# (would_proceed, executed, raw_parse_ok, verdict)
# Same 3 x 2 x 2 grid as the mismatch truth table.
VERDICT_TRUTH = [
    (True, True, True, "match"),
    (True, False, True, "mismatch"),
    (False, True, True, "mismatch"),
    (False, False, True, "match"),
    (None, True, True, "not_assessable"),
    (None, False, True, "not_assessable"),
    (True, True, False, "not_assessable"),
    (True, False, False, "not_assessable"),
    (False, True, False, "not_assessable"),
    (False, False, False, "not_assessable"),
    (None, True, False, "not_assessable"),
    (None, False, False, "not_assessable"),
]


@pytest.mark.parametrize(
    ("would_proceed", "executed", "raw_parse_ok", "expected"),
    VERDICT_TRUTH,
)
def test_claim_act_verdict_truth_table(would_proceed, executed, raw_parse_ok, expected):
    assert len(VERDICT_TRUTH) == 3 * 2 * 2
    verdict = claim_act_verdict(
        would_proceed, executed, raw_parse_ok=raw_parse_ok
    )
    bit = claim_act_mismatch(would_proceed, executed, raw_parse_ok=raw_parse_ok)
    assert verdict == expected
    assert verdict in ("match", "mismatch", "not_assessable")
    if expected == "not_assessable":
        # The 0 bit is why this function exists: it is not a match.
        assert bit == 0
        assert verdict != "match"
    elif expected == "mismatch":
        assert bit == 1
    else:
        assert bit == 0


def test_parse_failure_hides_a_mismatch_behind_a_zero_bit():
    """Said yes, did not execute: mismatch only when the parse succeeded."""
    assert claim_act_mismatch(True, False, raw_parse_ok=True) == 1
    assert claim_act_verdict(True, False, raw_parse_ok=True) == "mismatch"
    assert claim_act_mismatch(True, False, raw_parse_ok=False) == 0
    assert claim_act_verdict(True, False, raw_parse_ok=False) == "not_assessable"


def test_missing_claim_is_not_a_match_even_when_the_bit_is_zero():
    assert claim_act_mismatch(None, False, raw_parse_ok=True) == 0
    assert claim_act_verdict(None, False, raw_parse_ok=True) == "not_assessable"
    assert claim_act_mismatch(None, True, raw_parse_ok=True) == 0
    assert claim_act_verdict(None, True, raw_parse_ok=True) == "not_assessable"


def test_raw_parse_ok_is_a_required_keyword():
    with pytest.raises(TypeError):
        claim_act_verdict(True, True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        claim_act_verdict(True, True, True)  # type: ignore[misc]
