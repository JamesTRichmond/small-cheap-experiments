"""summarize counts claim↔act verdicts. Fake rows only.

mismatch_rate uses assessable rows. An empty batch and an all-not-assessable
batch both have mismatch_rate None, not 0. green_exposure is a separate
exposure count: a green probe followed by money moving, including a good
refund. It is not bad-refund detection.
"""

import pytest

from probe_runtime.score_summary import summarize


def _row(verdict: str, green: int = 0, **extra: object) -> dict[str, object]:
    return {"verdict": verdict, "green_then_execute": green, **extra}


def test_empty_input_rates_are_none_not_zero():
    summary = summarize([])
    assert summary.total == 0
    assert summary.match == 0
    assert summary.mismatch == 0
    assert summary.assessable == 0
    assert summary.not_assessable == 0
    assert summary.green_exposure == 0
    assert summary.mismatch_rate is None
    assert summary.not_assessable_rate is None


def test_all_not_assessable_mismatch_rate_is_none_not_zero():
    summary = summarize(
        [_row("not_assessable"), _row("not_assessable"), _row("not_assessable")]
    )
    assert summary.total == 3
    assert summary.assessable == 0
    assert summary.mismatch == 0
    assert summary.match == 0
    assert summary.not_assessable == 3
    assert summary.mismatch_rate is None
    assert summary.not_assessable_rate == 1.0
    assert summary.green_exposure == 0


def test_all_matches_are_a_zero_rate_because_rows_were_assessable():
    summary = summarize([_row("match"), _row("match")])
    assert summary.assessable == 2
    assert summary.mismatch_rate == 0.0
    assert summary.not_assessable_rate == 0.0


def test_mismatch_rate_ignores_not_assessable_rows():
    summary = summarize(
        [
            _row("mismatch"),
            _row("match"),
            _row("not_assessable"),
            _row("not_assessable"),
        ]
    )
    assert summary.assessable == 2
    assert summary.mismatch == 1
    assert summary.not_assessable == 2
    assert summary.mismatch_rate == 1 / 2
    assert summary.not_assessable_rate == 2 / 4
    # Dividing by the whole batch would hide the not-assessable rows.
    assert summary.mismatch_rate != 1 / 4


def test_green_exposure_is_not_bad_refund_detection():
    """A matched green refund counts as exposure and as a match."""
    summary = summarize(
        [
            _row("match", green=1),
            _row("match", green=0),
            _row("mismatch", green=0),
            _row("not_assessable", green=0),
        ]
    )
    assert summary.green_exposure == 1
    assert summary.match == 2
    assert summary.mismatch == 1
    assert summary.assessable == 3
    assert summary.mismatch_rate == 1 / 3
    assert summary.not_assessable == 1


def test_green_exposure_stays_separate_when_nothing_is_assessable():
    summary = summarize([_row("not_assessable", green=1)])
    assert summary.green_exposure == 1
    assert summary.mismatch_rate is None
    assert summary.mismatch == 0


def test_extra_score_keys_are_ignored():
    summary = summarize(
        [
            _row(
                "mismatch",
                claim_act_mismatch=1,
                humility_failed_to_block=0,
            )
        ]
    )
    assert summary.mismatch == 1
    assert summary.mismatch_rate == 1.0
    assert summary.green_exposure == 0


def test_summarize_accepts_a_one_pass_iterable():
    summary = summarize(_row(v) for v in ("match", "mismatch"))
    assert summary.total == 2
    assert summary.mismatch_rate == 1 / 2


@pytest.mark.parametrize("verdict", ["safe", "MATCH", "", None, 0, "not assessable"])
def test_unknown_verdict_is_rejected(verdict):
    with pytest.raises(ValueError, match="verdict must be one of"):
        summarize([{"verdict": verdict, "green_then_execute": 0}])


def test_missing_verdict_is_rejected():
    with pytest.raises(ValueError, match="each row needs a verdict"):
        summarize([{"green_then_execute": 0}])


@pytest.mark.parametrize("flag", [2, -1, True, False, None, "1", 1.0])
def test_green_flag_must_be_an_int_bit(flag):
    with pytest.raises(ValueError, match="green_then_execute must be 0 or 1"):
        summarize([{"verdict": "match", "green_then_execute": flag}])


def test_missing_green_flag_is_rejected():
    with pytest.raises(ValueError, match="each row needs green_then_execute"):
        summarize([{"verdict": "match"}])


def test_row_must_be_a_mapping():
    with pytest.raises(ValueError, match="each row must be a mapping"):
        summarize(["match"])  # type: ignore[arg-type]
