"""probes_discriminate on synthetic summaries. Not a probe validation.

Fixtures are invented verdict rows. They are not holdout probes, and a
passing gap does not mean a Risk team has checked anything.

GUESS: every min_gap below is a fixture cutoff so the test can name one.
probes_discriminate has no default. None of these numbers came from a
Risk team.
"""

import math

import pytest

from probe_runtime.score_summary import probes_discriminate, summarize


def _summary(verdicts: list[str]):
    return summarize(
        {"verdict": verdict, "green_then_execute": 0} for verdict in verdicts
    )


# Sloppy mismatches on 3 of 4 assessable rows (0.75). Careful mismatches
# on 1 of 4 (0.25). Gap is 0.5. The 0.75 and 0.25 are fixture arithmetic,
# not measured probe rates.
SLOPPY = _summary(["mismatch", "mismatch", "mismatch", "match"])
CAREFUL = _summary(["match", "match", "match", "mismatch"])


def test_sloppy_and_careful_summaries_can_clear_a_gap():
    result = probes_discriminate(SLOPPY, CAREFUL, min_gap=0.4)
    assert result.gap == 0.5
    assert result.passed is True
    assert result.min_gap == 0.4


def test_equal_scores_do_not_discriminate():
    same = _summary(["mismatch", "match"])
    result = probes_discriminate(same, same, min_gap=0.1)
    assert result.gap == 0.0
    assert result.passed is False


def test_gap_equal_to_min_gap_passes():
    result = probes_discriminate(SLOPPY, CAREFUL, min_gap=0.5)
    assert result.gap == 0.5
    assert result.passed is True


def test_gap_just_under_min_gap_fails():
    result = probes_discriminate(
        SLOPPY, CAREFUL, min_gap=math.nextafter(0.5, math.inf)
    )
    assert result.passed is False
    assert result.gap == 0.5


def test_careful_worse_than_sloppy_fails():
    result = probes_discriminate(CAREFUL, SLOPPY, min_gap=0.1)
    assert result.gap == -0.5
    assert result.passed is False


def test_not_assessable_rows_do_not_dilute_the_gap():
    """Nine unparsed rows must not turn one mismatch into a 0.1 rate."""
    sloppy = _summary(["mismatch"] + ["not_assessable"] * 9)
    careful = _summary(["match"])
    assert sloppy.mismatch_rate == 1.0
    assert careful.mismatch_rate == 0.0
    result = probes_discriminate(sloppy, careful, min_gap=0.5)
    assert result.gap == 1.0
    assert result.passed is True


def test_empty_summaries_have_no_gap():
    empty = summarize([])
    result = probes_discriminate(empty, empty, min_gap=0.2)
    assert empty.mismatch_rate is None
    assert result.gap is None
    assert result.passed is False


def test_all_not_assessable_is_not_a_zero_gap():
    blank = _summary(["not_assessable", "not_assessable"])
    result = probes_discriminate(blank, CAREFUL, min_gap=0.2)
    assert blank.mismatch_rate is None
    assert result.gap is None
    assert result.passed is False


def test_green_exposure_does_not_change_the_gap():
    sloppy = summarize(
        [
            {"verdict": "mismatch", "green_then_execute": 0},
            {"verdict": "match", "green_then_execute": 0},
        ]
    )
    careful_exposed = summarize(
        [
            {"verdict": "mismatch", "green_then_execute": 1},
            {"verdict": "match", "green_then_execute": 1},
        ]
    )
    assert sloppy.mismatch_rate == careful_exposed.mismatch_rate
    assert sloppy.green_exposure == 0
    assert careful_exposed.green_exposure == 2
    result = probes_discriminate(sloppy, careful_exposed, min_gap=0.1)
    assert result.gap == 0.0
    assert result.passed is False


def test_min_gap_is_required():
    with pytest.raises(TypeError):
        probes_discriminate(SLOPPY, CAREFUL)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "min_gap",
    [0, 0.0, -0.1, -1, True, False, None, "0.2", math.nan, math.inf, -math.inf],
)
def test_min_gap_must_be_a_positive_real(min_gap):
    with pytest.raises(ValueError, match="min_gap must be a real number greater than 0"):
        probes_discriminate(SLOPPY, CAREFUL, min_gap=min_gap)


def test_summaries_must_be_score_summaries():
    with pytest.raises(TypeError, match="summaries must come from summarize"):
        probes_discriminate({"mismatch_rate": 1.0}, CAREFUL, min_gap=0.2)  # type: ignore[arg-type]
