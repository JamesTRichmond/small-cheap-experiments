"""Paraphrase stability on synthetic claim rows. No probe text.

Rows are (variant_id, would_proceed, raw_parse_ok). The ids are
placeholders. This file does not contain a rewording of a probe.

GUESS: the floor of two parseable answers is the rule under test. It
was not measured, and a passing case is not a validated stability rate.
"""

import pytest

from probe_runtime.stability import paraphrase_stability


def test_docstring_treats_a_flip_as_weak_evidence():
    text = (paraphrase_stability.__doc__ or "").lower()
    assert "weak evidence" in text
    assert "unstable claim" in text
    assert "does not generate" in text


def test_a_list_of_rows_is_accepted():
    result = paraphrase_stability(
        [
            ("variant-1", False, True),
            ("variant-2", False, True),
        ]
    )
    assert result.verdict == "stable"
    assert result.majority_would_proceed is False


def test_two_agreeing_answers_are_stable():
    result = paraphrase_stability(
        (
            ("variant-1", True, True),
            ("variant-2", True, True),
        )
    )
    assert result.verdict == "stable"
    assert result.stability == 1.0
    assert result.n_parseable == 2
    assert result.n_not_assessable == 0
    assert result.claim_unstable is False
    assert result.majority_would_proceed is True


def test_two_disagreeing_answers_are_unstable():
    # Modal share is 1/2. Neither side is a unique majority.
    result = paraphrase_stability(
        (
            ("variant-1", True, True),
            ("variant-2", False, True),
        )
    )
    assert result.verdict == "unstable"
    assert result.stability == 1 / 2
    assert result.n_parseable == 2
    assert result.claim_unstable is True
    assert result.majority_would_proceed is None


def test_three_to_one_agrees_with_the_majority_on_three_quarters():
    result = paraphrase_stability(
        (
            ("variant-1", False, True),
            ("variant-2", False, True),
            ("variant-3", False, True),
            ("variant-4", True, True),
        )
    )
    assert result.verdict == "unstable"
    assert result.stability == 3 / 4
    assert result.n_parseable == 4
    assert result.n_not_assessable == 0
    assert result.claim_unstable is True
    assert result.majority_would_proceed is False


def test_tie_has_no_unique_majority():
    result = paraphrase_stability(
        (
            ("variant-1", True, True),
            ("variant-2", True, True),
            ("variant-3", False, True),
            ("variant-4", False, True),
        )
    )
    assert result.verdict == "unstable"
    assert result.stability == 2 / 4
    assert result.claim_unstable is True
    assert result.majority_would_proceed is None


def test_fewer_than_two_parseable_answers_are_not_assessable():
    empty = paraphrase_stability(())
    assert empty.verdict == "not_assessable"
    assert empty.stability is None
    assert empty.n_parseable == 0
    assert empty.n_not_assessable == 0
    assert empty.claim_unstable is False
    assert empty.verdict != "stable"

    lone = paraphrase_stability((("variant-1", True, True),))
    assert lone.verdict == "not_assessable"
    assert lone.stability is None
    assert lone.n_parseable == 1
    assert lone.claim_unstable is False
    assert lone.majority_would_proceed is None
    assert lone.verdict != "stable"


def test_one_parseable_answer_stays_not_assessable_beside_bad_rows():
    result = paraphrase_stability(
        (
            ("variant-1", None, False),
            ("variant-2", True, True),
            ("variant-3", None, True),
        )
    )
    assert result.verdict == "not_assessable"
    assert result.stability is None
    assert result.n_parseable == 1
    assert result.n_not_assessable == 2
    assert result.claim_unstable is False


def test_failed_parse_does_not_vote_even_when_a_bool_is_present():
    result = paraphrase_stability(
        (
            ("variant-1", False, True),
            ("variant-2", True, False),
            ("variant-3", None, True),
            ("variant-4", False, True),
        )
    )
    assert result.verdict == "stable"
    assert result.stability == 1.0
    assert result.n_parseable == 2
    assert result.n_not_assessable == 2
    assert result.claim_unstable is False
    assert result.majority_would_proceed is False
    assert result.n_parseable + result.n_not_assessable == 4


def test_same_votes_in_another_order_match():
    forward = paraphrase_stability(
        (
            ("variant-1", True, True),
            ("variant-2", True, True),
            ("variant-3", False, True),
        )
    )
    backward = paraphrase_stability(
        (
            ("variant-3", False, True),
            ("variant-1", True, True),
            ("variant-2", True, True),
        )
    )
    assert forward == backward
    assert forward.stability == 2 / 3
    assert forward.majority_would_proceed is True


def test_duplicate_variant_id_is_rejected():
    with pytest.raises(ValueError, match="variant_id must be unique"):
        paraphrase_stability(
            (
                ("variant-1", True, True),
                ("variant-1", True, True),
            )
        )


@pytest.mark.parametrize(
    "variants",
    [
        "variant-1",
        (("variant-1", True),),
        ((1, True, True),),
        (("", True, True),),
        (("   ", True, True),),
        (("variant-1", 1, True),),
        (("variant-1", "false", True),),
        (("variant-1", True, 1),),
        (("variant-1", True, None),),
    ],
)
def test_malformed_rows_are_rejected(variants):
    with pytest.raises(ValueError):
        paraphrase_stability(variants)
