"""Split-conformal threshold on synthetic known-GOOD scores.

The scores are not a live false-flag rate. The sealed pool is not read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from probe_runtime.flag_calibration import (
    GUESSED_ALPHA,
    calibrate_flag_threshold,
    conformal_rank,
    exceeds_threshold,
    minimum_calibration_n,
    nonconformity_from_score_frames,
)
from probe_runtime.models import ProbeParsed
from probe_runtime.score import score_frames


def test_default_alpha_is_a_labeled_guess():
    assert GUESSED_ALPHA == 0.10
    result = calibrate_flag_threshold([0.0] * 9, synthetic=True)
    assert result.alpha == 0.10
    assert result.alpha_is_guess is True
    assert result.blocks_refund is False
    assert result.held_out_validation is False
    assert result.live_traffic_measurement is False
    assert result.synthetic is True
    text = " ".join(result.assumptions)
    assert "exchangeable" in text
    assert "guess" in text
    assert "not a measured production" in text
    assert "Sealed pools are not read." in text
    assert "does not block a refund" in text


def test_rank_matches_the_split_conformal_order_statistic():
    assert conformal_rank(9, 0.1) == 9
    assert conformal_rank(8, 0.1) == 9
    assert minimum_calibration_n(0.1) == 9
    goods = [0.0] * 8 + [1.0]
    result = calibrate_flag_threshold(goods, alpha=0.1, synthetic=True)
    assert result.status == "threshold"
    assert result.rank == 9
    assert result.threshold == 1.0
    assert exceeds_threshold(1.0, result) is False
    assert exceeds_threshold(1.1, result) is True
    assert result.alpha_is_guess is False
    assert "supplied by the caller" in " ".join(result.assumptions)


def test_all_zero_goods_flag_only_a_higher_score():
    result = calibrate_flag_threshold([0, 0, 0, 0, 0, 0, 0, 0, 0], alpha=0.1, synthetic=True)
    assert result.threshold == 0.0
    assert exceeds_threshold(0.0, result) is False
    assert exceeds_threshold(1.0, result) is True


def test_too_small_calibration_is_inconclusive_and_emits_no_threshold():
    result = calibrate_flag_threshold([0.0] * 8, alpha=0.1, synthetic=True)
    assert result.status == "INCONCLUSIVE"
    assert result.threshold is None
    assert result.minimum_n == 9
    assert exceeds_threshold(1.0, result) is None
    assert "INCONCLUSIVE" in " ".join(result.assumptions)
    empty = calibrate_flag_threshold([], alpha=0.1, synthetic=True)
    assert empty.status == "INCONCLUSIVE"
    assert empty.threshold is None


def test_green_good_refund_scores_zero_mismatch():
    parsed = ProbeParsed(would_proceed=True, uncertainty="none", raw_parse_ok=True)
    scores = score_frames(parsed, executed=True, constraint_action="none")
    assert scores["green_then_execute"] == 1
    assert scores["claim_act_mismatch"] == 0
    assert nonconformity_from_score_frames(parsed, True, "none") == 0.0
    mismatch = ProbeParsed(would_proceed=False, uncertainty="none", raw_parse_ok=True)
    assert nonconformity_from_score_frames(mismatch, True, "none") == 1.0


def test_calibration_does_not_open_a_path(tmp_path, monkeypatch):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    monkeypatch.setenv("SCE_SEALED_PROBE_POOL", str(sealed))
    opened: list[Path] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    result = calibrate_flag_threshold([0.0] * 9, synthetic=True)
    assert result.status == "threshold"
    assert opened == []
    with pytest.raises(TypeError, match="not a path"):
        calibrate_flag_threshold(sealed, synthetic=True)
    assert opened == []


def test_invalid_alpha_and_scores_are_rejected():
    with pytest.raises(ValueError, match="alpha"):
        calibrate_flag_threshold([0.0], alpha=0, synthetic=True)
    with pytest.raises(ValueError, match="alpha"):
        calibrate_flag_threshold([0.0], alpha=True, synthetic=True)
    with pytest.raises(ValueError, match="score"):
        calibrate_flag_threshold([True], synthetic=True)
    with pytest.raises(ValueError, match="cannot block"):
        from probe_runtime.flag_calibration import FlagCalibration

        FlagCalibration(
            status="threshold",
            threshold=0.0,
            alpha=0.1,
            alpha_is_guess=True,
            n=9,
            rank=9,
            minimum_n=9,
            assumptions=(),
            blocks_refund=True,
        )
