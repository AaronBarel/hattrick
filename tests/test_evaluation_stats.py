import numpy as np
import pytest

from hattrick.evaluation import (
    calibration_curve,
    expected_calibration_error,
    paired_bootstrap_test,
)


def test_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_test(np.array([1.0]), np.array([1.0, 2.0]))


def test_bootstrap_identical_scores_are_not_significant():
    scores = np.arange(20, dtype=float)
    result = paired_bootstrap_test(scores, scores, n_boot=500, seed=1)
    assert result.observed_diff == 0
    assert not result.significant


def test_calibration_curve_covers_all_predictions():
    probs = [{"home_win": value, "draw": 1 - value, "away_win": 0.0}
             for value in np.linspace(0.1, 0.9, 20)]
    actual = ["home_win" if item["home_win"] > 0.5 else "draw" for item in probs]
    curve = calibration_curve(probs, actual, "home_win", n_bins=5)
    assert curve["n_matches"].sum() == len(probs)
    assert expected_calibration_error(probs, actual, "home_win") >= 0
