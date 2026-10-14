import numpy as np
import pytest
from hattrick.model.scorelines import outcome_probs, over_under, btts, correct_score_probs

def _toy_matrix():
    n = 4
    M = np.abs(np.random.default_rng(0).normal(size=(n, n)))
    return M / M.sum()

def test_outcome_probs_sum_to_one():
    assert pytest.approx(sum(outcome_probs(_toy_matrix()).values()), abs=1e-9) == 1.0

def test_over_under_sums_to_one():
    ou = over_under(_toy_matrix(), line=2.5)
    assert pytest.approx(ou["over"] + ou["under"], abs=1e-9) == 1.0

def test_btts_sums_to_one():
    b = btts(_toy_matrix())
    assert pytest.approx(b["yes"] + b["no"], abs=1e-9) == 1.0

def test_correct_score_sorted_descending():
    probs = [p for _, p in correct_score_probs(_toy_matrix(), top_n=5)]
    assert probs == sorted(probs, reverse=True)
