"""Regression tests for the degenerate-rating failure and the partial pooling
that fixes it.

The bug these cover is the dangerous kind: nothing raises, the parameters
look like numbers, and the model quietly claims a team's opponent will score
zero goals with probability ~1. It reproduced on real Premier League data a
few matchweeks into 2025/26, when newly-promoted Sunderland had played one
match and not conceded.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from hattrick.model.dixon_coles import DixonColesModel
from test_dixon_coles import make_synthetic_league


def _league_plus_newcomer(n_conceded=0, newcomer_matches=1, seed=7):
    """An established league, plus a promoted side that has just arrived and
    conceded `n_conceded` goals in `newcomer_matches` games."""
    matches, *_ = make_synthetic_league(n_teams=8, matches_per_pair=4, seed=seed)
    last = matches["date"].max()
    extra = []
    for k in range(newcomer_matches):
        extra.append({
            "date": last + pd.Timedelta(days=k + 1),
            "home": "Newcomer", "away": f"Team{k % 8}",
            "home_goals": 1, "away_goals": n_conceded,
        })
    return pd.concat([matches, pd.DataFrame(extra)], ignore_index=True)


def test_newcomer_with_a_clean_sheet_does_not_break_the_whole_fit():
    """The failure mode was global, not local: log(0) as a starting point made
    L-BFGS-B return ABNORMAL, so *every* team's rating came from a failed
    optimisation and rho was left at its initial value."""
    matches = _league_plus_newcomer(n_conceded=0, newcomer_matches=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a non-convergence warning fails the test
        model = DixonColesModel(xi=0.0).fit(matches)

    assert model._fit_result.success
    assert np.isfinite(model.params.log_likelihood)
    assert all(np.isfinite(v) and v > 0 for v in model.params.attack.values())
    assert all(np.isfinite(v) and v > 0 for v in model.params.defense.values())


def test_newcomer_clean_sheet_does_not_imply_opponent_scores_zero():
    """The consequential assertion: a promoted team's defense rating must
    stay in a sane range, so opponents get a plausible scoring rate rather
    than ~0."""
    matches = _league_plus_newcomer(n_conceded=0, newcomer_matches=1)
    model = DixonColesModel(xi=0.0).fit(matches)

    _, mu = model.rates("Newcomer", "Team1")
    assert 0.4 < mu < 3.0, f"opponent away rate {mu} is not a plausible goal expectation"

    league_mean_defense = float(np.mean([v for t, v in model.params.defense.items()
                                         if t != "Newcomer"]))
    assert model.params.defense["Newcomer"] > 0.3 * league_mean_defense


def test_shrinkage_pulls_low_data_teams_toward_the_league_mean():
    """This is the actual point of the prior, so test it directly: the same
    extreme one-match record should move the rating far less than an
    unpooled fit would."""
    matches = _league_plus_newcomer(n_conceded=0, newcomer_matches=1)

    pooled = DixonColesModel(xi=0.0, rating_prior_sd=0.35).fit(matches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unpooled = DixonColesModel(xi=0.0, prior_strength=0.0, rating_prior_sd=None).fit(matches)

    def gap(m):
        others = [v for t, v in m.params.defense.items() if t != "Newcomer"]
        return abs(np.log(m.params.defense["Newcomer"]) - np.log(np.mean(others)))

    assert gap(pooled) < gap(unpooled), "pooling should move the newcomer closer to the mean"


def test_shrinkage_barely_moves_a_team_with_a_full_history():
    """Pooling must be nearly inert where there is real data, otherwise it is
    just bias. A team with a full season should land in essentially the same
    place with and without the prior."""
    matches, *_ = make_synthetic_league(n_teams=8, matches_per_pair=6, seed=3)
    pooled = DixonColesModel(xi=0.0, rating_prior_sd=0.35).fit(matches)
    unpooled = DixonColesModel(xi=0.0, prior_strength=0.0, rating_prior_sd=None).fit(matches)

    for team in pooled.params.teams:
        rel = abs(pooled.params.attack[team] / unpooled.params.attack[team] - 1.0)
        assert rel < 0.12, f"{team} attack moved {rel:.1%} on a full season of data"


def test_pooling_does_not_destroy_parameter_recovery():
    """Shrinkage trades a little bias for variance; confirm the bias is small
    enough that ground-truth recovery still holds."""
    matches, true_attack, true_defense, true_gamma = make_synthetic_league(
        n_teams=8, matches_per_pair=6, seed=42)
    model = DixonColesModel(xi=0.0, rating_prior_sd=0.35).fit(matches)
    teams = list(true_attack.keys())
    fit_a = np.array([model.params.attack[t] for t in teams])
    fit_d = np.array([model.params.defense[t] for t in teams])
    assert np.corrcoef([true_attack[t] for t in teams], fit_a)[0, 1] > 0.85
    assert np.corrcoef([true_defense[t] for t in teams], fit_d)[0, 1] > 0.85
    assert abs(model.params.home_advantage - true_gamma) < 0.25


def test_priors_can_be_switched_off_for_a_pure_mle_fit():
    matches, *_ = make_synthetic_league(n_teams=6, matches_per_pair=4, seed=1)
    m = DixonColesModel(xi=0.0, prior_strength=0.0, rating_prior_sd=None).fit(matches)
    assert m.prior_strength == 0.0 and m.rating_prior_sd is None
    assert np.isfinite(m.params.log_likelihood)


def test_non_convergence_is_warned_about_not_swallowed():
    """A failed optimizer used to return silently with a NaN likelihood."""
    matches, *_ = make_synthetic_league(n_teams=4, matches_per_pair=2, seed=0)
    model = DixonColesModel(xi=0.0)

    class _Failed:
        success, message, fun, x = False, "ABNORMAL", np.nan, None

    real = model._fit_full

    def fake(*a, **k):
        alpha, beta, gamma, rho, _ = real(*a, **k)
        return alpha, beta, gamma, rho, _Failed()

    model._fit_full = fake
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model.fit(matches)
