"""Tests for the continuous Poisson likelihood and the xG fitting target."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from hattrick.model.dixon_coles import DixonColesModel, poisson_logpmf


def test_continuous_poisson_matches_scipy_exactly_on_integers():
    """The generalization must be a strict superset of the discrete pmf, or
    switching to it silently changes every goals-target fit."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 12, 500).astype(float)
    lam = rng.uniform(0.05, 6.0, 500)
    assert np.allclose(poisson_logpmf(x, lam), poisson.logpmf(x, lam), rtol=0, atol=1e-12)


def test_continuous_poisson_is_finite_where_scipy_is_not():
    assert np.isfinite(poisson_logpmf(1.34, 1.5))
    assert poisson.logpmf(1.34, 1.5) == -np.inf


def test_continuous_poisson_is_maximized_at_lambda_equals_x():
    """Sanity on the shape, not just the value: the MLE of a single Poisson
    observation is the observation itself, continuous x included."""
    for x in (0.7, 1.34, 3.9):
        lams = np.linspace(0.05, 8, 4000)
        assert lams[np.argmax(poisson_logpmf(x, lams))] == pytest.approx(x, abs=0.01)


def _synthetic_league(n_teams=10, rounds=6, seed=0, xg_noise=0.15, coverage=1.0):
    """A league with KNOWN attack/defense, where xG is a low-noise reading of
    the true rate and goals are a Poisson draw from it.

    This is the generative story the xG target assumes: both observe the same
    lambda, but xG observes it far more precisely. A model fit on xG should
    therefore recover the truth better than one fit on goals.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    atk = dict(zip(teams, rng.uniform(0.7, 1.4, n_teams)))
    dfn = dict(zip(teams, rng.uniform(0.7, 1.4, n_teams)))
    scale = np.mean(list(atk.values()))
    atk = {k: v / scale for k, v in atk.items()}
    dfn = {k: v * scale for k, v in dfn.items()}
    gamma, rows, day = 1.35, [], 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = atk[h] * dfn[a] * gamma
                mu = atk[a] * dfn[h]
                day += 1
                rows.append({
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
                    "home": h, "away": a,
                    "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                    "home_xg": max(lam * rng.lognormal(0, xg_noise), 1e-3),
                    "away_xg": max(mu * rng.lognormal(0, xg_noise), 1e-3),
                })
    df = pd.DataFrame(rows)
    if coverage < 1.0:
        drop = rng.random(len(df)) > coverage
        df.loc[drop, ["home_xg", "away_xg"]] = np.nan
    return df, atk, dfn, gamma


def _recovery(model, atk, dfn):
    teams = list(atk)
    a = np.corrcoef([atk[t] for t in teams], [model.params.attack[t] for t in teams])[0, 1]
    d = np.corrcoef([dfn[t] for t in teams], [model.params.defense[t] for t in teams])[0, 1]
    return a, d


def test_model_recovers_known_parameters_from_a_continuous_target():
    """The Task-4 analogue of the goals-based recovery test: continuous
    'goals' standing in for xG must still identify attack/defense."""
    m, atk, dfn, gamma = _synthetic_league(rounds=6, seed=42)
    model = DixonColesModel(xi=0.0, target="xg").fit(m)
    a, d = _recovery(model, atk, dfn)
    assert a > 0.9 and d > 0.9
    assert abs(model.params.home_advantage - gamma) < 0.2
    assert model.params.xg_coverage == pytest.approx(1.0)
    assert model.params.target == "xg"


def test_low_noise_xg_recovers_the_truth_better_than_goals_do():
    """The whole premise of the feature, stated as a test: when xG is a less
    noisy reading of the same lambda, fitting on it must beat fitting on
    goals. If this fails, the xG target is not earning its complexity."""
    wins = 0
    for seed in range(6):
        m, atk, dfn, _ = _synthetic_league(rounds=3, seed=seed, xg_noise=0.1)
        g = _recovery(DixonColesModel(xi=0.0, target="goals").fit(m), atk, dfn)
        x = _recovery(DixonColesModel(xi=0.0, target="xg").fit(m), atk, dfn)
        wins += (np.mean(x) > np.mean(g))
    assert wins >= 5, f"xG target beat goals in only {wins}/6 synthetic leagues"


def test_partial_xg_coverage_falls_back_per_match_and_records_it():
    m, atk, dfn, _ = _synthetic_league(rounds=4, seed=1, coverage=0.6)
    model = DixonColesModel(xi=0.0, target="xg").fit(m)
    assert 0.5 < model.params.xg_coverage < 0.7
    a, d = _recovery(model, atk, dfn)
    assert a > 0.85 and d > 0.85


def test_missing_xg_columns_warn_and_degrade_to_goals():
    m, atk, dfn, _ = _synthetic_league(rounds=3, seed=2)
    m = m.drop(columns=["home_xg", "away_xg"])
    with pytest.warns(RuntimeWarning, match="no home_xg/away_xg"):
        model = DixonColesModel(xi=0.0, target="xg").fit(m)
    assert model.params.xg_coverage == 0.0
    goals = DixonColesModel(xi=0.0, target="goals").fit(m)
    assert model.params.attack == pytest.approx(goals.params.attack, rel=1e-6)


def test_rho_is_estimated_from_goals_when_the_target_is_continuous():
    """rho describes the discrete scoreline process; a continuous target
    never touches the four cells tau adjusts, so it cannot inform rho. It
    must therefore come from the actual goals, and must not simply be left
    at its initial value."""
    m, *_ = _synthetic_league(rounds=6, seed=7)
    model = DixonColesModel(xi=0.0, target="xg").fit(m)
    assert model.params.rho != 0.0
    assert -0.99 < model.params.rho < 0.99


def test_goals_target_is_bit_for_bit_unchanged_by_the_refactor():
    """Backward compatibility: swapping scipy's logpmf for the continuous
    form must not move a goals-based fit."""
    m, *_ = _synthetic_league(rounds=4, seed=3)
    a = DixonColesModel(xi=0.0018).fit(m).params
    b = DixonColesModel(xi=0.0018, target="goals").fit(m).params
    assert a.attack == pytest.approx(b.attack)
    assert a.rho == pytest.approx(b.rho)


def test_invalid_target_rejected():
    with pytest.raises(ValueError, match="target must be"):
        DixonColesModel(target="npxg")
