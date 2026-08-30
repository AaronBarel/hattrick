import numpy as np
import pandas as pd
import pytest
from hattrick.model.dixon_coles import DixonColesModel


def make_synthetic_league(n_teams=8, matches_per_pair=2, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    true_attack = dict(zip(teams, rng.uniform(0.7, 1.4, n_teams)))
    true_defense = dict(zip(teams, rng.uniform(0.7, 1.4, n_teams)))
    scale = np.mean(list(true_attack.values()))
    true_attack = {k: v / scale for k, v in true_attack.items()}
    true_defense = {k: v * scale for k, v in true_defense.items()}
    true_gamma = 1.35
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    day = 0
    for _ in range(matches_per_pair):
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                lam = true_attack[home] * true_defense[away] * true_gamma
                mu = true_attack[away] * true_defense[home]
                x, y = rng.poisson(lam), rng.poisson(mu)
                rows.append({"date": base_date + pd.Timedelta(days=day), "home": home,
                             "away": away, "home_goals": x, "away_goals": y})
                day += 1
    return pd.DataFrame(rows), true_attack, true_defense, true_gamma


def test_ipf_and_fit_recover_known_parameters():
    matches, true_attack, true_defense, true_gamma = make_synthetic_league(
        n_teams=8, matches_per_pair=6, seed=42)
    model = DixonColesModel(xi=0.0)
    model.fit(matches)
    teams = list(true_attack.keys())
    true_a = np.array([true_attack[t] for t in teams])
    fit_a = np.array([model.params.attack[t] for t in teams])
    true_d = np.array([true_defense[t] for t in teams])
    fit_d = np.array([model.params.defense[t] for t in teams])
    assert np.corrcoef(true_a, fit_a)[0, 1] > 0.85
    assert np.corrcoef(true_d, fit_d)[0, 1] > 0.85
    assert abs(model.params.home_advantage - true_gamma) < 0.25


def test_fit_produces_valid_scoreline_matrix():
    from hattrick.model.markets import score_matrix, outcome_probs
    matches, *_ = make_synthetic_league(n_teams=6, matches_per_pair=3, seed=1)
    model = DixonColesModel(xi=0.0)
    model.fit(matches)
    M = score_matrix(model, "Team0", "Team1", max_goals=10)
    assert M.shape == (11, 11)
    assert pytest.approx(M.sum(), abs=1e-9) == 1.0
    assert np.all(M >= 0)
    probs = outcome_probs(M)
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0


def test_rates_raises_on_unknown_team():
    matches, *_ = make_synthetic_league(n_teams=4, matches_per_pair=2, seed=2)
    model = DixonColesModel(xi=0.0)
    model.fit(matches)
    with pytest.raises(KeyError):
        model.rates("Team0", "NotARealTeam")
