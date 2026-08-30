import numpy as np
import pandas as pd
import pytest

from hattrick.model.dixon_coles import DixonColesModel
from hattrick.model.season_sim import simulate_season


def _fit_toy_model():
    """4 teams, deliberately lopsided so we can sanity-check the simulator
    against a result we can reason about by hand: Strong should win the
    league far more often than Weak.
    """
    rng = np.random.default_rng(0)
    teams = ["Strong", "Mid1", "Mid2", "Weak"]
    true_attack = {"Strong": 2.2, "Mid1": 1.0, "Mid2": 1.0, "Weak": 0.3}
    true_defense = {"Strong": 0.5, "Mid1": 1.0, "Mid2": 1.0, "Weak": 2.0}
    gamma = 1.3
    rows = []
    day = 0
    for _ in range(6):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = true_attack[h] * true_defense[a] * gamma
                mu = true_attack[a] * true_defense[h]
                rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
                             "home": h, "away": a,
                             "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu)})
                day += 1
    matches = pd.DataFrame(rows)
    model = DixonColesModel(xi=0.0)
    model.fit(matches)
    return model, teams


def test_stronger_team_wins_title_more_often():
    model, teams = _fit_toy_model()
    played = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    remaining = pd.DataFrame([{"home": h, "away": a} for h in teams for a in teams if h != a])

    result = simulate_season(model, played, remaining, n_sims=2000, seed=1)

    assert result.title_prob["Strong"] > result.title_prob["Weak"]
    assert result.expected_rank["Strong"] < result.expected_rank["Weak"]
    # rank probabilities for each team should sum to 1 (every trial assigns exactly one rank)
    for t in teams:
        assert pytest.approx(sum(result.rank_probs[t]), abs=1e-9) == 1.0


def test_already_played_results_are_fixed_not_simulated():
    """A team that's already mathematically won the title (huge points lead,
    nothing left to play) should show ~100% title probability regardless of
    the model's rates - that's baked into 'played', not simulated.
    """
    model, teams = _fit_toy_model()
    played = pd.DataFrame([
        {"home": "Weak", "away": "Strong", "home_goals": 0, "away_goals": 0},
    ] * 0)  # no played matches needed for this check; use points directly instead
    # Simplest correct test: zero remaining fixtures -> title = whoever has most points already.
    played = pd.DataFrame([
        {"home": "Weak", "away": "Strong", "home_goals": 5, "away_goals": 0},
    ])
    remaining = pd.DataFrame(columns=["home", "away"])
    result = simulate_season(model, played, remaining, n_sims=100, seed=2)
    assert result.title_prob["Weak"] == 1.0
    assert result.title_prob["Strong"] == 0.0


def test_unseen_team_in_remaining_fixtures_is_skipped_not_crashed():
    model, teams = _fit_toy_model()
    played = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    remaining = pd.DataFrame([{"home": "Strong", "away": "NewlyPromoted"}])
    # Should not raise, even though "NewlyPromoted" has no fitted rating.
    result = simulate_season(model, played, remaining, n_sims=50, seed=3)
    assert "NewlyPromoted" in result.teams  # appears in the table (0 pts) even if unsimulated
