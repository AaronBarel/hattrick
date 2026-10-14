import numpy as np
import pandas as pd
import pytest

from hattrick.evaluation import (
    log_loss, brier_score, devig_bookmaker_probs, actual_outcome, backtest_by_league,
    walk_forward_backtest,
)


def test_log_loss_rewards_confident_correct_predictions():
    confident_correct = [{"home_win": 0.9, "draw": 0.05, "away_win": 0.05}]
    uniform = [{"home_win": 1/3, "draw": 1/3, "away_win": 1/3}]
    actual = ["home_win"]
    assert log_loss(confident_correct, actual) < log_loss(uniform, actual)


def test_log_loss_punishes_confident_wrong_predictions():
    confident_wrong = [{"home_win": 0.05, "draw": 0.05, "away_win": 0.9}]
    uniform = [{"home_win": 1/3, "draw": 1/3, "away_win": 1/3}]
    actual = ["home_win"]
    assert log_loss(confident_wrong, actual) > log_loss(uniform, actual)


def test_brier_score_zero_for_perfect_prediction():
    perfect = [{"home_win": 1.0, "draw": 0.0, "away_win": 0.0}]
    assert brier_score(perfect, ["home_win"]) == pytest.approx(0.0)


def test_devig_removes_overround():
    # Odds with an obvious overround: 1/2 + 1/4 + 1/4 = 1.5 (implies 50% margin)
    probs = devig_bookmaker_probs(2.0, 4.0, 4.0)
    assert probs is not None
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0
    # Ratios should be preserved: home was twice as likely as draw/away pre-devig
    assert probs["home_win"] == pytest.approx(2 * probs["draw"])


def test_devig_returns_none_for_invalid_odds():
    assert devig_bookmaker_probs(None, 3.0, 4.0) is None
    assert devig_bookmaker_probs(0.5, 3.0, 4.0) is None  # odds must be > 1.0


def test_actual_outcome_classification():
    assert actual_outcome(2, 1) == "home_win"
    assert actual_outcome(1, 2) == "away_win"
    assert actual_outcome(1, 1) == "draw"


def _synthetic_matches_with_odds(n_teams=6, n_rounds=10, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    attack = dict(zip(teams, rng.uniform(0.8, 1.3, n_teams)))
    defense = dict(zip(teams, rng.uniform(0.8, 1.3, n_teams)))
    gamma = 1.3
    rows = []
    day = 0
    for _ in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = attack[h] * defense[a] * gamma
                mu = attack[a] * defense[h]
                hg, ag = rng.poisson(lam), rng.poisson(mu)
                # synthetic "bookmaker" odds: true win probs, noised and with ~5% overround added
                p_h, p_d, p_a = 0.45, 0.25, 0.30  # rough placeholder shape, doesn't need to be exact
                rows.append({
                    "date": pd.Timestamp("2023-08-01") + pd.Timedelta(days=day),
                    "home": h, "away": a, "home_goals": hg, "away_goals": ag,
                    "b365h": round(1 / (p_h * 1.05), 2), "b365d": round(1 / (p_d * 1.05), 2),
                    "b365a": round(1 / (p_a * 1.05), 2),
                })
                day += 1
    return pd.DataFrame(rows)


def test_walk_forward_backtest_runs_end_to_end_with_market_comparison():
    matches = _synthetic_matches_with_odds(n_teams=6, n_rounds=8, seed=1)
    holdout_start = matches["date"].quantile(0.5)  # roughly the halfway point by time

    result = walk_forward_backtest(matches, holdout_start=holdout_start, xi=0.0,
                                    refit_every=15, max_goals=8)

    assert result.n_matches > 0
    assert result.model_log_loss > 0
    assert result.market_log_loss is not None
    assert result.n_with_market == result.n_matches
    assert len(result.per_match) == result.n_matches


def test_walk_forward_backtest_without_odds_columns():
    matches = _synthetic_matches_with_odds(n_teams=6, n_rounds=8, seed=2).drop(
        columns=["b365h", "b365d", "b365a"])
    holdout_start = matches["date"].quantile(0.5)
    result = walk_forward_backtest(matches, holdout_start=holdout_start, xi=0.0, refit_every=15)
    assert result.market_log_loss is None
    assert result.n_with_market == 0


def _league(league, seed, n_teams=6, n_rounds=6):
    m = _synthetic_matches_with_odds(n_teams=n_teams, n_rounds=n_rounds, seed=seed)
    m["home"] = league + "-" + m["home"]
    m["away"] = league + "-" + m["away"]
    m["league"] = league
    return m


def test_walk_forward_backtest_refuses_to_pool_leagues():
    """One fit across leagues would put teams that never meet on one rating
    scale. The backtest must refuse rather than quietly do it."""
    both = pd.concat([_league("A", 1), _league("B", 2)])
    with pytest.raises(ValueError, match="one league at a time"):
        walk_forward_backtest(both, holdout_start=both["date"].quantile(0.5))


def test_backtest_by_league_fits_each_league_separately_and_pools_per_match():
    a, b = _league("A", 1), _league("B", 2, n_teams=5)
    results = backtest_by_league(pd.concat([a, b]), holdout_start=a["date"].quantile(0.5),
                                 xi=0.0, refit_every=15)
    assert set(results) == {"A", "B", "ALL"}
    pooled = results["ALL"]
    assert pooled.n_matches == results["A"].n_matches + results["B"].n_matches
    assert set(pooled.per_match["league"]) == {"A", "B"}
    for attr in ("model_log_loss", "market_log_loss"):
        weighted = sum(getattr(results[k], attr) * results[k].n_matches for k in "AB")
        assert getattr(pooled, attr) == pytest.approx(weighted / pooled.n_matches)
    assert pooled.market_source == results["A"].market_source
