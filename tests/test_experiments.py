"""Tests for the model-comparison harness.

The failure these guard against is a comparison that looks rigorous and is
not: two configs scored on different match sets, or a "significant" verdict
that a broken bootstrap would hand out for free.
"""
import numpy as np
import pandas as pd
import pytest

from hattrick.experiments import ModelConfig, align, compare_configs, format_comparison


def _matches(seed=0, n_teams=8, rounds=5):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    atk = dict(zip(teams, rng.uniform(0.7, 1.4, n_teams)))
    rows, day = [], 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                lam, mu = atk[h] * 1.35, atk[a]
                rows.append({"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
                             "home": h, "away": a,
                             "home_goals": rng.poisson(lam), "away_goals": rng.poisson(mu),
                             "home_xg": lam, "away_xg": mu,
                             "b365ch": 2.0, "b365cd": 3.4, "b365ca": 4.0})
    return pd.DataFrame(rows)


def test_align_restricts_to_the_intersection_of_scored_fixtures():
    """A config that skipped hard matches must not be judged on the easy
    remainder."""
    base = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                         "home": ["A", "B", "C"], "away": ["B", "C", "A"]})
    fewer = base.iloc[:2].copy()
    out = align({"full": base, "partial": fewer})
    assert len(out["full"]) == len(out["partial"]) == 2
    assert list(out["full"]["home"]) == list(out["partial"]["home"])


def test_identical_configs_are_never_declared_significantly_different():
    """The most important null: a comparison that finds a difference between
    a model and itself is broken."""
    m = _matches()
    cfgs = [ModelConfig("a"), ModelConfig("b")]
    df = compare_configs({"L": m}, cfgs, holdout_start=m["date"].quantile(0.7), n_boot=2000)
    b = df[df["config"] == "b"].iloc[0]
    assert b["vs_baseline"] == pytest.approx(0.0, abs=1e-12)
    assert not b["significant"]


def test_a_deliberately_crippled_config_is_flagged_as_worse():
    """The complementary check: the harness must actually detect a real
    difference, not just fail to find fake ones."""
    m = _matches(rounds=8)
    cfgs = [ModelConfig("sane"),
            ModelConfig("over-decayed", xi=0.5)]  # forgets everything but last few days
    df = compare_configs({"L": m}, cfgs, holdout_start=m["date"].quantile(0.6), n_boot=2000)
    bad = df[df["config"] == "over-decayed"].iloc[0]
    assert bad["vs_baseline"] > 0, "a crippled model should score worse"


def test_pooling_across_leagues_weights_by_matches_not_by_league():
    big, small = _matches(seed=1, rounds=6), _matches(seed=2, n_teams=6, rounds=6)
    cfgs = [ModelConfig("a")]
    # A date inside both leagues' spans, so neither has an empty holdout.
    cutoff = min(big["date"].max(), small["date"].max()) - pd.Timedelta(days=40)
    df = compare_configs({"big": big, "small": small}, cfgs,
                         holdout_start=cutoff, n_boot=500)
    per_league = df.attrs["per_league"]
    assert set(per_league["league"]) == {"big", "small"}
    assert df.iloc[0]["n_matches"] == per_league["n"].sum()


def test_market_benchmark_is_reported_alongside():
    m = _matches()
    df = compare_configs({"L": m}, [ModelConfig("a")],
                         holdout_start=m["date"].quantile(0.7), n_boot=500)
    assert "market_score" in df.attrs and df.attrs["market_score"] > 0


def test_formatting_states_the_verdict_in_words():
    m = _matches()
    df = compare_configs({"L": m}, [ModelConfig("a"), ModelConfig("b", xi=0.01)],
                         holdout_start=m["date"].quantile(0.7), n_boot=500)
    text = format_comparison(df)
    assert "baseline" in text
    assert "noise" in text or "SIGNIFICANT" in text
