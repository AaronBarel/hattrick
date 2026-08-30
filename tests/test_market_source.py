"""Tests for the pre-closing vs. closing odds distinction.

The bug these guard against is silent and flattering: benchmarking the model
against a stale, days-before-kickoff price while calling it "the closing
line". That makes the model look better than it is, and nothing crashes, so
only an explicit test catches it.
"""
import numpy as np
import pandas as pd
import pytest

from hattrick.evaluation import (
    MARKET_SOURCES,
    devig_bookmaker_probs,
    resolve_market_source,
    walk_forward_backtest,
)


def _odds_frame(n, cols_present, coverage=1.0):
    """Build a frame with the given odds column sets populated."""
    df = pd.DataFrame({"idx": range(n)})
    for name in cols_present:
        h, d, a = MARKET_SOURCES[name][0]
        df[h], df[d], df[a] = 2.0, 3.5, 4.0
        if coverage < 1.0:
            blank = df.index >= int(n * coverage)
            df.loc[blank, [h, d, a]] = np.nan
    return df


def test_preference_order_prefers_sharpest_closing_line():
    df = _odds_frame(100, ["preclosing_b365", "closing_b365", "closing_avg", "closing_pinnacle"])
    name, cols, label = resolve_market_source(df, "auto")
    assert name == "closing_pinnacle"
    assert cols == ("psch", "pscd", "psca")
    assert "CLOSING" in label


def test_falls_through_to_next_closing_source_when_present():
    df = _odds_frame(100, ["preclosing_b365", "closing_b365"])
    name, _, _ = resolve_market_source(df, "auto")
    assert name == "closing_b365"


def test_partially_covered_source_is_rejected_not_silently_subsampled():
    """Pinnacle columns really do go blank partway through a season. Scoring
    on the surviving rows would compare against a non-random subsample, so
    the resolver must fall through rather than use it."""
    df = _odds_frame(100, ["closing_b365"])
    partial = _odds_frame(100, ["closing_pinnacle"], coverage=0.55)
    df = pd.concat([df, partial.drop(columns=["idx"])], axis=1)

    name, _, _ = resolve_market_source(df, "auto")
    assert name == "closing_b365", "55%-covered Pinnacle should not win over fully-covered Bet365"

    # ...but it is still selectable if the caller explicitly insists.
    name, _, _ = resolve_market_source(df, "closing_pinnacle")
    assert name == "closing_pinnacle"


def test_preclosing_is_last_resort_and_is_labelled_as_stale():
    df = _odds_frame(100, ["preclosing_b365"])
    name, cols, label = resolve_market_source(df, "auto")
    assert name == "preclosing_b365"
    assert cols == ("b365h", "b365d", "b365a")
    assert "PRE-CLOSING" in label and "weaker benchmark" in label


def test_no_odds_at_all_returns_none():
    assert resolve_market_source(pd.DataFrame({"idx": range(10)}), "auto") is None


def test_unknown_source_name_raises():
    with pytest.raises(ValueError, match="Unknown market_source"):
        resolve_market_source(_odds_frame(10, ["closing_b365"]), "closing_betfair")


def test_devig_removes_a_known_overround():
    """Hand-checkable: odds of 2.0/4.0/4.0 imply 0.5+0.25+0.25 = 1.0 exactly,
    so a fair book must come back unchanged."""
    p = devig_bookmaker_probs(2.0, 4.0, 4.0)
    assert p == pytest.approx({"home_win": 0.5, "draw": 0.25, "away_win": 0.25})

    # Halve every price -> overround of 2.0, same fair probabilities.
    p2 = devig_bookmaker_probs(1.0000001, 2.0, 2.0)
    assert sum(p2.values()) == pytest.approx(1.0)


def _synthetic_matches(seed=0, n_seasons=3):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows, date = [], pd.Timestamp("2022-08-01")
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                date += pd.Timedelta(days=1)
                rows.append({
                    "date": date, "home": h, "away": a,
                    "home_goals": rng.poisson(1.5), "away_goals": rng.poisson(1.1),
                })
    return pd.DataFrame(rows)


def test_backtest_reports_which_snapshot_it_scored_against():
    """End-to-end: the result must name its benchmark, and must prefer the
    closing columns when both snapshots are present."""
    m = _synthetic_matches()
    m["b365h"], m["b365d"], m["b365a"] = 2.0, 3.5, 4.0        # pre-closing
    m["b365ch"], m["b365cd"], m["b365ca"] = 2.1, 3.4, 3.9      # closing

    res = walk_forward_backtest(m, holdout_start=m["date"].quantile(0.7), refit_every=1000)
    assert res.market_source == "closing_b365"
    assert "CLOSING" in res.market_label
    assert res.market_label in res.summary()
    assert res.n_with_market > 0


def test_backtest_falls_back_to_preclosing_and_says_so():
    m = _synthetic_matches()
    m["b365h"], m["b365d"], m["b365a"] = 2.0, 3.5, 4.0
    res = walk_forward_backtest(m, holdout_start=m["date"].quantile(0.7), refit_every=1000)
    assert res.market_source == "preclosing_b365"
    assert "PRE-CLOSING" in res.summary()


def test_backtest_without_any_odds_still_scores_the_model():
    m = _synthetic_matches()
    res = walk_forward_backtest(m, holdout_start=m["date"].quantile(0.7), refit_every=1000)
    assert res.market_source is None
    assert res.market_log_loss is None
    assert res.model_log_loss > 0
