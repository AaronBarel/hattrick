"""Backtesting: walk-forward evaluation of the Dixon-Coles model against
actual results and, where available, the bookmaker's closing line.

Three things worth knowing before reading the numbers:

0. PRE-CLOSING IS NOT CLOSING. football-data.co.uk's `B365H/D/A` columns
   (loaded as `b365h/d/a`) are collected days before kickoff, not at
   kickoff; the true closing prices are the separate "C"-infixed columns
   (`b365ch/cd/ca`, `psch/pscd/psca`, `avgch/avgcd/avgca`). Benchmarking
   against the pre-closing line overstates the model, because a stale price
   is an easier target than the closing one. `walk_forward_backtest`
   therefore defaults to the sharpest *closing* set actually present in the
   data, and `BacktestResult.market_label` records which one it used - read
   that label before quoting any "beats the market" claim.

1. Log loss and Brier score are proper scoring rules for probabilistic
   classification - "proper" meaning a forecaster can't improve their
   expected score by reporting anything other than their true belief, unlike
   e.g. plain accuracy (which rewards confident guessing). Lower is better
   for both, and both are computed multiclass over {home_win, draw, away_win}.

2. Bookmaker odds embed a margin (the "overround"): 1/H + 1/D + 1/A sums to
   more than 1. We remove it (divide each implied probability by the sum)
   before comparing, so it's apples-to-apples - our probabilities against the
   bookmaker's fair, de-vigged probabilities rather than their quoted price.
   This is also the honest benchmark: beating the RAW quoted price is a much
   lower bar than beating the fair-value estimate behind it, and only the
   latter measures genuine forecasting skill.

Walk-forward, not a single train/test split: fit once on everything before
`holdout_start`, then step through the holdout period match by match,
periodically refitting on newly completed results (every `refit_every`
matches) so later predictions benefit from in-season form - closer to how
this system would actually be run live than a single static fit.

One league at a time: `walk_forward_backtest` fits a single model, so it must
be handed one league's matches. `backtest_by_league` runs it per league and
pools the per-match scores. Pooling the *fits* would put teams that never
meet on one rating scale - see `hattrick.config` for why that is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hattrick.model.dixon_coles import DixonColesModel
from hattrick.model.scorelines import score_matrix, outcome_probs

OUTCOMES = ["home_win", "draw", "away_win"]


def log_loss(probs: list[dict], actual: list[str]) -> float:
    eps = 1e-12
    losses = [-np.log(max(p[a], eps)) for p, a in zip(probs, actual)]
    return float(np.mean(losses))


def brier_score(probs: list[dict], actual: list[str]) -> float:
    scores = []
    for p, a in zip(probs, actual):
        s = sum((p[o] - (1.0 if o == a else 0.0)) ** 2 for o in OUTCOMES)
        scores.append(s)
    return float(np.mean(scores))


def per_match_log_loss(probs: list[dict], actual: list[str]) -> np.ndarray:
    eps = 1e-12
    return np.array([-np.log(max(p[a], eps)) for p, a in zip(probs, actual)])


def per_match_brier(probs: list[dict], actual: list[str]) -> np.ndarray:
    return np.array([
        sum((p[o] - (1.0 if o == a else 0.0)) ** 2 for o in OUTCOMES)
        for p, a in zip(probs, actual)
    ])


@dataclass
class BootstrapTestResult:
    n_matches: int
    observed_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    alpha: float
    significant: bool

    def summary(self, label_a: str = "A", label_b: str = "B") -> str:
        better = label_a if self.observed_diff < 0 else label_b
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (f"{label_a} - {label_b} mean diff: {self.observed_diff:+.4f} "
                f"([{self.ci_low:+.4f}, {self.ci_high:+.4f}] CI), "
                f"p={self.p_value:.3f} -> {verdict}; {better} better (n={self.n_matches})")


def paired_bootstrap_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10_000,
    alpha: float = 0.05, seed: int | None = None,
) -> BootstrapTestResult:
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")
    if len(scores_a) == 0:
        raise ValueError("scores must not be empty")
    rng = np.random.default_rng(seed)
    diffs = scores_a - scores_b
    boot_diffs = diffs[rng.integers(0, len(diffs), size=(n_boot, len(diffs)))].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = min(1.0, float(2 * min((boot_diffs >= 0).mean(), (boot_diffs <= 0).mean())))
    return BootstrapTestResult(
        n_matches=len(diffs), observed_diff=float(diffs.mean()), ci_low=float(ci_low),
        ci_high=float(ci_high), p_value=p_value, alpha=alpha,
        significant=not (ci_low <= 0 <= ci_high),
    )


def calibration_curve(
    probs: list[dict], actual: list[str], outcome: str, n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    predicted = np.array([p[outcome] for p in probs])
    observed = np.array([a == outcome for a in actual], dtype=float)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(predicted, np.linspace(0, 1, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    bin_idx = np.digitize(predicted, edges[1:-1], right=True)
    rows = []
    for index in range(len(edges) - 1):
        mask = bin_idx == index
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lower": edges[index], "bin_upper": edges[index + 1],
            "mean_predicted": float(predicted[mask].mean()),
            "empirical_frequency": float(observed[mask].mean()),
            "n_matches": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    probs: list[dict], actual: list[str], outcome: str, n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    curve = calibration_curve(probs, actual, outcome, n_bins, strategy)
    if curve.empty:
        return float("nan")
    weights = curve["n_matches"] / curve["n_matches"].sum()
    return float((weights * (curve["mean_predicted"] - curve["empirical_frequency"]).abs()).sum())


def devig_bookmaker_probs(home_odds, draw_odds, away_odds) -> dict[str, float] | None:
    """Decimal odds -> de-vigged (overround-removed) probabilities, or None
    if any odds are missing/invalid (common for older/lower-division data).

    Uses the multiplicative ("proportional") de-vig: divide each implied
    probability by their sum. This is the standard simple choice, and it is
    what we want here, but note it is not the only one - it assumes the
    bookmaker's margin is spread proportionally across outcomes. In practice
    margin is applied slightly more heavily to longshots (the favorite-
    longshot bias), so proportional de-vigging mildly overstates longshot
    probabilities. Shin's method corrects for this; we do not implement it,
    because the effect is small relative to the model differences we are
    trying to detect here, and because using the same de-vig on both sides
    of every comparison keeps it apples-to-apples.
    """
    vals = (home_odds, draw_odds, away_odds)
    if any(v is None or (isinstance(v, float) and np.isnan(v)) or v <= 1.0 for v in vals):
        return None
    raw = {"home_win": 1.0 / home_odds, "draw": 1.0 / draw_odds, "away_win": 1.0 / away_odds}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


# Candidate market benchmarks, best first. "Best" means sharpest: Pinnacle is
# a low-margin, high-limit book that accepts sharp action, so its closing
# line is the strongest freely available single-book estimate; the
# cross-book closing average is next; Bet365's own closing line next; and
# the PRE-closing snapshot is a clearly-labeled last resort that should
# never be quoted as "the closing line".
MARKET_SOURCES: dict[str, tuple[tuple[str, str, str], str]] = {
    "closing_pinnacle": (("psch", "pscd", "psca"), "de-vigged Pinnacle CLOSING odds"),
    "closing_avg": (("avgch", "avgcd", "avgca"), "de-vigged cross-book average CLOSING odds"),
    "closing_b365": (("b365ch", "b365cd", "b365ca"), "de-vigged Bet365 CLOSING odds"),
    "preclosing_b365": (("b365h", "b365d", "b365a"),
                         "de-vigged Bet365 PRE-CLOSING odds (collected days before kickoff - "
                         "a weaker benchmark than the closing line)"),
}

MARKET_SOURCE_PREFERENCE = ["closing_pinnacle", "closing_avg", "closing_b365", "preclosing_b365"]


def resolve_market_source(matches: pd.DataFrame, market_source: str = "auto",
                          min_coverage: float = 0.9) -> tuple[str, tuple[str, str, str], str] | None:
    """Pick which odds column set to benchmark against.

    With ``market_source="auto"``, walk MARKET_SOURCE_PREFERENCE and take the
    first set whose columns are present and non-null for at least
    ``min_coverage`` of rows. The coverage floor matters: Pinnacle columns
    exist but go blank partway through some recent seasons, and silently
    benchmarking on the surviving 55% of matches would compare the model
    against a non-random subsample. Falling through to a fully-covered set
    is the honest choice.

    Returns ``(name, (home_col, draw_col, away_col), human_label)``, or None
    if no source qualifies.
    """
    if market_source != "auto":
        if market_source not in MARKET_SOURCES:
            raise ValueError(f"Unknown market_source {market_source!r}; "
                             f"expected 'auto' or one of {sorted(MARKET_SOURCES)}")
        cols, label = MARKET_SOURCES[market_source]
        if not set(cols).issubset(matches.columns):
            return None
        return market_source, cols, label

    n = len(matches)
    for name in MARKET_SOURCE_PREFERENCE:
        cols, label = MARKET_SOURCES[name]
        if not set(cols).issubset(matches.columns):
            continue
        if n and matches[list(cols)].notna().all(axis=1).sum() / n >= min_coverage:
            return name, cols, label
    return None


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


@dataclass
class BacktestResult:
    n_matches: int
    model_log_loss: float
    model_brier: float
    n_with_market: int
    market_log_loss: float | None
    market_brier: float | None
    per_match: pd.DataFrame
    # Which odds snapshot the market columns came from, e.g. "closing_avg".
    # None when no odds were available. Always report this alongside any
    # model-vs-market claim: beating the pre-closing line is a much weaker
    # result than beating the closing line.
    market_source: str | None = None
    market_label: str | None = None

    def model_probs_list(self, df: pd.DataFrame | None = None) -> list[dict]:
        df = self.per_match if df is None else df
        return [{"home_win": r.model_home_win, "draw": r.model_draw, "away_win": r.model_away_win}
                for r in df.itertuples()]

    def market_probs_list(self, df: pd.DataFrame | None = None) -> list[dict]:
        df = self.per_match if df is None else df
        return [{"home_win": r.market_home_win, "draw": r.market_draw, "away_win": r.market_away_win}
                for r in df.itertuples()]

    def significance_vs_market(self, metric: str = "log_loss", n_boot: int = 10_000,
                               alpha: float = 0.05, seed: int | None = None) -> BootstrapTestResult | None:
        with_market = self.per_match.dropna(subset=["market_home_win", "market_draw", "market_away_win"])
        if with_market.empty:
            return None
        actuals = with_market["actual"].tolist()
        model_probs = self.model_probs_list(with_market)
        market_probs = self.market_probs_list(with_market)
        score_fn = per_match_log_loss if metric == "log_loss" else per_match_brier
        return paired_bootstrap_test(score_fn(model_probs, actuals), score_fn(market_probs, actuals),
                                     n_boot=n_boot, alpha=alpha, seed=seed)

    def calibration(self, outcome: str = "home_win", n_bins: int = 10,
                    strategy: str = "uniform", source: str = "model") -> pd.DataFrame:
        df = self.per_match
        if source == "model":
            probs = self.model_probs_list(df)
        elif source == "market":
            df = df.dropna(subset=["market_home_win", "market_draw", "market_away_win"])
            probs = self.market_probs_list(df)
        else:
            raise ValueError("source must be 'model' or 'market'")
        return calibration_curve(probs, df["actual"].tolist(), outcome, n_bins, strategy)

    def summary(self) -> str:
        lines = [
            f"Matches evaluated: {self.n_matches}",
            f"Model:   log loss={self.model_log_loss:.4f}  Brier={self.model_brier:.4f}",
        ]
        if self.market_log_loss is not None:
            lines.append(
                f"Market:  log loss={self.market_log_loss:.4f}  Brier={self.market_brier:.4f} "
                f"({self.market_label or 'unknown odds source'}, n={self.n_with_market})"
            )
            delta = self.model_log_loss - self.market_log_loss
            verdict = "model beats market" if delta < 0 else "market beats model"
            lines.append(f"Delta (model - market) log loss: {delta:+.4f}  -> {verdict}")
            significance = self.significance_vs_market()
            if significance is not None:
                lines.append(significance.summary("model", "market"))
        else:
            lines.append("No bookmaker odds available in this dataset for a market comparison.")
        return "\n".join(lines)


def walk_forward_backtest(
    matches: pd.DataFrame,
    holdout_start,
    xi: float = 0.0018,
    refit_every: int = 10,
    max_goals: int = 10,
    market_source: str = "auto",
    model_kwargs: dict | None = None,
) -> BacktestResult:
    """Walk-forward backtest against actual results and the closing line.

    `model_kwargs` is forwarded to every DixonColesModel built here (both the
    initial fit and each walk-forward refit), so feature ablations can vary
    the model without duplicating this loop.
    """
    if "league" in matches.columns and matches["league"].nunique() > 1:
        raise ValueError(
            f"walk_forward_backtest got {matches['league'].nunique()} leagues; it fits one "
            f"model, so pass one league at a time (see backtest_by_league)."
        )
    matches = matches.sort_values("date").reset_index(drop=True)
    holdout_start = pd.Timestamp(holdout_start)
    train = matches[matches["date"] < holdout_start].copy()
    holdout = matches[matches["date"] >= holdout_start].reset_index(drop=True)
    if train.empty or holdout.empty:
        raise ValueError("holdout_start leaves an empty train or holdout set - check the date.")

    model_kwargs = dict(model_kwargs or {})
    model = DixonColesModel(xi=xi, **model_kwargs)
    model.fit(train)

    # Resolve the benchmark on the HOLDOUT rows: coverage in the training
    # period is irrelevant, since only holdout matches are ever scored.
    resolved = resolve_market_source(holdout, market_source)
    if resolved is None:
        market_name = market_cols = market_label = None
    else:
        market_name, market_cols, market_label = resolved

    records = []

    for i, row in holdout.iterrows():
        home, away = row["home"], row["away"]
        if home not in model.params.attack or away not in model.params.attack:
            continue  # team the model has never seen - can't predict it, skip

        M = score_matrix(model, home, away, max_goals)
        model_probs = outcome_probs(M)
        actual = actual_outcome(row["home_goals"], row["away_goals"])

        market_probs = None
        if market_cols is not None:
            market_probs = devig_bookmaker_probs(*(row.get(c) for c in market_cols))

        records.append({
            "date": row["date"], "home": home, "away": away, "actual": actual,
            "model_home_win": model_probs["home_win"], "model_draw": model_probs["draw"],
            "model_away_win": model_probs["away_win"],
            "market_home_win": market_probs["home_win"] if market_probs else np.nan,
            "market_draw": market_probs["draw"] if market_probs else np.nan,
            "market_away_win": market_probs["away_win"] if market_probs else np.nan,
        })

        # Walk forward: fold this now-completed match into training data and
        # periodically refit, so later predictions see newer form.
        if (i + 1) % refit_every == 0:
            running_train = pd.concat([train, holdout.iloc[:i + 1]], ignore_index=True)
            model = DixonColesModel(xi=xi, **model_kwargs)
            model.fit(running_train)

    per_match = pd.DataFrame(records)
    if per_match.empty:
        raise ValueError("No holdout matches could be evaluated (all teams unseen in training data).")

    return _summarize(per_match, market_name, market_label)


def _summarize(per_match: pd.DataFrame, market_source: str | None,
               market_label: str | None) -> BacktestResult:
    model_probs_list = [{"home_win": r.model_home_win, "draw": r.model_draw, "away_win": r.model_away_win}
                         for r in per_match.itertuples()]
    actuals = per_match["actual"].tolist()
    model_ll = log_loss(model_probs_list, actuals)
    model_brier = brier_score(model_probs_list, actuals)

    market_ll = market_brier = None
    with_market = per_match.dropna(subset=["market_home_win", "market_draw", "market_away_win"])
    if len(with_market) > 0:
        market_probs_list = [{"home_win": r.market_home_win, "draw": r.market_draw, "away_win": r.market_away_win}
                              for r in with_market.itertuples()]
        market_actuals = with_market["actual"].tolist()
        market_ll = log_loss(market_probs_list, market_actuals)
        market_brier = brier_score(market_probs_list, market_actuals)

    return BacktestResult(
        n_matches=len(per_match), model_log_loss=model_ll, model_brier=model_brier,
        n_with_market=len(with_market), market_log_loss=market_ll, market_brier=market_brier,
        per_match=per_match, market_source=market_source, market_label=market_label,
    )


def backtest_by_league(matches: pd.DataFrame, holdout_start, **kwargs) -> dict[str, BacktestResult]:
    """Run `walk_forward_backtest` separately for each league.

    Returns `{league_code: BacktestResult}` plus an `"ALL"` entry whose scores
    are pooled at the per-match level across leagues - every match is still
    predicted by its own league's model. The pooled entry's market label is
    set only when every league used the same odds source.
    """
    out = {}
    for code, sub in matches.groupby("league", sort=False):
        res = walk_forward_backtest(sub, holdout_start, **kwargs)
        res.per_match.insert(0, "league", code)
        out[code] = res
    if not out:
        raise ValueError("No leagues to backtest.")
    sources = {(r.market_source, r.market_label) for r in out.values()}
    source, label = sources.pop() if len(sources) == 1 else (
        "mixed", "mixed odds sources - see each league's label")
    out["ALL"] = _summarize(pd.concat([r.per_match for r in out.values()], ignore_index=True),
                            source, label)
    return out
