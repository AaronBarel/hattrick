"""Model comparison harness: run several model configurations over one or
more leagues and report which differences are real.

Why this exists as a module rather than as notebook cells: the interesting
part of "did this feature help?" is the *significance test*, and getting that
subtly wrong (comparing on different match sets, pooling losses that came
from different holdouts, reading a point estimate as a result) is easy and
invisible. Notebook cells that do it get copy-pasted and drift. This is the
one implementation, and it is unit-tested.

The single rule this enforces: two configurations are only ever compared on
matches that BOTH of them scored. A model that skips more matches - because
it saw fewer teams in training, say - would otherwise get to be judged on an
easier subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hattrick.evaluation import (
    per_match_brier,
    per_match_log_loss,
    paired_bootstrap_test,
    walk_forward_backtest,
)


def synthetic_league(n_teams=20, rounds=4, seed=0, xg_noise=0.15, start="2022-08-01"):
    """A league with known ratings, goals, xG and a bookmaker line.

    Exists so the notebooks run standalone with no downloads. The generative
    story matches what the model assumes: xG is a low-noise reading of the
    same lambda that goals are a Poisson draw from, and the "market" is a
    de-vigged version of the true probabilities with a little noise and a
    5% overround, so it is a genuinely strong benchmark rather than a
    strawman.
    """
    from hattrick.model.scorelines import outcome_probs, score_matrix

    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(n_teams)]
    atk = dict(zip(teams, rng.lognormal(0, 0.22, n_teams)))
    dfn = dict(zip(teams, rng.lognormal(0, 0.22, n_teams)))
    scale = float(np.mean(list(atk.values())))
    atk = {k: v / scale for k, v in atk.items()}
    dfn = {k: v * scale for k, v in dfn.items()}
    gamma, rows, day = 1.35, [], 0

    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                lam, mu = atk[h] * dfn[a] * gamma, atk[a] * dfn[h]
                x, y = rng.poisson(lam), rng.poisson(mu)
                ph = 1.0 / (1.0 + np.exp(-(np.log(lam / mu))))
                p = {"home_win": 0.46 * ph + 0.10, "draw": 0.26,
                     "away_win": 0.46 * (1 - ph) + 0.10}
                tot = sum(p.values())
                p = {k: v / tot for k, v in p.items()}
                over = 1.05
                rows.append({
                    "date": pd.Timestamp(start) + pd.Timedelta(days=day),
                    "league": "SYN", "home": h, "away": a,
                    "home_goals": x, "away_goals": y,
                    "home_xg": max(lam * rng.lognormal(0, xg_noise), 1e-3),
                    "away_xg": max(mu * rng.lognormal(0, xg_noise), 1e-3),
                    "b365ch": over / p["home_win"], "b365cd": over / p["draw"],
                    "b365ca": over / p["away_win"],
                })
    return pd.DataFrame(rows)


def load_leagues(league_codes=None, seasons=None, with_xg=True, verbose=True):
    """Load real league data with xG attached, or fall back to synthetic.

    Returns `(matches_by_league, is_real)`. Every step degrades rather than
    raising, because the point of the notebooks is that they run for someone
    who has just cloned the repo and downloaded nothing.
    """
    try:
        from hattrick.config import load_config, resolve_path
        from hattrick.data.loader import build_matches_dataset

        cfg = load_config()
        league_codes = league_codes or [lg["code"] for lg in cfg["leagues"]]
        seasons = seasons or cfg["seasons"]
        raw_dir = resolve_path(cfg["data"]["raw_dir"])
        proc_dir = resolve_path(cfg["data"]["processed_dir"])

        # Prefer the already-built matches.csv. Rebuilding here would be
        # both slower and destructive: build_matches_dataset writes that
        # file, so calling it once per league would leave it holding only
        # the last league.
        processed = proc_dir / "matches.csv"
        if processed.exists():
            allm = pd.read_csv(processed, parse_dates=["date"])
        else:
            allm = build_matches_dataset(raw_dir, proc_dir, league_codes, seasons,
                                         write=False)

        has_xg = {"home_xg", "away_xg"}.issubset(allm.columns)
        out = {}
        for code in league_codes:
            m = allm[allm["league"] == code].reset_index(drop=True)
            if m.empty:
                continue
            if with_xg and not has_xg:
                try:
                    from hattrick.data.xg import fetch_understat_xg, merge_xg
                    m, rep = merge_xg(m, fetch_understat_xg([code], seasons))
                    cov = f"xG coverage {rep.coverage:.1%}"
                except Exception as exc:
                    cov = f"xG unavailable ({type(exc).__name__}) - falling back to goals"
            elif with_xg:
                cov = f"xG coverage {m['home_xg'].notna().mean():.1%} (from matches.csv)"
            else:
                cov = "goals only"
            if verbose:
                print(f"{code}: {len(m)} matches, {cov}")
            out[code] = m
        if out:
            return out, True
    except Exception as exc:
        if verbose:
            print(f"Real data unavailable ({type(exc).__name__}: {exc}).")

    if verbose:
        print("Falling back to synthetic data. To use real data, run:\n"
              "    uv run scripts/update_data.py && uv run scripts/update_xg.py")
    return {"SYN": synthetic_league()}, False


@dataclass
class ModelConfig:
    """One model variant to evaluate."""
    name: str
    xi: float = 0.0018
    model_kwargs: dict = field(default_factory=dict)

    def run(self, matches, holdout_start, **kw):
        return walk_forward_backtest(matches, holdout_start=holdout_start, xi=self.xi,
                                     model_kwargs=self.model_kwargs, **kw)


def _key(per_match):
    return list(zip(per_match["date"].astype("datetime64[ns]").astype(np.int64),
                    per_match["home"], per_match["away"]))


def align(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every result to the fixtures all of them scored.

    Without this, a config that silently skipped 30 hard matches would look
    better than one that attempted them.
    """
    common = None
    for df in results.values():
        ks = set(_key(df))
        common = ks if common is None else (common & ks)
    out = {}
    for name, df in results.items():
        mask = [k in common for k in _key(df)]
        out[name] = df.loc[mask].sort_values(["date", "home", "away"]).reset_index(drop=True)
    return out


def compare_configs(
    matches_by_league: dict[str, pd.DataFrame],
    configs: list[ModelConfig],
    holdout_start,
    baseline: str | None = None,
    metric: str = "log_loss",
    n_boot: int = 10_000,
    seed: int = 0,
    market_source: str = "auto",
    refit_every: int = 10,
) -> pd.DataFrame:
    """Evaluate every config on every league; return one tidy row per pair.

    Losses are pooled across leagues at the *per-match* level rather than by
    averaging each league's mean, so a 300-match league does not get the same
    weight as a 380-match one.

    `baseline` names the config every other config is tested against. The
    market is reported alongside but is not the baseline by default: the
    question a feature ablation answers is "did this change help *my model*",
    which is a different question from "does my model beat the closing line".
    """
    score = per_match_log_loss if metric == "log_loss" else per_match_brier
    baseline = baseline or configs[0].name

    pooled: dict[str, list] = {c.name: [] for c in configs}
    market_pooled: list = []
    per_league_rows = []

    for league, matches in matches_by_league.items():
        results, market = {}, None
        for cfg in configs:
            res = cfg.run(matches, holdout_start, market_source=market_source,
                          refit_every=refit_every)
            results[cfg.name] = res.per_match
            if market is None:
                market = (res.market_source, res.market_label)
        results = align(results)

        actual = results[baseline]["actual"].tolist()
        for name, df in results.items():
            probs = [{"home_win": r.model_home_win, "draw": r.model_draw,
                      "away_win": r.model_away_win} for r in df.itertuples()]
            s = score(probs, actual)
            pooled[name].append(s)
            per_league_rows.append({"league": league, "config": name,
                                    "n": len(df), metric: float(np.mean(s))})

        mdf = results[baseline].dropna(subset=["market_home_win"])
        if len(mdf):
            mprobs = [{"home_win": r.market_home_win, "draw": r.market_draw,
                       "away_win": r.market_away_win} for r in mdf.itertuples()]
            market_pooled.append(score(mprobs, mdf["actual"].tolist()))

    rows = []
    base = np.concatenate(pooled[baseline])
    for cfg in configs:
        vals = np.concatenate(pooled[cfg.name])
        row = {"config": cfg.name, "n_matches": len(vals), metric: float(vals.mean()),
               "vs_baseline": float(vals.mean() - base.mean())}
        if cfg.name != baseline:
            t = paired_bootstrap_test(vals, base, n_boot=n_boot, seed=seed)
            row.update({"ci_low": t.ci_low, "ci_high": t.ci_high, "p_value": t.p_value,
                        "significant": bool(t.significant)})
        else:
            row.update({"ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                        "significant": False})
        rows.append(row)

    out = pd.DataFrame(rows)
    if market_pooled:
        mv = np.concatenate(market_pooled)
        out.attrs["market_score"] = float(mv.mean())
        out.attrs["market_n"] = int(len(mv))
    out.attrs["per_league"] = pd.DataFrame(per_league_rows)
    out.attrs["baseline"] = baseline
    out.attrs["metric"] = metric
    return out


def format_comparison(df: pd.DataFrame) -> str:
    """Render a comparison table with the honest reading spelled out."""
    metric = df.attrs.get("metric", "log_loss")
    base = df.attrs.get("baseline")
    lines = [f"Baseline: {base}   (metric: {metric}, lower is better)"]
    if "market_score" in df.attrs:
        lines.append(f"Market benchmark: {df.attrs['market_score']:.4f} "
                     f"(n={df.attrs['market_n']})")
    lines.append("")
    for r in df.itertuples():
        if r.config == base:
            lines.append(f"  {r.config:<28s} {getattr(r, metric):.4f}   (baseline, n={r.n_matches})")
            continue
        verdict = ("SIGNIFICANT" if r.significant
                   else "not distinguishable from noise")
        direction = "better" if r.vs_baseline < 0 else "worse"
        lines.append(
            f"  {r.config:<28s} {getattr(r, metric):.4f}   {r.vs_baseline:+.4f} "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] p={r.p_value:.3f}  "
            f"-> {direction}, {verdict}")
    return "\n".join(lines)
