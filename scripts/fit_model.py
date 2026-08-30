#!/usr/bin/env python
"""Fit one Dixon-Coles model PER LEAGUE and save to artifacts/{league}/.

Models are never pooled across leagues. A single fit assumes one shared pool
of teams with a common scoring level, one home-advantage multiplier and one
rho; the Premier League and La Liga share none of those and never play each
other, so a pooled fit would rank a Spanish team against an English one on
the basis of nothing but their leagues' goal averages.
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from hattrick.config import (
    league_codes, league_name, load_config, model_artifact_path, resolve_path,
)
from hattrick.model.dixon_coles import DixonColesModel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-xg", action="store_true",
                    help="Fit the attack/defense rates on expected goals instead of actual "
                         "goals, falling back to goals per-match where xG is missing. "
                         "Requires scripts/update_xg.py to have run. rho is still estimated "
                         "from actual scorelines.")
    ap.add_argument("--promoted-prior", action="store_true",
                    help="Shrink newly-promoted teams toward a promoted-side rating rather "
                         "than the league mean. OFF by default: the effect is real and "
                         "large in the raw data (promoted sides average 0.91 points per "
                         "game against 1.45), and it visibly removes absurd ratings for "
                         "unknown teams, but it did NOT improve out-of-sample forecasts "
                         "when measured. See the README before enabling.")
    ap.add_argument("--league", action="append",
                    help="Fit only this league code (repeatable). Default: all configured.")
    args = ap.parse_args()

    cfg = load_config()
    matches_path = resolve_path(cfg["data"]["processed_dir"]) / "matches.csv"
    if not matches_path.exists():
        raise SystemExit(f"{matches_path} not found - run scripts/update_data.py first.")
    matches = pd.read_csv(matches_path, parse_dates=["date"])

    if "league" not in matches.columns:
        raise SystemExit(
            f"{matches_path} has no 'league' column, so its rows cannot be separated by "
            f"league and fitting them together would pool leagues into one model. "
            f"Re-run scripts/update_data.py to rebuild it."
        )

    target = "xg" if args.use_xg else "goals"
    codes = args.league or league_codes(cfg)
    fitted = 0

    for code in codes:
        sub = matches[matches["league"] == code]
        if sub.empty:
            print(f"{code}: no matches in {matches_path.name}; skipping.")
            continue

        print(f"\n=== {league_name(cfg, code)} ({code}) ===")
        print(f"Fitting on {len(sub)} matches "
              f"({sub['date'].min().date()} to {sub['date'].max().date()}), target={target} ...")
        model = DixonColesModel(xi=cfg["model"]["xi"], target=target,
                                promoted_teams="auto" if args.promoted_prior else None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(sub)
        for w in caught:
            print(f"  WARNING: {w.message}")

        p = model.params
        if args.promoted_prior:
            print(f"  promoted prior applied to: {p.promoted or 'no teams'}")
        if args.use_xg:
            print(f"  xG coverage: {p.xg_coverage:.1%} (remainder fell back to actual goals)")
        print(f"  home_advantage(gamma)={p.home_advantage:.3f}  rho={p.rho:.4f}  "
              f"log-likelihood={p.log_likelihood:.1f}")
        top = sorted(p.attack.items(), key=lambda kv: -kv[1])[:5]
        print("  Top 5 attack:", ", ".join(f"{t}={v:.2f}" for t, v in top))

        out = model_artifact_path(cfg, code)
        out.parent.mkdir(parents=True, exist_ok=True)
        p.to_json(str(out))
        print(f"  Saved -> {out.relative_to(resolve_path('.'))}")
        fitted += 1

    if not fitted:
        raise SystemExit("No leagues were fitted - check config.yaml and matches.csv.")
    print(f"\nFitted {fitted} league model(s).")


if __name__ == "__main__":
    main()
