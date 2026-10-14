#!/usr/bin/env python
"""Monte Carlo simulate the rest of the season using the fitted model.

Remaining fixtures come from `load_fixtures`, which uses the full-season
schedule written by scripts/update_fixtures.py (Understat) when it exists.
If that step has not run or failed for a league, it falls back to
football-data.co.uk's rolling fixtures.csv, which covers only the next few
days: the simulation then treats the table as far more settled than it is,
and this script says so. Pass --remaining-fixtures to supply your own
schedule instead.

Usage:
    python scripts/simulate_season.py
    python scripts/simulate_season.py --remaining-fixtures my_schedule.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from hattrick.config import (
    current_season_start, league_codes, league_name, load_config,
    load_league_models, resolve_path, sim_artifact_path,
)
from hattrick.data.loader import load_fixtures
from hattrick.model.season_sim import simulate_season


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remaining-fixtures", type=str, default=None,
                         help="CSV with columns date,home,away covering the full remaining season")
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--league", action="append",
                         help="League code to simulate (repeatable). Default: all configured. "
                              "A season table is per-league by construction, so each is "
                              "simulated separately with its own fitted model.")
    args = parser.parse_args()

    cfg = load_config()
    models = load_league_models(cfg)
    if not models:
        raise SystemExit("No fitted league models found - run scripts/fit_model.py first.")

    matches_path = resolve_path(cfg["data"]["processed_dir"]) / "matches.csv"
    all_matches_full = pd.read_csv(matches_path, parse_dates=["date"])

    for code in (args.league or league_codes(cfg)):
        model = models.get(code)
        if model is None:
            print(f"No fitted model for {code} - skipping.")
            continue
        simulate_one(cfg, args, code, model, all_matches_full)


def simulate_one(cfg, args, code, model, all_matches_full):
    league = {"code": code, "name": league_name(cfg, code)}
    all_matches = all_matches_full[all_matches_full["league"] == code]

    season_start = current_season_start(cfg)
    played = all_matches[all_matches["date"] >= season_start]

    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    if args.remaining_fixtures:
        remaining = pd.read_csv(args.remaining_fixtures, parse_dates=["date"])
    else:
        remaining = load_fixtures(raw_dir, league["code"])
        if remaining.attrs.get("source") == "rolling":
            print(f"NOTE: no full-season schedule for {league['name']}; simulating only the "
                  f"near-term fixtures in fixtures.csv, so the table will look more settled "
                  f"than it is. Run scripts/update_fixtures.py for the full schedule.\n")

    if played.empty and remaining.empty:
        print(f"No played or remaining fixtures for {league['name']} ({code}) - skipping.")
        return

    print(f"\nSimulating {league['name']}: {len(played)} played, {len(remaining)} remaining "
          f"(as of {season_start.date()}) ...")

    n_sims = args.n_sims or cfg["simulation"]["n_sims"]
    result = simulate_season(model, played, remaining, n_sims=n_sims,
                              max_goals=cfg["model"]["max_goals"], seed=42)

    df = result.summary_table()
    print(f"\n{n_sims:,} simulations:\n")
    print(df.to_string(index=False))

    artifact_out = sim_artifact_path(cfg, code)
    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(artifact_out, orient="records", indent=2)
    print(f"\nSaved -> {artifact_out}")


if __name__ == "__main__":
    main()
