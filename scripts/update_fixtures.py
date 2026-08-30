#!/usr/bin/env python
"""Fetch the full season fixture schedule for every configured league.

football-data.co.uk's fixtures.csv only covers a rolling few days and can be
entirely already-played right after a matchweek. Understat publishes the
whole season up front, so this gives the dashboard something to show and the
season simulator a real remaining-fixture list.

Optional: if this fails, load_fixtures falls back to football-data.co.uk's
rolling file.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from hattrick.config import league_codes, load_config, resolve_path
from hattrick.data.fixtures import current_season_code, fetch_understat_fixtures

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default=None,
                    help="Season code (e.g. 2627). Defaults to the season in progress.")
    args = ap.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    codes = league_codes(cfg)
    season = args.season or current_season_code()

    # Map Understat club names onto the spellings the fitted models use, which
    # come from football-data.co.uk. Without this the dashboard would show
    # fixtures whose teams the model cannot look up.
    names = {}
    matches_path = resolve_path(cfg["data"]["processed_dir"]) / "matches.csv"
    if matches_path.exists():
        m = pd.read_csv(matches_path, usecols=["league", "home", "away"])
        for code in codes:
            sub = m[m["league"] == code]
            if len(sub):
                names[code] = sorted(set(sub["home"]) | set(sub["away"]))

    print(f"Fetching {season} schedule for {codes} ...")
    try:
        fx = fetch_understat_fixtures(codes, season=season,
                                      team_names_by_league=names, raw_dir=raw_dir)
    except Exception as exc:
        print(f"\nFixture fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Falling back to football-data.co.uk's rolling fixtures.csv, which may "
              "contain only already-played matches.", file=sys.stderr)
        raise SystemExit(1)

    for code, why in (fx.attrs.get("skipped") or {}).items():
        print(f"  {code}: skipped - {why}")

    upcoming = fx[~fx["is_result"]]
    print(f"Wrote {len(fx)} fixtures ({len(upcoming)} still to play) -> "
          f"{raw_dir / 'fixtures_full.csv'}")
    for code in codes:
        sub = upcoming[upcoming["league"] == code]
        if len(sub):
            print(f"  {code}: {len(sub)} upcoming, next {sub['kickoff'].min()}")
        elif code not in (fx.attrs.get("skipped") or {}):
            print(f"  {code}: no upcoming fixtures")


if __name__ == "__main__":
    main()
