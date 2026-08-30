#!/usr/bin/env python
"""Fetch match xG and merge it onto the processed matches dataset.

Run after scripts/update_data.py. This is an OPTIONAL enhancement: if it
fails, the pipeline still works and the model simply fits on actual goals.

  uv run scripts/update_xg.py                  # scrape Understat via soccerdata
  uv run scripts/update_xg.py --csv my_xg.csv  # use a CSV instead of scraping

The CSV path expects columns: date, home, away, home_xg, away_xg. Use it when
Understat is unreachable or you have a better source (FBref export, StatsBomb
open data, a paid feed).
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from hattrick.config import load_config, resolve_path
from hattrick.data.xg import fetch_understat_xg, load_xg_csv, merge_xg

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Standardized xG CSV to use instead of scraping Understat.")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="Fail loudly if xG covers less of the dataset than this (default 0.5). "
                         "Low coverage is worth noticing: the model would silently fit most "
                         "matches on goals while reporting an xG fit.")
    args = ap.parse_args()

    cfg = load_config()
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    matches_path = processed_dir / "matches.csv"
    if not matches_path.exists():
        raise SystemExit(f"{matches_path} not found - run scripts/update_data.py first.")
    matches = pd.read_csv(matches_path, parse_dates=["date"])

    league_codes = [lg["code"] for lg in cfg["leagues"]]
    seasons = cfg["seasons"]

    if args.csv:
        print(f"Loading xG from {args.csv} ...")
        xg = load_xg_csv(args.csv)
    else:
        print(f"Fetching Understat xG for leagues={league_codes} seasons={seasons} ...")
        try:
            xg = fetch_understat_xg(league_codes, seasons)
        except Exception as exc:
            print(f"\nUnderstat fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("This is a scrape of a site with no API contract, so this can happen.\n"
                  "The pipeline still works without xG - fit_model.py will use actual\n"
                  "goals. To supply xG another way, export a CSV with columns\n"
                  "date,home,away,home_xg,away_xg and re-run with --csv.", file=sys.stderr)
            raise SystemExit(1)

    merged, report = merge_xg(matches, xg)
    print(report.summary())

    if report.coverage < args.min_coverage:
        raise SystemExit(
            f"\nCoverage {report.coverage:.1%} is below --min-coverage "
            f"{args.min_coverage:.0%}; not overwriting {matches_path}. "
            f"Fix the club-name mapping or lower the threshold deliberately."
        )

    merged.to_csv(matches_path, index=False)
    print(f"Wrote {matches_path} with home_xg/away_xg. "
          f"Fit with: uv run scripts/fit_model.py --use-xg")


if __name__ == "__main__":
    main()
