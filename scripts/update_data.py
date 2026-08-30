#!/usr/bin/env python
"""Download latest results + fixtures configured in config.yaml."""
from __future__ import annotations
import logging
from hattrick.config import load_config, resolve_path
from hattrick.data.football_data import download_all, download_fixtures
from hattrick.data.loader import build_matches_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    processed_dir = resolve_path(cfg["data"]["processed_dir"])
    league_codes = [lg["code"] for lg in cfg["leagues"]]
    seasons = cfg["seasons"]
    print(f"Downloading results for leagues={league_codes} seasons={seasons} ...")
    download_all(league_codes, seasons, raw_dir)
    print("Downloading upcoming fixtures ...")
    download_fixtures(raw_dir)
    print("Building cleaned matches dataset ...")
    matches = build_matches_dataset(raw_dir, processed_dir, league_codes, seasons)
    print(f"Done: {len(matches)} matches -> {processed_dir / 'matches.csv'}")


if __name__ == "__main__":
    main()
