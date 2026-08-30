"""Fetch historical results + odds and upcoming fixtures from football-data.co.uk.
Free, no auth. URL pattern: https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
Fixtures (all leagues): https://www.football-data.co.uk/fixtures.csv
"""
from __future__ import annotations
from pathlib import Path
import logging
import pandas as pd
import requests

logger = logging.getLogger(__name__)
BASE_URL = "https://www.football-data.co.uk"
RESULTS_URL_TMPL = BASE_URL + "/mmz4281/{season}/{div}.csv"
FIXTURES_URL = BASE_URL + "/fixtures.csv"


def current_season_code(today=None) -> str:
    """football-data.co.uk's compact code for the season in progress.

    European seasons run August-May, so anything from July onward belongs to
    the season starting that calendar year: 2026-08-24 -> "2627".

    This is computed rather than hardcoded on purpose. It used to be a
    module constant (`CURRENT_SEASON_HINT = "2526"`) that decided whether a
    cached file could be reused. Once the calendar moved past that season the
    constant silently made the *actual* current season cacheable, so new
    results would stop arriving and nobody would be told.
    """
    ts = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    start = ts.year if ts.month >= 7 else ts.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


class SeasonNotPublished(Exception):
    """The requested season/division file does not exist upstream yet."""


def _looks_like_results_csv(content: bytes) -> bool:
    """Is this actually a results CSV, or an error page wearing a .csv name?

    football-data.co.uk answers a request for an unpublished season with a
    "300 Multiple Choices" HTML page, at HTTP status 300 - which
    `raise_for_status()` does not treat as an error. Writing that to disk
    produces a .csv full of HTML that blows up the parser later, far from
    the cause. Check the payload, not just the status code.
    """
    head = content[:512].lstrip().lstrip(b"\xef\xbb\xbf").lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return False
    return b"hometeam" in head and b"awayteam" in head


def download_season(div: str, season: str, raw_dir: Path, force: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{div}_{season}.csv"

    # Never cache the season in progress - it gains matches every week.
    is_current = season == current_season_code()
    if dest.exists() and not force and not is_current:
        logger.info("Using cached %s", dest)
        return dest

    url = RESULTS_URL_TMPL.format(season=season, div=div)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if not _looks_like_results_csv(resp.content):
        raise SeasonNotPublished(
            f"{url} did not return a results CSV (HTTP {resp.status_code}, "
            f"{len(resp.content)} bytes). The season is probably not published yet."
        )

    dest.write_bytes(resp.content)
    logger.info("Downloaded %s -> %s", url, dest)
    return dest


def download_all(league_codes: list[str], seasons: list[str], raw_dir: Path) -> list[Path]:
    paths = []
    for div in league_codes:
        for season in seasons:
            try:
                paths.append(download_season(div, season, raw_dir))
            except (requests.HTTPError, SeasonNotPublished) as e:
                # Leagues start at different times, so early in a season some
                # divisions genuinely have no file yet. Skip, do not fail.
                logger.warning("Skipping %s %s: %s", div, season, e)
    return paths


def download_fixtures(raw_dir: Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "fixtures.csv"
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest
