"""Forward fixture schedules.

football-data.co.uk's `fixtures.csv` is a rolling near-term window, and a
narrow one: it is republished around midweek and, once a matchweek finishes,
it can contain nothing but already-played games until the next publish. Two
things go wrong if that is your only source.

1. The dashboard shows matches whose results are already known, which reads
   as the model predicting the past.
2. Filter those out and the dashboard is simply empty, with no way to tell
   "the source has not published yet" from "the app is broken".

Understat publishes the **entire season's** schedule up front - 380 fixtures
running to next May, flagged played/unplayed - so it answers both. This
module fetches that, maps club names onto the spellings the rest of the
pipeline uses, and writes `fixtures_full.csv`. `loader.load_fixtures` prefers
that file and falls back to football-data.co.uk's when it is absent, so
nothing here is a hard dependency.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hattrick.data.football_data import current_season_code
from hattrick.data.xg import LEAGUE_CODE_TO_UNDERSTAT, build_team_mapping

__all__ = ["current_season_code", "fetch_understat_fixtures", "FULL_FIXTURES_FILE"]

FULL_FIXTURES_FILE = "fixtures_full.csv"


def fetch_understat_fixtures(league_codes, season=None, team_names_by_league=None,
                             raw_dir=None) -> pd.DataFrame:
    """Fetch the full season schedule for each league.

    `team_names_by_league` maps a league code to the club names used
    elsewhere in the pipeline (i.e. football-data.co.uk spellings); when
    supplied, Understat's names are translated onto them so the fitted
    model can actually look the teams up. Without it the raw Understat
    names are returned unchanged.

    Leagues whose schedule is not published yet are skipped and recorded in
    the result's `.attrs["skipped"]`, rather than failing the whole fetch -
    leagues start at different times, so this is a normal August condition,
    not an error.

    Returns columns: league, date, kickoff (tz-aware UTC), home, away,
    is_result.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "soccerdata is required for full-season fixtures. Install with "
            "`uv sync --extra xg`, or rely on football-data.co.uk's fixtures.csv."
        ) from exc

    season = season or current_season_code()
    frames, skipped = [], {}
    for code in league_codes:
        league = LEAGUE_CODE_TO_UNDERSTAT.get(code)
        if league is None:
            skipped[code] = "not covered by Understat"
            continue
        try:
            sched = sd.Understat(leagues=league, seasons=season).read_schedule().reset_index()
        except Exception as exc:
            skipped[code] = f"{type(exc).__name__}: {exc}"
            continue

        # Leagues start at different times, so a schedule can simply not exist
        # yet - the Bundesliga's was unpublished while the other four were
        # already up. Understat answers that with a degenerate frame rather
        # than an error, so check the shape instead of trusting the call.
        required = {"date", "home_team", "away_team", "is_result"}
        if not required.issubset(sched.columns) or sched.empty:
            skipped[code] = f"no {season} schedule published yet"
            continue

        home = sched["home_team"].astype(str)
        away = sched["away_team"].astype(str)
        targets = (team_names_by_league or {}).get(code)
        if targets:
            mapping = build_team_mapping(sorted(set(home) | set(away)), sorted(targets))
            home = home.map(lambda n: mapping.get(n, n))
            away = away.map(lambda n: mapping.get(n, n))

        kickoff = pd.to_datetime(sched["date"], errors="coerce")
        frames.append(pd.DataFrame({
            "league": code,
            "date": kickoff.dt.normalize(),
            # Understat timestamps are UK local, like football-data.co.uk's.
            "kickoff": kickoff.dt.tz_localize("Europe/London", ambiguous=True,
                                              nonexistent="shift_forward").dt.tz_convert("UTC"),
            "home": home,
            "away": away,
            "is_result": sched["is_result"].astype("boolean").fillna(False),
        }).dropna(subset=["date", "home", "away"]))

    if not frames:
        raise ValueError(
            f"No {season} schedule available for any of {list(league_codes)}. "
            f"Reasons: {skipped}"
        )

    out = pd.concat(frames, ignore_index=True).sort_values("kickoff").reset_index(drop=True)
    out.attrs["skipped"] = skipped
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(raw_dir / FULL_FIXTURES_FILE, index=False)
    return out
