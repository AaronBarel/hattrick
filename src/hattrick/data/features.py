"""Derived per-team features that the raw match table does not carry.

Currently: which teams are newly promoted. Promotion matters because the
model's shrinkage prior has to decide what a data-poor team looks like, and
"league average" is the wrong answer for a side that has just come up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# European seasons run August-May, so July is the cut point.
SEASON_START_MONTH = 7


def season_of(dates) -> pd.Series:
    """Map dates to the calendar year their season began in."""
    d = pd.to_datetime(pd.Series(dates))
    return np.where(d.dt.month >= SEASON_START_MONTH, d.dt.year, d.dt.year - 1)


def detect_promoted_teams(matches: pd.DataFrame, season=None) -> set[str]:
    """Teams playing in `season` that were absent the season before.

    Detection is by absence, which is the only signal a results table
    carries, and it means two things are lumped together: genuine promotions,
    and teams missing from the previous season for any other reason (a data
    gap, a league that changed size, a club returning after two years away).
    All of them share the property that actually matters here - the model has
    no recent top-flight record for them - so treating them alike is
    reasonable, but it is a heuristic and not a promotion table.

    Requires at least two seasons of data; with one season there is nothing
    to compare against and this returns an empty set rather than guessing
    that every team is new.

    Handles multiple leagues in one frame, comparing within each league.
    """
    if matches.empty or "date" not in matches.columns:
        return set()

    df = matches.copy()
    df["_season"] = season_of(df["date"])
    leagues = df["league"].unique() if "league" in df.columns else [None]

    promoted: set[str] = set()
    for lg in leagues:
        sub = df if lg is None else df[df["league"] == lg]
        seasons = sorted(sub["_season"].unique())
        if len(seasons) < 2:
            continue
        target = seasons[-1] if season is None else season
        if target not in seasons:
            continue
        prev = [s for s in seasons if s < target]
        if not prev:
            continue

        cur_rows = sub[sub["_season"] == target]
        prev_rows = sub[sub["_season"] == prev[-1]]
        cur = set(cur_rows["home"]) | set(cur_rows["away"])
        before = set(prev_rows["home"]) | set(prev_rows["away"])
        promoted |= (cur - before)

    return promoted
