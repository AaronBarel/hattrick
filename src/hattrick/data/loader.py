"""Turn raw football-data.co.uk CSVs into one clean, standardized DataFrame."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# football-data.co.uk publishes two distinct odds snapshots per match, and
# conflating them is a real (and easy) mistake:
#
#   B365H/D/A    - PRE-CLOSING. Per football-data.co.uk's own notes file,
#                  these are collected Friday afternoons for weekend games
#                  and Tuesday afternoons for midweek games: typically 2-4
#                  days before kickoff, before team news, and therefore a
#                  materially weaker forecast than the closing line.
#   B365CH/CD/CA - CLOSING, i.e. the last price before kickoff. The "C"
#                  infix marks closing across every bookmaker column set.
#
# The distinction matters because the closing price is the standard benchmark
# in the forecasting literature: it aggregates all public information, weighted
# by the capital participants commit to it, and is close to the best freely
# available probability estimate. Scoring a model against the PRE-closing
# price flatters the model, because you are beating a stale number.
#
# We also ingest Pinnacle closing (PSC*) and the cross-book closing average
# (AvgC*). Pinnacle is a low-margin, high-limit book that welcomes sharp
# money, so its closing line is generally the sharpest single-book benchmark
# available for free; AvgC is a useful fallback with broader coverage.
KEEP_COLS = {
    "Div": "league", "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "home_goals", "FTAG": "away_goals",
    # Bet365 pre-closing (NOT closing - see above).
    "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
    # Bet365 closing.
    "B365CH": "b365ch", "B365CD": "b365cd", "B365CA": "b365ca",
    # Pinnacle closing.
    "PSCH": "psch", "PSCD": "pscd", "PSCA": "psca",
    # Cross-bookmaker average closing.
    "AvgCH": "avgch", "AvgCD": "avgcd", "AvgCA": "avgca",
}


def _parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")


# Without these, a season's rows load with the column silently absent and
# everything downstream degrades quietly instead of failing.
REQUIRED_COLS = ["league", "date", "home", "away", "home_goals", "away_goals"]


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    """Read a football-data.co.uk CSV, coping with their inconsistent encoding.

    Their files are not uniform: seasons up to 2023/24 are plain Latin-1,
    while 2024/25 onward are UTF-8 with a byte-order mark. Reading a
    BOM-prefixed file as Latin-1 succeeds - it just names the first column
    "\ufeffDiv" instead of "Div". Since the first column is `Div`, the league
    code, KEEP_COLS then drops it and every row from those seasons ends up
    with a null league. Nothing raises.

    That is exactly what happened here: `scripts/fit_model.py` filters by
    league, so half the dataset - every match from August 2024 onward -
    was silently excluded from every fit, leaving models trained on data that
    stopped two years short of the present.

    So: try UTF-8-with-BOM first (it also reads plain UTF-8 and ASCII), fall
    back to Latin-1 for the older files, and strip any stray BOM or
    whitespace from the column names either way.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = _read_csv_any_encoding(path)
    cols_present = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df[list(cols_present)].rename(columns=cols_present)

    # Fail loudly on a missing essential column. The odds columns are
    # genuinely optional (older seasons and lower divisions lack them), but
    # these six are what every downstream stage assumes, and losing one
    # silently is how the BOM bug above went unnoticed.
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: required column(s) {missing} missing after parsing. "
            f"Got: {sorted(df.columns)}. This usually means the file's encoding or "
            f"header layout changed upstream."
        )

    df["date"] = _parse_dates(df["date"])
    df = df.dropna(subset=["date", "home", "away", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df


def canonicalize_teams(matches: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Collapse spellings of the same club onto one identity, per league.

    football-data.co.uk sometimes changes how it writes a club mid-archive.
    In the 2026/27 files Atletico Madrid appears as "Atl. Madrid" and Rayo
    Vallecano as "Rayo Vallecano", where every prior season used "Ath Madrid"
    and "Vallecano" - for exactly ONE match each.

    Left alone, the model treats those as brand-new teams with a single match
    of history. That is bad in three separate ways: a top club's 153-match
    record is split, a phantom one-match team appears in the ratings, and
    promotion detection flags Atletico Madrid as newly promoted and applies a
    relegation-candidate prior to it.

    Names are grouped by their canonical form (the same normalization the xG
    merge uses, which already knows these two aliases) and the most frequent
    spelling in each group wins. Choosing by frequency rather than a
    hardcoded list means new renames are absorbed automatically, and the
    long-established spelling is the one that survives.
    """
    from hattrick.data.xg import normalize_team

    if matches.empty:
        return matches

    out = matches.copy()
    league_col = out["league"] if "league" in out.columns else pd.Series("_", index=out.index)
    merges = {}

    for lg in league_col.unique():
        mask = league_col == lg
        sub = out[mask]
        counts = pd.concat([sub["home"], sub["away"]]).value_counts()

        groups: dict[str, list[str]] = {}
        for name in counts.index:
            groups.setdefault(normalize_team(str(name)), []).append(str(name))

        mapping = {}
        for canon, names in groups.items():
            if len(names) < 2:
                continue
            winner = max(names, key=lambda n: counts[n])   # counts is sorted desc
            for n in names:
                if n != winner:
                    mapping[n] = winner
                    merges[(lg, n)] = (winner, int(counts[n]), int(counts[winner]))

        if mapping:
            out.loc[mask, "home"] = sub["home"].replace(mapping)
            out.loc[mask, "away"] = sub["away"].replace(mapping)

    if merges and verbose:
        for (lg, loser), (winner, n_lose, n_win) in sorted(merges.items()):
            print(f"  [{lg}] merged team identity {loser!r} ({n_lose} matches) "
                  f"-> {winner!r} ({n_win} matches)")

    return out


def build_matches_dataset(raw_dir: Path, processed_dir: Path, league_codes: list[str],
                           seasons: list[str], write: bool = True) -> pd.DataFrame:
    """Raw CSVs -> one clean matches frame, optionally persisted.

    `write=False` matters: this function's side effect is overwriting
    `processed_dir/matches.csv`, so calling it once per league in a loop
    leaves that file containing only the LAST league. Read-only callers
    (analysis, notebooks) must pass write=False.
    """
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    frames = []
    for div in league_codes:
        for season in seasons:
            p = raw_dir / f"{div}_{season}.csv"
            if p.exists():
                frames.append(load_raw_csv(p))
    if not frames:
        raise FileNotFoundError(f"No raw CSVs in {raw_dir} for {league_codes}/{seasons}. "
                                 "Run scripts/update_data.py first.")
    matches = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    matches = matches.drop_duplicates(subset=["date", "home", "away"])

    matches = canonicalize_teams(matches)

    # Belt and braces on the same failure: a null league means those rows
    # vanish from every per-league fit without warning.
    n_null = int(matches["league"].isna().sum())
    if n_null:
        raise ValueError(
            f"{n_null} of {len(matches)} matches have no league code after loading. "
            f"Those rows would be silently dropped by every per-league fit."
        )
    if write:
        processed_dir.mkdir(parents=True, exist_ok=True)
        matches.to_csv(processed_dir / "matches.csv", index=False)
    return matches


# football-data.co.uk publishes kickoff times in UK local time, with no
# timezone marker anywhere in the file.
FIXTURE_TZ = "Europe/London"


def _fixture_kickoffs(dates: pd.Series, times: pd.Series | None) -> pd.Series:
    """Combine the Date and Time columns into timezone-aware UTC kickoffs.

    Two details that matter for deciding what is still "upcoming":

    - The Time column exists and is worth using. A fixture at 15:00 today is
      in the past by the evening, and treating every fixture as midnight
      would keep already-played matches on the dashboard all day.
    - Those times are UK local, but the person reading the dashboard may not
      be. Comparing a naive UK timestamp against a naive local `now` is wrong
      by up to a day near midnight, which is exactly when someone checking
      tomorrow's card would notice. So localize to Europe/London and compare
      in UTC.

    A missing or unparseable time falls back to 23:59 rather than 00:00, so a
    date-only fixture stays visible for the whole of its match day instead of
    disappearing at midnight.
    """
    if times is None:
        parsed = pd.Series([pd.NaT] * len(dates), index=dates.index)
    else:
        parsed = pd.to_datetime(times, format="%H:%M", errors="coerce")

    offsets = pd.to_timedelta(
        parsed.dt.hour.fillna(23) * 3600 + parsed.dt.minute.fillna(59) * 60,
        unit="s",
    )
    naive = dates.dt.normalize() + offsets
    return (naive
            .dt.tz_localize(FIXTURE_TZ, ambiguous=True, nonexistent="shift_forward")
            .dt.tz_convert("UTC"))


def _load_football_data_fixtures(path: Path, league_code: str) -> pd.DataFrame:
    """football-data.co.uk's rolling near-term fixtures.csv."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/update_data.py first.")
    df = _read_csv_any_encoding(path)
    df = df[df["Div"] == league_code].copy()
    df["Date"] = _parse_dates(df["Date"])
    df = df.rename(columns={"HomeTeam": "home", "AwayTeam": "away", "Date": "date"})
    df = df.dropna(subset=["date", "home", "away"])
    if df.empty:
        return df
    df["kickoff"] = _fixture_kickoffs(df["date"], df.get("Time"))
    return df


def _load_full_schedule(path: Path, league_code: str) -> pd.DataFrame:
    """The whole-season schedule written by scripts/update_fixtures.py.

    Already carries a tz-aware UTC kickoff, so no re-localizing here. Played
    matches are dropped: this file covers the entire season, and its
    `is_result` flag is a more reliable "has this happened" signal than
    comparing a kickoff time to the clock.
    """
    df = pd.read_csv(path)
    df = df[df["league"] == league_code].copy()
    if df.empty:
        return df
    df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce", utc=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "is_result" in df.columns:
        played = df["is_result"].astype(str).str.lower().isin(["true", "1"])
        df = df[~played]
    return df.dropna(subset=["date", "kickoff", "home", "away"])


def load_fixtures(raw_dir: Path, league_code: str, upcoming_only: bool = True,
                  now=None) -> pd.DataFrame:
    """Upcoming fixtures for one league.

    Prefers `fixtures_full.csv` - the whole-season schedule written by
    `scripts/update_fixtures.py` - and falls back to football-data.co.uk's
    rolling `fixtures.csv` when that is absent.

    `upcoming_only=True` (the default) drops anything whose kickoff has
    already passed. Both sources contain played matches: football-data.co.uk
    keeps the last few days in its rolling window, and the full schedule
    covers the entire season by construction. Without this filter the
    dashboard shows games whose results are already known, which reads as
    the model predicting the past.

    Returns columns `date`, `kickoff` (tz-aware UTC), `home`, `away`, with
    `attrs["source"]` set to "full_schedule", "rolling", or None (no data), so
    callers can say whether the result covers the rest of the season or only
    the next few days.
    """
    raw_dir = Path(raw_dir)
    full = raw_dir / "fixtures_full.csv"
    df = _load_full_schedule(full, league_code) if full.exists() else None
    source = "full_schedule"

    # Fall back per league, not just per file. Leagues start at different
    # times, so the full schedule can legitimately cover four of five while
    # the fifth is unpublished - that league should still get whatever
    # football-data.co.uk has rather than showing nothing.
    if df is None or df.empty:
        fd = raw_dir / "fixtures.csv"
        df = _load_football_data_fixtures(fd, league_code) if fd.exists() else pd.DataFrame()
        source = "rolling"

    if df.empty:
        empty = pd.DataFrame(columns=["date", "kickoff", "home", "away"])
        empty.attrs["source"] = None
        return empty

    if upcoming_only:
        cutoff = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        df = df[df["kickoff"] >= cutoff]

    out = (df[["date", "kickoff", "home", "away"]]
           .sort_values("kickoff")
           .reset_index(drop=True))
    out.attrs["source"] = source
    return out
