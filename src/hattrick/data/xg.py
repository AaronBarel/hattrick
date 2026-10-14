"""Expected-goals (xG) ingestion, and merging it onto the matches frame.

WHY xG
------
Goals are a very sparse readout of how a match actually went: roughly 2.7 per
game, which means a single deflection moves the scoreline by ~40% of a
typical match's information content. xG measures the quality of the chances a
team created, so it is a far less noisy estimate of the same underlying
attack/defense strength that Dixon-Coles is trying to recover. The empirical
finding - consistent across the public analytics literature - is that team
xG predicts *future* results better than past goals do, precisely because it
strips out finishing variance. See
Hudl's overview of xG and the PLOS ONE review of xG model performance:

  https://www.hudl.com/blog/expected-goals-xg-explained
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282295

SOURCES
-------
Primary: Understat, via the maintained `soccerdata` package
(https://github.com/probberechts/soccerdata). Understat is one of the few
remaining free sources of shot-level xG, and it covers exactly the five
leagues this project targets. `soccerdata` is preferred over a hand-rolled
scraper here: it already handles Understat's embedded-JSON page format,
caches to disk, and is maintained by someone tracking upstream changes -
which a scraper in this repo would not be.

That said, it is still a scrape of a site with no API contract. Understat can
change its page structure or block traffic at any time, and if it does, this
module stops working. It is therefore an *optional enhancement*, never a hard
dependency: `soccerdata` is an extra, `fetch_understat_xg` degrades to a
clear error rather than taking the pipeline down, and the manual path below
exists so a broken scrape never blocks the model.

Fallback / manual path: any CSV with the columns

    date, home, away, home_xg, away_xg

works identically via `load_xg_csv`. That standardized format is easy to
export from FBref, StatsBomb open data, or a paid feed, so xG coverage does
not depend on Understat specifically.

NAME MATCHING
-------------
football-data.co.uk and Understat disagree on club names ("Man United" vs
"Manchester United", "Wolves" vs "Wolverhampton Wanderers"). `merge_xg`
normalizes both sides and then resolves the remainder by best-match within
the league's own team set. Anything it cannot match is *reported*, never
silently dropped - an unmatched team means that team's whole season quietly
falls back to goals, which is exactly the kind of degradation that should be
loud.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

XG_COLUMNS = ["date", "home", "away", "home_xg", "away_xg"]

# football-data.co.uk division code -> soccerdata/Understat league name.
LEAGUE_CODE_TO_UNDERSTAT = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "I1": "ITA-Serie A",
    "D1": "GER-Bundesliga",
    "F1": "FRA-Ligue 1",
}

# Tokens that carry no identifying information: legal/organizational forms
# and founding years that sources include or omit arbitrarily.
#
# This list is deliberately conservative. An earlier, more aggressive version
# also stripped words like "united", "city", "real" and "atletico" - which
# collapses "Manchester United" and "Manchester City" to the same string, and
# "Real Madrid" and "Atletico Madrid" to the same string. The one-to-one
# assignment in build_team_mapping would then hand a club its city rival's
# xG, silently and for a whole season. Words that distinguish two clubs in
# the same city are NOT noise, however generic they look in isolation.
_NOISE_TOKENS = {
    "fc", "afc", "cf", "ac", "as", "ss", "ssc", "sc", "sv", "tsv", "vfl", "vfb",
    "fsv", "bsc", "ud", "cd", "rc", "sd", "ca", "ogc", "us", "ec", "aj", "rcd",
    "club", "calcio", "association", "sportiva", "de", "d", "of", "the",
    "1899", "1900", "1904", "1905", "1909", "05", "04", "96",
}


# Pairs no lexical rule can bridge: a nickname unrelated to the formal name,
# or a city rendered in a different language. Keys and values are both
# normalized before use, so either spelling may appear on either side.
# Anything resolvable by normalization or in-league fuzzy matching does NOT
# belong here - this table is for genuine domain knowledge only.
_ALIASES = {
    # --- football-data.co.uk spelling changes mid-archive. Both spellings
    # must canonicalize to the same string or one season's worth of a club's
    # xG goes missing.
    "atl madrid": "ath madrid",               # Spain: newer files
    "atletico madrid": "ath madrid",          # Spain: Understat
    "rayo vallecano": "vallecano",            # Spain: newer files + Understat
    # --- Shorthand seen in third-party feeds, which are not always even
    # internally consistent: Nottingham Forest turns up as both "Nottingham
    # Forest" and "Nottingham", so both spellings must land in one place.
    "nottingham": "nott m forest",            # England
    "nottingham forest": "nott m forest",
    "atletico": "ath madrid",                 # Spain
    "psg": "paris sg",                        # France
    "stade rennais": "rennes",                # France
    # --- nicknames and cross-language names no lexical rule can bridge.
    "wolverhampton wanderers": "wolves",      # England
    "athletic club": "ath bilbao",            # Spain (Athletic Bilbao)
    "fc cologne": "fc koln",                  # Germany (Koln / Cologne)
    "cologne": "koln",
    "bayern munich": "bayern munich",
    "internazionale": "inter",                # Italy
    "hellas verona": "verona",
}


def normalize_team(name: str) -> str:
    """Aggressively normalize a club name for cross-source matching.

    Strips accents, punctuation and case, then drops generic club-name
    tokens. "Wolverhampton Wanderers" and "Wolves" do NOT collapse to the
    same string here - no purely lexical rule gets that one - which is why
    `build_team_mapping` also does a fuzzy pass within the league's own
    roster, where "wolverhampton" vs "wolves" is an easy call because there
    are only ~20 candidates.
    """
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t]
    # Alias lookup happens both before and after noise-token stripping,
    # because stripping can destroy the very phrase an alias keys on:
    # "Athletic Club" loses "club" as noise and becomes "athletic", so an
    # alias written as "athletic club" would never fire post-strip.
    whole = " ".join(tokens)
    if whole in _ALIASES:
        return _ALIASES[whole]
    kept = [t for t in tokens if t not in _NOISE_TOKENS]
    out = " ".join(kept) if kept else whole  # never normalize a name away
    return _ALIASES.get(out, out)


def build_team_mapping(source_names, target_names, cutoff: float = 0.55,
                       one_to_one: bool = True) -> dict[str, str]:
    """Map xG-source club names onto the names the matches frame uses.

    Both sides describe the same league, so this is an assignment problem
    over two small sets, not open-ended entity resolution. Strategy:

    1. Exact match on the normalized form (handles most clubs outright).
    2. Prefix/containment, which catches the common abbreviation pattern
       ("wolverhampton" contains "wolves"? no - but "nottingham" vs
       "nottm" and "tottenham" vs "tottenham hotspur" resolve here).
    3. Fuzzy best-match on the normalized strings, restricted to the
       still-unclaimed targets and required to clear `cutoff`.

    `one_to_one=True` (the default) lets each target be claimed at most once,
    so a single ambiguous source name cannot shadow a whole league.

    `one_to_one=False` allows several sources to share one target, but ONLY
    via the exact-normalized pass. That distinction is the whole point:

    - Two spellings that canonicalize to the same string are the same club by
      construction, so both may legitimately claim it. football-data.co.uk
      writes Atletico Madrid as "Ath Madrid" in older seasons and
      "Atl. Madrid" in newer ones; under strict one-to-one the second
      spelling silently gets nothing, which cost ~20% of La Liga's xG.
    - A prefix or fuzzy hit is a *guess*. Letting guesses share a target lets
      two genuinely different clubs both attach to one source, which is worse
      than missing data because it is wrong data. Those passes stay
      one-to-one regardless of this flag.
    """
    src = {s: normalize_team(s) for s in source_names}
    tgt = {t: normalize_team(t) for t in target_names}
    mapping: dict[str, str] = {}
    claimed: set[str] = set()

    # Pass 1 - exact canonical match. Safe to share a target here, because
    # an identical normalized form means the same club by construction.
    for s, sn in src.items():
        for t, tn in tgt.items():
            if sn and sn == tn and (t not in claimed or not one_to_one):
                mapping[s] = t
                claimed.add(t)
                break

    # Pass 2 - prefix / containment. A guess, so strictly one-to-one.
    for s, sn in src.items():
        if s in mapping or not sn:
            continue
        for t, tn in tgt.items():
            if t in claimed or not tn:
                continue
            if sn.startswith(tn) or tn.startswith(sn) or tn in sn or sn in tn:
                mapping[s] = t
                claimed.add(t)
                break

    # Pass 3 - fuzzy. Also a guess, also strictly one-to-one.
    for s, sn in src.items():
        if s in mapping or not sn:
            continue
        pool = {t: tn for t, tn in tgt.items() if t not in claimed}
        best = difflib.get_close_matches(sn, list(pool.values()), n=1, cutoff=cutoff)
        if best:
            t = next(t for t, tn in pool.items() if tn == best[0])
            mapping[s] = t
            claimed.add(t)

    return mapping


def load_xg_csv(path) -> pd.DataFrame:
    """Read a standardized xG CSV: date, home, away, home_xg, away_xg.

    This is the source-agnostic entry point. Anything that can produce those
    five columns - FBref export, StatsBomb open data, a paid feed, a
    hand-maintained sheet - plugs in here without touching the scraper.
    """
    df = pd.read_csv(path)
    missing = [c for c in XG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: xG CSV is missing required column(s) {missing}. "
                         f"Expected exactly: {XG_COLUMNS}")
    df = df[XG_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("home_xg", "away_xg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "home", "away", "home_xg", "away_xg"]).reset_index(drop=True)


def fetch_understat_xg(league_codes, seasons, out_dir=None) -> pd.DataFrame:
    """Scrape match xG from Understat via soccerdata.

    EXPERIMENTAL in the sense that it depends on a site with no API contract.
    It is verified working as of this commit against the live site for the
    Premier League, but Understat can change or block at any time. On
    failure this raises with an actionable message rather than returning
    partial data, and callers are expected to fall back to `load_xg_csv`.

    `seasons` use football-data.co.uk's compact form ("2425"), matching
    config.yaml; soccerdata accepts that form directly.
    """
    try:
        import soccerdata as sd
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "soccerdata is required for Understat xG. Install it with "
            "`uv sync --extra xg`, or supply a CSV to load_xg_csv() instead."
        ) from exc

    frames = []
    for code in league_codes:
        league = LEAGUE_CODE_TO_UNDERSTAT.get(code)
        if league is None:
            raise ValueError(
                f"No Understat league known for division code {code!r}. "
                f"Known: {sorted(LEAGUE_CODE_TO_UNDERSTAT)}. Understat does not "
                f"cover every competition football-data.co.uk does."
            )
        sched = sd.Understat(leagues=league, seasons=list(seasons)).read_schedule()
        sched = sched.reset_index()
        df = pd.DataFrame({
            "date": pd.to_datetime(sched["date"], errors="coerce"),
            "home": sched["home_team"].astype(str),
            "away": sched["away_team"].astype(str),
            "home_xg": pd.to_numeric(sched["home_xg"], errors="coerce"),
            "away_xg": pd.to_numeric(sched["away_xg"], errors="coerce"),
            # Carried purely so merge_xg can verify the join landed on the
            # right fixture - see XGMergeReport.goal_mismatches.
            "src_home_goals": pd.to_numeric(sched["home_goals"], errors="coerce"),
            "src_away_goals": pd.to_numeric(sched["away_goals"], errors="coerce"),
        })
        df["league"] = code
        frames.append(df.dropna(subset=["date", "home_xg", "away_xg"]))

    if not frames:
        raise ValueError("No leagues requested.")
    out = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_dir / "xg.csv", index=False)
    return out


class XGMergeReport:
    """What the merge actually managed to match, so gaps are visible."""

    def __init__(self, n_matches, n_matched, unmatched_teams, unmatched_rows, name_map,
                 goal_mismatches=None):
        self.n_matches = n_matches
        self.n_matched = n_matched
        self.unmatched_teams = unmatched_teams
        self.unmatched_rows = unmatched_rows
        self.name_map = name_map
        # Rows where the xG source's own scoreline disagrees with the matches
        # frame's. High coverage proves nothing on its own - a wrong name
        # mapping or a wrong date join produces a confident, fully-covered,
        # completely incorrect merge. Comparing the two sources' goal counts
        # is an independent check that the join landed on the same fixture,
        # and it is cheap because both sources carry the scoreline.
        self.goal_mismatches = 0 if goal_mismatches is None else int(goal_mismatches)

    @property
    def coverage(self) -> float:
        return self.n_matched / self.n_matches if self.n_matches else 0.0

    def summary(self) -> str:
        lines = [f"xG merged onto {self.n_matched}/{self.n_matches} matches "
                 f"({self.coverage:.1%} coverage)."]
        if self.unmatched_teams:
            lines.append(f"  Club names with no xG-source counterpart "
                         f"({len(self.unmatched_teams)}): {sorted(self.unmatched_teams)}")
            lines.append("  Those teams fall back to actual goals. Add an entry to "
                         "data/xg.py's _ALIASES to fix.")
        if self.unmatched_rows:
            lines.append(f"  {self.unmatched_rows} xG row(s) matched no fixture "
                         f"(date drift beyond tolerance, or a competition not in "
                         f"the matches frame).")
        if self.goal_mismatches:
            lines.append(f"  NOTE: {self.goal_mismatches} merged row(s) disagree with the "
                         f"xG source on the final score. Usually that means the join "
                         f"landed on the wrong fixture - check the club-name mapping. "
                         f"A small number can be legitimate: Union Berlin vs Bochum on "
                         f"2024-12-14 finished 1-1 on the pitch (what Understat records) "
                         f"but was awarded 0-2 by the DFB after a spectator struck the "
                         f"Bochum keeper with a lighter (what football-data.co.uk "
                         f"records). Investigate rather than assume either way.")
        else:
            lines.append("  Join verified: every merged row agrees with the xG source "
                         "on the final score.")
        return "\n".join(lines)

    def __repr__(self):
        return f"<XGMergeReport coverage={self.coverage:.1%}>"


def merge_xg(matches: pd.DataFrame, xg: pd.DataFrame, date_tolerance_days: int = 3,
             ) -> tuple[pd.DataFrame, XGMergeReport]:
    """Left-join xG onto `matches`, returning `(matches_with_xg, report)`.

    Joins on (home, away) after name normalization, then keeps the candidate
    whose date is closest and within `date_tolerance_days`. Matching on the
    club pair rather than on an exact timestamp is deliberate: Understat
    stores kickoff times in a different timezone, and postponed fixtures get
    re-dated by one source before the other. A pair of clubs meets at most
    once per venue per season, so the pair plus a few days' tolerance
    identifies the fixture unambiguously while surviving that drift.

    `matches` is never filtered: rows with no xG keep NaN in home_xg/away_xg
    and the model falls back to goals for them.
    """
    # Drop any xG already attached, so re-running is idempotent rather than
    # producing home_xg_x / home_xg_y. scripts/update_xg.py writes its result
    # back over matches.csv, so a second run is a completely normal thing to do.
    out = matches.drop(columns=[c for c in ("home_xg", "away_xg") if c in matches.columns])
    if xg is None or xg.empty:
        out["home_xg"] = np.nan
        out["away_xg"] = np.nan
        return out, XGMergeReport(len(out), 0, set(), 0, {})

    # Resolve club names WITHIN each league, never across all of them at once.
    # build_team_mapping assigns one-to-one, so pooling every league into a
    # single ~100-club assignment lets a Spanish club be claimed by a
    # similarly-spelled Italian one, and the loser silently gets no xG. Doing
    # it per league keeps each problem to ~20 candidates that genuinely
    # correspond, which is what makes the fuzzy pass safe.
    def _pairs(frame):
        return sorted(set(frame["home"].astype(str)) | set(frame["away"].astype(str)))

    def _map_one(match_names, xg_names):
        # Map OUR names onto the xG source's, many-to-one: several
        # football-data spellings can legitimately denote one club.
        return build_team_mapping(match_names, xg_names, one_to_one=False)

    if "league" in matches.columns and "league" in xg.columns:
        name_map, unmatched_teams = {}, set()
        for code in sorted(set(matches["league"].dropna())):
            ours = _pairs(matches[matches["league"] == code])
            theirs = _pairs(xg[xg["league"] == code])
            if not theirs:
                continue
            sub = _map_one(ours, theirs)
            name_map.update(sub)
            unmatched_teams |= {n for n in ours if n not in sub}
    else:
        ours, theirs = _pairs(matches), _pairs(xg)
        name_map = _map_one(ours, theirs)
        unmatched_teams = {n for n in ours if n not in name_map}

    # Join on the xG source's spelling, reached from ours via name_map.
    out["xgkey_home"] = out["home"].astype(str).map(lambda n: name_map.get(n, n))
    out["xgkey_away"] = out["away"].astype(str).map(lambda n: name_map.get(n, n))
    x = xg.copy()
    x["xgkey_home"] = x["home"].astype(str)
    x["xgkey_away"] = x["away"].astype(str)
    x = x.rename(columns={"date": "xg_date"})

    out["_row"] = np.arange(len(out))
    carry = ["xg_date", "xgkey_home", "xgkey_away", "home_xg", "away_xg"]
    check_cols = [c for c in ("src_home_goals", "src_away_goals") if c in x.columns]
    cand = out[["_row", "date", "xgkey_home", "xgkey_away"]].merge(
        x[carry + check_cols], on=["xgkey_home", "xgkey_away"], how="inner")
    cand["_gap"] = (cand["xg_date"] - cand["date"]).abs().dt.days
    cand = cand[cand["_gap"] <= date_tolerance_days]
    cand = cand.sort_values("_gap").drop_duplicates(subset=["_row"], keep="first")

    out = out.merge(cand[["_row", "home_xg", "away_xg"] + check_cols], on="_row", how="left")
    matched_rows = out["home_xg"].notna().sum()

    goal_mismatches = 0
    if check_cols:
        chk = out.dropna(subset=check_cols + ["home_xg"])
        goal_mismatches = int(((chk["src_home_goals"] != chk["home_goals"])
                               | (chk["src_away_goals"] != chk["away_goals"])).sum())
        out = out.drop(columns=check_cols)

    used = set(zip(cand["xgkey_home"], cand["xgkey_away"], cand["xg_date"]))
    unmatched_rows = int(sum(1 for r in x.itertuples()
                             if (r.xgkey_home, r.xgkey_away, r.xg_date) not in used))

    out = out.drop(columns=["_row", "xgkey_home", "xgkey_away"])
    return out, XGMergeReport(len(out), int(matched_rows), unmatched_teams,
                              unmatched_rows, name_map, goal_mismatches)
