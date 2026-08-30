"""Promotion detection, team-identity canonicalization, and the promoted prior.

The canonicalization tests matter most: football-data.co.uk renaming a club
mid-archive splits its history into two teams, one of which is a phantom with
a single match - and that phantom then gets flagged as newly promoted.
"""
import numpy as np
import pandas as pd
import pytest

from hattrick.data.features import detect_promoted_teams, season_of
from hattrick.data.loader import canonicalize_teams
from hattrick.model.dixon_coles import DixonColesModel


def _season(teams, year, league="E0", start_month=8):
    rows, day = [], 0
    for h in teams:
        for a in teams:
            if h == a:
                continue
            day += 1
            rows.append({"date": pd.Timestamp(year=year, month=start_month, day=1)
                                 + pd.Timedelta(days=day),
                         "league": league, "home": h, "away": a,
                         "home_goals": 1, "away_goals": 1})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- season_of
@pytest.mark.parametrize("date,expected", [
    ("2026-08-24", 2026), ("2026-07-01", 2026), ("2026-06-30", 2025),
    ("2027-05-30", 2026), ("2026-01-15", 2025),
])
def test_season_boundary_is_july(date, expected):
    assert season_of([pd.Timestamp(date)])[0] == expected


# --------------------------------------------------- promotion detection
def test_detects_teams_absent_the_previous_season():
    old = _season(["A", "B", "C", "D"], 2024)
    new = _season(["A", "B", "C", "E"], 2025)
    assert detect_promoted_teams(pd.concat([old, new])) == {"E"}


def test_single_season_cannot_identify_promotions():
    """With nothing to compare against, every team looks new. Returning an
    empty set is the honest answer; returning all of them would apply a
    relegation-candidate prior to the entire league."""
    assert detect_promoted_teams(_season(["A", "B", "C"], 2025)) == set()


def test_leagues_are_compared_independently():
    frames = [_season(["A", "B", "C"], 2024, "E0"), _season(["A", "B", "X"], 2025, "E0"),
              _season(["P", "Q", "R"], 2024, "SP1"), _season(["P", "Q", "Z"], 2025, "SP1")]
    assert detect_promoted_teams(pd.concat(frames)) == {"X", "Z"}


def test_a_renamed_club_is_not_reported_as_promoted():
    """The bug this guards: 'Ath Madrid' -> 'Atl. Madrid' in one 2026/27 file
    made a 153-match club look like a newcomer, which would have handed
    Atletico Madrid a relegation-candidate prior."""
    old = _season(["Ath Madrid", "B", "C"], 2024, "SP1")
    new = _season(["Atl. Madrid", "B", "C"], 2025, "SP1")
    merged = canonicalize_teams(pd.concat([old, new], ignore_index=True), verbose=False)
    assert detect_promoted_teams(merged) == set()


# ---------------------------------------------------- canonicalization
def test_duplicate_identities_collapse_to_the_most_frequent_spelling():
    many = _season(["Ath Madrid", "B", "C"], 2024, "SP1")
    one = pd.DataFrame([{"date": pd.Timestamp("2026-08-20"), "league": "SP1",
                         "home": "Atl. Madrid", "away": "B",
                         "home_goals": 1, "away_goals": 0}])
    out = canonicalize_teams(pd.concat([many, one], ignore_index=True), verbose=False)
    names = set(out["home"]) | set(out["away"])
    assert "Atl. Madrid" not in names
    assert "Ath Madrid" in names
    assert len(out) == len(many) + 1, "no rows may be lost"


def test_canonicalization_never_merges_same_city_rivals():
    """The catastrophic failure mode: giving a club its rival's record."""
    df = _season(["Man United", "Man City", "Real Madrid", "Ath Madrid"], 2024, "E0")
    out = canonicalize_teams(df, verbose=False)
    assert len(set(out["home"]) | set(out["away"])) == 4


def test_canonicalization_is_per_league():
    df = pd.concat([_season(["A", "B"], 2024, "E0"), _season(["A", "B"], 2024, "SP1")])
    out = canonicalize_teams(df, verbose=False)
    assert len(out) == len(df)


# ------------------------------------------------------- promoted prior
def _league_with_newcomer(seed=0, newcomer_matches=4):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    rows, day = [], 0
    for _ in range(4):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                rows.append({"date": pd.Timestamp("2024-08-01") + pd.Timedelta(days=day),
                             "league": "E0", "home": h, "away": a,
                             "home_goals": rng.poisson(1.5), "away_goals": rng.poisson(1.2)})
    for k in range(newcomer_matches):
        day += 1
        rows.append({"date": pd.Timestamp("2025-08-01") + pd.Timedelta(days=day),
                     "league": "E0", "home": "Newcomer", "away": teams[k % 8],
                     "home_goals": 1, "away_goals": 1})
    return pd.DataFrame(rows)


def test_promoted_prior_is_off_by_default():
    """It is measured as not improving forecasts, so it must be opt-in."""
    m = _league_with_newcomer()
    assert DixonColesModel().promoted_teams is None
    assert DixonColesModel(xi=0.0).fit(m).params.promoted == []


def test_promoted_team_is_rated_below_an_equivalent_unmarked_team():
    m = _league_with_newcomer()
    plain = DixonColesModel(xi=0.0).fit(m).params
    prom = DixonColesModel(xi=0.0, promoted_teams={"Newcomer"}).fit(m).params
    assert prom.attack["Newcomer"] < plain.attack["Newcomer"]
    assert prom.defense["Newcomer"] > plain.defense["Newcomer"]
    assert prom.promoted == ["Newcomer"]


def test_established_teams_are_essentially_untouched():
    m = _league_with_newcomer()
    plain = DixonColesModel(xi=0.0).fit(m).params
    prom = DixonColesModel(xi=0.0, promoted_teams={"Newcomer"}).fit(m).params
    for t in plain.teams:
        if t == "Newcomer":
            continue
        assert abs(prom.attack[t] / plain.attack[t] - 1) < 0.10


def test_offset_fades_once_the_model_has_its_own_evidence():
    """"Promoted" and "unknown to the model" are different sets. Applying the
    offset to a yo-yo club with a full record of its own is pure bias, and
    measurably made forecasts worse before this fade existed."""
    m = _league_with_newcomer(newcomer_matches=4)
    few = DixonColesModel(xi=0.0, promoted_teams={"Newcomer"},
                          promoted_fade_matches=30).fit(m).params

    # Same team, but now with a long record of its own.
    rich = _league_with_newcomer(newcomer_matches=60)
    many = DixonColesModel(xi=0.0, promoted_teams={"Newcomer"},
                           promoted_fade_matches=30).fit(rich).params
    assert "Newcomer" in few.promoted
    assert "Newcomer" not in many.promoted, "offset should have faded to nothing"


def test_fade_can_be_disabled():
    m = _league_with_newcomer(newcomer_matches=60)
    p = DixonColesModel(xi=0.0, promoted_teams={"Newcomer"},
                        promoted_fade_matches=0).fit(m).params
    assert "Newcomer" in p.promoted


def test_auto_detection_wires_through_fit():
    m = _league_with_newcomer()
    p = DixonColesModel(xi=0.0, promoted_teams="auto").fit(m).params
    assert p.promoted == ["Newcomer"]
