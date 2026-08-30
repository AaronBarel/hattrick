"""Tests for xG ingestion, club-name matching, and the merge.

The dangerous failure here is not a crash, it is a *confident wrong join*:
100% coverage while a club is quietly receiving its city rival's xG. Several
of these tests exist specifically to make that impossible.
"""
import numpy as np
import pandas as pd
import pytest

from hattrick.data.xg import (
    build_team_mapping,
    load_xg_csv,
    merge_xg,
    normalize_team,
)


# --------------------------------------------------------------- normalizing
def test_normalization_strips_noise_accents_and_punctuation():
    assert normalize_team("Paris Saint-Germain FC") == normalize_team("Paris Saint Germain")
    assert normalize_team("Bayer Leverkusen") == "bayer leverkusen"   # accents/none
    assert normalize_team("St. Pauli") == normalize_team("St Pauli")


def test_aliases_collapse_known_spelling_variants_of_one_club():
    """Atletico Madrid is written three ways across the two sources. All three
    must canonicalize together or the club loses xG for whole seasons."""
    forms = ["Atlético Madrid", "Atletico Madrid", "Ath Madrid", "Atl. Madrid"]
    assert len({normalize_team(f) for f in forms}) == 1

    for f in ["Rayo Vallecano", "Vallecano"]:
        assert normalize_team(f) == normalize_team("Vallecano")


def test_same_city_rivals_never_collapse_to_the_same_string():
    """The regression that motivated the conservative noise list. If these
    collapse, build_team_mapping's one-to-one assignment can hand United's
    xG to City for an entire season, silently."""
    pairs = [
        ("Manchester United", "Manchester City"),
        ("Real Madrid", "Atletico Madrid"),
        ("AC Milan", "Inter"),
        ("Real Sociedad", "Real Betis"),
    ]
    for a, b in pairs:
        assert normalize_team(a) != normalize_team(b), f"{a} and {b} collapsed"


def test_name_is_never_normalized_out_of_existence():
    assert normalize_team("FC") != ""
    assert normalize_team("Club") != ""


# ------------------------------------------------------------------- mapping
def test_mapping_resolves_real_abbreviations_seen_in_the_wild():
    understat = ["Manchester United", "Manchester City", "Newcastle United",
                 "Nottingham Forest", "Wolverhampton Wanderers", "Tottenham"]
    football_data = ["Man United", "Man City", "Newcastle", "Nott'm Forest",
                     "Wolves", "Tottenham"]
    m = build_team_mapping(understat, football_data)
    assert m["Manchester United"] == "Man United"
    assert m["Manchester City"] == "Man City"
    assert m["Newcastle United"] == "Newcastle"
    assert m["Nottingham Forest"] == "Nott'm Forest"
    assert m["Wolverhampton Wanderers"] == "Wolves"
    assert m["Tottenham"] == "Tottenham"


def test_mapping_handles_cross_language_and_nickname_aliases():
    m = build_team_mapping(["Athletic Club", "FC Cologne"], ["Ath Bilbao", "FC Koln"])
    assert m["Athletic Club"] == "Ath Bilbao"
    assert m["FC Cologne"] == "FC Koln"


def test_mapping_is_one_to_one():
    """No target may be claimed twice - that is the shape a rival-swap bug
    takes."""
    src = ["Manchester United", "Manchester City", "Manchester"]
    m = build_team_mapping(src, ["Man United", "Man City"])
    assert len(set(m.values())) == len(m.values())


def test_unresolvable_name_is_left_out_rather_than_guessed():
    m = build_team_mapping(["Wellington Phoenix"], ["Arsenal", "Chelsea"])
    assert "Wellington Phoenix" not in m


# ---------------------------------------------------------------- loading CSV
def test_manual_csv_path_accepts_the_standard_format(tmp_path):
    p = tmp_path / "xg.csv"
    p.write_text("date,home,away,home_xg,away_xg\n2024-08-16,Arsenal,Chelsea,1.8,0.9\n")
    df = load_xg_csv(p)
    assert list(df.columns) == ["date", "home", "away", "home_xg", "away_xg"]
    assert df.loc[0, "home_xg"] == pytest.approx(1.8)


def test_missing_columns_fail_loudly_with_a_useful_message(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("date,home,away\n2024-08-16,Arsenal,Chelsea\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_xg_csv(p)


# ----------------------------------------------------------------- merging
def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2024-08-17", "2024-08-18"]),
        "home": ["Man United", "Man City", "Wolves"],
        "away": ["Wolves", "Man United", "Man City"],
        "home_goals": [1, 3, 0],
        "away_goals": [0, 1, 2],
    })


def _xg(dates=("2024-08-16", "2024-08-17", "2024-08-18")):
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "home": ["Manchester United", "Manchester City", "Wolverhampton Wanderers"],
        "away": ["Wolverhampton Wanderers", "Manchester United", "Manchester City"],
        "home_xg": [1.7, 2.4, 0.6],
        "away_xg": [0.5, 1.1, 1.9],
    })


def test_merge_attaches_xg_to_the_right_fixture():
    out, rep = merge_xg(_matches(), _xg())
    assert rep.coverage == 1.0
    assert out.loc[0, "home_xg"] == pytest.approx(1.7)
    assert out.loc[1, "home_xg"] == pytest.approx(2.4)   # City home, not United
    assert out.loc[2, "away_xg"] == pytest.approx(1.9)


def test_merge_tolerates_date_drift_between_sources():
    """Understat timestamps kick-off in a different timezone, and postponed
    games get re-dated by one source before the other."""
    out, rep = merge_xg(_matches(), _xg(dates=("2024-08-17", "2024-08-18", "2024-08-19")))
    assert rep.coverage == 1.0


def test_merge_rejects_drift_beyond_tolerance():
    out, rep = merge_xg(_matches(), _xg(dates=("2024-09-16", "2024-09-17", "2024-09-18")))
    assert rep.coverage == 0.0
    assert out["home_xg"].isna().all()


def test_extra_source_rows_are_counted_not_silently_dropped():
    """An xG row that corresponds to no fixture must show up in the report."""
    m = _matches()
    xg = _xg()
    xg.loc[len(xg)] = [pd.Timestamp("2024-08-19"), "Sporting Lisbon", "Benfica", 1.0, 1.0]
    _, rep = merge_xg(m, xg)
    assert rep.unmatched_rows >= 1
    assert "matched no fixture" in rep.summary()


def test_our_clubs_without_an_xg_counterpart_are_named():
    """The actionable direction: which teams in OUR data will fall back to
    goals. Reporting it the other way round tells you about clubs you do not
    model and stays silent about the ones you do."""
    m = _matches()
    m.loc[len(m)] = [pd.Timestamp("2024-08-19"), "Accrington Stanley", "Wolves", 1, 0]
    _, rep = merge_xg(m, _xg())
    assert any("Accrington" in n for n in rep.unmatched_teams)
    assert "no xG-source counterpart" in rep.summary()


def test_two_spellings_of_one_club_both_get_xg():
    """football-data.co.uk renamed clubs mid-archive: Atletico Madrid appears
    as both "Ath Madrid" and "Atl. Madrid", Rayo Vallecano as both
    "Vallecano" and "Rayo Vallecano". A one-to-one mapping lets one spelling
    claim the club and silently starves the other - which cost about 20% of
    La Liga's xG coverage."""
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2025-08-16", "2024-08-17", "2025-08-17"]),
        "league": ["SP1"] * 4,
        "home": ["Ath Madrid", "Atl. Madrid", "Vallecano", "Rayo Vallecano"],
        "away": ["Getafe", "Getafe", "Getafe", "Getafe"],
        "home_goals": [1, 2, 0, 1], "away_goals": [0, 1, 0, 1],
    })
    xg = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2025-08-16", "2024-08-17", "2025-08-17"]),
        "league": ["SP1"] * 4,
        "home": ["Atletico Madrid", "Atletico Madrid", "Rayo Vallecano", "Rayo Vallecano"],
        "away": ["Getafe", "Getafe", "Getafe", "Getafe"],
        "home_xg": [1.4, 2.2, 0.7, 1.1], "away_xg": [0.6, 0.9, 0.5, 1.0],
    })
    out, rep = merge_xg(matches, xg)
    assert rep.coverage == 1.0, f"both spellings must resolve; got {rep.summary()}"
    assert out["home_xg"].notna().all()


def test_one_to_one_is_still_the_default():
    """The many-to-one relaxation must be opt-in: for same-league rivals,
    one-to-one is exactly what stops a club stealing another's data."""
    m = build_team_mapping(["Manchester United", "Manchester City"], ["Man United", "Man City"])
    assert len(set(m.values())) == 2


def test_matches_are_never_dropped_by_the_merge():
    """A left join, always: a match with no xG keeps NaN and falls back to
    goals downstream."""
    m = _matches()
    out, rep = merge_xg(m, _xg().iloc[:1])
    assert len(out) == len(m)
    assert out["home_xg"].isna().sum() == 2


def test_empty_xg_input_degrades_gracefully():
    out, rep = merge_xg(_matches(), pd.DataFrame())
    assert len(out) == 3
    assert out["home_xg"].isna().all()
    assert rep.coverage == 0.0


def test_score_disagreement_is_detected_and_reported():
    """The independent integrity check: high coverage alone cannot tell you
    the join landed on the right fixture, but a scoreline mismatch can."""
    xg = _xg()
    xg["src_home_goals"] = [1, 3, 0]
    xg["src_away_goals"] = [0, 1, 2]
    _, ok = merge_xg(_matches(), xg)
    assert ok.goal_mismatches == 0
    assert "Join verified" in ok.summary()

    xg.loc[1, "src_home_goals"] = 9      # source now disagrees on this fixture
    _, bad = merge_xg(_matches(), xg)
    assert bad.goal_mismatches == 1
    assert "disagree" in bad.summary()
