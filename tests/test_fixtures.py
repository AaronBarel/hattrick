"""Tests for upcoming-fixture selection.

The bug these cover was visible on the dashboard: it listed matches that had
already been played, because `load_fixtures` never filtered by date and
ignored the kickoff time column entirely. Showing a settled match as a
prediction is worse than showing nothing.
"""
import pandas as pd
import pytest

from hattrick.data.football_data import current_season_code, _looks_like_results_csv
from hattrick.data.loader import load_fixtures

HEADER = "Div,Date,Time,HomeTeam,AwayTeam\n"


def _write_rolling(tmp_path, rows):
    (tmp_path / "fixtures.csv").write_text(HEADER + rows)
    return tmp_path



def test_kickoff_time_is_used_not_just_the_date(tmp_path):
    """A 15:00 kickoff is in the past by the evening. Treating every fixture
    as midnight keeps settled matches on screen all day."""
    _write_rolling(tmp_path, "E0,24/08/2026,15:00,Arsenal,Chelsea\n"
                             "E0,24/08/2026,20:00,Everton,Fulham\n")
    # 18:00 UK on the same day: the first has kicked off, the second has not.
    now = pd.Timestamp("2026-08-24 17:00", tz="UTC")
    out = load_fixtures(tmp_path, "E0", now=now)
    assert list(out["home"]) == ["Everton"]


def test_kickoffs_are_interpreted_as_uk_time(tmp_path):
    """football-data.co.uk publishes UK local times with no zone marker.
    Reading them as UTC shifts every summer fixture by an hour, which flips
    the include/exclude decision for anything near the boundary."""
    _write_rolling(tmp_path, "E0,24/08/2026,20:00,Everton,Fulham\n")
    out = load_fixtures(tmp_path, "E0", upcoming_only=False)
    # 20:00 BST == 19:00 UTC
    assert out.loc[0, "kickoff"] == pd.Timestamp("2026-08-24 19:00", tz="UTC")


def test_missing_time_keeps_the_fixture_all_day(tmp_path):
    """A date-only fixture should not vanish at midnight."""
    _write_rolling(tmp_path, "E0,24/08/2026,,Arsenal,Chelsea\n")
    out = load_fixtures(tmp_path, "E0", now=pd.Timestamp("2026-08-24 12:00", tz="UTC"))
    assert len(out) == 1


def test_upcoming_only_can_be_turned_off(tmp_path):
    _write_rolling(tmp_path, "E0,01/01/2020,15:00,Arsenal,Chelsea\n")
    assert load_fixtures(tmp_path, "E0").empty
    assert len(load_fixtures(tmp_path, "E0", upcoming_only=False)) == 1


def test_results_are_sorted_by_kickoff(tmp_path):
    _write_rolling(tmp_path, "E0,30/08/2026,15:00,C,D\n"
                             "E0,28/08/2026,20:00,A,B\n")
    out = load_fixtures(tmp_path, "E0", now=pd.Timestamp("2026-08-27", tz="UTC"))
    assert list(out["home"]) == ["A", "C"]


# ---------------------------------------------------- full-season schedule
def _write_full(tmp_path, rows):
    (tmp_path / "fixtures_full.csv").write_text(
        "league,date,kickoff,home,away,is_result\n" + rows)
    return tmp_path


def test_full_schedule_is_preferred_over_the_rolling_file(tmp_path):
    _write_rolling(tmp_path, "E0,28/08/2026,20:00,Rolling,Source\n")
    _write_full(tmp_path, "E0,2026-08-28,2026-08-28 19:00:00+00:00,Full,Schedule,False\n")
    out = load_fixtures(tmp_path, "E0", now=pd.Timestamp("2026-08-27", tz="UTC"))
    assert list(out["home"]) == ["Full"]
    assert out.attrs["source"] == "full_schedule"


def test_full_schedule_drops_already_played_matches(tmp_path):
    _write_full(tmp_path,
                "E0,2026-08-21,2026-08-21 19:00:00+00:00,Played,Match,True\n"
                "E0,2026-08-28,2026-08-28 19:00:00+00:00,Future,Match,False\n")
    out = load_fixtures(tmp_path, "E0", now=pd.Timestamp("2026-08-20", tz="UTC"))
    assert list(out["home"]) == ["Future"], "is_result=True must be dropped"


def test_falls_back_per_league_not_just_per_file(tmp_path):
    """Leagues start at different times, so the full schedule can cover four
    of five. The fifth must still get whatever the rolling file has, rather
    than showing nothing because the other file exists."""
    _write_full(tmp_path, "E0,2026-08-28,2026-08-28 19:00:00+00:00,Arsenal,Chelsea,False\n")
    _write_rolling(tmp_path, "D1,29/08/2026,18:30,Bayern Munich,Dortmund\n")
    now = pd.Timestamp("2026-08-27", tz="UTC")
    e0, d1 = load_fixtures(tmp_path, "E0", now=now), load_fixtures(tmp_path, "D1", now=now)
    assert list(e0["home"]) == ["Arsenal"] and e0.attrs["source"] == "full_schedule"
    assert list(d1["home"]) == ["Bayern Munich"]
    assert d1.attrs["source"] == "rolling", "callers rely on this to warn about a near-term-only schedule"


def test_no_fixture_files_at_all_is_an_empty_frame_not_a_crash(tmp_path):
    out = load_fixtures(tmp_path, "E0")
    assert out.empty and list(out.columns) == ["date", "kickoff", "home", "away"]
    assert out.attrs["source"] is None


# ------------------------------------------------------------ season code
@pytest.mark.parametrize("today,expected", [
    ("2026-08-24", "2627"), ("2026-07-01", "2627"), ("2026-06-30", "2526"),
    ("2027-01-15", "2627"), ("2025-12-31", "2526"), ("2029-09-01", "2930"),
])
def test_current_season_code(today, expected):
    """This replaced a hardcoded constant that silently made the live season
    cacheable once the calendar moved past it, so new results stopped
    arriving with no warning."""
    assert current_season_code(today) == expected


def test_error_page_is_not_mistaken_for_a_results_csv():
    """football-data.co.uk answers an unpublished season with an HTML "300
    Multiple Choices" page at HTTP 300 - which raise_for_status() allows
    through. Writing that to disk breaks the CSV parser much later."""
    html = b'<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">\n<html><head><title>300 Multiple Choices</title>'
    assert not _looks_like_results_csv(html)
    assert _looks_like_results_csv(b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,1/1/24,A,B,1,0\n")
    assert _looks_like_results_csv(b"\xef\xbb\xbfDiv,Date,HomeTeam,AwayTeam,FTHG\n")
