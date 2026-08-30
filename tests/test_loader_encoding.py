"""Regression tests for the BOM-encoding bug that silently halved the dataset.

football-data.co.uk changed encoding partway through their archive: seasons
up to 2023/24 are Latin-1, 2024/25 onward are UTF-8 with a BOM. Reading a
BOM-prefixed file as Latin-1 does not raise - it just names the first column
"﻿Div". Since that first column is the league code, KEEP_COLS dropped
it and every row from those seasons got a null league. `fit_model.py` filters
by league, so every model was quietly trained on data that stopped in May
2024.

The tests below cover both the parsing and the guard that would have made it
loud in the first place.
"""
import pandas as pd
import pytest

from hattrick.data.loader import build_matches_dataset, load_raw_csv

HEADER = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A,"
          "B365CH,B365CD,B365CA\n")
ROWS = ("E0,12/08/2024,Arsenal,Chelsea,2,1,1.80,3.60,4.50,1.75,3.70,4.60\n"
        "E0,13/08/2024,Liverpool,Everton,3,0,1.50,4.20,6.00,1.45,4.30,6.20\n")


def _write(path, encoding, bom=False):
    data = HEADER + ROWS
    raw = data.encode(encoding)
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)
    return path


def test_bom_prefixed_file_still_yields_a_league_column(tmp_path):
    """The actual bug. Before the fix this produced a frame with no 'league'
    column at all, and no error."""
    p = _write(tmp_path / "E0_2425.csv", "utf-8", bom=True)
    df = load_raw_csv(p)
    assert "league" in df.columns
    assert set(df["league"]) == {"E0"}
    assert len(df) == 2


def test_plain_latin1_file_still_loads(tmp_path):
    """The older seasons must keep working, accents included."""
    data = HEADER + "SP1,12/08/2024,Alavés,Getafe,1,1,2.10,3.30,3.60,2.05,3.40,3.70\n"
    p = tmp_path / "SP1_2425.csv"
    p.write_bytes(data.encode("latin1"))
    df = load_raw_csv(p)
    assert set(df["league"]) == {"SP1"}
    assert len(df) == 1


def test_utf8_without_bom_loads(tmp_path):
    p = _write(tmp_path / "E0_2324.csv", "utf-8", bom=False)
    assert len(load_raw_csv(p)) == 2


def test_closing_odds_survive_a_bom_file(tmp_path):
    """The closing columns are what the backtest benchmarks against, so they
    must survive the same path."""
    p = _write(tmp_path / "E0_2425.csv", "utf-8", bom=True)
    df = load_raw_csv(p)
    assert {"b365ch", "b365cd", "b365ca"}.issubset(df.columns)
    assert df.loc[0, "b365ch"] == pytest.approx(1.75)


def test_missing_essential_column_raises_instead_of_degrading(tmp_path):
    """The guard that turns this whole class of bug from silent to loud."""
    p = tmp_path / "broken.csv"
    p.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG\n12/08/2024,Arsenal,Chelsea,2,1\n")
    with pytest.raises(ValueError, match="required column"):
        load_raw_csv(p)


def test_build_dataset_rejects_null_league_codes(tmp_path):
    """Second line of defense: even if a file slipped through, a dataset with
    unlabelled rows must not reach the per-league fits."""
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    _write(raw / "E0_2425.csv", "utf-8", bom=True)
    out = build_matches_dataset(raw, proc, ["E0"], ["2425"])
    assert out["league"].notna().all()
    assert len(out) == 2


def test_mixed_encoding_seasons_all_survive_the_same_build(tmp_path):
    """The real archive is mixed. Loading both eras together is the case that
    actually broke."""
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    raw.mkdir()
    old = raw / "E0_2324.csv"
    old.write_bytes((HEADER + "E0,12/08/2023,Arsenal,Chelsea,1,0,1.9,3.5,4.2,1.85,3.6,4.3\n")
                    .encode("latin1"))
    _write(raw / "E0_2425.csv", "utf-8", bom=True)

    out = build_matches_dataset(raw, proc, ["E0"], ["2324", "2425"])
    assert len(out) == 3, "a whole season went missing"
    assert out["league"].notna().all()
    assert out["date"].dt.year.nunique() == 2
