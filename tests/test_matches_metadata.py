"""data/matches_metadata.yaml must describe exactly the columns matches.csv has.

A column added to the loader without a description, or a description left
behind for a column that was dropped, fails here rather than drifting.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from hattrick.data.loader import KEEP_COLS
from hattrick.data.xg import XG_COLUMNS

REPO = Path(__file__).resolve().parents[1]
METADATA = yaml.safe_load((REPO / "data" / "matches_metadata.yaml").read_text())
COLUMNS = METADATA["columns"]


def _loader_columns() -> list[str]:
    """What update_data.py writes, plus what update_xg.py merges on."""
    xg = [c for c in XG_COLUMNS if c not in ("date", "home", "away")]
    return list(KEEP_COLS.values()) + xg


def test_metadata_covers_exactly_the_loader_columns():
    assert set(COLUMNS) == set(_loader_columns())


def test_metadata_lists_columns_in_file_order():
    assert list(COLUMNS) == _loader_columns()


@pytest.mark.parametrize("column", list(COLUMNS))
def test_every_column_is_fully_described(column):
    entry = COLUMNS[column]
    for field in ("type", "description", "source", "missing"):
        assert str(entry.get(field, "")).strip(), f"{column}: no {field}"


def test_metadata_matches_the_built_file_when_present():
    """Checks the real table too, when one has been built locally."""
    path = REPO / "data" / "processed" / "matches.csv"
    if not path.exists():
        pytest.skip("no data/processed/matches.csv - run scripts/update_data.py")
    header = list(pd.read_csv(path, nrows=0).columns)
    assert set(header) <= set(COLUMNS), f"undescribed: {set(header) - set(COLUMNS)}"
