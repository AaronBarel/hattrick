"""Club Elo (clubelo.com) - free, no auth, used only as an independent sanity
check against fitted Dixon-Coles ratings, not as a primary input."""
from __future__ import annotations
import io
import datetime as dt
import pandas as pd
import requests

API_BASE = "http://api.clubelo.com"


def get_ratings_as_of(date: dt.date | None = None) -> pd.DataFrame:
    date = date or dt.date.today()
    resp = requests.get(f"{API_BASE}/{date.isoformat()}", timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))
