"""Load config.yaml into a plain nested dict, and resolve artifact paths.

Artifacts are stored PER LEAGUE, under `artifacts/{league_code}/`. That is
not cosmetic. A single Dixon-Coles fit assumes one shared pool of teams with
a common scoring level, one home-advantage multiplier and one rho. Fitting
the Premier League and La Liga together would put every team on one
attack/defense scale even though they never play each other, so the relative
ratings between the two leagues would be determined by nothing but their
respective goal averages - and the "home advantage" would be an average of
two genuinely different quantities. Keeping the artifacts in separate
directories makes pooling impossible to do by accident.
"""
from __future__ import annotations
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def league_codes(cfg: dict) -> list[str]:
    return [lg["code"] for lg in cfg["leagues"]]


def league_name(cfg: dict, code: str) -> str:
    for lg in cfg["leagues"]:
        if lg["code"] == code:
            return lg.get("name", code)
    return code


def artifact_dir(cfg: dict, code: str | None = None) -> Path:
    """`artifacts/`, or `artifacts/{code}/` when a league is given."""
    base = resolve_path(cfg["model"].get("artifact_dir", "artifacts"))
    return base if code is None else base / code


def model_artifact_path(cfg: dict, code: str) -> Path:
    return artifact_dir(cfg, code) / "dixon_coles_params.json"


def sim_artifact_path(cfg: dict, code: str) -> Path:
    return artifact_dir(cfg, code) / "season_sim.json"


def current_season_start(cfg: dict, today=None):
    """When the season in progress began, for "already played this season".

    Reads `simulation.current_season_start` if set, otherwise derives 1 August
    of the current season's starting year. Deriving is the better default: a
    hardcoded date is correct for one year and then silently wrong, which
    shows up as the season table treating a finished season as in progress.
    """
    import pandas as pd

    configured = (cfg.get("simulation") or {}).get("current_season_start")
    if configured:
        return pd.Timestamp(configured)
    ts = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    return pd.Timestamp(year=ts.year if ts.month >= 7 else ts.year - 1, month=8, day=1)


def load_league_models(cfg: dict, codes: list[str] | None = None) -> dict:
    """Load one fitted model per league.

    Missing leagues are skipped rather than fatal, so a partially-fitted
    workspace still runs; the caller decides whether an empty result is an
    error. Returns `{league_code: DixonColesModel}`.
    """
    from hattrick.model.dixon_coles import DixonColesModel, DixonColesParams

    out = {}
    for code in (codes or league_codes(cfg)):
        path = model_artifact_path(cfg, code)
        if not path.exists():
            continue
        m = DixonColesModel(xi=cfg["model"]["xi"])
        m.params = DixonColesParams.from_json(str(path))
        out[code] = m
    return out
