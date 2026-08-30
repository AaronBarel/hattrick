"""Dashboard for the model's fixture probabilities and season simulation.

Run with:
    streamlit run app.py

Reads directly from the artifacts the CLI pipeline already produces
(artifacts/{league}/dixon_coles_params.json, artifacts/{league}/season_sim.json,
data/raw/fixtures_full.csv) and reuses the same hattrick functions the
scripts use - no logic is duplicated here, this is a view layer only.

Only *upcoming* fixtures are ever shown. Both fixture sources contain
already-played matches, and rendering those makes the model look like it is
predicting the past.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hattrick.config import (
    league_codes, league_name, load_config, load_league_models,
    model_artifact_path, resolve_path, sim_artifact_path,
)
from hattrick.data.loader import load_fixtures
from hattrick.model.dixon_coles import DixonColesModel
from hattrick.model.markets import all_markets
from hattrick.evaluation import OUTCOMES, expected_calibration_error, log_loss

st.set_page_config(page_title="Hattrick", layout="wide")


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_models(_cfg, artifact_mtime):
    """One fitted model per league - never a single pooled model."""
    return load_league_models(_cfg)


# Kickoffs are stored in UTC. Render them wherever the person running this
# actually is; fall back to UTC if the host has no usable local zone.
try:
    LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc
except Exception:  # pragma: no cover - defensive
    LOCAL_TZ = timezone.utc


def outcome_bar_chart(result: dict, home: str, away: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[result["home_win"], result["draw"], result["away_win"]],
        y=[f"{home} win", "Draw", f"{away} win"],
        orientation="h", text=[f"{v:.1%}" for v in result.values()], textposition="auto",
    ))
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%")
    return fig


def reliability_diagram(calibration: pd.DataFrame, outcome: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                               line=dict(dash="dash", color="gray")))
    for source, color in [("model", "#2563eb"), ("market", "#dc2626")]:
        values = calibration[(calibration["outcome"] == outcome) & (calibration["source"] == source)]
        if values.empty:
            continue
        fig.add_trace(go.Scatter(x=values["mean_predicted"], y=values["empirical_frequency"],
                                 mode="markers+lines", name=source.capitalize(),
                                 marker=dict(color=color, size=8)))
    fig.update_layout(height=350, xaxis_title="Mean predicted probability",
                      yaxis_title="Empirical frequency", xaxis_range=[0, 1], yaxis_range=[0, 1])
    return fig


cfg = get_config()
_paths = [model_artifact_path(cfg, c) for c in league_codes(cfg)]
artifact_mtime = tuple(p.stat().st_mtime_ns if p.exists() else None for p in _paths)
models = get_models(cfg, artifact_mtime)

st.title("⚽ Hattrick - multi-league soccer forecasting")

if not models:
    st.warning(
        "No fitted models found. Run this first, from the repo root:\n\n"
        "```\npython scripts/update_data.py\npython scripts/fit_model.py\n```"
    )
    st.stop()

FITTED = [c for c in league_codes(cfg) if c in models]

RAW_DIR = resolve_path(cfg["data"]["raw_dir"])


def _data_age(path):
    if not path.exists():
        return None
    return pd.Timestamp.now(tz="UTC") - pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")


def _fmt_age(delta):
    if delta is None:
        return "missing"
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() // 60)}m ago"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


_sched = RAW_DIR / "fixtures_full.csv"
_ages = {"results": _data_age(resolve_path(cfg["data"]["processed_dir"]) / "matches.csv"),
         "schedule": _data_age(_sched if _sched.exists() else RAW_DIR / "fixtures.csv")}
_stale = any(a is None or a > pd.Timedelta(days=2) for a in _ages.values())

st.caption(
    "Fitted leagues: " + ", ".join(f"{league_name(cfg, c)} ({c})" for c in FITTED)
    + f"  |  results updated {_fmt_age(_ages['results'])}"
    + f", schedule {_fmt_age(_ages['schedule'])}"
    + f"  |  now {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M')} UTC"
)
if _stale:
    st.warning("Data is more than two days old. Re-run `uv run scripts/run_pipeline.py` to refresh results, "
               "the fixture schedule, and every league's fit.")


def show_kickoff(ts) -> str:
    """Kickoffs are stored in UTC; show them in the viewer's local zone."""
    if pd.isna(ts):
        return "TBC"
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ).strftime("%a %d %b, %H:%M")


def no_fixtures_message(code):
    """Say WHY there is nothing to show.

    An empty fixture list is normal - between matchweeks, or before a league
    starts - but an unexplained empty dashboard reads as a broken app, so
    distinguish the causes.
    """
    try:
        played = load_fixtures(RAW_DIR, code, upcoming_only=False)
    except FileNotFoundError:
        return ("No fixture data downloaded yet. Run `uv run scripts/run_pipeline.py` (or "
                "`uv run scripts/update_data.py`).")
    if played.empty:
        return (f"No schedule published yet for {league_name(cfg, code)}. Leagues start at "
                f"different times, and Understat publishes each season's schedule shortly "
                f"before it begins. Nothing is wrong - re-run `uv run scripts/run_pipeline.py` closer to kickoff.")
    return (f"Every fixture on file for {league_name(cfg, code)} has already kicked off "
            f"(latest was {show_kickoff(played['kickoff'].max())}). Run `uv run scripts/run_pipeline.py` to pull "
            f"the next matchweek.")

tab_fixtures, tab_table, tab_validation = st.tabs(
    ["Fixtures", "Season Table", "Model Validation"])

# ---------------------------------------------------------------- Fixtures
with tab_fixtures:
    code = st.selectbox("League", FITTED, format_func=lambda c: league_name(cfg, c),
                        key="fixtures_league")
    model = models[code]
    try:
        fixtures = load_fixtures(RAW_DIR, code)
    except FileNotFoundError:
        fixtures = pd.DataFrame()

    if fixtures.empty:
        st.info(no_fixtures_message(code))
    else:
        horizon = st.slider("Show fixtures within (days)", 1, 60, 14, key="fixtures_horizon")
        cutoff = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=horizon)
        window = fixtures[fixtures["kickoff"] <= cutoff]

        known = window[window["home"].isin(model.params.attack)
                       & window["away"].isin(model.params.attack)]
        unknown = len(window) - len(known)
        st.caption(f"{len(known)} upcoming fixture(s) in the next {horizon} days"
                   + (f"; {unknown} hidden (promoted team with no fitted rating yet)"
                      if unknown else ""))
        if window.empty:
            st.info(f"Nothing within {horizon} days. Next kickoff is "
                    f"{show_kickoff(fixtures['kickoff'].min())}. Widen the slider to see more.")
        elif known.empty:
            st.info("Fixtures were found, but none involve teams the model has fitted "
                    "ratings for yet - typically newly promoted sides.")
        for _, row in known.iterrows():
            home, away = row["home"], row["away"]
            m = all_markets(model, home, away, cfg["model"]["max_goals"])
            with st.container(border=True):
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.subheader(f"{home} vs {away}")
                    st.caption(show_kickoff(row["kickoff"]))
                    ou = m["over_under_2_5"]; b = m["btts"]
                    st.write(f"**Over/Under 2.5:** Over {ou['over']:.1%} / Under {ou['under']:.1%}")
                    st.write(f"**BTTS:** Yes {b['yes']:.1%} / No {b['no']:.1%}")
                    top = ", ".join(f"{s[0]}-{s[1]} ({p:.1%})" for s, p in m["top_scorelines"][:3])
                    st.write(f"**Likely scorelines:** {top}")
                with c2:
                    st.plotly_chart(outcome_bar_chart(m["result"], home, away),
                                     use_container_width=True, key=f"chart_{home}_{away}")

# ------------------------------------------------------------- Season Table
with tab_table:
    code = st.selectbox("League", FITTED, format_func=lambda c: league_name(cfg, c),
                        key="table_league")
    sim_path = sim_artifact_path(cfg, code)
    if not sim_path.exists():
        st.info(
            "No season simulation found. Run:\n\n"
            "```\npython scripts/simulate_season.py\n```"
        )
    else:
        df = pd.read_json(sim_path)
        st.caption(
            "Monte Carlo simulation of remaining fixtures. NOTE: by default this only "
            "simulates near-term fixtures.csv, not the full remaining season - see "
            "scripts/simulate_season.py for how to supply a fuller schedule."
        )
        st.dataframe(
            df.style.background_gradient(subset=["title_%", "top4_%"], cmap="Greens")
                    .background_gradient(subset=["relegation_%"], cmap="Reds"),
            use_container_width=True, hide_index=True,
        )

# ------------------------------------------------------------ Model Validation
with tab_validation:
    per_match_path = resolve_path("artifacts/backtest_per_match.csv")
    calibration_path = resolve_path("artifacts/backtest_calibration.csv")
    if not per_match_path.exists() or not calibration_path.exists():
        st.info("No backtest validation artifacts found. Run the pipeline first.")
    else:
        per_match = pd.read_csv(per_match_path, parse_dates=["date"])
        calibration = pd.read_csv(calibration_path)
        model_probs = [{"home_win": r.model_home_win, "draw": r.model_draw, "away_win": r.model_away_win}
                       for r in per_match.itertuples()]
        actuals = per_match["actual"].tolist()
        st.metric("Model log loss", f"{log_loss(model_probs, actuals):.4f}")
        outcome = st.radio("Outcome", OUTCOMES, horizontal=True)
        st.plotly_chart(reliability_diagram(calibration, outcome), use_container_width=True)
        ece = expected_calibration_error(model_probs, actuals, outcome)
        st.caption(f"Model expected calibration error: {ece:.4f}")
