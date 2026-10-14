# Architecture

## Shape of the thing

Five independent per-league models behind one shared pipeline. Data flows
strictly one way; nothing writes backwards.

```
football-data.co.uk ──┐
  results, odds       │
                      ├──► data/raw/*.csv ──► loader ──► data/processed/matches.csv
Understat (soccerdata)│                                          │
  xG, forward schedule┘                                          │
                                                                 ▼
                                            DixonColesModel.fit()  (one per league)
                                                                 │
                                            artifacts/{league}/dixon_coles_params.json
                                                                 │
                          ┌──────────────────────┼────────────────────────┐
                          ▼                      ▼                        ▼
                  scorelines.py            season_sim.py            evaluation.py
                  (per-fixture             (Monte Carlo             (walk-forward
                   W/D/L, O/U, BTTS)        league table)            backtest)
                          │                      │                        │
                          │                      ▼                        ▼
                          │            artifacts/{league}/          experiments.py
                          │                season_sim.json          (config comparison)
                          │                      │                        │
                          └──────────────────────┴────────────────────────┘
                                                 │
                                                 ▼
                                        app.py  (dashboard)
```

**One model per league, never pooled.** A single Dixon-Coles fit assumes one
pool of teams sharing a scoring level, a home-advantage multiplier and a rho.
Leagues share none of those and their teams never meet, so a pooled fit would
rank a Spanish club against an English one on nothing but their leagues' goal
averages. Fitted home advantage ranges 1.19 (Serie A) to 1.39 (Bundesliga).
The per-league artifact directory makes pooling hard to do by accident.

## Modules

### Configuration
- **`config.py`** — loads `config.yaml`; resolves per-league artifact paths
  (`model_artifact_path`, `sim_artifact_path`); `load_league_models()` returns
  `{league_code: DixonColesModel}`. `current_season_start()` derives the
  in-progress season rather than reading a hardcoded date.

### Data (`data/`)
- **`football_data.py`** — downloads results/odds CSVs and the rolling
  fixtures file. Validates payloads are really CSVs (an unpublished season
  returns an HTML page at HTTP 300). `current_season_code()` computes the
  live season so the cache never staleness-traps new results.
- **`loader.py`** — raw CSVs → one clean `matches` frame. Handles mixed
  encodings, enforces required columns, and `canonicalize_teams()` collapses
  duplicate club identities. Also owns `load_fixtures()`, which returns only
  *upcoming* fixtures with tz-aware kickoffs.
- **`xg.py`** — Understat xG via `soccerdata`, plus a source-agnostic CSV
  path. Owns club-name normalization (`normalize_team`, `build_team_mapping`)
  used by every cross-source join in the project.
- **`fixtures.py`** — full forward season schedule from Understat, written to
  `fixtures_full.csv`.
- **`features.py`** — derived per-team features; currently promotion
  detection.
- **`clubelo.py`** — optional Club Elo ratings, used only as a sanity check.

### Model (`model/`)
- **`dixon_coles.py`** — the estimator. IPF warm start, L-BFGS-B refinement,
  partial pooling, optional continuous (xG) target and promoted-team prior.
- **`scorelines.py`** — scoreline matrix → derived match probabilities (W/D/L,
  over/under 2.5 goals, both teams to score, most likely scorelines).
- **`season_sim.py`** — Monte Carlo simulation of a league table from
  remaining fixtures.

### Evaluation
- **`evaluation.py`** — scoring rules, calibration, paired bootstrap,
  benchmark selection (`resolve_market_source`), `walk_forward_backtest` (one
  league) and `backtest_by_league` (each league with its own model, scores
  pooled per match).
- **`experiments.py`** — compares model configurations across leagues with
  significance tests; also provides the synthetic league the notebooks fall
  back to.

## Scripts

Run in this order, or let `run_pipeline.py` do it for you. The Streamlit
dashboard (`uv run streamlit run app.py`) reads whatever the pipeline last
produced and does not refresh anything itself.

| Script | Does | Optional? |
|---|---|---|
| `update_data.py` | results, odds, rolling fixtures | no |
| `update_xg.py` | match xG → merged into `matches.csv` | yes |
| `update_fixtures.py` | full forward season schedule | yes |
| `fit_model.py` | one model per league → `artifacts/{league}/` | no |
| `predict_fixtures.py` | upcoming fixture probabilities | — |
| `simulate_season.py` | league table Monte Carlo | — |
| `backtest.py` | walk-forward evaluation, per league | — |

The two optional steps depend on scraped sources with no availability
guarantee, so `run_pipeline.py` marks them non-fatal: the model falls back to
actual goals, and fixtures fall back to the rolling file.

## Artifacts and generated data

Everything below is reproducible and git-ignored.

```
data/raw/{LEAGUE}_{SEASON}.csv    results + odds
data/raw/fixtures.csv             rolling near-term fixtures
data/raw/fixtures_full.csv        full forward season schedule
data/processed/matches.csv        the one clean table everything fits on
artifacts/{league}/dixon_coles_params.json
artifacts/{league}/season_sim.json
artifacts/backtest_*.csv          backtest output for the dashboard
```

## Conventions

- **League codes** are football-data.co.uk's (`E0`, `SP1`, `I1`, `D1`, `F1`)
  and are the primary key everywhere: config, artifacts, the `league` column,
  and the Understat lookup table.
- **Club names** are football-data.co.uk's spellings. Every other source is
  mapped onto them via `data/xg.py`.
- **Times** are tz-aware UTC wherever a kickoff is involved; sources publish
  UK local time with no marker.
- **Seasons** are compact codes (`2627`) and run August–May; July is the cut.
