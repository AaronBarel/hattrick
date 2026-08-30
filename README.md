# Hattrick

A multi-league Dixon-Coles forecasting model for soccer: match outcome
probabilities, derived markets (over/under, both-teams-to-score, correct
score), and Monte Carlo season simulation — for the Premier League, La Liga,
Serie A, Bundesliga and Ligue 1. Every modeling choice traces to a
derivation or to a measurement recorded in [`docs/findings.md`](docs/findings.md),
and the results reported below include the ones that did not work.

## Status

- Fits **one Dixon-Coles model per league** 
- Uses scoreline matrix to produce W/D/L, over/under, BTTS and correct-score probabilities for upcoming fixtures
- Simulates the rest of a season by Monte Carlo for title, top-four and
  relegation probabilities.
- Evaluates walk-forward backtesting against actual results
  **and against the bookmaker's closing line**, with paired bootstrap
  confidence intervals and calibration curves.

**The model does not beat the bookmaker closing line.** Pooled over five
leagues across the 2025/26 season (1,738 matches):

| | Log loss |
|---|---|
| This model | 0.9952 |
| De-vigged closing line | **0.9778** |

Things that were tried and **did not help**, each measured rather than
assumed: fitting on expected goals instead of goals (+0.0009, p=0.72), a
promoted-team prior (+0.0003, p=0.80), and using a promoted team's
second-tier record as a predictor (correlation −0.08). Details and
confidence intervals in [`docs/findings.md`](docs/findings.md).

## Quick start

```bash
git clone https://github.com/<you>/hattrick
cd hattrick
uv sync                                     # core + test dependencies

uv run scripts/update_data.py               # results + odds from football-data.co.uk
uv run scripts/fit_model.py                 # one model per league -> artifacts/{league}/
uv run scripts/predict_fixtures.py          # upcoming fixture probabilities
uv run scripts/simulate_season.py           # Monte Carlo season tables
```

`uv run scripts/run_pipeline.py` runs the whole chain in order.

### Dashboard

```bash
uv run --extra ui streamlit run app.py
```

Opens on `http://localhost:8501`. It is a view layer over whatever the
pipeline last wrote, so run the steps above first — with no `artifacts/` it
starts fine but has nothing to show.

### Optional extras

```bash
uv sync --extra xg          # match xG from Understat, and the forward schedule
uv sync --extra ui          # Streamlit dashboard
uv sync --extra notebooks   # Jupyter
uv sync --all-extras        # everything at once
```

Each degrades gracefully when absent: the model falls back to actual goals
without `xg`, and only the dashboard needs `ui`.

## Model

Goals are modelled as Poisson counts with multiplicative rates:

```
lambda = attack_home * defense_away * gamma      (home goals)
mu     = attack_away * defense_home              (away goals)
```

plus the Dixon-Coles `tau` correction on the four lowest scorelines (real
football produces more 0-0s and 1-1s than independent Poissons predict), and
exponential time decay so recent matches count for more.

Fitting is two-stage: a closed-form IPF warm start, then L-BFGS-B on the
penalized likelihood. Two priors matter — a Gamma prior keeping every warm-
start rate positive, and **partial pooling** shrinking each team's log-rating
toward the league mean. Without the latter, a team with no goals in the
fitting window sits on a likelihood boundary, the optimizer fails, and
*every* rating comes back from a failed fit with a NaN likelihood. Nothing
raises.

## Documentation

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Module map, data flow, artifacts |
| [`docs/data.md`](docs/data.md) | Sources, schemas, club-name resolution, upstream hazards |
| [`docs/modeling.md`](docs/modeling.md) | Dixon-Coles derivation, priors, continuous xG target |
| [`docs/evaluation.md`](docs/evaluation.md) | Backtest protocol, scoring rules, significance |
| [`docs/findings.md`](docs/findings.md) | Every measured result, including the negative ones |
| [`docs/assumptions.md`](docs/assumptions.md) | Standing assumptions and where they break |
| [`docs/references.md`](docs/references.md) | Literature and data sources |

## Notebooks

```bash
uv sync --extra notebooks
cd notebooks && uv run jupyter nbconvert --to notebook --execute --inplace \
    01_model_evaluation.ipynb 02_feature_ablation.ipynb
```

- `01_model_evaluation.ipynb` — walk-forward backtest per league, reliability
  diagrams, expected calibration error, cumulative model-vs-market curve.
- `02_feature_ablation.ipynb` — does a change actually help? Every comparison
  carries a paired bootstrap interval.

## A note on data quality

Every data bug found in this project has been **silent**, which shaped how
the loader is written. football-data.co.uk switched to BOM-prefixed UTF-8
partway through its archive (nulling the league column, which would have
dropped half the dataset from every fit); it answers a request for an
unpublished season with an HTML page at HTTP 300; and it renamed two La Liga
clubs mid-archive, which split one club into two teams.

The rule the loader follows: **degrade gracefully when data is legitimately
absent, fail loudly when it is wrong.** Required columns are asserted, not
hoped for, and the xG merge verifies its join by comparing both sources'
scorelines rather than trusting a coverage percentage.

## Tests

```bash
uv run pytest                        # 128 tests
uv run pytest tests/test_docs.py     # check the docs match the code
```

## License

MIT. Data belongs to its respective providers; please respect their terms.

Development of this project was assisted with Claude Code
