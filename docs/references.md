# References

## Core method

- **Dixon, M. J., & Coles, S. G. (1997).** *Modelling Association Football
  Scores and Inefficiencies in the Football Betting Market.* Journal of the
  Royal Statistical Society: Series C, 46(2), 265–280.
  The model this project implements: multiplicative Poisson rates, the
  low-score `tau` correction, and exponential time decay.

- **Maher, M. J. (1982).** *Modelling Association Football Scores.*
  Statistica Neerlandica, 36(3), 109–118.
  The independent-Poisson predecessor; Dixon-Coles is Maher plus dependence
  and decay.

- **Karlis, D., & Ntzoufras, I. (2003).** *Analysis of sports data by using
  bivariate Poisson models.* JRSS Series D, 52(3), 381–393.
  Alternative treatment of the goal-count dependence that `rho` patches.

## Forecast evaluation

- **Gneiting, T., & Raftery, A. E. (2007).** *Strictly Proper Scoring Rules,
  Prediction, and Estimation.* JASA, 102(477), 359–378.
  Why log loss and Brier are used and accuracy is not.

- **Shin, H. S. (1993).** *Measuring the Incidence of Insider Trading in a
  Market for State-Contingent Claims.* Economic Journal, 103(420), 1141–1153.
  The de-vigging correction for favorite-longshot bias that we deliberately
  do not implement (`evaluation.md` explains why).

- **Efron, B., & Tibshirani, R. (1993).** *An Introduction to the Bootstrap.*
  Chapman & Hall. Underlies `paired_bootstrap_test`.

## Ratings and uncertainty over time

- **Glickman, M. E. (1999).** *Parameter estimation in large dynamic paired
  comparison experiments.* JRSS Series C, 48(3), 377–394.
  Glicko's ratings-deviation inflation over idle periods — the standard
  treatment of the season-boundary problem this project has not yet
  implemented.

- **An Adaptive Glicko-2 Rating Framework for Probabilistic Football
  Forecasting and Season Simulation** — https://arxiv.org/pdf/2607.01722
  Includes handling of teams moving between leagues, i.e. the promoted-team
  prior problem.

- **Bayesian weighted discrete-time dynamic models for association football
  prediction** — https://arxiv.org/pdf/2508.05891

## Expected goals

- **Expected goals in football: Improving model performance and demonstrating
  value.** PLOS ONE (2023).
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282295

- **What are Expected Goals (xG)?** Hudl.
  https://www.hudl.com/blog/expected-goals-xg-explained
  Both support xG over goals for forecasting *small-sample aggregates* — which
  is not the regime this model operates in. See `findings.md` §4.

## Data sources

| Source | URL | Used for |
|---|---|---|
| football-data.co.uk | https://www.football-data.co.uk | results, odds, fixtures, second tiers |
| Understat | https://understat.com | match xG, forward schedule |
| `soccerdata` | https://github.com/probberechts/soccerdata | maintained scraper wrapper |
| FBref | https://fbref.com | alternative xG (StatsBomb-powered), unused |
| StatsBomb Open Data | https://github.com/statsbomb/open-data | event-level, unused |

## Learning material

Aimed at someone already comfortable with classical ML, wanting the sports-
specifics of forecasting a low-scoring sport.

**Sports modeling**
1. Dixon & Coles (1997) — read the actual paper; it is short and this repo is
   an implementation of it.
2. *The Numbers Game* — Anderson & Sally. Why football is a low-information,
   high-variance sport, which is the constraint behind every result here.
3. **Understat / FBref** — build intuition for how much of a scoreline is
   noise before trusting any model.
4. **StatsBomb open data + their xG write-ups** — the shot-level modeling
   that match-level xG collapses away.
5. *Soccermatics* — David Sumpter. Poisson models for football, worked.
