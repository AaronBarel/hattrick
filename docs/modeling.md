# Modeling

Everything here is implemented in `model/dixon_coles.py` unless stated.

## The base model

Each match produces two goal counts, modelled as Poisson variables whose
rates are multiplicative in team strength:

```
lambda = attack_home * defense_away * gamma        (home goals)
mu     = attack_away * defense_home                (away goals)

X ~ Poisson(lambda),  Y ~ Poisson(mu)
```

`attack_i` is scoring strength; `defense_j` is **conceding** strength, so a
*higher* defense value means a weaker defense — it multiplies the opponent's
rate. `gamma` is the home-advantage multiplier, applied only to the home rate.

### Why multiplicative

A log-linear rate is the natural link for count data: it keeps rates positive
without constraints, and makes "twice as good an attack" mean the same thing
at every opponent strength. Taking logs turns the model into a two-way
additive layout, which is what makes the IPF warm start below work in closed
form.

### The low-score correction

Independent Poissons underproduce the draws that real football generates,
particularly 0-0 and 1-1. Dixon & Coles (1997) correct the four lowest cells
with a dependence parameter `rho`:

```
tau(0,0) = 1 - lambda*mu*rho
tau(0,1) = 1 + lambda*rho
tau(1,0) = 1 + mu*rho
tau(1,1) = 1 - rho
tau(x,y) = 1 otherwise
```

The joint likelihood is `tau(x,y) * Pois(x;lambda) * Pois(y;mu)`. This is a
local patch on the joint distribution, not a general correlation structure —
it touches only those four cells and leaves everything else independent.

### Time decay

Recent matches matter more. Each match `d` days before the fitting date gets
weight `exp(-xi * d)`, with `xi = 0.0018/day` (half-life ≈ 385 days).

**A known weakness:** decay is purely calendar-based, so an 80-day summer
break retains 87% of May's weight. The model therefore carries a squad's
pre-transfer-window form into a new season almost undiminished. See
`findings.md` for the measured size of the resulting early-season error.

## Fitting: two stages

### Stage 1 — IPF warm start

With `rho = 0` the log-likelihood is separable, and the maximum-likelihood
fixed point is a ratio of weighted goals to weighted expected goals. Iterating
these updates (iterative proportional fitting) converges fast and lands close
to the optimum, which matters because stage 2 is a ~60-dimensional non-convex
problem where the starting point decides whether it converges at all.

Each rate carries a Gamma(kappa, kappa) prior (mean 1.0, the neutral value
under this parametrisation), giving

```
alpha_k = (sum_w goals + kappa) / (sum_w expected + kappa)
```

i.e. the MLE plus one pseudo-match at exactly league average
(`prior_strength`, default 1.0).

**Why the prior is not optional.** Without it the update sits on a boundary
whenever a team's weighted numerator is zero — a promoted side that has not
yet conceded gets `defense = 0` exactly. Stage 2 then starts from
`log(0) = -inf`, L-BFGS-B returns ABNORMAL, and *every* team's rating comes
back from a failed optimisation with a NaN log-likelihood and rho stuck at
its initial value. Nothing raises. This reproduced on real Premier League
data in 2025/26.

### Stage 2 — penalized likelihood

L-BFGS-B over `log(attack)`, `log(defense)`, `log(gamma)`, `rho`, minimising
the negative weighted log-likelihood plus two penalties.

**Identifiability anchor.** The parametrisation is invariant under
`attack -> c*attack, defense -> defense/c`, so the scale must be pinned or the
optimizer wanders along a ridge. Penalty `1000 * (mean(log alpha)^2 +
mean(log beta)^2)`.

**Partial pooling.** A Gaussian prior on each team's log-rating about the
league mean log-rating:

```
penalty += (1 / (2*sd^2)) * sum_k (log r_k - mean(log r))^2
```

`rating_prior_sd = 0.35`, chosen a priori from the empirical spread of
log-ratings (attack spans ~0.5–1.7), **not** tuned on the holdout. Shrinking
toward the *mean* rather than toward zero keeps this orthogonal to the anchor:
it pools teams without constraining the league's overall scoring level.

This is the load-bearing fix for boundary ratings — the unpenalized optimizer
drives a zero-evidence rating to ~1e-3 regardless of where it starts, which
downstream reads as "the opponent scores zero goals with probability ~1", i.e.
a confident, badly wrong forecast.

A failed optimisation now emits a `RuntimeWarning` rather than returning
silently.

## The continuous (xG) target

`scipy.stats.poisson.logpmf` returns `-inf` for non-integer input, correctly —
a Poisson variate cannot equal 1.34. `poisson_logpmf()` computes the same
expression treated as a continuous quasi-likelihood:

```
log L(x; lambda) = x*log(lambda) - lambda - gammaln(x + 1)
```

`gammaln(x+1) = log(x!)` on integers and interpolates smoothly between them,
so this is numerically identical to scipy on integer inputs (asserted to 1e-12
in the tests) and finite in between.

**It is a quasi-likelihood, not a likelihood.** The score equations remain
unbiased for the mean, so fitted rates stay consistent, but the implied
`Var = mean` is wrong for xG, which is far less dispersed than a goal count.
Point estimates are fine; **no standard error may be read off it**, and none
is reported anywhere.

### rho cannot come from xG

`tau` only touches the four lowest *integer* cells. A continuous target never
lands on them, so tau is identically 1, rho has exactly zero gradient, and
the optimizer would return its starting value while reporting success. So
when `target="xg"`:

1. Fit `attack`, `defense`, `gamma` on xG with rho absent.
2. Hold those fixed and estimate rho alone from the **actual scorelines** — a
   well-conditioned 1-D maximisation, since only the tau term depends on rho.

Partial xG coverage falls back to actual goals per match, recorded in
`params.xg_coverage`. That mixes two targets with different noise levels and
weights them as equally precise — a mild misspecification, which is why the
coverage figure is recorded rather than assumed.

## The promoted-team prior

Optional (`promoted_teams`), **off by default because it was measured not to
help** — see `findings.md`.

Plain partial pooling shrinks a data-poor team toward the league average. For
a newly promoted side that target is wrong and the error is one-directional:
promoted teams average 0.91 points per game against 1.45. So promoted teams
are shrunk toward `mean + offset` instead, with offsets measured by fitting
each league-season in isolation:

```
attack offset  = -0.2465   (x0.78)
defense offset = +0.2183   (x1.24)
```

A single pooled offset is used rather than per-league ones: each league
contributes only ~8 promoted teams, and the spread *within* promoted teams
(sd 0.18/0.19) is already smaller than the prior's own width of 0.35.

**The offset fades with evidence** (`promoted_fade_matches`, default 30, on
time-decayed match count). "Promoted" and "unknown to the model" are different
sets: over a multi-season window most promoted sides are yo-yo clubs, and 8 of
14 flagged for the 2025/26 holdout already had 34–76 top-flight matches.
Handing those a relegation-candidate prior is pure bias, and measurably made
forecasts worse before the fade existed.

## From rates to match probabilities

`scorelines.py` builds a `(max_goals+1) x (max_goals+1)` scoreline matrix from
`lambda`, `mu`, `rho`, renormalises it (truncation at `max_goals = 10` loses a
little mass), and reads every derived probability off it: W/D/L from the
triangles, over/under 2.5 goals by summing the anti-diagonals, both teams to
score from the complement of the first row and column. One matrix for
everything, so all reported probabilities are mutually consistent by
construction.
