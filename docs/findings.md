# Findings

Every measured result, including — especially — the negative ones. **Append
here whenever you measure something.** A dead experiment that nobody recorded
gets re-run.

Unless stated otherwise: walk-forward backtest, holdout from 2025-08-01,
pooled across all five leagues at the per-match level, log loss (lower is
better), significance by paired bootstrap.

## Summary

| # | Change | Verdict |
|---|---|---|
| 1 | Benchmark against true closing odds | **Correctness fix** — old benchmark understated the gap by ~9% |
| 2 | Fix degenerate ratings (boundary + failed optimizer) | **Large win** — 1.0924 → 1.0482 |
| 3 | Partial pooling across teams | **Positive, borderline** — 1.0482 → 1.0245; p moves with configuration |
| 4 | xG as fitting target | **No effect** — +0.0009, p=0.72 |
| 6 | Promoted-team prior | **No effect / slightly negative** — shipped off by default |
| 7 | Second-tier record as promoted predictor | **No signal** — r = −0.08, killed before building |
| 8 | Team-identity canonicalization | **Correctness fix** — one club was fitted as two |

## 1. Pre-closing odds were labeled closing

Same model, four seasons of Premier League data, different benchmarks:

| Benchmark | Market log loss | Model − market |
|---|---|---|
| Bet365 pre-closing | 1.0180 | +0.0744 |
| Bet365 closing | 1.0124 | +0.0799 |
| Cross-book average closing | 1.0111 | +0.0813 |

Scoring against the stale price understated the model's deficit by ~9%.

## 2–3. Degenerate ratings, then partial pooling

A team with zero weighted goals sits on a likelihood boundary. IPF returned
`defense = 0` exactly, `log(0)` broke L-BFGS-B, and **every** team's rating
came back from a failed optimisation with a NaN likelihood and rho stuck at
its initial value. Nothing raised. Real consequence: the model predicted
Arsenal would score **0.001 goals** away at Sunderland — a forecast that is not merely
wrong but confidently, absurdly wrong.

| Fit | Model log loss | vs market (0.9769–1.0111) |
|---|---|---|
| Original | 1.0924 | +0.0813 |
| Optimizer no longer fails | 1.0482 | +0.0371 |
| \+ partial pooling | **1.0245** | +0.0134 |

**Honesty about the pooling row.** The bug fix is unambiguous — a NaN
likelihood is simply wrong. The additional pooling gain is *borderline*: at
tuned `xi=0.0009` with `refit_every=10` it is −0.0139, p=0.016; at default
`xi=0.0018` with `refit_every=20` it is −0.0081, p=0.068. Same direction and
size, crossing the significance line depending on refit cadence. This dataset
cannot resolve it cleanly. Kept because it is theoretically motivated and
because the underlying bug is real regardless.

## 4. xG as the fitting target — no effect

`xi` selected on 2024/25, scored once on 2025/26. n=1,745.

| Config | Log loss | vs goals | 95% CI | p |
|---|---|---|---|---|
| goals | 0.9932 | — | — | — |
| **xG** | 0.9941 | **+0.0009** | [−0.0042, +0.0061] | 0.72 |
| market (closing) | 0.9769 | — | — | — |

Stronger than "no effect detected": the interval **rules out** an improvement
beyond ~0.004 nats. Per-league deltas scatter around zero with no consistent
sign (+0.0009, +0.0008, −0.0045, +0.0062, +0.0002).

Confound tested and eliminated: a less noisy target plausibly wants shorter
memory, so `xi` might have handicapped xG. Sweeping `xi` on the validation
season, **both targets selected 0.0009** and landed within 0.0003 of each
other.

**Why the literature result doesn't transfer.** The usual finding compares
small-sample aggregates (xG over six matches beats goals over six). This model
already pools 2–4 seasons per team with time decay, so the finishing variance
xG removes has largely been averaged away — while xG's cost remains: it is a
model output, and it discards real, persistent finishing and goalkeeping
skill. At this sample size the two cancel.

Pipeline kept: it is the prerequisite for shot-level work, and the negative
result is only trustworthy because it was measured.

## 5. Promoted-team prior — no effect

The underlying fact is real and large. Across five leagues, 2023–25:

| | PPG | Goals for | Goals against |
|---|---|---|---|
| Established | 1.45 | 1.47 | 1.36 |
| Promoted | **0.91** | 1.07 | 1.74 |

Median promoted side finishes at the **85th percentile**. Offsets measured by
fitting each league-season in isolation: attack −0.2465, defense +0.2183
(×0.78, ×1.24), consistent in sign across all five leagues.

The prior does what it says — Hull v Aston Villa moves 41.9% → 31.0% against
the market's 25%; established teams shift under 1.5% — but did not improve
forecasts:

| Slice | Baseline | Promoted prior | p |
|---|---|---|---|
| Whole season | 0.9952 | 0.9954 | 0.80 |
| First 50 matches | 0.9806 | 0.9854 | 0.17 |
| First 50, promoted only | 1.0382 | 1.0579 | 0.18 |
| Disagreement > 15% | +0.0669 | +0.0863 | — |

Slightly worse everywhere, never significantly. **Off by default.**

**Design lesson that survived:** the first version was worse still (+0.0069 on
the first 50). "Promoted" and "unknown to the model" are different sets — over
a multi-season window most promoted sides are yo-yo clubs, and 8 of 14 flagged
for this holdout already had 34–76 top-flight matches (Sassuolo 76, Lorient
72, Köln 68). The offset now fades with time-decayed match count, halving the
damage without turning it positive.

## 6. Second-tier record predicts nothing

All five second tiers are free from the same source, so this was tested rather
than assumed. Across 41 promoted teams:

| Predictor | Correlation with top-flight PPG |
|---|---|
| Second-tier PPG | **−0.08** |
| Second-tier goal difference | +0.06 |
| Second-tier finishing position | −0.04 |

Second-tier *champions* averaged 0.888 PPG in the top flight; playoff
qualifiers averaged 1.018 — backwards. Cause is range restriction: every
promoted side is in the top few percent of the division below, so there is
almost no variance in the predictor.

Killed in ten minutes, before building the pipeline.

## Where season-start error actually lives

Excess loss vs the closing line, 2025/26:

| Stage | Established | Promoted |
|---|---|---|
| First ~40 matches | 0.0225 | 0.0241 |
| Rest of season | 0.0138 | 0.0288 |

Early-season error is **almost identical** for both groups. Promoted fixtures
are 27% of matches and 34% of excess loss — over-represented, not dominant.
So promotion is *not* where most season-start error lives; established teams
change over the summer too, and with `xi=0.0018` an 80-day break retains 87%
of May's weight.

## The result that outranks the rest

Bucketing the holdout by how far the model departs from the closing line:

| Disagreement | n | Model excess log loss |
|---|---|---|
| > 5% | 905 | +0.033 |
| > 10% | 321 | +0.055 |
| > 15% | 87 | +0.067 |
| > 20% | 16 | **+0.166** |

**Disagreement size is an inverse quality indicator.** The further the model
departs from the closing line, the more reliably it is the one that is wrong.
This is the sharpest single statement of where the model currently stands: it
is a reasonable forecaster that is comfortably beaten by the market, and its
confident disagreements are its worst predictions.

## Incidental

- **`xi = 0.0018` is slightly too fast.** Faster decay (0.0030) is
  significantly worse (+0.0028, p=0.013); slower (0.0009) marginally better
  (−0.0012, p=0.22), matching the independent validation sweep. Left unchanged
  because it is not significant on its own.
- **Home advantage and rho genuinely differ by league** — gamma 1.19 (Serie A)
  to 1.39 (Bundesliga); rho +0.015 (La Liga) to −0.088 (Premier League). This
  is the empirical justification for never pooling leagues.
- **xG coverage** is ~100% across all five leagues (7,125 / 7,127 matches).
