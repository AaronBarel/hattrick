# Evaluation

If the measurement is wrong, every conclusion downstream is wrong. This is the
part of the project most worth being pedantic about.

## Metrics

**Log loss** and **Brier score**, multiclass over `{home_win, draw, away_win}`,
lower is better. Both are *proper* scoring rules: a forecaster cannot improve
their expected score by reporting anything other than their true belief. Plain
accuracy is not proper — it rewards confident guessing — and is never used.

Log loss is the headline because it punishes confident errors hardest, which
is the failure mode that matters most for a forecast anyone acts on.

## Calibration

Log loss is an *average*, and averages hide a failure that matters: being
systematically overconfident in one probability band while compensating
elsewhere. `calibration_curve()` bins predictions and compares
mean predicted probability against observed frequency;
`expected_calibration_error()` reduces that to one number.

Points below the diagonal mean the model claims more confidence than outcomes
justify.

## Walk-forward protocol

`walk_forward_backtest()`. Not a single train/test split:

1. Fit on everything before `holdout_start`.
2. Step through the holdout match by match, predicting each before seeing it.
3. Refit every `refit_every` matches, folding in newly completed results.

This mirrors how the system would actually run. Nothing after a match ever
informs the prediction made for it.

`refit_every` trades fidelity against runtime (10 for careful work, 20 for
sweeps). It does move results slightly — a difference that flips significance
when you change the refit cadence is not a real difference.

## Choosing the benchmark

The market comparison is only meaningful against the **closing** line. The
preference order in `resolve_market_source()`:

1. `closing_pinnacle` — Pinnacle is low-margin, high-limit and welcomes sharp
   money, so its close is the sharpest single-book estimate available free.
2. `closing_avg` — cross-book average close.
3. `closing_b365` — Bet365 close.
4. `preclosing_b365` — **last resort**, clearly labeled as stale.

A source must cover **≥90% of holdout matches** to be selected. Pinnacle's
columns really do go blank partway through recent seasons, and scoring on the
surviving 55% would compare against a non-random subsample.

`BacktestResult.market_label` records which snapshot was used. **Read it
before quoting any "beats the market" claim.**

### De-vigging

Bookmaker odds embed a margin: `1/H + 1/D + 1/A > 1`. We divide each implied
probability by their sum (the multiplicative/proportional method) before
comparing, so it is model-vs-fair-value rather than model-vs-retail-price.
Beating the raw price is a much lower bar and tells you nothing about skill.

Proportional de-vigging assumes margin is spread evenly across outcomes. In
practice it is applied slightly more heavily to longshots (the
favorite-longshot bias), so this mildly overstates longshot probabilities.
Shin's method corrects for it; we do not implement it, because the effect is
small relative to the differences being detected and the same de-vig is
applied to both sides of every comparison.

## Significance

**A point estimate is not a result.** One season is ~380 matches per league;
differences of 0.01–0.02 nats are routinely indistinguishable from noise, and
reading them as real is the easiest way to adopt a feature that does nothing.

`paired_bootstrap_test()` resamples matches with replacement (default 10,000
draws) to build a confidence interval on the *difference* in mean per-match
loss. Paired, because both models saw the same fixtures — comparing unpaired
means throws away the pairing and inflates the interval.

## Comparing configurations

`experiments.compare_configs()` exists so the delicate parts have one tested
implementation rather than being copy-pasted between notebook cells. It
enforces:

- **Configs are scored on the same fixtures** (`align()`). A variant that
  skipped matches would otherwise be judged on an easier subset.
- **Losses pool across leagues at the per-match level**, so a 380-match league
  is not given the same weight as a 306-match one.
- **Every difference carries a bootstrap CI**, and `format_comparison()` spells
  out the verdict in words.

## Protocol for tuning

Any hyperparameter chosen by looking at the holdout has contaminated it.

The protocol used here: **select on a validation season, score once on a later
test season.** For example `xi` was swept on 2024/25 and the winner scored on
2025/26. Where a value was instead chosen a priori (`rating_prior_sd = 0.35`,
from the empirical spread of log-ratings), that is stated explicitly, and the
sensitivity curve is reported as *descriptive* rather than as a selection.

## The bar for new work

Log loss going down is necessary but not sufficient, for two reasons this
project has measured rather than assumed.

**Differences at this sample size are usually noise.** A season is ~380
matches per league; a 0.01-nat movement is routinely indistinguishable from
zero. Report a confidence interval or do not report the result.

**The model does not beat the closing line.** Pooled over five leagues in
2025/26 it scores 0.9952 against the closing line's 0.9778. Any change should
be judged on whether it closes that gap, not on whether the number drifted
down. A second diagnostic from `findings.md`: model-market **disagreement size
is an inverse quality indicator** — the further the model departs from the
closing line, the more reliably it is the one that is wrong. A change that
increases large disagreements without improving the tail has made things
worse.

## Notebooks

- `01_model_evaluation.ipynb` — per-league backtest, reliability diagrams,
  ECE, cumulative model-minus-market curve (which distinguishes "consistently
  slightly worse" from "one catastrophic night").
- `02_feature_ablation.ipynb` — configuration comparison with bootstrap
  intervals, and a per-league breakdown.

Both fall back to a synthetic league with known ratings when real data is
absent, so every cell runs on a fresh clone.
