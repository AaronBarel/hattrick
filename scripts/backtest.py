#!/usr/bin/env python
"""Walk-forward backtest: how good are the model's probabilities, really?

Fits on everything before --holdout-start, then evaluates match by match
through the holdout period (refitting periodically), scoring against actual
results with log loss / Brier score, and against Bet365's de-vigged closing
odds where available as the "did we beat the market" benchmark.

Usage:
    python scripts/backtest.py --holdout-start 2024-08-01
"""
from __future__ import annotations

import argparse

import pandas as pd

from hattrick.config import load_config, resolve_path
from hattrick.evaluation import OUTCOMES, expected_calibration_error, walk_forward_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-start", type=str, required=True,
                         help="YYYY-MM-DD - matches before this date are training data only")
    parser.add_argument("--refit-every", type=int, default=10,
                         help="refit the model every N holdout matches (default: ~1 gameweek)")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--save-csv", type=str, default="artifacts/backtest_per_match.csv")
    parser.add_argument("--save-calibration", type=str, default="artifacts/backtest_calibration.csv")
    args = parser.parse_args()

    cfg = load_config()
    matches_path = resolve_path(cfg["data"]["processed_dir"]) / "matches.csv"
    if not matches_path.exists():
        raise SystemExit(f"{matches_path} not found - run scripts/update_data.py first.")

    matches = pd.read_csv(matches_path, parse_dates=["date"])
    print(f"Backtesting on {len(matches)} matches, holdout starting {args.holdout_start} ...")

    result = walk_forward_backtest(
        matches, holdout_start=args.holdout_start, xi=cfg["model"]["xi"],
        refit_every=args.refit_every, max_goals=cfg["model"]["max_goals"],
    )

    print()
    print(result.summary())

    out_path = resolve_path(args.save_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.per_match.to_csv(out_path, index=False)
    print(f"\nPer-match predictions saved -> {out_path}")

    calibration_frames = []
    model_probs = result.model_probs_list(result.per_match)
    actuals = result.per_match["actual"].tolist()
    for outcome in OUTCOMES:
        curve = result.calibration(outcome, n_bins=args.calibration_bins, source="model")
        curve["outcome"], curve["source"] = outcome, "model"
        calibration_frames.append(curve)
        if result.market_log_loss is not None:
            curve = result.calibration(outcome, n_bins=args.calibration_bins, source="market")
            curve["outcome"], curve["source"] = outcome, "market"
            calibration_frames.append(curve)

    calibration_path = resolve_path(args.save_calibration)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(calibration_frames, ignore_index=True).to_csv(calibration_path, index=False)
    print(f"Calibration curves saved -> {calibration_path}")
    print("\nExpected Calibration Error:")
    for outcome in OUTCOMES:
        ece = expected_calibration_error(model_probs, actuals, outcome, args.calibration_bins)
        print(f"  {outcome:10s} model ECE = {ece:.4f}")


if __name__ == "__main__":
    main()
