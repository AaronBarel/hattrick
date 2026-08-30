#!/usr/bin/env python
"""Run the complete forecasting pipeline in dependency order."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_step(name: str, script: str, *args: str, optional: bool = False) -> None:
    """Run one pipeline step.

    `optional=True` steps enhance the pipeline but must never take it down -
    they depend on scraped sources with no availability guarantee, and the
    model degrades sensibly without them (goals instead of xG, the rolling
    fixtures file instead of the full schedule).
    """
    print(f"\n=== {name} ===", flush=True)
    try:
        subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        if not optional:
            raise
        print(f"[skipped] {name} failed (exit {exc.returncode}); continuing without it.",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-start", default="2024-08-01",
                        help="Backtest holdout date (YYYY-MM-DD).")
    parser.add_argument("--refit-every", type=int, default=10,
                        help="Refit the backtest model every N matches.")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="Number of season simulations (uses config default when omitted).")
    parser.add_argument("--skip-update", action="store_true",
                        help="Reuse existing raw and processed data instead of downloading.")
    parser.add_argument("--skip-xg", action="store_true",
                        help="Do not fetch xG (faster; the model uses actual goals).")
    parser.add_argument("--promoted-prior", action="store_true",
                        help="Enable the promoted-team prior in fitting. Measured as no "
                             "better than the default - see the README.")
    parser.add_argument("--use-xg", action="store_true",
                        help="Fit on expected goals. Measured as no better than goals "
                             "on this data - see the README before enabling.")
    args = parser.parse_args()

    if not args.skip_update:
        run_step("Update data", "update_data.py")
        # Both optional: scraped, and the pipeline degrades cleanly without them.
        if not args.skip_xg:
            run_step("Update xG", "update_xg.py", optional=True)
        run_step("Update fixtures", "update_fixtures.py", optional=True)

    fit_args = (["--use-xg"] if args.use_xg else []) + \
               (["--promoted-prior"] if args.promoted_prior else [])
    run_step("Fit model", "fit_model.py", *fit_args)
    run_step("Predict fixtures", "predict_fixtures.py")

    simulation_args = [] if args.n_sims is None else ["--n-sims", str(args.n_sims)]
    run_step("Simulate season", "simulate_season.py", *simulation_args)
    run_step("Backtest", "backtest.py", "--holdout-start", args.holdout_start,
             "--refit-every", str(args.refit_every))
    print("\nPipeline complete. Explore outputs in notebooks/01_model_evaluation.ipynb.")


if __name__ == "__main__":
    main()
