"""Monte Carlo season-table simulation.

Why simulation and not a closed form: with 20 interacting teams, "P(Arsenal
finishes top 4)" depends on the joint outcome of ~380 correlated matches (via
shared opponents) - there's no clean analytic formula for that once you're
past win probabilities for a single match. Simulation sidesteps this
entirely: draw one full set of remaining results per trial from the model's
own scoreline distributions, compute that trial's final table, repeat
thousands of times, and the table-outcome probabilities just fall out of the
empirical distribution across trials.

Everything here reuses `model/scorelines.score_matrix` - one fitted model, same
scoreline distributions used for match-level predictions, no separate model
needed for the season view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hattrick.model.dixon_coles import DixonColesModel
from hattrick.model.scorelines import score_matrix


def _sample_scoreline(M: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Draw one (home_goals, away_goals) pair from the scoreline matrix."""
    flat = M.ravel()
    idx = rng.choice(len(flat), p=flat)
    n = M.shape[0]
    return idx // n, idx % n


def _table_from_results(points: dict[str, int], goal_diff: dict[str, int]) -> list[str]:
    """Rank teams: points desc, then goal difference desc (standard EPL
    tiebreak; head-to-head is a further tiebreak we skip for simplicity -
    it almost never changes title/relegation probabilities in practice)."""
    teams = list(points.keys())
    teams.sort(key=lambda t: (-points[t], -goal_diff[t]))
    return teams


@dataclass
class SeasonSimResult:
    teams: list[str]
    n_sims: int
    # per-team: probability of finishing at each rank (1-indexed), and summary stats
    rank_probs: dict[str, list[float]] = field(default_factory=dict)
    title_prob: dict[str, float] = field(default_factory=dict)
    top4_prob: dict[str, float] = field(default_factory=dict)
    relegation_prob: dict[str, float] = field(default_factory=dict)
    expected_points: dict[str, float] = field(default_factory=dict)
    expected_rank: dict[str, float] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for t in self.teams:
            rows.append({
                "team": t,
                "expected_pts": round(self.expected_points[t], 1),
                "expected_rank": round(self.expected_rank[t], 1),
                "title_%": round(100 * self.title_prob[t], 1),
                "top4_%": round(100 * self.top4_prob[t], 1),
                "relegation_%": round(100 * self.relegation_prob[t], 1),
            })
        return pd.DataFrame(rows).sort_values("expected_rank").reset_index(drop=True)


def simulate_season(
    model: DixonColesModel,
    played: pd.DataFrame,
    remaining_fixtures: pd.DataFrame,
    n_sims: int = 10_000,
    max_goals: int = 10,
    relegation_spots: int = 3,
    top4_spots: int = 4,
    seed: int | None = None,
) -> SeasonSimResult:
    """
    played: DataFrame with columns home, away, home_goals, away_goals - results
        already in the books, contribute fixed points to every trial.
    remaining_fixtures: DataFrame with columns home, away - not yet played;
        simulated fresh (independently) in every trial.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(set(played["home"]) | set(played["away"])
                    | set(remaining_fixtures["home"]) | set(remaining_fixtures["away"]))

    # Fixed contribution from already-played matches (same every trial).
    base_points = {t: 0 for t in teams}
    base_gd = {t: 0 for t in teams}
    for _, row in played.iterrows():
        h, a, hg, ag = row["home"], row["away"], int(row["home_goals"]), int(row["away_goals"])
        base_gd[h] += hg - ag
        base_gd[a] += ag - hg
        if hg > ag:
            base_points[h] += 3
        elif hg < ag:
            base_points[a] += 3
        else:
            base_points[h] += 1
            base_points[a] += 1

    # Pre-compute each remaining fixture's scoreline matrix once - it doesn't
    # change across trials, only the random draw from it does.
    remaining = []
    for _, row in remaining_fixtures.iterrows():
        h, a = row["home"], row["away"]
        if h not in model.params.attack or a not in model.params.attack:
            continue  # unseen team (e.g. promoted side with no fitted rating) - skip
        remaining.append((h, a, score_matrix(model, h, a, max_goals)))

    rank_counts = {t: np.zeros(len(teams), dtype=int) for t in teams}
    points_sum = {t: 0 for t in teams}
    rank_sum = {t: 0 for t in teams}

    for _ in range(n_sims):
        points = dict(base_points)
        gd = dict(base_gd)
        for h, a, M in remaining:
            hg, ag = _sample_scoreline(M, rng)
            gd[h] += hg - ag
            gd[a] += ag - hg
            if hg > ag:
                points[h] += 3
            elif hg < ag:
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1

        table = _table_from_results(points, gd)  # rank 1 = table[0]
        for rank, team in enumerate(table):
            rank_counts[team][rank] += 1
            rank_sum[team] += rank + 1
        for t in teams:
            points_sum[t] += points[t]

    n_teams = len(teams)
    result = SeasonSimResult(teams=teams, n_sims=n_sims)
    for t in teams:
        probs = rank_counts[t] / n_sims
        result.rank_probs[t] = probs.tolist()
        result.title_prob[t] = float(probs[0])
        result.top4_prob[t] = float(probs[:top4_spots].sum())
        result.relegation_prob[t] = float(probs[n_teams - relegation_spots:].sum())
        result.expected_points[t] = points_sum[t] / n_sims
        result.expected_rank[t] = rank_sum[t] / n_sims

    return result
