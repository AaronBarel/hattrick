"""Scoreline matrix -> derived match probabilities (W/D/L, totals, BTTS, correct score)."""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson
from hattrick.model.dixon_coles import DixonColesModel, _tau


def score_matrix(model: DixonColesModel, home: str, away: str, max_goals: int = 10) -> np.ndarray:
    lam, mu = model.rates(home, away)
    rho = model.params.rho
    goals = np.arange(max_goals + 1)
    M = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    x = np.array([0, 0, 1, 1]); y = np.array([0, 1, 0, 1])
    tau_vals = _tau(x, y, np.full(4, lam), np.full(4, mu), rho)
    M[0, 0] *= tau_vals[0]; M[0, 1] *= tau_vals[1]
    M[1, 0] *= tau_vals[2]; M[1, 1] *= tau_vals[3]
    M = np.clip(M, 0, None)
    M /= M.sum()
    return M


def outcome_probs(M: np.ndarray) -> dict[str, float]:
    return {"home_win": float(np.tril(M, -1).sum()), "draw": float(np.trace(M)),
            "away_win": float(np.triu(M, 1).sum())}


def over_under(M: np.ndarray, line: float = 2.5) -> dict[str, float]:
    n = M.shape[0]
    totals = np.add.outer(np.arange(n), np.arange(n))
    over = float(M[totals > line].sum())
    return {"over": over, "under": 1.0 - over}


def btts(M: np.ndarray) -> dict[str, float]:
    yes = float(M[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def correct_score_probs(M: np.ndarray, top_n: int = 6):
    n = M.shape[0]
    flat = [((i, j), float(M[i, j])) for i in range(n) for j in range(n)]
    flat.sort(key=lambda t: -t[1])
    return flat[:top_n]


def match_probabilities(model: DixonColesModel, home: str, away: str, max_goals: int = 10) -> dict:
    M = score_matrix(model, home, away, max_goals)
    lam, mu = model.rates(home, away)
    return {"home": home, "away": away, "lambda_home": lam, "mu_away": mu,
            "result": outcome_probs(M), "over_under_2_5": over_under(M, 2.5),
            "btts": btts(M), "top_scorelines": correct_score_probs(M), "matrix": M}
