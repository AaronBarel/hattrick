"""Dixon-Coles Poisson model. Multiplicative parametrization:
    lambda (home rate) = attack_home * defense_away * home_advantage
    mu     (away rate) = attack_away * defense_home
Two-stage fit: IPF warm start (closed-form, no rho) -> L-BFGS-B refinement
(adds the rho low-score correction, which breaks IPF's separability).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import poisson


@dataclass
class DixonColesParams:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    rho: float
    fitted_as_of: str | None = None
    log_likelihood: float | None = None
    # Provenance: which target the rates were fit on, and what fraction of
    # matches actually had xG (the rest fell back to goals).
    target: str = "goals"
    xg_coverage: float = 0.0
    # Teams given the promoted-side shrinkage target rather than the league
    # mean. Recorded so a surprising rating can be traced to its prior.
    promoted: list = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DixonColesParams":
        with open(path) as f:
            return cls(**json.load(f))


def poisson_logpmf(x, lam):
    """Poisson log-density, generalized to continuous x >= 0.

        log L(x; lam) = x*log(lam) - lam - gammaln(x + 1)

    `scipy.stats.poisson.logpmf` computes exactly this but returns -inf for
    non-integer x, because as a *probability mass function* that is correct:
    a Poisson variate cannot equal 1.34. We need the same expression treated
    as a continuous quasi-likelihood, because the whole point of fitting on
    expected goals is that the target is not an integer.

    `gammaln(x+1) = log(x!)` for integer x and interpolates smoothly between
    integers, so this agrees with `poisson.logpmf` exactly on integer inputs
    (asserted in the tests) while remaining finite and smooth in between.

    Using a Poisson likelihood on a non-count target makes this a
    quasi-likelihood rather than a true likelihood: the score equations are
    still unbiased for the mean, so the fitted rates remain consistent, but
    the implied variance (Var = mean) is wrong for xG, which is much less
    dispersed than a goal count. Point estimates are fine; any standard
    error read off this likelihood would be too small. We do not report
    such standard errors anywhere.
    """
    x = np.asarray(x, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return x * np.log(lam) - lam - gammaln(x + 1.0)


def _tau(x, y, lam, mu, rho):
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0); m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0); m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu[m10] * rho
    tau[m11] = 1 - rho
    return tau


class DixonColesModel:
    """Dixon-Coles fitter.

    Parameters
    ----------
    xi:
        Exponential time-decay rate, in units of 1/day. A match `d` days old
        gets weight `exp(-xi * d)`.
    prior_strength:
        Pseudo-count (kappa) shrinking each team's IPF rate toward the league
        average of 1.0. See `_fit_ipf` for the derivation and for the failure
        this prevents; 0.0 recovers the unsmoothed maximum-likelihood IPF.
    rating_prior_sd:
        Standard deviation of a Gaussian prior on each team's log-rating
        about the league mean log-rating - i.e. partial pooling across
        teams. See `_fit_full`. `None` disables it and recovers the
        unpenalized maximum-likelihood fit.
    """

    def __init__(self, xi: float = 0.0018, prior_strength: float = 1.0,
                 rating_prior_sd: float | None = 0.35, target: str = "goals",
                 promoted_teams=None, promoted_prior: tuple[float, float] = (-0.25, 0.22),
                 promoted_fade_matches: float = 30.0):
        if target not in ("goals", "xg"):
            raise ValueError(f"target must be 'goals' or 'xg', got {target!r}")
        self.target = target
        self.promoted_teams = promoted_teams
        self.promoted_prior = promoted_prior
        self.promoted_fade_matches = float(promoted_fade_matches)
        self.xi = xi
        self.prior_strength = float(prior_strength)
        self.rating_prior_sd = rating_prior_sd
        self.params: DixonColesParams | None = None
        self._fit_result = None

    def _weights(self, dates, as_of):
        days = (as_of - dates).dt.days.to_numpy(dtype=float)
        return np.exp(-self.xi * np.clip(days, 0, None))

    def _fit_ipf(self, home_idx, away_idx, x, y, w, n_teams, n_iter=300, tol=1e-9):
        """Closed-form IPF warm start, with a Gamma prior on each rate.

        The plain IPF update is `alpha_k = (goals scored) / (expected goals
        scored)`, the maximum-likelihood fixed point of the no-rho model.
        That update sits on a boundary when a team's numerator is zero: a
        newly-promoted side that has not yet conceded gets `beta_k = 0`
        exactly. `_fit_full` then starts from `log(0) = -inf`, L-BFGS-B
        immediately returns ABNORMAL, and every parameter - not just that
        team's - comes back from a failed optimisation with a NaN
        log-likelihood and rho stuck at its initial value. Downstream that
        surfaces as a team predicted to concede exactly zero goals with
        probability one - a forecast that is not merely wrong but confidently,
        absurdly wrong. This is not hypothetical; it reproduces on real
        Premier League data a few matchweeks into 2025/26.

        The fix is to give each rate a Gamma(kappa, kappa) prior, whose mean
        is 1.0 - the neutral value under this model's multiplicative
        parametrisation, since ratings are identified only up to the
        attack/defense scale anchor. The posterior-mode update is then

            alpha_k = (sum_w goals + kappa) / (sum_w expected + kappa)

        which is the ordinary MLE plus one pseudo-match at exactly league
        average. With kappa = 1 a team that has played a full season barely
        moves (one pseudo-goal against ~50), while a team with two matches
        is pulled firmly toward the league mean - which is the honest
        estimate, because two matches genuinely do not identify a rating.
        """
        kappa = self.prior_strength
        alpha = np.ones(n_teams); beta = np.ones(n_teams); gamma = 1.3
        for _ in range(n_iter):
            prev = np.concatenate([alpha, beta, [gamma]])
            new_alpha = alpha.copy()
            for k in range(n_teams):
                hm = home_idx == k; am = away_idx == k
                numer = np.sum(w[hm] * x[hm]) + np.sum(w[am] * y[am]) + kappa
                denom = (gamma * np.sum(w[hm] * beta[away_idx[hm]])
                         + np.sum(w[am] * beta[home_idx[am]]) + kappa)
                if denom > 0:
                    new_alpha[k] = numer / denom
            alpha = new_alpha
            new_beta = beta.copy()
            for k in range(n_teams):
                hm = home_idx == k; am = away_idx == k
                numer = np.sum(w[hm] * y[hm]) + np.sum(w[am] * x[am]) + kappa
                denom = (gamma * np.sum(w[am] * alpha[home_idx[am]])
                         + np.sum(w[hm] * alpha[away_idx[hm]]) + kappa)
                if denom > 0:
                    new_beta[k] = numer / denom
            beta = new_beta
            expected_home = np.sum(w * alpha[home_idx] * beta[away_idx])
            if expected_home > 0:
                gamma = np.sum(w * x) / expected_home
            cur = np.concatenate([alpha, beta, [gamma]])
            if np.max(np.abs(cur - prev)) < tol:
                break
        scale = alpha.mean()
        return alpha / scale, beta * scale, gamma

    def _fit_full(self, home_idx, away_idx, x, y, w, n_teams, alpha0, beta0, gamma0,
                  rho0=-0.05, fit_rho=True, off_a=None, off_b=None):
        """Stage 2: L-BFGS-B refinement.

        `fit_rho=False` drops the low-score correction from the objective
        and freezes rho at `rho0`. That is required when the target is
        continuous (xG): tau only ever fires on the (0,0)/(0,1)/(1,0)/(1,1)
        cells, so with a target like 1.34 no mask matches, tau is
        identically 1, and rho has exactly zero gradient - the optimizer
        would return whatever it started with while looking converged.
        `fit()` estimates rho separately from actual goals instead.
        """
        n = n_teams
        # Belt and braces: the Gamma prior in _fit_ipf should already keep
        # every rate strictly positive, but a non-finite starting point makes
        # L-BFGS-B fail on ALL parameters rather than just the offending one,
        # so it is worth being unconditionally safe about it here too.
        floor = 1e-6
        alpha0 = np.clip(np.nan_to_num(alpha0, nan=1.0, posinf=1.0), floor, None)
        beta0 = np.clip(np.nan_to_num(beta0, nan=1.0, posinf=1.0), floor, None)
        gamma0 = float(np.clip(np.nan_to_num(gamma0, nan=1.3, posinf=1.3), floor, None))
        off_a = np.zeros(n) if off_a is None else np.asarray(off_a, dtype=float)
        off_b = np.zeros(n) if off_b is None else np.asarray(off_b, dtype=float)
        theta0 = np.concatenate([np.log(alpha0), np.log(beta0), [np.log(gamma0)], [rho0]])
        bounds = [(None, None)] * (2 * n + 1) + [((-0.99, 0.99) if fit_rho else (rho0, rho0))]

        def nll(theta):
            log_alpha = theta[:n]; log_beta = theta[n:2*n]
            log_gamma = theta[2*n]; rho = theta[2*n+1]
            alpha = np.exp(log_alpha); beta = np.exp(log_beta); gamma = np.exp(log_gamma)
            lam = alpha[home_idx] * beta[away_idx] * gamma
            mu = alpha[away_idx] * beta[home_idx]
            ll = poisson_logpmf(x, lam) + poisson_logpmf(y, mu)
            if fit_rho:
                tau = _tau(x, y, lam, mu, rho)
                ll = ll + np.log(np.clip(tau, 1e-10, None))
            # Hard identifiability anchor: the multiplicative parametrisation
            # is invariant to alpha -> c*alpha, beta -> beta/c, so the scale
            # has to be pinned down or the optimizer wanders along a ridge.
            penalty = 1000.0 * (log_alpha.mean() ** 2 + log_beta.mean() ** 2)

            # Partial pooling across teams. Without this, a team whose
            # weighted goal count is zero in the fitting window - a promoted
            # side a few matches into the season that has not yet conceded -
            # has its maximum-likelihood rating at the boundary, and the
            # optimizer happily drives beta_k to ~1e-3. That is not a
            # near-miss: it says the opponent scores zero goals with
            # probability ~1, which poisons every derived market for that
            # fixture.
            #
            # A Gaussian prior log(rating_k) ~ N(mean_log_rating, sd^2)
            # contributes sum((log r_k - mean)^2) / (2 sd^2) to the negative
            # log-likelihood. Shrinking toward the *mean* rather than toward
            # zero keeps this orthogonal to the anchor above: it pools teams
            # together without constraining the league's overall scoring
            # level, which still has to be fit from the data.
            #
            # sd = 0.35 is weakly informative: Premier League attack ratings
            # span roughly 0.5-1.7, so log-ratings have an empirical spread
            # near 0.3, and a prior of comparable width barely moves a team
            # with a full season of matches while firmly pooling one with
            # two.
            if self.rating_prior_sd is not None:
                c = 1.0 / (2.0 * self.rating_prior_sd ** 2)
                # Each team is shrunk toward the league mean PLUS its own
                # offset. The offset is zero for established teams, so this
                # reduces exactly to plain partial pooling when no team is
                # marked promoted.
                target_alpha = log_alpha.mean() + off_a
                target_beta = log_beta.mean() + off_b
                penalty += c * (np.sum((log_alpha - target_alpha) ** 2)
                                + np.sum((log_beta - target_beta) ** 2))
            return -np.sum(w * ll) + penalty

        result = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
        theta = result.x
        alpha = np.exp(theta[:n]); beta = np.exp(theta[n:2*n])
        gamma = np.exp(theta[2*n]); rho = theta[2*n+1]
        return alpha, beta, gamma, rho, result

    def _resolve_target(self, matches):
        """Choose the per-match fitting target, blending xG in where present.

        Returns `(x, y, coverage)`. With `target="goals"` this is just the
        goal columns. With `target="xg"` it is xG wherever xG exists and
        actual goals elsewhere.

        The fallback is deliberate (a league or season with no xG coverage
        must still fit, not crash) but it is not free: the two targets have
        different noise levels, so a partially-covered fit weights xG-backed
        and goals-backed matches as if they were equally precise. That is a
        mild misspecification, and the reason `params.xg_coverage` is
        recorded - read it before trusting an xG fit.
        """
        goals_x = matches["home_goals"].to_numpy(dtype=float)
        goals_y = matches["away_goals"].to_numpy(dtype=float)
        if self.target == "goals":
            return goals_x, goals_y, 0.0

        if not {"home_xg", "away_xg"}.issubset(matches.columns):
            warnings.warn(
                "target='xg' but the matches frame has no home_xg/away_xg columns; "
                "falling back to actual goals. Run scripts/update_xg.py to fetch xG.",
                RuntimeWarning, stacklevel=3,
            )
            return goals_x, goals_y, 0.0

        xg_x = pd.to_numeric(matches["home_xg"], errors="coerce").to_numpy(dtype=float)
        xg_y = pd.to_numeric(matches["away_xg"], errors="coerce").to_numpy(dtype=float)
        have = np.isfinite(xg_x) & np.isfinite(xg_y) & (xg_x >= 0) & (xg_y >= 0)
        coverage = float(have.mean()) if len(have) else 0.0
        if coverage == 0.0:
            warnings.warn("target='xg' but no match has usable xG; using actual goals.",
                          RuntimeWarning, stacklevel=3)
            return goals_x, goals_y, 0.0
        return np.where(have, xg_x, goals_x), np.where(have, xg_y, goals_y), coverage

    def _fit_rho_on_goals(self, home_idx, away_idx, matches, w, alpha, beta, gamma):
        """Estimate rho from actual scorelines with the rates held fixed.

        rho is a property of the *discrete* goal process - Dixon-Coles added
        it because real football produces more 0-0s and 1-1s than independent
        Poissons predict. xG cannot speak to it at all: tau only touches the
        four lowest integer cells, and a continuous target never lands on
        them. So when the rates are fit on xG, rho has to come from somewhere
        else, and the natural source is the scorelines themselves.

        With alpha/beta/gamma fixed, only the tau term depends on rho, so
        this collapses to a well-conditioned one-dimensional maximisation.
        """
        gx = matches["home_goals"].to_numpy(dtype=float)
        gy = matches["away_goals"].to_numpy(dtype=float)
        lam = alpha[home_idx] * beta[away_idx] * gamma
        mu = alpha[away_idx] * beta[home_idx]

        def neg(rho):
            tau = _tau(gx, gy, lam, mu, rho)
            return -np.sum(w * np.log(np.clip(tau, 1e-10, None)))

        res = minimize_scalar(neg, bounds=(-0.99, 0.99), method="bounded")
        return float(res.x) if res.success else 0.0

    def _prior_offsets(self, teams, matches):
        """Per-team shrinkage targets, offset from the league mean.

        Plain partial pooling shrinks a data-poor team toward the league
        average. For a newly-promoted side that is the wrong target and the
        error is large and one-directional: measured across all five leagues
        and three seasons, promoted teams average 0.91 points per game
        against 1.45 for established teams, score 1.07 goals against 1.47,
        concede 1.74 against 1.36, and finish at the 85th percentile of their
        table. Shrinking them toward average makes the model systematically
        overrate exactly the teams it knows least about - which shows up
        directly as overconfident forecasts for promoted sides early in a
        season.

        The default offsets are measured, not assumed. Fitting each
        league-season in isolation (no time decay, so the ratings are that
        season's own truth) and comparing promoted with established teams
        gives, in log space, attack -0.2465 and defense +0.2183 - that is,
        promoted sides attack at about 0.78x and concede at about 1.24x the
        league average. The sign is consistent across all five leagues
        (attack -0.15 to -0.36, defense +0.14 to +0.34).

        A single pooled offset is used rather than per-league ones because
        each league contributes only ~8 promoted teams, so per-league
        estimates are noisy; and the spread *within* promoted teams
        (sd 0.18 attack, 0.19 defense) is smaller than the shrinkage prior's
        own width of 0.35, so the residual uncertainty is already expressed.

        Note what this is NOT: an attempt to say which promoted team is best.
        A team's second-tier record turns out to carry no signal for its
        top-flight performance (correlation -0.08 over 41 promoted teams),
        because every promoted side is in the top few percent of the division
        below and there is almost no variance to exploit. So all promoted
        teams get the same offset, and the data moves them apart from there.
        """
        n = len(teams)
        off_a = np.zeros(n)
        off_b = np.zeros(n)
        if self.promoted_teams is None:
            return off_a, off_b, set()

        if self.promoted_teams == "auto":
            from hattrick.data.features import detect_promoted_teams
            promoted = detect_promoted_teams(matches)
        else:
            promoted = set(self.promoted_teams)

        # Fade the offset out as the model accumulates its own evidence.
        #
        # This is not a refinement, it is the difference between the feature
        # working and actively hurting. "Promoted" and "unknown to the model"
        # are not the same set: over a multi-season fitting window most
        # promoted sides are yo-yo clubs. For the 2025/26 holdout, 8 of 14
        # teams flagged as promoted already had 34-76 top-flight matches in
        # the window - Sassuolo 76, Lorient 72, Koln 68. Handing those teams
        # a relegation-candidate prior adds bias while reducing no variance,
        # and measurably made forecasts worse.
        #
        # Weight is by time-decayed match count, not a raw count, because a
        # team whose only top-flight record is three seasons old is
        # effectively unknown too - which is exactly what the decay already
        # says about it.
        counts = {}
        if self.promoted_fade_matches > 0:
            as_of = matches["date"].max()
            w = self._weights(matches["date"], as_of)
            for team in promoted:
                m = ((matches["home"] == team) | (matches["away"] == team)).to_numpy()
                counts[team] = float(w[m].sum())

        da, db = self.promoted_prior
        applied = set()
        for i, t in enumerate(teams):
            if t not in promoted:
                continue
            if self.promoted_fade_matches > 0:
                scarcity = max(0.0, 1.0 - counts.get(t, 0.0) / self.promoted_fade_matches)
            else:
                scarcity = 1.0
            if scarcity <= 0.0:
                continue
            off_a[i] = da * scarcity
            off_b[i] = db * scarcity
            applied.add(t)
        return off_a, off_b, applied

    def fit(self, matches: pd.DataFrame, as_of=None) -> "DixonColesModel":
        matches = matches.sort_values("date").reset_index(drop=True)
        as_of = as_of or matches["date"].max()
        w = self._weights(matches["date"], as_of)
        teams = sorted(set(matches["home"]) | set(matches["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        home_idx = matches["home"].map(idx).to_numpy()
        away_idx = matches["away"].map(idx).to_numpy()
        n_teams = len(teams)

        x, y, xg_coverage = self._resolve_target(matches)
        # rho is only identifiable from the joint fit when the target is
        # integer goals; on a continuous target it is estimated afterwards.
        continuous = xg_coverage > 0.0

        off_a, off_b, promoted = self._prior_offsets(teams, matches)

        alpha0, beta0, gamma0 = self._fit_ipf(home_idx, away_idx, x, y, w, n_teams)
        alpha, beta, gamma, rho, result = self._fit_full(
            home_idx, away_idx, x, y, w, n_teams, alpha0, beta0, gamma0,
            rho0=0.0 if continuous else -0.05, fit_rho=not continuous,
            off_a=off_a, off_b=off_b,
        )
        if continuous:
            rho = self._fit_rho_on_goals(home_idx, away_idx, matches, w, alpha, beta, gamma)

        self.params = DixonColesParams(
            teams=teams, attack=dict(zip(teams, alpha.tolist())),
            defense=dict(zip(teams, beta.tolist())), home_advantage=float(gamma),
            rho=float(rho),
            fitted_as_of=str(as_of.date()) if hasattr(as_of, "date") else str(as_of),
            log_likelihood=float(-result.fun),
            target=self.target, xg_coverage=xg_coverage,
            promoted=sorted(promoted),
        )
        self._fit_result = result

        # A failed optimisation used to be returned silently, with rho left
        # at its initial value and a NaN log-likelihood. Callers cannot see
        # that from the parameters alone, so say it out loud.
        if not result.success or not np.isfinite(result.fun):
            warnings.warn(
                f"Dixon-Coles optimizer did not converge ({result.message!r}); "
                f"log-likelihood={-result.fun}. Parameters should not be trusted. "
                f"n_teams={n_teams}, n_matches={len(matches)}.",
                RuntimeWarning, stacklevel=2,
            )
        return self

    def rates(self, home: str, away: str) -> tuple[float, float]:
        if self.params is None:
            raise RuntimeError("Model not fitted yet - call .fit() first.")
        p = self.params
        for team in (home, away):
            if team not in p.attack:
                raise KeyError(f"Unknown team '{team}' - not in training data.")
        lam = p.attack[home] * p.defense[away] * p.home_advantage
        mu = p.attack[away] * p.defense[home]
        return lam, mu
