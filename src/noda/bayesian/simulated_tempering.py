#!/usr/bin/env python3
"""
simulated_tempering.py

Simulated Tempering (ST) MCMC sampler.

State
─────
    (x, k)  where x ∈ ℝ^d  and  k ∈ {0, …, K-1}.

Tempered posterior at temperature index k:
    π_{β_k}(x) ∝ L(x)^{β_k} · prior(x)

Joint target on the augmented space:
    p(x, k) ∝ π_{β_k}(x) · exp(g_k)

The g_k ("pseudo-priors") are learned during tuning so the marginal over
k is roughly uniform; otherwise the chain gets stuck at hot temperatures.

Moves
─────
  1. Within-temperature random-walk Metropolis on x using
        x' = x + σ_k · L_k · z,    z ~ N(0, I)
     where L_k L_kᵀ = Σ_k is a Cholesky factor learned per temperature.
  2. Temperature swap k → k' = k ± 1.

Evidence (β_0 = 1 is the cold/target temperature, β_{K-1} small)
──────────────────────────────────────────────────────────────
    log Z = log Z(β=0) + ∫_0^1 ⟨log L⟩_β dβ
With a properly normalised prior, log Z(β=0) = 0, so
    log Z ≈ ∑_k trapezoid weight × ⟨log L⟩_{β_k}.

Conventions
───────────
β ladder is stored in DECREASING order:  betas[0] = 1, betas[K-1] = β_min.
The "cold" chain (target posterior) lives at index k = 0.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Helper: numerically stable log-sum-exp
# ─────────────────────────────────────────────────────────────────────────
def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


# ─────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────
class SimulatedTempering:
    """
    Parameters
    ----------
    log_likelihood : callable
        x ↦ log L(x).   Must be finite for any x with finite log_prior(x).
    log_prior : callable
        x ↦ log π(x).   Return -inf outside the support (hard bounds OK).
    n_dim : int
        Dimension of x.
    x_init : (d,) array
        Starting point. Must satisfy log_prior(x_init) > -inf.
    prop_cov : (d, d) array, optional
        Initial proposal covariance shared by all temperatures.
        Recommended: pass the inverse Hessian / Minuit covariance scaled by
        2.38² / d (Roberts–Rosenthal optimal scaling for Gaussian targets).
    n_temps : int
        Number of temperatures K (≥ 4 recommended; 8–16 typical).
    beta_min : float
        Smallest inverse temperature; β_0 = 1.   Hotter → more like the prior.
    rng : np.random.Generator
        For reproducibility.
    """

    def __init__(
        self,
        log_likelihood: Callable[[np.ndarray], float],
        log_prior:      Callable[[np.ndarray], float],
        n_dim:          int,
        x_init:         np.ndarray,
        prop_cov:       Optional[np.ndarray] = None,
        n_temps:        int = 12,
        beta_min:       float = 0.01,
        rng:            Optional[np.random.Generator] = None,
    ):
        self.log_L     = log_likelihood
        self.log_pi    = log_prior
        self.d         = int(n_dim)
        self.K         = int(n_temps)
        self.rng       = rng if rng is not None else np.random.default_rng()
        self.betas     = np.geomspace(1.0, beta_min, self.K)  # decreasing
        self.g         = np.zeros(self.K)                     # log pseudo-priors

        if prop_cov is None:
            prop_cov = np.eye(self.d) * 1e-4
        prop_cov = 0.5 * (prop_cov + prop_cov.T)               # symmetrise
        # Guard against non-PD input
        try:
            self._chol_base = np.linalg.cholesky(prop_cov)
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(prop_cov)
            evals = np.clip(evals, 1e-12, None)
            self._chol_base = evecs * np.sqrt(evals)

        # Per-temperature scale on top of the shared Cholesky factor.
        # Hotter chains take bigger steps — initialise on a small geometric ramp.
        self.scales = np.geomspace(1.0, 3.0, self.K)

        # Current state
        x0 = np.asarray(x_init, dtype=float).copy()
        lp0 = self.log_pi(x0)
        if not np.isfinite(lp0):
            raise ValueError("x_init is outside the prior support (log_prior = -inf).")
        ll0 = self.log_L(x0)
        if not np.isfinite(ll0):
            raise ValueError("log_likelihood(x_init) is not finite.")
        self.x          = x0
        self.k          = 0
        self.log_L_x    = ll0
        self.log_pi_x   = lp0

        # Stats
        self._n_within        = np.zeros(self.K, dtype=int)
        self._n_within_accept = np.zeros(self.K, dtype=int)
        self._n_swap          = 0
        self._n_swap_accept   = 0
        self._visits          = np.zeros(self.K, dtype=int)

    # ──────────────────────────────────────────────────────────────────────
    # Internal moves
    # ──────────────────────────────────────────────────────────────────────
    def _log_target(self, log_L_x: float, log_pi_x: float, k: int) -> float:
        """log of the tempered, pseudo-prior-augmented target at (x, k)."""
        return self.betas[k] * log_L_x + log_pi_x + self.g[k]

    def _within_temperature_step(self) -> bool:
        """Random-walk Metropolis on x at current k.  Returns acceptance bool."""
        z       = self.rng.standard_normal(self.d)
        x_prop  = self.x + self.scales[self.k] * (self._chol_base @ z)
        lp_prop = self.log_pi(x_prop)
        self._n_within[self.k] += 1
        if not np.isfinite(lp_prop):
            return False
        ll_prop = self.log_L(x_prop)
        if not np.isfinite(ll_prop):
            return False
        log_alpha = (self._log_target(ll_prop, lp_prop, self.k)
                     - self._log_target(self.log_L_x, self.log_pi_x, self.k))
        if np.log(self.rng.random()) < log_alpha:
            self.x        = x_prop
            self.log_L_x  = ll_prop
            self.log_pi_x = lp_prop
            self._n_within_accept[self.k] += 1
            return True
        return False

    def _swap_temperature_step(self) -> bool:
        """Propose k → k ± 1 with symmetric proposal (boundary correction)."""
        k = self.k
        if k == 0:
            k_new = 1
            log_q = np.log(0.5) - np.log(1.0)            # log q(k'→k) − log q(k→k')
        elif k == self.K - 1:
            k_new = self.K - 2
            log_q = np.log(0.5) - np.log(1.0)
        else:
            k_new = k + (1 if self.rng.random() < 0.5 else -1)
            if k_new == 0 or k_new == self.K - 1:
                log_q = np.log(1.0) - np.log(0.5)
            else:
                log_q = 0.0
        self._n_swap += 1
        log_alpha = ((self.betas[k_new] - self.betas[k]) * self.log_L_x
                     + (self.g[k_new] - self.g[k])
                     + log_q)
        if np.log(self.rng.random()) < log_alpha:
            self.k = k_new
            self._n_swap_accept += 1
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Tuning (burn-in) — adapt proposal scales and pseudo-priors
    # ──────────────────────────────────────────────────────────────────────
    def tune(
        self,
        n_iter: int,
        target_accept: float = 0.234,
        n_swap_per_step: int = 1,
        learn_g: bool = True,
        adapt_scales: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Burn-in with adaptive proposal scales (Robbins–Monro on log scale)
        and Wang-Landau-style updates of the pseudo-priors g_k.

        After tuning, statistics are reset before production sampling.
        """
        # Wang-Landau-style learning rate that decays over the tuning window.
        wl_initial = 1.0
        for it in range(1, n_iter + 1):
            self._within_temperature_step()
            for _ in range(n_swap_per_step):
                self._swap_temperature_step()
            self._visits[self.k] += 1

            if adapt_scales:
                # Diminishing-adaptation: γ_t = it^{-0.6}
                gamma = it ** -0.6
                accept_rate = (self._n_within_accept[self.k]
                               / max(1, self._n_within[self.k]))
                log_scale = np.log(self.scales[self.k])
                log_scale += gamma * (accept_rate - target_accept)
                # Clamp to a sane range to prevent runaway
                self.scales[self.k] = float(np.exp(np.clip(log_scale, -8, 6)))

            if learn_g:
                # Wang-Landau: decrement g_k for the visited temperature.
                # This DISCOURAGES staying there if it's over-visited and
                # encourages the chain to move on, equalising visits.
                wl = wl_initial / np.sqrt(it)
                self.g[self.k] -= wl
                # Periodically re-anchor so g_0 = 0
                if it % 500 == 0:
                    self.g -= self.g[0]

            if verbose and it % max(1, n_iter // 10) == 0:
                vis = self._visits / max(1, self._visits.sum())
                ar  = (self._n_within_accept /
                       np.maximum(1, self._n_within))
                sr  = self._n_swap_accept / max(1, self._n_swap)
                print(f"[tune {it:>7d}/{n_iter}]  swap_acc={sr:.3f}  "
                      f"within_acc={ar.mean():.3f}  "
                      f"visit_min/max={vis.min():.3f}/{vis.max():.3f}")

        # Reset counters for production
        self._n_within[:]        = 0
        self._n_within_accept[:] = 0
        self._n_swap             = 0
        self._n_swap_accept      = 0
        self._visits[:]          = 0

    # ──────────────────────────────────────────────────────────────────────
    # Production sampling
    # ──────────────────────────────────────────────────────────────────────
    def sample(
        self,
        n_iter:           int,
        thin:             int = 1,
        n_swap_per_step:  int = 1,
        store_all_temps:  bool = True,
        verbose:          bool = False,
    ) -> dict:
        """
        Run the production chain.   No adaptation here (preserves
        detailed balance).

        Returns
        -------
        dict with keys
            'samples_cold' : (N, d)  samples at β = 1 only
            'samples_all'  : (M, d)  every stored sample, any temperature
            'k_all'        : (M,)    temperature index for samples_all
            'logL_all'     : (M,)    log-likelihood for samples_all
            'betas'        : (K,)    inverse-temperature ladder
            'visits'       : (K,)    number of steps spent at each k
            'g'            : (K,)    pseudo-priors at end of tuning
            'diagnostics'  : dict
        """
        assert thin >= 1
        n_store = n_iter // thin

        samples_all = np.empty((n_store, self.d), dtype=float) if store_all_temps else None
        k_all       = np.empty(n_store, dtype=np.int32) if store_all_temps else None
        logL_all    = np.empty(n_store, dtype=float)    if store_all_temps else None

        cold_buf = []

        store_idx = 0
        for it in range(1, n_iter + 1):
            self._within_temperature_step()
            for _ in range(n_swap_per_step):
                self._swap_temperature_step()
            self._visits[self.k] += 1

            if it % thin == 0:
                if store_all_temps:
                    samples_all[store_idx] = self.x
                    k_all[store_idx]       = self.k
                    logL_all[store_idx]    = self.log_L_x
                if self.k == 0:
                    cold_buf.append(self.x.copy())
                store_idx += 1

            if verbose and it % max(1, n_iter // 10) == 0:
                ar = (self._n_within_accept /
                      np.maximum(1, self._n_within))
                sr = self._n_swap_accept / max(1, self._n_swap)
                print(f"[sample {it:>7d}/{n_iter}]  swap_acc={sr:.3f}  "
                      f"within_acc={ar.mean():.3f}  "
                      f"cold_frac={self._visits[0]/self._visits.sum():.3f}")

        samples_cold = np.asarray(cold_buf) if cold_buf else np.empty((0, self.d))

        diag = {
            "within_accept_per_temp": (self._n_within_accept /
                                       np.maximum(1, self._n_within)).copy(),
            "swap_accept":            self._n_swap_accept / max(1, self._n_swap),
            "visits":                 self._visits.copy(),
            "scales":                 self.scales.copy(),
        }

        return {
            "samples_cold": samples_cold,
            "samples_all":  samples_all if samples_all is not None else samples_cold,
            "k_all":        k_all       if k_all       is not None else np.zeros(len(samples_cold), dtype=np.int32),
            "logL_all":     logL_all    if logL_all    is not None else np.array([self.log_L(x) for x in samples_cold]),
            "betas":        self.betas.copy(),
            "visits":       self._visits.copy(),
            "g":            self.g.copy(),
            "diagnostics":  diag,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Evidence via thermodynamic integration
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def log_evidence(samples_all: np.ndarray,
                     k_all:       np.ndarray,
                     logL_all:    np.ndarray,
                     betas:       np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        log Z ≈ ∫_0^1 ⟨log L⟩_β dβ  (prior is normalised, so log Z(β=0) = 0).

        Trapezoidal rule on the temperature ladder, with ⟨log L⟩_β estimated
        as the mean log L of samples whose temperature index equals that β.

        Returns
        -------
        log_Z : float
        log_Z_err : float
            Crude uncertainty: jackknife over temperature contributions.
        mean_logL_per_temp : (K,)
        """
        K = len(betas)
        # betas is decreasing; for integration we want ascending β from 0 to 1.
        order   = np.argsort(betas)
        b_asc   = betas[order]
        mean_logL = np.full(K, np.nan)
        for j, k_idx in enumerate(order):
            mask = (k_all == k_idx)
            if mask.any():
                mean_logL[j] = logL_all[mask].mean()
            else:
                mean_logL[j] = np.nan

        # Prepend β = 0, ⟨log L⟩_0 = E_prior[log L].  We don't have prior-only
        # samples; the smallest β in the ladder is the best proxy.  This is the
        # standard approximation; making β_min small enough mitigates it.
        b_full = np.concatenate(([0.0], b_asc))
        m_full = np.concatenate(([mean_logL[0]], mean_logL))

        # Trapezoid weights
        trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy 2 rename
        log_Z = float(trapezoid(m_full, b_full))

        # Per-interval contributions for a rough error bar
        contribs = 0.5 * (m_full[1:] + m_full[:-1]) * np.diff(b_full)
        # Jackknife-style: drop each interval, recompute
        n_int = len(contribs)
        jack = np.array([log_Z - c for c in contribs])
        log_Z_err = float(np.sqrt((n_int - 1) / n_int * np.sum((jack - jack.mean())**2)))

        return log_Z, log_Z_err, mean_logL