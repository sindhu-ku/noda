#!/usr/bin/env python3
"""
ptmcmc.py — Parallel-Tempering MCMC with Differential-Evolution moves.

Designed for posteriors with well-separated modes (JUNO's dm2_31 wiggle
minima and the NO/IO degeneracy) at moderate dimension (5–100 params).

Algorithm
─────────
* K temperatures with inverse temperatures betas[0]=1 … betas[K-1]=beta_min.
  Each temperature holds its own ensemble of `nwalkers` walkers targeting
      pi_k(x) ∝ L(x)^{beta_k} · prior(x)          (prior NOT tempered)

* Within-temperature move: Differential Evolution (ter Braak 2006) with the
  emcee-style red/black split — proposals for one half of the ensemble use
  differences of two walkers from the OTHER half at the same temperature:
      x' = x + gamma · (x_a − x_b) + jitter
  gamma = 2.38/sqrt(2d) normally; with probability `p_bigjump`, gamma = 1,
  which jumps directly between modes once walkers populate them.
  DE is tuning-free: proposal shape/scale adapts to the local covariance
  automatically because it is built from the ensemble itself.

* Optional user-supplied involution `jump` (f∘f = id, |det Jf| = 1), applied
  with probability `p_jump` instead of the DE move. Used for the NO↔IO
  dm2_31 sign flip; acceptance is the plain Metropolis ratio.

* Replica exchange between adjacent temperatures after every sweep.

* Adaptive temperature ladder during burn-in (endpoints fixed, interior
  gaps adjusted to equalise pairwise swap acceptance, gain decays with
  sweep index so production sweeps are strictly Markovian).

* Evidence via thermodynamic integration over the stored per-temperature
  mean log-likelihoods:  log Z = log Z(beta=0) + ∫ ⟨log L⟩_beta dbeta.

Parallelism
───────────
All likelihood evaluations of a half-sweep are independent; pass `map_fn`
(e.g. a multiprocessing Pool map) to fan them out. `log_prior` is called
vectorised in the parent, so out-of-bounds proposals never cost a
likelihood call.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Optional


class PTSampler:
    """
    Parameters
    ----------
    log_like : callable(x: (d,) array) -> float
        Log-likelihood of a single point. This is the tempered part.
    log_prior : callable(X: (n, d) array) -> (n,) array
        VECTORISED log-prior; return -inf outside the support.
    ndim : int
    nwalkers : int, optional
        Walkers per temperature (even, >= 6). Default max(2*ndim + 2, 16).
    ntemps : int
    beta_min : float
        Hottest inverse temperature. Choose so that beta_min * dchi2_barrier
        is O(1) for the barriers you need to cross.
    betas : array, optional
        Explicit decreasing ladder (overrides ntemps / beta_min).
    jump : callable(X: (n, d)) -> (n, d), optional
        Involution mode-jump proposal (e.g. dm2_31 sign flip).
    p_jump : float
        Per-walker probability of proposing `jump` instead of a DE move.
    p_bigjump : float
        Probability that a DE move uses gamma = 1 (mode hopping).
    map_fn : callable(func, iterable) -> list, optional
        Parallel map for likelihood evaluations. Default: builtin map.
    rng : np.random.Generator, optional
    """

    def __init__(self, log_like, log_prior, ndim, *,
                 nwalkers: Optional[int] = None,
                 ntemps: int = 10,
                 beta_min: float = 0.02,
                 betas: Optional[np.ndarray] = None,
                 jump: Optional[Callable] = None,
                 p_jump: float = 0.1,
                 p_bigjump: float = 0.2,
                 map_fn: Optional[Callable] = None,
                 rng: Optional[np.random.Generator] = None):
        self.log_like  = log_like
        self.log_prior = log_prior
        self.ndim      = int(ndim)
        self.jump      = jump
        self.p_jump    = float(p_jump) if jump is not None else 0.0
        self.p_bigjump = float(p_bigjump)
        self.map_fn    = map_fn if map_fn is not None else \
                         (lambda f, xs: list(map(f, xs)))
        self.rng       = rng if rng is not None else np.random.default_rng()

        if betas is not None:
            self.betas = np.asarray(betas, dtype=float)
            if not np.all(np.diff(self.betas) < 0) or self.betas[0] != 1.0:
                raise ValueError("betas must be decreasing and start at 1.0")
        else:
            self.betas = np.geomspace(1.0, beta_min, int(ntemps))
        self.ntemps = len(self.betas)

        if nwalkers is None:
            nwalkers = max(2 * self.ndim + 2, 16)
        nwalkers = int(nwalkers)
        if nwalkers % 2:
            nwalkers += 1
        if nwalkers < 6:
            raise ValueError("nwalkers must be >= 6")
        self.nwalkers = nwalkers
        self._h1 = slice(0, nwalkers // 2)
        self._h2 = slice(nwalkers // 2, nwalkers)

        self.gamma0 = 2.38 / np.sqrt(2 * self.ndim)
        self.gscale = np.ones(self.ntemps)       # per-temp DE scale factor
        self._jitter = None                      # per-dim jitter, set at init

        # state
        self.x    = None                         # (T, W, d)
        self.logl = None                         # (T, W)
        self.logp = None                         # (T, W)

        # counters
        self._n_de   = np.zeros(self.ntemps); self._a_de   = np.zeros(self.ntemps)
        self._n_jump = np.zeros(self.ntemps); self._a_jump = np.zeros(self.ntemps)
        self._n_swap = np.zeros(self.ntemps - 1)
        self._a_swap = np.zeros(self.ntemps - 1)
        self._pair_acc_last = np.full(self.ntemps - 1, 0.5)

    # ──────────────────────────────────────────────────────────────────────
    def initialize(self, x0: np.ndarray):
        """x0: (ntemps, nwalkers, ndim) initial positions, all inside the
        prior support."""
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (self.ntemps, self.nwalkers, self.ndim):
            raise ValueError(f"x0 must have shape "
                             f"{(self.ntemps, self.nwalkers, self.ndim)}")
        flat = x0.reshape(-1, self.ndim)
        lp = np.asarray(self.log_prior(flat), dtype=float)
        if not np.all(np.isfinite(lp)):
            raise ValueError(f"{np.sum(~np.isfinite(lp))} initial walkers "
                             "are outside the prior support")
        ll = np.asarray(self.map_fn(self.log_like, list(flat)), dtype=float)
        if not np.all(np.isfinite(ll)):
            raise ValueError(f"{np.sum(~np.isfinite(ll))} initial walkers "
                             "have non-finite log-likelihood")
        self.x    = x0.copy()
        self.logl = ll.reshape(self.ntemps, self.nwalkers)
        self.logp = lp.reshape(self.ntemps, self.nwalkers)
        # jitter: tiny relative to the initial per-dim spread
        spread = flat.std(axis=0)
        spread[spread == 0] = 1e-12
        self._jitter = 1e-3 * spread

    # ──────────────────────────────────────────────────────────────────────
    def _half_update(self, first_half: bool):
        sl, csl = (self._h1, self._h2) if first_half else (self._h2, self._h1)
        cur  = self.x[:, sl]                       # views
        lc   = self.logl[:, sl]
        pc   = self.logp[:, sl]
        comp = self.x[:, csl]
        T, nh, d = cur.shape
        nc  = comp.shape[1]
        rng = self.rng

        # DE partners a != b from the complementary half, same temperature
        a = rng.integers(0, nc, size=(T, nh))
        b = rng.integers(0, nc - 1, size=(T, nh))
        b[b >= a] += 1
        xa = np.take_along_axis(comp, a[..., None], axis=1)
        xb = np.take_along_axis(comp, b[..., None], axis=1)

        big   = rng.random((T, nh)) < self.p_bigjump
        gamma = np.where(big, 1.0,
                         self.gamma0 * self.gscale[:, None]
                         * (1.0 + 0.1 * rng.standard_normal((T, nh))))
        prop = (cur + gamma[..., None] * (xa - xb)
                + self._jitter * rng.standard_normal(cur.shape))

        jump_mask = np.zeros((T, nh), dtype=bool)
        if self.jump is not None and self.p_jump > 0:
            jump_mask = rng.random((T, nh)) < self.p_jump
            if jump_mask.any():
                prop[jump_mask] = self.jump(cur[jump_mask])

        lp = np.asarray(self.log_prior(prop.reshape(-1, d)),
                        dtype=float).reshape(T, nh)
        ll = np.full((T, nh), -np.inf)
        ok = np.isfinite(lp)
        if ok.any():
            vals = self.map_fn(self.log_like, list(prop[ok]))
            ll[ok] = np.asarray(vals, dtype=float)

        log_alpha = self.betas[:, None] * (ll - lc) + (lp - pc)
        acc = np.log(rng.random((T, nh))) < log_alpha

        cur[acc] = prop[acc]
        lc[acc]  = ll[acc]
        pc[acc]  = lp[acc]

        de_mask = ~jump_mask
        self._n_de   += de_mask.sum(axis=1)
        self._a_de   += (acc & de_mask).sum(axis=1)
        self._n_jump += jump_mask.sum(axis=1)
        self._a_jump += (acc & jump_mask).sum(axis=1)

    # ──────────────────────────────────────────────────────────────────────
    def _temperature_swaps(self):
        """Replica exchange between adjacent temperature pairs, hot → cold,
        with random walker pairing (as in ptemcee)."""
        rng = self.rng
        W = self.nwalkers
        for k in range(self.ntemps - 1, 0, -1):
            perm  = rng.permutation(W)
            dbeta = self.betas[k - 1] - self.betas[k]
            log_alpha = dbeta * (self.logl[k, perm] - self.logl[k - 1])
            # expected acceptance (low-noise signal for ladder adaptation)
            self._pair_acc_last[k - 1] = float(
                np.exp(np.minimum(0.0, log_alpha)).mean())
            acc = np.log(rng.random(W)) < log_alpha
            self._n_swap[k - 1] += W
            self._a_swap[k - 1] += acc.sum()
            if not acc.any():
                continue
            i_hot = perm[acc]
            for arr in (self.x, self.logl, self.logp):
                tmp = arr[k - 1, acc].copy()
                arr[k - 1, acc] = arr[k, i_hot]
                arr[k, i_hot] = tmp

    # ──────────────────────────────────────────────────────────────────────
    def _adapt(self, sweep_idx: int, adapt_ladder: bool, adapt_scales: bool,
               t0: float = 100.0):
        gain = t0 / (sweep_idx + t0)
        if adapt_scales:
            # nudge per-temp DE scale toward ~24% acceptance
            with np.errstate(invalid='ignore'):
                ar = np.where(self._n_de > 0, self._a_de / np.maximum(self._n_de, 1), 0.24)
            self.gscale *= np.exp(0.5 * gain * (ar - 0.24))
            np.clip(self.gscale, 0.1, 10.0, out=self.gscale)
        if adapt_ladder and self.ntemps > 2:
            T = 1.0 / self.betas
            S = np.log(np.diff(T))
            S += gain * (self._pair_acc_last - self._pair_acc_last.mean())
            gaps  = np.exp(S)
            T_new = T[0] + np.concatenate(([0.0], np.cumsum(gaps)))
            # keep both endpoints fixed
            T_new = T[0] + (T_new - T[0]) * (T[-1] - T[0]) / (T_new[-1] - T_new[0])
            self.betas = 1.0 / T_new

    def _reset_counters(self):
        self._n_de[:] = 0;   self._a_de[:] = 0
        self._n_jump[:] = 0; self._a_jump[:] = 0
        self._n_swap[:] = 0; self._a_swap[:] = 0

    # ──────────────────────────────────────────────────────────────────────
    def run(self, nsweeps: int, *, burnin: int = 0, thin: int = 1,
            adapt_ladder: bool = True, adapt_scales: bool = True,
            record_all_temps: bool = False,
            progress_every: int = 0) -> dict:
        """
        Burn-in (with adaptation) then production (no adaptation).

        Returns dict:
          chain      (nkept, nwalkers, ndim)   cold-chain positions
          logl       (nkept, nwalkers)         cold-chain log L
          logl_all   (nkept, ntemps, nwalkers) log L at every temperature
          chain_all  (nkept, ntemps, nwalkers, ndim)  [record_all_temps only]
          betas      final ladder
          acc_de, acc_jump   per-temperature acceptance fractions
          acc_swap           per-pair swap acceptance fractions
        """
        if self.x is None:
            raise RuntimeError("call initialize() first")
        assert thin >= 1

        for it in range(1, burnin + 1):
            self._half_update(True)
            self._half_update(False)
            self._temperature_swaps()
            self._adapt(it, adapt_ladder, adapt_scales)
            if progress_every and it % progress_every == 0:
                self._progress(f"burn-in {it}/{burnin}")
        self._reset_counters()

        nkept = nsweeps // thin
        chain    = np.empty((nkept, self.nwalkers, self.ndim))
        logl     = np.empty((nkept, self.nwalkers))
        logl_all = np.empty((nkept, self.ntemps, self.nwalkers))
        chain_all = (np.empty((nkept, self.ntemps, self.nwalkers, self.ndim))
                     if record_all_temps else None)

        j = 0
        for it in range(1, nsweeps + 1):
            self._half_update(True)
            self._half_update(False)
            self._temperature_swaps()
            if it % thin == 0 and j < nkept:
                chain[j]    = self.x[0]
                logl[j]     = self.logl[0]
                logl_all[j] = self.logl
                if chain_all is not None:
                    chain_all[j] = self.x
                j += 1
            if progress_every and it % progress_every == 0:
                self._progress(f"sample {it}/{nsweeps}")

        out = {
            'chain':    chain[:j],
            'logl':     logl[:j],
            'logl_all': logl_all[:j],
            'betas':    self.betas.copy(),
            'acc_de':   self._a_de / np.maximum(self._n_de, 1),
            'acc_jump': self._a_jump / np.maximum(self._n_jump, 1),
            'acc_swap': self._a_swap / np.maximum(self._n_swap, 1),
            'gscale':   self.gscale.copy(),
        }
        if chain_all is not None:
            out['chain_all'] = chain_all[:j]
        return out

    def _progress(self, tag):
        ar = self._a_de.sum() / max(self._n_de.sum(), 1)
        sw = self._a_swap.sum() / max(self._n_swap.sum(), 1)
        msg = (f"[pt {tag}]  acc_de={ar:.3f}  acc_swap={sw:.3f}  "
               f"maxlogL={self.logl[0].max():.2f}")
        if self.jump is not None:
            aj = self._a_jump.sum() / max(self._n_jump.sum(), 1)
            msg += f"  acc_jump={aj:.3f}"
        print(msg, flush=True)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def log_evidence(logl_all: np.ndarray, betas: np.ndarray):
        """
        Thermodynamic integration:
            log Z = ∫_0^1 ⟨log L⟩_beta dbeta      (normalised prior)
        The beta=0 endpoint is approximated by the hottest temperature.

        Returns (log_Z, mean_logL_per_temp).
        """
        mean_ll = logl_all.mean(axis=(0, 2))          # (ntemps,)
        order = np.argsort(betas)
        b = np.concatenate(([0.0], betas[order]))
        m = np.concatenate(([mean_ll[order][0]], mean_ll[order]))
        trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # numpy 2 rename
        return float(trapezoid(m, b)), mean_ll
