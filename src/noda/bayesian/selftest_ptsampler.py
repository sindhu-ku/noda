#!/usr/bin/env python3
"""
selftest_ptsampler.py — validate PTSampler on a synthetic multimodal target.

Target (d = 5): x0 is a two-Gaussian mixture mimicking the NO/IO split
    0.7 · N(-3, 0.4²)  +  0.3 · N(+3, 0.4²)
and x1..x4 are standard normals (nuisances). Uniform prior on [-10, 10]^5.

Checks
  * recovered mode weight of the x0 > 0 mode  ≈ 0.30
  * mode means/widths
  * TI evidence vs the analytic value  log Z = -5 log 20
    (likelihood integrates to 1 and lies well inside the prior box)

Run:  python -m noda.bayesian.selftest_ptsampler
"""
import numpy as np
from .ptmcmc import PTSampler

D      = 5
W1, W2 = 0.7, 0.3
MU, SIG = 3.0, 0.4
LO, HI = -10.0, 10.0


def log_like(x):
    g1 = W1 * np.exp(-0.5 * ((x[0] + MU) / SIG) ** 2)
    g2 = W2 * np.exp(-0.5 * ((x[0] - MU) / SIG) ** 2)
    ll = np.log((g1 + g2) / (SIG * np.sqrt(2 * np.pi)))
    ll += -0.5 * np.sum(x[1:] ** 2) - (D - 1) * 0.5 * np.log(2 * np.pi)
    return ll


LOGPRIOR_CONST = -D * np.log(HI - LO)


def log_prior(X):
    ok = np.all((X >= LO) & (X <= HI), axis=1)
    return np.where(ok, LOGPRIOR_CONST, -np.inf)


def jump(X):
    Y = X.copy()
    Y[:, 0] = -Y[:, 0]          # involution between the two modes
    return Y


def main():
    rng = np.random.default_rng(42)
    # beta_min small enough to resolve <logL>(beta) near beta=0 — needed for
    # the TI evidence check; parameter estimation alone is fine with a much
    # shallower ladder (e.g. ntemps=8, beta_min=0.05).
    ntemps, nwalkers = 12, 32
    s = PTSampler(log_like, log_prior, D, nwalkers=nwalkers, ntemps=ntemps,
                  beta_min=0.005, jump=jump, p_jump=0.1, rng=rng)

    # start ALL walkers in the WRONG (minor) mode on purpose
    x0 = rng.normal(0, 0.2, size=(ntemps, nwalkers, D))
    x0[..., 0] += MU
    s.initialize(x0)

    res = s.run(4000, burnin=1000, thin=1, progress_every=1000)

    flat = res['chain'].reshape(-1, D)
    w_plus = float(np.mean(flat[:, 0] > 0))
    logZ, mean_ll = PTSampler.log_evidence(res['logl_all'], res['betas'])
    logZ_true = -D * np.log(HI - LO)

    mode_m = flat[flat[:, 0] < 0, 0]
    mode_p = flat[flat[:, 0] > 0, 0]
    print("\n── results ────────────────────────────────────────────")
    print(f"weight(x0>0)      = {w_plus:.3f}   (true {W2:.3f})")
    print(f"mode-: mean {mode_m.mean():+.3f} std {mode_m.std():.3f}  "
          f"(true {-MU:+.1f}, {SIG})")
    print(f"mode+: mean {mode_p.mean():+.3f} std {mode_p.std():.3f}  "
          f"(true {+MU:+.1f}, {SIG})")
    print(f"nuisance x1: mean {flat[:,1].mean():+.3f} std {flat[:,1].std():.3f}"
          f"  (true +0.0, 1.0)")
    print(f"log Z (TI)        = {logZ:.3f}   (true {logZ_true:.3f})")
    print(f"acc_de  per temp  = {np.array2string(res['acc_de'], precision=2)}")
    print(f"acc_jump per temp = {np.array2string(res['acc_jump'], precision=2)}")
    print(f"acc_swap per pair = {np.array2string(res['acc_swap'], precision=2)}")
    print(f"final betas       = {np.array2string(res['betas'], precision=3)}")

    ok = True
    if abs(w_plus - W2) > 0.05:
        ok = False; print("FAIL: mode weight off by > 0.05")
    if abs(mode_m.mean() + MU) > 0.1 or abs(mode_p.mean() - MU) > 0.1:
        ok = False; print("FAIL: mode means off")
    if abs(flat[:, 1].std() - 1.0) > 0.1:
        ok = False; print("FAIL: nuisance width off")
    if abs(logZ - logZ_true) > 0.5:
        ok = False; print("FAIL: evidence off by > 0.5")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
