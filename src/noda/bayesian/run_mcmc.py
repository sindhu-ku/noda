#!/usr/bin/env python3
"""
run_mcmc.py — Bayesian parallel-tempering MCMC for NODA.

Posterior = exp(-chi2/2) with chi2 the EXACT closure Minuit minimises
(run_minuit_bundled with return_cost_fn=True), so every pull term acts as
its Gaussian prior and nothing can drift between the frequentist fit and
the Bayesian sampling. Flat priors within the Minuit limits on top.

Multimodality in dm2_31 (wiggle minima within an ordering, and NO vs IO)
is handled by the PT ladder + DE big-jump moves + an explicit NO<->IO
sign-flip involution move when --ordering both.

Orderings share one chain: the sampled dm2_31 (Minuit units, 1e-3 eV2) has
prior support [-3.5,-1.5] U [1.5,3.5]; the likelihood dispatches on its
sign to the per-ordering cost closure (each built with its own NMO_opt,
pull centres, and fixed-nuisance values, with nuosc.op_nom swapped per
call). The NO/IO posterior odds then drop out of the cold-chain sign
fraction.

Typical usage
─────────────
  # fast: sample the 5 shared osc params, nuisances fixed at the per-
  # ordering Minuit best fit (marginalisation only via CM-type chi2)
  run_noda_mcmc --config config/my.yaml --ordering both --params osc \\
                --output mcmc_osc.npz

  # full: sample every free Minuit parameter (expensive; use many cores)
  run_noda_mcmc --config config/my.yaml --ordering both --params all \\
                --nsweeps 4000 --ncores 16 --output mcmc_all.npz

Analyse with noda_mcmc_results.
"""
import os
# nl_geant4 uses numba @njit(parallel=True); numba's default OpenMP threading
# layer (GNU libgomp) aborts fork()ed pool workers, so force the fork-safe
# workqueue layer. Must be set BEFORE numba is first imported (it gets pulled
# in via the noda imports below). Single-threaded OpenMP for the same reason,
# and to avoid oversubscription while the pool samples.
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('OMP_NUM_THREADS', '1')
# one numba thread per pool worker — benchmarked ~equal per-eval speed and
# avoids ncores x ncores oversubscription while the pool samples
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

# Every chi2 eval allocates and frees ~10-30 MB of large numpy temporaries
# (fine x fine response-matrix combinations). glibc's dynamic mmap threshold
# migrates those onto the brk heap, where small long-lived cache entries pin
# the freed blocks: each pool worker's RSS then ratchets by MBs per eval and
# the condor cgroup counts every freed-but-held page (observed: 102 GB on a
# 5-core CNAF job in ~5 min of sampling). Pinning M_MMAP_THRESHOLD makes all
# >128 KB allocations mmap-backed so free() returns them to the OS; the
# mmap/page-fault cost is negligible against a >=60 ms eval. Set before the
# pool fork so every worker inherits it.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _libc.mallopt(-3, 131072)    # M_MMAP_THRESHOLD (also disables dynamic adj)
    _libc.mallopt(-1, 131072)    # M_TRIM_THRESHOLD
except Exception:
    pass

import argparse
import multiprocessing as mp
import time
from copy import deepcopy
from datetime import datetime

import numpy as np

from .. import nuosc
from ..fc.fc_toy import build_setups
from ..fit import combined_fit as minuit_new
from ..fit.minuit import load_prefit_cache
from ..fit.shared_nuisances import SHARED_PARAM_NAMES
from .ptmcmc import PTSampler


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side likelihood (inherited by fork; only x gets pickled per call)
# ─────────────────────────────────────────────────────────────────────────────
_LOGLIKE = None


def _eval_loglike(x):
    return _LOGLIKE(x)


class NodaPosterior:
    """log L(x) = -chi2(x)/2, dispatching on sign(dm2_31) between the
    per-ordering cost closures."""

    def __init__(self, j_dm31):
        self.j_dm31 = j_dm31
        self.entries = {}          # sign -> dict(cost, base, idx, op)

    def add_ordering(self, sign, cost_fn, base_values, idx, op_snapshot):
        self.entries[int(sign)] = {
            'cost': cost_fn,
            'base': np.asarray(base_values, dtype=float),
            'idx':  np.asarray(idx, dtype=int),
            'op':   dict(op_snapshot),
        }

    def __call__(self, x):
        sign = 1 if x[self.j_dm31] > 0 else -1
        e = self.entries.get(sign)
        if e is None:
            return -np.inf
        # pull centres etc. read module-global nuosc.op_nom at call time
        nuosc.op_nom.update(e['op'])
        v = e['base'].copy()
        v[e['idx']] = x
        try:
            c = float(e['cost'](*v))
        except Exception:
            return -np.inf
        return -0.5 * c if np.isfinite(c) else -np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Bundle assembly (mirrors main._run_multi_bundle, with NMO forced)
# ─────────────────────────────────────────────────────────────────────────────
def build_all_bundles(setups, force_NO):
    from ..main import _build_juno_bundle, _build_tao_bundle, _build_dyb_bundle
    bundles, args_fit, first2 = [], None, None
    for s in setups:
        s2 = dict(s)
        for key in ('args_juno', 'args_dyb', 'args_tao'):
            if s.get(key) is not None:
                s2[key] = deepcopy(s[key])
                s2[key].NMO_opt = bool(force_NO)
        bundles.append(_build_juno_bundle(s2))
        if args_fit is None:
            args_fit, first2 = s2['args_juno'], s2
    if (args_fit.fit_type == 'PMOP_combined'
            and getattr(args_fit, 'include_tao', False)
            and first2['ensp_nom_tao'] is not None):
        bundles.append(_build_tao_bundle(first2))
    if (args_fit.fit_type == 'PMOP_combined'
            and getattr(args_fit, 'include_dyb', False)):
        if args_fit.use_dagflow:
            from ..DYB_model import make_dyb_model
            bundles.append(_build_dyb_bundle(
                first2, dyb_model=make_dyb_model(first2['args_dyb'])))
        elif first2['cm_dyb'] is not None:
            bundles.append(_build_dyb_bundle(first2))
    return bundles, args_fit


def seed_fit(setups, policy, prefit, force_NO, scan_n=0):
    """Migrad seed for one ordering: best-fit values (fixed-nuisance base
    vector + walker init), limits, and the chi2 closure itself.

    scan_n > 0 adds a conditional dm2_31 profile scan: after the free fit,
    dm2_31 is fixed and walked over an scan_n-point grid across its Minuit
    window (each migrad warm-starts from the previous point), and the free
    fit is redone from the deepest grid point. Protects fits without a
    DYB/LBL dm2_31 pull from seeding the MCMC in a secondary wiggle
    minimum."""
    bundles, args_fit = build_all_bundles(setups, force_NO)
    m, cost = minuit_new.run_minuit_bundled(
        bundles, policy, args_fit, prefit_cache=prefit,
        toy_quiet=True, return_cost_fn=True)
    if scan_n:
        free_val  = float(m.fval)
        free_dm31 = float(m.values['dm2_31'])
        lo, hi = m.limits['dm2_31']
        m.fixed['dm2_31'] = True
        best_g, best_val = free_dm31, free_val
        scan = []
        for g in np.linspace(lo, hi, int(scan_n)):
            m.values['dm2_31'] = g
            m.migrad()
            scan.append((float(g), float(m.fval)))
            if m.fmin.is_valid and m.fval < best_val - 1e-4:
                best_g, best_val = float(g), float(m.fval)
        print(f"[MCMC] dm2_31 conditional scan "
              f"({'NO' if force_NO else 'IO'}): "
              + "  ".join(f"{g:+.3f}:{v:.2f}" for g, v in scan), flush=True)
        m.values['dm2_31'] = best_g
        m.migrad()                       # re-converge nuisances at best point
        m.fixed['dm2_31'] = False
        m.migrad()                       # release dm2_31 from that basin
        if best_val < free_val - 1e-3:
            print(f"[MCMC]   scan re-seeded the fit: chi2 {m.fval:.3f} @ "
                  f"dm2_31={m.values['dm2_31']:+.4f}  (free fit had "
                  f"{free_val:.3f} @ {free_dm31:+.4f})", flush=True)
    op_snap = dict(nuosc.op_nom)     # per-ordering pull centres
    return m, cost, op_snap, bundles


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",       required=True,
                   help="Config YAML or multi-phase manifest (as for run_noda)")
    p.add_argument("--base_path",    default=None)
    p.add_argument("--prefit-cache", default=None,
                   help="Optional prefit NPZ (as for run_fc_observed); if "
                        "omitted in a prefit fit_mode, the DYB+TAO prefit is "
                        "computed on the fly per ordering")
    p.add_argument("--output",       required=True, help="output NPZ path")
    p.add_argument("--ordering",     choices=("NO", "IO", "both"), default="both",
                   help="both: one chain over both orderings with the "
                        "sign-flip jump move (gives Bayesian NMO odds)")
    p.add_argument("--params",       choices=("osc", "all"), default="osc",
                   help="osc: sample the shared osc params only, everything "
                        "else fixed at the per-ordering Minuit best fit "
                        "(fast); all: sample every free Minuit parameter")
    p.add_argument("--sample-extra", default="",
                   help="comma-separated extra Minuit params to sample on "
                        "top of --params osc (e.g. Nrea,eff_nuis)")
    p.add_argument("--nsweeps",   type=int, default=2000)
    p.add_argument("--burnin",    type=int, default=500)
    p.add_argument("--thin",      type=int, default=1)
    p.add_argument("--ntemps",    type=int, default=10)
    p.add_argument("--beta-min",  type=float, default=0.02,
                   help="hottest inverse temperature; use ~1e-3 with more "
                        "temps if you want an accurate TI evidence")
    p.add_argument("--nwalkers",  type=int, default=0,
                   help="walkers per temperature (0: max(2*ndim+2, 16))")
    p.add_argument("--p-jump",    type=float, default=0.1,
                   help="probability of the NO<->IO sign-flip move (both mode)")
    p.add_argument("--p-bigjump", type=float, default=0.2,
                   help="probability a DE move uses gamma=1 (mode hopping)")
    p.add_argument("--init-dm31-spread", type=float, default=0.25,
                   help="fraction of initial walkers whose dm2_31 is drawn "
                        "uniformly over the prior window instead of at the "
                        "seed minimum — populates secondary wiggle minima "
                        "from the start (raise to ~0.5 for JUNO-only fits "
                        "without DYB/LBL constraints)")
    p.add_argument("--ncores",    type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--seed-scan", type=int, default=0,
                   help="N>0: after each ordering's free seed fit, profile "
                        "dm2_31 over an N-point grid across its window "
                        "(warm-started conditional fits) and re-seed from "
                        "the global conditional minimum. Use ~25-41 for "
                        "fits without DYB/LBL pulls, where the free fit "
                        "can land in a secondary wiggle minimum")
    p.add_argument("--bound-nsig", type=float, default=10.0,
                   help="half-width, in seed-fit sigma, of the synthesised "
                        "flat prior for params without Minuit limits")
    p.add_argument("--resume", default=None,
                   help="NPZ from a previous run: restart the walkers from "
                        "its final state (same config/params required). "
                        "Burn-in and ladder adaptation are skipped, so "
                        "chained jobs pay burn-in only once. Analyse all "
                        "segments together with noda_mcmc_results "
                        "--input seg1.npz seg2.npz ...")
    # Remaining --key value pairs are forwarded as config overrides,
    # exactly like run_noda / run_fc_observed CLI flags.
    return p.parse_known_args()


# ─────────────────────────────────────────────────────────────────────────────
def _bounds_for(sampled, seeds, nsig):
    """Flat-prior bounds per sampled param: Minuit limits where set, else
    best-fit ± nsig*sigma widened over all orderings."""
    lo = np.full(len(sampled), -np.inf)
    hi = np.full(len(sampled),  np.inf)
    for m, _, _ in seeds:
        for i, pname in enumerate(sampled):
            l, h = m.limits[pname]
            v, e = m.values[pname], max(abs(m.errors[pname]), 1e-12)
            l = l if np.isfinite(l) else v - nsig * e
            h = h if np.isfinite(h) else v + nsig * e
            lo[i] = min(lo[i], l) if np.isfinite(lo[i]) else l
            hi[i] = max(hi[i], h) if np.isfinite(hi[i]) else h
    return lo, hi


def _init_walkers(rng, seeds, sampled, lo, hi, in_support,
                  ntemps, nwalkers, j31, abs_win, dm31_spread=0.0):
    """Scatter walkers around the per-ordering best fits (alternating by
    walker index so both red/black halves contain every mode).

    With dm31_spread > 0, that fraction of walkers gets its dm2_31 drawn
    uniformly over the prior window (keeping the branch's sign) instead of
    near the seed minimum — so secondary wiggle minima (e.g. JUNO-only,
    no DYB/LBL pull) are populated from sweep 0 rather than having to be
    discovered by the hot chains."""
    d = len(sampled)
    branches = []
    for m, _, _ in seeds:
        c = np.array([m.values[p] for p in sampled], dtype=float)
        s = np.array([max(abs(m.errors[p]), 1e-10) for p in sampled])
        s = np.minimum(s, 0.1 * (hi - lo))       # migrad errors can be junk
        branches.append((c, s))
    X = np.empty((ntemps, nwalkers, d))
    for t in range(ntemps):
        for w in range(nwalkers):
            c, s0 = branches[w % len(branches)]
            spread31 = rng.random() < dm31_spread
            s = s0.copy()
            for _ in range(1000):
                x = c + 0.5 * s * rng.standard_normal(d)
                if spread31:
                    sign = 1.0 if c[j31] > 0 else -1.0
                    if abs_win is not None:
                        x[j31] = sign * rng.uniform(*abs_win)
                    else:
                        x[j31] = rng.uniform(lo[j31], hi[j31])
                if np.all(x >= lo) and np.all(x <= hi) and in_support(x):
                    break
                s *= 0.8                          # shrink toward best fit
            else:
                x = np.clip(c, lo, hi)
            X[t, w] = x
    return X


def main(argv=None):
    global _LOGLIKE
    cli, extra = parse()
    t0 = datetime.now()

    setups, policy = build_setups(cli.config, base_path=cli.base_path,
                                  remaining_argv=extra)
    if extra:
        print(f"[MCMC] config overrides: {' '.join(extra)}", flush=True)
    a0 = setups[0]['args_juno']
    if cli.prefit_cache:
        prefit = load_prefit_cache(cli.prefit_cache)
    elif (getattr(a0, 'fit_mode', 'combined').startswith('prefit')
            and getattr(a0, 'use_prefit', False)):
        # same auto-load rule as main._run_multi_bundle
        prefit = load_prefit_cache(a0.prefit_path)
        print(f"[MCMC] loaded prefit cache from config: {a0.prefit_path}",
              flush=True)
    else:
        prefit = None

    # ── 1. seed Minuit fits (one per ordering) ───────────────────────────
    orderings = {'NO': [1], 'IO': [0], 'both': [1, 0]}[cli.ordering]
    seeds = []                               # [(m, cost, op_snap)] per ordering
    seed_bundles = []
    for true_no in orderings:
        name = 'NO' if true_no else 'IO'
        print(f"[MCMC] seed fit ({name}) ...", flush=True)
        m, cost, op_snap, bundles = seed_fit(setups, policy, prefit,
                                             force_NO=true_no,
                                             scan_n=cli.seed_scan)
        print(f"[MCMC]   {name}: chi2_min={m.fval:.3f}  "
              f"dm2_31={m.values['dm2_31']:+.4f}", flush=True)
        seeds.append((m, cost, op_snap))
        seed_bundles.append(bundles)

    # ── 1b. LBL-pull solar pinning (match the frequentist convention) ────
    # run_minuit_bundled pins args.sin2_th12_lbl / dm2_21_lbl on its JUNO
    # bundles to ITS OWN best fit after migrad, so each closure already uses
    # fixed solar in f_lbl. The frequentist NMO-significance fit evaluates
    # the OPPOSITE ordering with the DATA fit's solar (m_opp runs after the
    # pin), so in both-ordering mode re-pin every branch to the primary
    # ordering's best-fit solar.
    if len(seeds) == 2 and getattr(a0, 'pull_dm2_31_lbl', False):
        i_pri = orderings.index(1 if bool(a0.NMO_opt) else 0)
        m_pri = seeds[i_pri][0]
        s12  = float(m_pri.values['sin2_th12'])
        dm21 = float(m_pri.values['dm2_21']) * 1e-5
        for bundles in seed_bundles:
            for b in bundles:
                if b.kind == 'juno':
                    b.detector.args.sin2_th12_lbl = s12
                    b.detector.args.dm2_21_lbl    = dm21
        print(f"[MCMC] f_lbl solar pinned to "
              f"{'NO' if orderings[i_pri] else 'IO'} data fit: "
              f"sin2_th12={s12:.5f}  dm2_21={dm21:.4e} eV2", flush=True)

    # ── 2. sampled-parameter set ─────────────────────────────────────────
    m0 = seeds[0][0]
    names_all = list(m0.parameters)
    free = [p for p in names_all if not m0.fixed[p]]
    for m, _, _ in seeds[1:]:
        if list(m.parameters) != names_all:
            raise RuntimeError("NO/IO fits disagree on the parameter list")
    if cli.params == 'osc':
        sampled = [p for p in SHARED_PARAM_NAMES if p in free]
        for p in [s.strip() for s in cli.sample_extra.split(',') if s.strip()]:
            if p not in free:
                raise ValueError(f"--sample-extra {p!r} is not a free Minuit "
                                 f"parameter")
            if p not in sampled:
                sampled.append(p)
    else:
        sampled = free
    if 'dm2_31' not in sampled:
        raise RuntimeError("dm2_31 must be free (it is the sampled NMO axis)")
    j31 = sampled.index('dm2_31')
    idx = np.array([names_all.index(p) for p in sampled])
    d   = len(sampled)
    print(f"[MCMC] sampling {d} parameters: {sampled}", flush=True)
    n_fixed = len(names_all) - d
    if n_fixed:
        print(f"[MCMC] {n_fixed} params held at per-ordering best fit "
              f"(profile-style; their uncertainty is NOT propagated unless "
              f"it is inside the chi2 covariance)", flush=True)

    # ── 3. flat priors from the Minuit limits ────────────────────────────
    lo, hi = _bounds_for(sampled, seeds, cli.bound_nsig)
    both = len(seeds) == 2
    if both:
        a, b = seeds[0][0].limits['dm2_31']       # NO window, e.g. (1.5, 3.5)
        abs_win = (min(abs(a), abs(b)), max(abs(a), abs(b)))
        lo[j31], hi[j31] = -abs_win[1], abs_win[1]
        widths = hi - lo
        widths[j31] = 2.0 * (abs_win[1] - abs_win[0])   # union support
    else:
        abs_win = None
        widths = hi - lo
    log_prior_const = -float(np.sum(np.log(widths)))

    def log_prior(X):
        ok = np.all((X >= lo) & (X <= hi), axis=1)
        if both:
            av = np.abs(X[:, j31])
            ok &= (av >= abs_win[0]) & (av <= abs_win[1])
        return np.where(ok, log_prior_const, -np.inf)

    def in_support(x):
        return np.isfinite(log_prior(x[None, :])[0])

    # ── 4. posterior with per-ordering dispatch ──────────────────────────
    post = NodaPosterior(j31)
    for true_no, (m, cost, op_snap) in zip(orderings, seeds):
        base = np.array([m.values[p] for p in names_all], dtype=float)
        post.add_ordering(+1 if true_no else -1, cost, base, idx, op_snap)
    _LOGLIKE = post

    # sanity + timing at the seed best fit(s)
    for true_no, (m, _, _) in zip(orderings, seeds):
        xbf = np.array([m.values[p] for p in sampled], dtype=float)
        t = time.perf_counter()
        ll = post(xbf)
        dt = time.perf_counter() - t
        if not np.isfinite(ll):
            raise RuntimeError("posterior not finite at the seed best fit")
        if abs(-2.0 * ll - m.fval) > max(1e-3 * abs(m.fval), 0.1):
            print(f"[MCMC] note: -2 logL at {('NO' if true_no else 'IO')} "
                  f"best fit = {-2*ll:.3f} vs Minuit fval = {m.fval:.3f} "
                  f"(small shifts are expected from the post-fit f_lbl "
                  f"solar pinning; large ones mean a fixed-param mismatch)",
                  flush=True)
        print(f"[MCMC] chi2 eval: {1e3*dt:.1f} ms", flush=True)

    # ── 5. NO<->IO involution jump ───────────────────────────────────────
    jump = None
    if both:
        c31 = (seeds[0][0].values['dm2_31'] + seeds[1][0].values['dm2_31'])

        def jump(X):
            Y = X.copy()
            Y[:, j31] = c31 - Y[:, j31]
            return Y
        print(f"[MCMC] NO<->IO jump: dm2_31 -> {c31:+.4f} - dm2_31", flush=True)

    # ── 6. sampler ───────────────────────────────────────────────────────
    rng = np.random.default_rng(cli.seed)
    nwalkers = cli.nwalkers or 0
    pool = None
    map_fn = None
    if cli.ncores > 1:
        # CPython dirties COW pages just by *reading* inherited objects
        # (refcounts + GC header bits), so each fork worker slowly privatises
        # the ~GB fit image. Freeze the fully-built heap into the permanent
        # generation so the cyclic GC never walks (and never dirties) it —
        # workers keep GC for their own new allocations.
        import gc
        gc.collect()
        gc.freeze()
        # maxtasksperchild: hard upper bound on per-worker RSS growth from
        # glibc heap ratcheting (each eval frees ~10-30 MB of large numpy
        # temporaries that malloc may keep; the freed-but-held pages count
        # against the condor cgroup). Recycling re-forks from the parent
        # image, so respawn is cheap and only the worker caches go cold.
        pool = mp.get_context('fork').Pool(cli.ncores, maxtasksperchild=400)

        _nbatch = [0]
        _next_rss = [0.0]

        def _rss_mb(pid):
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) // 1024
            except OSError:
                pass
            return 0

        def map_fn(_f, xs):
            _nbatch[0] += 1
            if time.time() >= _next_rss[0]:  # memory growth curve for the logs
                _next_rss[0] = time.time() + 300
                w = [_rss_mb(p.pid) for p in pool._pool]
                print(f"[MCMC] rss MB after {_nbatch[0]-1} batches: "
                      f"parent={_rss_mb(os.getpid())} workers={sum(w)} "
                      f"(max {max(w) if w else 0})", flush=True)
            return pool.map(_eval_loglike, xs,
                            chunksize=max(1, len(xs) // (4 * cli.ncores)))

    if cli.resume:
        prev = np.load(cli.resume, allow_pickle=True)
        prev_names = [str(s) for s in prev['param_names']]
        if prev_names != sampled:
            raise RuntimeError(f"--resume param set {prev_names} != current "
                               f"{sampled} (same config/--params needed)")
        x0 = np.asarray(prev['state_x'], dtype=float)
        sampler = PTSampler(post, log_prior, d,
                            nwalkers=x0.shape[1],
                            betas=np.asarray(prev['betas'], dtype=float),
                            jump=jump, p_jump=cli.p_jump,
                            p_bigjump=cli.p_bigjump,
                            map_fn=map_fn, rng=rng)
        if 'gscale' in prev:
            sampler.gscale = np.asarray(prev['gscale'], dtype=float).copy()
        burnin = 0
        print(f"[MCMC] resuming from {cli.resume} "
              f"(burn-in/adaptation skipped)", flush=True)
    else:
        sampler = PTSampler(post, log_prior, d,
                            nwalkers=nwalkers or None,
                            ntemps=cli.ntemps, beta_min=cli.beta_min,
                            jump=jump, p_jump=cli.p_jump,
                            p_bigjump=cli.p_bigjump,
                            map_fn=map_fn, rng=rng)
        x0 = _init_walkers(rng, seeds, sampled, lo, hi, in_support,
                           sampler.ntemps, sampler.nwalkers,
                           j31, abs_win, dm31_spread=cli.init_dm31_spread)
        burnin = cli.burnin
    print(f"[MCMC] ntemps={sampler.ntemps} nwalkers={sampler.nwalkers} "
          f"-> {(cli.nsweeps + burnin) * sampler.ntemps * sampler.nwalkers:,} "
          f"chi2 evals on {cli.ncores} cores", flush=True)

    sampler.initialize(x0)
    res = sampler.run(cli.nsweeps, burnin=burnin, thin=cli.thin,
                      progress_every=max(1, (cli.nsweeps + burnin) // 20))

    # ── 7. evidence + NMO odds ───────────────────────────────────────────
    # Never let a summary-stat failure lose the chain: anything that breaks
    # here is recomputable from the saved npz by noda_mcmc_results.
    logZ, mean_ll, f_no = float('nan'), np.full(sampler.ntemps, np.nan), float('nan')
    try:
        logZ, mean_ll = PTSampler.log_evidence(res['logl_all'], res['betas'])
        flat31 = res['chain'][:, :, j31].ravel()
        f_no = float(np.mean(flat31 > 0))
        print(f"\n[MCMC] TI log-evidence = {logZ:.3f} "
              f"(needs --beta-min ~1e-3 and more temps to be accurate)")
        if both:
            print(f"[MCMC] P(NO|data) = {f_no:.4f}   P(IO|data) = {1-f_no:.4f}")
            if 0 < f_no < 1:
                bf = f_no / (1 - f_no)
                # with equal prior volume per ordering, posterior odds = Bayes
                # factor; 2 ln B is on the same scale as the frequentist dchi2
                print(f"[MCMC] posterior odds NO/IO = {bf:.3g}   "
                      f"2 ln(B) = {2*np.log(bf):.2f}")
            signs = res['chain'][:, :, j31] > 0
            flips = int(np.sum(signs[1:] != signs[:-1]))
            print(f"[MCMC] cold-chain NO<->IO transitions: {flips} "
                  f"(want >~100 for a stable Bayes factor)")
    except Exception:
        import traceback
        traceback.print_exc()
        print("[MCMC] summary stats failed; saving chain anyway", flush=True)

    # ── 8. save ──────────────────────────────────────────────────────────
    meta_values = {('NO' if tn else 'IO'): {p: float(m.values[p]) for p in names_all}
                   for tn, (m, _, _) in zip(orderings, seeds)}
    meta_errors = {('NO' if tn else 'IO'): {p: float(m.errors[p]) for p in names_all}
                   for tn, (m, _, _) in zip(orderings, seeds)}
    meta_fval   = {('NO' if tn else 'IO'): float(m.fval)
                   for tn, (m, _, _) in zip(orderings, seeds)}
    os.makedirs(os.path.dirname(cli.output) or '.', exist_ok=True)
    np.savez_compressed(
        cli.output,
        chain=res['chain'], logl=res['logl'], logl_all=res['logl_all'],
        betas=res['betas'],
        state_x=sampler.x, gscale=res['gscale'],
        acc_de=res['acc_de'], acc_jump=res['acc_jump'],
        acc_swap=res['acc_swap'],
        param_names=np.array(sampled, dtype=str),
        all_param_names=np.array(names_all, dtype=str),
        bounds_lo=lo, bounds_hi=hi, j_dm31=j31,
        log_prior_const=log_prior_const,
        ordering=cli.ordering, both_mode=both,
        logZ_TI=logZ, mean_logl_per_temp=mean_ll, f_NO=f_no,
        seed_values=np.array([meta_values], dtype=object),
        seed_errors=np.array([meta_errors], dtype=object),
        seed_fval=np.array([meta_fval], dtype=object),
        config=str(cli.config),
        cli=str(vars(cli)))
    print(f"[MCMC] wrote {cli.output}")

    if pool is not None:
        pool.close()
        pool.join()
    print(f"[MCMC] total wall time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
