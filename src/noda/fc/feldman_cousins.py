"""
feldman_cousins.py — toy spectrum generation & fit wrappers for FC studies.

Place in noda/fc/.

Two test statistics, both built on top of run_minuit_bundled() in
fit_mode='prefit_*':

  * dm²₃₁ FC interval: per toy, fit twice — dm²₃₁ floating, then fixed at the
                       true value. TS = chi²_fix - chi²_min ≥ 0.
                       Both fits assume the same NMO; one prefit needed.

  * NMO test:          per toy, fit once with NO and once with IO.
                       T = chi²_IO - chi²_NO  (signed; > 0 favors NO).
                       Two prefits needed (one per assumed NMO).

Only the JUNO observation is fluctuated — DYB+TAO are held at Asimov so each
prefit is reusable across all toys.

Multi-phase support: every per-phase JUNO dataset (one setup dict from
main.setup_one_phase per manifest row) gets its own toy observation,
generated from ONE shared nuisance sample resolved through the ParamPolicy
(so parameters correlated across phases are sampled once, per-phase
parameters use their own Minuit value). The fits bundle all phases together
exactly like main.py does.
"""

from __future__ import annotations

import os
import numpy as np
from copy import deepcopy
from typing import Optional, Dict, Any

from .. import nuosc
from ..noda import *  # Spectrum, etc.
from ..fit import combined_fit as minuit_new
from ..fit.shared_nuisances import get_shape_nuis_names


# ─────────────────────────────────────────────────────────────────────────────
# Param-name plumbing (phases/shells)
# ─────────────────────────────────────────────────────────────────────────────

def base_param(name: str) -> str:
    """'abc1__pP25C__sFV1' → 'abc1' (strip the policy mangling suffix)."""
    return name.split('__', 1)[0]


def make_resolver(policy, phase: str, shell: str):
    """base name → Minuit name for one (phase, shell) dataset."""
    return lambda base: policy.resolve(base, phase, shell)


def n_shape_of(args_juno) -> int:
    return len(args_juno.seg_bins_nonuniform) - 1


def build_fit_bundles(setups, *, force_NO, toy_obs_list=None):
    """
    One JUNO DetectorBundle per setup, with the toy observation (if given)
    swapped in and NMO_opt forced. Returns (bundles, args_fit) where
    args_fit is the first phase's args copy — it drives the global fit
    behaviour inside run_minuit_bundled.
    """
    from ..main import _build_juno_bundle
    bundles, args_fit = [], None
    for i, s in enumerate(setups):
        s2 = dict(s)
        s2['args_juno'] = deepcopy(s['args_juno'])
        s2['args_juno'].NMO_opt = bool(force_NO)
        if toy_obs_list is not None:
            s2['ensp_nom_juno'] = make_toy_ensp(s['ensp_nom_juno'],
                                                toy_obs_list[i])
        bundles.append(_build_juno_bundle(s2))
        if args_fit is None:
            args_fit = s2['args_juno']
    return bundles, args_fit


def frozen_dm31_scan(cost_fn, param_names, base_values, dm31_grid_phys):
    """Evaluate the run_minuit_bundled chi2 closure with ALL nuisances frozen
       at base_values, scanning dm²₃₁ over dm31_grid_phys (physical eV²)."""
    j   = list(param_names).index('dm2_31')
    out = np.empty(len(dm31_grid_phys))
    for k, dm_phys in enumerate(dm31_grid_phys):
        v = list(base_values)
        v[j] = dm_phys * 1e3                  # physical eV² → Minuit units
        out[k] = cost_fn(*v)                  # ← YOUR chi2
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Common refit helper
# ─────────────────────────────────────────────────────────────────────────────

def maybe_refit(*, setups, policy, toy_obs_list, prefit_cache,
                initial_fit, cost_fn, true_phys, force_NO, toy_dm2_31,
                refit_gap=0.05e-3):
    """
    Frozen-nuisance scan around true_phys to check for a better minimum.
    If the scan minimum is far from the free-fit minimum, refit from there.
    Returns the best fit (original or refitted).

    Parameters
    ----------
    initial_fit   : Minuit result from the free fit
    cost_fn       : chi2 closure returned alongside initial_fit
    true_phys     : centre of the scan grid (physical eV²)
    force_NO      : NMO assumption passed to run_one_toy_fit
    refit_gap     : threshold (physical eV²) above which a refit is triggered
    """
    names     = list(initial_fit.parameters)
    base      = [initial_fit.values[p] for p in names]
    dm31_free = initial_fit.values['dm2_31'] * 1e-3   # physical eV²

    dm31_grid     = np.linspace(true_phys - 1e-3, true_phys + 1e-3, 100)
    chi2_scan     = frozen_dm31_scan(cost_fn, names, base, dm31_grid)
    dm31_scan_min = float(dm31_grid[int(np.argmin(chi2_scan))])

    best = initial_fit
    if abs(dm31_scan_min - dm31_free) > refit_gap:
        refit = run_one_toy_fit(
            setups=setups, policy=policy, toy_obs_list=toy_obs_list,
            prefit_cache=prefit_cache,
            fix_dm2_31_to=None, force_NO=force_NO, toy_dm2_31=toy_dm2_31,
            true_dm2_31=dm31_scan_min)
        if refit.fval < initial_fit.fval:
            best = refit
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Asimov prediction & toy generation
# ─────────────────────────────────────────────────────────────────────────────

def build_juno_asimov(juno_det, args_juno, backgrounds: Dict[str, Any],
                      true_dm2_31: float, true_NO: bool,
                      true_sin2_12: Optional[float] = None,
                      true_sin2_13: Optional[float] = None,
                      true_dm2_21: Optional[float] = None) -> np.ndarray:
    """
    Total predicted JUNO spectrum (signal + bkgs) at the requested truth point,
    all nuisances at zero. Returns a numpy array (bin_cont).

    `true_dm2_31` is the PHYSICAL value (eV²); pass e.g. 2.50e-3.
    """
    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=true_NO)
    nom = nuosc.op_nom

    sin2_12 = true_sin2_12 if true_sin2_12 is not None else nom['sin2_th12']
    sin2_13 = true_sin2_13 if true_sin2_13 is not None else nom['sin2_th13']
    dm2_21  = true_dm2_21  if true_dm2_21  is not None else nom['dm2_21']

    pn = juno_det.get_param_names()
    n_NL   = sum(1 for p in pn if p.startswith("NL"))
    n_rea  = sum(1 for p in pn if p.startswith("snf_noneq_norm"))
    n_core = sum(1 for p in pn if p.startswith("core_power"))

    bckgs = {}
    for k, info in backgrounds["JUNO"].items():
        bckgs[k] = {"pull": 0.0, "sigma": info["rate_unc"],
                    "b2b": info["b2b_unc"], "name": info["name"],
                    "b2b_scale": info['b2b_scale']}

    NL = ([args_juno.kB, args_juno.fC, args_juno.kI, args_juno.LY_shared]
          if args_juno.NL != "pull" else [0.0] * n_NL)
    abc = [args_juno.a, args_juno.b, args_juno.c]
    n_nuis = n_shape_of(args_juno)

    out = juno_det.build_spectrum(
        sin2_12=sin2_12, sin2_13=sin2_13,
        dm2_21=dm2_21, dm2_31=true_dm2_31, me_rho=args_juno.me_rho,
        Nrea=0.0, NL=NL, abc=abc,
        bckgs_for_chi2=bckgs,
        snf_noneq_norm=[0.0] * n_rea,
        eff_nuis=0.0,
        ThU_ratio=args_juno.Th_U_ratio,
        core_powers_delta=[0.0] * n_core,
        efission=list(np.array(args_juno.efission) * 1e-2),
        nuisance=np.zeros(n_nuis),
        opp=False)

    s_tot = out[1]
    return np.asarray(s_tot.bin_cont, dtype=float)

def sample_er_nuisances(rng, central_values, covariance_er, resolve):
    """
    Sample [a, b, c, kB, fC, kI, LY_shared] jointly from the ER covariance,
    centred on the best-fit values from the initial fit. `resolve` maps the
    base parameter names to this dataset's Minuit names.
    """
    names = [resolve(n) for n in
             ['abc0', 'abc1', 'abc2', 'NL0', 'NL1', 'NL2', 'NL3']]
    theta0 = np.array([central_values[n] for n in names])
    L = np.linalg.cholesky(covariance_er + 1e-12 * np.eye(len(theta0)))
    z = rng.standard_normal(len(theta0))
    theta_sampled = theta0 + L @ z
    return dict(zip(names, theta_sampled))


def sample_nuisances(rng, central, param_names, prefit_cov, shape_param_names,
                     truncate_below_zero, truncate_below_mone, nominal_uncertainties):
    """
    Sample one nuisance vector.
    - Shape nuisances: correlated multivariate Gaussian (Cholesky of prefit_cov)
    - Other nuisances: independent Gaussian at nominal uncertainty
    - Some are truncated to be ≥ 0
    Central values come from the initial fit.
    """
    sample = dict(central)   # start at central values, will overwrite

    # ── shape nuisances via Cholesky of prefit cov ───────────────────────────
    if prefit_cov is not None and len(shape_param_names):
        L = np.linalg.cholesky(prefit_cov + 1e-12 * np.eye(len(prefit_cov)))  # jitter
        z = rng.standard_normal(len(shape_param_names))
        delta = L @ z
        for i, name in enumerate(shape_param_names):
            sample[name] = central[name] + delta[i]

    # ── independent Gaussians for other nuisances ────────────────────────────
    for name in param_names:
        if name in shape_param_names:
            continue
        sigma = nominal_uncertainties.get(name, None)
        if sigma is None or sigma <= 0:
            continue   # not a free nuisance; leave at central
        val = central[name] + rng.normal(0.0, sigma)

        if name in truncate_below_zero and val < 0.0:
            while val < 0.0:
                val = central[name] + rng.normal(0.0, sigma)
        if name in truncate_below_mone and val < -1.0:
            while val < -1.0:
                val = central[name] + rng.normal(0.0, sigma)
        sample[name] = val
    return sample


def build_juno_from_sampled(juno_det, args_juno, backgrounds, sample,
                             true_dm2_31, true_NO, resolve=lambda n: n):
    """Like build_juno_asimov but uses sampled nuisances instead of zeros.
       `resolve` maps base parameter names to this dataset's Minuit names."""
    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=true_NO)
    nom = nuosc.op_nom
    pn = juno_det.get_param_names()

    bckgs = {}
    for k, info in backgrounds["JUNO"].items():
        bckgs[k] = {"pull": sample[resolve(f"bkg_{k}")], "sigma": info["rate_unc"],
                    "b2b": info["b2b_unc"], "name": info["name"],
                    "b2b_scale": info['b2b_scale']}

    NL      = [sample[resolve(p)] for p in pn if p.startswith("NL")]
    abc     = [sample[resolve(f"abc{i}")] for i in range(3)]
    snf     = [sample[resolve(p)] for p in pn if p.startswith("snf_noneq_norm")]
    core_d  = [sample[resolve(p)] for p in pn if p.startswith("core_power")]
    efis    = [sample[resolve(p)] for p in pn if p.startswith("efission")]
    nuis    = [sample[f"nuis{i}"] for i in range(n_shape_of(args_juno))]

    out = juno_det.build_spectrum(
        sin2_12=sample['sin2_th12'], sin2_13=sample['sin2_th13'],
        dm2_21=sample['dm2_21'] * 1e-5, dm2_31=true_dm2_31,
        me_rho=sample['me_rho'],
        Nrea=sample[resolve('Nrea')], NL=NL, abc=abc,
        bckgs_for_chi2=bckgs, snf_noneq_norm=snf,
        eff_nuis=sample[resolve('eff_nuis')],
        ThU_ratio=sample[resolve('ThU_ratio')],
        core_powers_delta=core_d, efission=efis,
        nuisance=nuis, opp=False)
    return np.asarray(out[1].bin_cont, dtype=float)


def fluctuate_poisson(prediction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent Poisson per bin, clipped to ≥ 0."""
    return rng.poisson(np.maximum(prediction, 0.0)).astype(float)


def make_toy_ensp(ensp_nom_juno: dict, toy_obs_array: np.ndarray) -> dict:
    """
    Deep-copy ensp_nom_juno with the *observed* spectrum bin_cont replaced.

    ASSUMPTION: the observed JUNO spectrum used by JUNODetector.chi2_contribution
    lives in ensp_nom_juno['rdet']. If that's a different key, change it here.
    """
    ensp_toy = deepcopy(ensp_nom_juno)
    ensp_toy['rtot_toy'] = Spectrum(
        bins=ensp_toy['rdet'].bins,
        bin_cont=np.asarray(toy_obs_array, dtype=float))
    return ensp_toy


# ─────────────────────────────────────────────────────────────────────────────
# Single-fit wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_fit(*, setups, policy,
                    toy_obs_list,
                    prefit_cache: dict,
                    fix_dm2_31_to: Optional[float] = None,
                    force_NO: bool = True,
                    toy_dm2_31: float = None,
                    true_dm2_31: Optional[float] = None,
                    return_cost_fn: bool = False):
    """
    Run one JUNO-only bundled fit on a list of toy observations (one per
    phase/shell setup), with `prefit_cache` plugged into run_minuit_bundled
    so the DYB+TAO prefit step is skipped.
    """
    bundles, args_fit = build_fit_bundles(setups, force_NO=force_NO,
                                          toy_obs_list=toy_obs_list)
    return minuit_new.run_minuit_bundled(
        bundles, policy, args_fit,
        prefit_cache=prefit_cache,
        toy_quiet=True,
        fix_dm2_31_to=fix_dm2_31_to,
        toy_dm2_31=toy_dm2_31,
        true_dm2_31=true_dm2_31,
        return_cost_fn=return_cost_fn)


# ─────────────────────────────────────────────────────────────────────────────
# dm²₃₁ FC: single prefit (matches the assumed NMO of all fits)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_dm31(*, seed: int, true_dm2_31_minuit: float, true_NO: bool,
                     setups, policy, truth_bundles,
                     initial_fits: dict,
                     prefit_cache: dict) -> dict:
    """
    One toy for the dm²₃₁ FC scan. Both fits are at the same NMO (true_NO),
    so one prefit cache (matching that NMO) is sufficient.

    `true_dm2_31_minuit` is in 1e-3 eV² (Minuit convention; positive number).
    Sign is set by `true_NO`: NO uses +, IO uses −.
    `truth_bundles` are the nominal (non-toy) JUNO bundles, one per setup,
    used only to build the truth prediction.
    """
    fits_start = datetime.now()

    rng       = np.random.default_rng(seed)
    sign      = +1.0 if true_NO else -1.0
    true_phys = sign * true_dm2_31_minuit * 1e-3   # physical eV²

    args_juno_truth   = setups[0]['args_juno']
    shape_param_names = get_shape_nuis_names(n_shape_of(args_juno_truth))
    grid_key          = float(true_dm2_31_minuit) if true_NO else -float(true_dm2_31_minuit)
    fit_data          = initial_fits['grid_best_fits'][grid_key]
    central_values    = fit_data['values']

    nominal_uncertainties = initial_fits['nominal_uncertainties']

    sample = sample_nuisances(
        rng,
        central=central_values,
        param_names=initial_fits['param_names'],
        prefit_cov=prefit_cache.get('nuis_cov') if prefit_cache else None,
        shape_param_names=shape_param_names,
        truncate_below_zero=initial_fits['truncate_below_zero'],
        truncate_below_mone=initial_fits['truncate_below_mone'],
        nominal_uncertainties=nominal_uncertainties)

    for name in initial_fits['param_names']:
        base = base_param(name)
        if base.startswith('efiss'):
            nominal_uncertainties[name] = args_juno_truth.efission_err[int(base[-1])]*1e-2

    # ── one toy observation per phase/shell, from the SAME sample ────────────
    toy_obs_list = []
    for b in truth_bundles:
        resolve = make_resolver(policy, b.phase, b.shell)
        if b.args.NL == "real":
            er_sample = sample_er_nuisances(rng, central_values,
                                            b.detector.covariance_er, resolve)
            sample.update(er_sample)
        pred = build_juno_from_sampled(
            b.detector, b.args, b.backgrounds, sample,
            true_dm2_31=true_phys, true_NO=true_NO, resolve=resolve)
        toy_obs_list.append(fluctuate_poisson(pred, rng))

    toy_dm2_31 = None
    if args_juno_truth.pull_dm2_31:
        nuosc.SetOscillationParameters(opt=args_juno_truth.PDG_opt, NO=true_NO)
        dyb_sigma  = nuosc.op_nom['dm2_31_err']
        toy_dm2_31 = rng.normal(loc=true_phys, scale=dyb_sigma)
    if args_juno_truth.pull_dm2_31_lbl:
        lbl_sigma  = args_juno_truth.dm2_31_lbl_err
        toy_dm2_31 = rng.normal(loc=abs(true_phys), scale=lbl_sigma)

    # ── free fit ──────────────────────────────────────────────────────────────
    free, cost_fn = run_one_toy_fit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=true_NO, toy_dm2_31=toy_dm2_31,
        true_dm2_31=true_phys, return_cost_fn=True)

    # ── scan + optional refit ─────────────────────────────────────────────────
    free = maybe_refit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        initial_fit=free, cost_fn=cost_fn, true_phys=true_phys,
        force_NO=true_NO, toy_dm2_31=toy_dm2_31)

    # ── fixed fit at true dm²₃₁ ───────────────────────────────────────────────
    fixed = run_one_toy_fit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        fix_dm2_31_to=true_phys * 1e3, force_NO=true_NO, toy_dm2_31=toy_dm2_31)

    fits_stop = datetime.now()

    return {
        'seed':       int(seed),
        'true_dm31':  float(true_dm2_31_minuit),
        'true_NO':    int(true_NO),
        'chi2_free':  free.fval,
        'chi2_fix':   fixed.fval,
        'TS_dm31':    fixed.fval - free.fval,
        'best_dm31':  free.values['dm2_31'],
        'best_dm21':  free.values['dm2_21'],
        'best_sin12': free.values['sin2_th12'],
        'best_sin13': free.values['sin2_th13'],
        'free_valid': int(free.fmin.is_valid),
        'fix_valid':  int(fixed.fmin.is_valid),
        'time': (fits_stop - fits_start).total_seconds()
    }


# ─────────────────────────────────────────────────────────────────────────────
# NMO FC: TWO prefits — one per assumed-NMO of the JUNO fit
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_nmo(*, seed: int, true_NO: bool,
                    setups, policy, truth_bundles,
                    prefit_cache: dict) -> dict:
    """
    One toy for the NMO test. Toy generated under `true_NO`. JUNO is then fit
    once with NO and once with IO. T = chi²_IO - chi²_NO (> 0 favours NO).
    """
    fits_start = datetime.now()

    rng = np.random.default_rng(seed)
    args_juno_truth = setups[0]['args_juno']
    nuosc.SetOscillationParameters(opt=args_juno_truth.PDG_opt, NO=true_NO)
    true_phys = nuosc.op_nom['dm2_31']   # physical eV², signed by NMO below

    # Asimov-based toys (no nuisance sampling needed for NMO sensitivity)
    toy_obs_list = []
    for b in truth_bundles:
        asimov = build_juno_asimov(
            b.detector, b.args, b.backgrounds,
            true_dm2_31=true_phys, true_NO=true_NO)
        toy_obs_list.append(fluctuate_poisson(asimov, rng))

    toy_dm2_31 = None
    if args_juno_truth.pull_dm2_31:
        nuosc.SetOscillationParameters(opt=args_juno_truth.PDG_opt, NO=true_NO)
        dyb_sigma  = nuosc.op_nom['dm2_31_err']
        toy_dm2_31 = rng.normal(loc=true_phys, scale=dyb_sigma)
    if args_juno_truth.pull_dm2_31_lbl:
        lbl_sigma  = args_juno_truth.dm2_31_lbl_err
        toy_dm2_31 = rng.normal(loc=args_juno_truth.dm2_31_lbl, scale=lbl_sigma)

    # True physical dm²₃₁ for each ordering (scan centre)
    true_phys_NO = +abs(true_phys)
    true_phys_IO = -abs(true_phys)

    # ── IO fit + refit ────────────────────────────────────────────────────────
    fit_IO, cost_fn_IO = run_one_toy_fit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=False, return_cost_fn=True,
        toy_dm2_31=toy_dm2_31)

    fit_IO = maybe_refit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        initial_fit=fit_IO, cost_fn=cost_fn_IO, true_phys=true_phys_IO,
        force_NO=False, toy_dm2_31=toy_dm2_31)

    # ── NO fit + refit ────────────────────────────────────────────────────────
    fit_NO, cost_fn_NO = run_one_toy_fit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=True, return_cost_fn=True,
        toy_dm2_31=toy_dm2_31)

    fit_NO = maybe_refit(
        setups=setups, policy=policy, toy_obs_list=toy_obs_list,
        prefit_cache=prefit_cache,
        initial_fit=fit_NO, cost_fn=cost_fn_NO, true_phys=true_phys_NO,
        force_NO=True, toy_dm2_31=toy_dm2_31)

    fits_stop = datetime.now()

    return {
        'seed':     int(seed),
        'true_NO':  int(true_NO),
        'chi2_NO':  fit_NO.fval,
        'chi2_IO':  fit_IO.fval,
        'T_nmo':    fit_IO.fval - fit_NO.fval,   # > 0 favors NO
        'NO_valid': bool(fit_NO.fmin.is_valid),
        'IO_valid': bool(fit_IO.fmin.is_valid),
        'dm31_NO':  fit_NO.values['dm2_31'],
        'dm31_IO':  fit_IO.values['dm2_31'],
        'time': (fits_stop - fits_start).total_seconds()
    }
