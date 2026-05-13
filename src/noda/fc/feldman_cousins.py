"""
feldman_cousins.py — toy spectrum generation & fit wrappers for FC studies.

Place in noda/fc/.

Two test statistics, both built on top of run_minuit() in fit_mode='prefit_*':

  * dm²₃₁ FC interval: per toy, fit twice — dm²₃₁ floating, then fixed at the
                       true value. TS = chi²_fix - chi²_min ≥ 0.
                       Both fits assume the same NMO; one prefit needed.

  * NMO test:          per toy, fit once with NO and once with IO.
                       T = chi²_IO - chi²_NO  (signed; > 0 favors NO).
                       Two prefits needed (one per assumed NMO).

Only the JUNO observation is fluctuated — DYB+TAO are held at Asimov so each
prefit is reusable across all toys.
"""

from __future__ import annotations

import os
import numpy as np
from copy import deepcopy
from typing import Optional, Dict, Any

from .. import nuosc
from ..noda import *  # Spectrum, etc.


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
                    "b2b": info["b2b_unc"], "name": info["name"],"b2b_scale": info['b2b_scale']}

    NL = ([args_juno.kB, args_juno.fC, args_juno.kI, args_juno.LY_shared]
          if args_juno.NL != "pull" else [0.0] * n_NL)
    abc = [args_juno.a, args_juno.b, args_juno.c]
    n_nuis = len(args_juno.seg_bins_nonuniform)

    out = juno_det.build_spectrum(
        sin2_12=sin2_12, sin2_13=sin2_13,
        dm2_21=dm2_21, dm2_31=true_dm2_31, me_rho=args_juno.me_rho,
        Nrea=0.0, NL=NL, abc=abc,
        bckgs_for_chi2=bckgs,
        snf_noneq_norm=[0.0] * n_rea,
        eff_nuis=0.0,
        ThU_ratio=args_juno.Th_U_ratio,
        core_powers_delta=[0.0] * n_core,
        efission=list(args_juno.efission),
        nuisance=np.zeros(n_nuis),
        opp=False)

    s_tot = out[1]
    return np.asarray(s_tot.bin_cont, dtype=float)


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
    ensp_toy['rtot_toy'] = Spectrum(bins=ensp_toy['rdet'].bins, bin_cont = np.asarray(toy_obs_array, dtype=float))
    return ensp_toy


# ─────────────────────────────────────────────────────────────────────────────
# Prefit cache I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_prefit_cache(path: str) -> dict:
    """Read a prefit cache produced by run_minuit (save_prefit_path)."""
    z = np.load(path)
    C = z['C_reco']
    return {'nuis_best': z['nuis_best'],
            'nuis_cov':  z['nuis_cov'],
            'C_reco':    None if C.size == 0 else C}


# ─────────────────────────────────────────────────────────────────────────────
# Single-fit wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_fit(*, run_minuit_fn,
                    toy_obs_juno: np.ndarray,
                    ensp_nom_juno: dict,
                    prefit_cache: dict,
                    fix_dm2_31_to: Optional[float] = None,
                    force_NO: bool = True,
                    common_kwargs: dict) -> dict:
    """
    Run one JUNO-only fit on a toy observation, with `prefit_cache` plugged
    into run_minuit so the DYB+TAO prefit step is skipped.
    """
    args_juno = deepcopy(common_kwargs['args_juno'])
    args_juno.NMO_opt = int(force_NO)

    ensp_juno_toy = make_toy_ensp(ensp_nom_juno, toy_obs_juno)

    kwargs = dict(common_kwargs)
    kwargs['args_juno']     = args_juno
    kwargs['ensp_nom_juno'] = ensp_juno_toy

    m = run_minuit_fn(
        prefit_cache=prefit_cache,
        toy_quiet=True,
        fix_dm2_31_to=fix_dm2_31_to,
        **kwargs)

    return {
        'fval':      float(m.fval),
        'is_valid':  bool(m.fmin.is_valid),
        'sin2_th12': float(m.values['sin2_th12']),
        'sin2_th13': float(m.values['sin2_th13']),
        'dm2_21':    float(m.values['dm2_21']),
        'dm2_31':    float(m.values['dm2_31']),
        'me_rho':    float(m.values['me_rho']),
    }


# ─────────────────────────────────────────────────────────────────────────────
# dm²₃₁ FC: single prefit (matches the assumed NMO of all fits)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_dm31(*, seed: int, true_dm2_31_minuit: float, true_NO: bool,
                     run_minuit_fn,
                     juno_det, args_juno_truth, backgrounds,
                     ensp_nom_juno: dict,
                     prefit_cache: dict,
                     common_kwargs: dict) -> dict:
    """
    One toy for the dm²₃₁ FC scan. Both fits are at the same NMO (true_NO),
    so one prefit cache (matching that NMO) is sufficient.

    `true_dm2_31_minuit` is in 1e-3 eV² (Minuit convention; positive number).
    Sign is set by `true_NO`: NO uses +, IO uses −.
    """
    fits_start  = datetime.now()
    
    rng = np.random.default_rng(seed)
    sign      = +1.0 if true_NO else -1.0
    true_phys = sign * true_dm2_31_minuit

    asimov  = build_juno_asimov(juno_det, args_juno_truth, backgrounds,
                                true_dm2_31=true_phys, true_NO=true_NO)
    toy_obs = fluctuate_poisson(asimov, rng)

    free = run_one_toy_fit(
        run_minuit_fn=run_minuit_fn, toy_obs_juno=toy_obs,
        ensp_nom_juno=ensp_nom_juno, prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=true_NO, common_kwargs=common_kwargs)

    fixed = run_one_toy_fit(
        run_minuit_fn=run_minuit_fn, toy_obs_juno=toy_obs,
        ensp_nom_juno=ensp_nom_juno, prefit_cache=prefit_cache,
        fix_dm2_31_to=true_dm2_31_minuit, force_NO=true_NO,
        common_kwargs=common_kwargs)
    
    fits_stop  = datetime.now()
    
    return {
        'seed':       int(seed),
        'true_dm31':  float(true_dm2_31_minuit),
        'true_NO':    int(true_NO),
        'chi2_free':  free['fval'],
        'chi2_fix':   fixed['fval'],
        'TS_dm31':    abs(fixed['fval'] - free['fval']),
        'best_dm31':  free['dm2_31'],
        'best_dm21':  free['dm2_21'],
        'best_sin12': free['sin2_th12'],
        'best_sin13': free['sin2_th13'],
        'free_valid': int(free['is_valid']),
        'fix_valid':  int(fixed['is_valid']),
        'time': (fits_stop-fits_start).total_seconds()
    }


# ─────────────────────────────────────────────────────────────────────────────
# NMO FC: TWO prefits — one per assumed-NMO of the JUNO fit
# ─────────────────────────────────────────────────────────────────────────────

def run_one_toy_nmo(*, seed: int, true_NO: bool,
                    run_minuit_fn,
                    juno_det, args_juno_truth, backgrounds,
                    ensp_nom_juno: dict,
                    prefit_cache: dict,
                    common_kwargs: dict) -> dict:
    """
    One toy for the NMO test. Toy generated under `true_NO`. Then JUNO fit
    once with NO assumption (using prefit_cache)
    """
    fits_start  = datetime.now()
    rng = np.random.default_rng(seed)

    asimov = build_juno_asimov(
        juno_det, args_juno_truth, backgrounds,
        true_dm2_31=nuosc.op_nom['dm2_31'], true_NO=true_NO)
    toy_obs = fluctuate_poisson(asimov, rng)

    fit_NO = run_one_toy_fit(
        run_minuit_fn=run_minuit_fn, toy_obs_juno=toy_obs,
        ensp_nom_juno=ensp_nom_juno, prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=True, common_kwargs=common_kwargs)

    fit_IO = run_one_toy_fit(
        run_minuit_fn=run_minuit_fn, toy_obs_juno=toy_obs,
        ensp_nom_juno=ensp_nom_juno, prefit_cache=prefit_cache,
        fix_dm2_31_to=None, force_NO=False, common_kwargs=common_kwargs)

    fits_stop  = datetime.now()
    return {
        'seed':     int(seed),
        'true_NO':  int(true_NO),
        'chi2_NO':  fit_NO['fval'],
        'chi2_IO':  fit_IO['fval'],
        'T_nmo':    abs(fit_IO['fval'] - fit_NO['fval']),   # > 0 favors NO
        'NO_valid': int(fit_NO['is_valid']),
        'IO_valid': int(fit_IO['is_valid']),
        'dm31_NO':  fit_NO['dm2_31'],
        'dm31_IO':  fit_IO['dm2_31'],
        'time': (fits_stop-fits_start).total_seconds()
    }