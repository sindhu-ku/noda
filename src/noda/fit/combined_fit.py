#!/usr/bin/env python3
"""
combined_fit.py

Combined JUNO + DYB/TAO fit via iMinuit.

Changes from the original monolithic run_minuit:
  - 'nd' is gone; detectors are explicit: dyb / tao
  - me_rho is a SHARED parameter (index 4, after the 4 osc params)
  - TAO has its own nuisance block in the Minuit parameter vector
  - All spectrum building is delegated to JUNODetector / DYBDetector / TAODetector

Minuit parameter vector layout
───────────────────────────────────────────────────────────────────────────
  [shared block]
    0  sin2_th12
    1  sin2_th13
    2  dm2_21
    3  dm2_31
    4  me_rho
  [JUNO block]
    5  Nrea
    6  NL0 … NL{n_NL-1}
    +  abc0 abc1 abc2
    +  bkg_{key} × n_bckgs_juno
    +  snf_noneq_norm0 … snf_noneq_norm{n_rea-1}
    +  eff_nuis
    +  ThU_ratio
    +  nuis0 … nuis{n_nuis-1}     (shape nuisances, PMOP_combined only)
    +  core_power0 … core_power9
    +  efission0 … efission{n_efis-1}
  [TAO block]  (only when fit_type == "PMOP_combined" and use_tao == True)
    +  Nrea_tao
    +  NL_tao0 … NL_tao4
    +  abc_tao0 abc_tao1 abc_tao2
    +  bkg_tao
    +  core_power_tao
    +  eff_nuis_tao
  [DYB block]  (only when fit_type == "PMOP_combined" and use_tao == False)
    +  Nrea_dyb
───────────────────────────────────────────────────────────────────────────
"""

import sys
import numpy as np
import os
import inspect
import matplotlib.pyplot as plt
from iminuit import Minuit

from .. import nuosc as nuosc
from ..noda import *
from .juno_fit import JUNODetector
from .dyb_fit  import DYBDetector
from .tao_fit  import TAODetector
from .shared_nuisances import *
from .minuit import *

FULL_CHI2  = 0
DYB_BINS   = 0
FREE_PARAM = 0

def run_dyb_tao_prefit(dyb_det, tao_det, E_segments, unc_dyb, unc_tao,
                       args_juno, args_tao):
    """
    Fit DYB+TAO with oscillation params fixed to nominal.
    Returns: (nuis_best, nuis_cov)
    """
    n_nuis = len(E_segments)
    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=bool(args_juno.NMO_opt))
    sin2_12 = nuosc.op_nom["sin2_th12"]
    sin2_13 = nuosc.op_nom["sin2_th13"]
    dm2_21  = nuosc.op_nom["dm2_21"]
    dm2_31  = nuosc.op_nom["dm2_31"]

    tao_param_names  = tao_det.get_param_names()  if tao_det else []
    tao_start_values = tao_det.get_start_values() if tao_det else []
    dyb_param_names  = dyb_det.get_param_names()  if dyb_det else []
    dyb_start_values = dyb_det.get_start_values() if dyb_det else []

    nuis_names  = get_shape_nuis_names(n_nuis)
    nuis_starts = get_shape_nuis_start_values(n_nuis)

    all_names  = nuis_names  + tao_param_names  + dyb_param_names
    all_starts = nuis_starts + tao_start_values + dyb_start_values

    def prefit_chi2(*params):
        nuisance_vals   = list(params[:n_nuis])
        tao_vals        = list(params[n_nuis : n_nuis + len(tao_param_names)])
        dyb_vals        = list(params[n_nuis + len(tao_param_names):])
        tao_params_dict = dict(zip(tao_param_names, tao_vals))
        dyb_params_dict = dict(zip(dyb_param_names, dyb_vals))

        chi2_tao = 0.0
        chi2_dyb = 0.0

        if tao_det is not None:
            (chi2_tao, _, _s, _st) = tao_det.chi2_contribution(
                sin2_12=sin2_12, sin2_13=sin2_13,
                dm2_21=dm2_21, dm2_31=dm2_31,
                me_rho=0.0,
                Nrea_tao=tao_params_dict['Nrea_tao'],
                NL_tao=[tao_params_dict[f'NL_tao{i}'] for i in range(5)],
                snf_noneq_norm_tao=[tao_params_dict[f'snf_noneq_norm_tao{i}']
                                    for i in range(2)],
                abc_tao=[tao_params_dict['abc_tao0'],
                         tao_params_dict['abc_tao1'],
                         tao_params_dict['abc_tao2']],
                core_power_tao=tao_params_dict['core_power_tao'],
                eff_nuis_tao=tao_params_dict['eff_nuis_tao'],
                unc=unc_tao, nuisance=nuisance_vals, opp=False)

        if dyb_det is not None:
            chi2_dyb, _s, _st = dyb_det.chi2_contribution(
                nuisance_coarse=np.array(nuisance_vals),
                snf_noneq_norm_dyb=dyb_params_dict['snf_noneq_norm_dyb'],
                E_segments=E_segments)

        return chi2_tao + chi2_dyb

    prefit_chi2.__signature__ = inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for n in all_names
    ])

    m_pre = Minuit(prefit_chi2, *all_starts, name=all_names)
    if tao_det: tao_det.apply_limits(m_pre)
    if dyb_det: dyb_det.apply_limits(m_pre)
    for n in nuis_names:
        m_pre.limits[n] = [-1, None]

    m_pre.migrad()
    m_pre.hesse()

    nuis_best = np.array([m_pre.values[n] for n in nuis_names])
    idx       = [list(m_pre.parameters).index(n) for n in nuis_names]
    nuis_cov  = np.array(m_pre.covariance)[np.ix_(idx, idx)]

    print("DYB+TAO pre-fit complete.")
    print("  nuis_best:", nuis_best)
    print("  nuis_cov diag:", np.diag(nuis_cov))
    return nuis_best, nuis_cov

def propagate_nuis_to_reco(juno_det, nuis_best, nuis_cov,
                            sin2_12, sin2_13, dm2_21, dm2_31,
                            fixed_juno_params, args_juno, unc_juno,
                            eps=1e-4):
    """
    Propagate nuisance covariance into JUNO reco-energy space.
    J[b, i] = dS_reco[b] / dnuis[i]  evaluated at nuis_best.
    Returns C_reco = J @ nuis_cov @ J.T  (n_reco_bins x n_reco_bins)
    """
    n_nuis = len(nuis_best)
    def spectrum_at(nuis):
        (_, s_tot, _, _, _, _, _, _, _) = juno_det.build_spectrum(
            sin2_12=sin2_12, sin2_13=sin2_13,
            dm2_21=dm2_21, dm2_31=dm2_31, me_rho=0.0,
            nuisance=list(nuis),
            Nrea=fixed_juno_params['Nrea'],
            NL=fixed_juno_params['NL'],
            abc=fixed_juno_params['abc'],
            bckgs_for_chi2=fixed_juno_params['bckgs'],
            snf_noneq_norm=fixed_juno_params['snf_noneq_norm'],
            eff_nuis=fixed_juno_params['eff_nuis'],
            ThU_ratio=fixed_juno_params['ThU_ratio'],
            core_powers_delta=fixed_juno_params['core_powers_delta'],
            efission=fixed_juno_params['efission'],
            opp=False)
        return np.array(s_tot.bin_cont)
    S0 = spectrum_at(nuis_best)
    n_reco = len(S0)
    J = np.zeros((n_reco, n_nuis))

    for i in range(n_nuis):
        nuis_up        = nuis_best.copy()
        nuis_up[i]    += eps
        nuis_dn        = nuis_best.copy()
        nuis_dn[i]    -= eps
        J[:, i]        = (spectrum_at(nuis_up) - spectrum_at(nuis_dn)) / (2 * eps)

    C_reco = J @ nuis_cov @ J.T
    return C_reco
    
def run_minuit(ndays, ndays_dyb=0, ndays_tao=0,
               ensp_nom_juno={}, ensp_nom_dyb={}, ensp_nom_tao={},
               unc_juno='', unc_dyb='', unc_tao='', unc_corr='',
               rm=[], rm_dyb=[], rm_tao=[],
               cm_juno={}, cm_dyb={}, cm_tao={}, cm_corr='',
               args_juno='', args_dyb='', args_tao='',
               dyb_model=None, backgrounds=None,
               ene_leak_tao=None,
               toy=False, fix_param=None):
    """
    Combined JUNO + DYB/TAO fit.

    - Set ensp_nom_tao / args_tao / rm_tao / cm_tao to use TAO.
    - Set ensp_nom_dyb / args_dyb / rm_dyb / cm_dyb / dyb_model to use DYB.
    - args_juno.fit_type == "PMOP_combined" activates the near-detector term.
    - Which near detector is active is determined by bool(ensp_nom_tao).
    """

    # ── load reactor summary ─────────────────────────────────────────────────
    core_names, core_powers_std, core_fractions, core_baselines = \
        load_reactor_summary(args_juno.reactor_summary_yaml)

    E_segments = np.array(args_juno.seg_bins_nonuniform)
    print("Shape nuisance segments: ", E_segments)

    s_ibd = sp.interpolate.interp1d(
        ensp_nom_juno['sibd'].GetBinCenters(),
        ensp_nom_juno['sibd'].bin_cont,
        kind='slinear', bounds_error=False, fill_value=(0.0, 0.0))

    s_bump_lin = sp.interpolate.interp1d(
        ensp_nom_juno['bump_corr'].GetBinCenters(),
        ensp_nom_juno['bump_corr'].bin_cont,
        kind='slinear', bounds_error=False,
        fill_value=(ensp_nom_juno['bump_corr'].bin_cont[0],
                    ensp_nom_juno['bump_corr'].bin_cont[-1]))

    # ── decide which near detector is active ─────────────────────────────────
    use_tao = args_juno.include_tao and args_juno.fit_type == "PMOP_combined"
    use_dyb = args_juno.include_dyb and args_juno.fit_type == "PMOP_combined" 

    # ── instantiate detectors ────────────────────────────────────────────────
    juno_det = JUNODetector(
        ndays=ndays,
        ensp_nom=ensp_nom_juno,
        rm=rm, cm=cm_juno,
        args=args_juno,
        backgrounds=backgrounds,
        core_baselines=core_baselines,
        core_powers_std=core_powers_std,
        core_fractions=core_fractions,
        E_segments=E_segments,
        s_ibd=s_ibd,
        s_bump_lin=s_bump_lin)

    dyb_det = None
    if use_dyb:
        dyb_det = DYBDetector(
            ensp_nom_dyb=ensp_nom_dyb,
            args_dyb=args_dyb,
            args_juno=args_juno,
            cm_dyb=cm_dyb,
            rm_dyb=rm_dyb,
            dyb_model=dyb_model)

    tao_det = None
    if use_tao:
        tao_det = TAODetector(
            ndays=ndays_tao,
            ensp_nom=ensp_nom_tao,
            rm=rm_tao, cm=cm_tao,
            args=args_tao,
            core_baselines=args_tao.core_baselines,
            core_powers_std=args_tao.core_powers,
            s_ibd=s_ibd, ene_leak_tao=ene_leak_tao, 
            E_segments=E_segments)

    global DYB_BINS
    if dyb_det is not None:
        DYB_BINS = dyb_det.n_dyb_bins
    elif tao_det is not None:
        DYB_BINS = len(ensp_nom_tao['rdet'].bins) - 1

    # ── print fit configuration ───────────────────────────────────────────────
    if not args_juno.silent_print:
        print("************Fit configuration************")
        print(f"Fit type: {args_juno.fit_type}")
        if args_juno.fit_type == "geo":
            print(f"Th/U free: {args_juno.geo_fit_type == 'UThfree'}")
            print(f"Mantle fit: {args_juno.geo_fit_type == 'mantle'}")
            print(f"OPfixed: {args_juno.geo_OPfixed}")
        if args_juno.fit_type == "PMOP_combined":
            print(f"Near detector: {'TAO' if use_tao else 'DYB' if use_dyb else 'none'}")

    unc_new_juno = unc_juno if unc_juno == 'stat' else 'stat+' + unc_juno
    if not args_juno.silent_print:
        print("Uncertainty JUNO: ", unc_new_juno)

    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=bool(args_juno.NMO_opt))
    fit_mode = getattr(args_juno, 'fit_mode', 'combined')  # 'combined' | 'prefit_wsegs' | 'prefit_wosegs'
    
 
    # ════════════════════════════════════════════════════════════════════════
    #  Parameter bookkeeping
    # ════════════════════════════════════════════════════════════════════════

    # ── JUNO-specific ────────────────────────────────────────────────────────

    juno_param_names = juno_det.get_param_names()
    juno_start_values = juno_det.get_start_values()
    n_juno = len(juno_param_names)
    n_nuis = len(E_segments)
    
    bckgs = {}
    for key, info in backgrounds["JUNO"].items():
        bckgs[key] = {"pull": 0.0, "sigma": info["rate_unc"],
                      "b2b": info["b2b_unc"], "name": info["name"]}
    bckgs_keys = list(bckgs.keys())

    n_NL    = sum(1 for p in juno_param_names if p.startswith("NL"))
    n_abc   = sum(1 for p in juno_param_names if p.startswith("abc"))
    n_bckgs = sum(1 for p in juno_param_names if p.startswith("bkg_"))
    n_rea   = sum(1 for p in juno_param_names if p.startswith("snf_noneq_norm"))
    n_core  = sum(1 for p in juno_param_names if p.startswith("core_power"))
    n_efis  = sum(1 for p in juno_param_names if p.startswith("efission"))
    
    # ── TAO-specific (only populated when use_tao) ────────────────────────────
    if use_tao:
        tao_param_names  = tao_det.get_param_names()
        tao_start_values = tao_det.get_start_values()
        n_tao            = len(tao_param_names)
    else:
        tao_param_names  = []
        tao_start_values = []
        n_tao            = 0

    # ── DYB-specific (only Nrea_dyb when use_dyb) ────────────────────────────
    if use_dyb:
        dyb_param_names  = dyb_det.get_param_names()
        dyb_start_values = dyb_det.get_start_values()
        n_dyb            = len(dyb_param_names)
    else:
        dyb_param_names  = []
        dyb_start_values = []
        n_dyb            = 0

    # ════════════════════════════════════════════════════════════════════════
    #  Build flat Minuit parameter vector
    #  Layout: [shared(5)] [juno] [tao or dyb]
    # ════════════════════════════════════════════════════════════════════════

    start_values = (
        get_shared_start_values(args_juno)          # sin2_12 sin2_13 dm2_21 dm2_31 me_rho
        + juno_start_values
        + get_shape_nuis_start_values(n_nuis)
        + tao_start_values
        + dyb_start_values
    )

    param_names = (
        SHARED_PARAM_NAMES                           # sin2_th12 sin2_th13 dm2_21 dm2_31 me_rho
        + juno_param_names
        + get_shape_nuis_names(n_nuis)
        + tao_param_names
        + dyb_param_names
    )

    # ── offsets for wrapper slicing ───────────────────────────────────────────
    # shared block is 5 params; Nrea is index 5; the rest follow
    # The wrapper receives params[5:] as *all_nuisance_params after the 5 shared params
    # so offsets below are relative to position 0 of all_nuisance_params

    # ════════════════════════════════════════════════════════════════════════
    #  Inner chi2  (delegates to detector objects)
    # ════════════════════════════════════════════════════════════════════════
    nuis_best    = np.zeros(n_nuis)
    nuis_cov_inv = None
    C_reco       = None
    
    if fit_mode in ('prefit_wsegs', 'prefit_wosegs'):
        nuis_best, nuis_cov = run_dyb_tao_prefit(
            dyb_det=dyb_det, tao_det=tao_det,
            E_segments=E_segments,
            unc_dyb=unc_dyb, unc_tao=unc_tao,
            args_juno=args_juno, args_tao=args_tao)
    
        # seed start values from pre-fit
        for i, n in enumerate(get_shape_nuis_names(n_nuis)):
            start_values[param_names.index(n)] = nuis_best[i]
    
        if fit_mode == 'prefit_wsegs':
            nuis_cov_inv = np.linalg.inv(nuis_cov)
    
        elif fit_mode == 'prefit_wosegs':
            # build fixed_juno_params dict for Jacobian evaluation
            jp_fixed = dict(
                Nrea=0.0,
                NL=[args_juno.kB, args_juno.fC, args_juno.kI, args_juno.LY_shared] if args_juno.NL != "pull"
                    else [0.0] * n_NL,
                abc=[args_juno.a, args_juno.b, args_juno.c],
                bckgs={k: {**v, "pull": 0.0} for k, v in bckgs.items()},
                snf_noneq_norm=[0.0] * n_rea,
                eff_nuis=0.0,
                ThU_ratio=args_juno.Th_U_ratio,
                core_powers_delta=[0.0] * n_core,
                efission=list(args_juno.efission))   
            C_reco = propagate_nuis_to_reco(
                juno_det=juno_det,
                nuis_best=nuis_best, nuis_cov=nuis_cov,
                sin2_12=nuosc.op_nom["sin2_th12"],
                sin2_13=nuosc.op_nom["sin2_th13"],
                dm2_21=nuosc.op_nom["dm2_21"],
                dm2_31=nuosc.op_nom["dm2_31"],
                fixed_juno_params=jp_fixed,
                args_juno=args_juno, unc_juno=unc_juno)
            # pass C_reco into juno_det so CNP chi2 picks it up
            juno_det.set_external_cov(C_reco)
            plot_correlation_pair(nuis_cov, C_reco, ensp_nom_juno, n_nuis,
                      save_path=f"cov_propagation.png")
    def chi2(sin2_12, sin2_13, dm2_21, dm2_31, me_rho,
             Nrea=0.0,
             NL=None, abc=None, bckgs=None,
             snf_noneq_norm=None, eff_nuis=0.0,
             core_powers_delta=[], efission=[],
             Ngeo=0.0, NU=0.0, NTh=0.0, Nmantle=0.0,
             opp=False, plot_fit=False,
             ThU_ratio=args_juno.Th_U_ratio,nuisance=None, 
             tao_params=None, dyb_params=None):

        chi2_dyb        = 0.0
        chi2_tao        = 0.0
        chi2_penalty_er = 0.0

        dm2_31_phys = dm2_31 * 1e-3
        dm2_21_phys = dm2_21 * 1e-5

        
        # ── JUNO ─────────────────────────────────────────────────────────────
        (chi2_juno, chi2_nolog, chi2_penalty_er,
         s_juno, s_tot_juno,
         fit_bckgs_spectra, rfis_param_post) = juno_det.chi2_contribution(
            sin2_12=sin2_12, sin2_13=sin2_13,
            dm2_21=dm2_21_phys, dm2_31=dm2_31_phys, me_rho=me_rho,
            Nrea=Nrea, NL=NL, abc=abc,
            bckgs_for_chi2=bckgs,
            snf_noneq_norm=snf_noneq_norm,
            eff_nuis=eff_nuis, 
            ThU_ratio=ThU_ratio,
            core_powers_delta=core_powers_delta,
            efission=efission,
            nuisance=nuisance,
            unc=unc_juno, opp=opp)

        sp_obs_tot = juno_det._get_obs_spectrum()


        # ── near detector ─────────────────────────────────────────────────────
        if args_juno.fit_type == "PMOP_combined":
            if use_tao:
                (chi2_tao, _, s_tao, s_tot_tao) = tao_det.chi2_contribution(
                    sin2_12=sin2_12, sin2_13=sin2_13,
                    dm2_21=dm2_21_phys, dm2_31=dm2_31_phys,
                    me_rho=me_rho,
                    Nrea_tao=tao_params['Nrea_tao'],
                    NL_tao=[tao_params[f'NL_tao{i}'] for i in range(5)],
                    snf_noneq_norm_tao=[tao_params[f'snf_noneq_norm_tao{i}'] for i in range(2)],
                    abc_tao=[tao_params['abc_tao0'],
                              tao_params['abc_tao1'],
                              tao_params['abc_tao2']],
                    core_power_tao=tao_params['core_power_tao'],
                    eff_nuis_tao=tao_params['eff_nuis_tao'],
                    unc=unc_tao,nuisance=nuisance, opp=opp)

            if use_dyb:
                nuisance_coarse = np.array(nuisance)
                chi2_dyb, s_dyb, s_tot_dyb = dyb_det.chi2_contribution(
                    nuisance_coarse=nuisance_coarse,
                    snf_noneq_norm_dyb=dyb_params['snf_noneq_norm_dyb'],
                    E_segments=E_segments)

        # ── plot ──────────────────────────────────────────────────────────────
        if plot_fit:
            global FULL_CHI2
            FULL_CHI2 = chi2_nolog + chi2_dyb + chi2_tao + chi2_penalty_er
            plot_fit_residuals(s_tot_juno, sp_obs_tot, s_juno,
                               fit_bckgs_spectra, ndays, args_juno,
                               write_output=args_juno.write_output)
            ensp_nom_juno['ribd'].Plot(
                f"{args_juno.plots_folder}/rfis_posfit.png",
                xlabel="Antineutrino energy (MeV)",
                ylabel="Events/ 23 keV",
                extra_spectra=[rfis_param_post],
                leg_labels=['pre-fit', 'post-fit'],
                colors=['steelblue', 'brown'],
                xmin=0.7, xmax=12.0, ymin=1e-3,
                log_scale=False,
                title=f"CPC 2021, {len(E_segments)}")
            # ensp_nom_juno['ribd'].WritetoROOT("JUNO_unosc_prefit", "Postfit_Apr13.root")
            # rfis_param_post.WritetoROOT("JUNO_unosc_postfit", "Postfit_Apr13.root")
            # ensp_nom_tao['rtot'].WritetoROOT("TAO_prefit", "Postfit_Apr13.root")
            #s_tot_tao.WritetoROOT("TAO_postfit", "Postfit_Apr16.root")
            # ensp_nom_dyb['rdet'].WritetoROOT("DYB_prefit", "Postfit_Apr13.root")
            # s_tot_dyb.WritetoROOT("DYB_postfit", "Postfit_Apr13.root")

        chi2_nuis_penalty = 0.0
        if nuis_cov_inv is not None:
            delta = np.array(nuisance) - nuis_best
            chi2_nuis_penalty = float(delta @ nuis_cov_inv @ delta)


        
        return chi2_juno + chi2_dyb + chi2_tao + chi2_penalty_er +chi2_nuis_penalty

    # ════════════════════════════════════════════════════════════════════════
    #  chi2 wrapper  — slices the flat Minuit parameter vector
    #
    #  params[:5]  = sin2_12, sin2_13, dm2_21, dm2_31, me_rho  (shared)
    #  params[5:]  = JUNO nuisances + (TAO or DYB nuisances)
    # ════════════════════════════════════════════════════════════════════════

    def chi2_wrapper(sin2_12, sin2_13, dm2_21, dm2_31, me_rho,
                     *all_nuisance_params, opp=False, plot_fit=False):
        offset = 0
 
        # ── JUNO block: slice n_juno values, map by name ──────────────────────
        juno_vals = all_nuisance_params[offset : offset + n_juno]; offset += n_juno
        jp = dict(zip(juno_param_names, juno_vals))
 
        # ── shape nuisances ───────────────────────────────────────────────────
        nuisance_vals = all_nuisance_params[offset : offset + n_nuis]; offset += n_nuis
 
        # ── TAO block ─────────────────────────────────────────────────────────
        tao_params_dict = None
        if use_tao:
            tao_vals = all_nuisance_params[offset : offset + n_tao]; offset += n_tao
            tao_params_dict = dict(zip(tao_param_names, tao_vals))
 
        # ── DYB block ─────────────────────────────────────────────────────────
        dyb_params_dict = None
        if use_dyb:
            dyb_vals = all_nuisance_params[offset : offset + n_dyb]; offset += n_dyb
            dyb_params_dict = dict(zip(dyb_param_names, dyb_vals))
 
        # ── unpack JUNO params by name ────────────────────────────────────────
        NL_vals   = [jp[p] for p in juno_param_names if p.startswith("NL")]
        abc_vals  = [jp[p] for p in juno_param_names if p.startswith("abc")]
        rea_vals  = [jp[p] for p in juno_param_names if p.startswith("snf_noneq_norm")]
        core_vals = [jp[p] for p in juno_param_names if p.startswith("core_power")]
        efis_vals = [jp[p] for p in juno_param_names if p.startswith("efission")]
 
        bckgs_for_chi2 = {}
        for key in bckgs_keys:
            info = dict(bckgs[key])
            info["pull"] = jp[f"bkg_{key}"]
            bckgs_for_chi2[key] = info
 
        return chi2(
            sin2_12=sin2_12, sin2_13=sin2_13,
            dm2_21=dm2_21, dm2_31=dm2_31, me_rho=me_rho,
            Nrea=jp["Nrea"],
            NL=NL_vals, abc=abc_vals,
            bckgs=bckgs_for_chi2,
            snf_noneq_norm=rea_vals,
            eff_nuis=jp["eff_nuis"],
            ThU_ratio=jp["ThU_ratio"],
            core_powers_delta=core_vals,
            efission=efis_vals,
            nuisance=nuisance_vals,
            tao_params=tao_params_dict,
            dyb_params=dyb_params_dict,
            opp=opp, plot_fit=plot_fit)
    # ════════════════════════════════════════════════════════════════════════
    #  Minuit object
    # ════════════════════════════════════════════════════════════════════════

    def make_chi2_func(n_fixed=5, n_total=None, **extra_kwargs):
        def f(*params):
            return chi2_wrapper(*params[:n_fixed], *params[n_fixed:], **extra_kwargs)
        f.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(f"p{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for i in range(n_total)
            ])
        return f

    chi2_func = make_chi2_func(n_fixed=5, n_total=len(start_values))
    m = Minuit(chi2_func, *start_values, name=param_names)

    # ── limits ───────────────────────────────────────────────────────────────
    # shared
    apply_shared_limits(m, args_juno, n_shape_bins=n_nuis)

    # JUNO
    juno_det.apply_limits(m)

    # TAO
    if use_tao:
        tao_det.apply_limits(m)

    # DYB
    if use_dyb:
        dyb_det.apply_limits(m)

    if not args_juno.silent_print:
        print("-----Parameter init values and limits-----")
        for p in m.parameters:
            print(p, m.values[p], m.limits[p])

  #  m.fixed['nuis20'] = True
    if fit_mode == 'prefit_wosegs':
        for i in range(n_nuis):
            m.fixed[f'nuis{i}'] = True
        # near-detector params are irrelevant now — fix them too
        for n in tao_param_names + dyb_param_names:
            m.fixed[n] = True
    elif fit_mode == 'combined':
        # existing behaviour: fix nuis for non-PMOP
        if args_juno.fit_type != 'PMOP_combined':
            for i in range(n_nuis):
                m.fixed[f'nuis{i}'] = True
    first_fix_juno = juno_det.get_first_fix()

    first_fix_tao = tao_det.get_first_fix() if use_tao else []
    first_fix_dyb = dyb_det.get_first_fix() if use_dyb else []

    first_fix = first_fix_juno + first_fix_tao + first_fix_dyb

    # ── stage 1: fix osc + me_rho + all nuisances ─────────────────────────────
    for p in SHARED_PARAM_NAMES:   # sin2_th12..me_rho
        m.fixed[p] = True
    for p in first_fix:
        m.fixed[p] = True
    m.migrad()

    # ── stage 2: free osc + me_rho ───────────────────────────────────────────
    print("freeing shared params (osc + me_rho)")
    for p in SHARED_PARAM_NAMES:
        m.fixed[p] = False
    m.migrad()
    m.strategy = 2

    # ── stage 3: free everything else ─────────────────────────────────────────
    print("freeing everything else")
    for p in first_fix:
        if not m.fixed[p]:
            continue
        m.fixed[p] = False



    # keep NL and abc fixed in stage 3 (same as original)
    # for i in range(n_NL):
    #     m.fixed[f'NL{i}'] = True
    # for i in range(n_abc):
    #     m.fixed[f'abc{i}'] = True

    m.migrad()
    m.hesse()

    # ── diagnostics ───────────────────────────────────────────────────────────
    if not args_juno.silent_print:
        print("Function minimum value: ", m.fval)
        nbins = len(ensp_nom_juno["rdet"].bins) - 1
        if args_juno.fit_type == "PMOP_combined":
            nbins += DYB_BINS
        global FREE_PARAM
        FREE_PARAM = 3 + n_nuis
        ndf        = nbins - FREE_PARAM
        print("Free parameters", FREE_PARAM)
        print("Nbins",           nbins)
        print("chi2/ndf: ",      m.fval / ndf)
        print("-----Good things, should be True-----")
        print("FMin valid?             ", m.fmin.is_valid)
        print("Has covariance?         ", m.fmin.has_covariance)
        print("Covariance accurate?    ", m.fmin.has_accurate_covar)
        print("Minos valid?            ", m.fmin.is_valid)
        print("-----Bad things, should be False-----")
        print("Above EDM threshold?    ", m.fmin.is_above_max_edm)
        print("Has reached call limit? ", m.fmin.has_reached_call_limit)
        print("Hesse failed?           ", m.fmin.hesse_failed)

    # ── write postfit ROOT file ───────────────────────────────────────────────
    if args_juno.write_output:
        output_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_nuis = output_file.mkdir("nuisances")
        dir_post = output_file.mkdir("postfit")

    if not args_juno.silent_print:
        print("************Other parameter fit results************")
        for i in range(len(SHARED_PARAM_NAMES), len(m.values)):
            print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")

        print("************Background rates************")
        # JUNO backgrounds: offset = n_shared(5) + 1(Nrea) + n_NL + n_abc = 5+1+n_NL+3
        juno_bkg_offset = len(SHARED_PARAM_NAMES) + 1 + n_NL + n_abc
        for j, key in enumerate(bckgs_keys):
            idx      = juno_bkg_offset + j
            nuis     = m.values[idx]
            nuis_err = m.errors[idx]
            if key == "geo":
                start_rate = float(args_juno.geo_tnu_rate) * float(args_juno.tnu_to_cpd)
            else:
                start_rate = backgrounds["JUNO"][key]["rate"]
            rate_unc        = backgrounds["JUNO"][key].get("rate_unc", None)
            rate_fit        = start_rate * (1 + nuis)
            rate_err        = start_rate * nuis_err
            deviation_sigma = ((rate_fit - start_rate) / (start_rate * rate_unc)
                               if rate_unc else float('nan'))
            print(f"{key}: {rate_fit:.4f} ± {rate_err:.4f} cpd "
                  f"(rel unc: {nuis_err:.3%}; deviation: {deviation_sigma:.2f} σ)")
            if args_juno.write_output:
                dir_nuis.cd()
                write_parameter(f"{key}", rate_fit, rate_err)

        print("************Normalization parameters************")
        print(f"Rflux: {1.0 * (1 + m.values['Nrea'])} +/- {1.0 * m.errors['Nrea']}")
        print(f"Eff:   {args_juno.detector_efficiency * (1 + m.values['eff_nuis'])} "
              f"+/- {args_juno.detector_efficiency * m.errors['eff_nuis']}")

        print("************Shared parameters************")
        print(f"me_rho: {m.values['me_rho']} +/- {m.errors['me_rho']}")

        print("************JUNO Detector response************")
        print(f"a:  {m.values['abc0']} +/- {m.errors['abc0']}")
        print(f"b:  {m.values['abc1']} +/- {m.errors['abc1']}")
        print(f"c:  {m.values['abc2']} +/- {m.errors['abc2']}")
        print(f"kB: {m.values['NL0']}  +/- {m.errors['NL0']}")
        print(f"fC: {m.values['NL1']}  +/- {m.errors['NL1']}")
        print(f"kI: {m.values['NL2']}  +/- {m.errors['NL2']}")
        print(f"LY: {m.values['NL3']}  +/- {m.errors['NL3']}")

        if use_tao:
            print("************TAO Detector response************")
            for pname in tao_param_names:
                print(f"{pname}: {m.values[pname]} +/- {m.errors[pname]}")

        if args_juno.write_output:
            dir_nuis.cd()
            write_parameter("Rflux", 1.0 * (1 + m.values['Nrea']), 1.0 * m.errors['Nrea'])
            write_parameter("Eff",
                            args_juno.detector_efficiency * (1 + m.values['eff_nuis']),
                            args_juno.detector_efficiency * m.errors['eff_nuis'])
            for tag, pname in [("a","abc0"),("b","abc1"),("c","abc2"),
                                ("kB","NL0"),("fC","NL1"),("kI","NL2"),("LY","NL3")]:
                write_parameter(tag, m.values[pname], m.errors[pname])
            write_parameter("me_rho", m.values['me_rho'], m.errors['me_rho'])

    if args_juno.write_output:
        output_file.Close()

    # ── correlation plot ──────────────────────────────────────────────────────
    if args_juno.plot_correlation:
        corr = m.covariance.correlation()
        import seaborn as sns
        plt.figure(figsize=(30, 30))
        sns.heatmap(corr, annot=True, fmt=".2f",
                    xticklabels=m.parameters, yticklabels=m.parameters,
                    cmap="coolwarm", vmin=-1, vmax=1, cbar=False)
        plt.tight_layout()
        plt.savefig("Plots/corr.pdf")

    # ── dm2_31 profile ────────────────────────────────────────────────────────
    if args_juno.profile_dm2_31:
        profile_dm2_31(m, args_juno, NMO_opt=bool(args_juno.NMO_opt),
                       write=args_juno.write_output)

    # ── MINOS ────────────────────────────────────────────────────────────────
    merrors = bool(args_juno.calc_minuit_errors)
    if merrors:
        m.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31")

    # ── oscillation results ───────────────────────────────────────────────────
    osc_params_nom = [
        nuosc.op_nom["sin2_th12"],
        nuosc.op_nom["sin2_th13"],
        nuosc.op_nom["dm2_21"] * 1e5,
        nuosc.op_nom["dm2_31"] * 1e3,
    ]

    if args_juno.write_output:
        output_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_osc = output_file.mkdir("oscillation_parameters")
        dir_osc.cd()

    if not args_juno.silent_print and not args_juno.is_data:
        print("************Oscillation parameter fit results************")

    for i in range(4):
        err = (max(abs(m.merrors[i].upper), abs(m.merrors[i].lower))
               if merrors else m.errors[i])
        val    = m.values[i]
        exp_val = osc_params_nom[i]
        rel_unc         = err / val if val != 0 else float('nan')
        deviation_sigma = (val - exp_val) / err if err != 0 else float('nan')
        print(f"{m.parameters[i]}: {val:.6g} +/- {err:.6g}; "
              f"rel unc: {rel_unc:.3%}; deviation: {deviation_sigma:.2f} σ")
        if args_juno.write_output:
            write_osc_params(str(m.parameters[i]), val, err)

    if args_juno.write_output:
        output_file.Close()

    # ── post-fit plot ─────────────────────────────────────────────────────────
    best_params = [m.values[name] for name in m.parameters]
    chi2_NO     = chi2_wrapper(*best_params, plot_fit=True)

    # ── solar contour ─────────────────────────────────────────────────────────
    if args_juno.solar_contour:
        x1, y1 = zip(*contour_1sigma)   # noqa: F821
        x2, y2 = zip(*contour_2sigma)   # noqa: F821
        plt.figure(figsize=(7, 6))
        plt.plot(x1, y1, 'b-',  label='68% CL (1σ)')
        plt.plot(x2, y2, 'r--', label='95% CL (2σ)')
        plt.plot(m.values["sin2_th12"], m.values["dm2_21"], 'ko', label='Best fit')
        plt.xlabel(r'$\sin^2\theta_{12}$')
        plt.ylabel(r'$\Delta m^2_{21}$ [eV$^2$]')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("Plots/solar_contour.png")

    # ── NMO significance ──────────────────────────────────────────────────────
    if args_juno.NMO_sig:
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt,
                                       NO=not bool(args_juno.NMO_opt))
        m.values['dm2_31'] = nuosc.op_nom['dm2_31']
        chi2_func_opp = make_chi2_func(n_fixed=5, n_total=len(start_values), opp=True)
        m_opp = Minuit(chi2_func_opp, *m.values, name=param_names)
        m_opp.migrad()
        print(m_opp)
        if merrors:
            m_opp.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31")
        best_params_IO = [m_opp.values[name] for name in m_opp.parameters]
        chi2_IO = chi2_wrapper(*best_params_IO)
        if args_juno.profile_dm2_31:
            profile_dm2_31(m_opp, args_juno,
                           NMO_opt=not bool(args_juno.NMO_opt), write=False)
        print("NMO significance: ", chi2_NO, chi2_IO)


    # ── profile / matrix plots ────────────────────────────────────────────────
    if args_juno.plot_minuit_matrix or args_juno.plot_minuit_profiles:
        os.makedirs(f"{args_juno.plots_folder}/Chi2_profiles/Minuit", exist_ok=True)

    if args_juno.plot_minuit_profiles:
        print("Plotting chi2 profiles")
        for i, pname in enumerate(m.parameters):
            plotname = (f"{args_juno.plots_folder}/Chi2_profiles/Minuit/"
                        f"chi2_{args_juno.stat_opt}_{pname}_{unc_juno}.png")
            plot_profile(m, i, pname, plotname)

    if args_juno.print_minuit_correlation:
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())

    if args_juno.plot_minuit_matrix:
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1, 2, 3])
        plt.show()

    return m