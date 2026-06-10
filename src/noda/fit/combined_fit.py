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

def _plot_relative_uncertainty(obs_vals, obs_err, bins, detector_label, args,
                               tag=None, output_name=""):
    """
    Standalone √Var/N vs energy plot, one per detector call.
    Uses whatever obs_err was already computed by the caller.
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rel_unc = obs_err / np.maximum(np.abs(obs_vals), 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bin_centers, rel_unc, 'o-', color='black', markersize=3)
    ax.set_xlabel("Reconstructed Energy [MeV]")
    ax.set_ylabel(r"Relative Uncertainty ($\sqrt{\mathrm{Var}}/N$)")
    ax.set_title(f"{detector_label} Relative Uncertainty")
    ax.grid(True, alpha=0.3)

    os.makedirs("Plots", exist_ok=True)
    if output_name:
        figname = f"Plots/{output_name}_relunc.png"
    else:
        suffix  = f"_{tag}" if tag else ""
        figname = (f"Plots/rel_unc_{args.file_prefix}"
                   f"_{detector_label}{suffix}.png")
    plt.tight_layout()
    plt.savefig(figname, dpi=200)
    plt.close(fig)
    print(f"[plot] saved {figname}")

def plot_fit_residuals_all(s_tot, s_obs, s_signal, bckgs, ndays, args,
                       *,
                       detector_label="JUNO",
                       signal_label=None,
                       tag=None,
                       chi2_value=None,
                       ndf=None,
                       write_output=False, output_name="", cm=None, eff=None):
    """
    Plot observed vs fitted spectra with residuals and chi²/ndf.

    detector_label : "JUNO" | "TAO" | "DYB" | other — used in legend/title
    signal_label   : label for the s_signal curve (default depends on detector)
    tag            : unique string appended to filename (e.g. "phase1_ALL");
                     when None, falls back to args.file_prefix alone
    chi2_value     : chi² shown in the textbox; falls back to FULL_CHI2 global
    ndf            : ndf used in chi²/ndf; falls back to (n_bins - FREE_PARAM)
    """
    if chi2_value is None:
        chi2_value = FULL_CHI2
    default_sig_label = {
        "JUNO": r"Reactor $\bar{\nu}_e$ expectation",
        "TAO":  r"TAO reactor $\bar{\nu}_e$",
        "DYB":  r"Daya Bay reactor $\bar{\nu}_e$",
    }
    if signal_label is None:
        signal_label = default_sig_label.get(detector_label, f"{detector_label} signal")

    if write_output:
        output_file = (f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}"
                           f"_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root")
        if detector_label == "JUNO":
            s_signal.WritetoROOT(f"reactor",
                             f"{output_file}",
                             directory="juno/postfit")

        s_tot.WritetoROOT(f"totalpred",
                             f"{output_file}",
                             directory=f"{detector_label.lower()}/postfit")

    os.makedirs("Plots", exist_ok=True)
    bin_centers  = s_tot.bins
    bin_centers2 = 0.5 * (s_tot.bins[:-1] + s_tot.bins[1:])

    obs_vals = s_obs.bin_cont
    obs_err = np.sqrt(np.diag(cm.data))

    if detector_label != "DYB":
        obs_err += np.sqrt(np.maximum(obs_vals, 1e-12))

    fit_vals = s_tot.bin_cont
    fit_err  = np.sqrt(np.maximum(fit_vals, 1e-12))

    residuals = (obs_vals - fit_vals) / np.maximum(obs_err, 1e-12)

    chi2_bin  = (obs_vals - fit_vals) ** 2 / np.maximum(obs_err, 1e-12) ** 2

    total_chi2 = float(np.sum(chi2_bin))
    if ndf is None:
        ndf = max(1, len(chi2_bin) - FREE_PARAM)
    chi2_ndf = float(chi2_value) / ndf

    # cumulative chi²/ndf, high-energy → low-energy
    chi2_bin_rev          = chi2_bin[::-1]
    cum_chi2_rev          = np.cumsum(chi2_bin_rev)
    ndf_bins              = np.arange(1, len(cum_chi2_rev) + 1)
    cum_chi2_ndf          = (cum_chi2_rev / ndf_bins)[::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1, 1]})


    # ── top panel: data + fit + signal + backgrounds ────────────────────
    ax1.errorbar(bin_centers2, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label=f'{detector_label} data')
    ax1.step(bin_centers, np.append(fit_vals, fit_vals[-1]),
             where='post', color='blue', lw=1.5, label='Total Fit')
    if s_signal is not None:
        sig_cpd = s_signal.GetIntegral() / ndays if ndays else 0.0
        ax1.step(bin_centers, np.append(s_signal.bin_cont, s_signal.bin_cont[-1]),
                 where='post', color='red', lw=1.2, linestyle='-',
                 label=f'{signal_label}: {sig_cpd:.2f} cpd')

    colors = plt.cm.tab10.colors
    name_to_key = {
        "Accidentals": "acc", "Geoneutrinos": "geo",
        r"$^{9}$Li / $^{8}$He": "lihe", "Fast neutrons": "fastn",
        "Double neutrons": "doublen",
        r"$^{13}$C ($\alpha$, n) $^{16}$O": "alphan",
        "Atmospheric neutrinos": "atm", "World reactor neutrinos": "wr",
        r"$^{214}$Bi": "bipo",
    }
    if bckgs:
        if detector_label == "JUNO":
            for i, (key, spec) in enumerate(bckgs.items()):
                cpd = spec.GetIntegral() / ndays if ndays else 0.0
                ax1.step(bin_centers, np.append(spec.bin_cont, spec.bin_cont[-1]),
                         where='post', color=colors[i % len(colors)], lw=1,
                         label=f'{key}: {cpd:.2f} cpd')
                if write_output and name_to_key.get(key):
                    spec.WritetoROOT(f"{name_to_key.get(key)}",
                                     f"{output_file}", "juno/postfit")
        if detector_label == "TAO":
                cpd = bckgs.GetIntegral() / ndays if ndays else 0.0
                ax1.step(bin_centers, np.append(bckgs.bin_cont, bckgs.bin_cont[-1]),
                         where='post', color=colors[0], lw=1,
                         label=f'bckg: {cpd:.2f} cpd')
    ax1.set_ylabel("Events")
    title_tag = f" [{tag}]" if tag else ""
    ax1.set_title(f"{detector_label}{title_tag}", fontsize=10)
    leg = ax1.legend(fontsize=9, loc="upper right", frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bin_centers[0], bin_centers[-1])

    # ── residuals panel ─────────────────────────────────────────────────
    ax2.errorbar(bin_centers2, residuals, yerr=np.ones_like(residuals),
                 fmt='o', color='blue', markersize=3)
    ax2.step(bin_centers, np.append(residuals, residuals[-1]), where='post', color='blue', lw=1)
    ax2.axhline(0, color='k', lw=1)
    for nsigma, alpha in zip([1, 2, 3], [0.25, 0.15, 0.1]):
       ax2.fill_between(bin_centers, -nsigma, nsigma, color='blue', alpha=alpha)
    ax2.set_ylabel("Residuals [σ]")
    ax2.set_ylim(-5.0, 5.0)
    ax2.grid(True, alpha=0.3)

    # ── chi² panel ──────────────────────────────────────────────────────
    ax3.step(bin_centers, np.append(chi2_bin, chi2_bin[-1]),
             where='post', color='steelblue', lw=1.5, label='Local χ²')
    ax3.axhline(1.0, color='gray', ls='--', lw=1)
    ax3.set_ylabel("Local χ²", color='steelblue')
    ax3.tick_params(axis='y', colors='steelblue')
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.step(bin_centers, np.append(cum_chi2_ndf, cum_chi2_ndf[-1]),
              where='post', color='brown', lw=1.5,
              label='Cumulative χ²/ndf (high→low)')
    ax3b.axhline(chi2_ndf, color='darkred', ls='--', lw=1)
    ax3b.set_ylabel("Cumulative χ²/ndf", color='brown')
    ax3b.tick_params(axis='y', colors='brown')
    ax3.set_ylim(0, 5)
    ax3b.set_ylim(0, 5)

    # ── toy goodness-of-fit ─────────────────────────────────────────────
    n_toys = 1000
    rng = np.random.default_rng(12345)
    toy_data  = rng.poisson(np.maximum(fit_vals, 1e-12), size=(n_toys, len(fit_vals)))
    chi2_toys = np.sum((toy_data - fit_vals[None, :]) ** 2
                       / np.maximum(obs_err, 1e-12) ** 2, axis=1)
    p_val = float(np.sum(chi2_toys > total_chi2)) / n_toys
    print(f"\n--- {detector_label}{title_tag} goodness-of-fit ---")
    print(f"Observed χ² = {total_chi2:.2f} for {ndf} dof → χ²/ndf = {chi2_ndf:.3f}")
    print(f"Toy mean χ² = {chi2_toys.mean():.2f} ± {chi2_toys.std():.2f}")
    print(f"p-value = {p_val:.3f}")

    if detector_label == "JUNO":
        textstr = (f"{detector_label}{title_tag}\n"
               f"Livetime = {ndays:.2f} days\n"
               f"χ²/ndf = {chi2_value:.2f}/{ndf} = {chi2_ndf:.2f}\n"
               f"Efficiency = {eff*100.:.2f}")
    else:
        textstr = (f"{detector_label}{title_tag}\n"
               f"Livetime = {ndays:.2f} days\n"
               f"χ²/ndf = {chi2_value:.2f}/{ndf} = {chi2_ndf:.2f}")

    ax1.text(0.2, 0.9, textstr, transform=ax1.transAxes, fontsize=9,
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_xlabel("Reconstructed Energy [MeV]")
    plt.tight_layout()

    # ── unique filename ─────────────────────────────────────────────────
    if output_name:
        figname = f"Plots/{output_name}.png"
    else:
        suffix = f"_{tag}" if tag else ""
        figname = f"Plots/fit_residuals_{args.file_prefix}_{detector_label}{suffix}.png"
    plt.savefig(figname, dpi=200)
    plt.close(fig)
    print(f"[plot] saved {figname}")

    _plot_relative_uncertainty(
        obs_vals=obs_vals,
        obs_err=obs_err,          # already correct for every detector
        bins=s_tot.bins,
        detector_label=detector_label,
        args=args,
        tag=tag,
        output_name=output_name,
    )

def run_dyb_tao_prefit(dyb_det, tao_det, E_segments, unc_dyb, unc_tao,
                       args_juno, args_dyb, args_tao, ndays_dyb, ndays_tao, dyb_exp, tao_exp, ibd_prefit):
    """
    Fit DYB+TAO with oscillation params fixed to nominal.
    Returns: (nuis_best, nuis_cov)
    """
    n_nuis = len(E_segments) -1
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
            (chi2_tao, _, _s, _st, _, _) = tao_det.chi2_contribution(
                Nrea_tao=tao_params_dict['Nrea_tao'],
                NL_tao=[tao_params_dict[f'NL_tao{i}'] for i in range(5)],
                snf_noneq_norm_tao=[tao_params_dict[f'snf_noneq_norm_tao{i}']
                                    for i in range(2)],
                abc_tao=[tao_params_dict['abc_tao0'],
                         tao_params_dict['abc_tao1'],
                         tao_params_dict['abc_tao2']],
                core_power_tao=tao_params_dict['core_power_tao'],
                eff_nuis_tao=tao_params_dict['eff_nuis_tao'],
                bckg_rate=tao_params_dict['bckg_rate'],
                unc=unc_tao, nuisance=nuisance_vals)


        if dyb_det is not None:
            noneq=0.0
            if len(dyb_vals):
                noneq = dyb_params_dict['snf_noneq_norm_dyb']
            chi2_dyb, s_obs_dyb, s_tot_dyb, _, _ = dyb_det.chi2_contribution(
                nuisance_coarse=np.array(nuisance_vals),
                snf_noneq_norm_dyb=noneq,
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

        # Stage A: shape nuisances fixed at 0, fit TAO + DYB only
    for n in nuis_names:
        m_pre.fixed[n] = True
    m_pre.migrad()

    # Stage B: free shape nuisances, refit
    for n in nuis_names:
        m_pre.fixed[n] = False
    m_pre.migrad()
    m_pre.hesse()


    nuis_best = np.array([m_pre.values[n] for n in nuis_names])
    idx       = [list(m_pre.parameters).index(n) for n in nuis_names]
    nuis_cov  = np.array(m_pre.covariance)[np.ix_(idx, idx)]
    if not args_juno.silent_print:
        print("DYB+TAO pre-fit complete.")
        print("  nuis_best:", nuis_best)
        print("  nuis_cov diag:", np.diag(nuis_cov))

    nbins  =0
    if dyb_det is not None:
        noneq=0.0
        if len(dyb_start_values):
            noneq = m_pre.values['snf_noneq_norm_dyb']
        chi2_dyb, s_obs_dyb, s_tot_dyb, cm_dyb, ibd_postfit = dyb_det.chi2_contribution(
            nuisance_coarse=nuis_best,
            snf_noneq_norm_dyb=noneq,
            E_segments=E_segments)
        nbins += len(s_obs_dyb.bins)

    if tao_det is not None:
        (chi2_tao, _, s_obs_tao, s_tot_tao, bckg, cm_tao) = tao_det.chi2_contribution(
            Nrea_tao=m_pre.values['Nrea_tao'],
            NL_tao=[m_pre.values[f'NL_tao{i}'] for i in range(5)],
            snf_noneq_norm_tao=[m_pre.values[f'snf_noneq_norm_tao{i}']
                                for i in range(2)],
            abc_tao=[m_pre.values['abc_tao0'],
                     m_pre.values['abc_tao1'],
                     m_pre.values['abc_tao2']],
            core_power_tao=m_pre.values['core_power_tao'],
            eff_nuis_tao=m_pre.values['eff_nuis_tao'],
            bckg_rate=m_pre.values['bckg_rate'],
            unc=unc_tao, nuisance=nuis_best)
        nbins += len(s_obs_tao.bins)

    if dyb_det is not None:
        plot_fit_residuals_all(
                s_tot=s_tot_dyb,
                s_obs=s_obs_dyb,
                s_signal=dyb_exp,
                bckgs=None,
                ndays=ndays_dyb,
                args=args_dyb,
                detector_label="DYB",
                tag="best-fit",
                chi2_value=m_pre.fval,
                cm=cm_dyb,
                ndf=nbins -len(E_segments),
                write_output=args_dyb.write_output
            )

        ibd_prefit.Plot(
                f"{args_juno.plots_folder}/rfis_posfit.png",
                xlabel="Antineutrino energy (MeV)",
                ylabel="Events",
                extra_spectra=[ibd_postfit],
                leg_labels=['pre-fit', 'post-fit'],
                colors=['steelblue', 'brown'],
                xmin=0.7, xmax=12.0, ymin=1e-3,
                log_scale=False,
                title=f"PRL 2025, {len(E_segments)} segments")
    if tao_det is not None:
        plot_fit_residuals_all(
                s_tot=s_tot_tao,
                s_obs=s_obs_tao,
                s_signal=tao_exp,
                bckgs=bckg,
                ndays=ndays_tao,
                args=args_tao,
                detector_label="TAO",
                tag="best-fit",
                chi2_value=chi2_tao,
                cm=cm_tao,
                ndf=nbins  - len(E_segments),
                write_output=args_tao.write_output
            )



    return m_pre, nuis_best, nuis_cov

def propagate_nuis_to_reco(juno_det, nuis_best, nuis_cov,
                            sin2_12, sin2_13, dm2_21, dm2_31, me_rho,
                            fixed_juno_params, args_juno, unc_juno,
                            eps=1e-4):
    n_nuis = len(nuis_best)

    def spectrum_at(nuis):
        (_, s_tot, _, _, _, _, _, _, _) = juno_det.build_spectrum(
            sin2_12=sin2_12, sin2_13=sin2_13,
            dm2_21=dm2_21, dm2_31=dm2_31, me_rho=me_rho,
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

    LOWER = -1.0 + 1e-3                    # stay safely above the log(0) wall

    for i in range(n_nuis):
        # Adaptive step from this knot's posterior σ (with a floor)
        sigma_i = float(np.sqrt(max(nuis_cov[i, i], 1e-16)))
        h = max(eps, 0.1 * sigma_i)

        # Choose stencil based on how close we are to the wall
        if nuis_best[i] - h > LOWER:
            # central
            up = nuis_best.copy(); up[i] += h
            dn = nuis_best.copy(); dn[i] -= h
            col = (spectrum_at(up) - spectrum_at(dn)) / (2 * h)
        elif nuis_best[i] + h <= 5.0:
            # forward only (knot is pinned at / very near −1)
            up = nuis_best.copy(); up[i] += h
            col = (spectrum_at(up) - S0) / h
        else:
            col = np.zeros(n_reco)         # extreme case — shouldn't happen

        # Guard against any residual NaN/Inf
        if not np.all(np.isfinite(col)):
            print(f"[propagate] non-finite Jacobian column {i} "
                  f"(nuis_best[{i}]={nuis_best[i]:.4f}); zeroing.")
            col = np.zeros(n_reco)

        J[:, i] = col

    return J @ nuis_cov @ J.T

def run_minuit(ndays, ndays_dyb=0, ndays_tao=0, s_ibd=None, s_bump_lin=None,
               ensp_nom_juno={}, ensp_nom_dyb={}, ensp_nom_tao={},
               unc_juno='', unc_dyb='', unc_tao='', unc_corr='',
               rm=[], rm_dyb=[], rm_tao=[],
               cm_juno={}, cm_dyb={}, cm_tao={}, cm_corr='',
               args_juno='', args_dyb='', args_tao='',
               dyb_model=None, backgrounds=None,
               ene_leak_tao=None,
               toy=False, fix_param=None,
               # ── FC additions ──────────────────────────────────────────────
               prefit_cache=None,        # dict {'nuis_best','nuis_cov','C_reco'}
               prefit_only=False,        # return after prefit, skip JUNO fit
               return_prefit=False,      # return (m, prefit_dict) instead of m
               return_cost_fn=False,
               fix_dm2_31_to=None,# in 1e-3 eV² (Minuit convention)
               toy_dm2_31=None,
               true_dm2_31=None,
               toy_quiet=False):         # silence ALL post-fit output
    full_fit_start = datetime.now()
    """
    Combined JUNO + DYB/TAO fit.

    - Set ensp_nom_tao / args_tao / rm_tao / cm_tao to use TAO.
    - Set ensp_nom_dyb / args_dyb / rm_dyb / cm_dyb / dyb_model to use DYB.
    - args_juno.fit_type == "PMOP_combined" activates the near-detector term.
    - Which near detector is active is determined by bool(ensp_nom_tao).
    """

    # ── FC: silence everything post-fit when running toys ────────────────────
    if toy_quiet:
        args_juno.silent_print             = True
        args_juno.write_output             = False
        args_juno.run_hesse                = False
        args_juno.plot_correlation         = False
        args_juno.profile_dm2_31           = False
        args_juno.calc_minuit_errors       = False
        args_juno.NMO_sig                  = False
        args_juno.solar_contour            = False
        args_juno.plot_minuit_profiles     = False
        args_juno.plot_minuit_matrix       = False
        args_juno.print_minuit_correlation = False

    # ── load reactor summary ─────────────────────────────────────────────────
    core_names, core_powers_std, core_fractions, core_baselines = \
        load_reactor_summary(args_juno.reactor_summary_yaml)

    E_segments = np.array(args_juno.seg_bins_nonuniform)
    if not args_juno.silent_print: print("Shape nuisance segments: ", E_segments)


    # ── decide which near detector is active ─────────────────────────────────
    use_tao = args_juno.include_tao and args_juno.fit_type == "PMOP_combined"
    use_dyb = args_juno.include_dyb and args_juno.fit_type == "PMOP_combined"

    if not args_juno.silent_print:
        if use_tao: print("Using TAO")
        if use_dyb: print("Using DYB")
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
            s_ibd=s_ibd,
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
    n_nuis = len(E_segments) -1

    bckgs = {}
    for key, info in backgrounds["JUNO"].items():
        bckgs[key] = {"pull": 0.0, "sigma": info["rate_unc"],
                      "b2b": info["b2b_unc"], "name": info["name"], "b2b_scale": info["b2b_scale"]}
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
        get_shared_start_values(args_juno, dm2_31=true_dm2_31)          # sin2_12 sin2_13 dm2_21 dm2_31 me_rho
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
    nuis_cov     = None
    nuis_cov_inv = None
    C_reco       = None

    if args_juno.fit_type == "PMOP_combined" and fit_mode.startswith('prefit'):
        if prefit_cache is None:
            m_pre, nuis_best, nuis_cov = run_dyb_tao_prefit(
                dyb_det=dyb_det, tao_det=tao_det,
                E_segments=E_segments,
                unc_dyb=unc_dyb, unc_tao=unc_tao,
                args_juno=args_juno, args_tao=args_tao)
        else:
            nuis_best = np.asarray(prefit_cache['nuis_best'])
            nuis_cov  = np.asarray(prefit_cache['nuis_cov'])
            m_pre     = None
            if not args_juno.silent_print:
                print("Using CACHED DYB+TAO prefit.")

        # seed start values from pre-fit
        for i, n in enumerate(get_shape_nuis_names(n_nuis)):
            start_values[param_names.index(n)] = nuis_best[i]

        if fit_mode == 'prefit_wsegs':
            nuis_cov_inv = np.linalg.inv(nuis_cov)

        elif fit_mode == 'prefit_wosegs':
            if prefit_cache is not None and prefit_cache.get('C_reco') is not None:
                C_reco = np.asarray(prefit_cache['C_reco'])
            else:
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
                    me_rho = args_juno.me_rho,
                    fixed_juno_params=jp_fixed,
                    args_juno=args_juno, unc_juno=unc_juno)
            # pass C_reco into juno_det so CNP chi2 picks it up
            juno_det.set_external_cov(C_reco)
            # plot_correlation_pair(nuis_cov, C_reco, ensp_nom_juno, n_nuis,
            #           save_path=f"cov_propagation.png")

        # ── FC: persist freshly-computed prefit if config asks for it ────────

        if args_juno.save_prefit and prefit_cache is None:
            os.makedirs("prefit_data", exist_ok=True)
            np.savez_compressed(
                f"{args_juno.prefit_path}",
                nuis_best=nuis_best,
                nuis_cov=nuis_cov,
                C_reco=C_reco if C_reco is not None else np.array([]))
            if not args_juno.silent_print:
                print(f"[run_minuit] Saved prefit cache")

        # ── FC: early return if caller only wanted the prefit ────────────────
        if prefit_only:
            return {'nuis_best': nuis_best,
                    'nuis_cov':  nuis_cov,
                    'C_reco':    C_reco}
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
            unc=unc_juno, opp=opp, toy_dm2_31=toy_dm2_31)

        sp_obs_tot = juno_det._get_obs_spectrum()


        # ── near detector ─────────────────────────────────────────────────────
        if args_juno.fit_type == "PMOP_combined" and fit_mode == "combined":
            if use_tao:
                (chi2_tao, _, s_tao, s_tot_tao) = tao_det.chi2_contribution(
                    Nrea_tao=tao_params['Nrea_tao'],
                    NL_tao=[tao_params[f'NL_tao{i}'] for i in range(5)],
                    snf_noneq_norm_tao=[tao_params[f'snf_noneq_norm_tao{i}'] for i in range(2)],
                    abc_tao=[tao_params['abc_tao0'],
                              tao_params['abc_tao1'],
                              tao_params['abc_tao2']],
                    core_power_tao=tao_params['core_power_tao'],
                    eff_nuis_tao=tao_params['eff_nuis_tao'],
                    bckg_rate=tao_params['bckg_rate'],
                    unc=unc_tao,nuisance=nuisance)

            if use_dyb:
                noneq = 0.0
                if args_dyb.model == "CPC2021":
                    noneq = dyb_params['snf_noneq_norm_dyb']
                nuisance_coarse = np.array(nuisance)
                chi2_dyb, s_dyb, s_tot_dyb = dyb_det.chi2_contribution(
                    nuisance_coarse=nuisance_coarse,
                    snf_noneq_norm_dyb=noneq,
                    E_segments=E_segments)

        # ── plot ──────────────────────────────────────────────────────────────
        if plot_fit:
            global FULL_CHI2
            FULL_CHI2 = chi2_nolog + chi2_dyb + chi2_tao + chi2_penalty_er
            plot_fit_residuals_all(s_tot_juno, sp_obs_tot, s_juno,
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
                title=f"PRL 2025 + TAO, {len(E_segments)} segments")
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




    if fit_mode != "combined":
        start_values = start_values[0:len(start_values)-len(tao_param_names)-len(dyb_param_names)]
        param_names = param_names[0:len(param_names)-len(tao_param_names)-len(dyb_param_names)]

    if not args_juno.silent_print: print(start_values)
    if not args_juno.silent_print: print(param_names)

    chi2_func = make_chi2_func(n_fixed=5, n_total=len(start_values))
    # inside run_minuit, just after chi2_func = make_chi2_func(...)
    # import time
    # _chi2_count = [0]
    # _chi2_total = [0.0]
    # _chi2_raw = chi2_func
    # def chi2_timed(*p):
    #     t = time.perf_counter()
    #     r = _chi2_raw(*p)
    #     _chi2_total[0] += time.perf_counter() - t
    #     _chi2_count[0] += 1
    #     return r
    # chi2_timed.__signature__ = chi2_func.__signature__
    # chi2_func = chi2_timed
    m = Minuit(chi2_func, *start_values, name=param_names)

    # ── FC: pin dm²₃₁ at requested value before any migrad stage ─────────────
    if fix_dm2_31_to is not None:
        m.values['dm2_31'] = fix_dm2_31_to
        m.fixed['dm2_31']  = True

    final_fit_start = datetime.now()
    # ── limits ───────────────────────────────────────────────────────────────
    # shared
    apply_shared_limits(m, args_juno, n_shape_bins=n_nuis)

    # JUNO
    juno_det.apply_limits(m)

    # TAO
    if fit_mode == "combined":
        if use_tao:
            tao_det.apply_limits(m)

        # DYB
        if use_dyb:
            dyb_det.apply_limits(m)



    if not args_juno.silent_print:
        print("-----Parameter init values and limits-----")
        for p in m.parameters:
            print(p, m.values[p], m.limits[p])

    if args_juno.fit_type != 'PMOP_combined' or fit_mode == 'prefit_wosegs':
        if not args_juno.silent_print: print("Fixing shape nuisances")
        for i in range(n_nuis):
            m.fixed[f'nuis{i}'] = True

    first_fix_juno = juno_det.get_first_fix()

    first_fix_tao = tao_det.get_first_fix() if use_tao and fit_mode == "combined" else []
    first_fix_dyb = dyb_det.get_first_fix() if use_dyb and fit_mode == "combined" else []

    first_fix = first_fix_juno + first_fix_tao + first_fix_dyb


    # ── stage 1: fix osc + me_rho + all nuisances ─────────────────────────────
    for p in SHARED_PARAM_NAMES:   # sin2_th12..me_rho
        m.fixed[p] = True
    for p in first_fix:
        m.fixed[p] = True

    if not (args_juno.fit_type == 'PMOP_combined' and fit_mode == 'prefit_wosegs'):
        m.migrad()
    # ── stage 2: free osc + me_rho ───────────────────────────────────────────
    if not args_juno.silent_print: print("freeing shared params (osc + me_rho)")
    for p in SHARED_PARAM_NAMES:
        if p == 'dm2_31' and fix_dm2_31_to is not None:
            continue                    # FC: keep dm²₃₁ pinned
        m.fixed[p] = False
    m.migrad()
    #m.strategy = 2

    # ── stage 3: free everything else ─────────────────────────────────────────
    if not args_juno.silent_print: print("freeing everything else")
    for p in first_fix:
        if not m.fixed[p]:
            continue
        m.fixed[p] = False


    m.migrad()
    migrad_time = datetime.now()
    if args_juno.run_hesse: m.hesse()
    hesse_time = datetime.now()
    #n = _chi2_count[0]; t = _chi2_total[0]
    #print(f"chi2 evals: {n}, total {t:.1f}s, mean {1000*t/n:.1f}ms")
    # ── FC: short-circuit for toy fits — return Minuit object now ────────────
    if toy_quiet:
        if return_prefit:
            return m, {'nuis_best': nuis_best,
                       'nuis_cov':  nuis_cov,
                       'C_reco':    C_reco}
        if return_cost_fn:
            return m, chi2_wrapper
        return m

    # ── diagnostics ───────────────────────────────────────────────────────────
    if not args_juno.silent_print:
        print("Function minimum value: ", m.fval)
        nbins = len(ensp_nom_juno["rdet"].bins) - 1
        if args_juno.fit_type == "PMOP_combined":
            nbins += DYB_BINS
        global FREE_PARAM
        FREE_PARAM = 3 #+ n_nuis
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
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root", "update")
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
                start_rate = float(args_juno.geo_tnu_rate) * float(args_juno.tnu_to_cpd) * args_juno.detector_efficiency
            elif key == "wr" or key == "atm":
                start_rate = backgrounds["JUNO"][key]["rate"]*args_juno.detector_efficiency
            else:
                start_rate = backgrounds["JUNO"][key]["rate"]
            rate_unc        = backgrounds["JUNO"][key].get("rate_unc", None)
            rate_fit        = start_rate * (1 + nuis) * (1 +m.values['eff_nuis'])
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

        if use_tao and not prefit_cache:
            print("************TAO Detector response************")
            for pname in tao_param_names:
                if fit_mode == "combined":
                    print(f"{pname}: {m.values[pname]} +/- {m.errors[pname]}")
                else:
                    print(f"{pname}: {m_pre.values[pname]} +/- {m_pre.errors[pname]}")

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
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root", "update")
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
    chi2_true     = m.fval
    plotchi2 = chi2_wrapper(*best_params, plot_fit=True)

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
        #start_values[3] = nuosc.op_nom['dm2_31'] * 1e3
        m.values['dm2_31'] = nuosc.op_nom['dm2_31'] * 1e3
        chi2_func_opp = make_chi2_func(n_fixed=5, n_total=len(start_values), opp=True)
        m_opp = Minuit(chi2_func_opp, *m.values, name=param_names)
        apply_shared_limits(m_opp, args_juno, n_shape_bins=n_nuis)
        juno_det.apply_limits(m_opp)
        if fit_mode == "combined":
            if use_tao: tao_det.apply_limits(m_opp)
            if use_dyb: dyb_det.apply_limits(m_opp)

        if args_juno.fit_type != 'PMOP_combined' or fit_mode == 'prefit_wosegs':
            for i in range(n_nuis):
                m_opp.fixed[f'nuis{i}'] = True

        for p in SHARED_PARAM_NAMES:
            m_opp.fixed[p] = True
        for p in first_fix:
            m_opp.fixed[p] = True

        if not (args_juno.fit_type == 'PMOP_combined' and fit_mode == 'prefit_wosegs'):
            m_opp.migrad()

        for p in SHARED_PARAM_NAMES:
            m_opp.fixed[p] = False
        m_opp.migrad()
        m_opp.strategy = 2

        for p in first_fix:
            if m_opp.fixed[p]: m_opp.fixed[p] = False
        m_opp.migrad()

        if not args_juno.silent_print: print(m_opp)

        chi2_opp = m_opp.fval

        if not args_juno.silent_print: print("NMO significance: ",chi2_true, chi2_opp)


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

    print('Migrad: ', migrad_time - final_fit_start)
    print('Hesse: ' , hesse_time - migrad_time)
    print("Final fit time:", datetime.now()-final_fit_start)
    return m
from .dispatcher import (DetectorBundle, build_chi2,
                         apply_all_limits, collect_first_fix)
from .param_policy import ParamPolicy


def run_minuit_bundled(bundles, policy, args_juno, *,
                       prefit_cache=None, prefit_only=False,
                       return_cost_fn=False,
                       fix_dm2_31_to=None, toy_dm2_31=None,
                       true_dm2_31=None, toy_quiet=False):
    """
    Multi-bundle entry point. Mirrors the three fit modes of the original
    run_minuit, including on-the-fly DYB+TAO prefit generation when no
    cache is provided, and full post-fit reporting.
    """


    if toy_quiet:
        args_juno.silent_print = True
        args_juno.run_hesse    = False
        args_juno.write_output = False

    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt,
                                   NO=bool(args_juno.NMO_opt))

    fit_type  = args_juno.fit_type
    fit_mode       = getattr(args_juno, 'fit_mode', 'combined')
    is_prefit_mode = (args_juno.fit_type == 'PMOP_combined'
                      and fit_mode.startswith('prefit'))
    n_shape        = len(args_juno.seg_bins_nonuniform) -1
    n_nuis         = n_shape                                  # legacy alias
    E_segments     = np.array(args_juno.seg_bins_nonuniform)

    # ── 1. Find TAO/DYB bundles BEFORE we filter them out ────────────────
    tao_bundle       = next((b for b in bundles if b.kind == 'tao'), None)
    dyb_bundle       = next((b for b in bundles if b.kind == 'dyb'), None)
    juno_bundles_all = [b for b in bundles if b.kind == 'juno']
    use_tao = tao_bundle is not None
    use_dyb = dyb_bundle is not None

    # ── 2. Resolve prefit data — cache, or compute on the fly ────────────
    nuis_best = None
    nuis_cov  = None
    C_reco    = None
    m_pre     = None                          # captured for TAO printout

    if fit_type == "PMOP_combined":
        if is_prefit_mode:
            if prefit_cache is not None:
                nuis_best = np.asarray(prefit_cache['nuis_best'])
                nuis_cov  = np.asarray(prefit_cache['nuis_cov'])
                cr = prefit_cache.get('C_reco')
                if cr is not None:
                    cr = np.asarray(cr)
                    if cr.size > 0:
                        C_reco = cr
                if not args_juno.silent_print:
                    print(f"[bundled] using cached prefit "
                          f"({'C_reco present' if C_reco is not None else 'no C_reco'})")
            else:
                if tao_bundle is None and dyb_bundle is None:
                    raise ValueError(
                        f"fit_mode={fit_mode!r}: no prefit cache AND no TAO/DYB "
                        "bundles available — cannot compute the prefit.")
                if not args_juno.silent_print:
                    print("[bundled] no cache — running DYB+TAO prefit on the fly")

                m_pre, nuis_best, nuis_cov = run_dyb_tao_prefit(
                    dyb_det=dyb_bundle.detector if dyb_bundle else None,
                    tao_det=tao_bundle.detector if tao_bundle else None,
                    E_segments=E_segments,
                    unc_dyb=(dyb_bundle.unc_key if dyb_bundle else 'all'),
                    unc_tao=(tao_bundle.unc_key if tao_bundle else 'all'),
                    args_juno=args_juno,
                    args_tao=(tao_bundle.args if tao_bundle else None),
                    args_dyb=(dyb_bundle.args if dyb_bundle else None),
                    ndays_dyb=dyb_bundle.detector.ndays if dyb_bundle else 0.0,
                    ndays_tao=tao_bundle.detector.ndays if tao_bundle else 0.0,
                    dyb_exp=dyb_bundle.detector.ensp_nom['rtot'] if dyb_bundle else None,
                    tao_exp=tao_bundle.detector.ensp_nom['rtot'] if tao_bundle else None,
                    ibd_prefit=dyb_bundle.detector.ensp_nom['ribd'])

        # ── 3. prefit_wosegs: compute C_reco if missing from cache ───────────
        if fit_mode == 'prefit_wosegs' and C_reco is None:
            if not juno_bundles_all:
                raise ValueError(
                    "prefit_wosegs requires at least one JUNO bundle to "
                    "compute C_reco via propagate_nuis_to_reco.")
            b0        = juno_bundles_all[0]
            juno_det0 = b0.detector
            args0     = b0.args
            unc_juno0 = b0.unc_key
            juno_param_names0 = juno_det0.get_param_names()
            n_NL_0   = sum(1 for p in juno_param_names0 if p.startswith("NL"))
            n_rea_0  = sum(1 for p in juno_param_names0 if p.startswith("snf_noneq_norm"))
            n_core_0 = sum(1 for p in juno_param_names0 if p.startswith("core_power"))
            bckgs_for_jp = {}
            for key, info in b0.backgrounds["JUNO"].items():
                bckgs_for_jp[key] = {"pull": 0.0,
                                      "sigma": info["rate_unc"],
                                      "b2b":   info["b2b_unc"],
                                      "name":  info["name"],
                                      "b2b_scale": info["b2b_scale"]}
            jp_fixed = dict(
                Nrea=0.0,
                NL=([args0.kB, args0.fC, args0.kI, args0.LY_shared]
                    if args0.NL != "pull" else [0.0] * n_NL_0),
                abc=[args0.a, args0.b, args0.c],
                bckgs=bckgs_for_jp,
                snf_noneq_norm=[0.0] * n_rea_0,
                eff_nuis=0.0,
                ThU_ratio=args0.Th_U_ratio,
                core_powers_delta=[0.0] * n_core_0,
                efission=list(np.array(args0.efission)*1e-2))
            if not args_juno.silent_print:
                print("[bundled] computing C_reco via propagate_nuis_to_reco "
                      f"(using JUNO bundle {b0.phase}/{b0.shell})")
            C_reco = propagate_nuis_to_reco(
                juno_det=juno_det0,
                nuis_best=nuis_best, nuis_cov=nuis_cov,
                sin2_12=nuosc.op_nom["sin2_th12"],
                sin2_13=nuosc.op_nom["sin2_th13"],
                dm2_21=nuosc.op_nom["dm2_21"],
                dm2_31=nuosc.op_nom["dm2_31"],
                me_rho=args_juno.me_rho,
                fixed_juno_params=jp_fixed,
                args_juno=args_juno, unc_juno=unc_juno0)

        # ── 4. Save freshly-computed prefit to disk if requested ─────────────
        if (is_prefit_mode and prefit_cache is None
                and getattr(args_juno, 'save_prefit', False)):
            out_path = args_juno.prefit_path
            out_dir  = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                out_path,
                nuis_best=nuis_best,
                nuis_cov=nuis_cov,
                C_reco=C_reco if C_reco is not None else np.array([]))
            if not args_juno.silent_print:
                print(f"[bundled] saved prefit cache → {out_path}")

        # ── 5. FC early return ───────────────────────────────────────────────
        if prefit_only:
            return {'nuis_best': nuis_best,
                    'nuis_cov':  nuis_cov,
                    'C_reco':    C_reco}

        # ── 6. Drop TAO/DYB from the JUNO fit in any prefit mode ─────────────
        if fit_mode != 'combined':
            n_pre = len(bundles)
            bundles = juno_bundles_all
            if not args_juno.silent_print and len(bundles) != n_pre:
                print(f"[bundled] fit_mode={fit_mode}: dropped "
                      f"{n_pre - len(bundles)} non-JUNO bundle(s) — they're "
                      "absorbed into the prefit cache")

        # ── 7. prefit_wosegs: stamp C_reco onto every JUNO detector ──────────
        if fit_mode == 'prefit_wosegs':
            for b in bundles:
                if b.kind == 'juno':
                    b.detector.set_external_cov(C_reco)

    # ── 8. Build chi² closure ────────────────────────────────────────────
    chi2_func, names, starts, routing = build_chi2(bundles, policy, args_juno)
    #chi2_func = time_chi2(chi2_func, label='chi2')
    if fit_type == "PMOP_combined":
        # ── 9. prefit_wsegs: add (Δ @ nuis_cov_inv @ Δ) penalty to closure ───
        if fit_mode == 'prefit_wsegs':
            nuis_cov_inv  = np.linalg.inv(nuis_cov)
            nuis_best_arr = np.asarray(nuis_best[:n_shape], dtype=float)
            shape_idx     = [names.index(f'nuis{i}') for i in range(n_shape)]
            _orig_chi2 = chi2_func
            _orig_sig  = chi2_func.__signature__
            def _chi2_with_penalty(*flat):
                v = _orig_chi2(*flat)
                d = np.array([flat[i] for i in shape_idx]) - nuis_best_arr
                return v + float(d @ nuis_cov_inv @ d)
            _chi2_with_penalty.__signature__ = _orig_sig
            chi2_func = _chi2_with_penalty

        # ── 10. Seed shape-nuisance starts from prefit (both prefit modes) ───
        if is_prefit_mode and nuis_best is not None:
            for i in range(min(n_shape, len(nuis_best))):
                starts[names.index(f'nuis{i}')] = float(nuis_best[i])

    # ── 11. Optional FC dm²₃₁ seed ───────────────────────────────────────
    if true_dm2_31 is not None:
        starts[names.index('dm2_31')] = true_dm2_31 * 1e3

    # ── 12. Build Minuit ─────────────────────────────────────────────────
    m = Minuit(chi2_func, *starts, name=names)
    final_fit_start = datetime.now()
    apply_shared_limits(m, args_juno, n_shape_bins=n_shape)
    apply_all_limits(m, bundles, routing)

    if fix_dm2_31_to is not None:
        m.values['dm2_31'] = fix_dm2_31_to
        m.fixed['dm2_31']  = True

    if not args_juno.silent_print:
        print(f"[bundled] {len(bundles)} bundles, {len(names)} Minuit params")
        for b in bundles:
            print(f"   bundle: {b.kind:5s} phase={b.phase} shell={b.shell}")

    # ── 13. Shape-nuisance fixing rule (matches original run_minuit) ─────
    if args_juno.fit_type != 'PMOP_combined' or fit_mode == 'prefit_wosegs':
        for i in range(n_shape):
            m.fixed[f'nuis{i}'] = True

    first_fix = collect_first_fix(bundles, routing)

    # ── 14. Stage 1: fix shared + first_fix nuisances; Migrad ────────────
    for p in SHARED_PARAM_NAMES:
        m.fixed[p] = True
    for p in first_fix:
        m.fixed[p] = True
    if not (args_juno.fit_type == 'PMOP_combined' and fit_mode == 'prefit_wosegs'):
        m.migrad()

    # ── 15. Stage 2: free osc + me_rho ───────────────────────────────────
    for p in SHARED_PARAM_NAMES:
        if p == 'dm2_31' and fix_dm2_31_to is not None:
            continue
        m.fixed[p] = False


    m.migrad()

    # ── 16. Stage 3: free everything else ────────────────────────────────
    for p in first_fix:
        if m.fixed[p]:
            m.fixed[p] = False

    m.migrad()
    migrad_time = datetime.now()

    if args_juno.run_hesse and not toy_quiet:
        m.hesse()
    hesse_time = datetime.now()

    # ── 17. FC short-circuit ─────────────────────────────────────────────
    if toy_quiet:
        if return_cost_fn:
            return m, chi2_func
        return m

    # ═══════════════════════════════════════════════════════════════════════
    #  POST-FIT REPORTING
    # ═══════════════════════════════════════════════════════════════════════

    # First JUNO bundle is used for header info / counts
    b_first = juno_bundles_all[0] if juno_bundles_all else None
    bckgs_keys = (list(b_first.backgrounds['JUNO'].keys())
                  if b_first is not None else [])
    juno_param_names = (b_first.detector.get_param_names()
                        if b_first is not None else [])
    n_NL  = sum(1 for p in juno_param_names if p.startswith("NL"))
    n_abc = sum(1 for p in juno_param_names if p.startswith("abc"))

    # Convenience: route a JUNO base name through the first JUNO bundle's
    # mapping (gives plain name when global, mangled otherwise)
    def juno_param(base):
        return routing[id(b_first)][base] if b_first is not None else base

    unc_juno_first = b_first.unc_key if b_first is not None else 'stat'

    # ── 18. Diagnostics block ────────────────────────────────────────────
    if not args_juno.silent_print:
        print("Function minimum value: ", m.fval)
        global FREE_PARAM
        FREE_PARAM =3 #+ n_nuis
        print("Free parameters", FREE_PARAM)
        print("-----Good things, should be True-----")
        print("FMin valid?             ", m.fmin.is_valid)
        print("Has covariance?         ", m.fmin.has_covariance)
        print("Covariance accurate?    ", m.fmin.has_accurate_covar)
        print("Minos valid?            ", m.fmin.is_valid)
        print("-----Bad things, should be False-----")
        print("Above EDM threshold?    ", m.fmin.is_above_max_edm)
        print("Has reached call limit? ", m.fmin.has_reached_call_limit)
        print("Hesse failed?           ", m.fmin.hesse_failed)

    # ── 19. ROOT output (postfit) ────────────────────────────────────────
    output_file = None
    dir_nuis = dir_post = dir_post_dyb = None
    if args_juno.write_output:
        output_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root",
            "update")
        juno_dir = output_file.GetDirectory("juno")
        juno_dir.cd()
        write_param_dbl("chi2/ndof", m.fval/len(b_first.detector.ensp_nom['rdet'].bins) -4)
        dir_nuis = juno_dir.mkdir("nuisances")
        dir_post = juno_dir.mkdir("postfit")
    if not args_juno.silent_print:
        print("************Other parameter fit results************")
        for i in range(len(SHARED_PARAM_NAMES), len(m.values)):
            print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")

        # ── Background rates per JUNO bundle ─────────────────────────────
        print("************Background rates************")
        for b in juno_bundles_all:
            tag_pre = (f"[{b.phase}/{b.shell}] "
                       if len(juno_bundles_all) > 1 else "")
            for key, info in b.backgrounds['JUNO'].items():
                mangled  = routing[id(b)][f'bkg_{key}']
                nuis     = m.values[mangled]
                nuis_err = m.errors[mangled]
                rate_unc        = info.get("rate_unc", None)
                if key == "geo":
                    start_rate = (float(b.args.geo_tnu_rate)
                                  * float(b.args.tnu_to_cpd)) * float(b.args.detector_efficiency)
                    rate_fit = start_rate * (1 + nuis) * (1+m.values[routing[id(b)]['eff_nuis']])
                elif key == "wr" or key == "atm":
                    start_rate = info["rate"]*float(b.args.detector_efficiency)
                    rate_fit = start_rate * (1 + nuis) * (1+m.values[routing[id(b)]['eff_nuis']])
                else:
                    start_rate = info["rate"]
                    rate_fit  = start_rate * (1 + nuis)

                rate_err        = start_rate * nuis_err
                deviation_sigma = ((rate_fit - start_rate)
                                   / (start_rate * rate_unc)
                                   if rate_unc else float('nan'))
                print(f"{tag_pre}{key}: {rate_fit:.4f} ± {rate_err:.4f} cpd "
                      f"(rel unc: {rate_err/rate_fit:.3%}; "
                      f"deviation: {deviation_sigma:.2f} σ)")
                if args_juno.write_output:
                    dir_nuis.cd()
                    if rate_unc != 0.0:
                        write_parameter(f"{tag_pre}{key}", nuis/rate_unc, nuis_err/rate_unc)
                    else:
                        write_parameter(f"{tag_pre}{key}", nuis, nuis_err)

        # ── Normalization parameters per JUNO bundle ─────────────────────
        print("************Normalization parameters************")
        for b in juno_bundles_all:
            tag_pre = (f"[{b.phase}/{b.shell}] "
                       if len(juno_bundles_all) > 1 else "")
            n_name = routing[id(b)]['Nrea']
            e_name = routing[id(b)]['eff_nuis']
            print(f"{tag_pre}Rflux: {1.0 + m.values[n_name]} +/- {m.errors[n_name]}")
            print(f"{tag_pre}Eff:   "
                  f"{b.args.detector_efficiency * (1 + m.values[e_name])} "
                  f"+/- {b.args.detector_efficiency * m.errors[e_name]}")
            if args_juno.write_output:
                dir_nuis.cd()
                write_parameter(f"{tag_pre}Rflux", m.values[n_name]/b.detector.args.r2_unc, m.errors[n_name]/b.detector.args.r2_unc)
                write_parameter(f"{tag_pre}eff_pull", m.values[e_name]/b.detector.args.eff_unc, m.errors[e_name]/b.detector.args.eff_unc)

        # ── Shared (osc + me_rho) ────────────────────────────────────────
        print("************Shared parameters************")
        print(f"me_rho: {m.values['me_rho']} +/- {m.errors['me_rho']}")

        # ── JUNO detector response (NL + abc) — usually global ───────────
        if b_first is not None:
            print("************JUNO Detector response************")
            print(f"a:  {m.values[juno_param('abc0')]} "
                  f"+/- {m.errors[juno_param('abc0')]}")
            print(f"b:  {m.values[juno_param('abc1')]} "
                  f"+/- {m.errors[juno_param('abc1')]}")
            print(f"c:  {m.values[juno_param('abc2')]} "
                  f"+/- {m.errors[juno_param('abc2')]}")
            print(f"kB: {m.values[juno_param('NL0')]}  "
                  f"+/- {m.errors[juno_param('NL0')]}")
            print(f"fC: {m.values[juno_param('NL1')]}  "
                  f"+/- {m.errors[juno_param('NL1')]}")
            print(f"kI: {m.values[juno_param('NL2')]}  "
                  f"+/- {m.errors[juno_param('NL2')]}")
            print(f"LY: {m.values[juno_param('NL3')]}  "
                  f"+/- {m.errors[juno_param('NL3')]}")

        # ── TAO detector response ────────────────────────────────────────
        if use_tao and tao_bundle is not None:
            print("************TAO Detector response************")
            tao_pnames = tao_bundle.detector.get_param_names()
            for pname in tao_pnames:
                mangled = routing[id(tao_bundle)].get(pname, pname) \
                          if id(tao_bundle) in routing else pname
                # In combined mode TAO is in the JUNO Minuit object.
                # In prefit modes it was fit by m_pre (if we ran it fresh).
                if fit_mode == "combined" and mangled in m.parameters:
                    print(f"{pname}: {m.values[mangled]} "
                          f"+/- {m.errors[mangled]}")
                elif m_pre is not None and pname in m_pre.parameters:
                    print(f"{pname}: {m_pre.values[pname]} "
                          f"+/- {m_pre.errors[pname]}  (from prefit)")
                else:
                    pass    # cached prefit — m_pre unavailable; skip

        # ── ROOT writes for JUNO response ────────────────────────────────
        if args_juno.write_output and b_first is not None:
            dir_nuis.cd()
            sigma_prior = np.sqrt(np.diag(b_first.detector.covariance_er))

            det_params = [("a","abc0"), ("b","abc1"), ("c","abc2"),
                          ("kB","NL0"), ("fC","NL1"), ("kI","NL2"), ("LY","NL3")]

            for i, (tag, base) in enumerate(det_params):
                name    = juno_param(base)
                nominal = m.init_params[name].value
                dev     = (m.values[name] - nominal) / sigma_prior[i]   # element0 (pull)
                rel_unc =  m.errors[name]            / sigma_prior[i]   # element1 (err in σ units)
                write_parameter(tag, dev, rel_unc)

            write_parameter('me_rho', (m.values['me_rho'] - b_first.detector.args.me_rho) / b_first.detector.args.me_rho_scale, m.errors['me_rho']/b_first.detector.args.me_rho_scale)
            write_parameter('ratio', (m.values['ThU_ratio'] - b_first.detector.args.Th_U_ratio) / b_first.detector.args.Th_U_ratio_err, m.errors['ThU_ratio']/b_first.detector.args.Th_U_ratio_err)
            for i, (tag, base) in enumerate([("YJ-C1","core_power0"), ("YJ-C2","core_power1"), ("YJ-C3","core_power2"),
                              ("YJ-C4","core_power3"), ("YJ-C5","core_power4"), ("YJ-C6", "core_power5"),
                              ("TS-C1","core_power6"), ("TS-C2","core_power7"), ("FCG-C1-4", "core_power8"), ("DYB-C", "core_power9")]):
                name = juno_param(base)
                write_parameter(tag, m.values[name]/b_first.detector.args.core_flux_unc, m.errors[name]/b_first.detector.args.core_flux_unc)


    if args_juno.write_output and output_file is not None:
        dir_post.cd()
        write_param_dbl("Efficiency", args_juno.detector_efficiency*(1+m.values["eff_nuis"]))

    if args_juno.write_output:
        dyb_dir = output_file.GetDirectory("dyb")
        dyb_dir.cd()
        dir_nuis_dyb = dyb_dir.mkdir("nuisances")
        dir_nuis_dyb.cd()
        for i in range(n_nuis):
            if fit_mode == 'prefit_wosegs':
                write_parameter(f"rea_shape_f{i}", m_pre.values[f"nuis{i}"], m_pre.errors[f"nuis{i}"])
            else:
                write_parameter(f"rea_shape_f{i}", m.values[f"nuis{i}"], m.errors[f"nuis{i}"])

        output_file.Close()
    # ── 20. Correlation heat-map ─────────────────────────────────────────
    if args_juno.plot_correlation:
        import re
        corr = np.array(m.covariance.correlation())

        def _drop(p):
            return (re.fullmatch(r"nuis\d+", p)          # shape nuisances
                    or p.startswith("core_power")        # power corrections (+_tao)
                    or p.startswith("snf_noneq_norm")   # SNF / non-eq (+_tao/_dyb)
                    or p.startswith("efission"))

        keep   = [i for i, p in enumerate(m.parameters) if not _drop(p)]
        corr   = corr[np.ix_(keep, keep)]
        labels = [m.parameters[i] for i in keep]

        import seaborn as sns
        plt.figure(figsize=(12, 12))          # smaller figure → smaller squares
        ax = sns.heatmap(corr,
                         annot=False,          # remove the numbers
                         xticklabels=labels, yticklabels=labels,
                         cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                         square=True)          # force cells to be actual squares
        ax.tick_params(labelsize=16)           # bigger axis labels
        plt.tight_layout()

        plt.savefig(f"Plots/corr_{args_juno.file_prefix}", dpi=200)

    # ── 21. dm²₃₁ profile ────────────────────────────────────────────────
    if args_juno.profile_dm2_31:
        profile_dm2_31(m, args_juno, NMO_opt=bool(args_juno.NMO_opt),
                       write=args_juno.write_output)

    # ── 22. MINOS errors ─────────────────────────────────────────────────
    merrors = bool(args_juno.calc_minuit_errors)
    if merrors:
        m.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31")

    # ── 23. Oscillation parameter results ────────────────────────────────
    osc_params_nom = [
        nuosc.op_nom["sin2_th12"],
        nuosc.op_nom["sin2_th13"],
        nuosc.op_nom["dm2_21"] * 1e5,
        nuosc.op_nom["dm2_31"] * 1e3,
    ]


    if not args_juno.silent_print: # and not args_juno.is_data:
        if args_juno.write_output:
                output_file = ROOT.TFile(
                            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
                            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root",
                            "update")

                dir_osc = output_file.mkdir("oscillation_parameters")
                dir_osc.cd()
        print("************Oscillation parameter fit results************")
        for i in range(4):
            err = (max(abs(m.merrors[i].upper), abs(m.merrors[i].lower))
                   if merrors else m.errors[i])
            val             = m.values[i]
            exp_val         = osc_params_nom[i]
            rel_unc         = err / val if val != 0 else float('nan')
            deviation_sigma = (val - exp_val) / err if err != 0 else float('nan')
            print(f"{m.parameters[i]}: {val:.6g} +/- {err:.6g}; "
                  f"rel unc: {rel_unc:.3%}; deviation: {deviation_sigma:.2f} σ")
            if args_juno.write_output:
                write_osc_params(str(m.parameters[i]), val, err)


    if args_juno.write_output:
        output_file.Close()

    # ── 24. Post-fit residual plot (one per JUNO bundle) ─────────────────
    chi2_true = m.fval
    if not args_juno.silent_print:
        # Replay each JUNO bundle's chi2_contribution at the best-fit point
        # to obtain spectra and feed plot_fit_residuals.
        values_best = {n: m.values[n] for n in m.parameters}
        for b in juno_bundles_all:
            mapping = routing[id(b)]
            def gv(base): return values_best[mapping[base]]
            base_names = b.detector.get_param_names()
            NL_vals = [gv(p) for p in base_names if p.startswith('NL')]
            bckgs_for_chi2 = {}
            for k, raw in b.backgrounds['JUNO'].items():
                bckgs_for_chi2[k] = {
                    'pull':      gv(f'bkg_{k}'),
                    'sigma':     raw['rate_unc'],
                    'b2b':       raw['b2b_unc'],
                    'name':      raw['name'],
                    'b2b_scale': raw['b2b_scale'],
                }
            (_chi2j, _nolog, _erpen, s_juno, s_tot_juno,
             fit_bckgs, _rfis, cm_juno) = b.detector.chi2_contribution(
                sin2_12=values_best['sin2_th12'],
                sin2_13=values_best['sin2_th13'],
                dm2_21 =values_best['dm2_21'] * 1e-5,
                dm2_31 =values_best['dm2_31'] * 1e-3,
                me_rho =values_best['me_rho'],
                Nrea     = gv('Nrea'),
                NL       = NL_vals,
                abc      = [gv('abc0'), gv('abc1'), gv('abc2')],
                bckgs_for_chi2 = bckgs_for_chi2,
                snf_noneq_norm = [gv(p) for p in base_names
                                  if p.startswith('snf_noneq_norm')],
                eff_nuis = gv('eff_nuis'),
                ThU_ratio= gv('ThU_ratio'),
                core_powers_delta = [gv(p) for p in base_names
                                     if p.startswith('core_power')],
                efission = [gv(p) for p in base_names
                            if p.startswith('efission')],
                nuisance = [values_best[f'nuis{i}'] for i in range(n_shape)],
                unc=b.unc_key)
            sp_obs = b.detector._get_obs_spectrum()

            plot_fit_residuals_all(
                s_tot_juno, sp_obs, s_juno, fit_bckgs,
                b.detector.ndays, b.args,
                detector_label="JUNO",
                tag=f"{b.phase}_{b.shell}" if len(juno_bundles_all) > 1 else None,
                chi2_value=m.fval,
                ndf = len(sp_obs.bins)-4,
                write_output=args_juno.write_output, cm=cm_juno, eff=b.detector.args.detector_efficiency*(1+m.values['eff_nuis']))
    # ── 25. Solar contour ────────────────────────────────────────────────
    if args_juno.solar_contour:
        try:
            from .. import contour_1sigma, contour_2sigma  # type: ignore
        except Exception:
            contour_1sigma = []; contour_2sigma = []
        if contour_1sigma and contour_2sigma:
            x1, y1 = zip(*contour_1sigma)
            x2, y2 = zip(*contour_2sigma)
            plt.figure(figsize=(7, 6))
            plt.plot(x1, y1, 'b-',  label='68% CL (1σ)')
            plt.plot(x2, y2, 'r--', label='95% CL (2σ)')
            plt.plot(m.values["sin2_th12"], m.values["dm2_21"], 'ko', label='Best fit')
            plt.xlabel(r'$\sin^2\theta_{12}$')
            plt.ylabel(r'$\Delta m^2_{21}$ [eV$^2$]')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("Plots/solar_contour.png")

    # ── 26. NMO significance ─────────────────────────────────────────────
    chi2_opp = None
    if args_juno.NMO_sig:
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt,
                                       NO=not bool(args_juno.NMO_opt))
        # Build opposite-ordering closure (re-uses the same routing & bundles)
        chi2_func_opp, names_opp, starts_opp, routing_opp = build_chi2(
            bundles, policy, args_juno, opp=True)
        # Seed at the current best-fit values, but flip dm²₃₁ central to the
        # opposite-ordering nominal (same as the original)
        seeds = list(m.values)
        seeds[names_opp.index('dm2_31')] = nuosc.op_nom['dm2_31'] * 1e3
        m_opp = Minuit(chi2_func_opp, *seeds, name=names_opp)
        apply_shared_limits(m_opp, args_juno, n_shape_bins=n_shape)
        apply_all_limits(m_opp, bundles, routing_opp)
        if args_juno.fit_type != 'PMOP_combined' or fit_mode == 'prefit_wosegs':
            for i in range(n_shape):
                m_opp.fixed[f'nuis{i}'] = True
        for p in SHARED_PARAM_NAMES: m_opp.fixed[p] = True
        for p in collect_first_fix(bundles, routing_opp): m_opp.fixed[p] = True
        if not (args_juno.fit_type == 'PMOP_combined' and fit_mode == 'prefit_wosegs'):
            m_opp.migrad()
        for p in SHARED_PARAM_NAMES: m_opp.fixed[p] = False
        m_opp.migrad()
        for p in collect_first_fix(bundles, routing_opp):
            if m_opp.fixed[p]: m_opp.fixed[p] = False
        m_opp.migrad()
        chi2_opp = m_opp.fval
        if not args_juno.silent_print:
            #print(m_opp)
            print(f"NMO significance: chi2 = {chi2_true}, chi2_opp = {chi2_opp}, "
                  f"Δχ² = {chi2_opp - chi2_true:.2f}")

    # ── 27. Profile / matrix plots ───────────────────────────────────────
    if args_juno.plot_minuit_matrix or args_juno.plot_minuit_profiles:
        os.makedirs(f"{args_juno.plots_folder}/Chi2_profiles/Minuit",
                    exist_ok=True)
    if args_juno.plot_minuit_profiles:
        print("Plotting chi2 profiles")
        for i, pname in enumerate(m.parameters):
            plotname = (f"{args_juno.plots_folder}/Chi2_profiles/Minuit/"
                        f"chi2_{args_juno.stat_opt}_{pname}_{unc_juno_first}.png")
            plot_profile(m, i, pname, plotname)
    if args_juno.print_minuit_correlation:
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())
    if args_juno.plot_minuit_matrix:
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1, 2, 3])
        plt.show()

    print('Migrad: ', migrad_time - final_fit_start)
    print('Hesse: ',  hesse_time - migrad_time)
    print("Final fit time:", datetime.now() - final_fit_start)

    if return_cost_fn:
        return m, chi2_func
    return m
