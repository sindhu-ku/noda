#!/usr/bin/env python3
"""
minuit.py

Helper functions used across the fit.
run_minuit() has moved to combined_fit.py.
Spectrum building has moved to juno_fit.py / dyb_fit.py / tao_fit.py.
"""

import sys
from .. import nuosc as nuosc
from ..noda import *
import numpy as np
import os
from iminuit import Minuit
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.linalg import pinvh

# Global counters updated by combined_fit.py
FULL_CHI2  = 0
DYB_BINS   = 0
FREE_PARAM = 0

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_pair(nuis_cov, C_reco, ensp_nom_juno, n_nuis, save_path="cov_propagation.png"):
    
    # compute correlation matrices from covariances
    def to_corr(cov):
        std = np.sqrt(np.diag(cov))
        # avoid division by zero
        outer = np.outer(std, std)
        outer[outer == 0] = 1.0
        return cov / outer

    corr_seg  = to_corr(nuis_cov)
    corr_reco = to_corr(C_reco)

    n_reco = C_reco.shape[0]

    seg_labels  = [f"$S_{{{i}}}$"  for i in range(n_nuis)]
    reco_labels = [f"$D_{{{i}}}$"  for i in range(n_reco)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(r"Covariance propagation: segment space $\rightarrow$ reconstructed space",
                 fontsize=14)

    kw = dict(cmap="RdBu_r", vmin=-1, vmax=1, square=True,
              cbar_kws={"label": r"Correlation $\rho$", "shrink": 0.8})

    # ── left: segment covariance ──────────────────────────────────────────
    sns.heatmap(corr_seg, ax=axes[0],
                xticklabels=seg_labels, yticklabels=seg_labels, **kw)
    axes[0].set_title(r"Segment factors $-\ C_{\rm seg}$ (DYB+TAO)")
    axes[0].tick_params(axis='both', labelsize=8)

    # ── right: reco covariance ────────────────────────────────────────────
    # only label every 5th bin to avoid crowding
    reco_xticks = [l if i % 5 == 0 else "" for i, l in enumerate(reco_labels)]
    reco_yticks = [l if i % 5 == 0 else "" for i, l in enumerate(reco_labels)]
    sns.heatmap(corr_reco, ax=axes[1],
                xticklabels=reco_xticks, yticklabels=reco_yticks, **kw)
    axes[1].set_title(r"Reconstructed bins $-\ V_{\rm segments}$ (JUNO)")
    axes[1].tick_params(axis='both', labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")



def f_lbl(dm2_31, sin2_12, sin2_13, dm2_21, theta23, dcp, ordering):
    if ordering == "NO":
        return dm2_31
    elif ordering == "IO":
        safe_sin2_12 = np.clip(sin2_12, 1e-10, 1.0)
        safe_sin2_13 = np.clip(sin2_13, 1e-10, 1.0)
        c12  = np.sqrt(1 - safe_sin2_12)
        s13  = np.sqrt(safe_sin2_13)
        term = dm2_21 * (
            2 * c12**2
            - np.sin(2 * np.arcsin(np.sqrt(safe_sin2_12)))
            * s13
            * (np.tan(np.radians(theta23)) * np.cos(np.radians(dcp)) * 2))
        return -dm2_31 + term


def write_parameter(name, val, err):
    vec    = ROOT.TVectorD(2)
    vec[0] = val
    vec[1] = err
    vec.Write(name)


def write_osc_params(name, val, err):
    value = ROOT.TParameter("double")(f"{name}", val)
    error = ROOT.TParameter("double")(f"{name}_err", err)
    value.Write("", ROOT.TObject.kOverwrite)
    error.Write("", ROOT.TObject.kOverwrite)


def profile_dm2_31(m, args_juno, NMO_opt=True, write=False):
    from scipy.interpolate import CubicSpline
    print("Profiling dm2_31 using built-in mnprofile...")

    if NMO_opt:
        res = m.mnprofile("dm2_31", size=int(args_juno.profile_points), bound=(1.5, 3.5))
    else:
        res = m.mnprofile("dm2_31", size=int(args_juno.profile_points), bound=(-3.5, -1.5))

    dm2_scan_vals = res[0]
    chi2_vals     = res[1]
    delta_chi2    = chi2_vals - np.min(chi2_vals)

    dm2_fine       = np.linspace(dm2_scan_vals.min(), dm2_scan_vals.max(), 500)
    cs             = CubicSpline(dm2_scan_vals, chi2_vals)
    chi2_smooth    = cs(dm2_fine)
    delta_chi2_smooth = chi2_smooth - np.min(chi2_smooth)

    plt.figure(figsize=(7, 5))
    plt.plot(dm2_fine, delta_chi2_smooth, label="Smoothed Profile", lw=2)
    plt.xlabel(r"$\Delta m^2_{31}$ [10$^{-3}$ eV$^2$]", fontsize=14)
    plt.ylabel(r"$\Delta \chi^2$", fontsize=14)
    plt.axhline(1.0, color="green", ls="--", label=r"1$\sigma$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Plots/profile_dm2_31_NO_{NMO_opt}_{args_juno.file_prefix}.png")

    best_index       = np.argmin(delta_chi2)
    best_fit_val     = dm2_scan_vals[best_index]

    if write:
        n_points = len(dm2_fine)
        graph    = ROOT.TGraph(n_points,
                               dm2_fine.astype(float),
                               delta_chi2_smooth.astype(float))
        graph.SetName("dm2_31")
        graph.SetMarkerStyle(2)
        out_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_prof = out_file.mkdir("profiles")
        dir_prof.cd()
        graph.Write()
        out_file.Close()

    best_dm2_from_scan    = dm2_fine[np.argmin(chi2_smooth)]
    print(best_dm2_from_scan)
    m.values["dm2_31"]   = best_dm2_from_scan
    print(m.values["dm2_31"])
    m.migrad()


def plot_fit_residuals(s_tot_juno, s_obs, s_juno, bckgs, ndays, args_juno,
                       write_output=False):
    """
    Plot observed vs fitted spectra with residuals and local/cumulative χ²/ndf.
    """
    from .minuit import FULL_CHI2, DYB_BINS, FREE_PARAM  # read updated globals

    if write_output:
        output_file = (f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
                       f"_dm2_31_pull_{args_juno.pull_dm2_31}.root")
        s_juno.WritetoROOT("reactor", f"{output_file}", directory="postfit")

    os.makedirs("Plots", exist_ok=True)
    bin_centers  = s_tot_juno.bins
    bin_centers2 = 0.5 * (s_tot_juno.bins[:-1] + s_tot_juno.bins[1:])

    obs_vals = s_obs.bin_cont
    obs_err  = np.sqrt(np.maximum(obs_vals, 1e-12))
    fit_vals = s_tot_juno.bin_cont
    fit_err  = np.sqrt(np.maximum(fit_vals, 1e-12))

    residuals = (obs_vals - fit_vals) / np.maximum(obs_err, 1e-12)
    chi2_bin  = (obs_vals - fit_vals) ** 2 / np.maximum(obs_err, 1e-12) ** 2

    total_chi2 = np.sum(chi2_bin)
    ndf        = len(chi2_bin) + DYB_BINS - FREE_PARAM
    chi2_ndf   = FULL_CHI2 / ndf

    chi2_bin_rev          = chi2_bin[::-1]
    cum_chi2_rev          = np.cumsum(chi2_bin_rev)
    ndf_bins              = np.arange(1, len(cum_chi2_rev) + 1)
    cum_chi2_ndf_rev      = cum_chi2_rev / ndf_bins
    cum_chi2_ndf          = cum_chi2_ndf_rev[::-1]
    cum_chi2_global_ndf_rev = cum_chi2_rev / (len(chi2_bin) - 3)
    cum_chi2_global_ndf   = cum_chi2_global_ndf_rev[::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1.errorbar(bin_centers2, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label='Observed data')
    ax1.step(bin_centers, np.append(fit_vals, fit_vals[-1]),
             where='mid', color='blue', lw=1.5, label='Total Fit')
    ax1.step(bin_centers, np.append(s_juno.bin_cont, s_juno.bin_cont[-1]),
             where='mid', color='red', lw=1.2, linestyle='-',
             label=f'Reactor $\\bar{{\\nu}}_e$ expectation: '
                   f'{s_juno.GetIntegral()/ndays:.2f} cpd')

    colors = plt.cm.tab10.colors
    name_to_key = {
        "Accidentals": "acc", "Geoneutrinos": "geo",
        r"$^{9}$Li / $^{8}$He": "lihe", "Fast neutrons": "fastn",
        "Double neutrons": "doublen",
        r"$^{13}$C ($\alpha$, n) $^{16}$O": "alphan",
        "Atmospheric neutrinos": "atm", "World reactor neutrinos": "wr",
        r"$^{214}$Bi": "bipo"
    }
    for i, (key, spec) in enumerate(bckgs.items()):
        cpd = spec.GetIntegral() / ndays
        ax1.step(bin_centers, np.append(spec.bin_cont, spec.bin_cont[-1]),
                 where='mid', color=colors[i % len(colors)], lw=1,
                 label=f'{key}: {cpd:.2f} cpd')
        if write_output:
            spec.WritetoROOT(f"{name_to_key.get(key)}", f"{output_file}", "postfit")

    ax1.set_ylabel("Events")
    leg = ax1.legend(fontsize=9, loc="upper right", frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.7, 12.0)

    ax2.errorbar(bin_centers2, residuals, yerr=np.ones_like(residuals),
                 fmt='o', color='blue', markersize=3)
    ax2.axhline(0, color='k', lw=1)
    for nsigma, alpha in zip([1, 2, 3], [0.25, 0.15, 0.1]):
        ax2.fill_between(bin_centers, -nsigma, nsigma, color='blue', alpha=alpha)
    ax2.set_ylabel("Residuals [σ]")
    ax2.set_ylim(-5.0, 3.0)
    ax2.grid(True, alpha=0.3)

    ax3.step(bin_centers, np.append(chi2_bin, chi2_bin[-1]),
             where='mid', color='steelblue', lw=1.5, label='Local χ²/ndf')
    ax3.axhline(1.0, color='gray', ls='--', lw=1)
    ax3.set_ylabel("Local χ²/ndf", color='steelblue')
    ax3.tick_params(axis='y', colors='steelblue')
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.step(bin_centers, np.append(cum_chi2_ndf, cum_chi2_ndf[-1]),
              where='mid', color='brown', lw=1.5,
              label='Cumulative χ²/ndf (high→low)')
    ax3b.axhline(chi2_ndf, color='darkred', ls='--', lw=1)
    ax3b.set_ylabel("Cumulative χ²/ndf", color='brown')
    ax3b.tick_params(axis='y', colors='brown')
    ax3.set_ylim(0, 5)
    ax3b.set_ylim(0, 5)

    n_toys    = 1000
    chi2_toys = []
    for _ in range(n_toys):
        toy_data  = np.random.poisson(np.maximum(fit_vals, 1e-12))
        chi2_toy  = np.sum((toy_data - fit_vals) ** 2 / np.maximum(obs_err, 1e-12) ** 2)
        chi2_toys.append(chi2_toy)
    chi2_toys = np.array(chi2_toys)
    p_val     = np.sum(chi2_toys > total_chi2) / len(chi2_toys)

    print(f"\n--- Toy-based goodness-of-fit test ---")
    print(f"Observed χ² = {total_chi2:.2f} for {ndf} dof → χ²/ndf = {chi2_ndf:.3f}")
    print(f"Toy mean χ² = {chi2_toys.mean():.2f} ± {chi2_toys.std():.2f}")
    print(f"p-value = {p_val:.3f}\n")

    textstr = (f"Livetime = {ndays:.2f} days\n"
               f"χ²/ndf = {FULL_CHI2:.2f}/{ndf} = {chi2_ndf:.2f}")
    ax1.text(0.02, 0.9, textstr, transform=ax1.transAxes, fontsize=9,
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax3.set_xlabel("Reconstructed Energy [MeV]")
    plt.tight_layout()
    plt.savefig(f"Plots/fit_residuals_{args_juno.file_prefix}.png", dpi=200)
    plt.close(fig)

    fig2, (ax_cum1, ax_cum2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax_cum1.step(bin_centers, np.append(cum_chi2_ndf, cum_chi2_ndf[-1]),
                 where='mid', color='purple', lw=1.5,
                 label='Cumulative χ² (high→low)')
    ax_cum1.set_ylabel("Cumulative χ²/ ndf")
    ax_cum1.legend(loc="upper left", fontsize=9)
    ax_cum1.grid(True, alpha=0.3)

    ax_cum2.step(bin_centers, np.append(cum_chi2_global_ndf, cum_chi2_global_ndf[-1]),
                 where='mid', color='darkgreen', lw=1.5,
                 label='Global cumulative χ²/ndf')
    ax_cum2.set_xlabel("Reconstructed Energy [MeV]")
    ax_cum2.set_ylabel("Cummulative χ²/global ndf")
    ax_cum2.legend(loc="upper right", fontsize=9)
    ax_cum2.grid(True, alpha=0.3)
    ax_cum1.set_xlim(0.7, 12.0)
    ax_cum2.set_xlim(0.7, 12.0)
    plt.tight_layout()
    plt.savefig("Plots/cumulative_chi2.png", dpi=200)
    plt.close(fig2)

def get_toy_results(m, extra, merrors):
    params = []
    for i in range(len(m.values)):
        params.append(m.values[i])
        if merrors:
            params.append(max(abs(m.merrors[i].upper), abs(m.merrors[i].lower)))
        else:
            params.append(m.errors[i])
    params.extend(extra)
    del m
    return params


def round_errors(param, neg_err, pos_err):
    m_err = neg_err
    p_err = pos_err
    if param == 'sin2_12':
        m_err = round(neg_err, 5); p_err = round(pos_err, 5)
    elif param == 'sin2_13':
        m_err = round(neg_err, 5); p_err = round(pos_err, 5)
    elif param == 'dm2_21':
        m_err = round(neg_err, 7); p_err = round(pos_err, 7)
    elif param == 'dm2_32':
        m_err = round(neg_err, 9); p_err = round(pos_err, 9)
    return m_err, p_err


def plot_profile(m, i, param, plotname):
    x, y, blah = m.mnprofile(param)
    plt.plot(x, y)
    plt.axvline(x=m.values[i], color='black', linestyle='dashed')
    plt.axvspan(m.values[i] + m.merrors[i].lower,
                m.values[i] + m.merrors[i].upper,
                color='gray', alpha=0.3, label='Vertical Band')
    m_err, p_err = round_errors(param, m.merrors[i].lower, m.merrors[i].upper)
    plt.title(f'{param} {m.values[i]} {m_err} + {p_err}')
    plt.ylabel('FCN')
    plt.xlabel(param)
    plt.savefig(plotname)
    plt.close()
