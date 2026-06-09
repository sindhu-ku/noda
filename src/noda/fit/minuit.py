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


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def time_chi2(chi2_func, *, label='chi2'):
    """Wrap a chi² closure to count calls and accumulate wall time."""
    import time, atexit
    state = {'n': 0, 't': 0.0, 'min': float('inf'), 'max': 0.0}
    def wrapped(*p):
        t0 = time.perf_counter()
        r = chi2_func(*p)
        dt = time.perf_counter() - t0
        state['n']  += 1
        state['t']  += dt
        state['min'] = min(state['min'], dt)
        state['max'] = max(state['max'], dt)
        return r
    wrapped.__signature__ = chi2_func.__signature__
    def _report():
        n, t = state['n'], state['t']
        if n == 0: return
        print(f"[{label}] {n} evals, total {t:.2f}s, "
              f"mean {1000*t/n:.2f}ms, min {1000*state['min']:.2f}ms, "
              f"max {1000*state['max']:.2f}ms")
    atexit.register(_report)
    return wrapped
    
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

def write_param_dbl(name, val):
    value = ROOT.TParameter("double")(f"{name}", val)
    value.Write("", ROOT.TObject.kOverwrite)
    
def profile_dm2_31(m, args_juno, NMO_opt=True, write=False):
    import csv
    from scipy.interpolate import CubicSpline
    print("Profiling dm2_31 using built-in mnprofile...")
    if NMO_opt:
        res = m.mnprofile("dm2_31", size=int(args_juno.profile_points), bound=(1.5, 3.5))
    else:
        res = m.mnprofile("dm2_31", size=int(args_juno.profile_points), bound=(-3.5, -1.5))
    dm2_scan_vals = res[0]
    chi2_vals     = res[1]
    delta_chi2    = chi2_vals - np.min(chi2_vals)
    dm2_fine          = np.linspace(dm2_scan_vals.min(), dm2_scan_vals.max(), 500)
    cs                = CubicSpline(dm2_scan_vals, chi2_vals)
    chi2_smooth       = cs(dm2_fine)
    delta_chi2_smooth = chi2_smooth - np.min(chi2_smooth)

    # --- Save raw scan points and smoothed curve to CSV ---
    csv_path = f"Plots/profile_dm2_31_NO_{NMO_opt}_{args_juno.file_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dm2_31_raw", "chi2_raw", "delta_chi2_raw",
                         "dm2_31_smooth", "chi2_smooth", "delta_chi2_smooth"])
        # Pad raw columns with empty strings where lengths differ (raw < 500)
        for i in range(len(dm2_fine)):
            raw_dm2   = dm2_scan_vals[i] if i < len(dm2_scan_vals) else ""
            raw_chi2  = chi2_vals[i]     if i < len(chi2_vals)     else ""
            raw_dchi2 = delta_chi2[i]    if i < len(delta_chi2)    else ""
            writer.writerow([raw_dm2, raw_chi2, raw_dchi2,
                             dm2_fine[i], chi2_smooth[i], delta_chi2_smooth[i]])
    print(f"Saved profile scan to {csv_path}")

    plt.figure(figsize=(6, 5))
    plt.plot(dm2_fine, delta_chi2_smooth, label="Smoothed Profile", lw=2)
    plt.xlabel(r"$\Delta m^2_{31}$ [10$^{-3}$ eV$^2$]", fontsize=14)
    plt.ylabel(r"$\Delta \chi^2$", fontsize=14)
    plt.axhline(1.0, color="green", ls="--", label=r"1$\sigma$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Plots/profile_dm2_31_NO_{NMO_opt}_{args_juno.file_prefix}.png", dpi=300)
    best_index    = np.argmin(delta_chi2)
    best_fit_val  = dm2_scan_vals[best_index]
    if write:
        n_points = len(dm2_fine)
        graph    = ROOT.TGraph(n_points,
                               dm2_fine.astype(float),
                               delta_chi2_smooth.astype(float))
        graph.SetName("dm2_31")
        graph.SetMarkerStyle(2)
        out_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root", "update")
        dir_prof = out_file.mkdir("profiles")
        dir_prof.cd()
        graph.Write()
        out_file.Close()
    best_dm2_from_scan  = dm2_fine[np.argmin(chi2_smooth)]
    print(best_dm2_from_scan)
    m.values["dm2_31"] = best_dm2_from_scan
    print(m.values["dm2_31"])
    m.migrad()

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


def load_prefit_cache(path: str) -> dict:
    """Read a prefit cache produced by run_minuit (save_prefit_path)."""
    z = np.load(path)
    C = z['C_reco']
    return {'nuis_best': z['nuis_best'],
            'nuis_cov':  z['nuis_cov'],
            'C_reco':    None if C.size == 0 else C}

