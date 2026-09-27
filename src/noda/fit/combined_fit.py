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
    +  core_power0 … core_power{n_cores-1}
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

import re


def read_external_osc(path):
    """Read the 4 oscillation parameters (Minuit units: sin²θ₁₂, sin²θ₁₃ raw;
    dm²₂₁ ×1e-5; dm²₃₁ ×1e-3) from a Postfit ROOT written by a previous run.

    Used by `fix_osc_to`: the stat-only refit pins the osc params at these
    EXTERNAL values (e.g. the full all-phase AllP fit) instead of this run's
    own full-fit best fit. Returns {name: value} for the 4 OSC_PARAM_NAMES.
    """
    f = ROOT.TFile(path, "read")
    if f.IsZombie():
        raise FileNotFoundError(f"fix_osc_to: cannot open ROOT file '{path}'")
    d = f.GetDirectory("oscillation_parameters")
    if not d:
        f.Close()
        raise KeyError(f"fix_osc_to: no 'oscillation_parameters' folder in {path}")
    out = {}
    for name in OSC_PARAM_NAMES:
        p = d.Get(name)
        if not p:
            f.Close()
            raise KeyError(f"fix_osc_to: '{name}' not found in {path}")
        out[name] = float(p.GetVal())
    f.Close()
    return out


# Exact ORSA nuisance labels keyed by reactor-summary core name. ORSA's
# labels don't follow a single rule (TPL has a numeric suffix, DYB/FC don't),
# so they're listed explicitly to match ORSA exactly.
ORSA_CORE_LABELS = {
    "YJ1": "YJ-C1", "YJ2": "YJ-C2", "YJ3": "YJ-C3",
    "YJ4": "YJ-C4", "YJ5": "YJ-C5", "YJ6": "YJ-C6",
    "TS1": "TS-C1", "TS2": "TS-C2",
    "DYB": "DYB-C", "TPL": "TPL-C1", "FC": "FCG-C1-4",
}

def _core_display_tag(name):
    """Map a reactor-summary core name to the exact ORSA export label.

    Falls back to a '<plant>-C<n>' / '<name>-C' derivation only for names not
    in ORSA_CORE_LABELS, so an unexpected core never crashes the export.
    """
    if name in ORSA_CORE_LABELS:
        return ORSA_CORE_LABELS[name]
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(name))
    if m:
        return f"{m.group(1)}-C{m.group(2)}"
    return f"{name}-C"

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
                       write_output=False, output_name="", cm=None, eff=None,
                       root_suffix=""):
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
            # Per-shell top-level folder (ORSA layout: juno_inner / juno_outer),
            # bare object names instead of a per-object shell suffix.
            from ..spectra import juno_output_dirname
            jdir = juno_output_dirname(args)
            s_signal.WritetoROOT("reactor", f"{output_file}",
                                 directory=f"{jdir}/postfit")
            s_tot.WritetoROOT("totalpred", f"{output_file}",
                              directory=f"{jdir}/postfit")
        else:
            s_tot.WritetoROOT(f"totalpred{root_suffix}",
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

    residuals = (obs_vals-fit_vals)/ np.maximum(obs_err, 1e-12)

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
    total_cpd = np.sum(fit_vals) / ndays if ndays else 0.0
    ax1.errorbar(bin_centers2, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label=f'{detector_label} data')
    ax1.step(bin_centers, np.append(fit_vals, fit_vals[-1]),
             where='post', color='blue', lw=1.5, label=f'Total Fit: {total_cpd:.2f} cpd')
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
                    spec.WritetoROOT(name_to_key.get(key),
                                     f"{output_file}", f"{jdir}/postfit")
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

# Post-fit background display-name -> short histogram key (matches the per-shell
# names written by plot_fit_residuals_all).
_BCKG_NAME_TO_KEY = {
    "Accidentals": "acc", "Geoneutrinos": "geo",
    r"$^{9}$Li / $^{8}$He": "lihe", "Fast neutrons": "fastn",
    "Double neutrons": "doublen",
    r"$^{13}$C ($\alpha$, n) $^{16}$O": "alphan",
    "Atmospheric neutrinos": "atm", "World reactor neutrinos": "wr",
    r"$^{214}$Bi": "bipo",
}


def _rebin_to(coarse_edges, fine_edges, fine_cont):
    """Sum fine-bin contents into a coarser (aligned, subset-edge) grid.

    Each fine bin is assigned to the coarse bin containing its centre, so this
    works for non-uniform binnings as long as the coarse edges are (close to) a
    subset of the fine edges. Returns an array of len(coarse_edges)-1."""
    coarse_edges = np.asarray(coarse_edges, float)
    centers = 0.5 * (np.asarray(fine_edges, float)[:-1] + np.asarray(fine_edges, float)[1:])
    idx = np.searchsorted(coarse_edges, centers, side='right') - 1
    out = np.zeros(len(coarse_edges) - 1)
    for i, c in zip(idx, np.asarray(fine_cont, float)):
        if 0 <= i < len(out):
            out[i] += c
    return out


def plot_postfit_summed(shell_results, args,
                        *,
                        detector_label="JUNO",
                        signal_label=None,
                        chi2_value=None,
                        ndf=None,
                        write_output=False,
                        output_name=""):
    """
    Plot the post-fit spectra *summed over multiple shells* (one figure for
    the whole detector instead of one per shell).

    This is deliberately a separate function from ``plot_fit_residuals_all``:
    it consumes the per-shell post-fit spectra already computed in the replay
    loop and stacks them onto a common reco grid.

    shell_results : list of dicts, one per shell bundle, each holding
        {'s_tot', 's_obs', 's_signal', 'bckgs', 'cm', 'ndays',
         'phase', 'shell'}
        as returned by the JUNO best-fit replay.  All shells are assumed to
        share the same reco binning (they do in the AllP configs); any shell
        whose bins differ from the first is skipped with a warning.
    chi2_value : total fit χ² shown in the textbox (defaults to FULL_CHI2).
    """
    if not shell_results:
        print("[plot] plot_postfit_summed: no shell results — nothing to do")
        return
    if chi2_value is None:
        chi2_value = FULL_CHI2

    default_sig_label = {
        "JUNO": r"Reactor $\bar{\nu}_e$ expectation",
        "TAO":  r"TAO reactor $\bar{\nu}_e$",
        "DYB":  r"Daya Bay reactor $\bar{\nu}_e$",
    }
    if signal_label is None:
        signal_label = default_sig_label.get(detector_label,
                                             f"{detector_label} signal")

    # ── reference grid from the first shell ──────────────────────────────
    ref          = shell_results[0]
    ref_bins     = ref['s_tot'].bins
    n_bins       = len(ref['s_tot'].bin_cont)
    bin_centers  = ref_bins
    bin_centers2 = 0.5 * (ref_bins[:-1] + ref_bins[1:])

    # ── accumulators ─────────────────────────────────────────────────────
    # Shells differ in livetime, so cpd (counts/day) cannot be recovered from
    # the summed counts with a single ndays — accumulate cpd per shell as
    # Σ(counts_shell / ndays_shell), while the plotted Events are summed counts.
    obs_vals = np.zeros(n_bins)
    fit_vals = np.zeros(n_bins)
    sig_vals = np.zeros(n_bins)
    cov_sum  = np.zeros((n_bins, n_bins))
    bckg_sum = {}                       # key -> summed bin_cont
    fit_cpd  = 0.0
    sig_cpd  = 0.0
    bckg_cpd = {}                       # key -> summed cpd
    used     = 0

    for r in shell_results:
        bins = r['s_tot'].bins
        if (len(bins) != len(ref_bins)
                or not np.allclose(bins, ref_bins, rtol=0, atol=1e-9)):
            print(f"[plot] plot_postfit_summed: skipping shell "
                  f"{r.get('phase')}/{r.get('shell')} — reco binning "
                  f"differs from reference, cannot sum")
            continue
        nd = r['ndays']
        obs_vals += r['s_obs'].bin_cont
        fit_vals += r['s_tot'].bin_cont
        fit_cpd  += np.sum(r['s_tot'].bin_cont) / nd if nd else 0.0
        if r.get('s_signal') is not None:
            sig_vals += r['s_signal'].bin_cont
            sig_cpd  += np.sum(r['s_signal'].bin_cont) / nd if nd else 0.0
        if r.get('cm') is not None:
            cov_sum += r['cm'].data
        for key, spec in (r.get('bckgs') or {}).items():
            bckg_sum.setdefault(key, np.zeros(n_bins))
            bckg_sum[key] += spec.bin_cont
            bckg_cpd[key] = bckg_cpd.get(key, 0.0) + (
                np.sum(spec.bin_cont) / nd if nd else 0.0)
        used += 1

    if used == 0:
        print("[plot] plot_postfit_summed: no shells shared the reference "
              "binning — nothing plotted")
        return

    # data error: summed shell covariance + Poisson on the summed counts
    obs_err = np.sqrt(np.maximum(np.diag(cov_sum), 0.0))
    if detector_label != "DYB":
        obs_err = obs_err + np.sqrt(np.maximum(obs_vals, 1e-12))
    obs_err = np.maximum(obs_err, 1e-12)

    residuals = (obs_vals - fit_vals) / obs_err
    chi2_bin  = (obs_vals - fit_vals) ** 2 / obs_err ** 2
    if ndf is None:
        ndf = max(1, n_bins - FREE_PARAM)
    chi2_ndf = float(chi2_value) / ndf

    os.makedirs("Plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 7),
        gridspec_kw={'height_ratios': [3, 1]})

    # ── top panel: summed data + fit + signal + backgrounds ──────────────
    ax1.errorbar(bin_centers2, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label=f'{detector_label} data (Σ shells)')
    ax1.step(bin_centers, np.append(fit_vals, fit_vals[-1]),
             where='post', color='blue', lw=1.5,
             label=f'Total Fit: {fit_cpd:.2f} cpd')
    if np.any(sig_vals):
        ax1.step(bin_centers, np.append(sig_vals, sig_vals[-1]),
                 where='post', color='red', lw=1.2,
                 label=f'{signal_label}: {sig_cpd:.2f} cpd')

    colors = plt.cm.tab10.colors
    for i, (key, cont) in enumerate(bckg_sum.items()):
        ax1.step(bin_centers, np.append(cont, cont[-1]),
                 where='post', color=colors[i % len(colors)], lw=1,
                 label=f'{key}: {bckg_cpd.get(key, 0.0):.2f} cpd')

    ax1.set_ylabel("Events")
    ax1.set_title(f"{detector_label} — summed over {used} shell(s)",
                  fontsize=10)
    leg = ax1.legend(fontsize=9, loc="upper right", frameon=True,
                     edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bin_centers[0], bin_centers[-1])

    shells_txt = ", ".join(f"{r.get('phase')}/{r.get('shell')}"
                           for r in shell_results)
    textstr = (f"{detector_label} (Σ {used} shells)\n"
               f"χ²/ndf = {chi2_value:.2f}/{ndf} = {chi2_ndf:.2f}\n"
               f"shells: {shells_txt}")
    ax1.text(0.02, 0.88, textstr, transform=ax1.transAxes, fontsize=8,
             va='center', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── residuals panel ──────────────────────────────────────────────────
    ax2.errorbar(bin_centers2, residuals, yerr=np.ones_like(residuals),
                 fmt='o', color='blue', markersize=3)
    ax2.step(bin_centers, np.append(residuals, residuals[-1]),
             where='post', color='blue', lw=1)
    ax2.axhline(0, color='k', lw=1)
    for nsigma, alpha in zip([1, 2, 3], [0.25, 0.15, 0.1]):
        ax2.fill_between(bin_centers, -nsigma, nsigma, color='blue', alpha=alpha)
    ax2.set_ylabel("Residuals [σ]")
    ax2.set_ylim(-5.0, 5.0)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Reconstructed Energy [MeV]")

    plt.tight_layout()
    if output_name:
        figname = f"Plots/{output_name}.png"
    else:
        figname = (f"Plots/postfit_summed_{args.file_prefix}_"
                   f"{detector_label}.png")
    plt.savefig(figname, dpi=200)
    plt.close(fig)
    print(f"[plot] saved {figname}  (summed over {used} shells)")

    if write_output:
        output_file = (f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}"
                       f"_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root")
        # The whole-detector sum is a NODA-only diagnostic with no ORSA
        # equivalent; keep it in a clearly-named top-level folder so it does
        # not collide with the per-shell juno_inner / juno_outer dirs.
        sum_dir = "juno_summed" if detector_label == "JUNO" else detector_label.lower()

        # Shells can sit on different reco grids (e.g. inner finer than outer).
        # Pick the COARSEST grid and rebin every (finer) shell down to it, so the
        # shells sum without double counting. This is what makes the summed
        # totalpred/reactor/backgrounds true inner+outer sums rather than the
        # first shell only.
        coarse = min((r['s_tot'].bins for r in shell_results), key=len)

        def _sum_shells(getter):
            tot = np.zeros(len(coarse) - 1)
            for r in shell_results:
                spec = getter(r)
                if spec is None:
                    continue
                b = spec.bins
                if len(b) == len(coarse) and np.allclose(b, coarse, rtol=0, atol=1e-9):
                    tot = tot + spec.bin_cont
                else:
                    tot = tot + _rebin_to(coarse, b, spec.bin_cont)
            return tot

        Spectrum(bin_cont=_sum_shells(lambda r: r['s_tot']), bins=coarse).WritetoROOT(
            "totalpred", output_file, directory=f"{sum_dir}/postfit")
        if any(r.get('s_signal') is not None for r in shell_results):
            Spectrum(bin_cont=_sum_shells(lambda r: r.get('s_signal')), bins=coarse).WritetoROOT(
                "reactor", output_file, directory=f"{sum_dir}/postfit")
        # Summed observed data at {sum_dir}/data (mirrors the per-shell
        # juno_<shell>/data location); only when fitting real data, as per shell.
        if getattr(args, 'is_data', False):
            Spectrum(bin_cont=_sum_shells(lambda r: r.get('s_obs')), bins=coarse).WritetoROOT(
                "data", output_file, directory=sum_dir)

        # Summed background components, written like totalpred/reactor. JUNO only
        # (its bckgs is a {display-name: Spectrum} dict; TAO/DYB differ).
        if detector_label == "JUNO":
            bkeys = []
            for r in shell_results:
                for k in (r.get('bckgs') or {}):
                    if k not in bkeys:
                        bkeys.append(k)
            n_written = 0
            for k in bkeys:
                short = _BCKG_NAME_TO_KEY.get(k)
                if not short:
                    continue
                cont = _sum_shells(lambda r, k=k: (r.get('bckgs') or {}).get(k))
                Spectrum(bin_cont=cont, bins=coarse).WritetoROOT(
                    short, output_file, directory=f"{sum_dir}/postfit")
                n_written += 1
            print(f"[write] {sum_dir}/postfit: summed totalpred, reactor + "
                  f"{n_written} backgrounds on the {len(coarse) - 1}-bin grid")

            # Combined livetime + efficiency into juno_summed (mirrors the
            # per-shell juno_<shell>/LivetimeDays and postfit/Efficiency). The
            # shells share the SAME livetime (same exposure), so it is taken
            # once -- not summed. Each shell's efficiency is its fraction of the
            # whole detector, so the combined detector efficiency is their SUM.
            lts  = [r['ndays'] for r in shell_results if r.get('ndays')]
            effs = [r['eff_postfit'] for r in shell_results
                    if r.get('eff_postfit') is not None]
            wf = ROOT.TFile(output_file, "update")
            d  = wf.GetDirectory(sum_dir) or wf.mkdir(sum_dir)
            if lts:
                lt = lts[0]
                if not all(abs(x - lt) < 1e-6 for x in lts):
                    print(f"[write] WARNING: shells have different livetimes "
                          f"{lts}; storing the first in {sum_dir}/LivetimeDays")
                d.cd()
                write_param_dbl("LivetimeDays", float(lt))
            if effs:
                dpost = d.GetDirectory("postfit") or d.mkdir("postfit")
                dpost.cd()
                write_param_dbl("Efficiency", float(sum(effs)))
            force_vector_streamers(wf)
            wf.Close()
            print(f"[write] {sum_dir}: LivetimeDays={lts[0] if lts else '--'}, "
                  f"postfit Efficiency={sum(effs) if effs else '--'}")

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
            tao_kwargs = dict(
                Nrea_tao=tao_params_dict['Nrea_tao'],
                snf_noneq_norm_tao=[tao_params_dict[f'snf_noneq_norm_tao{i}']
                                    for i in range(2)],
                abc_tao=[tao_params_dict['abc_tao0'],
                         tao_params_dict['abc_tao1'],
                         tao_params_dict['abc_tao2']],
                core_power_tao=tao_params_dict['core_power_tao'],
                eff_nuis_tao=tao_params_dict['eff_nuis_tao'],
                bckg_rate=tao_params_dict['bckg_rate'],
                unc=unc_tao, nuisance=nuisance_vals)
            if 'NL_tao0' in tao_params_dict:
                tao_kwargs['NL_tao'] = [tao_params_dict[f'NL_tao{i}'] for i in range(6)]
            (chi2_tao, _, _s, _st, _, _) = tao_det.chi2_contribution(**tao_kwargs)


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

    # The old-style TAO NL sector floats 5 near-collinear shape pulls
    # (positronNLpull0-4 correlate up to 0.99) plus the LY energy scale. With
    # Minuit's default strategy 1 the coarse finite-difference Hessian over that
    # ill-conditioned (cond ~1e7) sub-space comes out non-positive-definite, so
    # Hesse reports inaccurate errors (pull σ > 1, and the covariance is not
    # trustworthy). Strategy 2 uses more accurate derivatives and recovers an
    # accurate, positive-definite covariance. Only needed when TAO actually
    # floats the NL pulls (nl_in_leak False); the baked-in path has no NL sector.
    if tao_det is not None and not getattr(args_tao, "nl_in_leak", False):
        m_pre.strategy = 2
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
        tao_kwargs = dict(
            Nrea_tao=m_pre.values['Nrea_tao'],
            snf_noneq_norm_tao=[m_pre.values[f'snf_noneq_norm_tao{i}']
                                for i in range(2)],
            abc_tao=[m_pre.values['abc_tao0'],
                     m_pre.values['abc_tao1'],
                     m_pre.values['abc_tao2']],
            core_power_tao=m_pre.values['core_power_tao'],
            eff_nuis_tao=m_pre.values['eff_nuis_tao'],
            bckg_rate=m_pre.values['bckg_rate'],
            unc=unc_tao, nuisance=nuis_best)
        if 'NL_tao0' in m_pre.parameters:
            tao_kwargs['NL_tao'] = [m_pre.values[f'NL_tao{i}'] for i in range(6)]
        (chi2_tao, _, s_obs_tao, s_tot_tao, bckg, cm_tao) = \
            tao_det.chi2_contribution(**tao_kwargs)
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

    if args_juno.plot_correlation:
        import re
        corr = np.array(m_pre.covariance.correlation())

        labels = list(m_pre.parameters)

        import seaborn as sns
        plt.figure(figsize=(12, 12))          # smaller figure → smaller squares
        ax = sns.heatmap(corr,
                         annot=False,          # remove the numbers
                         xticklabels=labels, yticklabels=labels,
                         cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                         square=True)          # force cells to be actual squares
        ax.tick_params(labelsize=16)           # bigger axis labels
        plt.tight_layout()

        plt.savefig(f"Plots/corr_{args_juno.file_prefix}_corr_TAO.png", dpi=200)

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

from .dispatcher import (DetectorBundle, build_chi2,
                         apply_all_limits, collect_first_fix)
from .param_policy import ParamPolicy


def run_minuit_bundled(bundles, policy, args_juno, *,
                       prefit_cache=None,        # dict {'nuis_best','nuis_cov','C_reco'}
                       prefit_only=False,        # return after prefit, skip JUNO fit
                       return_prefit=False,      # return (m, prefit_dict) instead of m
                       return_cost_fn=False,
                       fix_dm2_31_to=None,       # in 1e-3 eV² (Minuit convention)
                       toy_dm2_31=None,
                       true_dm2_31=None,
                       toy_quiet=False):         # silence ALL post-fit output
    """
    Multi-bundle entry point. Mirrors the three fit modes of the original
    run_minuit, including on-the-fly DYB+TAO prefit generation when no
    cache is provided, and full post-fit reporting.
    """


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
                    ibd_prefit=(dyb_bundle.detector.ensp_nom['ribd']
                                if dyb_bundle else None))

        # Per-bundle segment→reco covariance (prefit_wosegs). Each JUNO bundle
        # may have its own reco binning (e.g. different shells), so C_reco must
        # be built from that bundle's own detector / response, not shared.
        def _compute_C_reco_for(b):
            det    = b.detector
            a_b    = b.args
            pnames = det.get_param_names()
            n_NL   = sum(1 for p in pnames if p.startswith("NL"))
            n_rea  = sum(1 for p in pnames if p.startswith("snf_noneq_norm"))
            n_core = sum(1 for p in pnames if p.startswith("core_power"))
            bckgs_for_jp = {key: {"pull": 0.0,
                                  "sigma": info["rate_unc"],
                                  "b2b":   info["b2b_unc"],
                                  "name":  info["name"],
                                  "b2b_scale": info["b2b_scale"]}
                            for key, info in b.backgrounds["JUNO"].items()}
            jp_fixed = dict(
                Nrea=0.0,
                NL=([a_b.kB, a_b.fC, a_b.kI, a_b.LY_shared]
                    if a_b.NL != "pull" else [0.0] * n_NL),
                abc=[a_b.a, a_b.b, a_b.c],
                bckgs=bckgs_for_jp,
                snf_noneq_norm=[0.0] * n_rea,
                eff_nuis=0.0,
                ThU_ratio=a_b.Th_U_ratio,
                core_powers_delta=[0.0] * n_core,
                efission=list(np.array(a_b.efission)*1e-2))
            return propagate_nuis_to_reco(
                juno_det=det,
                nuis_best=nuis_best, nuis_cov=nuis_cov,
                sin2_12=nuosc.op_nom["sin2_th12"],
                sin2_13=nuosc.op_nom["sin2_th13"],
                dm2_21=nuosc.op_nom["dm2_21"],
                dm2_31=nuosc.op_nom["dm2_31"],
                me_rho=args_juno.me_rho,
                fixed_juno_params=jp_fixed,
                args_juno=args_juno, unc_juno=b.unc_key)

        # ── 3. prefit_wosegs: compute C_reco (b0) if missing from cache ──────
        if fit_mode == 'prefit_wosegs' and C_reco is None:
            if not juno_bundles_all:
                raise ValueError(
                    "prefit_wosegs requires at least one JUNO bundle to "
                    "compute C_reco via propagate_nuis_to_reco.")
            b0 = juno_bundles_all[0]
            if not args_juno.silent_print:
                print("[bundled] computing C_reco via propagate_nuis_to_reco "
                      f"(using JUNO bundle {b0.phase}/{b0.shell})")
            C_reco = _compute_C_reco_for(b0)

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
        # A single cached/precomputed C_reco only matches one binning; with
        # several JUNO shells (different binning) recompute per bundle so the
        # external covariance matches each detector's reco-bin count.
        if fit_mode == 'prefit_wosegs':
            multi = len(juno_bundles_all) > 1
            for b in bundles:
                if b.kind != 'juno':
                    continue
                cov_b = (_compute_C_reco_for(b)
                         if (multi or C_reco is None) else C_reco)
                # [TEST] optionally replace each shell's reco covariance with an
                # externally-supplied one (ORSA's DYB_TAO_JUNO_covariance_*.txt),
                # matched to the shell by bin count. Env-gated, no yaml change.
                import os as _os
                _ext_map = {int(_os.environ.get('NODA_EXT_COV_INNER_NBINS', '168')):
                                _os.environ.get('NODA_EXT_COV_INNER'),
                            int(_os.environ.get('NODA_EXT_COV_OUTER_NBINS', '84')):
                                _os.environ.get('NODA_EXT_COV_OUTER')}
                _n = getattr(cov_b, 'shape', (0,))[0]
                _ext = _ext_map.get(_n)
                if _ext:
                    _O = np.loadtxt(_ext)
                    if _O.shape == cov_b.shape:
                        print(f"[EXT-COV] shell nbins={_n}: NODA diag mean="
                              f"{np.mean(np.diag(cov_b)):.4g} -> ORSA diag mean={np.mean(np.diag(_O)):.4g}; "
                              f"using {_ext.split('/')[-1]}")
                        cov_b = _O
                    else:
                        print(f"[EXT-COV] shape mismatch {_O.shape} vs {cov_b.shape}; keeping NODA")
                b.detector.set_external_cov(cov_b)

    # ── 8. Build chi² closure ────────────────────────────────────────────
    chi2_func, names, starts, routing = build_chi2(bundles, policy, args_juno,
                                                   toy_dm2_31=toy_dm2_31)
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

        # ── 9b. Efficiency inter-shell correlation (ORSA multivariate_gauss) ─
        # Couple the per-shell eff_nuis with a joint Gaussian prior whose
        # covariance Σ has diagonal = eff_unc_shell² and off-diagonal =
        # eff_corr_unc² (the shared "target proton" variance). build_spectrum
        # already emits the per-shell diagonal (eff_nuis/eff_unc)², so here we
        # add only the extra coupling Σ⁻¹ − diag(1/eff_unc²); the net efficiency
        # penalty is then Δ·Σ⁻¹·Δ. Equivalent to a shared target-proton
        # nuisance, but expressed as a covariance prior like ORSA does.
        if getattr(args_juno, 'eff_corr_split', False):
            juno_bs = [b for b in bundles if b.kind == 'juno']
            eff_mangled, eff_unc = [], []
            for b in juno_bs:
                mn = routing[id(b)].get('eff_nuis')
                if mn is not None and mn not in eff_mangled:
                    eff_mangled.append(mn)
                    eff_unc.append(float(b.args.eff_unc))
            if len(eff_mangled) >= 2:
                ecu   = float(getattr(args_juno, 'eff_corr_unc', 0.01))
                Sig   = np.full((len(eff_mangled), len(eff_mangled)), ecu ** 2)
                for i, su in enumerate(eff_unc):
                    Sig[i, i] = su ** 2
                coupling = np.linalg.inv(Sig) - np.diag([1.0 / su ** 2 for su in eff_unc])
                eff_idx  = [names.index(mn) for mn in eff_mangled]
                _base_eff = chi2_func
                _sig_eff  = chi2_func.__signature__
                def _chi2_eff_cov(*flat, _b=_base_eff, _M=coupling, _idx=eff_idx):
                    v = _b(*flat)
                    d = np.array([flat[i] for i in _idx])
                    return v + float(d @ _M @ d)
                _chi2_eff_cov.__signature__ = _sig_eff
                chi2_func = _chi2_eff_cov
                if not args_juno.silent_print:
                    print(f"[bundled] efficiency inter-shell covariance prior on "
                          f"{eff_mangled}, eff_unc={eff_unc}, off-diag "
                          f"eff_corr_unc={ecu} (ORSA multivariate_gauss form)")

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

    # Minuit strategy 0 = fewest chi² evaluations during minimisation (no
    # careful numerical Hessian between steps). The minimum location is
    # unchanged; accurate parameter errors still come from the explicit
    # m.hesse() below. This is the main speed lever for the many-parameter
    # multi-shell fit and for FC toys (which don't run hesse at all). Override
    # with `minuit_strategy: 1` (or 2) in the config if convergence regresses.
    #m.strategy = int(getattr(args_juno, 'minuit_strategy', 0))

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

    m.strategy =2 
    m.migrad()
    migrad_time = datetime.now()

    if args_juno.run_hesse and not toy_quiet:
        m.hesse()
    hesse_time = datetime.now()

    # ── 17. FC short-circuit ─────────────────────────────────────────────
    if toy_quiet:
        if return_prefit:
            return m, {'nuis_best': nuis_best,
                       'nuis_cov':  nuis_cov,
                       'C_reco':    C_reco}
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
    # Per-JUNO-bundle ROOT folders: id(bundle) -> (juno_dir, nuisances, postfit).
    # Each shell gets its OWN self-contained top-level juno_<shell> folder
    # (ORSA layout) carrying the FULL nuisance set, so shared parameters are
    # duplicated with identical values across shells.
    juno_shell_dirs = {}
    from ..spectra import juno_output_dirname
    if args_juno.write_output:
        output_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root",
            "update")
        chi2_ndof_val = (m.fval / len(b_first.detector.ensp_nom['rdet'].bins) - 4
                         if b_first is not None else 0.0)
        # "update" mode reuses an existing Postfit file, so these dirs may
        # already exist from a previous run — mkdir() returns nullptr in that
        # case, which later crashes on .cd(). Reuse the existing dir if present.
        for b in juno_bundles_all:
            jname = juno_output_dirname(b.args)
            jd = output_file.GetDirectory(jname) or output_file.mkdir(jname)
            jd.cd()
            write_param_dbl("chi2/ndof", chi2_ndof_val)
            nd = jd.GetDirectory("nuisances") or jd.mkdir("nuisances")
            pd = jd.GetDirectory("postfit") or jd.mkdir("postfit")
            juno_shell_dirs[id(b)] = (jd, nd, pd)
    try:
        if not args_juno.silent_print:
            print("************Other parameter fit results************")
            for i in range(len(SHARED_PARAM_NAMES), len(m.values)):
                print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")

            # ── Background rates per JUNO bundle ─────────────────────────────
            # Each shell's full nuisance set is written into its OWN
            # juno_<shell>/nuisances folder with bare names; shared parameters
            # (same mangled Minuit name across bundles) are duplicated into
            # every shell folder with identical values (ORSA convention).
            print("************Background rates************")
            multi = len(juno_bundles_all) > 1

            def _bkg_start_fit(b, key, info):
                """(start_rate, fitted_rate) for one background in one shell,
                including the detection-efficiency factor for external fluxes."""
                nuis    = m.values[routing[id(b)][f'bkg_{key}']]
                eff_fac = 1.0 + m.values[routing[id(b)]['eff_nuis']]
                if key == "geo":
                    sr = (float(b.args.geo_tnu_rate) * float(b.args.tnu_to_cpd)
                          * float(b.args.detector_efficiency))
                    return sr, sr * (1 + nuis) * eff_fac
                if key in ("wr", "atm"):
                    sr = info["rate"] * float(b.args.detector_efficiency)
                    return sr, sr * (1 + nuis) * eff_fac
                sr = info["rate"]
                return sr, sr * (1 + nuis)

            # A background is "shell-correlated" when its Minuit parameter is the
            # SAME mangled name across all shells (correlate_shell: true). Those
            # are reported once as a combined total over shells; the rest per
            # shell. The per-shell ROOT nuisance write below is unchanged.
            for key in juno_bundles_all[0].backgrounds['JUNO'].keys():
                info0    = juno_bundles_all[0].backgrounds['JUNO'][key]
                rate_unc = info0.get("rate_unc", None)
                mangs    = {routing[id(bb)][f'bkg_{key}'] for bb in juno_bundles_all}
                shared   = multi and len(mangs) == 1
                if shared:
                    b0       = juno_bundles_all[0]
                    nuis_err = m.errors[routing[id(b0)][f'bkg_{key}']]
                    sr_tot = rf_tot = 0.0
                    for bb in juno_bundles_all:
                        sr, rf = _bkg_start_fit(bb, key, bb.backgrounds['JUNO'][key])
                        sr_tot += sr; rf_tot += rf
                    rate_err = sr_tot * nuis_err
                    dev = ((rf_tot - sr_tot) / (sr_tot * rate_unc)
                           if rate_unc else float('nan'))
                    print(f"[total] {key}: {rf_tot:.4f} ± {rate_err:.4f} cpd "
                          f"(rel unc: {rate_err/rf_tot:.3%}; deviation: {dev:.2f} σ)")
                else:
                    for b in juno_bundles_all:
                        tag_pre  = f"[{b.phase}/{b.shell}] " if multi else ""
                        info     = b.backgrounds['JUNO'][key]
                        ru       = info.get("rate_unc", None)
                        nuis_err = m.errors[routing[id(b)][f'bkg_{key}']]
                        sr, rf   = _bkg_start_fit(b, key, info)
                        rate_err = sr * nuis_err
                        dev = ((rf - sr) / (sr * ru) if ru else float('nan'))
                        print(f"{tag_pre}{key}: {rf:.4f} ± {rate_err:.4f} cpd "
                              f"(rel unc: {rate_err/rf:.3%}; deviation: {dev:.2f} σ)")

            # ROOT nuisance write — per shell, unchanged (ORSA convention: shared
            # params are duplicated identically into each shell folder).
            if args_juno.write_output:
                for b in juno_bundles_all:
                    _, dir_nuis, _ = juno_shell_dirs[id(b)]
                    dir_nuis.cd()
                    for key, info in b.backgrounds['JUNO'].items():
                        mangled  = routing[id(b)][f'bkg_{key}']
                        nuis     = m.values[mangled]
                        nuis_err = m.errors[mangled]
                        rate_unc = info.get("rate_unc", None)
                        if rate_unc != 0.0:
                            write_parameter(f"{key}", nuis/rate_unc, nuis_err/rate_unc)
                        else:
                            write_parameter(f"{key}", nuis, nuis_err)

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
                    _, dir_nuis, _ = juno_shell_dirs[id(b)]
                    dir_nuis.cd()
                    write_parameter("Rflux", m.values[n_name]/b.detector.args.r2_unc, m.errors[n_name]/b.detector.args.r2_unc)
                    write_parameter("eff_pull", m.values[e_name]/b.detector.args.eff_unc, m.errors[e_name]/b.detector.args.eff_unc)

            # ── Shared (osc + me_rho) ────────────────────────────────────────
            print("************Shared parameters************")
            print(f"me_rho: {m.values['me_rho']} +/- {m.errors['me_rho']}")

            # ── JUNO detector response (NL + abc) ────────────────────────────
            # abc/NL may be shared (one Minuit param) or uncorrelated per
            # phase/shell (a distinct mangled param per bundle). Print one block
            # per JUNO bundle, deduping by mangled name so the shared case still
            # prints a single block.
            if b_first is not None:
                print("************JUNO Detector response************")
                resp_labels = [("a", "abc0"), ("b", "abc1"), ("c", "abc2"),
                               ("kB", "NL0"), ("fC", "NL1"),
                               ("kI", "NL2"), ("LY", "NL3")]
                printed_resp: set[str] = set()
                for b in juno_bundles_all:
                    mp = routing[id(b)]
                    # Skip if this bundle's response params were already printed
                    # (i.e. they resolve to the same shared Minuit names).
                    sig = tuple(mp[base] for _, base in resp_labels)
                    if sig in printed_resp:
                        continue
                    printed_resp.add(sig)
                    if len(juno_bundles_all) > 1:
                        print(f"--- [{b.phase}/{b.shell}] ---")
                    for tag, base in resp_labels:
                        name = mp[base]
                        print(f"{tag}: {m.values[name]}  +/- {m.errors[name]}")

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
            # One full block per shell, into that shell's own nuisances folder.
            # Response (a,b,c,kB,fC,kI,LY) is free per shell so each bundle's
            # routing yields its own values; me_rho/ratio/cores are shared and
            # therefore duplicated with identical values across shells.
            if args_juno.write_output and b_first is not None:
                det_params = [("a","abc0"), ("b","abc1"), ("c","abc2"),
                              ("kB","NL0"), ("fC","NL1"), ("kI","NL2"), ("LY","NL3")]

                for b in juno_bundles_all:
                    _, dir_nuis, _ = juno_shell_dirs[id(b)]
                    dir_nuis.cd()
                    mp = routing[id(b)]
                    sigma_prior = np.sqrt(np.diag(b.detector.covariance_er))

                    for i, (tag, base) in enumerate(det_params):
                        name    = mp[base]
                        nominal = m.init_params[name].value
                        dev     = (m.values[name] - nominal) / sigma_prior[i]   # element0 (pull)
                        rel_unc =  m.errors[name]            / sigma_prior[i]   # element1 (err in σ units)
                        write_parameter(tag, dev, rel_unc)

                    write_parameter('me_rho', (m.values['me_rho'] - b.detector.args.me_rho) / b.detector.args.me_rho_scale, m.errors['me_rho']/b.detector.args.me_rho_scale)
                    write_parameter('ratio', (m.values['ThU_ratio'] - b.detector.args.Th_U_ratio) / b.detector.args.Th_U_ratio_err, m.errors['ThU_ratio']/b.detector.args.Th_U_ratio_err)
                    core_names = b.detector.core_names
                    if core_names is None:
                        core_names = [f"core{i}" for i in range(b.detector.n_cores)]
                    for i, core_name in enumerate(core_names):
                        tag  = _core_display_tag(core_name)
                        name = mp[f"core_power{i}"]
                        write_parameter(tag, m.values[name]/b.detector.args.core_flux_unc, m.errors[name]/b.detector.args.core_flux_unc)


        if args_juno.write_output and output_file is not None and b_first is not None:
            from ..spectra import _build_nl_graph, _build_resolution_graph
            # Per-shell post-fit Efficiency + NL/resolution curves, into each
            # shell's own juno_<shell>/{postfit,response/postfit}. eff_nuis and
            # the response params are uncorrelated per (phase, shell), so route
            # each through that bundle's own mapping.
            for b in juno_bundles_all:
                jd, _, dir_post = juno_shell_dirs[id(b)]
                mp = routing[id(b)]
                dir_post.cd()
                write_param_dbl("Efficiency",
                                b.args.detector_efficiency
                                * (1 + m.values[mp['eff_nuis']]))

                resp_dir = jd.GetDirectory("response") or jd.mkdir("response")
                resp_post = resp_dir.GetDirectory("postfit") or resp_dir.mkdir("postfit")
                resp_post.cd()
                if args_juno.NL == "pull":
                    nl_graph = _build_nl_graph(
                        b.detector.ensp_nom, b.args,
                        NL_pulls=[m.values[mp[f'NL{i}']] for i in range(5)])
                else:
                    nl_graph = _build_nl_graph(
                        None, b.args,
                        kB=m.values[mp['NL0']],
                        fC=m.values[mp['NL1']],
                        kI=m.values[mp['NL2']],
                        LY_shared=m.values[mp['NL3']])
                nl_graph.Write("", ROOT.TObject.kOverwrite)
                _build_resolution_graph(
                    m.values[mp['abc0']],
                    m.values[mp['abc1']],
                    m.values[mp['abc2']]).Write("", ROOT.TObject.kOverwrite)

            # ── TAO post-fit response curves (tao/response/postfit) ─────────
            # In combined mode the TAO parameters live in the JUNO Minuit object
            # (mangled via routing); in the prefit modes they come from m_pre.
            if use_tao and tao_bundle is not None:
                from ..spectra import _build_nl_graph, _build_resolution_graph

                def tao_val(pname):
                    mangled = routing[id(tao_bundle)].get(pname, pname) \
                              if id(tao_bundle) in routing else pname
                    if fit_mode == "combined" and mangled in m.parameters:
                        return m.values[mangled]
                    if m_pre is not None and pname in m_pre.parameters:
                        return m_pre.values[pname]
                    return None    # cached prefit — best-fit values unavailable

                tao_nl_in_leak = getattr(tao_bundle.detector.args, "nl_in_leak", False)
                # When NL is baked into the leakage matrix there is no post-fit
                # NL curve to draw (and no NL_tao params); only write resolution.
                tao_NL  = None if tao_nl_in_leak else [tao_val(f"NL_tao{i}") for i in range(6)]
                tao_abc = [tao_val(f"abc_tao{i}") for i in range(3)]
                if None not in tao_abc and (tao_nl_in_leak or None not in tao_NL):
                    tao_dir = (output_file.GetDirectory("tao")
                               or output_file.mkdir("tao"))
                    resp_dir_tao = (tao_dir.GetDirectory("response")
                                    or tao_dir.mkdir("response"))
                    resp_post_tao = (resp_dir_tao.GetDirectory("postfit")
                                     or resp_dir_tao.mkdir("postfit"))
                    resp_post_tao.cd()
                    if not tao_nl_in_leak:
                        _build_nl_graph(tao_bundle.detector.ensp_nom,
                                        tao_bundle.detector.args,
                                        NL_pulls=tao_NL).Write("", ROOT.TObject.kOverwrite)
                    _build_resolution_graph(*tao_abc).Write("", ROOT.TObject.kOverwrite)

        if args_juno.write_output and args_juno.fit_type == "PMOP_combined" and not args_juno.use_prefit:
            dyb_dir = output_file.GetDirectory("dyb") or output_file.mkdir("dyb")
            dyb_dir.cd()
            dir_nuis_dyb = dyb_dir.GetDirectory("nuisances") or dyb_dir.mkdir("nuisances")
            dir_nuis_dyb.cd()
            for i in range(n_nuis):
                if fit_mode == 'prefit_wosegs':
                    write_parameter(f"rea_shape_f{i}", m_pre.values[f"nuis{i}"], m_pre.errors[f"nuis{i}"])
                else:
                    write_parameter(f"rea_shape_f{i}", m.values[f"nuis{i}"], m.errors[f"nuis{i}"])

        # ── TAO nuisances (tao/nuisances) — ORSA convention: TVectorD(2) of
        #    [pull, err] in prior-σ units, same names as the ORSA export ────────
        if (args_juno.write_output and use_tao and tao_bundle is not None
                and args_juno.fit_type == "PMOP_combined"):
            args_tao_w = tao_bundle.detector.args

            def tao_val_err(pname):
                mangled = routing[id(tao_bundle)].get(pname, pname) \
                          if id(tao_bundle) in routing else pname
                if fit_mode == "combined" and mangled in m.parameters:
                    return m.values[mangled], m.errors[mangled]
                if m_pre is not None and pname in m_pre.parameters:
                    return m_pre.values[pname], m_pre.errors[pname]
                return None

            # (output name, Minuit name, nominal, prior sigma)
            tao_export = [("Eff",   "eff_nuis_tao",        0.0, args_tao_w.eff_unc),
                          ("Rflux", "Nrea_tao",            0.0, args_tao_w.r2_unc),
                          ("TS1",   "core_power_tao",      0.0, args_tao_w.core_flux_unc),
                          ("bkg",   "bckg_rate",           0.0, args_tao_w.bckg_rate_unc),
                          ("snf_tao",   "snf_noneq_norm_tao0", 0.0, args_tao_w.snf_unc),
                          ("noneq_tao", "snf_noneq_norm_tao1", 0.0, args_tao_w.noneq_unc)]
            # NL sector (only when NL is NOT baked into the leakage matrix):
            # 5 shape pulls a0..a4 = NL_tao0..4 (prior NL_sd), plus the separate
            # energy-scale LY = NL_tao5 (prior ene_scale_unc). Earlier this
            # mislabelled NL_tao4 (a shape pull) as "LY" and divided its error by
            # ene_scale_unc, producing the spurious "LY err ≈ 140"; NL_tao5 (the
            # real LY) was never written.
            if not getattr(args_tao_w, "nl_in_leak", False):
                tao_export += ([(f"a{i}", f"NL_tao{i}", 0.0, args_tao_w.NL_sd)
                                for i in range(5)]
                               + [("LY", "NL_tao5", 0.0, args_tao_w.ene_scale_unc)])
            tao_export += [("a", "abc_tao0", args_tao_w.a, args_tao_w.a_err),
                           ("b", "abc_tao1", args_tao_w.b, args_tao_w.b_err),
                           ("c", "abc_tao2", args_tao_w.c, args_tao_w.c_err)]

            fetched = {tag: tao_val_err(pname) for tag, pname, _, _ in tao_export}
            if any(v is None for v in fetched.values()):
                print("[bundled] TAO nuisances not written — best-fit values "
                      "unavailable (cached prefit, no m_pre)")
            else:
                tao_dir = output_file.GetDirectory("tao") or output_file.mkdir("tao")
                dir_nuis_tao = (tao_dir.GetDirectory("nuisances")
                                or tao_dir.mkdir("nuisances"))
                dir_nuis_tao.cd()
                for tag, pname, nominal, sigma in tao_export:
                    val, err = fetched[tag]
                    write_parameter(tag, (val - nominal) / sigma, err / sigma)
                # segment (shape) nuisances, mirrored here like in the ORSA file
                for i in range(n_nuis):
                    if fit_mode == 'prefit_wosegs' and m_pre is not None:
                        write_parameter(f"rea_shape_f{i}",
                                        m_pre.values[f"nuis{i}"], m_pre.errors[f"nuis{i}"])
                    else:
                        write_parameter(f"rea_shape_f{i}",
                                        m.values[f"nuis{i}"], m.errors[f"nuis{i}"])
    finally:
        if output_file is not None:
            force_vector_streamers(output_file)
            output_file.Close()
    # ── 19b. Post-fit spectra replay + residual plots ────────────────────
    # Runs right after the fit (before correlation/profiling/minos) so the
    # juno/dyb/tao postfit spectra reach the ROOT file even if a later
    # post-fit step crashes or the run is interrupted.
    chi2_true = m.fval
    if not args_juno.silent_print:
        # Replay each JUNO bundle's chi2_contribution at the best-fit point
        # to obtain spectra and feed plot_fit_residuals.
        values_best = {n: m.values[n] for n in m.parameters}
        juno_shell_results = []   # per-shell spectra for the summed plot
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

            # Per-bundle suffix (set in main._dataset_suffix) so multiple
            # shells/phases don't overwrite each other's postfit spectra; it
            # matches the prefit names written in spectra._write_juno_output.
            root_suffix = getattr(b.args, 'root_suffix', '')

            plot_fit_residuals_all(
                s_tot_juno, sp_obs, s_juno, fit_bckgs,
                b.detector.ndays, b.args,
                detector_label="JUNO",
                tag=f"{b.phase}_{b.shell}",
                chi2_value=m.fval,
                ndf = len(sp_obs.bins)-4,
                write_output=args_juno.write_output, cm=cm_juno,
                eff=b.detector.args.detector_efficiency*(1+gv('eff_nuis')),
                root_suffix=root_suffix)

            # stash spectra for the summed-over-shells plot below
            juno_shell_results.append({
                's_tot':    s_tot_juno,
                's_obs':    sp_obs,
                's_signal': s_juno,
                'bckgs':    fit_bckgs,
                'cm':       cm_juno,
                'ndays':    b.detector.ndays,
                'phase':    b.phase,
                'shell':    b.shell,
                'eff_postfit': b.detector.args.detector_efficiency * (1 + gv('eff_nuis')),
            })

        # ── summed post-fit spectrum over all JUNO shells ────────────────
        if len(juno_shell_results) > 1:
            plot_postfit_summed(
                juno_shell_results, args_juno,
                detector_label="JUNO",
                chi2_value=m.fval,
                ndf=len(juno_shell_results[0]['s_obs'].bins) - 4,
                write_output=args_juno.write_output)

        # ── TAO / DYB replay at the best-fit point (combined mode) ───────
        # In the prefit modes these outputs are produced by
        # run_dyb_tao_prefit; in combined mode the bundles stay in the main
        # Minuit fit (id in routing), so replay them from m here.
        nuis_vals = [values_best[f'nuis{i}'] for i in range(n_shape)]

        if tao_bundle is not None and id(tao_bundle) in routing:
            map_tao = routing[id(tao_bundle)]
            def gv_tao(base): return values_best[map_tao[base]]
            tao_kwargs = dict(
                Nrea_tao=gv_tao('Nrea_tao'),
                abc_tao=[gv_tao('abc_tao0'), gv_tao('abc_tao1'),
                         gv_tao('abc_tao2')],
                snf_noneq_norm_tao=[gv_tao(f'snf_noneq_norm_tao{i}')
                                    for i in range(2)],
                core_power_tao=gv_tao('core_power_tao'),
                eff_nuis_tao=gv_tao('eff_nuis_tao'),
                bckg_rate=gv_tao('bckg_rate'),
                unc=tao_bundle.unc_key, nuisance=nuis_vals)
            if 'NL_tao0' in map_tao:
                tao_kwargs['NL_tao'] = [gv_tao(f'NL_tao{i}') for i in range(6)]
            (chi2_tao, _, s_obs_tao, s_tot_tao,
             bckg_tao, cm_tao) = tao_bundle.detector.chi2_contribution(**tao_kwargs)
            plot_fit_residuals_all(
                s_tot=s_tot_tao,
                s_obs=s_obs_tao,
                s_signal=tao_bundle.detector.ensp_nom['rtot'],
                bckgs=bckg_tao,
                ndays=tao_bundle.detector.ndays,
                args=tao_bundle.args,
                detector_label="TAO",
                tag="best-fit",
                chi2_value=chi2_tao,
                cm=cm_tao,
                ndf=len(s_obs_tao.bins) - len(E_segments),
                write_output=tao_bundle.args.write_output)

        if (dyb_bundle is not None and id(dyb_bundle) in routing
                and not getattr(args_juno, 'use_dagflow', False)):
            map_dyb = routing[id(dyb_bundle)]
            noneq = (values_best[map_dyb['snf_noneq_norm_dyb']]
                     if 'snf_noneq_norm_dyb' in map_dyb else 0.0)
            (chi2_dyb, s_obs_dyb, s_tot_dyb,
             cm_dyb, ibd_postfit) = dyb_bundle.detector.chi2_contribution(
                nuisance_coarse=np.array(nuis_vals),
                snf_noneq_norm_dyb=noneq,
                E_segments=E_segments)
            plot_fit_residuals_all(
                s_tot=s_tot_dyb,
                s_obs=s_obs_dyb,
                s_signal=dyb_bundle.detector.ensp_nom['rtot'],
                bckgs=None,
                ndays=dyb_bundle.detector.ndays,
                args=dyb_bundle.args,
                detector_label="DYB",
                tag="best-fit",
                chi2_value=chi2_dyb,
                cm=cm_dyb,
                ndf=len(s_obs_dyb.bins) - len(E_segments),
                write_output=dyb_bundle.args.write_output)
            dyb_bundle.detector.ensp_nom['ribd'].Plot(
                f"{args_juno.plots_folder}/rfis_posfit.png",
                xlabel="Antineutrino energy (MeV)",
                ylabel="Events",
                extra_spectra=[ibd_postfit],
                leg_labels=['pre-fit', 'post-fit'],
                colors=['steelblue', 'brown'],
                xmin=0.7, xmax=12.0, ymin=1e-3,
                log_scale=False,
                title=f"PRL 2025, {len(E_segments)} segments")

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

    # Pin the LBL-projection SOLAR params (theta12, dm2_21) to THIS data fit's
    # best-fit values, so the observed dm2_31 profile and the NMO-significance
    # fit use the same fixed solar in f_lbl as the FC toys do — instead of the
    # per-point floating solar. (f_lbl reads args.sin2_th12_lbl/dm2_21_lbl.)
    if getattr(args_juno, 'pull_dm2_31_lbl', False):
        _s12_data  = float(m.values['sin2_th12'])
        _dm21_data = float(m.values['dm2_21']) * 1e-5     # Minuit -> physical eV²
        for _b in juno_bundles_all:
            _b.detector.args.sin2_th12_lbl = _s12_data
            _b.detector.args.dm2_21_lbl    = _dm21_data

    # ── 21. dm²₃₁ profile ────────────────────────────────────────────────
    if args_juno.profile_dm2_31:
        profile_dm2_31(m, args_juno, NMO_opt=bool(args_juno.NMO_opt),
                       write=args_juno.write_output,
                       conditional=bool(getattr(args_juno,
                                                'profile_dm2_31_conditional',
                                                False)))

    # ── 21b. 2D geo-rate vs dm²₂₁ Δχ² map (conditional and/or profiled) ───
    if getattr(args_juno, 'geo_dm21_2d', False):
        from .minuit import scan_geo_dm21_2d
        scan_geo_dm21_2d(m, chi2_func, args_juno,
                         routing=routing,
                         juno_bundles_all=juno_bundles_all,
                         NMO_opt=bool(args_juno.NMO_opt))

    # ── 21c. 2D solar (sin²θ₁₂, Δm²₂₁) profiled contour via mncontour ─────
    if getattr(args_juno, 'solar_mncontour', False):
        from .minuit import scan_solar_contour
        scan_solar_contour(m, args_juno, NMO_opt=bool(args_juno.NMO_opt))

    # ── 21d. 2D (sin²θ₁₃, Δm²₃₁) profiled contour via mncontour ───────────
    if getattr(args_juno, 'th13_dm31_2d', False):
        from .minuit import scan_th13_dm31_contour
        scan_th13_dm31_contour(m, chi2_func, args_juno,
                               NMO_opt=bool(args_juno.NMO_opt))

    # ── 22. MINOS errors ─────────────────────────────────────────────────
    merrors = bool(args_juno.calc_minuit_errors)
    if merrors:
        m.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31", "bkg_geo")

        # ── Total geo rate with MINOS (asymmetric) errors ────────────────
        # The geo nuisance is shared across shells (a single Minuit param, hence
        # the bare "bkg_geo" name profiled above), so there is ONE geo rate: the
        # per-shell start rates are summed and scaled by the shared (1+nuis). The
        # error is that shared nuisance's MINOS error propagated to the summed
        # rate (Σ start_rate · σ), referenced to the nominal rate — same
        # convention as the "Background rates" block, NOT a full multi-nuisance
        # (incl. per-shell efficiency) propagation of the total event count.
        geo_bundles = [b for b in juno_bundles_all
                       if "geo" in b.backgrounds['JUNO']
                       and routing[id(b)]['bkg_geo'] in m.merrors]
        if geo_bundles:
            mangled     = routing[id(geo_bundles[0])]['bkg_geo']
            nuis        = m.values[mangled]
            me          = m.merrors[mangled]
            nuis_err    = m.errors[mangled]
            start_total = 0.0
            rate_total  = 0.0
            for b in geo_bundles:
                sr  = (float(b.args.geo_tnu_rate)
                       * float(b.args.tnu_to_cpd)
                       * float(b.args.detector_efficiency))
                eff = m.values[routing[id(b)]['eff_nuis']]
                start_total += sr
                rate_total  += sr * (1 + nuis) * (1 + eff)
            err_lo    = start_total * abs(me.lower)
            err_hi    = start_total * abs(me.upper)
            hesse_err = start_total * nuis_err
            n_sh      = len(geo_bundles)
            print("************Total geo rate (MINOS errors)************")
            print(f"geo{f' (Σ over {n_sh} shells)' if n_sh > 1 else ''}: "
                  f"{rate_total:.4f} +{err_hi:.4f} / -{err_lo:.4f} cpd "
                  f"(rel unc: +{err_hi/rate_total:.3%} / -{err_lo/rate_total:.3%}; "
                  f"Hesse: ±{hesse_err:.4f} cpd = {hesse_err/rate_total:.3%}; "
                  f"minos valid: {me.is_valid})")

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
            dir_osc = (output_file.GetDirectory("oscillation_parameters")
                       or output_file.mkdir("oscillation_parameters"))
            dir_osc.cd()
        try:
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
        finally:
            if args_juno.write_output:
                force_vector_streamers(output_file)
                output_file.Close()

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
        # Opposite-ordering fit: confine dm2_31 to the OTHER ordering's sign
        # domain so it stays in that basin (LBL pull + oscillation flip at 0).
        apply_shared_limits(m_opp, args_juno, n_shape_bins=n_shape,
                            nmo_opt=not bool(args_juno.NMO_opt))
        apply_all_limits(m_opp, bundles, routing_opp)
        m_opp.strategy = int(getattr(args_juno, 'minuit_strategy', 0))
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

    # ── 28. Stat-only refit ──────────────────────────────────────────────
    # Two-stage "statistics-only" measurement: the full fit above already found
    # the joint best fit with every nuisance floating. Here we PIN every nuisance
    # at that best-fit value and re-fit ONLY the parameters of interest — the 4
    # oscillation params plus the shared geo rate — so the resulting Hesse errors
    # carry no nuisance freedom (statistics only). The central values are
    # essentially unchanged (they already sit at the conditional minimum); only
    # the errors shrink. Results OVERWRITE the normal `oscillation_parameters`
    # folder (write_osc_params uses kOverwrite), so the Postfit carries the
    # stat-only values/errors.
    #
    # NB: `unc: stat` already makes the χ² covariance statistical (systematics
    # enter this fit through the pull penalties, not a systematic cov matrix), so
    # fixing the pulls is all that is needed for a genuine stat-only error.
    # `stat_only_osc` is a stricter variant: it ALSO pins the 4 oscillation
    # parameters at their full-fit best-fit values, so only the geo rate (and
    # any other non-osc POI) floats in the refit. Plain `stat_only` leaves the
    # osc params free and pins only the nuisances/systematics.
    stat_only     = getattr(args_juno, 'stat_only', False)
    stat_only_osc = getattr(args_juno, 'stat_only_osc', False)
    # `fix_osc_to`: path to a Postfit ROOT from a previous run (e.g. the full
    # all-phase AllP fit). When set, the stat-only refit pins the 4 osc params
    # at THOSE external best-fit values instead of this run's own full fit —
    # implies the osc-pinned variant (only geo floats).
    fix_osc_to = (getattr(args_juno, 'fix_osc_to', '') or '').strip()
    osc_ext    = None
    if fix_osc_to:
        osc_ext       = read_external_osc(fix_osc_to)
        stat_only_osc = True
    if stat_only or stat_only_osc:
        label = ("nuisances + osc fixed at best fit" if stat_only_osc
                 else "nuisances fixed at best fit")
        if osc_ext is not None:
            label = f"nuisances fixed at best fit; osc fixed to {fix_osc_to}"
        print(f"\n************ Stat-only refit ({label}) ************")

        poi = list(OSC_PARAM_NAMES)                     # sin2_th12/13, dm2_21/31
        # Shared geo pull (its Minuit name may be mangled per bundle) stays free.
        geo_name    = None
        geo_bundles = [b for b in juno_bundles_all
                       if 'geo' in b.backgrounds.get('JUNO', {})]
        if geo_bundles:
            cand = routing[id(geo_bundles[0])].get('bkg_geo')
            if cand in m.parameters:
                geo_name = cand
        if geo_name is not None:
            poi.append(geo_name)

        # stat_only_osc: drop the oscillation params from the POI set so they
        # are pinned at their full-fit best fit alongside the nuisances. The
        # resulting geo Hesse error then carries neither nuisance NOR
        # oscillation freedom.
        if stat_only_osc:
            poi = [p for p in poi if p not in OSC_PARAM_NAMES]

        # fix_osc_to: move the osc params to the EXTERNAL best fit before pinning
        # so the refit floats only geo around the AllP osc values, not this run's.
        if osc_ext is not None:
            for name, val in osc_ext.items():
                m.values[name] = val
            print("  osc pinned to external values: "
                  + ", ".join(f"{k}={v:.6g}" for k, v in osc_ext.items()))

        # Pin everything except the POIs at the full-fit best fit.
        for p in m.parameters:
            m.fixed[p] = (p not in poi)
        if fix_dm2_31_to is not None:          # respect an externally-fixed dm2_31
            m.fixed['dm2_31'] = True

        m.strategy = 2
        if any(not m.fixed[p] for p in m.parameters):   # something left to fit
            m.migrad()
        m.hesse()
        free_now = [p for p in poi if not m.fixed[p]]
        print(f"Stat-only fmin = {m.fval:.4f}  (valid: {m.fmin.is_valid}, "
              f"free: {free_now})")

        out_stat = None
        if args_juno.write_output:
            out_stat = ROOT.TFile(
                f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
                f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root",
                "update")
            d_stat = (out_stat.GetDirectory("oscillation_parameters")
                      or out_stat.mkdir("oscillation_parameters"))
            d_stat.cd()
        try:
            print("************ Stat-only oscillation parameters ************")
            osc_tag = "osc fixed at best fit" if stat_only_osc else "stat only"
            for i in range(4):
                val     = m.values[i]
                err     = m.errors[i]
                exp_val = osc_params_nom[i]
                rel_unc = err / val if val != 0 else float('nan')
                dev     = (val - exp_val) / err if err != 0 else float('nan')
                print(f"{m.parameters[i]}: {val:.6g} +/- {err:.6g} ({osc_tag}); "
                      f"rel unc: {rel_unc:.3%}; deviation: {dev:.2f} σ")
                if out_stat is not None:
                    write_osc_params(str(m.parameters[i]), val, err)

            # Total (Σ shells) geo rate with its stat-only Hesse error, using the
            # same nominal-referenced convention as the MINOS geo-rate block.
            if geo_name is not None:
                nuis        = m.values[geo_name]
                nuis_err    = m.errors[geo_name]
                start_total = 0.0
                rate_total  = 0.0
                for b in geo_bundles:
                    sr  = (float(b.args.geo_tnu_rate) * float(b.args.tnu_to_cpd)
                           * float(b.args.detector_efficiency))
                    eff = m.values[routing[id(b)]['eff_nuis']]
                    start_total += sr
                    rate_total  += sr * (1 + nuis) * (1 + eff)
                err_total = start_total * nuis_err
                n_sh      = len(geo_bundles)
                print(f"geo{f' (Σ over {n_sh} shells)' if n_sh > 1 else ''}: "
                      f"{rate_total:.4f} +/- {err_total:.4f} cpd (stat only; "
                      f"rel unc: "
                      f"{err_total / rate_total if rate_total else float('nan'):.3%})")

                # Overwrite the per-shell geo NUISANCE TVectorD ([pull, err]) with
                # the stat-only value/error. This is what downstream readers use
                # (scripts/geo_free_83.py read_geo_pull → {juno_dir}/nuisances/geo);
                # section 18 wrote it earlier with the FULL-fit error, so without
                # this the "stat-only" file would carry the stat+syst geo error.
                # Mirror section 18's units exactly (pull/rate_unc when the geo is
                # constrained, raw pull when free) so the reader un-normalises
                # correctly.
                if out_stat is not None:
                    write_osc_params("geo_rate", rate_total, err_total)
                    from ..spectra import juno_output_dirname
                    for b in geo_bundles:
                        info     = b.backgrounds['JUNO']['geo']
                        rate_unc = info.get("rate_unc", None)
                        jname    = juno_output_dirname(b.args)
                        jd = (out_stat.GetDirectory(jname)
                              or out_stat.mkdir(jname))
                        nd = jd.GetDirectory("nuisances") or jd.mkdir("nuisances")
                        nd.cd()
                        if rate_unc:                      # constrained (rate_unc>0)
                            gv, ge = nuis / rate_unc, nuis_err / rate_unc
                        else:                             # free geo (rate_unc 0/None)
                            gv, ge = nuis, nuis_err
                        vec = ROOT.TVectorD(2); vec[0] = gv; vec[1] = ge
                        vec.Write("geo", ROOT.TObject.kOverwrite)
                    d_stat.cd()                           # restore osc dir context
        finally:
            if out_stat is not None:
                force_vector_streamers(out_stat)
                out_stat.Close()

    if return_cost_fn:
        return m, chi2_func
    return m
