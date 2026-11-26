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
from datetime import datetime
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


def lbl_pull_target(args, ordering, sin2_12, sin2_13, dm2_21):
    """(center, sigma) for the LBL dm2_31 Gaussian pull, in SIGNED dm2_31.

    NO -> configured dm2_31_lbl (positive), error dm2_31_lbl_err.
    IO -> if an independent IO measurement is configured (dm2_31_lbl_IO, and
          optionally dm2_31_lbl_err_IO) use it DIRECTLY as -|value| with that
          error; otherwise PROJECT the NO central onto IO via f_lbl (legacy
          behaviour), keeping dm2_31_lbl_err. theta23/dcp and the LBL solar
          params come from args / nuosc.op_nom (the data best-fit solar set on
          args when available). Data-fit path only; FC toys keep their own
          per-ordering precomputed centrals."""
    err = args.dm2_31_lbl_err
    if ordering == "IO":
        direct = getattr(args, 'dm2_31_lbl_IO', None)
        if direct is not None:
            return -abs(direct), (getattr(args, 'dm2_31_lbl_err_IO', None) or err)
    th23 = nuosc.op_nom.get('theta23', args.theta23)
    dcp  = nuosc.op_nom.get('dcp',     args.dcp)
    s12  = getattr(args, 'sin2_th12_lbl', None)
    s12  = s12 if s12 is not None else sin2_12
    dm21 = getattr(args, 'dm2_21_lbl', None)
    dm21 = dm21 if dm21 is not None else dm2_21
    return f_lbl(args.dm2_31_lbl, s12, sin2_13, dm21, th23, dcp, ordering), err


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

def profile_dm2_31(m, args_juno, NMO_opt=True, write=False, conditional=False):
    import csv
    from scipy.interpolate import CubicSpline
    full_bound = (1.5, 3.5) if NMO_opt else (-3.5, -1.5)

    # ── optional splitting: scan only one chunk of the dm2_31 range ──────────
    # Run with --profile_nchunks N --profile_chunk i (0..N-1) --profile_csv path
    # --profile_make_graph 0 to do a memory-light slice, then combine the CSVs
    # into the TGraph with combine_profile.py.
    nchunks = int(getattr(args_juno, 'profile_nchunks', 1) or 1)
    chunk   = int(getattr(args_juno, 'profile_chunk', 0) or 0)
    npoints = int(args_juno.profile_points)
    if nchunks > 1:
        lo, hi = full_bound
        w      = (hi - lo) / nchunks
        bound  = (lo + chunk * w, lo + (chunk + 1) * w)
        size   = max(2, npoints // nchunks)
        print(f"Profiling dm2_31 chunk {chunk + 1}/{nchunks} over {bound} "
              f"({size} pts)")
    else:
        bound, size = full_bound, npoints

    if conditional:
        # conditional scan: nuisance/oscillation parameters stay fixed at
        # their current best-fit values instead of being re-minimized
        print("Conditional scan of dm2_31 using built-in profile...")
        res = m.profile("dm2_31", size=size, bound=bound)
    else:
        print("Profiling dm2_31 using built-in mnprofile...")
        res = m.mnprofile("dm2_31", size=size, bound=bound)
    dm2_scan_vals = res[0]
    chi2_vals     = res[1]

    # ── raw (dm2_31, chi2) -> CSV. RAW chi2 (NOT delta): the global minimum
    #    subtraction is done by combine_profile.py after all chunks are merged.
    csv_path = getattr(args_juno, 'profile_csv', '') or ''
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["dm2_31", "chi2"])
            for xv, cv in zip(dm2_scan_vals, chi2_vals):
                wtr.writerow([float(xv), float(cv)])
        print(f"Wrote {len(dm2_scan_vals)} profile points to {csv_path}")

    # ── when splitting, stop here; combine_profile.py builds the TGraph ──────
    if not bool(getattr(args_juno, 'profile_make_graph', True)):
        return

    delta_chi2    = chi2_vals - np.min(chi2_vals)
    dm2_fine          = np.linspace(dm2_scan_vals.min(), dm2_scan_vals.max(), 500)
    cs                = CubicSpline(dm2_scan_vals, chi2_vals)
    chi2_smooth       = cs(dm2_fine)
    delta_chi2_smooth = chi2_smooth - np.min(chi2_smooth)

    # --- Save raw scan points and smoothed curve to CSV ---
    scan_tag = "conditional_dm2_31" if conditional else "profile_dm2_31"
    # csv_path = f"Plots/{scan_tag}_NO_{NMO_opt}_{args_juno.file_prefix}.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["dm2_31_raw", "chi2_raw", "delta_chi2_raw",
    #                      "dm2_31_smooth", "chi2_smooth", "delta_chi2_smooth"])
    #     # Pad raw columns with empty strings where lengths differ (raw < 500)
    #     for i in range(len(dm2_fine)):
    #         raw_dm2   = dm2_scan_vals[i] if i < len(dm2_scan_vals) else ""
    #         raw_chi2  = chi2_vals[i]     if i < len(chi2_vals)     else ""
    #         raw_dchi2 = delta_chi2[i]    if i < len(delta_chi2)    else ""
    #         writer.writerow([raw_dm2, raw_chi2, raw_dchi2,
    #                          dm2_fine[i], chi2_smooth[i], delta_chi2_smooth[i]])
    # print(f"Saved profile scan to {csv_path}")

    plt.figure(figsize=(6, 5))
    plt.plot(dm2_fine, delta_chi2_smooth,
             label="Smoothed Conditional Scan" if conditional else "Smoothed Profile",
             lw=2)
    plt.xlabel(r"$\Delta m^2_{31}$ [10$^{-3}$ eV$^2$]", fontsize=14)
    plt.ylabel(r"$\Delta \chi^2$", fontsize=14)
    plt.axhline(1.0, color="green", ls="--", label=r"1$\sigma$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Plots/{scan_tag}_NO_{NMO_opt}_{args_juno.file_prefix}.png", dpi=300)
    best_index    = np.argmin(delta_chi2)
    best_fit_val  = dm2_scan_vals[best_index]
    if write:
        n_points = len(dm2_fine)
        graph    = ROOT.TGraph(n_points,
                               dm2_fine.astype(float),
                               chi2_smooth.astype(float))
        graph.SetName("chi2_profile" if conditional else "dm2_31")
        graph.SetMarkerStyle(2)
        out_file = ROOT.TFile(
            f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}"
            f"_dm2_31_pull_{args_juno.pull_dm2_31}_{args_juno.fit_mode}.root", "update")
        try:
            dir_prof = out_file.GetDirectory("profiles") or out_file.mkdir("profiles")
            dir_prof.cd()
            graph.Write()
            # raw (unsmoothed) scan points, chi2 as returned by Minuit
            graph_raw = ROOT.TGraph(len(dm2_scan_vals),
                                    np.asarray(dm2_scan_vals, dtype=float),
                                    np.asarray(chi2_vals, dtype=float))
            graph_raw.SetName(("chi2_profile" if conditional else "dm2_31") + "_raw")
            graph_raw.SetMarkerStyle(2)
            graph_raw.Write()
        finally:
            force_vector_streamers(out_file)
            out_file.Close()
    best_dm2_from_scan  = dm2_fine[np.argmin(chi2_smooth)]
    print(best_dm2_from_scan)
    m.values["dm2_31"] = best_dm2_from_scan
    print(m.values["dm2_31"])
    m.migrad()


def _geo_param_and_nominal(m, routing, juno_bundles_all):
    """Resolve the shared geo Minuit-param name and the nominal total geo rate.

    Returns (geo_name, tot_nom_cpd). geo_name is the (possibly mangled) Minuit
    parameter carrying the geo pull; tot_nom_cpd is Σ_shells geo_tnu_rate ·
    tnu_to_cpd · efficiency — the nominal-referenced total to which the pull is
    applied (rate = tot_nom · (1 + pull)), matching the MINOS geo-rate block and
    scripts/fit_data_geo.py. Falls back to a name-search when routing is absent.
    """
    geo_name = None
    tot_nom  = 0.0
    geo_bundles = [b for b in (juno_bundles_all or [])
                   if 'geo' in getattr(b, 'backgrounds', {}).get('JUNO', {})]
    if routing is not None and geo_bundles:
        cand = routing[id(geo_bundles[0])].get('bkg_geo')
        if cand in m.parameters:
            geo_name = cand
    if geo_name is None:
        if 'bkg_geo' in m.parameters:
            geo_name = 'bkg_geo'
        else:
            geo_name = next((p for p in m.parameters if 'bkg_geo' in p), None)
    for b in geo_bundles:
        a = b.args
        tot_nom += (float(a.geo_tnu_rate) * float(a.tnu_to_cpd)
                    * float(a.detector_efficiency))
    return geo_name, tot_nom


def scan_geo_dm21_2d(m, chi2_func, args_juno, *, routing=None,
                     juno_bundles_all=None, NMO_opt=True):
    """2D Δχ² map of the geo rate vs dm²₂₁.

    Both axes are Minuit parameters of the joint fit: dm²₂₁ (shared, ×1e-5 eV²)
    and the geo pull ``bkg_geo``. Two flavours, selected by ``geo_dm21_mode``:

      * ``conditional`` — freeze all other parameters at the global best fit and
        vary only the two axes (fast; same idea as profile_dm2_31_conditional).
      * ``profiled``    — at each grid point fix the two axes and re-run Migrad
        over every other nuisance (true profile-likelihood surface; slow, so it
        is chunkable via geo_dm21_nchunks / geo_dm21_chunk like the dm2_31
        profile). A final job with geo_dm21_make_graph=True assembles the chunks.

      * ``both`` runs conditional first, then profiled.

    Config keys (all under fit_options, all optional):
      geo_dm21_2d: true            enable this scan
      geo_dm21_mode: both          conditional | profiled | both
      geo_dm21_ngeo: 41            geo-axis grid points
      geo_dm21_ndm21: 41           dm²₂₁-axis grid points
      geo_dm21_nsigma: 4.0         half-width (in σ) when explicit ranges absent
      geo_dm21_geo_range: [lo,hi]  geo axis range in cpd (overrides nsigma)
      geo_dm21_dm21_range: [lo,hi] dm²₂₁ axis range in 1e-5 eV² (overrides nsigma)
      geo_dm21_nchunks: 1          split profiled grid into N batches
      geo_dm21_chunk: 0            which batch this job runs (0..N-1)
      geo_dm21_make_graph: true    assemble chunks + plot (set false in batches)
      geo_dm21_mncontour: true     overlay profiled 1/2/3σ mncontour lines
      geo_dm21_tag: ''             filename tag (default: file_prefix)
    """
    import csv
    import glob

    geo_name, tot_nom = _geo_param_and_nominal(m, routing, juno_bundles_all)
    if geo_name is None:
        print("[geo2d] no bkg_geo parameter in the fit — skipping 2D scan")
        return
    if tot_nom <= 0:
        print("[geo2d] WARNING: nominal geo rate is 0; geo axis will be plotted "
              "as the raw pull instead of cpd")

    # mncontour (and a clean profiled reference) require a VALID minimum.
    # Some configs — e.g. phase1 with a cached prefit — leave the fit
    # marginally above the EDM goal (is_valid=False), which makes mncontour
    # refuse. Re-tighten with a higher Minuit strategy before proceeding.
    if not m.fmin.is_valid:
        print(f"[geo2d] fit minimum flagged invalid "
              f"(edm={m.fmin.edm:.2g} > goal {m.fmin.edm_goal:.2g}); "
              f"re-tightening with strategy 2")
        m.strategy = 2
        for _ in range(3):
            m.migrad()
            if m.fmin.is_valid:
                break
        # Restore a cheaper strategy: strategy 2 is only needed to re-validate
        # the minimum, NOT for the many per-pixel Migrads of the profiled grid
        # (which would run several-fold slower if they inherited strategy 2).
        m.strategy = 1
        print(f"[geo2d] after re-tighten: valid={m.fmin.is_valid}, "
              f"edm={m.fmin.edm:.2g} (strategy reset to 1 for the scan)")

    names   = list(m.parameters)
    base    = list(m.values)                       # global best-fit vector
    ig      = names.index(geo_name)
    idm     = names.index('dm2_21')
    chi2_min_global = float(m.fval)

    mode   = str(getattr(args_juno, 'geo_dm21_mode', 'both')).lower()
    ngeo   = int(getattr(args_juno, 'geo_dm21_ngeo', 41))
    ndm    = int(getattr(args_juno, 'geo_dm21_ndm21', 41))
    nsig   = float(getattr(args_juno, 'geo_dm21_nsigma', 4.0))
    tag    = getattr(args_juno, 'geo_dm21_tag', '') or args_juno.file_prefix
    do_mnc = bool(getattr(args_juno, 'geo_dm21_mncontour', True))
    mnc_sz = int(getattr(args_juno, 'geo_dm21_mncontour_size', 200))

    # ── axis grids ───────────────────────────────────────────────────────
    # Geo axis lives in PULL space internally (that is the Minuit param);
    # cpd = tot_nom·(1+pull). dm²₂₁ axis lives in Minuit units (×1e-5 eV²).
    geo_range = getattr(args_juno, 'geo_dm21_geo_range', None)
    dm_range  = getattr(args_juno, 'geo_dm21_dm21_range', None)

    pull_best = base[ig]
    dm_best   = base[idm]
    if geo_range:
        rlo, rhi   = float(geo_range[0]), float(geo_range[1])
        pull_grid  = (np.linspace(rlo, rhi, ngeo) / tot_nom - 1.0
                      if tot_nom > 0 else np.linspace(rlo, rhi, ngeo))
    else:
        gerr      = float(m.errors[geo_name])
        pull_grid = np.linspace(pull_best - nsig * gerr,
                                pull_best + nsig * gerr, ngeo)
        pull_grid = np.clip(pull_grid, -0.999, None)   # bkg_geo limit is [-1, …]
    if dm_range:
        dm_grid = np.linspace(float(dm_range[0]), float(dm_range[1]), ndm)
    else:
        derr    = float(m.errors['dm2_21'])
        dm_grid = np.linspace(dm_best - nsig * derr, dm_best + nsig * derr, ndm)

    geo_cpd_grid = (tot_nom * (1.0 + pull_grid) if tot_nom > 0 else pull_grid)

    outdir = getattr(args_juno, 'main_data_folder', '.') or '.'
    os.makedirs(outdir, exist_ok=True)
    prefix = f"{outdir}/geo_dm21_2d_{tag}_NO_{NMO_opt}"

    def _plot(chi2, kind):
        dchi2 = chi2 - np.nanmin(chi2)
        X, Y  = np.meshgrid(geo_cpd_grid, dm_grid, indexing='ij')
        plt.figure(figsize=(7, 6))
        levels = np.linspace(0, min(30.0, np.nanmax(dchi2)), 31)
        cf = plt.contourf(X, Y, dchi2, levels=levels, cmap='viridis_r',
                          extend='max')
        plt.colorbar(cf, label=r"$\Delta\chi^2$")
        # White solid = Δχ² contours from THIS grid (2.30/6.18/11.83 = 1/2/3σ,
        # 2 d.o.f.). These are always drawn, so a map is never contour-less.
        cl = plt.contour(X, Y, dchi2, levels=[2.30, 6.18, 11.83],
                         colors='white', linewidths=1.6)
        fmt = {2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'}
        plt.clabel(cl, fmt=fmt, fontsize=9)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='white', lw=1.6,
                   label=f'{kind} $\\Delta\\chi^2$ 1/2/3$\\sigma$')]
        # Dashed = profiled mncontour lines (true joint CLs), overlaid.
        if do_mnc:
            try:
                for clv, col in [(1, 'red'), (2, 'orange'), (3, 'yellow')]:
                    pts = np.asarray(m.mncontour(geo_name, 'dm2_21',
                                                 cl=clv, size=mnc_sz))
                    if pts.ndim == 2 and len(pts):
                        px = (tot_nom * (1.0 + pts[:, 0]) if tot_nom > 0
                              else pts[:, 0])
                        plt.plot(px, pts[:, 1], '--', color=col, lw=1.5)
                        legend_handles.append(
                            Line2D([0], [0], color=col, lw=1.5, ls='--',
                                   label=f'mncontour {clv}$\\sigma$'))
            except Exception as e:
                print(f"[geo2d] mncontour overlay failed: {e}")
        gx = tot_nom * (1.0 + pull_best) if tot_nom > 0 else pull_best
        star, = plt.plot(gx, dm_best, 'r*', ms=15, mec='k', label='best fit')
        legend_handles.append(star)
        plt.legend(handles=legend_handles, loc='upper right', fontsize=8)
        plt.xlabel("Total geo rate [cpd]" if tot_nom > 0 else "geo pull")
        plt.ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
        plt.title(f"{kind} $\\Delta\\chi^2$  ({tag})", fontsize=10)
        plt.tight_layout()
        png = f"{prefix}_{kind}.png"
        plt.savefig(png, dpi=200)
        plt.close()
        np.savez(f"{prefix}_{kind}.npz",
                 geo_cpd=geo_cpd_grid, geo_pull=pull_grid, dm2_21=dm_grid,
                 chi2=chi2, dchi2=dchi2, tot_nom=tot_nom,
                 best_pull=pull_best, best_dm2_21=dm_best,
                 chi2_min_global=chi2_min_global)
        print(f"[geo2d] wrote {png} and {prefix}_{kind}.npz")

        # ── true per-pixel COLZ heatmap (real χ² at every grid point) ────
        disp = getattr(args_juno, 'geo_dm21_title', '') or tag
        plt.figure(figsize=(7.6, 6)); ax = plt.gca()
        pc = ax.pcolormesh(X, Y, dchi2, cmap='viridis_r', shading='auto',
                           vmin=0, vmax=min(20.0, float(np.nanmax(dchi2))))
        plt.colorbar(pc, ax=ax, label=r"$\Delta\chi^2$")
        ax.contour(X, Y, dchi2, levels=[1, 2.30, 4, 6.18, 9, 11.83, 16, 20],
                   colors='k', linewidths=0.5, alpha=0.5)
        ax.plot(gx, dm_best, 'r*', ms=15, mec='k', label='best fit')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlabel("Total geo rate [cpd]" if tot_nom > 0 else "geo pull")
        ax.set_ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
        ax.set_title(f"{kind} $\\Delta\\chi^2$ map, COLZ (real per-pixel)  "
                     f"({disp})", fontsize=10)
        plt.tight_layout()
        cpng = f"{prefix}_{kind}_colz.png"
        plt.savefig(cpng, dpi=200); plt.close()
        print(f"[geo2d] wrote {cpng}")

    # ── mncontour-only (profiled contour lines, no grid/surface) ─────────
    # Fast: skips the whole grid scan — just the fit (already done) plus three
    # profiled contour traces. geo_dm21_mncontour_size controls smoothness.
    if mode == 'mncontour':
        print(f"[geo2d] mncontour-only: profiled 1/2/3σ contours "
              f"(size={mnc_sz} points each)")
        from matplotlib.lines import Line2D
        plt.figure(figsize=(7, 6))
        saved = {}
        handles = []
        for clv, col in [(1, 'red'), (2, 'orange'), (3, 'green')]:
            try:
                pts = np.asarray(m.mncontour(geo_name, 'dm2_21',
                                             cl=clv, size=mnc_sz))
            except Exception as e:
                print(f"[geo2d] mncontour cl={clv}σ failed: {e}")
                continue
            if pts.ndim != 2 or not len(pts):
                continue
            px = tot_nom * (1.0 + pts[:, 0]) if tot_nom > 0 else pts[:, 0]
            py = pts[:, 1]
            px = np.append(px, px[0])         # close the loop
            py = np.append(py, py[0])
            plt.plot(px, py, '-', color=col, lw=1.8)
            handles.append(Line2D([0], [0], color=col, lw=1.8,
                                  label=f'{clv}$\\sigma$'))
            saved[f'geo_cpd_{clv}sig'] = px
            saved[f'dm2_21_{clv}sig']  = py
        gx = tot_nom * (1.0 + pull_best) if tot_nom > 0 else pull_best
        star, = plt.plot(gx, dm_best, 'k*', ms=15, label='best fit')
        handles.append(star)
        plt.legend(handles=handles, loc='upper right', fontsize=9)
        plt.xlabel("Total geo rate [cpd]" if tot_nom > 0 else "geo pull")
        plt.ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
        plt.title(f"profiled contours (mncontour)  ({tag})", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png = f"{prefix}_mncontour.png"
        plt.savefig(png, dpi=200)
        plt.close()
        np.savez(f"{prefix}_mncontour.npz", tot_nom=tot_nom,
                 best_geo_cpd=gx, best_dm2_21=dm_best, **saved)
        print(f"[geo2d] wrote {png} and {prefix}_mncontour.npz")
        return

    # ── mncontour MAP (profiled Δχ² surface as nested iso-Δχ² lines) ──────
    # The profiled 2D Δχ² map WITHOUT a grid: trace one mncontour per Δχ²
    # level (cl = chi2.cdf(level, df=2) for the 2-dof joint region), then
    # stack the level sets. Fully profiled, cheaper/smoother than the nested-
    # Migrad grid. Levels via geo_dm21_map_levels (Δχ² values).
    if mode == 'mncontour_map':
        from scipy.stats import chi2 as _chi2dist
        from scipy.interpolate import griddata
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        levels = getattr(args_juno, 'geo_dm21_map_levels', None)
        if not levels:
            levels = [1.0, 2.30, 4.0, 6.18, 9.0, 11.83, 16.0, 20.0]
        levels = sorted(float(x) for x in levels)
        # Display name for titles (e.g. "P26B phase 1"); falls back to tag.
        disp = getattr(args_juno, 'geo_dm21_title', '') or tag
        print(f"[geo2d] mncontour_map: {len(levels)} profiled iso-Δχ² lines "
              f"at Δχ²={levels} (size={mnc_sz} each)")
        norm = Normalize(vmin=0.0, vmax=max(levels))
        cmap = plt.get_cmap('viridis_r')
        gx = tot_nom * (1.0 + pull_best) if tot_nom > 0 else pull_best
        # scatter of (geo, dm2_21, Δχ²): best fit (0) + every contour point.
        sx, sy, sz = [gx], [dm_best], [0.0]

        plt.figure(figsize=(7.4, 6))
        saved = {}
        for lev in levels:
            prob = float(_chi2dist.cdf(lev, df=2))    # Δχ²(2 dof) → CL prob
            try:
                pts = np.asarray(m.mncontour(geo_name, 'dm2_21',
                                             cl=prob, size=mnc_sz))
            except Exception as e:
                print(f"[geo2d]   Δχ²={lev} (cl={prob:.4f}) failed: {e}")
                continue
            if pts.ndim != 2 or not len(pts):
                continue
            px = tot_nom * (1.0 + pts[:, 0]) if tot_nom > 0 else pts[:, 0]
            py = pts[:, 1]
            sx += list(px); sy += list(py); sz += [lev] * len(px)
            px = np.append(px, px[0]); py = np.append(py, py[0])
            plt.plot(px, py, '-', color=cmap(norm(lev)), lw=1.6)
            saved[f'geo_cpd_L{lev:g}'] = px
            saved[f'dm2_21_L{lev:g}']  = py
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label=r"$\Delta\chi^2$")
        plt.plot(gx, dm_best, 'r*', ms=15, mec='k', label='best fit')
        plt.legend(loc='upper right', fontsize=9)
        plt.xlabel("Total geo rate [cpd]" if tot_nom > 0 else "geo pull")
        plt.ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
        plt.title(f"profiled $\\Delta\\chi^2$ map (mncontour)  ({disp})",
                  fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png = f"{prefix}_mncontour_map.png"
        plt.savefig(png, dpi=200); plt.close()
        np.savez(f"{prefix}_mncontour_map.npz", tot_nom=tot_nom,
                 levels=np.array(levels), best_geo_cpd=gx,
                 best_dm2_21=dm_best, **saved)
        print(f"[geo2d] wrote {png} and {prefix}_mncontour_map.npz")

        # ── filled COLZ heatmap: interpolate Δχ² from the contour points ──
        sx, sy, sz = np.array(sx), np.array(sy), np.array(sz)
        if len(sx) > 8:
            ggx = np.linspace(sx.min(), sx.max(), 400)
            ggy = np.linspace(sy.min(), sy.max(), 400)
            GX, GY = np.meshgrid(ggx, ggy)
            GZ = griddata((sx, sy), sz, (GX, GY), method='linear')  # nan outside
            plt.figure(figsize=(7.6, 6)); ax = plt.gca()
            pc = ax.pcolormesh(GX, GY, GZ, cmap='viridis_r', shading='auto',
                               vmin=0, vmax=max(levels))
            plt.colorbar(pc, ax=ax, label=r"$\Delta\chi^2$")
            for lev in levels:
                kx = f'geo_cpd_L{lev:g}'
                if kx in saved:
                    ax.plot(saved[kx], saved[f'dm2_21_L{lev:g}'],
                            'k-', lw=0.5, alpha=0.4)
            ax.plot(gx, dm_best, 'r*', ms=15, mec='k', label='best fit')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel("Total geo rate [cpd]" if tot_nom > 0 else "geo pull")
            ax.set_ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
            ax.set_title(f"profiled $\\Delta\\chi^2$ map, COLZ  ({disp})",
                         fontsize=10)
            plt.tight_layout()
            cpng = f"{prefix}_mncontour_map_colz.png"
            plt.savefig(cpng, dpi=200); plt.close()
            print(f"[geo2d] wrote {cpng}")
        return

    # ── 3D cube Δχ²(geo, dm2_21, sin2_th12) — one dataset, all plots ──────
    # Conditional cube (other nuisances frozen at best fit). Saved to npz; the
    # corner plot, slices, E[dm2|geo,sin12] map and 3D shell all derive from it.
    if mode == 'cube':
        disp = getattr(args_juno, 'geo_dm21_title', '') or tag
        idx_th = names.index('sin2_th12')
        th_best = base[idx_th]
        nth = int(getattr(args_juno, 'geo_dm21_nth12', 31))
        th_range = getattr(args_juno, 'geo_dm21_th12_range', None)
        if th_range:
            th_grid = np.linspace(float(th_range[0]), float(th_range[1]), nth)
        else:
            therr = float(m.errors['sin2_th12'])
            th_grid = np.linspace(th_best - nsig * therr,
                                  th_best + nsig * therr, nth)
        ntot = ngeo * ndm * nth
        print(f"[geo2d] cube: {ngeo}×{ndm}×{nth} conditional "
              f"Δχ²(geo,dm2_21,sin2_th12) = {ntot} evals")
        cube = np.empty((ngeo, ndm, nth))
        v = list(base)
        for i, pu in enumerate(pull_grid):
            v[ig] = pu
            for j, dv in enumerate(dm_grid):
                v[idm] = dv
                for k, tv in enumerate(th_grid):
                    v[idx_th] = tv
                    cube[i, j, k] = float(chi2_func(*v))
            if (i + 1) % 5 == 0:
                print(f"[geo2d]   geo {i + 1}/{ngeo}", flush=True)
        dcube = cube - float(np.min(cube))
        gx = tot_nom * (1.0 + pull_best) if tot_nom > 0 else pull_best
        # ── ALWAYS save the cube first (the reusable dataset) ────────────
        cube_npz = f"{prefix}_cube.npz"
        np.savez(cube_npz, dchi2=dcube, chi2=cube,
                 geo_cpd=geo_cpd_grid, geo_pull=pull_grid, dm2_21=dm_grid,
                 sin2_th12=th_grid, tot_nom=tot_nom, best_geo_cpd=gx,
                 best_dm2_21=dm_best, best_sin2_th12=th_best,
                 chi2_min_global=chi2_min_global)
        print(f"[geo2d] wrote {cube_npz}  — all plots derive from this")

        AX = [("Total geo rate [cpd]", geo_cpd_grid, gx),
              (r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]", dm_grid, dm_best),
              (r"$\sin^2\theta_{12}$", th_grid, th_best)]

        def _fill(ax, Xv, Yv, Z):
            XX, YY = np.meshgrid(Xv, Yv, indexing='ij')
            zmax = float(np.nanmax(Z))
            lv = np.linspace(0, min(20.0, zmax if zmax > 0 else 1.0), 31)
            cf = ax.contourf(XX, YY, Z, levels=lv, cmap='viridis_r',
                             extend='max')
            ax.contour(XX, YY, Z, levels=[2.30, 6.18, 11.83],
                       colors='white', linewidths=1.0)
            return cf

        # ── (1) corner / triangle plot ───────────────────────────────────
        try:
            fig, axs = plt.subplots(3, 3, figsize=(11, 10))
            for r in range(3):
                for c in range(3):
                    ax = axs[r][c]
                    if c > r:
                        ax.axis('off'); continue
                    if r == c:
                        oth = tuple(a for a in range(3) if a != r)
                        prof = dcube.min(axis=oth)
                        ax.plot(AX[r][1], prof, 'b-')
                        ax.axhline(1.0, color='gray', ls=':')
                        ax.axvline(AX[r][2], color='r', ls='--')
                        ax.set_ylim(0, 12)
                    else:
                        oth = ({0, 1, 2} - {r, c}).pop()
                        proj = dcube.min(axis=oth)   # axes (min(r,c), max(r,c))
                        _fill(ax, AX[c][1], AX[r][1], proj)  # x=c (<r), y=r
                        ax.plot(AX[c][2], AX[r][2], 'r*', ms=12, mec='k')
                    if r == 2:
                        ax.set_xlabel(AX[c][0], fontsize=9)
                    if c == 0 and r > 0:
                        ax.set_ylabel(AX[r][0], fontsize=9)
            fig.suptitle(f"profiled corner: geo, $\\Delta m^2_{{21}}$, "
                         f"$\\sin^2\\theta_{{12}}$  ({disp})", fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{prefix}_cube_corner.png", dpi=200); plt.close()
            print(f"[geo2d] wrote {prefix}_cube_corner.png")
        except Exception as e:
            print(f"[geo2d] corner plot failed: {e}")

        # ── (2) slices: geo–dm2 maps at several sin2_th12 values ─────────
        try:
            ksel = np.unique(np.linspace(0, nth - 1, 5).astype(int))
            fig, axs = plt.subplots(1, len(ksel), figsize=(3.7 * len(ksel), 4.3),
                                    sharey=True)
            axs = np.atleast_1d(axs)
            for ax, k in zip(axs, ksel):
                cf = _fill(ax, geo_cpd_grid, dm_grid, dcube[:, :, k])
                ax.plot(gx, dm_best, 'r*', ms=11, mec='k')
                ax.set_title(rf"$\sin^2\theta_{{12}}$={th_grid[k]:.4f}", fontsize=9)
                ax.set_xlabel("geo rate [cpd]", fontsize=9)
            axs[0].set_ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]", fontsize=9)
            fig.colorbar(cf, ax=list(axs), label=r"$\Delta\chi^2$")
            fig.suptitle(f"geo–$\\Delta m^2_{{21}}$ slices vs "
                         f"$\\sin^2\\theta_{{12}}$  ({disp})", fontsize=10)
            plt.savefig(f"{prefix}_cube_slices.png", dpi=200); plt.close()
            print(f"[geo2d] wrote {prefix}_cube_slices.png")
        except Exception as e:
            print(f"[geo2d] slices plot failed: {e}")

        # ── (3) E[dm2 | geo, sin2_th12] conditional-prediction map ───────
        try:
            kmin = np.argmin(dcube, axis=1)         # (ngeo, nth)
            dm_exp = dm_grid[kmin]
            XX, YY = np.meshgrid(geo_cpd_grid, th_grid, indexing='ij')
            plt.figure(figsize=(7.8, 6)); ax = plt.gca()
            pc = ax.pcolormesh(XX, YY, dm_exp, cmap='plasma', shading='auto')
            plt.colorbar(pc, ax=ax,
                         label=r"expected $\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
            ax.plot(gx, th_best, 'w*', ms=15, mec='k', label='best fit')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel("Total geo rate [cpd]")
            ax.set_ylabel(r"$\sin^2\theta_{12}$")
            ax.set_title(r"$E[\Delta m^2_{21}\,|\,$geo, "
                         rf"$\sin^2\theta_{{12}}]$  ({disp})", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{prefix}_cube_dm2_expected.png", dpi=200); plt.close()
            print(f"[geo2d] wrote {prefix}_cube_dm2_expected.png")
        except Exception as e:
            print(f"[geo2d] E[dm2|geo,sin12] map failed: {e}")

        # ── (4) 3D Δχ² shell scatter (isosurface proxy; no skimage) ──────
        try:
            GG, DD, TT = np.meshgrid(geo_cpd_grid, dm_grid, th_grid,
                                     indexing='ij')
            sel = dcube <= 11.83                    # inside 3σ-ish (2-dof lvl)
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(GG[sel], DD[sel], TT[sel], c=dcube[sel],
                            cmap='viridis_r', vmin=0, vmax=11.83, s=4, alpha=0.25)
            ax.scatter([gx], [dm_best], [th_best], c='r', marker='*', s=200,
                       edgecolor='k')
            fig.colorbar(sc, ax=ax, shrink=0.6, label=r"$\Delta\chi^2$")
            ax.set_xlabel("geo [cpd]"); ax.set_ylabel(r"$\Delta m^2_{21}$")
            ax.set_zlabel(r"$\sin^2\theta_{12}$")
            ax.set_title(f"3D Δχ² shell (≤11.83)  ({disp})", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{prefix}_cube_3d.png", dpi=200); plt.close()
            print(f"[geo2d] wrote {prefix}_cube_3d.png")
        except Exception as e:
            print(f"[geo2d] 3D shell plot failed: {e}")
        return

    # ── conditional map ──────────────────────────────────────────────────
    if mode in ('conditional', 'both'):
        print(f"[geo2d] conditional scan: {ngeo}×{ndm} grid "
              f"(others frozen at best fit)")
        chi2 = np.empty((ngeo, ndm))
        for i, pull in enumerate(pull_grid):
            for j, dmv in enumerate(dm_grid):
                v = list(base)
                v[ig] = pull
                v[idm] = dmv
                chi2[i, j] = float(chi2_func(*v))
        _plot(chi2, 'conditional')

    # ── profiled map (chunkable) ─────────────────────────────────────────
    if mode in ('profiled', 'both'):
        nchunks = max(1, int(getattr(args_juno, 'geo_dm21_nchunks', 1)))
        chunk   = int(getattr(args_juno, 'geo_dm21_chunk', 0))
        make_gr = bool(getattr(args_juno, 'geo_dm21_make_graph', True))

        flat = [(i, j) for i in range(ngeo) for j in range(ndm)]
        per  = -(-len(flat) // nchunks)                # ceil division
        mine = flat[chunk * per:(chunk + 1) * per]
        print(f"[geo2d] profiled scan chunk {chunk + 1}/{nchunks}: "
              f"{len(mine)} of {len(flat)} points (Migrad per point)")

        # A profiled Δχ² grid only needs the fval at each minimum, not per-pixel
        # errors, so strategy 0 (fewest chi² evaluations, no careful Hessian) is
        # the fastest and the minimum location is unchanged — the main per-pixel
        # speed lever for these single-shell fits.
        m.strategy = 0
        # Optional: also pin sin2_th12 at a chosen value → a PROFILED sin2_th12
        # SLICE (geo, dm2_21 scanned, sin2_th12 fixed, all other nuisances
        # floated). Two such slices give the profiled two-slice side-by-side.
        fix_th = getattr(args_juno, 'geo_dm21_fix_sin12', None)
        idx_th = names.index('sin2_th12')
        if fix_th not in (None, '', 0):
            fix_th = float(fix_th)
            print(f"[geo2d] profiled slice at fixed sin2_th12 = {fix_th:.5f}")
        else:
            fix_th = None
        # Checkpoint file: partial cube saved every few pixels so an
        # interrupted (killed/overnight) run keeps its progress. Assemble/plot
        # it later from this npz even if the job never reaches the end.
        ckpt = f"{prefix}_profiled_partial.npz"
        chi2_part = np.full((ngeo, ndm), np.nan)
        for n, (i, j) in enumerate(mine):
            for k, val in enumerate(base):            # reset to best fit
                m.values[k] = val
            m.values[ig]  = pull_grid[i]
            m.values[idm] = dm_grid[j]
            m.fixed[geo_name] = True
            m.fixed['dm2_21'] = True
            if fix_th is not None:
                m.values[idx_th] = fix_th
                m.fixed['sin2_th12'] = True
            m.migrad()
            chi2_part[i, j] = float(m.fval)
            m.fixed[geo_name] = False
            m.fixed['dm2_21'] = False
            if fix_th is not None:
                m.fixed['sin2_th12'] = False
            if (n + 1) % 20 == 0:
                print(f"[geo2d]   {n + 1}/{len(mine)} done", flush=True)
            if (n + 1) % 25 == 0 or (n + 1) == len(mine):
                np.savez(ckpt, geo_cpd=geo_cpd_grid, geo_pull=pull_grid,
                         dm2_21=dm_grid, chi2=chi2_part, tot_nom=tot_nom,
                         best_pull=pull_best, best_dm2_21=dm_best,
                         done=n + 1, total=len(mine))

        chunk_path = f"{prefix}_profiled_chunk{chunk}of{nchunks}.npz"
        np.savez(chunk_path, geo_cpd=geo_cpd_grid, geo_pull=pull_grid,
                 dm2_21=dm_grid, chi2=chi2_part, tot_nom=tot_nom,
                 best_pull=pull_best, best_dm2_21=dm_best)
        print(f"[geo2d] wrote chunk → {chunk_path}")

        # Restore m to the global best fit BEFORE plotting: the profiled loop
        # left it at the last grid point (a non-minimum), which would both make
        # the mncontour overlay inside _plot fail and corrupt the downstream
        # MINOS / osc-param reporting. Re-seat best-fit values, unfix, re-fit.
        for k, val in enumerate(base):
            m.values[k] = val
        m.fixed[geo_name] = False
        m.fixed['dm2_21'] = False
        m.migrad()

        if make_gr:
            merged = chi2_part.copy()
            for f in glob.glob(f"{prefix}_profiled_chunk*of{nchunks}.npz"):
                d = np.load(f)
                mask = ~np.isnan(d['chi2'])
                merged[mask] = d['chi2'][mask]
            if np.isnan(merged).any():
                miss = int(np.isnan(merged).sum())
                print(f"[geo2d] WARNING: {miss} grid points still missing — "
                      f"run the remaining chunks, then re-run with "
                      f"geo_dm21_make_graph=True")
            _plot(merged, 'profiled')


def scan_th13_dm31_contour(m, chi2_func, args_juno, *, NMO_opt=True):
    """Conditional 2D Δχ² map of (sin²θ₁₃, Δm²₃₁): freeze every other parameter
    at the global best fit and vary only these two on a grid (fast — pure χ²
    evals, no per-point re-minimisation), same idea as the geo-vs-dm2_21
    ``conditional`` mode. The DYB dm²₃₁ pull, when active, enters ``chi2_func``,
    so the surface tightens along dm²₃₁ accordingly. Saves the Δχ² grid to npz +
    a PNG; to compare with/without the DYB pull, run twice (pull_dm2_31 on/off,
    different th13_dm31_tag) and overlay the two npz files.

    Config keys (fit_options, all optional):
      th13_dm31_2d: true       enable this map
      th13_dm31_n: 61          grid points per axis
      th13_dm31_nsigma: 4.0    half-width per axis (in Hesse σ)
      th13_dm31_tag: ''        filename tag (default: file_prefix)
    """
    names = list(m.parameters)
    if 'sin2_th13' not in names or 'dm2_31' not in names:
        print("[th13dm31] sin2_th13/dm2_31 not free in the fit — skipping")
        return
    base = list(m.values)
    ix, iy = names.index('sin2_th13'), names.index('dm2_31')

    n     = int(getattr(args_juno, 'th13_dm31_n', 61))
    nth   = int(getattr(args_juno, 'th13_dm31_nth13', n))
    ndm   = int(getattr(args_juno, 'th13_dm31_ndm',   n))
    nsig  = float(getattr(args_juno, 'th13_dm31_nsigma', 4.0))
    tag   = getattr(args_juno, 'th13_dm31_tag', '') or args_juno.file_prefix
    pull  = bool(getattr(args_juno, 'pull_dm2_31', False))
    xb, yb = float(base[ix]), float(base[iy])          # dm2_31 in ×1e-3 eV²
    ex, ey = float(m.errors['sin2_th13']), float(m.errors['dm2_31'])
    # Explicit ranges (th13_dm31_th13_range / _dm31_range) override the ±nσ box.
    # Use a wide dm2_31 range to expose the JUNO wiggle-degeneracy secondary
    # minima (present when the DYB dm2_31 pull is off).
    xr = getattr(args_juno, 'th13_dm31_th13_range', None)
    yr = getattr(args_juno, 'th13_dm31_dm31_range', None)
    xg = (np.linspace(float(xr[0]), float(xr[1]), nth) if xr
          else np.linspace(xb - nsig * ex, xb + nsig * ex, nth))
    yg = (np.linspace(float(yr[0]), float(yr[1]), ndm) if yr
          else np.linspace(yb - nsig * ey, yb + nsig * ey, ndm))

    outdir = getattr(args_juno, 'main_data_folder', '.') or '.'
    os.makedirs(outdir, exist_ok=True)
    prefix = f"{outdir}/th13_dm31_contour_{tag}_NO_{NMO_opt}"
    print(f"[th13dm31] conditional {nth}×{ndm} grid (sin²θ₁₃, Δm²₃₁) "
          f"th13=[{xg[0]:.4g},{xg[-1]:.4g}] dm31=[{yg[0]:.4g},{yg[-1]:.4g}], "
          f"pull_dm2_31={pull} ({nth*ndm} χ² evals)")

    t0   = datetime.now()
    chi2 = np.empty((nth, ndm))
    for i, xv in enumerate(xg):
        for j, yv in enumerate(yg):
            v = list(base); v[ix] = xv; v[iy] = yv
            chi2[i, j] = float(chi2_func(*v))
    dchi2 = chi2 - np.nanmin(chi2)
    print(f"[th13dm31] grid done in {datetime.now() - t0}")

    np.savez(f"{prefix}.npz", sin2_th13=xg, dm2_31=yg, chi2=chi2, dchi2=dchi2,
             best_sin2_th13=xb, best_dm2_31=yb, pull_dm2_31=pull,
             chi2min_global=float(m.fval))

    X, Y = np.meshgrid(xg, yg, indexing='ij')
    plt.figure(figsize=(6.5, 6))
    cl = plt.contour(X, Y, dchi2, levels=[2.30, 6.18, 11.83],
                     colors=['red', 'orange', 'green'], linewidths=1.7)
    plt.clabel(cl, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'},
               fontsize=9)
    plt.plot(xb, yb, 'k*', ms=14, label='best fit')
    plt.xlabel(r'$\sin^2\theta_{13}$')
    plt.ylabel(r'$\Delta m^2_{31}$ [$10^{-3}\,$eV$^2$]')
    plt.title(f'(sin²θ₁₃, Δm²₃₁) conditional  {tag}  (DYB pull={pull})',
              fontsize=10)
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{prefix}.png", dpi=200); plt.close()
    print(f"[th13dm31] wrote {prefix}.png and {prefix}.npz")


def scan_solar_contour(m, args_juno, *, NMO_opt=True):
    """Profiled 2D solar contour in (sin²θ₁₂, Δm²₂₁) via Minuit ``mncontour``.

    Unlike the legacy grid scan (``scan.py`` + ``grid_scan_results.py``, which
    evaluates a covariance-matrix χ² on a brute-force grid and marginalises with
    ``np.amin``), this traces the TRUE joint confidence regions: every point on
    each contour is a full re-minimisation with all other parameters — every
    nuisance AND the other oscillation params — profiled out. It reuses the fit
    minimum already found in ``m`` (no extra base fit), so its only cost is the
    contour tracing itself.

    Cost scales as ``size × len(cls)`` re-minimisations, so a coarse ``size``
    gives a quick, linearly-extrapolable time estimate for a fine one. Each
    contour and the total are timed and printed.

    Config keys (all under fit_options, all optional):
      solar_mncontour: true          enable this contour
      solar_mncontour_size: 100      points traced per contour (cost knob)
      solar_mncontour_cls: [1,2,3]   confidence levels (σ) to draw
      solar_tag: ''                  filename tag (default: file_prefix)
    """
    from matplotlib.lines import Line2D

    # mncontour refuses on a minimum flagged invalid (edm above goal). Some
    # configs (cached prefit, etc.) leave the fit marginally invalid; re-tighten
    # with a higher strategy before tracing, then restore a cheaper strategy.
    if not m.fmin.is_valid:
        print(f"[solar] fit minimum flagged invalid "
              f"(edm={m.fmin.edm:.2g} > goal {m.fmin.edm_goal:.2g}); "
              f"re-tightening with strategy 2")
        m.strategy = 2
        for _ in range(3):
            m.migrad()
            if m.fmin.is_valid:
                break
        m.strategy = 1
        print(f"[solar] after re-tighten: valid={m.fmin.is_valid}, "
              f"edm={m.fmin.edm:.2g} (strategy reset to 1)")

    size = int(getattr(args_juno, 'solar_mncontour_size', 100))
    cls  = getattr(args_juno, 'solar_mncontour_cls', None) or [1, 2, 3]
    cls  = [int(c) for c in cls]
    tag  = getattr(args_juno, 'solar_tag', '') or args_juno.file_prefix

    th_best = float(m.values['sin2_th12'])
    dm_best = float(m.values['dm2_21'])

    outdir = getattr(args_juno, 'main_data_folder', '.') or '.'
    os.makedirs(outdir, exist_ok=True)
    prefix = f"{outdir}/solar_contour_{tag}_NO_{NMO_opt}"

    print(f"[solar] mncontour: profiled (sin²θ₁₂, Δm²₂₁) contours for "
          f"cls={cls}σ, size={size} points each "
          f"({size * len(cls)} re-minimisations total)")

    saved    = {}
    contours = []                                  # (clv, col, px, py)
    colours  = {1: 'red', 2: 'orange', 3: 'green'}
    t_all    = datetime.now()
    for clv in cls:
        col = colours.get(clv, 'blue')
        t0  = datetime.now()
        try:
            pts = np.asarray(m.mncontour('sin2_th12', 'dm2_21',
                                         cl=clv, size=size))
        except Exception as e:
            print(f"[solar] mncontour cl={clv}σ failed: {e}")
            continue
        dt = datetime.now() - t0
        print(f"[solar]   cl={clv}σ: {len(pts)} points in {dt} "
              f"({dt.total_seconds() / max(len(pts), 1):.3f} s/point)")
        if pts.ndim != 2 or not len(pts):
            continue
        px = np.append(pts[:, 0], pts[0, 0])       # close the loop
        py = np.append(pts[:, 1], pts[0, 1])
        contours.append((clv, col, px, py))
        saved[f'sin2_th12_{clv}sig'] = px
        saved[f'dm2_21_{clv}sig']    = py

    # ── 1D profiled Δχ² for the side panels ──────────────────────────────
    # mnprofile is the TRUE profile: every scan point re-minimises over all
    # other parameters (nuisances + the other osc param), so these marginals
    # are consistent with the 2D contours above. subtract the global minimum.
    do_prof  = bool(getattr(args_juno, 'solar_profiles', True))
    psize    = int(getattr(args_juno, 'solar_profile_size', 40))
    pbound   = float(getattr(args_juno, 'solar_profile_bound', 4.0))
    chi2_min = float(m.fval)
    prof     = {}
    if do_prof:
        for name in ('sin2_th12', 'dm2_21'):
            t0 = datetime.now()
            try:
                x, y, _ok = m.mnprofile(name, size=psize, bound=pbound)
            except Exception as e:
                print(f"[solar] mnprofile {name} failed: {e}")
                continue
            x = np.asarray(x, float)
            dchi2 = np.asarray(y, float) - chi2_min
            prof[name] = (x, dchi2)
            dt = datetime.now() - t0
            print(f"[solar]   profile {name}: {len(x)} pts in {dt} "
                  f"({dt.total_seconds() / max(len(x), 1):.3f} s/point)")
            saved[f'profile_{name}_x']     = x
            saved[f'profile_{name}_dchi2'] = dchi2

    # ── corner-style figure: 2D contour + marginal Δχ² profiles ──────────
    have_prof = do_prof and ('sin2_th12' in prof) and ('dm2_21' in prof)
    if have_prof:
        fig = plt.figure(figsize=(7.5, 7.5))
        gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1],
                               height_ratios=[1, 4], wspace=0.06, hspace=0.06)
        ax_main  = fig.add_subplot(gs[1, 0])
        ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_leg   = fig.add_subplot(gs[0, 1]); ax_leg.axis('off')
    else:
        fig, ax_main = plt.subplots(figsize=(7, 6))
        ax_top = ax_right = ax_leg = None

    handles = []
    for clv, col, px, py in contours:
        ax_main.plot(px, py, '-', color=col, lw=1.8)
        handles.append(Line2D([0], [0], color=col, lw=1.8,
                              label=f'{clv}$\\sigma$'))
    star, = ax_main.plot(th_best, dm_best, 'k*', ms=15, label='best fit')
    handles.append(star)
    ax_main.set_xlabel(r"$\sin^2\theta_{12}$")
    ax_main.set_ylabel(r"$\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]")
    ax_main.grid(True, alpha=0.3)
    if contours:                                   # lock view to contour extent
        ax = np.concatenate([c[2] for c in contours])
        ay = np.concatenate([c[3] for c in contours])
        xpad = 0.05 * (np.ptp(ax) or 1.0)
        ypad = 0.05 * (np.ptp(ay) or 1.0)
        ax_main.set_xlim(ax.min() - xpad, ax.max() + xpad)
        ax_main.set_ylim(ay.min() - ypad, ay.max() + ypad)

    _levels = [(1, 'red'), (4, 'orange'), (9, 'green')]    # Δχ² 1/2/3σ (1 dof)
    if ax_top is not None:
        xg, dchi = prof['sin2_th12']
        ax_top.plot(xg, dchi, 'k-', lw=1.5)
        for lv, c in _levels:
            ax_top.axhline(lv, color=c, ls=':', lw=1.0)
        ax_top.set_ylabel(r"$\Delta\chi^2$")
        ax_top.set_ylim(0, 10)
        ax_top.grid(True, alpha=0.3)
        plt.setp(ax_top.get_xticklabels(), visible=False)
    if ax_right is not None:
        yg, dchiy = prof['dm2_21']
        ax_right.plot(dchiy, yg, 'k-', lw=1.5)
        for lv, c in _levels:
            ax_right.axvline(lv, color=c, ls=':', lw=1.0)
        ax_right.set_xlabel(r"$\Delta\chi^2$")
        ax_right.set_xlim(0, 10)
        ax_right.grid(True, alpha=0.3)
        plt.setp(ax_right.get_yticklabels(), visible=False)

    if ax_leg is not None:
        ax_leg.legend(handles=handles, loc='center', fontsize=10, frameon=False)
    else:
        ax_main.legend(handles=handles, loc='upper right', fontsize=9)

    png = f"{prefix}_mncontour.png"
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    np.savez(f"{prefix}_mncontour.npz",
             best_sin2_th12=th_best, best_dm2_21=dm_best,
             size=size, cls=np.array(cls), **saved)
    print(f"[solar] total time (contours+profiles): {datetime.now() - t_all}")
    print(f"[solar] wrote {png} and {prefix}_mncontour.npz")


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
    out = {'nuis_best': z['nuis_best'],
           'nuis_cov':  z['nuis_cov'],
           'C_reco':    None if C.size == 0 else C}
    # optional full DYB+TAO prefit best fit (newer caches)
    if 'pre_names' in z.files:
        out.update(pre_names=[str(n) for n in z['pre_names']],
                   pre_values=z['pre_values'], pre_errors=z['pre_errors'],
                   pre_fval=float(z['pre_fval']))
    return out
