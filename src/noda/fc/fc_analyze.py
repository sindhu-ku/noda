#!/usr/bin/env python3
"""
fc_analyze.py — aggregate per-task FC CSVs into critical-Δχ² curves
and NMO test-statistic distributions.

Usage:
  python -m fc_analyze --input-dir fc_runs/dm31_scan/ \\
                                  --output-dir fc_plots/

Reads every fc_task_*.csv in --input-dir, concatenates, and produces:
  * dm²₃₁ FC critical Δχ² vs. true_dm31 (68%, 95%, 99.7% CL)
  * Comparison to the χ²₁ asymptotic (1, 4, 9)
  * NMO test stat T = chi²_IO − chi²_NO histograms under each true ordering
  * Median √Δχ² sensitivity for NMO
  * A combined CSV summary
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cls", default="0.6827,0.9545,0.9973",
                   help="Comma-separated CLs for FC critical values")
    p.add_argument("--observed-csv", default=None,
                   help="If provided, overlay observed Δχ² from this CSV "
                        "(columns: true_dm31, TS_obs)")
    return p.parse_args()


def load_all(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "fc_task_*.csv")))
    if not files:
        raise SystemExit(f"No fc_task_*.csv found in {input_dir}")
    print(f"[FC] reading {len(files)} files from {input_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"[FC] loaded {len(df)} rows; modes: {df['mode'].unique()}")
    return df


# ─── dm²₃₁ FC critical curve ─────────────────────────────────────────────────
def analyze_dm31(df: pd.DataFrame, cls, out_dir: str, observed_csv=None):
    sub = df[df['mode'] == 'dm31'].copy()
    if sub.empty:
        return
    valid = sub[(sub['free_valid'] == 1) & (sub['fix_valid'] == 1)
                & np.isfinite(sub['TS_dm31']) & (sub['TS_dm31'] >= 0)].copy()
    print(f"[FC] dm31: {len(valid)}/{len(sub)} valid toys")

    # group by (true_dm31, true_NO) and quantile
    rows = []
    for (dm31, no), g in valid.groupby(['true_dm31', 'true_NO']):
        ts = g['TS_dm31'].values
        if len(ts) < 30:
            print(f"[WARN] only {len(ts)} toys at dm31={dm31}, NO={no}")
        out = {'true_dm31': dm31, 'true_NO': no, 'n_toys': len(ts)}
        for cl in cls:
            out[f'crit_{int(cl*1e4):04d}'] = float(np.quantile(ts, cl))
        rows.append(out)
    summary = pd.DataFrame(rows).sort_values(['true_NO', 'true_dm31'])
    summary.to_csv(os.path.join(out_dir, "fc_dm31_critical.csv"), index=False)

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for no in sorted(summary['true_NO'].unique()):
        s = summary[summary['true_NO'] == no]
        for cl in cls:
            col = f'crit_{int(cl*1e4):04d}'
            ax.plot(s['true_dm31'], s[col], '-o',
                    label=f"NO={no} CL={cl:.4f}")
    # asymptotic χ²₁ levels
    for cl in cls:
        ax.axhline(chi2_dist.ppf(cl, df=1), ls=':', color='gray', alpha=0.5)
    ax.set_xlabel(r"true $\Delta m^2_{31}\ (\times 10^{-3}\ \mathrm{eV}^2)$")
    ax.set_ylabel(r"FC critical $\Delta\chi^2$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fc_dm31_critical.pdf"))

    # If observed available, build the FC interval
    if observed_csv:
        obs = pd.read_csv(observed_csv)
        # Interpolate critical at each true_dm31, find points where TS_obs < critical
        for no in sorted(summary['true_NO'].unique()):
            s = summary[summary['true_NO'] == no]
            for cl in cls:
                col = f'crit_{int(cl*1e4):04d}'
                # naive: a true_dm31 is "in" the interval if observed Δχ² there
                # is below the critical for that true_dm31
                merged = obs.merge(s, on='true_dm31', how='inner')
                inside = merged[merged['TS_obs'] < merged[col]]
                if not inside.empty:
                    print(f"NO={no} CL={cl}: interval covers "
                          f"[{inside['true_dm31'].min():.4f}, "
                          f"{inside['true_dm31'].max():.4f}] × 1e-3 eV²")


# ─── NMO sensitivity ──────────────────────────────────────────────────────────
def analyze_nmo(df: pd.DataFrame, out_dir: str):
    sub = df[df['mode'] == 'nmo'].copy()
    if sub.empty:
        return
    valid = sub[(sub['NO_valid'] == 1) & (sub['IO_valid'] == 1)
                & np.isfinite(sub['T_nmo'])].copy()
    print(f"[FC] nmo: {len(valid)}/{len(sub)} valid toys")

    T_NO = valid.loc[valid['true_NO'] == 1, 'T_nmo'].values
    T_IO = valid.loc[valid['true_NO'] == 0, 'T_nmo'].values

    sigma_NO = np.sqrt(np.median(T_NO))   if len(T_NO) else np.nan
    sigma_IO = np.sqrt(-np.median(T_IO))  if len(T_IO) else np.nan

    summary = pd.DataFrame([{
        'sigma_if_true_NO': sigma_NO,
        'sigma_if_true_IO': sigma_IO,
        'median_T_true_NO': np.median(T_NO) if len(T_NO) else np.nan,
        'median_T_true_IO': np.median(T_IO) if len(T_IO) else np.nan,
        'n_toys_NO':        len(T_NO),
        'n_toys_IO':        len(T_IO),
    }])
    summary.to_csv(os.path.join(out_dir, "fc_nmo_summary.csv"), index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(T_NO):
        ax.hist(T_NO, bins=50, alpha=0.5, label=f'true NO ({len(T_NO)})',
                color='C0', density=True)
    if len(T_IO):
        ax.hist(T_IO, bins=50, alpha=0.5, label=f'true IO ({len(T_IO)})',
                color='C3', density=True)
    ax.axvline(0, color='k', lw=0.7)
    ax.set_xlabel(r"$T = \chi^2_{IO} - \chi^2_{NO}$")
    ax.set_ylabel("density")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fc_nmo_distribution.pdf"))
    
def report_dm31_overall(df: pd.DataFrame, out_dir: str):
    """One median ± 1σ per ordering, pooled across all dm31-mode toys."""
    sub = df[df['mode'] == 'dm31'].copy()
    valid = sub[(sub['free_valid'] == 1) & np.isfinite(sub['best_dm31'])]
    if valid.empty:
        return

    rows = []
    print(f"\n{'ord':<4} {'median':>10} {'+1σ':>8} {'−1σ':>8} "
          f"{'mean':>10} {'std':>8} {'n':>6}")
    print("─" * 60)

    for no, g in valid.groupby('true_NO'):
        bf = g['best_dm31'].values
        lo, med, hi = np.quantile(bf, [0.1587, 0.5, 0.8413])
        label = "NO" if no else "IO"
        print(f"{label:<4} {med:>10.4f} {hi-med:>+8.4f} {-(med-lo):>+8.4f} "
              f"{np.mean(bf):>10.4f} {np.std(bf, ddof=1):>8.4f} {len(bf):>6d}")
        rows.append({
            'ordering':         label,
            'best_dm31_median': med,
            'plus_1sig':        hi - med,
            'minus_1sig':       med - lo,
            'best_dm31_mean':   np.mean(bf),
            'best_dm31_std':    np.std(bf, ddof=1),
            'n_toys':           len(bf),
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "fc_dm31_overall.csv"), index=False)
def plot_dm31_profile(df: pd.DataFrame, out_dir: str, nbins: int = 60):
    """
    Empirical Δχ² profile of dm²₃₁ from the toy best-fit distribution,
    one curve per ordering. Overlays a Gaussian (parabolic) fit.
    Horizontal lines mark Δχ² = 1, 4, 9 (1σ, 2σ, 3σ for one parameter).
    """
    from scipy.optimize import curve_fit

    sub = df[df['mode'] == 'dm31'].copy()
    valid = sub[(sub['free_valid'] == 1) & np.isfinite(sub['best_dm31'])]
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: 'C0', 0: 'C3'}     # NO blue, IO red

    for no, g in valid.groupby('true_NO'):
        label = "NO" if no else "IO"
        bf = g['best_dm31'].values
        mu, sg = np.mean(bf), np.std(bf, ddof=1)

        # Empirical Δχ² from binned density
        counts, edges = np.histogram(bf, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = counts / (counts.sum() * np.diff(edges))
        with np.errstate(divide='ignore'):
            nll = -2.0 * np.log(density)
        nll -= np.nanmin(nll)

        # Only keep bins with enough toys to avoid noisy tails
        ok = np.isfinite(nll) & (counts >= 5)
        ax.plot(centers[ok], nll[ok], 'o', ms=3, color=colors[no],
                label=f'{label}  toys: {mu:.4f} ± {sg:.4f}  (n={len(bf)})')

        # Parabolic (= Gaussian) overlay, fit on the central Δχ² < 6 region
        def parab(x, x0, s, c):
            return ((x - x0) / s)**2 + c
        try:
            mask = ok & (nll < 6)
            popt, _ = curve_fit(parab, centers[mask], nll[mask],
                                p0=[mu, sg, 0.0])
            xs = np.linspace(centers[ok].min(), centers[ok].max(), 400)
            ax.plot(xs, parab(xs, *popt) - popt[2], '-', color=colors[no],
                    lw=1.2, alpha=0.7,
                    label=f'{label}  fit: {popt[0]:.4f} ± {abs(popt[1]):.4f}')
            # Mark best fit and ±1σ
            ax.axvline(popt[0], color=colors[no], lw=0.8, alpha=0.5)
            ax.axvspan(popt[0] - abs(popt[1]), popt[0] + abs(popt[1]),
                       color=colors[no], alpha=0.07)
        except Exception as e:
            print(f"[plot_dm31_profile] parabola fit failed for {label}: {e}")

    for level, ls, txt in [(1, '--', '1σ'), (4, ':', '2σ'), (9, '-.', '3σ')]:
        ax.axhline(level, ls=ls, color='gray', alpha=0.5)
        ax.text(0.005, level + 0.15, txt,
                transform=ax.get_yaxis_transform(),
                color='gray', fontsize=9, va='bottom')

    ax.set_xlabel(r'$\Delta m^2_{31}\ (\times 10^{-3}\ \mathrm{eV}^2)$')
    ax.set_ylabel(r'$\Delta\chi^2$')
    ax.set_ylim(0, 12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper center', ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fc_dm31_profile.pdf'))
    fig.savefig(os.path.join(out_dir, 'fc_dm31_profile.png'), dpi=150)
    plt.close(fig)

def analyze_nmo_significance(df: pd.DataFrame, out_dir: str):
    """
    Median expected NMO significance from FC toys, both Asimov (√|median T|)
    and FC p-value crossing. Reconstructs the signed T from chi2_NO and chi2_IO.
    """
    from scipy.stats import norm

    sub = df[df['mode'] == 'nmo'].copy()
    valid = sub[(sub['NO_valid'] == 1) & (sub['IO_valid'] == 1)
                & np.isfinite(sub['chi2_NO']) & np.isfinite(sub['chi2_IO'])].copy()
    if valid.empty:
        print("[FC] no valid NMO toys"); return

    valid['T_signed'] = valid['chi2_IO'] - valid['chi2_NO']
    T_NO = valid.loc[valid['true_NO'] == 1, 'T_signed'].values
    T_IO = valid.loc[valid['true_NO'] == 0, 'T_signed'].values

    print(f"\n[FC] NMO toys: n(true NO) = {len(T_NO)},  n(true IO) = {len(T_IO)}")
    if not len(T_NO) or not len(T_IO):
        print("[FC] need toys under BOTH orderings to compute significance")
        return

    # ── medians under each true ordering ─────────────────────────────────────
    med_NO = float(np.median(T_NO))
    med_IO = float(np.median(T_IO))

    # ── Method 1: Asimov / √median ───────────────────────────────────────────
    sig_asimov_NO = np.sqrt(max(0,  med_NO))   # σ to reject IO if NO is true
    sig_asimov_IO = np.sqrt(max(0, -med_IO))   # σ to reject NO if IO is true

    # ── Method 2: FC p-value crossing ────────────────────────────────────────
    # If NO is true: the typical experiment yields T = med_NO.
    # The IO p-value of this observation is the fraction of true-IO toys with T ≥ med_NO.
    n_xover_IO = int((T_IO >= med_NO).sum())
    n_xover_NO = int((T_NO <= med_IO).sum())
    p_NO = n_xover_IO / len(T_IO)
    p_IO = n_xover_NO / len(T_NO)

    def p_to_sigma(p, n_alt):
        # one-sided; if no crossings, report a lower bound based on 1/n_alt
        if p == 0:
            return norm.isf(1.0 / n_alt), True   # σ_min for "p < 1/n"
        return norm.isf(p), False

    sig_fc_NO, bound_NO = p_to_sigma(p_NO, len(T_IO))
    sig_fc_IO, bound_IO = p_to_sigma(p_IO, len(T_NO))

    # ── print summary ────────────────────────────────────────────────────────
    print("\n  NMO significance (one-sided, median expected):")
    print(f"  Truth = NO  →  reject IO at:")
    print(f"      Asimov √|med T| :  σ = √{med_NO:.2f}  =  {sig_asimov_NO:.2f}")
    print(f"      FC p-value      :  σ = {sig_fc_NO:.2f}"
          + (f"  (LOWER BOUND, no crossing in {len(T_IO)} toys, p<{1/len(T_IO):.1e})"
             if bound_NO else f"  (p = {p_NO:.3e}, {n_xover_IO} crossings)"))
    print(f"  Truth = IO  →  reject NO at:")
    print(f"      Asimov √|med T| :  σ = √{-med_IO:.2f}  =  {sig_asimov_IO:.2f}")
    print(f"      FC p-value      :  σ = {sig_fc_IO:.2f}"
          + (f"  (LOWER BOUND, no crossing in {len(T_NO)} toys, p<{1/len(T_NO):.1e})"
             if bound_IO else f"  (p = {p_IO:.3e}, {n_xover_NO} crossings)"))

    pd.DataFrame([
        {'truth': 'NO', 'median_T': med_NO,
         'sigma_asimov': sig_asimov_NO, 'sigma_fc': sig_fc_NO,
         'p_fc': p_NO, 'sigma_fc_is_lower_bound': bound_NO,
         'n_truth': len(T_NO), 'n_alt': len(T_IO)},
        {'truth': 'IO', 'median_T': med_IO,
         'sigma_asimov': sig_asimov_IO, 'sigma_fc': sig_fc_IO,
         'p_fc': p_IO, 'sigma_fc_is_lower_bound': bound_IO,
         'n_truth': len(T_IO), 'n_alt': len(T_NO)},
    ]).to_csv(os.path.join(out_dir, "fc_nmo_significance.csv"), index=False)

    # ── plot the T distributions ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = min(T_NO.min(), T_IO.min()), max(T_NO.max(), T_IO.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(T_NO, bins=bins, alpha=0.55, color='C0', density=True,
            label=f'true NO  (n={len(T_NO)})')
    ax.hist(T_IO, bins=bins, alpha=0.55, color='C3', density=True,
            label=f'true IO  (n={len(T_IO)})')
    ax.axvline(med_NO, color='C0', lw=1.5,
               label=f'median(T|NO)={med_NO:.2f}')
    ax.axvline(med_IO, color='C3', lw=1.5,
               label=f'median(T|IO)={med_IO:.2f}')
    ax.axvline(0, color='k', lw=0.7, alpha=0.6)
    ax.set_xlabel(r'$T = \chi^2_{IO} - \chi^2_{NO}$')
    ax.set_ylabel('density')
    ax.set_title(
        f'NMO sensitivity (FC):  '
        f'σ(NO)={sig_fc_NO:.2f}{"⁺" if bound_NO else ""},  '
        f'σ(IO)={sig_fc_IO:.2f}{"⁺" if bound_IO else ""}')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fc_nmo_significance.pdf"))
    fig.savefig(os.path.join(out_dir, "fc_nmo_significance.png"), dpi=150)
    plt.close(fig)

def main():
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    cls = [float(x) for x in args.cls.split(',')]
    df = load_all(args.input_dir)
    analyze_dm31(df, cls, args.output_dir, args.observed_csv)
    plot_dm31_profile(df, args.output_dir)  
    analyze_nmo(df, args.output_dir)
    analyze_nmo_significance(df, args.output_dir)
    report_dm31_overall(df, args.output_dir)
    print(f"[FC] done — outputs in {args.output_dir}")


if __name__ == "__main__":
    main()