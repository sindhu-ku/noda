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
import uproot

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cls", default="0.6827", #,0.9545,0.9973",
                   help="Comma-separated CLs for FC critical values")
    p.add_argument("--observed-root", default=None,
                   help="If provided, overlay observed Δχ² from this root "
                        "(columns: true_dm31, TS_obs)")
    p.add_argument("--t-obs-nmo", type=float, default=None,
               help="Observed T_nmo = chi2_IO - chi2_NO from your dataset")
    return p.parse_args()


def load_all(input_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "fc_task_*.csv")))
    if not files:
        raise SystemExit(f"No fc_task_*.csv found in {input_dir}")
    print(f"[FC] reading {len(files)} files from {input_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Coerce every non-'mode' column to numeric. A single malformed cell (empty
    # / 'nan' / partial write from a crashed toy) otherwise makes the whole
    # column object-dtype, which blows up np.isfinite ("ufunc 'isfinite' not
    # supported"). errors='coerce' turns bad cells into NaN so the downstream
    # isfinite/valid filters simply drop them.
    bad_before = {}
    for c in df.columns:
        if c == 'mode':
            continue
        coerced = pd.to_numeric(df[c], errors='coerce')
        n_bad = int(coerced.isna().sum() - df[c].isna().sum())
        if n_bad > 0:
            bad_before[c] = n_bad
        df[c] = coerced
    if bad_before:
        print(f"[FC] WARNING: non-numeric cells coerced to NaN (dropped): {bad_before}")
    print(f"[FC] loaded {len(df)} rows; modes: {df['mode'].unique()}")
    return df

def plot_fc_interval(df: pd.DataFrame, observed_root: str, cls, out_dir: str):
    # 1. Filter valid toys
    sub = df[df['mode'] == 'dm31'].copy()
    valid = sub[(sub['free_valid'] == 1) & (sub['fix_valid'] == 1)
                 & (sub['TS_dm31'] >= 0)]
    if valid.empty:
        return
    print(f"Valid toys: {len(valid)}/{len(sub)}")

    # 2. Build the coarse FC critical curves
    crit_rows = []
    for (dm31, no), g in valid.groupby(['true_dm31', 'true_NO']):
        if not no and dm31 > 0:
            dm31 = -1 * dm31
        print(f"Using {len(g)} toys for {dm31} {no}")
        row = {'true_dm31': dm31, 'true_NO': no}

        for cl in cls:
            row[f'crit_{int(cl*1e4):04d}'] = float(np.quantile(g['TS_dm31'], cl))
        crit_rows.append(row)

    crit = pd.DataFrame(crit_rows).sort_values(['true_NO', 'true_dm31'])

    # 3. Load the fine data profile directly from the ROOT file
    with uproot.open(f"{observed_root}:profiles/chi2_profile") as graph:
        x_vals, y_vals = graph.values()
        obs = pd.DataFrame({'true_dm31': x_vals, 'TS_obs': y_vals})
        obs['true_NO'] = obs['true_dm31'].mean() > 0

    # 4. Plot and extract intervals
    for no in sorted(crit['true_NO'].unique()):
        c = crit[crit['true_NO'] == no].copy()
        o = obs[obs['true_NO'] == no].copy()

        if o.empty:
            continue

        label = "NO" if no else "IO"

        c = c.sort_values('true_dm31', ascending=True)
        o = o.sort_values('true_dm31', ascending=True)

        x_fine   = o['true_dm31'].values
        ts_int   = o['TS_obs'].values
        best_idx = int(np.argmin(ts_int))
        best_dm31 = float(x_fine[best_idx])

        # ===== parabolic refinement of the minimum (match pedro.py) ==========
        # Fit a parabola through the 3 points around the discrete min and take
        # its vertex, for a sub-grid-precise best fit. Falls back to the
        # discrete min at the edges or if degenerate.
        i = best_idx
        if 0 < i < len(x_fine) - 1:
            x1, x2, x3 = x_fine[i-1], x_fine[i], x_fine[i+1]
            y1, y2, y3 = ts_int[i-1], ts_int[i], ts_int[i+1]
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            if denom != 0:
                A = (x3*(y2-y1) + x2*(y1-y3) + x1*(y3-y2)) / denom
                B = (x3*x3*(y1-y2) + x2*x2*(y3-y1) + x1*x1*(y2-y3)) / denom
                if A != 0:
                    best_dm31 = float(-B / (2*A))
        # ====================================================================

        print(f"\n  {label} hypothesis: best fit = {best_dm31:.4f} × 1e-3 eV²")

        if no:
            sign = 1.0
            color='steelblue'
        else:
            sign = -1.0
            color='brown'
        x_lo, x_hi = sign * 2.1, sign * 2.9

        fig, ax = plt.subplots(figsize=(5, 4))
        axins = ax.inset_axes([0.55, 0.6, 0.3, 0.3])

        # ── helper: find connected crossing interval around best fit ─────────
        def find_interval(x, ts_shifted, best_x):
            sc = np.where(np.diff(np.sign(ts_shifted)))[0]
            crossings = []
            for i in sc:
                x0, x1 = x[i], x[i + 1]
                y0, y1 = ts_shifted[i], ts_shifted[i + 1]
                crossings.append(x0 - y0 * (x1 - x0) / (y1 - y0))
            left  = [xc for xc in crossings if xc < best_x]
            right = [xc for xc in crossings if xc > best_x]
            lo = max(left)  if left  else None
            hi = min(right) if right else None
            return lo, hi

        # ── standard Wilks 1-sigma (TS == 1.0) ───────────────────────────────
        lo_w, hi_w = find_interval(x_fine, ts_int - 1.0, best_dm31)
        if lo_w is not None and hi_w is not None:
            print(f"    Standard 1-sigma profile (TS == 1.0 crossings): "
                  f"[{lo_w:.4f}, {hi_w:.4f}] "
                  f"(+{hi_w - best_dm31:.4f}/-{best_dm31 - lo_w:.4f})")
        elif lo_w is None:
            print(f"    Standard 1-sigma profile: no left crossing found")
        else:
            print(f"    Standard 1-sigma profile: no right crossing found")

        # ── FC critical curves ────────────────────────────────────────────────
        print(c)
        for cl, ls in zip(cls, ['-']):
            col = f'crit_{int(cl*1e4):04d}'
            avg_crit = float(np.mean(c[col]))   # mean FC critical chi2 over the scan
            ax.plot(c['true_dm31'], c[col], color='black', linestyle=':', marker='o',
                    markersize=3,
                    label=fr'FC crit. {cl*100:.2f}% ($\langle\chi^2\rangle$={avg_crit:.2f})')
            axins.plot(c['true_dm31'], c[col], color='black', linestyle=':', marker='o',
                       markersize=3)

            crit_int = np.interp(x_fine, c['true_dm31'], c[col])
            lo_fc, hi_fc = find_interval(x_fine, ts_int - crit_int, best_dm31)

            if lo_fc is not None and hi_fc is not None:
                ax.axvspan(lo_fc, hi_fc, color=color, alpha=0.06)
                axins.axvspan(lo_fc, hi_fc, color=color, alpha=0.06)

                up_err  = np.round(hi_fc - best_dm31, 4)
                low_err = np.round(best_dm31 - lo_fc, 4)
                print(f"    FC {cl*100:.2f}% CL: [{lo_fc:.4f}, {hi_fc:.4f}]   "
                      f"(+{up_err:.4f}/-{low_err:.4f})")
                ax.axvline(best_dm31, color=color, ls='-', lw=1,
                           label=fr"best fit = {best_dm31:.3f}$^{{+{up_err:.3f}}}_{{-{low_err:.3f}}}$")
                axins.axvline(best_dm31, color=color, ls='-', lw=1)
            elif lo_fc is None:
                print(f"    FC {cl*100:.2f}% CL: no left crossing found")
            else:
                print(f"    FC {cl*100:.2f}% CL: no right crossing found")

        # ── observed profile ──────────────────────────────────────────────────
        ax.plot(o['true_dm31'], o['TS_obs'], '-', color=color, lw=2,
                label='observed profile')
        axins.plot(o['true_dm31'], o['TS_obs'], '-', color=color, lw=2)

        # ── inset formatting ──────────────────────────────────────────────────
        axins.set_xlim(x_lo, x_hi)
        axins.set_ylim(0, 2)
        axins.tick_params(labelsize=6)
        axins.grid(alpha=0.3)

        ax.set_xlabel(fr'$\Delta m^2_{{31}}\ (\times 10^{{-3}}\ \mathrm{{eV}}^2)$')
        ax.set_ylabel(fr'$\Delta\chi^2$')
        ax.set_title(f'FC interval — {label} hypothesis')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 50)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fig.savefig(os.path.join(out_dir, f'fc_interval_{label}.png'), dpi=300)
        plt.close(fig)
        
def analyze_nmo_significance(df: pd.DataFrame, out_dir: str, T_obs: float = None):
    from scipy.stats import norm

    sub = df[df['mode'] == 'nmo'].copy()
    valid = sub[(sub['NO_valid'] == 1) & (sub['IO_valid'] == 1)]
       #         & np.isfinite(sub['chi2_NO']) & np.isfinite(sub['chi2_IO'])].copy()
    
    if valid.empty:
        print("[FC] no valid NMO toys")
        return

    # T = chi2_IO - chi2_NO. Positive favors NO, Negative favors IO.
    valid['T_signed'] = valid['chi2_IO'] - valid['chi2_NO']
    T_NO = valid.loc[valid['true_NO'] == 1, 'T_signed'].values
    T_IO = valid.loc[valid['true_NO'] == 0, 'T_signed'].values

    print(f"\n[FC] NMO toys: n(true NO) = {len(T_NO)}, n(true IO) = {len(T_IO)}")
    if not len(T_NO) or not len(T_IO):
        print("[FC] need toys under BOTH orderings")
        return

    med_NO = float(np.median(T_NO))
    med_IO = float(np.median(T_IO))
    
    # Asimov sigma is the simple sqrt(delta_chi2) approximation
    sig_asimov_NO = np.sqrt(max(0,  med_NO))
    sig_asimov_IO = np.sqrt(max(0, -med_IO))

    # Helper for p-value to Z-score conversion (ORSA Convention)
    def p_to_sigma_compare(p, n_alt):
        is_bound = False
        if p <= 0:
            p = 1.0 / n_alt  # Conservative estimate if zero crossings
            is_bound = True
        
        # One-sided: Standard Frequentist/FC approach
        s1 = norm.isf(p)
        # Two-sided: ORSA/Conservative approach (as per image formula)
        s2 = norm.isf(p / 2.0)
        return s1, s2, is_bound

    # ── EXPECTED SIGNIFICANCE (Median Sensitivity) ──────────────────────────
    n_xover_IO = int((T_IO >= med_NO).sum())
    n_xover_NO = int((T_NO <= med_IO).sum())
    
    p_exp_NO = n_xover_IO / len(T_IO)
    p_exp_IO = n_xover_NO / len(T_NO)

    s1_exp_NO, s2_exp_NO, b_exp_NO = p_to_sigma_compare(p_exp_NO, len(T_IO))
    s1_exp_IO, s2_exp_IO, b_exp_IO = p_to_sigma_compare(p_exp_IO, len(T_NO))

    # ── OBSERVED SIGNIFICANCE ────────────────────────────────────────────────
    s1_obs_IO = s2_obs_IO = p_obs_IO = None
    s1_obs_NO = s2_obs_NO = p_obs_NO = None

    if T_obs is not None:
        p_obs_IO = (T_IO >= T_obs).sum() / len(T_IO)
        p_obs_NO = (T_NO <= T_obs).sum() / len(T_NO)
        s1_obs_IO, s2_obs_IO, _ = p_to_sigma_compare(p_obs_IO, len(T_IO))
        s1_obs_NO, s2_obs_NO, _ = p_to_sigma_compare(p_obs_NO, len(T_NO))

    # ── PRINT RESULTS ────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'NMO SIGNIFICANCE REPORT':^70}")
    print("="*70)
    print(f"{'Hypothesis':<20} | {'p-value':<10} | {'1-sided σ':<10} | {'2-sided σ':<10}")
    print("-" * 70)
    
    label_NO = "Reject IO (True NO)"
    label_IO = "Reject NO (True IO)"
    
    print(f"{label_NO:<20} | {p_exp_NO:<10.4e} | {s1_exp_NO:<10.2f} | {s2_exp_NO:<10.2f} (Expected)")
    print(f"{label_IO:<20} | {p_exp_IO:<10.4e} | {s1_exp_IO:<10.2f} | {s2_exp_IO:<10.2f} (Expected)")

    if T_obs is not None:
        print("-" * 70)
        print(f"OBSERVED T = {T_obs:.3f} ({'favors NO' if T_obs > 0 else 'favors IO'})")
        print(f"{'Reject IO (Obs)':<20} | {p_obs_IO:<10.4e} | {s1_obs_IO:<10.2f} | {s2_obs_IO:<10.2f}")
        print(f"{'Reject NO (Obs)':<20} | {p_obs_NO:<10.4e} | {s1_obs_NO:<10.2f} | {s2_obs_NO:<10.2f}")
    print("="*70)

    # ── PLOTTING ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    lo, hi = min(T_NO.min(), T_IO.min()), max(T_NO.max(), T_IO.max())
    if T_obs is not None:
        lo, hi = min(lo, T_obs - 2), max(hi, T_obs + 2)
    
    bins = np.linspace(lo, hi, 70)
    ax.hist(T_NO, bins=bins, alpha=0.4, color='C0', density=True, label=f'True NO ($n={len(T_NO)}$)')
    ax.hist(T_IO, bins=bins, alpha=0.4, color='C3', density=True, label=f'True IO ($n={len(T_IO)}$)')

    print(np.std(T_NO), np.std(T_IO))
    ax.axvline(med_NO, color='C0', lw=2, ls='--', label=f'Med(T|NO): {med_NO:.1f}')
    ax.axvline(med_IO, color='C3', lw=2, ls='--', label=f'Med(T|IO): {med_IO:.1f}')
    ax.axvline(0, color='k', lw=1, alpha=0.5)

    if T_obs is not None:
        ax.axvline(T_obs, color='darkgreen', lw=2.5, label=f'$T_{{obs}}$ = {T_obs:.2f}')

    # Add text box with significance summary
    sig_text = (f"Expected Sensitivity (1-sided / 2-sided):\n"
                f"Reject IO: {s1_exp_NO:.2f}σ / {s2_exp_NO:.2f}σ\n"
                f"Reject NO: {s1_exp_IO:.2f}σ / {s2_exp_IO:.2f}σ")
    if T_obs is not None:
        sig_text += (f"\n\nObserved Significance:\n"
                     f"Reject IO: {s1_obs_IO:.2f}σ / {s2_obs_IO:.2f}σ")
    
    ax.text(0.02, 0.97, sig_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    ax.set_title('NMO Test Statistic Distributions ($T = \chi^2_{IO} - \chi^2_{NO}$)')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fc_nmo_significance.png"), dpi=300)
    plt.close(fig)

    # ── SAVE TO CSV ──────────────────────────────────────────────────────────
    pd.DataFrame([
        {'test': 'Reject IO', 'p_value': p_exp_NO, 'sigma_1side': s1_exp_NO, 'sigma_2side': s2_exp_NO, 'type': 'expected'},
        {'test': 'Reject NO', 'p_value': p_exp_IO, 'sigma_1side': s1_exp_IO, 'sigma_2side': s2_exp_IO, 'type': 'expected'},
        {'test': 'Reject IO', 'p_value': p_obs_IO, 'sigma_1side': s1_obs_IO, 'sigma_2side': s2_obs_IO, 'type': 'observed'},
        {'test': 'Reject NO', 'p_value': p_obs_NO, 'sigma_1side': s1_obs_NO, 'sigma_2side': s2_obs_NO, 'type': 'observed'},
    ]).to_csv(os.path.join(out_dir, "fc_nmo_significance_summary.csv"), index=False)
    T_crit = np.median(T_NO)          # from your toys
    crossings = (T_IO >= T_crit).sum()
    p_est = crossings / len(T_IO)
    toys_for_100 = int(100 / p_est) if p_est > 0 else float('inf')
    print(f"Current crossings: {crossings}/{len(T_IO)}")
    print(f"Estimated toys needed for 100 crossings: {toys_for_100}")
    print(f"Estimated jobs still needed: {max(0, toys_for_100 - len(T_IO)) // 50}")
def plot_nmo_gaussian_comparison(df: pd.DataFrame, out_dir: str):
    """
    Overlay a Gaussian (built from each distribution's own median and width)
    on the two NMO test-statistic histograms, purely to show that the T|NO
    and T|IO distributions are NOT Gaussian / symmetric.

    Deliberately minimal: just the two histograms and their matching
    Gaussians — no median lines, no T_obs, no significance numbers.
    """
    from scipy.stats import norm

    sub = df[df['mode'] == 'nmo'].copy()
    valid = sub[(sub['NO_valid'] == 1) & (sub['IO_valid'] == 1)
                & np.isfinite(sub['chi2_NO']) & np.isfinite(sub['chi2_IO'])].copy()
    if valid.empty:
        print("[FC] no valid NMO toys — skipping gaussian comparison")
        return

    valid['T_signed'] = valid['chi2_IO'] - valid['chi2_NO']
    T_NO = valid.loc[valid['true_NO'] == 1, 'T_signed'].values
    T_IO = valid.loc[valid['true_NO'] == 0, 'T_signed'].values
    if not len(T_NO) or not len(T_IO):
        print("[FC] need toys under BOTH orderings — skipping gaussian comparison")
        return

    lo  = min(T_NO.min(), T_IO.min())
    hi  = max(T_NO.max(), T_IO.max())
    bins = np.linspace(lo, hi, 70)
    x    = np.linspace(lo, hi, 500)

    fig, ax = plt.subplots(figsize=(9, 6))
    for T, color, label in ((T_NO, 'C0', 'True NO'),
                            (T_IO, 'C3', 'True IO')):
        # Gaussian uses the distribution's OWN median (location) and width.
        mu    = float(np.median(T))
        sigma = float(np.std(T))
        ax.hist(T, bins=bins, density=True, alpha=0.4, color=color, label=label)
        if sigma > 0:
            ax.plot(x, norm.pdf(x, mu, sigma), color=color, lw=2)

    ax.set_title(r'NMO Test Statistic ($T = \chi^2_{IO} - \chi^2_{NO}$)')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fc_nmo_gaussian_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"[FC] saved {os.path.join(out_dir, 'fc_nmo_gaussian_comparison.png')}")


def main():
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)
    cls = [float(x) for x in args.cls.split(',')]
    df = load_all(args.input_dir)

    if 'nmo' in df['mode'].unique():
        plot_nmo_gaussian_comparison(df, args.output_dir)
    if args.t_obs_nmo:
        analyze_nmo_significance(df, args.output_dir, T_obs=args.t_obs_nmo)
    if args.observed_root:
        plot_fc_interval(df, args.observed_root, cls, args.output_dir)
    print(f"[FC] done — outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
