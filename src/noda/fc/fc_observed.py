import argparse, os
from datetime import datetime
import numpy as np

from noda.fc.fc_toy import build_setups
from noda.fc.feldman_cousins import base_param, build_fit_bundles
from noda.fit import combined_fit as minuit_new
from noda.fit.minuit import load_prefit_cache

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True,
                   help="Config YAML or multi-phase manifest (as for run_noda)")
    p.add_argument("--prefit-cache", required=True)
    p.add_argument("--output",       required=True, help="output NPZ path")
    p.add_argument("--candidates",   type=float, nargs='+',
                   default=[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                   help="Absolute dm²₃₁ candidate values in 1e-3 eV²")
    return p.parse_args()

def main():
    cli = parse()
    t0 = datetime.now()

    # Setup is independent of the fit hypothesis, so build it once
    setups, policy = build_setups(cli.config)
    print(f"[FC-obs] Base setup ready in {datetime.now() - t0} "
          f"({len(setups)} phase/shell dataset(s))", flush=True)

    prefit = load_prefit_cache(cli.prefit_cache)
    print(f"[FC-obs] Loaded prefit", flush=True)

    def run_fit(true_no, fix_dm2_31_to=None):
        """One bundled JUNO-only fit at the requested NMO (fresh bundles)."""
        bundles, args_fit = build_fit_bundles(setups, force_NO=bool(true_no))
        return minuit_new.run_minuit_bundled(
            bundles, policy, args_fit,
            prefit_cache=prefit, toy_quiet=True,
            fix_dm2_31_to=fix_dm2_31_to)

    # Dictionaries and arrays to be saved to NPZ
    grid_best_fits = {}
    nominal_uncertainties = {}
    truncate_below_zero = []
    truncate_below_mone = []
    param_names = []  # Initialize global parameter names array

    # Loop over both Normal (1) and Inverted (0) Orderings
    for true_no in [1, 0]:
        ordering_name = "NO" if true_no else "IO"
        print(f"\n" + "="*50)
        print(f"[FC-obs] Starting scan for {ordering_name} (NMO_opt={true_no})")
        print("="*50)

        # ── 1. Free Fit (Required for NMO baseline toys) ──
        print(f"[FC-obs] Free fit for {ordering_name} central values...", flush=True)
        m_free = run_fit(true_no)

        # Save the global best fit using the integer NMO as the key (1 or 0)
        grid_best_fits[true_no] = {
            'true_dm31':   float(m_free.values['dm2_31']),
            'true_NO':     true_no,
            'values':      {p: float(m_free.values[p]) for p in m_free.parameters},
            'errors':      {p: float(m_free.errors[p]) for p in m_free.parameters},
            'cov':         np.array(m_free.covariance) if m_free.covariance is not None else None,
        }

        # Extract limits/nuisances AND param_names only once (during the NO pass)
        if true_no == 1:
            param_names = list(m_free.parameters)
            args0       = setups[0]['args_juno']
            backgrounds = setups[0]['backgrounds']

            nominal_uncertainties["me_rho"]    = args0.me_rho_scale
            nominal_uncertainties["ThU_ratio"] = args0.Th_U_ratio_err
            nominal_uncertainties["Nrea"]      = args0.r2_unc
            # Per-phase/shell parameters carry a '__p…__s…' suffix; match on
            # the base name so every dataset's copy gets its uncertainty.
            for name in param_names:
                base = base_param(name)
                if base.startswith('core'):
                    nominal_uncertainties[name] = args0.core_flux_unc
                if base.startswith('efiss'):
                    nominal_uncertainties[name] = args0.efission_err[int(base[-1])]*1e-2
                if base.startswith('eff'):
                    nominal_uncertainties[name] = args0.eff_unc
                if base.startswith('snf'):
                    nominal_uncertainties[name] = args0.snf_unc
                if args0.NL == "pull" and base.startswith('NL'):
                    nominal_uncertainties[name] = (args0.ene_scale_unc
                                                   if base == 'NL4'
                                                   else args0.NL_sd)
                if args0.NL == "pull" and base.startswith('abc'):
                    nominal_uncertainties[name] = {'abc0': args0.a_err,
                                                   'abc1': args0.b_err,
                                                   'abc2': args0.c_err}[base]
                if base.startswith('bkg_'):
                    info = backgrounds["JUNO"].get(base[4:], {})
                    nominal_uncertainties[name] = info.get("rate_unc", 1.0)

            truncate_below_zero = [
                p for p in m_free.parameters
                if m_free.limits[p] is not None
                and m_free.limits[p][0] == 0.0
            ]

            truncate_below_mone = [
                p for p in m_free.parameters
                if m_free.limits[p] is not None
                and m_free.limits[p][0] == -1.0
            ]


        # ── 2. Fixed Grid Fits (Required for dm31 FC scan toys) ──
        print(f"[FC-obs] Running fixed grid fits for {ordering_name}...", flush=True)
        for cand in cli.candidates:
            # Apply sign: positive for NO, negative for IO
            fit_cand = cand if true_no else -cand

            m_f = run_fit(true_no, fix_dm2_31_to=fit_cand)

            # Save using the signed float as the lookup key (e.g., 2.50 or -2.50)
            grid_best_fits[fit_cand] = {
                'true_dm31':   fit_cand,
                'true_NO':     true_no,
                'values':      {p: float(m_f.values[p]) for p in m_f.parameters},
                'errors':      {p: float(m_f.errors[p]) for p in m_f.parameters},
                'cov':         np.array(m_f.covariance) if m_f.covariance is not None else None,
            }

            print(f"[FC-obs]   Saved central values for dm31={fit_cand:+.4f}, fmin: {m_f.fval}", flush=True)

    # ── 3. Save to NPZ ──
    print("\n" + "="*50)
    os.makedirs(os.path.dirname(cli.output) or '.', exist_ok=True)
    np.savez_compressed(
        cli.output,
        grid_best_fits=np.array([grid_best_fits], dtype=object),
        truncate_below_zero=np.array(truncate_below_zero, dtype=str),
        truncate_below_mone=np.array(truncate_below_mone, dtype=str),
        nominal_uncertainties=np.array([nominal_uncertainties], dtype=object),
        param_names=np.array(param_names, dtype=str)
    )

    print(f"[FC-obs] Wrote {len(grid_best_fits)} parameter maps to {cli.output}")
    print(f"[FC-obs] Total time: {datetime.now() - t0}")

if __name__ == "__main__":
    main()
