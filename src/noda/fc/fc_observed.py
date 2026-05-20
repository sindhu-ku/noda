import argparse, os
from copy import deepcopy
from datetime import datetime
import numpy as np

from noda.fc.fc_toy import (build_args_from_yaml, build_setup)
from noda.fit import combined_fit as minuit_new
from noda.fit.minuit import load_prefit_cache

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--prefit-cache", required=True)
    p.add_argument("--output",       required=True, help="output NPZ path")
    p.add_argument("--candidates",   type=float, nargs='+',
                   default=[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                   help="Absolute dm²₃₁ candidate values in 1e-3 eV²")
    return p.parse_args()

def main():
    cli = parse()
    t0 = datetime.now()

    args_juno, args_dyb, args_tao = build_args_from_yaml(cli.config)
    args_juno_fit = deepcopy(args_juno)
    args_juno_fit.run_hesse = True  # Ensure Hesse is run so covariance matrix is populated

    # Setup is independent of the fit hypothesis, so build it once
    setup = build_setup(args_juno, args_dyb, args_tao)
    print(f"[FC-obs] Base setup ready in {datetime.now() - t0}", flush=True)

    common_kwargs = dict(setup)
    prefit = load_prefit_cache(cli.prefit_cache)
    print(f"[FC-obs] Loaded prefit", flush=True)

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

        # Update the fit arguments for the current ordering
        args_juno_fit.NMO_opt = bool(true_no)
        common_kwargs['args_juno'] = args_juno_fit
        
        # ── 1. Free Fit (Required for NMO baseline toys) ──
        print(f"[FC-obs] Free fit for {ordering_name} central values...", flush=True)
        m_free = minuit_new.run_minuit(prefit_cache=prefit, toy_quiet=True,
                                       fix_dm2_31_to=None, **common_kwargs)
        
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
            nominal_uncertainties["me_rho"]    = args_juno_fit.me_rho_scale
            nominal_uncertainties["ThU_ratio"] = args_juno_fit.Th_U_ratio_err
            nominal_uncertainties["Nrea"]  = args_juno_fit.r2_unc
            for name in param_names:
                if name.startswith('core'):
                    nominal_uncertainties[name] = args_juno_fit.core_flux_unc
                if name.startswith('efiss'):
                    nominal_uncertainties[name] = args_juno_fit.efission_err[int(name[-1])]*1e-2
                if name.startswith('eff'):
                    nominal_uncertainties[name] = args_juno_fit.eff_unc
                if name.startswith('snf'):
                    nominal_uncertainties[name] = args_juno_fit.snf_unc
                if args_juno_fit.NL == "pull" and name.startswith('NL'):
                    nominal_uncertainties[name] = args_juno_fit.NL_sd
            if args_juno_fit.NL == "pull": 
                nominal_uncertainties['NL4'] = args_juno_fit.ene_scale_unc
                nominal_uncertainties['abc0'] = args_juno_fit.a_err
                nominal_uncertainties['abc1'] = args_juno_fit.b_err
                nominal_uncertainties['abc2'] = args_juno_fit.c_err
        
            
            if 'backgrounds' in common_kwargs and 'JUNO' in common_kwargs['backgrounds']:
                for key, info in common_kwargs['backgrounds']["JUNO"].items():
                    nominal_uncertainties[f"bkg_{key}"] = info.get("rate_unc", 1.0)
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
                
            m_f = minuit_new.run_minuit(prefit_cache=prefit, toy_quiet=True,
                                        fix_dm2_31_to=fit_cand, **common_kwargs)

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