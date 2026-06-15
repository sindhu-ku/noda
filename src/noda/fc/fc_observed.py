import argparse, os
from datetime import datetime
import numpy as np

from noda.fc.fc_toy import build_setups
from noda.fc.feldman_cousins import base_param, build_fit_bundles, make_resolver
from noda.fit import combined_fit as minuit_new
from noda.fit.minuit import load_prefit_cache

def parse():
    # allow_abbrev=False so extra `--key value` config overrides aren't
    # accidentally matched as prefixes of the known flags below.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config",       required=True,
                   help="Config YAML or multi-phase manifest (as for run_noda)")
    p.add_argument("--base_path",    default=None,
                   help="Root of the NODA tree, substituted for ${NODA_BASE} "
                        "in all config/background paths. Overrides $NODA_BASE "
                        "and the config base_path key (as for run_noda).")
    p.add_argument("--prefit-cache", required=True)
    p.add_argument("--output",       required=True, help="output NPZ path")
    p.add_argument("--mode",         choices=("dm31", "nmo", "both"),
                   default="both",
                   help="dm31: free + fixed-grid fits; nmo: free fits ONLY "
                        "(no grid scan) — use this for the LBL NMO test where "
                        "run_fc_toy only needs the data free-fit dm2_31; "
                        "both: free + grid (default).")
    p.add_argument("--candidates",   type=float, nargs='+',
                   default=[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                   help="Absolute dm²₃₁ candidate values in 1e-3 eV² "
                        "(used by dm31/both modes only)")
    # Any other --key value pairs are forwarded as config overrides, exactly
    # like run_noda CLI flags (e.g. --pull_dm2_31_lbl 1 --PDG_opt NuFitLBL).
    return p.parse_known_args()

def main():
    cli, extra = parse()
    t0 = datetime.now()

    # Setup is independent of the fit hypothesis, so build it once. `extra`
    # carries config overrides (--pull_dm2_31_lbl 1, --PDG_opt NuFitLBL, …)
    # so the data fit uses the SAME constraints as the matching run_fc_toy run.
    setups, policy = build_setups(cli.config, base_path=cli.base_path,
                                  remaining_argv=extra)
    if extra:
        print(f"[FC-obs] config overrides: {' '.join(extra)}", flush=True)
    print(f"[FC-obs] mode={cli.mode}  "
          f"(grid fits {'OFF' if cli.mode == 'nmo' else 'ON'})", flush=True)
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

            # Shared (cross-dataset) scalar priors.
            nominal_uncertainties["me_rho"]    = args0.me_rho_scale
            nominal_uncertainties["ThU_ratio"] = args0.Th_U_ratio_err
            nominal_uncertainties["Nrea"]      = args0.r2_unc

            # Per-dataset priors: per-phase/shell parameters carry a
            # '__p…__s…' suffix. Route each mangled Minuit name to the
            # (phase, shell) it belongs to and take THAT dataset's own
            # uncertainties — eff_unc and background rate_unc differ per shell
            # in a manifest run, so using setups[0] for all would sample the
            # other shells' nuisances at the wrong width. A shared (correlated)
            # parameter resolves to the same mangled name from every setup and
            # is written identically.
            for s in setups:
                a       = s['args_juno']
                bkg     = s['backgrounds']
                resolve = make_resolver(policy, a.phase, a.shell)
                for name in param_names:
                    base = base_param(name)
                    if resolve(base) != name:
                        continue            # not this dataset's parameter
                    if base.startswith('core'):
                        nominal_uncertainties[name] = a.core_flux_unc
                    if base.startswith('efiss'):
                        nominal_uncertainties[name] = a.efission_err[int(base[-1])]*1e-2
                    if base.startswith('eff'):
                        nominal_uncertainties[name] = a.eff_unc
                    if base.startswith('snf'):
                        nominal_uncertainties[name] = a.snf_unc
                    if a.NL == "pull" and base.startswith('NL'):
                        nominal_uncertainties[name] = (a.ene_scale_unc
                                                       if base == 'NL4'
                                                       else a.NL_sd)
                    if a.NL == "pull" and base.startswith('abc'):
                        nominal_uncertainties[name] = {'abc0': a.a_err,
                                                       'abc1': a.b_err,
                                                       'abc2': a.c_err}[base]
                    if base.startswith('bkg_'):
                        info = bkg["JUNO"].get(base[4:], {})
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
        # Skipped entirely in nmo mode: the NMO test only needs the free-fit
        # (integer-key) dm2_31, which is already stored above.
        if cli.mode == "nmo":
            print(f"[FC-obs] nmo mode: skipping fixed-grid fits for "
                  f"{ordering_name}", flush=True)
            continue

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
