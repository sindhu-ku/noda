#!/usr/bin/env python3
"""
fc_toy.py — single Feldman-Cousins task (one cluster job runs one of these).

Each invocation:
  1. Builds the same setup as main.py (manifest → one setup per phase/shell,
     spectra, RMs, CMs, ParamPolicy, JUNO bundles)
  2. Loads the prefit cache(s) from disk — produced by `run_noda` runs with
     `save_prefit_path` set in the YAML config (one run per NMO ordering).
  3. Loops over its batch of toys at one (true_dm31, true_no) point
  4. Writes a CSV with one row per toy

The --config may be a plain single-phase YAML or a multi-phase manifest
(`common:` + `phases:`), exactly as for run_noda. Each phase/shell gets its
own toy observation; the fits bundle all phases via run_minuit_bundled.

PREFIT CACHES
-------------
The prefit fixes oscillation parameters to nominal NO or IO values, so it
depends on the assumed NMO. For NMO mode (or `both`) you must supply BOTH
caches; for dm31 mode only the one matching --true-no is needed.

Toy seed = seed_base + task_id * n_toys + i  → unique, reproducible.

Usage examples:
  # dm31 FC at true NO
  run_fc_toy --config config/groupA.yaml \\
             --task-id 0 --n-toys 50 --seed-base 0 \\
             --true-dm31 2.50 --true-no 1 \\
             --mode dm31 \\
             --output-dir fc_runs/dm31_2p50_NO/ \\
             --prefit-cache fc_runs/prefit_NO.npz

  # NMO test (true NO toys); needs both prefits
  run_fc_toy --config config/groupA.yaml \\
             --task-id 0 --n-toys 50 \\
             --true-dm31 -2.50 --true-no 1 \\
             --mode nmo \\
             --output-dir fc_runs/nmo_trueNO/ \\
             --prefit-cache fc_runs/prefit_NO.npz \\
             --prefit-cache fc_runs/prefit_IO.npz
"""

import argparse
import os
import sys
import yaml
import numpy as np
from datetime import datetime

# package imports
from noda.main import (_build_juno_bundle,
                       build_setups as _main_build_setups, build_policy)
from noda.fit.minuit import load_prefit_cache
from noda.fc.feldman_cousins import (run_one_toy_dm31, run_one_toy_nmo,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True,
                   help="Config YAML or multi-phase manifest (as for run_noda)")
    p.add_argument("--base_path",   default=None,
                   help="Root of the NODA tree, substituted for ${NODA_BASE} "
                        "in all config/background paths. Overrides $NODA_BASE "
                        "and the config base_path key (as for run_noda).")
    p.add_argument("--task-id",     type=int, required=True)
    p.add_argument("--n-toys",      type=int, default=50)
    p.add_argument("--seed-base",   type=int, default=0,
                   help="Global seed offset; toy seed = seed_base + task_id*n_toys + i")
    p.add_argument("--true-dm31",   type=float, default=2.5e-3,
                   help="True dm²₃₁ in 1e-3 eV² (always positive; sign comes from --true-no)")
    p.add_argument("--true-no",     type=int, choices=(0, 1), default=1)
    p.add_argument("--mode",   choices=("dm31", "nmo", "both"), default="dm31")
    p.add_argument("--pull_dm2_31",     type=int, choices=(0, 1), default=None,
                   help="Override the DYB dm2_31 constraint on/off (default: "
                        "per --mode). Use with --pull_dm2_31_lbl for combined.")
    p.add_argument("--pull_dm2_31_lbl", type=int, choices=(0, 1), default=None,
                   help="Override the LBL dm2_31 constraint on/off (default: "
                        "per --mode).")
    p.add_argument("--PDG_opt", default=None,
                   help="Oscillation parameter set for the toys (e.g. NuFitLBL, "
                        "PDG2024, PDG2025, DYB). Overrides the config value AND "
                        "the per-mode default for every dataset. MUST match the "
                        "--PDG_opt used for run_fc_observed so the toy truth and "
                        "the cached initial fits share the same osc parameters.")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--initial-fits", help="path to initial fits of the grid points")
    p.add_argument("--prefit-cache", help="Path to .npz prefit cache produced")
    p.add_argument("--nuis-center", choices=("profiled", "prior"),
                   default="profiled",
                   help="Nuisance CENTRES for toy generation. 'profiled' "
                        "(default): conditional best-fit values (profiled FC, the "
                        "current behaviour). 'prior': prefit nominal centres with "
                        "constrained pulls at 0 = a-priori Highland-Cousins (the "
                        "Group A method). Both modes use prefit WIDTHS. 'prior' "
                        "needs an initial-fits NPZ from the updated run_fc_observed.")
    return p.parse_args()

def load_initial_fits(filepath):
    """Extracts the exact data needed for toy sampling from the npz archive."""
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find initial fits file: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    print(data)
    # Unpack the 0-d object arrays back into standard Python objects
    return {
        'grid_best_fits': data['grid_best_fits'][0],
        'truncate_below_zero': data['truncate_below_zero'].tolist(),
        'truncate_below_mone': data['truncate_below_mone'].tolist(),
        'nominal_uncertainties': data['nominal_uncertainties'][0],
        'prior_central_values': (data['prior_central_values'][0]
                                 if 'prior_central_values' in data.files else None),
        'param_names': data['param_names'].tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Setup builder — replicates the relevant parts of main.py
# ─────────────────────────────────────────────────────────────────────────────
def build_setups(config_path, dm2_31=None, base_path=None, remaining_argv=None):
    """
    Manifest-aware setup: one setup dict per (phase, shell) row plus the
    ParamPolicy, built through the SAME helpers run_noda uses
    (noda.main.build_setups / build_policy) so the two never drift. This
    includes NODA_BASE resolution (CLI base_path > $NODA_BASE > config
    base_path > repo root), so FC honours ${NODA_BASE} configs exactly like
    run_noda without the caller having to export anything.

    `remaining_argv` are extra `--key value` CLI overrides forwarded to the
    per-phase config parser, so any config option (e.g. --pull_dm2_31_lbl 1
    --PDG_opt NuFitLBL) can be set from the command line as for run_noda.
    Returns (setups, policy).
    """
    with open(config_path) as f:
        manifest_doc = yaml.safe_load(f) or {}

    setups = _main_build_setups(config_path, dm2_31=dm2_31, base_path=base_path,
                                remaining_argv=remaining_argv)
    policy = build_policy(setups, manifest_doc)
    return setups, policy


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
import csv

# ── CSV schemas ──────────────────────────────────────────────────────────────
DM31_FIELDS = ['mode', 'seed', 'true_dm31', 'true_NO',
               'chi2_free', 'chi2_fix', 'TS_dm31',
               'best_dm31', 'best_dm21', 'best_sin12', 'best_sin13',
               'free_valid', 'fix_valid', 'time']
NMO_FIELDS  = ['mode', 'seed', 'true_NO',
               'chi2_NO', 'chi2_IO', 'T_nmo',
               'NO_valid', 'IO_valid',
               'dm31_NO', 'dm31_IO', 'time']

def fieldnames_for_mode(mode):
    if mode == 'dm31': return DM31_FIELDS
    if mode == 'nmo':  return NMO_FIELDS
    seen, out = set(), []
    for f in DM31_FIELDS + NMO_FIELDS:
        if f not in seen:
            out.append(f); seen.add(f)
    return out


def main():
    cli = parse()
    t0 = datetime.now()
    initial_fits = load_initial_fits(cli.initial_fits)

    if cli.nuis_center == "prior" and initial_fits.get('prior_central_values') is None:
        raise SystemExit(
            "[FC] --nuis-center prior requires 'prior_central_values' in the "
            "initial-fits NPZ; regenerate it with run_fc_observed (updated).")
    print(f"[FC] nuisance centres: {cli.nuis_center} "
          f"({'a-priori Highland-Cousins / Group A' if cli.nuis_center == 'prior' else 'profiled FC'})",
          flush=True)

    if cli.mode == "dm31":
        sign   = +1.0 if bool(cli.true_no) else -1.0
        dm2_31 = sign * cli.true_dm31 * 1e-3
    else:
        dm2_31 = None
    setups, policy = build_setups(cli.config, dm2_31=dm2_31,
                                  base_path=cli.base_path)
    print(f"[FC] setup built in {datetime.now() - t0} "
          f"({len(setups)} phase/shell dataset(s))", flush=True)

    # truth args: applied to EVERY phase/shell dataset
    for s in setups:
        s['args_juno'].toymc   = True
        s['args_juno'].NMO_opt = bool(cli.true_no)
        if cli.mode == "dm31":
            s['args_juno'].pull_dm2_31     = True
            s['args_juno'].pull_dm2_31_lbl = False
        if cli.mode == "nmo":
            s['args_juno'].pull_dm2_31     = False
            s['args_juno'].pull_dm2_31_lbl = True
            s['args_juno'].PDG_opt = "NuFitLBL"
        # Explicit CLI overrides win over the per-mode defaults above, so you can
        # run e.g. combined DYB+LBL: --mode nmo --pull_dm2_31 1 --pull_dm2_31_lbl 1
        if cli.pull_dm2_31 is not None:
            s['args_juno'].pull_dm2_31     = bool(cli.pull_dm2_31)
        if cli.pull_dm2_31_lbl is not None:
            s['args_juno'].pull_dm2_31_lbl = bool(cli.pull_dm2_31_lbl)
        # --PDG_opt overrides both the config value and the nmo-mode NuFitLBL
        # default; leave it unset to keep those.
        if cli.PDG_opt is not None:
            s['args_juno'].PDG_opt = cli.PDG_opt

    # LBL projection uses the DATA best-fit SOLAR params (theta12, dm2_21) from
    # the observed fit, fixed across all toys — not nuosc nominal, not each
    # toy's own solar. Same grid_best_fits key the dm2_31 central uses.
    _free = initial_fits['grid_best_fits'].get(int(cli.true_no))
    if _free is not None:
        s12_data  = float(_free['values']['sin2_th12'])
        dm21_data = float(_free['values']['dm2_21']) * 1e-5   # Minuit -> physical eV²
        for s in setups:
            s['args_juno'].sin2_th12_lbl = s12_data
            s['args_juno'].dm2_21_lbl    = dm21_data
        print(f"[FC] LBL projection solar (data best fit): "
              f"sin2_th12={s12_data:.5f}, dm2_21={dm21_data:.3e}", flush=True)

    # nominal bundles for the truth prediction (one per phase/shell)
    truth_bundles = [_build_juno_bundle(s) for s in setups]

    prefit = (load_prefit_cache(cli.prefit_cache)
              if cli.prefit_cache and os.path.exists(cli.prefit_cache) else None)
    print(f"[FC] loaded prefit from {cli.prefit_cache}", flush=True)

    # ── prepare the output CSV (append-mode, header only if new) ──────────────
    os.makedirs(cli.output_dir, exist_ok=True)
    out_csv = os.path.join(
        cli.output_dir,
        f"fc_task_{cli.mode}_{cli.task_id:05d}_dm31_{cli.true_dm31:.4f}_NO{cli.true_no}.csv")
    new_file = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    fieldnames = fieldnames_for_mode(cli.mode)

    seed0 = cli.seed_base + cli.task_id * cli.n_toys

    with open(out_csv, 'a', newline='', buffering=1) as fh:    # line-buffered
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction='ignore')
        if new_file:
            writer.writeheader()
            fh.flush(); os.fsync(fh.fileno())

        n_written = 0
        for i in range(cli.n_toys):
            seed  = seed0 + i
            toy_t = datetime.now()

            if cli.mode == "dm31":
                r = run_one_toy_dm31(
                    seed=seed,
                    true_dm2_31_minuit=cli.true_dm31,
                    true_NO=bool(cli.true_no),
                    setups=setups,
                    policy=policy,
                    truth_bundles=truth_bundles,
                    initial_fits=initial_fits,
                    prefit_cache=prefit,
                    nuis_center=cli.nuis_center)
                r['mode'] = 'dm31'
                writer.writerow(r); fh.flush(); n_written += 1

            if cli.mode == "nmo":
                r = run_one_toy_nmo(
                    seed=seed,
                    true_NO=bool(cli.true_no),
                    setups=setups,
                    policy=policy,
                    truth_bundles=truth_bundles,
                    initial_fits=initial_fits,
                    prefit_cache=prefit,
                    nuis_center=cli.nuis_center)
                r['mode'] = 'nmo'
                writer.writerow(r); fh.flush(); n_written += 1

            print(f"[FC] task={cli.task_id} toy {i+1}/{cli.n_toys} "
                  f"seed={seed} done in {datetime.now() - toy_t} "
                  f"({n_written} rows in {out_csv})", flush=True)

    print(f"[FC] wrote {n_written} rows to {out_csv}")
    print(f"[FC] total task time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
