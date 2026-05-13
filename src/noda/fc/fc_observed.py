#!/usr/bin/env python3
"""
fc_observed.py — produce the observed TS profile for ONE dataset.

By default, uses the observed JUNO spectrum loaded from the YAML config
(your real data). Optional --mock flag replaces it with synthetic data
for sensitivity studies / testing.
"""
import argparse, os
from copy import deepcopy
from datetime import datetime
import numpy as np, pandas as pd

from noda.fc.fc_toy import (build_args_from_yaml, build_setup, make_juno_det)
from noda.fit import combined_fit as minuit_new
from noda.fc.feldman_cousins import (load_prefit_cache, make_toy_ensp,
                                      build_juno_asimov, fluctuate_poisson)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--prefit-cache", required=True)
    p.add_argument("--output",       required=True, help="output CSV path")
    p.add_argument("--candidates",   type=float, nargs='+',
                   default=[2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                   help="dm²₃₁ candidate values in 1e-3 eV² (Minuit convention)")
    p.add_argument("--true-no", type=int, choices=(0, 1), default=1,
                   help="ordering ASSUMED IN THE FIT (NMO_opt)")

    # ── Optional: only used for sensitivity studies / testing ────────────────
    p.add_argument("--mock", choices=("asimov", "toy"), default=None,
                   help="Replace the real observed spectrum with synthetic data. "
                        "Omit this flag for real data.")
    p.add_argument("--mock-truth-dm31", type=float, default=2.5,
                   help="(--mock only) truth dm²₃₁ × 1e-3 eV²")
    p.add_argument("--mock-truth-no",   type=int, choices=(0, 1), default=1,
                   help="(--mock only) truth ordering")
    p.add_argument("--mock-seed", type=int, default=0,
                   help="(--mock toy only) RNG seed")
    return p.parse_args()


def main():
    cli = parse()
    t0 = datetime.now()

    args_juno, args_dyb, args_tao = build_args_from_yaml(cli.config)
    setup = build_setup(args_juno, args_dyb, args_tao)
    print(f"[FC-obs] setup ready in {datetime.now() - t0}", flush=True)

    common_kwargs = dict(setup)

    # ── default: use whatever real data the config loaded ────────────────────
    if cli.mock is None:
        tag = "real_data"
        print(f"[FC-obs] using observed JUNO spectrum from the YAML config")
    else:
        # synthetic override for testing
        E_segments = np.array(args_juno.seg_bins_nonuniform)
        juno_det = make_juno_det(setup, E_segments)
        sign = +1 if cli.mock_truth_no else -1
        asimov = build_juno_asimov(
            juno_det, args_juno, setup['backgrounds'],
            true_dm2_31=sign * cli.mock_truth_dm31 * 1e-3,
            true_NO=bool(cli.mock_truth_no))
        if cli.mock == "asimov":
            obs_data = asimov
            tag = f"mock_asimov_dm{cli.mock_truth_dm31:.4f}_NO{cli.mock_truth_no}"
        else:
            obs_data = fluctuate_poisson(asimov, np.random.default_rng(cli.mock_seed))
            tag = f"mock_toy{cli.mock_seed}_dm{cli.mock_truth_dm31:.4f}_NO{cli.mock_truth_no}"
            args_juno_fit.toymc   = True
        common_kwargs['ensp_nom_juno'] = make_toy_ensp(setup['ensp_nom_juno'], obs_data)
        print(f"[FC-obs] using synthetic data: {tag}")

    args_juno_fit = deepcopy(args_juno)
    args_juno_fit.NMO_opt = int(cli.true_no)
    common_kwargs['args_juno'] = args_juno_fit

    prefit = load_prefit_cache(cli.prefit_cache)
    print(f"[FC-obs] loaded prefit", flush=True)

    print(f"[FC-obs] free fit ...", flush=True)
    m = minuit_new.run_minuit(prefit_cache=prefit, toy_quiet=True,
                              fix_dm2_31_to=None, **common_kwargs)
    chi2_min  = float(m.fval)
    best_dm31 = float(m.values['dm2_31'])
    print(f"[FC-obs]   χ²_min = {chi2_min:.3f},  best dm²₃₁ = {best_dm31:.4f}")

    rows = []
    for cand in cli.candidates:
        m_f = minuit_new.run_minuit(prefit_cache=prefit, toy_quiet=True,
                                    fix_dm2_31_to=cand, **common_kwargs)
        chi2_fix = float(m_f.fval)
        ts_obs   = chi2_fix - chi2_min
        rows.append({'true_dm31': cand, 'true_NO': cli.true_no,
                     'chi2_fix':  chi2_fix, 'chi2_min': chi2_min,
                     'best_dm31': best_dm31, 'TS_obs': ts_obs,
                     'dataset':   tag})
        print(f"[FC-obs]   dm31={cand:.4f}  χ²_fix={chi2_fix:.3f}  TS_obs={ts_obs:.3f}",
              flush=True)

    os.makedirs(os.path.dirname(cli.output) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(cli.output, index=False)
    print(f"[FC-obs] wrote {len(rows)} rows to {cli.output}")
    print(f"[FC-obs] total time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()