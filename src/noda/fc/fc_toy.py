#!/usr/bin/env python3
"""
fc_toy.py — single Feldman-Cousins task (one cluster job runs one of these).

Each invocation:
  1. Builds the same setup as main.py (spectra, RMs, CMs, detectors)
  2. Loads the prefit cache(s) from disk — produced by `run_noda` runs with
     `save_prefit_path` set in the YAML config (one run per NMO ordering).
  3. Loops over its batch of toys at one (true_dm31, true_no) point
  4. Writes a CSV with one row per toy

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
import pandas as pd
from datetime import datetime

# package imports
import noda.spectra_new_single as spec
import noda.matrices as mat
from noda.fit import combined_fit as minuit_new
from noda.fc.feldman_cousins import (
    load_prefit_cache, run_one_toy_dm31, run_one_toy_nmo,
)
from noda.noda import (
    CovMatrix, Spectrum, GetCovMatrixFromROOT, GetRelCM, LoadObject,
    rebin_cov_matrix, calculate_ndays,
)
from noda import nuosc


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--task-id",     type=int, required=True)
    p.add_argument("--n-toys",      type=int, default=50)
    p.add_argument("--seed-base",   type=int, default=0,
                   help="Global seed offset; toy seed = seed_base + task_id*n_toys + i")
    p.add_argument("--true-dm31",   type=float, required=True,
                   help="True dm²₃₁ in 1e-3 eV² (always positive; sign comes from --true-no)")
    p.add_argument("--true-no",     type=int, choices=(0, 1), default=1)
    p.add_argument("--mode",   choices=("dm31", "nmo", "both"), default="dm31")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--prefit-cache",
                   help="Path to .npz prefit cache produced")
    return p.parse_args()



# ─────────────────────────────────────────────────────────────────────────────
# Setup builder — replicates the relevant parts of main.py
# ─────────────────────────────────────────────────────────────────────────────
def build_args_from_yaml(config_path):
    """Reproduce main.py's three-parser config parse."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    import argparse as ap
    parsers = {'juno': ap.ArgumentParser(), 'dyb': ap.ArgumentParser(),
               'tao':  ap.ArgumentParser()}
    sections = {
        "fit_options":       ["juno", "dyb", "tao"],
        "common_det_inputs": ["juno", "dyb", "tao"],
        "juno_inputs":       ["juno"],
        "dyb_inputs":        ["dyb"],
        "tao_inputs":        ["tao"],
    }
    for sec, dets in sections.items():
        if sec not in config: continue
        for opt, val in config[sec].items():
            for d in dets:
                parsers[d].add_argument(f"--{opt}", default=val)

    return (parsers['juno'].parse_args([]),
            parsers['dyb'].parse_args([]),
            parsers['tao'].parse_args([]))


def build_setup(args_juno, args_dyb, args_tao):
    """
    Build everything main.py builds, up to (but not including) the run_minuit
    call. Returns a dict ready to splat into run_minuit().
    """
    from noda.main import (
        _build_ibd_interpolator, _build_bump_interpolator,
    )

    ndays     = calculate_ndays(args_juno)
    ndays_dyb = calculate_ndays(args_dyb)
    ndays_tao = calculate_ndays(args_tao)

    ebins     = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins)
    ebins_dyb = np.linspace(args_dyb.min_ene,  args_dyb.max_ene,  args_dyb.bins)
    ebins_tao = np.linspace(args_tao.min_ene,  args_tao.max_ene,  args_tao.bins)

    s_ibd      = _build_ibd_interpolator(ebins)
    s_bump_lin = _build_bump_interpolator(args_juno.input_data_file)

    with open(args_juno.backgrounds_yaml) as f:
        backgrounds = yaml.safe_load(f)

    ensp_nom_juno, rm = spec.CreateSpectra(
        ndays=ndays, ebins=ebins, detector="JUNO", backgrounds=backgrounds,
        s_ibd=s_ibd, s_bump_lin=s_bump_lin, args=args_juno)

    ensp_nom_dyb, rm_dyb = spec.CreateSpectra(
        ndays=ndays_dyb, ebins=ebins_dyb, detector="DYB", backgrounds=backgrounds,
        s_ibd=s_ibd, s_bump_lin=s_bump_lin, args=args_dyb)

    ensp_nom_tao, rm_tao, ene_leak_tao = spec.CreateSpectra(
        ndays=ndays_tao, ebins=ebins_tao, detector="TAO", backgrounds=backgrounds,
        s_ibd=s_ibd, s_bump_lin=s_bump_lin, args=args_tao)

    # JUNO covariance
    unc_max_juno = max(args_juno.unc_list_juno, key=len)
    unc_list_juno = unc_max_juno.split('+') if '+' in unc_max_juno else [unc_max_juno]
    cm_juno = {}
    for u in unc_list_juno:
        cache = (f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_"
                 f"{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_"
                 f"{args_juno.stat_opt}_{args_juno.bins}bins.dat")
        if os.path.isfile(cache) and not args_juno.FORCE_CALC_CM:
            cm_juno[u] = LoadObject(cache)
        else:
            cm_juno[u] = mat.GetCM(ensp=ensp_nom_juno, resp_matrix=rm, ndays=ndays,
                                   unc=u, detector="JUNO", args=args_juno,
                                   backgrounds=backgrounds)
    full_unc = unc_list_juno[0] if len(unc_list_juno) == 1 else '+'.join(unc_list_juno)
    cm_juno[full_unc] = CovMatrix(
        data=np.zeros((len(ensp_nom_juno['rdet'].bins)-1,) * 2), bins=ebins)
    for u in full_unc.split('+'):
        cm_juno[full_unc] += cm_juno[u]

    # TAO CM
    cm_tao = {'all': rebin_cov_matrix(
        GetCovMatrixFromROOT(args_tao.data_file, args_tao.FV_cov_matrix),
        args_tao.bins_nonuniform)}

    # DYB CM
    dyb_V = np.loadtxt(args_dyb.dyb_cm)
    if args_dyb.model == "PRL":
        cov_75 = dyb_V.reshape(75, 75)
        dyb_V = cov_75[50:75, 50:75] * 1e-86
    DYB_CM = CovMatrix(dyb_V, bins=ensp_nom_dyb['rdet'].bins)
    if args_dyb.model == "CPC":
        DYB_spectrum = pd.read_csv(args_dyb.dyb_spectrum, sep=r'\s+',
                                   names=["Bin_Center", "Sigma", "Rel_Unc"])
    else:
        DYB_spectrum = pd.read_csv(args_dyb.dyb_spectrum, sep=r'\s+',
                                   names=["Bin_lowE", "Bin_highE", "Sigma", "Rel_Unc"])
    DYB_sp = Spectrum(bins=ensp_nom_dyb['rdet'].bins,
                      bin_cont=DYB_spectrum["Sigma"].to_numpy())
    DYB_rel_cm = GetRelCM(DYB_CM, DYB_sp)
    cm_dyb = {'all': DYB_rel_cm}

    return dict(
        ndays=ndays, ndays_dyb=ndays_dyb, ndays_tao=ndays_tao,
        s_ibd=s_ibd, s_bump_lin=s_bump_lin,
        ensp_nom_juno=ensp_nom_juno, ensp_nom_dyb=ensp_nom_dyb,
        ensp_nom_tao=ensp_nom_tao,
        unc_juno=full_unc, unc_dyb='all', unc_tao='all',
        rm=rm, rm_dyb=rm_dyb, rm_tao=rm_tao,
        cm_juno=cm_juno, cm_dyb=cm_dyb, cm_tao=cm_tao,
        args_juno=args_juno, args_dyb=args_dyb, args_tao=args_tao,
        dyb_model=None, backgrounds=backgrounds, ene_leak_tao=ene_leak_tao,
    )


def make_juno_det(setup, E_segments):
    from noda.fit.juno_fit import JUNODetector
    from noda.fit.combined_fit import load_reactor_summary
    args_juno = setup['args_juno']
    core_names, core_powers_std, core_fractions, core_baselines = \
        load_reactor_summary(args_juno.reactor_summary_yaml)
    return JUNODetector(
        ndays=setup['ndays'],
        ensp_nom=setup['ensp_nom_juno'],
        rm=setup['rm'], cm=setup['cm_juno'],
        args=args_juno,
        backgrounds=setup['backgrounds'],
        core_baselines=core_baselines,
        core_powers_std=core_powers_std,
        core_fractions=core_fractions,
        E_segments=E_segments,
        s_ibd=setup['s_ibd'],
        s_bump_lin=setup['s_bump_lin'])


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

    args_juno, args_dyb, args_tao = build_args_from_yaml(cli.config)
    args_juno.toymc = True
    setup = build_setup(args_juno, args_dyb, args_tao)
    print(f"[FC] setup built in {datetime.now() - t0}", flush=True)

    E_segments = np.array(args_juno.seg_bins_nonuniform)
    juno_det_truth = make_juno_det(setup, E_segments)

    prefit = (load_prefit_cache(cli.prefit_cache)
              if cli.prefit_cache and os.path.exists(cli.prefit_cache) else None)
    print(f"[FC] loaded prefit from {cli.prefit_cache}", flush=True)

    common_kwargs = dict(setup)

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

            if cli.mode in ("dm31", "both"):
                args_juno.pull_dm2_31     = True
                args_juno.pull_dm2_31_lbl = False
                r = run_one_toy_dm31(
                    seed=seed,
                    true_dm2_31_minuit=cli.true_dm31,
                    true_NO=bool(cli.true_no),
                    run_minuit_fn=minuit_new.run_minuit,
                    juno_det=juno_det_truth,
                    args_juno_truth=args_juno,
                    backgrounds=setup['backgrounds'],
                    ensp_nom_juno=setup['ensp_nom_juno'],
                    prefit_cache=prefit,
                    common_kwargs=common_kwargs)
                r['mode'] = 'dm31'
                writer.writerow(r); fh.flush(); n_written += 1

            if cli.mode in ("nmo", "both"):
                args_juno.pull_dm2_31     = False
                args_juno.pull_dm2_31_lbl = True
                r = run_one_toy_nmo(
                    seed=seed,
                    true_NO=bool(cli.true_no),
                    run_minuit_fn=minuit_new.run_minuit,
                    juno_det=juno_det_truth,
                    args_juno_truth=args_juno,
                    backgrounds=setup['backgrounds'],
                    ensp_nom_juno=setup['ensp_nom_juno'],
                    prefit_cache=prefit,
                    common_kwargs=common_kwargs)
                r['mode'] = 'nmo'
                writer.writerow(r); fh.flush(); n_written += 1

            print(f"[FC] task={cli.task_id} toy {i+1}/{cli.n_toys} "
                  f"seed={seed} done in {datetime.now() - toy_t} "
                  f"({n_written} rows in {out_csv})", flush=True)

    print(f"[FC] wrote {n_written} rows to {out_csv}")
    print(f"[FC] total task time: {datetime.now() - t0}")


if __name__ == "__main__":
    main()