#!/usr/bin/env python3
"""
main.py — multi-phase fitting driver (one JUNO bundle per phase/shell,
plus optionally one TAO and one DYB bundle from the FIRST phase's config).

`manifest.load(args.config)` either returns [('p0', 'ALL', cfg_path, {})]
for a single-phase YAML (top-level keys like `fit_options`, `juno_inputs`,
…) or [(phase, shell, cfg, overrides), …] for a manifest YAML (top-level
keys are phase names mapping to config paths or override blocks).

Per-phase setup lives in setup_one_phase(); the fit always goes through
minuit_new.run_minuit_bundled().
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime

from .model import assemble as spec
from .model import matrices as mat
from .noda import *
from .fit import combined_fit as minuit_new
from .fit.manifest import load as load_manifest
from .fit.param_policy import policy_from_backgrounds, _ident
from .fit.dispatcher import DetectorBundle
from .fit.juno_fit import JUNODetector
from .fit.tao_fit  import TAODetector
from .fit.dyb_fit  import DYBDetector
from .fit.combined_fit import load_reactor_summary
from .fit.minuit import load_prefit_cache
from .physics.constants import *


# ─────────────────────────────────────────────────────────────────────────────
# Portable base path — lets the same configs run on any cluster
# ─────────────────────────────────────────────────────────────────────────────
# Every file path in the config / background YAMLs is written relative to the
# token ``${NODA_BASE}``. At load time it is replaced by the resolved root of
# the NODA tree, chosen in this priority:
#   1. CLI flag        --base_path
#   2. env var         NODA_BASE
#   3. config key      base_path  (top-level or in fit_options/common_det_inputs)
#   4. auto-detect     the repo root that contains this source file
# To move to a new cluster you only need `export NODA_BASE=/path/to/noda/noda`
# (or pass --base_path); no YAML edits required.
_DEFAULT_BASE = str(Path(__file__).resolve().parents[2])   # …/noda/noda
NODA_BASE = os.path.abspath(os.path.expanduser(
    os.environ.get('NODA_BASE') or _DEFAULT_BASE))


def _expand_base(obj, base=None):
    """Recursively substitute ${NODA_BASE} in every string of a config tree."""
    base = NODA_BASE if base is None else base
    if isinstance(obj, str):
        return obj.replace('${NODA_BASE}', base)
    if isinstance(obj, dict):
        return {k: _expand_base(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_base(v, base) for v in obj]
    return obj


def _resolve_base_path(cli_base, manifest_doc):
    """CLI > env > config base_path > auto-detected repo root."""
    cand = (cli_base
            or os.environ.get('NODA_BASE')
            or (manifest_doc.get('base_path') if manifest_doc else None)
            or ((manifest_doc.get('fit_options') or {}).get('base_path')
                if manifest_doc else None)
            or ((manifest_doc.get('common_det_inputs') or {}).get('base_path')
                if manifest_doc else None)
            or _DEFAULT_BASE)
    return os.path.abspath(os.path.expanduser(cand))


# ─────────────────────────────────────────────────────────────────────────────
# IBD cross-section interpolator (unchanged from previous main.py)
# ─────────────────────────────────────────────────────────────────────────────
def _build_ibd_interpolator(ebins=None, emax=12.0):
    # IBD cross-section tabulated on 0..emax MeV (1 keV steps); zero outside.
    # emax follows the largest detector max_ene in the yaml, never below 12 MeV.
    hbar_MeV = 1.0e-6 * hbar / qe
    MeV_to_cm = (hbar * c * (1.0 / (1.0e6 * qe)))**2 * 1.0e4
    me2, fsq, gsq = me**2, f**2, g**2
    y2 = (delta_np**2 - me2) * 0.5
    M = (mn + mp) * 0.5
    sigma0 = (2.0 * np.pi**2 /
              (phase_space * (fsq + 3.0*gsq) * (me**5) * n_lifetime / hbar_MeV))
    n_points = int((emax - 0.0) / 0.001) + 1
    enu_array = np.linspace(0.0, emax, n_points)
    cos_theta_arr = np.linspace(-1.0, 1.0, 100)
    dc = 2.0 / (len(cos_theta_arr) - 1)
    xsec = np.zeros_like(enu_array)
    for i, Enu in enumerate(enu_array):
        Ee0 = Enu - delta_np
        if Ee0 <= me: continue
        pe0 = np.sqrt(Ee0**2 - me2)
        ve0 = pe0 / Ee0
        sum_sigma_0 = 0.0
        sum_sigma_1 = 0.0
        for j, ctheta in enumerate(cos_theta_arr):
            Ee1 = Ee0 * (1.0 - Enu / M * (1.0 - ve0 * ctheta)) - y2 / M
            if Ee1 > me:
                pe1 = np.sqrt(Ee1**2 - me2)
                ve1 = pe1 / Ee1
                t0 = (sigma0 * 0.5 * ((fsq + 3.0*gsq) + (fsq - gsq) * ve1 * ctheta)
                      * Ee1 * pe1)
                g1 = 2.0 * g * (f + f2) * ((2.0 * Ee0 + delta_np) * (1.0 - ve0*ctheta) - me2/Ee0)
                g2 = (fsq + gsq) * (delta_np * (1.0 + ve0*ctheta) + me2/Ee0)
                A_val = (Ee0 + delta_np) * (1.0 - ctheta/ve0) - delta_np
                g3 = (fsq + 3.0 * gsq) * A_val
                g4 = (fsq - gsq) * A_val * ve0 * ctheta
                t1 = (-0.5 * sigma0 * Ee0 * pe0 * (g1 + g2 + g3 + g4) / M)
            else:
                t0 = 0.0; t1 = 0.0
            w = 0.5 if (j == 0 or j == len(cos_theta_arr) - 1) else 1.0
            sum_sigma_0 += w * t0
            sum_sigma_1 += w * t1
        xsec[i] = (sum_sigma_0 + sum_sigma_1) * dc * MeV_to_cm
    return sp.interpolate.interp1d(enu_array, xsec, kind='cubic',
                                    bounds_error=False, fill_value=(0.0, 0.0))


def _build_bump_interpolator(input_data_file):
    bump = GetSpectrumFromROOT(input_data_file, "DYBFluxBump_ratio")
    return sp.interpolate.interp1d(
        bump.GetBinCenters(), bump.bin_cont, kind="slinear",
        bounds_error=False,
        fill_value=(bump.bin_cont[0], bump.bin_cont[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# Config parsing — pulled out so setup_one_phase can call it per config file
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_SECTIONS = ('fit_options', 'common_det_inputs', 'juno_inputs',
                   'dyb_inputs', 'tao_inputs')
# Flat (un-sectioned) manifest overrides are per-phase JUNO quantities
# (livetime, eff, backgrounds, reactor power, detector response, …), so
# they are looked up in these sections, in this order:
OVERRIDE_SEARCH_ORDER = ('juno_inputs', 'fit_options', 'common_det_inputs')


def _apply_manifest_overrides(config, overrides):
    """
    Merge per-phase manifest overrides into the common config dict.

    Two forms are accepted per key:
      - a section name ('juno_inputs', 'tao_inputs', …) mapping to a dict
        → merged into that section explicitly;
      - a flat key (e.g. 'stat_opt', 'eff_unc', 'core_powers')
        → replaces the key in the first section of OVERRIDE_SEARCH_ORDER
          that already defines it. Unknown keys raise, so typos don't
          silently fall back to the common value.
    """
    for key, val in overrides.items():
        if key in CONFIG_SECTIONS:
            if not isinstance(val, dict):
                raise ValueError(
                    f"manifest override '{key}' names a config section and "
                    f"must map to a dict of keys")
            config.setdefault(key, {}).update(val)
            continue
        for section in OVERRIDE_SEARCH_ORDER:
            if key in config.get(section, {}):
                config[section][key] = val
                break
        else:
            raise KeyError(
                f"manifest override '{key}' not found in common config "
                f"sections {OVERRIDE_SEARCH_ORDER}; nest it under an "
                f"explicit section (e.g. 'juno_inputs:') to add a new key")


def _parse_phase_config(cfg_path, remaining_argv=None, overrides=None):
    """
    Run the existing argparse-from-YAML logic against ONE config file.
    Returns (args_juno, args_dyb, args_tao). `remaining_argv` is the
    command-line argv minus --config; pass None for nested loads where
    you don't want CLI overrides applied. `overrides` is the per-phase
    dict from a common-config manifest, applied on top of the YAML.
    """
    if remaining_argv is None:
        remaining_argv = []

    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        _apply_manifest_overrides(config, overrides)

    # Substitute ${NODA_BASE} (after overrides, so injected paths expand too).
    config = _expand_base(config)

    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--config', default=cfg_path)

    parser_juno = argparse.ArgumentParser(parents=[conf_parser])
    parser_dyb  = argparse.ArgumentParser(parents=[conf_parser])
    parser_tao  = argparse.ArgumentParser(parents=[conf_parser])

    sections = {
        "fit_options":       [parser_juno, parser_dyb, parser_tao],
        "common_det_inputs": [parser_juno, parser_dyb, parser_tao],
        "juno_inputs":       [parser_juno],
        "dyb_inputs":        [parser_dyb],
        "tao_inputs":        [parser_tao],
    }
    for section, parsers in sections.items():
        if section not in config: continue
        for opt, val in config[section].items():
            for p in parsers:
                if isinstance(val, bool):
                    p.add_argument(f"--{opt}", default=val,
                                   type=lambda x: str(x).lower()
                                                  in ("1","true","yes","y","t"))
                else:
                    p.add_argument(f"--{opt}", default=val,
                                   type=type(val) if val is not None else str)

    # CLI overrides (remaining_argv) must reach all three detectors, not just
    # JUNO — otherwise shared fit_options flags like --is_data silently apply
    # to JUNO only, leaving DYB/TAO on their config values (e.g. JUNO Asimov but
    # DYB/TAO on real data → biased combined fit). parse_known_args lets each
    # detector pick up the shared flags it defines while ignoring the others'
    # detector-specific flags (juno_inputs/dyb_inputs/tao_inputs) instead of
    # erroring on them.
    args_juno = parser_juno.parse_args(remaining_argv)
    args_dyb, _  = parser_dyb.parse_known_args(remaining_argv)
    args_tao, _  = parser_tao.parse_known_args(remaining_argv)
    return args_juno, args_dyb, args_tao


# ─────────────────────────────────────────────────────────────────────────────
# Per-phase setup — wraps everything the old main.py did for one config
# ─────────────────────────────────────────────────────────────────────────────
def _dataset_suffix(phase, shell, rows):
    """Per-dataset ROOT-name suffix so multiple phases/shells don't overwrite
    each other's stored spectra. Mirrors the nuisance naming (`__p…__s…`) and
    only includes the dimension(s) that actually vary across `rows`. A single
    dataset (legacy single-config run) yields '' — names stay unchanged."""
    if len(rows) <= 1:
        return ''
    suf = ''
    if len({r[0] for r in rows}) > 1: suf += f'__p{_ident(phase)}'
    if len({r[1] for r in rows}) > 1: suf += f'__s{_ident(shell)}'
    if not suf:                       # bundles differ but labels identical
        suf = f'__p{_ident(phase)}__s{_ident(shell)}'
    return suf


def setup_one_phase(cfg_path, *, phase_label='p0', shell_label='ALL',
                    remaining_argv=None, overrides=None, dm2_31=None,
                    root_suffix=''):
    """
    Build the full stack of objects needed for one phase (or one shell of
    one phase): args namespaces, spectra, response matrices, covariance
    matrices, backgrounds, ndays. Returns a dict — caller assembles
    DetectorBundles from it.

    `dm2_31` (physical eV², signed) overrides the nominal value when
    generating the JUNO spectra — used by the FC machinery to build
    truth spectra at a grid point.
    """
    print(f"\n══════ setup phase={phase_label} shell={shell_label}  ({cfg_path}) ══════")
    if overrides:
        print(f"  manifest overrides: {', '.join(overrides)}")
    args_juno, args_dyb, args_tao = _parse_phase_config(cfg_path, remaining_argv,
                                                        overrides=overrides)

    # Stamp dataset identity so the output writers can suffix per phase/shell.
    for _a in (args_juno, args_dyb, args_tao):
        _a.phase = phase_label
        _a.shell = shell_label
    # Only JUNO has multiple datasets (shells); TAO/DYB stay unsuffixed.
    args_juno.root_suffix = root_suffix
    args_dyb.root_suffix  = ''
    args_tao.root_suffix  = ''

    if not os.path.exists(args_juno.main_data_folder):
        os.makedirs(args_juno.main_data_folder, exist_ok=True)

    ndays     = calculate_ndays(args_juno)
    ndays_dyb = calculate_ndays(args_dyb)
    ndays_tao = calculate_ndays(args_tao)
    print(f"  livetimes: JUNO={ndays:.1f} d  DYB={ndays_dyb:.1f} d  TAO={ndays_tao:.1f} d")

    ebins     = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins)
    ebins_dyb = np.linspace(args_dyb.min_ene,  args_dyb.max_ene,  args_dyb.bins)
    ebins_tao = np.linspace(args_tao.min_ene,  args_tao.max_ene,  args_tao.bins)

    os.makedirs(args_juno.data_matrix_folder, exist_ok=True)
    os.makedirs(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}",
                exist_ok=True)

    # ── interpolators ───────────────────────────────────────────────────
    s_ibd      = _build_ibd_interpolator(
        ebins, emax=max(12.0, float(args_juno.max_ene), float(args_dyb.max_ene),
                        float(args_tao.max_ene)))
    s_bump_lin = _build_bump_interpolator(args_juno.input_data_file)

    # ── per-phase energy-response covariance (optional) ──────────────────
    # If the phase config sets `er_cov_file`, load it; otherwise the
    # JUNODetector falls back to the module-global covariance_er.
    covariance_er = None
    er_cov_file = getattr(args_juno, 'det_resp_file', None)
    if er_cov_file:
        if er_cov_file.endswith('.npz'):
            covariance_er = np.load(er_cov_file)
        else:
            covariance_er = np.loadtxt(er_cov_file)

        if not args_juno.silent_print:
            print(f"  loaded per-phase ER covariance: {er_cov_file}")
            print(covariance_er)

    # ── backgrounds ─────────────────────────────────────────────────────
    with open(args_juno.backgrounds_yaml, 'r') as f:
        backgrounds = _expand_base(yaml.safe_load(f))
    backgrounds = backgrounds or {}
    if not backgrounds.get('JUNO'):        # empty / missing list -> no backgrounds
        backgrounds['JUNO'] = {}
    from .model import backgrounds as bkgdef
    bkgdef.register(backgrounds)           # display name -> key for post-fit output

    # ── spectra ─────────────────────────────────────────────────────────
    ensp_nom_juno, resp_matrix = spec.CreateSpectra(
        ndays=ndays, ebins=ebins, detector="JUNO",
        backgrounds=backgrounds, s_ibd=s_ibd, s_bump_lin=s_bump_lin,
        args=args_juno, dm2_31=dm2_31)

    ensp_nom_dyb, resp_matrix_dyb = None, None
    ensp_nom_tao, resp_matrix_tao, ene_leak_tao = None, None, None

    if args_juno.fit_type == 'PMOP_combined':
        if not args_juno.use_dagflow:
            ensp_nom_dyb, resp_matrix_dyb = spec.CreateSpectra(
                ndays=ndays_dyb, ebins=ebins_dyb, detector="DYB",
                backgrounds=backgrounds, s_ibd=s_ibd, s_bump_lin=s_bump_lin,
                args=args_dyb)
        # honour include_tao at SETUP time too: with a prefit cache the TAO
        # bundle is never used, so building its spectra is wasted work (and
        # forces its inputs to exist). Default True = previous behaviour.
        if getattr(args_juno, 'include_tao', True):
            ensp_nom_tao, resp_matrix_tao, ene_leak_tao = spec.CreateSpectra(
                ndays=ndays_tao, ebins=ebins_tao, detector="TAO",
                backgrounds=backgrounds, s_ibd=s_ibd, s_bump_lin=s_bump_lin,
                args=args_tao)

    # ── JUNO covariance matrices (load-or-compute + cache to disk) ──────
    unc_max_juno = max(args_juno.unc_list_juno, key=len)
    unc_list_juno = unc_max_juno.split('+') if '+' in unc_max_juno else [unc_max_juno]
    cm_juno = {}

    def _cm_cache(u):
        return (f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_"
                f"{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_"
                f"NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_"
                f"{args_juno.bins}bins.dat")

    for u in unc_list_juno:
        cache = _cm_cache(u)
        if os.path.isfile(cache) and not args_juno.FORCE_CALC_CM:
            print(f"  loading CM: {os.path.basename(cache)}")
            cm_juno[u] = LoadObject(cache)
        else:
            print(f"  building CM: {os.path.basename(cache)}")
            cm_juno[u] = mat.GetCM(
                ensp=ensp_nom_juno, resp_matrix=resp_matrix,
                ndays=ndays, unc=u, detector="JUNO",
                args=args_juno, backgrounds=backgrounds)
        cm_juno[u].Dump(f"{args_juno.data_matrix_folder}/csv_"
                        f"{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
        if args_juno.PLOT_CM:
            cm_juno[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_JUNO_{u}.png")
        del cm_juno[u]                # mimic old main.py memory pattern

    # second pass: rebuild combined-key CMs by summing individual components
    unc_list_new_juno = []
    for unc in args_juno.unc_list_juno:
        unc = unc.replace('stat+', "")            # stat handled inside chi2
        unc_list_new_juno.append(unc.replace(args_juno.unc_corr_ind + '+', ''))

    for full_unc in unc_list_new_juno:
        single = full_unc.split('+')
        cm_juno[full_unc] = CovMatrix(
            data=np.zeros((len(ensp_nom_juno['rdet'].bins) - 1,) * 2),
            bins=ebins)
        for u in single:
            cm_juno[u] = LoadObject(_cm_cache(u))
            cm_juno[full_unc] += cm_juno[u]

    unc_key_juno = unc_list_new_juno[0]

    # ── TAO covariance ──────────────────────────────────────────────────
    cm_tao = None
    if args_juno.fit_type == 'PMOP_combined':
        cm_tao = {'all': rebin_cov_matrix(
            GetCovMatrixFromROOT(args_tao.data_file, args_tao.FV_cov_matrix),
            args_tao.bins_nonuniform)}
        # When NL is baked into the leakage matrix (nl_in_leak) there are no NL
        # pull nuisances, so the NL uncertainty is carried as a covariance term
        # instead. Nonlinearity800_RelativeCovariance is already on the analysis
        # binning (bins_nonuniform), so it is added directly (no rebin).
        nl_cov_name = getattr(args_tao, 'NL_cov_matrix', None)
        if getattr(args_tao, 'nl_in_leak', False) and nl_cov_name:
            nl_cov = GetCovMatrixFromROOT(args_tao.data_file, nl_cov_name)
            if nl_cov.data.shape != cm_tao['all'].data.shape:
                raise ValueError(
                    f"NL covariance shape {nl_cov.data.shape} != TAO covariance "
                    f"shape {cm_tao['all'].data.shape}; expected both on "
                    f"bins_nonuniform ({len(args_tao.bins_nonuniform)} groups).")
            cm_tao['all'].data = cm_tao['all'].data + nl_cov.data
            if not args_juno.silent_print:
                print(f"[TAO] added NL covariance {nl_cov_name} "
                      f"(diag mean={np.mean(np.diag(nl_cov.data)):.3e})")

    # ── DYB covariance (only in direct PMOP path) ───────────────────────
    cm_dyb = None
    if args_juno.fit_type == 'PMOP_combined' and not args_juno.use_dagflow:
        dyb_V = np.loadtxt(args_dyb.dyb_cm)
        if args_dyb.model == "PRL":
            cov_75 = dyb_V.reshape(75, 75)
            # same fission fractions as the DYB flux/data scaling in spectra.py
            ef_dyb = compute_extrafactors(
                args_dyb, alpha=[args_dyb.U235_scale, args_dyb.U238_scale,
                                 args_dyb.Pu239_scale, args_dyb.Pu241_scale])
            dyb_V = cov_75[50:75, 50:75] * 1e-86 * ef_dyb * ef_dyb * ndays_dyb * ndays_dyb
            print("DYB CM scaling 1e-86: ", ef_dyb * ef_dyb * ndays_dyb * ndays_dyb)
        DYB_CM = CovMatrix(dyb_V, bins=ensp_nom_dyb['rdet'].bins)
        DYB_CM.ScaleByBinWidth()
        if args_dyb.is_data:
            cm_dyb = {'all':  DYB_CM} #GetRelCM(DYB_CM, ensp_nom_dyb['data'])}
        else:
            cm_dyb = {'all': DYB_CM} # GetRelCM(DYB_CM, ensp_nom_dyb['rtot'])}

    return {
        'phase':            phase_label,
        'shell':            shell_label,
        'args_juno':        args_juno,
        'args_dyb':         args_dyb,
        'args_tao':         args_tao,
        'backgrounds':      backgrounds,
        'ndays':            ndays,
        'ndays_dyb':        ndays_dyb,
        'ndays_tao':        ndays_tao,
        'ensp_nom_juno':    ensp_nom_juno,
        'ensp_nom_dyb':     ensp_nom_dyb,
        'ensp_nom_tao':     ensp_nom_tao,
        'rm':               resp_matrix,
        'rm_dyb':           resp_matrix_dyb,
        'rm_tao':           resp_matrix_tao,
        'cm_juno':          cm_juno,
        'cm_dyb':           cm_dyb,
        'cm_tao':           cm_tao,
        'unc_key_juno':     unc_key_juno,
        's_ibd':            s_ibd,
        's_bump_lin':       s_bump_lin,
        'ene_leak_tao':     ene_leak_tao,
        'covariance_er':    covariance_er,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bundle builders
# ─────────────────────────────────────────────────────────────────────────────
def _build_juno_bundle(setup):
    s = setup
    args_juno = s['args_juno']
    if args_juno.CI_powers:
        core_baselines = args_juno.core_baselines
        core_powers_std    = args_juno.core_powers
        core_fractions = None
        core_names = None
    else:
        core_names, core_powers_std, core_fractions, core_baselines = \
            load_reactor_summary(args_juno.reactor_summary_yaml)
    det = JUNODetector(
        ndays=s['ndays'],
        ensp_nom=s['ensp_nom_juno'],
        rm=s['rm'], cm=s['cm_juno'],
        args=args_juno,
        backgrounds=s['backgrounds'],
        core_baselines=core_baselines,
        core_powers_std=core_powers_std,
        core_fractions=core_fractions,
        core_names=core_names,
        E_segments=np.array(args_juno.seg_bins_nonuniform),
        s_ibd=s['s_ibd'], s_bump_lin=s['s_bump_lin'],
        covariance_er=s.get('covariance_er'))
    return DetectorBundle(
        phase=s['phase'], shell=s['shell'], kind='juno',
        args=args_juno, detector=det,
        backgrounds=s['backgrounds'], cm=s['cm_juno'],
        unc_key=s['unc_key_juno'], ensp_nom=s['ensp_nom_juno'])


def _build_tao_bundle(setup):
    """One TAO bundle from the FIRST phase's config — TAO doesn't get split."""
    s = setup
    args_tao = s['args_tao']
    det = TAODetector(
        ndays=s['ndays_tao'],
        ensp_nom=s['ensp_nom_tao'],
        rm=s['rm_tao'], cm=s['cm_tao'],
        args=args_tao,
        core_baselines=args_tao.core_baselines,
        core_powers_std=args_tao.core_powers,
        s_ibd=s['s_ibd'],
        s_bump_lin=s['s_bump_lin'],
        ene_leak_tao=s['ene_leak_tao'],
        E_segments=np.array(s['args_juno'].seg_bins_nonuniform))
    return DetectorBundle(
        phase='-', shell='-', kind='tao',
        args=args_tao, detector=det,
        backgrounds=s['backgrounds'], cm=s['cm_tao'],
        unc_key='all', ensp_nom=s['ensp_nom_tao'])


def _build_dyb_bundle(setup, dyb_model=None):
    """One DYB bundle from the FIRST phase's config."""
    s = setup
    det = DYBDetector(
        ensp_nom_dyb=s['ensp_nom_dyb'],
        ndays=s['ndays_dyb'],
        args_dyb=s['args_dyb'],
        args_juno=s['args_juno'],
        cm_dyb=s['cm_dyb'],
        rm_dyb=s['rm_dyb'],
        s_ibd=s['s_ibd'],
        dyb_model=dyb_model)
    return DetectorBundle(
        phase='-', shell='-', kind='dyb',
        args=s['args_dyb'], detector=det,
        backgrounds=s['backgrounds'], cm=s['cm_dyb'] or {},
        unc_key='all', ensp_nom=s['ensp_nom_dyb'] or {})


# ─────────────────────────────────────────────────────────────────────────────
# Multi-bundle path — N JUNO bundles + at most 1 TAO + 1 DYB
# ─────────────────────────────────────────────────────────────────────────────
def build_setups(config_path, *, remaining_argv=None, dm2_31=None,
                 base_path=None):
    """Manifest → one setup_one_phase() dict per (phase, shell) row, with the
    per-dataset ROOT suffix applied exactly as the main run_noda flow does.

    Shared by run_noda (main) and the FC drivers so the setup never drifts
    between the two. `dm2_31` (physical eV², signed) is forwarded to
    setup_one_phase for FC truth-spectrum builds.

    NODA_BASE is resolved and published here (CLI base_path > $NODA_BASE >
    config `base_path:` > repo root) BEFORE any setup is built, so every
    ${NODA_BASE} token in the configs expands identically whether the entry
    point is run_noda or an FC driver — the FC tools no longer depend on the
    caller exporting NODA_BASE.
    """
    with open(config_path, 'r') as f:
        manifest_doc = yaml.safe_load(f) or {}
    global NODA_BASE
    NODA_BASE = _resolve_base_path(base_path, manifest_doc)
    os.environ['NODA_BASE'] = NODA_BASE        # so child modules see it too
    print(f"[base] NODA_BASE = {NODA_BASE}")

    rows = load_manifest(config_path)
    return [setup_one_phase(cfg,
                            phase_label=phase, shell_label=shell,
                            remaining_argv=remaining_argv,
                            overrides=phase_ov, dm2_31=dm2_31,
                            root_suffix=_dataset_suffix(phase, shell, rows))
            for phase, shell, cfg, phase_ov in rows]


def build_policy(setups, manifest_doc):
    """ParamPolicy from the first dataset's JUNO backgrounds plus the manifest
    `systematics:` override — identical to the construction in run_noda.

    The per-phase / per-shell suffixes are only enabled when the run actually
    spans more than one phase / shell (same rule as _dataset_suffix), so a
    single-phase single-shell run stores nuisances under bare names."""
    sys_override = (manifest_doc.get('systematics')
                    if isinstance(manifest_doc, dict) else None)
    phases = {s['args_juno'].phase for s in setups}
    shells = {s['args_juno'].shell for s in setups}
    return policy_from_backgrounds(
        setups[0]['backgrounds']['JUNO'], sys_override=sys_override,
        multi_phase=len(phases) > 1, multi_shell=len(shells) > 1)


def _run_multi_bundle(setups, manifest_doc):
    """
    Build all bundles, build the param policy, run the bundled fit.
    """
    first = setups[0]
    args_juno = first['args_juno']

    bundles = []

    # ── one JUNO bundle per phase × shell ────────────────────────────────
    for s in setups:
        bundles.append(_build_juno_bundle(s))

    # ── ONE TAO bundle (if requested), taken from FIRST phase ────────────
    if (args_juno.fit_type == 'PMOP_combined'
            and getattr(args_juno, 'include_tao', False)
            and first['ensp_nom_tao'] is not None):
        bundles.append(_build_tao_bundle(first))

    # ── ONE DYB bundle (if requested), taken from FIRST phase ────────────
    if (args_juno.fit_type == 'PMOP_combined'
            and getattr(args_juno, 'include_dyb', False)):
        if args_juno.use_dagflow:
            from .model.DYB_model import make_dyb_model
            dyb_model = make_dyb_model(first['args_dyb'])
            bundles.append(_build_dyb_bundle(first, dyb_model=dyb_model))
        elif first['cm_dyb'] is not None:
            bundles.append(_build_dyb_bundle(first))

    # ── param-correlation policy ─────────────────────────────────────────
    policy = build_policy(setups, manifest_doc)

    # ── prefit cache ────────────────────────────────────────────────────
    prefit_cache = None
    if (getattr(args_juno, 'fit_mode', 'combined').startswith('prefit')
            and getattr(args_juno, 'use_prefit', False)):
        prefit_cache = load_prefit_cache(args_juno.prefit_path)

    # ── go ──────────────────────────────────────────────────────────────
    print(f"\n══════ running bundled fit  "
          f"({sum(1 for b in bundles if b.kind=='juno')} JUNO "
          f"+ {sum(1 for b in bundles if b.kind=='tao')} TAO "
          f"+ {sum(1 for b in bundles if b.kind=='dyb')} DYB bundles) ══════")
    return minuit_new.run_minuit_bundled(
        bundles, policy, args_juno, prefit_cache=prefit_cache)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    start = datetime.now()

    # ── locate the config file ───────────────────────────────────────────
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument('--config', default='config/legacy/groupA_paperconfig.yaml',
                             help='Path to config file or manifest YAML')
    conf_parser.add_argument('--base_path', default=None,
                             help='Root of the NODA tree, substituted for '
                                  '${NODA_BASE} in all config/background paths. '
                                  'Overrides $NODA_BASE and auto-detection.')
    args, remaining_argv = conf_parser.parse_known_args(argv)

    # ── load manifest (a plain single-phase YAML is a 1-row manifest) ─────
    # Needed here for _run_multi_bundle's param policy; build_setups loads it
    # again internally to resolve NODA_BASE.
    with open(args.config, 'r') as f:
        manifest_doc = yaml.safe_load(f) or {}

    # CLI overrides apply to every phase (so flags like --use_prefit 1
    # take effect uniformly). Per-phase differences live either in
    # per-phase config files or, with a `common:` + `phases:` manifest,
    # in the per-phase override blocks. build_setups resolves/publishes
    # NODA_BASE (CLI > env > config base_path > repo root) before building.
    setups = build_setups(args.config, remaining_argv=remaining_argv,
                          base_path=args.base_path)
    m = _run_multi_bundle(setups, manifest_doc)

    print(f"\nTotal wall time: {datetime.now() - start}")



if __name__ == "__main__":
    main(sys.argv[1:])
