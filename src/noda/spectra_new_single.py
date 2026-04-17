#!/usr/bin/env python3
"""
spectra.py – Reactor antineutrino spectra builder for JUNO / TAO / DYB.

Public API
----------
CreateSpectra(ndays, ebins, detector, backgrounds, args) -> (ensp, resp_matrix)
    or -> (ensp, resp_matrix, ene_leak_nd)   when detector == "TAO"
"""

import os
import sys
import csv

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import simpson, trapezoid
from scipy import constants
from joblib import Parallel, delayed
from datetime import datetime

from .noda import *


# ---------------------------------------------------------------------------
# Constants / index ranges
# ---------------------------------------------------------------------------

# Core index slices for JUNO (10 cores total: 6 YJ + 2 TS + 2 HM)
IDX_YJ = range(0, 6)
IDX_TS = range(6, 8)
IDX_HM = range(8, 10)

_CORE_GROUP = {}
for _i in IDX_YJ: _CORE_GROUP[_i] = ("YJ", _i - IDX_YJ.start + 1)
for _i in IDX_TS: _CORE_GROUP[_i] = ("TS", _i - IDX_TS.start + 1)
for _i in IDX_HM: _CORE_GROUP[_i] = ("HM", _i - IDX_HM.start + 1)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def interp_to_target(source_spec, target_spec):
    """Linearly interpolate *source_spec* onto the bin centres of *target_spec*."""
    src_x = source_spec.GetBinCenters()
    src_y = source_spec.bin_cont
    tgt_x = target_spec.GetBinCenters()
    f = sp.interpolate.interp1d(
        src_x, src_y, kind="linear",
        bounds_error=False,
        fill_value=(src_y[0], src_y[-1]),
    )
    new_spec = target_spec.Copy()
    new_spec.bin_cont = f(tgt_x)
    return new_spec


def _zero_spectrum(ebins):
    return Spectrum(bins=ebins, bin_cont=np.zeros(len(ebins) - 1))


# ---------------------------------------------------------------------------
# Energy-response chain helpers
# ---------------------------------------------------------------------------

def _apply_positron_energy(spec, args):
    return spec.GetWithPositronEnergy(
        type=args.posE_shift,
        inputfile=args.input_data_file,
        tf2name=args.pos_ene_TF2,
    )


def _apply_nonlinearity(spec, args, ensp):
    if args.NL == "pull":
        return spec.GetWithModifiedEnergy(mode="spectrum", spectrum=ensp["scintNL"])
    return spec.GetWithModifiedEnergy(
        mode="G4", kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared
    )


def _apply_full_energy_response(spec, args, ensp, resp_matrix):
    """positron shift → non-linearity → detector response."""
    s = _apply_positron_energy(spec, args)
    s = _apply_nonlinearity(s, args, ensp)
    return s.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)


# ---------------------------------------------------------------------------
# Response matrix
# ---------------------------------------------------------------------------

def _load_or_build_resp_matrix(args, detector, ebins):
    cache_path = (
        f"{args.data_matrix_folder}/"
        f"rm_{detector}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat"
    )

    if os.path.isfile(cache_path) and not bool(args.FORCE_CALC_RM):
        return LoadRespMatrix(cache_path)

    resp_matrix = CalcRespMatrix_abc(args.a, args.b, args.c, escale=1, ebins=ebins, pebins=ebins)

    if detector != "JUNO":
        ene_leak_nd = _build_energy_leak(args, detector, ebins)
        resp_matrix = RespMatrix(data=ene_leak_nd.data @ resp_matrix.data, ebins=ebins, pebins=ebins)

    resp_matrix.Save(cache_path)
    return resp_matrix


def _build_energy_leak(args, detector, ebins):
    if detector == "TAO":
        return CalcEnergyLeak(
            rootfile=args.data_spectrum,
            histname=args.IAV_hist,
            ebins=ebins,
            pebins=ebins,
        )
    if detector == "DYB":
        iav_data = np.loadtxt(args.iav_matrix_file)
        row_sums = iav_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return RespMatrix(data=iav_data / row_sums, ebins=ebins, pebins=ebins)
    raise ValueError(f"Unknown detector for energy-leak calculation: {detector!r}")


# ---------------------------------------------------------------------------
# IBD cross-section / flux helpers
# ---------------------------------------------------------------------------

def _build_ibd_interpolator(args):
    """Return (spectrum, interpolator) for the IBD cross-section."""
    sibd_spec = GetSpectrumFromROOT(args.input_data_file, "IBDXsec_VogelBeacom_DYB")
    s_ibd = sp.interpolate.interp1d(
        sibd_spec.GetBinCenters(), sibd_spec.bin_cont,
        kind="slinear", bounds_error=False, fill_value=(0.0, 0.0),
    )
    return sibd_spec, s_ibd


def _build_bump_interpolator(args):
    bump = GetSpectrumFromROOT(args.input_data_file, "DYBFluxBump_ratio")
    return sp.interpolate.interp1d(
        bump.GetBinCenters(), bump.bin_cont,
        kind="slinear", bounds_error=False,
        fill_value=(bump.bin_cont[0], bump.bin_cont[-1]),
    ), bump


def _build_huber_mueller_flux(args, fracs, ebins):
    """Sum Huber-Mueller isotope fluxes, rebinned to *ebins*."""
    flux = (
        GetSpectrumFromROOT(args.input_data_file, "HuberMuellerFlux_U235",  scale=fracs["U235"])
        + GetSpectrumFromROOT(args.input_data_file, "HuberMuellerFlux_U238",  scale=fracs["U238"])
        + GetSpectrumFromROOT(args.input_data_file, "HuberMuellerFlux_Pu239", scale=fracs["Pu239"])
        + GetSpectrumFromROOT(args.input_data_file, "HuberMuellerFlux_Pu241", scale=fracs["Pu241"])
    )
    flux.Rebin(ebins, mode="spline-not-keep-norm")
    return flux


def _weight_flux_with_ibd(flux, s_ibd, s_bump_lin, args, ndays, extrafactor):
    """Apply IBD cross-section, optional bump, day scaling, and baseline factor."""
    rfis = flux.Copy()
    rfis.WeightWithFunction(s_ibd)
    rfis.ScaleByBinWidth()
    if args.apply_bump:
        rfis = rfis.GetWeightedWithFunction(s_bump_lin)
    rfis.GetScaled(ndays)
    rfis.GetScaled(extrafactor)
    return rfis


# ---------------------------------------------------------------------------
# Non-linearity loading
# ---------------------------------------------------------------------------

def _load_nonlinearity(ensp, args, detector):
    """Populate ensp['scintNL'] and ensp['NL_pull'] when NL == 'pull'."""
    if args.NL != "pull":
        return

    ensp["scintNL"] = GetSpectrumFromROOT(args.NL_data_file, args.nl_hist_name)
    ensp["NL_pull"] = [
        GetSpectrumFromROOT(args.NL_data_file, f"{args.nl_pull_hist_name}{k}")
        for k in range(4)
    ]

    # Special rescaling for the J22rc0 parametrisation
    if detector == "JUNO" and args.nl_hist_name == "J22rc0_positronScintNL":
        old_nl = GetSpectrumFromROOT(args.input_data_file, "positronScintNL")
        ratio = ensp["scintNL"].bin_cont / old_nl.bin_cont
        ensp["NL_pull"] = [
            Spectrum(bins=p.bins, bin_cont=p.bin_cont * ratio)
            for p in ensp["NL_pull"]
        ]


# ---------------------------------------------------------------------------
# SNF / Non-equilibrium corrections  (JUNO per-core path)
# ---------------------------------------------------------------------------

def _build_snf_noneq_corrections_per_core(ensp, args, ebins, core_fractions, core_baselines, core_powers, resp_matrix):
    """
    Build oscillated, smeared SNF and Non-Eq spectra for each of the 10 JUNO
    cores and accumulate group totals.  Modifies *ensp* in-place.
    """
    n_cores = len(core_powers)

    # --- global HM-only SNF / Noneq base ratios ---
    for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
        ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
        ensp[key].Rebin(ebins, mode="spline-not-keep-norm")

    for suffix, fc_key, db_key in [
        ("snf",   "snf_FC",   "snf_DB"),
        ("noneq", "noneq_FC", "noneq_DB"),
    ]:
        ensp[fc_key] = interp_to_target(ensp[f"{suffix}0"], ensp["rfis_core8"]).GetWeightedWithSpectrum(ensp["rfis_core8"])
        ensp[db_key] = interp_to_target(ensp[f"{suffix}0"], ensp["rfis_core9"]).GetWeightedWithSpectrum(ensp["rfis_core9"])
        del ensp[f"{suffix}0"]

    # --- per-core SNF from dedicated histograms (YJ + TS only; HM uses global) ---
    snf_core = []
    noneq_core = []

    for i in range(n_cores):
        prefix, core_idx = _CORE_GROUP[i]
        rfis_i = ensp[f"rfis_core{i}"]

        # SNF
        if i in IDX_HM:
            # HM cores: use global ratio already weighted per-core
            fc_or_db = ensp["snf_FC"] if i == 8 else ensp["snf_DB"]
            snf_core.append(fc_or_db.Copy())
        else:
            spec = GetSpectrumFromROOT(args.snf_file, f"Ratio{prefix}ReactorSNF{core_idx}")
            spec.Rebin(ebins, mode="spline-not-keep-norm")
            snf_core.append(interp_to_target(spec, rfis_i).GetWeightedWithSpectrum(rfis_i))

        # Non-equilibrium
        if i in IDX_HM:
            fc_or_db = ensp["noneq_FC"] if i == 8 else ensp["noneq_DB"]
            noneq_core.append(fc_or_db.Copy())
        else:
            group_noneq = None
            for iso, frac in core_fractions[i].items():
                if iso == "U238":
                    continue
                spec = GetSpectrumFromROOT(args.noneq_file, f"{prefix}{core_idx}_{iso}")
                spec.Rebin(ebins, mode="spline-not-keep-norm")
                weighted = interp_to_target(spec.GetScaledFit(frac * 0.01), rfis_i).GetWeightedWithSpectrum(rfis_i)
                group_noneq = weighted.Copy() if group_noneq is None else group_noneq + weighted
            noneq_core.append(group_noneq)

    # --- YJ shutdown contributions ---
    def _load_shutdown(hist, core_idx):
        spec = GetSpectrumFromROOT(args.snf_file, hist)
        spec.Rebin(ebins, mode="spline-not-keep-norm")
        return interp_to_target(spec, ensp[f"rfis_core{core_idx}"]).GetWeightedWithSpectrum(ensp[f"rfis_core{core_idx}"])

    snf_core[0] += _load_shutdown("RatioYJReactor1Shutdown", 0)
    snf_core[3] += _load_shutdown("RatioYJReactor4Shutdown", 3)

    # --- Accumulate group totals ---
    for grp in ("snf", "noneq"):
        for lbl in ("YJ", "TS", "HM", "final"):
            ensp[f"{grp}_{lbl}"] = _zero_spectrum(ebins)

    for i in range(n_cores):
        s_i = snf_core[i]
        n_i = noneq_core[i]

        s_osc = s_i.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", args=args,
        )
        n_osc = n_i.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", args=args,
        )

        final_snf_i   = _apply_full_energy_response(s_osc, args, ensp, resp_matrix)
        final_noneq_i = _apply_full_energy_response(n_osc, args, ensp, resp_matrix)

        ensp["snf_final"]   += final_snf_i
        ensp["noneq_final"] += final_noneq_i

        group_label = _CORE_GROUP[i][0]   # "YJ", "TS", or "HM"
        ensp[f"snf_{group_label}"]   += final_snf_i
        ensp[f"noneq_{group_label}"] += final_noneq_i

        print(f"{group_label}{i}  snf={s_i.GetIntegral():.2f}  noneq={n_i.GetIntegral():.2f}")

    print(f"Final SNF Integral:   {ensp['snf_final'].GetIntegral():.2f}")
    print(f"Final Noneq Integral: {ensp['noneq_final'].GetIntegral():.2f}")


def _build_snf_noneq_simple(ensp, args, ebins, detector, resp_matrix):
    """
    SNF / Non-Eq for the single-baseline path (TAO, DYB, CI-powers JUNO).
    Modifies *ensp* in-place.
    """
    for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
        ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
        ensp[key].Rebin(ebins, mode="spline-not-keep-norm")

    ensp["snf"]   = ensp["snf0"].GetWeightedWithSpectrum(ensp["rfis"])
    ensp["noneq"] = ensp["noneq0"].GetWeightedWithSpectrum(ensp["rfis"])

    if detector == "DYB":
        ensp["noneq"] = ensp["noneq"].GetScaledFit(0.0071 / 0.0065)

    del ensp["snf0"], ensp["noneq0"]

    if detector != "DYB":
        osc_kwargs = dict(
            L=args.core_baselines, core_powers=args.core_powers,
            me_rho=args.me_rho, ene_mode="true", args=args,
        )
        ensp["snf_osc"]   = ensp["snf"].GetOscillated(**osc_kwargs)
        ensp["noneq_osc"] = ensp["noneq"].GetOscillated(**osc_kwargs)
    else:
        ensp["snf_osc"]   = ensp["snf"]
        ensp["noneq_osc"] = ensp["noneq"]

    if detector != "DYB":
        ensp["snf_final"]   = _apply_full_energy_response(ensp["snf_osc"],   args, ensp, resp_matrix)
        ensp["noneq_final"] = _apply_full_energy_response(ensp["noneq_osc"], args, ensp, resp_matrix)
    else:
        # DYB: positron shift only, no NL
        ensp["snf_final"]   = _apply_positron_energy(ensp["snf_osc"],   args).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        ensp["noneq_final"] = _apply_positron_energy(ensp["noneq_osc"], args).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)


# ---------------------------------------------------------------------------
# Reactor flux builders
# ---------------------------------------------------------------------------

def _build_reactor_flux_per_core(ensp, args, ebins, s_ibd, s_bump_lin, ndays,
                                  core_names, core_powers, core_fractions, core_baselines):
    """Build per-core IBD-weighted reactor fluxes. Modifies *ensp* in-place."""
    rfis_sum  = None
    rfis0_sum = None

    extrafactors = compute_extrafactors_per_core(args, core_powers, core_baselines, core_fractions)

    for i, (power, baseline, fracs, ef) in enumerate(
        zip(core_powers, core_baselines, core_fractions, extrafactors)
    ):
        rfis0_i = _build_huber_mueller_flux(args, fracs, ebins)
        rfis0_sum = rfis0_i if rfis0_sum is None else rfis0_sum + rfis0_i
        ensp[f"rfis0_core{i}"] = rfis0_i

        rfis_i = _weight_flux_with_ibd(rfis0_i, s_ibd, s_bump_lin, args, ndays, ef)
        ensp[f"rfis_core{i}"] = rfis_i
        rfis_sum = rfis_i if rfis_sum is None else rfis_sum + rfis_i

        print(f"Core {i} IBD/day: {rfis_i.GetIntegral() / ndays:.2f}")

    ensp["rfis0"] = rfis0_sum
    ensp["rfis"]  = rfis_sum


def _build_reactor_flux_simple(ensp, args, ebins, s_ibd, s_bump_lin, ndays):
    """Build a single combined IBD-weighted reactor flux. Modifies *ensp* in-place."""
    fracs = dict(
        U235=args.U235_scale, U238=args.U238_scale,
        Pu239=args.Pu239_scale, Pu241=args.Pu241_scale,
    )
    rfis0 = _build_huber_mueller_flux(args, fracs, ebins)
    ensp["rfis0"] = rfis0

    rfis_b = rfis0.Copy()
    rfis_b.WeightWithFunction(s_ibd)
    rfis_b.ScaleByBinWidth()
    if args.apply_bump:
        rfis_b = rfis_b.GetWeightedWithFunction(s_bump_lin)

    ensp["rfis"] = rfis_b.Copy()
    ensp["rfis"].GetScaled(ndays)
    ensp["rfis_fit"] = ensp["rfis"].Copy()

    extrafactors = compute_extrafactors(args)
    print(f"extrafactors: {extrafactors}")
    ensp["rfis"].GetScaled(extrafactors)


# ---------------------------------------------------------------------------
# Geo-neutrino spectra  (JUNO analytic path)
# ---------------------------------------------------------------------------

def _build_geo_spectra(ensp, args, ebins, s_ibd, ndays, resp_matrix, name, rate, ratio):
    """
    Build uranium + thorium geo-neutrino spectra for a single component
    (*name* = 'geo', 'crust', or 'mantle').  Returns the summed spectrum.
    """
    def _geo_component(hist, scale_rate, name):
        spec = GetSpectrumFromROOT(args.geo_file, hist)
        spec.Rebin(ebins, mode="spline-not-keep-norm")
        spec.ScaleByBinWidth()
        spec.WeightWithFunction(s_ibd)
        ensp[f'rfis0_geo{name}'] = spec
        spec_scaled = spec.GetScaledFit(
            ndays * scale_rate / (args.duty_cycle * spec.GetIntegral())
        )
        vis = _apply_positron_energy(spec_scaled, args)
        vis_nl = _apply_nonlinearity(vis, args, ensp)
        reco = vis_nl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        if args.rebin_nonuniform:
            reco.Rebin_nonuniform(args.bins_nonuniform)
        return reco

    rate_u  = rate / (1 + ratio)
    rate_th = rate * ratio / (1 + ratio)

    ensp[f"{name}u"]  = _geo_component("geoU",  rate_u, "u")
    ensp[f"{name}th"] = _geo_component("geoTh", rate_th, "th")
    return ensp[f"{name}u"] + ensp[f"{name}th"]


# ---------------------------------------------------------------------------
# Background loading  (JUNO)
# ---------------------------------------------------------------------------

def _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays):
    """Load and scale all JUNO backgrounds. Returns combined background spectrum."""
    bckg = None
    for key, info in backgrounds["JUNO"].items():
        ensp[key] = GetSpectrumFromROOT(info["rootfile"], info["histname"])

        if key == "wr":
            ensp[key] = _process_world_reactor(ensp[key], args, ensp, resp_matrix, ebins)

        scale = float(info["rate"]) * ndays / (args.duty_cycle * ensp[key].GetIntegral())
        ensp[key].GetScaled(scale)

        if args.rebin_nonuniform:
            ensp[key].Rebin_nonuniform(args.bins_nonuniform)

        print(f"  {key}: {ensp[key].GetIntegral():.2f}  (rate={info['rate']})")
        bckg = ensp[key] if bckg is None else bckg + ensp[key]

    return bckg


def _process_world_reactor(wr_spec, args, ensp, resp_matrix, ebins):
    wr_spec.Rebin(ebins, mode="spline-not-keep-norm")
    pos = _apply_positron_energy(wr_spec, args)
    ensp['rfis_wr_pos'] = pos
    vis = _apply_nonlinearity(pos, args, ensp)
    return vis.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)


# ---------------------------------------------------------------------------
# Detector-specific total spectrum assembly
# ---------------------------------------------------------------------------

def _assemble_total_spectrum_dyb(ensp, args, ndays):
    extrafactors = compute_extrafactors(args)
    ensp["rtot"] = ensp["rdet"]

    if args.is_data:
        dyb = pd.read_csv(
            args.dyb_spectrum, delim_whitespace=True,
            names=["Bin_Center", "Sigma", "Rel_Unc"],
        )
        ensp["rtot"] = Spectrum(bins=ensp["rdet"].bins, bin_cont=dyb["Sigma"].to_numpy())
        ensp["rtot"].GetScaled(extrafactors)
        ensp["rtot"].GetScaled(ndays)
        ensp["rtot"].ScaleByBinWidth()
        if args.rebin_nonuniform:
            ensp["rtot"].Rebin_nonuniform(args.bins_nonuniform)
        ensp["data"] = ensp["rtot"]

    return (
        [ensp["rtot"]],
        ["DYB expectation", "DYB data"],
        ["brown", "steelblue"],
    )


def _assemble_total_spectrum_tao(ensp, args):
    ensp["bckg"] = GetSpectrumFromROOT(args.data_spectrum, args.bckg_hist_name)
    ensp["bckg"].ScaleByBinWidth()
    if args.rebin_nonuniform:
        ensp["bckg"].Rebin_nonuniform(args.bins_nonuniform)

    ensp["rtot"] = ensp["rdet"] + ensp["bckg"]

    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        ensp["data"].ScaleByBinWidth()
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

    return  ([ensp['data'], ensp['bckg']],
            ['TAO expectation', 'TAO data', 'TAO background'],
            ['brown', 'steelblue', 'orange'])

def _assemble_total_spectrum_juno(ensp, args):
    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

    events = {
        "rtot":   ensp["rtot"].GetIntegral(),
        "rdet":   ensp["rdet"].GetIntegral(),
        "rbckg":  ensp["bckg"].GetIntegral(),
        "ribd":   ensp["ribd"].GetIntegral(),
        "rosc":   ensp["rosc"].GetIntegral(),
        "acc":    ensp["acc"].GetIntegral(),
        "fastn":  ensp["fastn"].GetIntegral(),
        "lihe":   ensp["lihe"].GetIntegral(),
        "alphan": ensp["alphan"].GetIntegral(),
        "geo":    ensp["geo"].GetIntegral(),
        "atm":    ensp["atm"].GetIntegral(),
        "wr":     ensp["wr"].GetIntegral(),
    }
    _print_juno_event_summary(events)

    extra = [ensp[k] for k in ("acc", "geo", "lihe", "fastn", "alphan", "atm", "wr", "doublen", "bipo")]
    labels = [
        "Reactor antineutrinos", "Accidentals", "Geoneutrinos",
        "Li9/He8", "Fast neutrons", "(alpha, n)", "Atmospheric",
        "Reactors > 300 km", "Double neutrons", "BiPo214",
    ]
    colors = ["black", "darkred", "green", "navy", "orange", "magenta",
              "lightblue", "yellow", "lightgreen", "purple", "grey"]
    return extra, labels, colors


def _print_juno_event_summary(ev):
    print("\n" + "=" * 46)
    print(f"  Reac × IBD:          {ev['ribd']:>10.2f} events")
    print(f"  Oscillated:          {ev['rosc']:>10.2f} events")
    print(f"  Detected:            {ev['rdet']:>10.2f} events")
    print(f"  IBD + BG:            {ev['rtot']:>10.2f} events")
    print(f"  Backgrounds (total): {ev['rbckg']:>10.2f} events")
    print(f"    accidentals:       {ev['acc']:>10.2f}")
    print(f"    Li-9/He-8:         {ev['lihe']:>10.2f}")
    print(f"    fast n:            {ev['fastn']:>10.2f}")
    print(f"    (alpha,n):         {ev['alphan']:>10.2f}")
    print(f"    geo-nu:            {ev['geo']:>10.2f}")
    print(f"    atmospheric:       {ev['atm']:>10.2f}")
    print(f"    global reactors:   {ev['wr']:>10.2f}")
    print("=" * 46 + "\n")


# ---------------------------------------------------------------------------
# Optional write-out
# ---------------------------------------------------------------------------

def _write_juno_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}.root"
    ensp["rdet"].WritetoROOT("reactor", tag, "prefit")
    for key in backgrounds["JUNO"]:
        ensp[key].WritetoROOT(key, tag, "prefit")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def CreateSpectra(ndays=10, ebins=None, detector="JUNO", backgrounds=None, args=None):
    """
    Build all reactor antineutrino spectra for *detector* over *ndays* days.

    Parameters
    ----------
    ndays      : float  – live days
    ebins      : array  – bin edges in MeV
    detector   : str    – "JUNO", "TAO", or "DYB"
    backgrounds: dict   – background config (used for JUNO)
    args       : namespace – all configuration parameters

    Returns
    -------
    (ensp, resp_matrix)                 for JUNO / DYB
    (ensp, resp_matrix, ene_leak_nd)    for TAO
    """
    _ensure_dir(args.plots_folder)

    ensp  = {}
    binw  = (ebins[1] - ebins[0]) * 1000   # keV

    print(f"\n{'='*50}")
    print(f"  Building spectra  |  detector={detector}  ndays={ndays}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # 1. Common ingredients
    # ------------------------------------------------------------------
    sibd_spec, s_ibd  = _build_ibd_interpolator(args)
    ensp["sibd"]      = sibd_spec
    s_bump_lin, ensp["bump_corr"] = _build_bump_interpolator(args)

    resp_matrix = _load_or_build_resp_matrix(args, detector, ebins)
    # Keep ene_leak_nd for TAO return value
    if detector == "TAO":
        ene_leak_nd = _build_energy_leak(args, detector, ebins)

    _load_nonlinearity(ensp, args, detector)

    # ------------------------------------------------------------------
    # 2. Reactor flux  (per-core JUNO path vs. simple single-flux path)
    # ------------------------------------------------------------------
    print("# Building reactor flux …")

    if detector == "JUNO" and not args.CI_powers:
        core_names, core_powers, core_fractions, core_baselines = load_reactor_summary(args.reactor_summary_yaml)
        _build_reactor_flux_per_core(
            ensp, args, ebins, s_ibd, s_bump_lin, ndays,
            core_names, core_powers, core_fractions, core_baselines,
        )
    else:
        core_baselines = args.core_baselines
        core_powers    = args.core_powers
        _build_reactor_flux_simple(ensp, args, ebins, s_ibd, s_bump_lin, ndays)

    print(f"  Expected IBDs (no osc):  {ensp['rfis'].GetIntegral():.2f}")

    # ------------------------------------------------------------------
    # 3. SNF + Non-equilibrium corrections
    # ------------------------------------------------------------------
    print("# Building SNF / non-eq corrections …")

    if detector == "JUNO" and not args.CI_powers:
        _build_snf_noneq_corrections_per_core(
            ensp, args, ebins, core_fractions, core_baselines, core_powers, resp_matrix,
        )
    else:
        _build_snf_noneq_simple(ensp, args, ebins, detector, resp_matrix)

    # ------------------------------------------------------------------
    # 4. Oscillated spectrum
    # ------------------------------------------------------------------
    ensp["ribd"] = ensp["rfis"].Copy()

    if detector != "DYB":
        ensp["rosc"] = ensp["ribd"].GetOscillated(
            L=list(core_baselines), core_powers=list(core_powers),
            me_rho=args.me_rho, ene_mode="true", args=args,
        )
    else:
        ensp["rosc"] = ensp["ribd"]

    ensp["rosc"].Plot(
        f"{args.plots_folder}/osc_spectrum_{detector}.png",
        xlabel="Neutrino energy (MeV)",
        ylabel=f"Events per {binw:.1f} keV",
        extra_spectra=[ensp["ribd"]],
        leg_labels=["oscillated", "unoscillated"],
        colors=["black", "darkred"],
        xmin=0, xmax=10, ymin=0, ymax=None, log_scale=False,
    )

    # ------------------------------------------------------------------
    # 5. Energy-response chain for the main reactor spectrum
    # ------------------------------------------------------------------
    print("# Applying energy-response chain …")

    ensp["rvis_nonl"]      = _apply_positron_energy(ensp["rosc"], args)
    ensp["rvis_nonl_temp"] = _apply_positron_energy(ensp["ribd"], args)

    if detector != "DYB":
        ensp["rvis"]      = _apply_nonlinearity(ensp["rvis_nonl"],      args, ensp)
        ensp["rvis_temp"] = _apply_nonlinearity(ensp["rvis_nonl_temp"], args, ensp)
    else:
        ensp["rvis"]      = ensp["rvis_nonl"]
        ensp["rvis_temp"] = ensp["rvis_nonl_temp"]

    ensp["rdet"] = ensp["rvis"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

    # Add corrections
    if detector == "DYB":
        ensp["rdet"] = ensp["rdet"] + ensp["noneq_final"]
    else:
        ensp["rdet"] = ensp["rdet"] + ensp["snf_final"] + ensp["noneq_final"]

    if args.rebin_nonuniform:
        ensp["rdet"].Rebin_nonuniform(args.bins_nonuniform)


    # ------------------------------------------------------------------
    # 7. Backgrounds + total spectrum
    # ------------------------------------------------------------------
    print("# Assembling total spectrum …")

    if detector == "JUNO":
        ensp["bckg"] = _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays)
        # ------------------------------------------------------------------
        # 6. Geo-neutrino spectra  (JUNO analytic path)
        # ------------------------------------------------------------------
        if args.geo_spectra == "ana":
            print("# Building analytic geo-neutrino spectra …")
            if args.geo_fit_type == "mantle":
                ensp["geomantle"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "mantle",
                    args.mantle_rate.get(args.mantle_model), args.mantle_Th_U_ratio.get(args.mantle_model),
                )
                ensp["geocrust"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "crust",
                    float(args.crust_rate_tnu) * args.tnu_to_cpd, args.crust_Th_U_ratio,
                )
                ensp["geo_ana"] = ensp["geocrust"] + ensp["geomantle"]
            else:
                ensp["geo_ana"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "geo",
                    float(args.geo_tnu_rate) * float(args.tnu_to_cpd), args.Th_U_ratio,
                )
                ensp["bckg"] = ensp["bckg"] - ensp["geo"] + ensp["geo_ana"]
                ensp["geo"]  = ensp["geo_ana"]
                print(f"  geo: {ensp['geo'].GetIntegral():.2f}")
        ensp["rtot"] = ensp["rdet"] + ensp["bckg"]
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_juno(ensp, args)

        if args.write_output:
            _write_juno_output(ensp, args, backgrounds)

    elif detector == "DYB":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_dyb(ensp, args, ndays)

    elif detector == "TAO":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_tao(ensp, args)

    ensp["rdet"].Plot(
            f"{args.plots_folder}/det_spectra_{detector}.png",
            xlabel="Reconstructed energy (MeV)",
            ylabel="Events / bin",
            extra_spectra=extra_spectra,
            leg_labels=leg_labels,
            colors=colors,
            xmin=0.7, xmax=12.0, ymin=1e-3, log_scale=False,
        )
    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    _print_diagnostics(ensp, detector)

    if detector == "TAO":
        return ensp, resp_matrix, ene_leak_nd
    return ensp, resp_matrix


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostics(ensp, detector):
    print("\n--- Spectral integrals ---")
    for key in ("rfis0", "rfis", "ribd", "rosc", "rvis_nonl", "rvis", "rdet",
                "snf_final", "noneq_final", "rtot"):
        if key in ensp:
            print(f"  {key:<20s}: {ensp[key].GetIntegral():.2f}")
    print()
    print(f"  2–6 MeV (ribd): {ensp['ribd'].GetIntegral(left_edge=2, right_edge=6):.2f}")
