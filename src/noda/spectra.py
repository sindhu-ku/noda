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
from scipy.integrate import simpson, trapezoid, quad
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

def _apply_positron_energy(spec, args, s_ibd=None):
    return spec.GetWithPositronEnergy(
        type=args.posE_shift,
        s_ibd=s_ibd,
        inputfile=args.input_data_file,
        tf2name=args.pos_ene_TF2,
    )


def _apply_nonlinearity(spec, args, ensp):
    if args.NL == "pull":
        return spec.GetWithModifiedEnergy(mode="spectrum", spectrum=ensp["scintNL"])
    return spec.GetWithModifiedEnergy(
        mode="G4", kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared
    )


def _apply_full_energy_response(spec, args, ensp, resp_matrix, s_ibd):
    """positron shift → non-linearity → detector response."""
    s = _apply_positron_energy(spec, args, s_ibd=s_ibd)
    s = _apply_nonlinearity(s, args, ensp)
    return s.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)


# ---------------------------------------------------------------------------
# Response matrix
# ---------------------------------------------------------------------------

def _load_or_build_resp_matrix(args, detector, ebins):
    # --- NEW: use external DYB response matrix file ---
    if detector == "DYB" and args.model == "PRL":
        return _load_dyb_resp_matrix(args.dyb_resp_matrix, ebins)

    cache_path = (
        f"{args.data_matrix_folder}/"
        f"rm_{detector}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat"
    )
    if os.path.isfile(cache_path) and not bool(args.FORCE_CALC_RM):
        return LoadRespMatrix(cache_path)
    resp_matrix = CalcRespMatrix_abc(args.a, args.b, args.c, escale=1, ebins=ebins, pebins=ebins)
    if detector == "DYB":
        ene_leak_nd = _build_energy_leak(args, detector, ebins)
        resp_matrix = RespMatrix(data=ene_leak_nd.data @ resp_matrix.data, ebins=ebins, pebins=ebins)
    resp_matrix.Save(cache_path)
    return resp_matrix


def _load_dyb_resp_matrix(filepath, ebins):
    data = np.loadtxt(filepath, comments="#")  # shape as stored: (240_reco, 240_true)

    if data.shape != (240, 240):
        raise ValueError(f"Expected 240×240 DYB response matrix, got {data.shape}")

    return RespMatrix(data=data, ebins=ebins, pebins=ebins)  # → (true, reco)

def _build_energy_leak(args, detector, ebins):
    if detector == "TAO":
        return CalcEnergyLeak(
            rootfile=args.data_file,
            histname=args.IAV_hist,
            ebins=ebins,
            pebins=ebins,
            ecrop=0.8
        )
    if detector == "DYB":
        iav_data = np.loadtxt(args.iav_matrix_file)
        row_sums = iav_data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return RespMatrix(data=iav_data / row_sums, ebins=ebins, pebins=ebins)
    raise ValueError(f"Unknown detector for energy-leak calculation: {detector!r}")


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
        GetSpectrumFromROOT(args.NL_data_file, f"{args.nl_pull_hist_name.replace('key', str(k))}")
        for k in range(4)
    ]

    # new_nonl = Spectrum(bins=ensp['scintNL'].bins,
    #                     bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
    # new_nonl.bin_cont = (ensp['scintNL'].bin_cont
    #     + 0.5 * (ensp['NL_pull'][0].bin_cont - ensp['scintNL'].bin_cont)
    #     + 0.5 * (ensp['NL_pull'][1].bin_cont - ensp['scintNL'].bin_cont)
    #     + 0.5 * (ensp['NL_pull'][2].bin_cont - ensp['scintNL'].bin_cont)
    #     + 0.5 * (ensp['NL_pull'][3].bin_cont - ensp['scintNL'].bin_cont))*1.05

    # ensp['scintNL'].WritetoROOT("nominal", "NL_groupA.root")
    # new_nonl.WritetoROOT("modified", "NL_groupA.root")

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

# def _build_snf_noneq_corrections_per_core(ensp, args, ebins, core_fractions, core_baselines, core_powers, resp_matrix, s_ibd, dm2_31):
#     """
#     Build oscillated, smeared SNF and Non-Eq spectra for each of the 10 JUNO
#     cores and accumulate group totals.  Modifies *ensp* in-place.
#     """
#     n_cores = len(core_powers)

#     # --- global HM-only SNF / Noneq base ratios ---
#     for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
#         ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
#         ensp[key].Rebin(ebins, mode="spline-not-keep-norm")

#     for suffix, fc_key, db_key in [
#         ("snf",   "snf_FC",   "snf_DB"),
#         ("noneq", "noneq_FC", "noneq_DB"),
#     ]:
#         ensp[fc_key] = interp_to_target(ensp[f"{suffix}0"], ensp["rfis_core8"]).GetWeightedWithSpectrum(ensp["rfis_core8"])
#         ensp[db_key] = interp_to_target(ensp[f"{suffix}0"], ensp["rfis_core9"]).GetWeightedWithSpectrum(ensp["rfis_core9"])
#         del ensp[f"{suffix}0"]

#     # --- per-core SNF from dedicated histograms (YJ + TS only; HM uses global) ---
#     snf_core = []
#     noneq_core = []

#     for i in range(n_cores):
#         prefix, core_idx = _CORE_GROUP[i]
#         rfis_i = ensp[f"rfis_core{i}"]

#         # SNF
#         if i in IDX_HM:
#             # HM cores: use global ratio already weighted per-core
#             fc_or_db = ensp["snf_FC"] if i == 8 else ensp["snf_DB"]
#             snf_core.append(fc_or_db.Copy())
#         else:
#             spec = GetSpectrumFromROOT(args.snf_file, f"Ratio{prefix}ReactorSNF{core_idx}")
#             spec.Rebin(ebins, mode="spline-not-keep-norm")
#             snf_core.append(interp_to_target(spec, rfis_i).GetWeightedWithSpectrum(rfis_i))

#         # Non-equilibrium
#         if i in IDX_HM:
#             fc_or_db = ensp["noneq_FC"] if i == 8 else ensp["noneq_DB"]
#             noneq_core.append(fc_or_db.Copy())
#         else:
#             group_noneq = None
#             for iso, frac in core_fractions[i].items():
#                 if iso == "U238":
#                     continue
#                 spec = GetSpectrumFromROOT(args.noneq_file, f"{prefix}{core_idx}_{iso}")
#                 spec.Rebin(ebins, mode="spline-not-keep-norm")
#                 weighted = interp_to_target(spec.GetScaledFit(frac * 0.01), rfis_i).GetWeightedWithSpectrum(rfis_i)
#                 group_noneq = weighted.Copy() if group_noneq is None else group_noneq + weighted
#             noneq_core.append(group_noneq)

#     # --- YJ shutdown contributions ---
#     def _load_shutdown(hist, core_idx):
#         spec = GetSpectrumFromROOT(args.snf_file, hist)
#         spec.Rebin(ebins, mode="spline-not-keep-norm")
#         return interp_to_target(spec, ensp[f"rfis_core{core_idx}"]).GetWeightedWithSpectrum(ensp[f"rfis_core{core_idx}"])

#     snf_core[0] += _load_shutdown("RatioYJReactor1Shutdown", 0)
#     snf_core[3] += _load_shutdown("RatioYJReactor4Shutdown", 3)

#     # --- Accumulate group totals ---
#     for grp in ("snf", "noneq"):
#         for lbl in ("YJ", "TS", "HM", "final", "osc"):
#             ensp[f"{grp}_{lbl}"] = _zero_spectrum(ebins)

#     for i in range(n_cores):
#         s_i = snf_core[i]
#         n_i = noneq_core[i]

#         s_osc = s_i.GetOscillated(
#             L=[core_baselines[i]], core_powers=[core_powers[i]],
#             me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
#         )
#         n_osc = n_i.GetOscillated(
#             L=[core_baselines[i]], core_powers=[core_powers[i]],
#             me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
#         )
#         ensp["snf_osc"] +=  s_osc
#         ensp["noneq_osc"] += n_osc

#         final_snf_i   = _apply_full_energy_response(s_osc, args, ensp, resp_matrix, s_ibd)
#         final_noneq_i = _apply_full_energy_response(n_osc, args, ensp, resp_matrix, s_ibd)

#         ensp["snf_final"]   += final_snf_i
#         ensp["noneq_final"] += final_noneq_i

#         group_label = _CORE_GROUP[i][0]   # "YJ", "TS", or "HM"
#         ensp[f"snf_{group_label}"]   += final_snf_i
#         ensp[f"noneq_{group_label}"] += final_noneq_i

#         print(f"{group_label}{i}  snf={s_i.GetIntegral():.2f}  noneq={n_i.GetIntegral():.2f}")

#     print(f"Final SNF Integral:   {ensp['snf_final'].GetIntegral():.2f}")
#     print(f"Final Noneq Integral: {ensp['noneq_final'].GetIntegral():.2f}")


def _build_snf_noneq_simple(ensp, args, ebins, detector, resp_matrix, s_ibd, ene_leak=None):
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

    if detector == "JUNO":
        ensp["snf_final"]   = _apply_full_energy_response(ensp["snf"],   args, ensp, resp_matrix, s_ibd)
        ensp["noneq_final"] = _apply_full_energy_response(ensp["noneq"], args, ensp, resp_matrix, s_ibd)
    elif detector == "DYB":
        # DYB: positron shift only, no NL
        ensp["snf_final"]   = _apply_positron_energy(ensp["snf"],   args, s_ibd=s_ibd).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        ensp["noneq_final"] = _apply_positron_energy(ensp["noneq"], args, s_ibd=s_ibd).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
    else:
        ensp["snf_final_temp"]   = _apply_positron_energy(ensp["snf"],   args, s_ibd=s_ibd).ApplyDetResp(ene_leak, pecrop=args.ene_crop)
        ensp["noneq_final_temp"] = _apply_positron_energy(ensp["noneq"], args, s_ibd=s_ibd).ApplyDetResp(ene_leak, pecrop=args.ene_crop)

        ensp["snf_final"] = _apply_nonlinearity(ensp["snf_final_temp"],   args, ensp).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        ensp["noneq_final"] =  _apply_nonlinearity(ensp["noneq_final_temp"],   args, ensp).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

def _build_snf_noneq_corrections_per_core(ensp, args,ndays, ebins, core_fractions,
                                          core_baselines, core_powers,
                                          resp_matrix, s_ibd, dm2_31):
    """
    Per-core SNF / Non-Eq corrections using the COMMON flux-ratio histograms
    (SNF_FluxRatio / NonEq_FluxRatio from args.input_data_file) for every core.

    Each core's correction is the global ratio weighted by that core's
    IBD-folded flux (rfis_core{i}), oscillated with the core's own
    baseline/power, then run through the full energy-response chain.
    Group totals (YJ / TS / HM), the summed final spectra, and the
    pre-response oscillated sums are accumulated in-place on *ensp*.

    NOTE: core_fractions is accepted for call-site compatibility but no
    longer used — all cores share the common ratio.
    """
    n_cores = len(core_powers)

    # --- common SNF / Non-eq flux ratios (same histogram for all cores) ---
    for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
        ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
        ensp[key].Rebin(ebins, mode="spline-not-keep-norm")

    # --- per-core, weighted by each core's flux (already on ebins → no interp) ---
    snf_core   = [ensp["snf0"].GetWeightedWithSpectrum(ensp[f"rfis_core{i}"])
                  for i in range(n_cores)]
    noneq_core = [ensp["noneq0"].GetWeightedWithSpectrum(ensp[f"rfis_core{i}"])
                  for i in range(n_cores)]

    del ensp["snf0"], ensp["noneq0"]

    # --- zero the accumulators ---
    for grp in ("snf", "noneq"):
        for lbl in ("YJ", "TS", "HM", "final", "osc"):
            ensp[f"{grp}_{lbl}"] = _zero_spectrum(ebins)

    for i in range(n_cores):


        s_osc = snf_core[i].GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args)
        n_osc = noneq_core[i].GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args)

        ensp["snf_osc"]   += s_osc
        ensp["noneq_osc"] += n_osc

        final_snf_i   = _apply_full_energy_response(s_osc, args, ensp, resp_matrix, s_ibd)
        final_noneq_i = _apply_full_energy_response(n_osc, args, ensp, resp_matrix, s_ibd)

        ensp["snf_final"]   += final_snf_i
        ensp["noneq_final"] += final_noneq_i

        grp = _CORE_GROUP[i][0]   # "YJ", "TS", or "HM"
        ensp[f"snf_{grp}"]   += final_snf_i
        ensp[f"noneq_{grp}"] += final_noneq_i

        print(f"{grp}{i}  snf={snf_core[i].GetIntegral()/ndays:.2f}  "
              f"noneq={noneq_core[i].GetIntegral()/ndays:.2f}")

    print(f"Final SNF Integral:   {ensp['snf_final'].GetIntegral()/ndays:.2f}")
    print(f"Final Noneq Integral: {ensp['noneq_final'].GetIntegral()/ndays:.2f}")
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
    #ensp['rfis'].GetScaled(1e+43)
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
        data = np.loadtxt(hist)
        energies = data[:, 0]*1e-3
        flux = data[:, 1]
        bin_edges = np.append(energies, energies[-1] + (energies[1] - energies[0]))
        spec = Spectrum(
            bin_cont=flux,
            bins=bin_edges,
            xlabel="Energy [MeV]",
            ylabel="Flux"
        )
        #spec = GetSpectrumFromROOT(args.geo_file, hist)
        spec.Rebin(ebins, mode="spline-not-keep-norm")
        spec.ScaleByBinWidth()
        spec.WeightWithFunction(s_ibd)
        ensp[f'rfis0_geo{name}'] = spec
        spec_scaled = spec.GetScaledFit(
            ndays * scale_rate / (args.duty_cycle * spec.GetIntegral())
        )
        vis = _apply_positron_energy(spec_scaled, args, s_ibd=s_ibd)
        vis_nl = _apply_nonlinearity(vis, args, ensp)
        reco = vis_nl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        reco.GetScaled(args.detector_efficiency)
        if args.rebin_nonuniform:
            reco.Rebin_nonuniform(args.bins_nonuniform)
        return reco

    rate_u  = rate / (1 + ratio)
    rate_th = rate * ratio / (1 + ratio)

    ensp[f"{name}u"]  = _geo_component(args.geoU_file,  rate_u, "u")
    ensp[f"{name}th"] = _geo_component(args.geoTh_file, rate_th, "th")
    return ensp[f"{name}u"] + ensp[f"{name}th"]


# ---------------------------------------------------------------------------
# Background loading  (JUNO)
# ---------------------------------------------------------------------------

def _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays, s_ibd):
    """Load and scale all JUNO backgrounds. Returns combined background spectrum."""
    bckg = None
    for key, info in backgrounds["JUNO"].items():
        ensp[key] = GetSpectrumFromROOT(info["rootfile"], info["histname"])

        if key == "wr":
            ensp[key] = _process_world_reactor(ensp[key], args, ensp, resp_matrix, ebins, s_ibd)

        scale = float(info["rate"]) * ndays / (args.duty_cycle * ensp[key].GetIntegral())
        ensp[key].GetScaled(scale)

        if key == "wr" or key == "geo" or key == "atm":
            ensp[key].GetScaled(args.detector_efficiency)
        if args.rebin_nonuniform:
            ensp[key].Rebin_nonuniform(args.bins_nonuniform)

        print(f"  {key}: {ensp[key].GetIntegral():.2f}  (rate={info['rate']*args.detector_efficiency})")
        bckg = ensp[key] if bckg is None else bckg + ensp[key]

    return bckg


def _process_world_reactor(wr_spec, args, ensp, resp_matrix, ebins, s_ibd):
    wr_spec.Rebin(ebins, mode="spline-not-keep-norm")
    pos = _apply_positron_energy(wr_spec, args,s_ibd=s_ibd)
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
        if args.model == "CPC":
            names = ["Bin_Center", "Sigma", "Rel_Unc"]
        else:
            names = ["Bin_lowE", "Bin_highE", "Sigma", "bin_err"]
        dyb = pd.read_csv(
            args.dyb_spectrum, sep='\s+',
            names=names,
        )

        ensp["data"] = Spectrum(bins=ensp["rdet"].bins, bin_cont=dyb["Sigma"].to_numpy())
        if args.model == "PRL":
            ensp["data"].GetScaled(1e-43)

        ensp["data"].GetScaled(extrafactors)
        ensp["data"].GetScaled(ndays)
        ensp["data"].ScaleByBinWidth()


        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

        return (
                [ensp["data"]],
                ["DYB expectation", "DYB data"],
                ["brown", "steelblue"],
            )

    if args.rebin_nonuniform:
        ensp["rtot"].Rebin_nonuniform(args.bins_nonuniform)


    return (
        [ensp["rtot"]],
        ["DYB expectation", "DYB data"],
        ["brown", "steelblue"],
    )


def _assemble_total_spectrum_tao(ensp, args, ndays):
    ensp["bckg"] = GetSpectrumFromROOT(args.data_file, args.bckg_hist_name)
    ensp["bckg"].ScaleByBinWidth()
    #ensp["bckg"].GetScaled(ndays*args.bckg_rate)
    print("TAO background: ", ensp['bckg'].GetIntegral())
    if args.rebin_nonuniform:
        ensp["bckg"].Rebin_nonuniform(args.bins_nonuniform)

    ensp["rtot"] = ensp["rdet"] + ensp["bckg"]

    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        ensp["data"].ScaleByBinWidth()
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)


        return  ([ensp['data'], ensp['bckg'], ensp['rtot']],
                ['TAO reactor expectation', 'TAO data', 'TAO background', 'TAO reactor+bckg expectation'],
                ['brown', 'steelblue', 'orange', 'black'])
    return  ([ensp['rtot'], ensp['bckg']],
                ['TAO expectation', 'TAO data', 'TAO background'],
                ['brown', 'steelblue', 'orange'])

def _assemble_total_spectrum_juno(ensp, args):

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

    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

        extra.append(ensp['data'])
        labels.append('Data')
        colors.append('blue')
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

def _build_nl_graph(ensp, args, kB=None, fC=None, kI=None, LY_shared=None):
    """Positron NL curve (Evis/Edep vs Edep), same convention as the ORSA
    export (juno/response/<stage>/nl_positron): Geant4 model anchored to the
    2.224589 MeV gamma.  Parameters default to the pre-fit nominals."""
    if args.NL == "pull":
        e = 0.5 * (ensp["scintNL"].bins[:-1] + ensp["scintNL"].bins[1:])
        ratio = ensp["scintNL"].bin_cont
    else:
        kB = args.kB if kB is None else kB
        fC = args.fC if fC is None else fC
        kI = args.kI if kI is None else kI
        LY_shared = args.LY_shared if LY_shared is None else LY_shared
        e = np.arange(1.022, 12.0, 0.001)
        ratio = (G4model.eval(e, kB, fC=fC, kI=kI) * LY_shared
                 / G4model_gamma.eval(2.224589, kB, fC=fC, kI=kI))
    graph = ROOT.TGraph(len(e),
                        np.ascontiguousarray(e, dtype=float),
                        np.ascontiguousarray(ratio, dtype=float))
    graph.SetName("nl_positron")
    return graph


def _write_juno_output(ensp,ndays, args, backgrounds):
    from .fit.minuit import write_param_dbl
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    juno_dir = output_file.mkdir("juno")
    juno_dir.cd()
    write_param_dbl("LivetimeDays", ndays)

    dir_pre = juno_dir.mkdir("prefit")
    dir_pre.cd()
    write_param_dbl("Efficiency", args.detector_efficiency)

    dir_resp = juno_dir.mkdir("response")
    dir_resp.cd()
    dir_resp_prefit = dir_resp.mkdir('prefit')
    dir_resp_prefit.cd()
    _build_nl_graph(ensp, args).Write("", ROOT.TObject.kOverwrite)
    output_file.Close()


    ensp["rdet"].WritetoROOT("reactor", tag, "juno/prefit")
    ensp["rtot"].WritetoROOT("totalpred", tag, "juno/prefit")
    for key in backgrounds["JUNO"]:
        ensp[key].WritetoROOT(key, tag, "juno/prefit")
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "juno")



def _write_dyb_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    output_file.mkdir("dyb")
    output_file.Close()

    ensp["rdet"].WritetoROOT("totalpred", tag, "dyb/prefit")
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "dyb")


def _write_tao_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    output_file.mkdir("tao")
    output_file.Close()

    ensp["rtot"].WritetoROOT("totalpred", tag, "tao/prefit")
    ensp["bckg"].WritetoROOT("bckg", tag, "tao/prefit")
    ensp["rdet"].WritetoROOT("reactor", tag, "tao/prefit")
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "tao")
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def CreateSpectra(ndays=10, ebins=None, detector="JUNO", s_ibd=None, s_bump_lin=None, backgrounds=None, args=None, dm2_31=None):
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


    resp_matrix = _load_or_build_resp_matrix(args, detector, ebins)
    # Keep ene_leak_nd for TAO return value
    ene_leak_nd=None
    if detector == "TAO":
        ene_leak_nd = _build_energy_leak(args, detector, ebins)

    _load_nonlinearity(ensp, args, detector)

    # ------------------------------------------------------------------
    # 2. Reactor flux  (per-core JUNO path vs. simple single-flux path)
    # ------------------------------------------------------------------
    print("# Building reactor flux …")

    if detector == "JUNO": # and not args.CI_powers:
        print("Building spectrum with unblinded powers")
        core_names, core_powers, core_fractions, core_baselines = load_reactor_summary(args.reactor_summary_yaml)
        _build_reactor_flux_per_core(
            ensp, args, ebins, s_ibd, s_bump_lin, ndays,
            core_names, core_powers, core_fractions, core_baselines,
        )
        _build_snf_noneq_corrections_per_core(
            ensp, args,ndays, ebins, core_fractions, core_baselines, core_powers, resp_matrix,s_ibd, dm2_31,
        )
    else:
        print("Building spectrum with powers from config")
        core_baselines = args.core_baselines
        core_powers    = args.core_powers
        _build_reactor_flux_simple(ensp, args, ebins, s_ibd, s_bump_lin, ndays)
        _build_snf_noneq_simple(ensp, args, ebins, detector, resp_matrix, s_ibd, ene_leak=ene_leak_nd)

    print(f"  Expected IBDs (no osc):  {ensp['rfis'].GetIntegral():.2f}")

    # ------------------------------------------------------------------
    # 3. SNF + Non-equilibrium corrections
    # ------------------------------------------------------------------
    print("# Building SNF / non-eq corrections …")

    #if detector == "JUNO" and not args.CI_powers:

    # else:
    #
    # ------------------------------------------------------------------
    # 4. Oscillated spectrum
    # ------------------------------------------------------------------
    ensp["ribd"] = ensp["rfis"].Copy()
    if detector == "JUNO":
        ensp["rosc"] = ensp["ribd"].GetOscillated(
            L=list(core_baselines), core_powers=list(core_powers),
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
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

    if detector == "DYB" and args.model == "PRL":
        ensp["rdet"] = ensp["rosc"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

    else:
        ensp["rvis_nonl"]      = _apply_positron_energy(ensp["rosc"], args,s_ibd=s_ibd)
        ensp["rvis_nonl_temp"] = _apply_positron_energy(ensp["ribd"], args, s_ibd=s_ibd)
        #ensp["rvis_nonl_temp_wcorr"] = _apply_positron_energy((ensp["rosc"]+ensp['snf_osc']+ensp['noneq_osc']), args, s_ibd=s_ibd)
        if detector == "TAO":
            ensp["rvis_nonl_leak"] = ensp["rvis_nonl"].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)
            ensp["rvis_nonl_temp_leak"] = ensp["rvis_nonl_temp"].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)

        else:
            ensp["rvis_nonl_leak"] = ensp["rvis_nonl"]
            ensp["rvis_nonl_temp_leak"] = ensp["rvis_nonl_temp"]

        if detector != "DYB":
            ensp["rvis"]      = _apply_nonlinearity(ensp["rvis_nonl_leak"],      args, ensp)
            ensp["rvis_temp"] = _apply_nonlinearity(ensp["rvis_nonl_temp_leak"], args, ensp)
            #ensp["rvis_temp_wcorr"] = _apply_nonlinearity(ensp["rvis_nonl_temp_wcorr"], args, ensp)
        else:
            ensp["rvis"]      = ensp["rvis_nonl"]
            ensp["rvis_temp"] = ensp["rvis_nonl_temp"]
            #ensp["rvis_temp_wcorr"] = ensp["rvis_nonl_temp_wcorr"]

        ensp["rdet"] = ensp["rvis"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        ensp["rdet_temp"] = ensp["rvis_temp"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        #ensp["rdet_temp_wcorr"] = ensp["rvis_temp_wcorr"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

    # Add corrections
    if detector == "DYB":
        if args.model == "CPC":
            ensp["rdet"] = ensp["rdet"] + ensp["noneq_final"]
        else:
            ensp["rdet"] = ensp["rdet"]
    else:
        ensp["rdet"] = ensp["rdet"] + ensp["snf_final"] + ensp["noneq_final"]

    if args.rebin_nonuniform:
        ensp["rdet"].Rebin_nonuniform(args.bins_nonuniform)


    # ------------------------------------------------------------------
    # 7. Backgrounds + total spectrum
    # ------------------------------------------------------------------
    print("# Assembling total spectrum …")

    if detector == "JUNO":
        ensp["bckg"] = _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays, s_ibd=s_ibd)
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
            _write_juno_output(ensp, ndays, args, backgrounds)

    elif detector == "DYB":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_dyb(ensp, args, ndays)
        if args.write_output: _write_dyb_output(ensp, args, backgrounds)
    elif detector == "TAO":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_tao(ensp, args, ndays)
        if args.write_output: _write_tao_output(ensp, args, backgrounds)
    ensp["rdet"].Plot(
            f"{args.plots_folder}/det_spectra_{detector}.png",
            xlabel="Reconstructed energy (MeV)",
            ylabel="Events / bin",
            extra_spectra=extra_spectra,
            leg_labels=leg_labels,
            colors=colors,
            xmin=0.7, xmax=12.0, ymin=1e-3, log_scale=False,
        )

    if detector != "DYB":
        ensp['rvis_nonl'].Plot(f"{args.plots_folder}/stages_{detector}.png", \
        extra_spectra = [ensp['rvis_nonl_leak'],ensp['rvis']], colors=['black', 'blue', 'orange'],\
        leg_labels = ['rvis_nonl', 'rvis_nonl_leak', 'rvis'], xmin=0.7, xmax=2.0)
    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    _print_diagnostics(ensp, detector, args)

    if detector == "TAO":
        return ensp, resp_matrix, ene_leak_nd
    return ensp, resp_matrix


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostics(ensp, detector, args):
    print("\n--- Spectral integrals ---")
    for key in ("rfis0", "rfis", "ribd", "rosc", "rvis_nonl", "rvis", "rdet",
                "snf_final", "noneq_final", "rtot"):
        if key in ensp:
            print(f"  {key:<20s}: {ensp[key].GetIntegral():.2f}")
    if args.is_data:
        print(f"data: {ensp['data'].GetIntegral():.2f}")
    print(f"  2–6 MeV (ribd): {ensp['ribd'].GetIntegral(left_edge=2, right_edge=6):.2f}")
