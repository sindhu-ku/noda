"""Reactor antineutrino prediction: core layout, Huber-Mueller flux,
flux x IBD cross-section, per-core oscillation/response chain,
SNF / non-equilibrium corrections, GroupC per-core exports.

Split out of model/spectra.py (pure move)."""
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

from ..noda import *
from .detection import _apply_full_energy_response, _apply_nonlinearity, _apply_oec_correction, _apply_positron_energy, _ensure_dir, _zero_spectrum, interp_to_target


# ---------------------------------------------------------------------------
# Constants / index ranges
# ---------------------------------------------------------------------------

# Core index layout for the per-core JUNO build. Order matches
# load_reactor_summary(): 6 near YJ cores, 2 near TS cores, then the distant
# combined plants (DYB, TPL, FC, …), all handled identically as the "HM" group.
# Generic in the number of distant plants, so adding one (e.g. TPL, 11 cores
# total) needs no change here.
N_YJ = 6
N_TS = 2
HM_START = N_YJ + N_TS            # every core at index >= HM_START is an HM plant

# kept for the (commented-out) legacy builder further down
IDX_YJ = range(0, N_YJ)
IDX_TS = range(N_YJ, HM_START)
IDX_HM = range(HM_START, HM_START + 3)


def _core_group(i):
    """Return (plant-label, 1-based index-within-group) for JUNO core index i.

    YJ: 0..5, TS: 6..7, HM (distant combined plants DYB/TPL/FC/…): >= HM_START.
    """
    if i < N_YJ:      return ("YJ", i + 1)
    if i < HM_START:  return ("TS", i - N_YJ + 1)
    return ("HM", i - HM_START + 1)


def _build_per_core_steps(ensp, args, resp_matrix, s_ibd, core_names,
                          core_baselines, core_powers, dm2_31):
    """Per-core reactor spectra at every step of the JUNO chain, mirroring the
    total-spectrum names written in <shell>/prefit:

      unosc      IBD-weighted flux (true E), per-core rfis_core{i}
      osc        oscillated at the core's own baseline
      osc_wcorr  osc + oscillated SNF + non-eq (true E)
      unosc_dep  unosc after positron energy shift (deposited)
      dep        osc after positron energy shift
      vis        dep after non-linearity
      snf, noneq SNF / non-eq corrections after the full response (reco)
      rec        vis ⊗ response + snf + noneq (+ OEC), fine reco binning
      reactor    rec rebinned to the analysis binning
    Sum over cores of `reactor` reproduces the total prefit `reactor`.
    Returns {core_name: {step: Spectrum}}."""
    out = {}
    for i, name in enumerate(core_names):
        unosc = ensp[f"rfis_core{i}"]
        osc = unosc.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", args=args, dm2_31=dm2_31)
        snf_f, neq_f = ensp[f"snf_final_core{i}"], ensp[f"noneq_final_core{i}"]
        dep = _apply_positron_energy(osc, args, s_ibd=s_ibd)
        vis = _apply_nonlinearity(dep, args, ensp)
        rec = vis.ApplyDetResp(resp_matrix, pecrop=args.ene_crop) + snf_f + neq_f
        _apply_oec_correction(rec, args, label="reactor")
        reactor = rec.Copy()
        if args.rebin_nonuniform:
            reactor.Rebin_nonuniform(args.bins_nonuniform)
        out[name] = {
            "unosc": unosc, "osc": osc,
            "osc_wcorr": osc + ensp[f"snf_osc_core{i}"] + ensp[f"noneq_osc_core{i}"],
            "unosc_dep": _apply_positron_energy(unosc, args, s_ibd=s_ibd),
            "dep": dep, "vis": vis, "snf": snf_f, "noneq": neq_f,
            "rec": rec, "reactor": reactor,
        }
    tot = sum(v["reactor"].GetIntegral() for v in out.values())
    print(f"[per_core] {len(out)} cores; sum of per-core reactor = {tot:.3f} "
          f"(total prefit reactor = {ensp['rdet'].GetIntegral():.3f})")
    return out


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
# SNF / Non-equilibrium corrections  (JUNO per-core path)
# ---------------------------------------------------------------------------

def _build_snf_noneq_corrections_per_core(ensp, args, ndays, ebins, core_fractions, core_baselines, core_powers, resp_matrix, s_ibd, dm2_31, core_names=None):
    """
    Build oscillated, smeared SNF and Non-Eq spectra for each of the 10 JUNO
    cores and accumulate group totals.  Modifies *ensp* in-place.
    """
    n_cores = len(core_powers)

    # --- global HM SNF / Noneq base flux-ratio shapes ---
    # The distant combined plants (DYB, TPL, FC, …) all share these shapes;
    # each is weighted by its own per-core fission flux in the loop below.
    for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
        ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
        ensp[key].Rebin(ebins, mode="spline-not-keep-norm")

    # --- per-core SNF / Non-Eq ---
    #   YJ + TS (cores 0-7): dedicated per-core histograms.
    #   HM (cores >= HM_START: DYB / TPL / FC / …): the global flux ratio
    #     weighted by each plant's own fission flux. Generic in HM-plant count.
    snf_core = []
    noneq_core = []

    name_i = lambda i: core_names[i] if core_names is not None else None

    for i in range(n_cores):
        prefix, core_idx = _core_group(i)
        rfis_i = ensp[f"rfis_core{i}"]

        if prefix == "HM":
            # TLP (TPL) carries no SNF contribution — zero it out while keeping
            # the binning consistent with the other per-core spectra.
            if name_i(i) == "TPL":
                snf_core.append(_zero_spectrum(rfis_i.bins))
            else:
                snf_core.append(
                    interp_to_target(ensp["snf0"], rfis_i).GetWeightedWithSpectrum(rfis_i))
            noneq_core.append(
                interp_to_target(ensp["noneq0"], rfis_i).GetWeightedWithSpectrum(rfis_i))
            continue

        # SNF (YJ/TS dedicated histogram)
        spec = GetSpectrumFromROOT(args.snf_file, f"Ratio{prefix}ReactorSNF{core_idx}")
        spec.Rebin(ebins, mode="spline-not-keep-norm")
        snf_core.append(interp_to_target(spec, rfis_i).GetWeightedWithSpectrum(rfis_i))

        # Non-equilibrium (sum over isotopes)
        group_noneq = None
        for iso, frac in core_fractions[i].items():
            if iso == "U238":
                continue
            spec = GetSpectrumFromROOT(args.noneq_file, f"{prefix}{core_idx}_{iso}")
            spec.Rebin(ebins, mode="spline-not-keep-norm")
            weighted = interp_to_target(spec.GetScaledFit(frac*0.01), rfis_i).GetWeightedWithSpectrum(rfis_i)
            group_noneq = weighted.Copy() if group_noneq is None else group_noneq + weighted
        noneq_core.append(group_noneq)

    del ensp["snf0"], ensp["noneq0"]

    # --- YJ shutdown contributions ---
    def _load_shutdown(hist, core_idx):
        spec = GetSpectrumFromROOT(args.snf_file, hist)
        spec.Rebin(ebins, mode="spline-not-keep-norm")
        return interp_to_target(spec, ensp[f"rfis_core{core_idx}"]).GetWeightedWithSpectrum(ensp[f"rfis_core{core_idx}"])

    shutdown_core = [None] * n_cores
    shutdown_core[0] = _load_shutdown("RatioYJReactor1Shutdown", 0)
    shutdown_core[3] = _load_shutdown("RatioYJReactor4Shutdown", 3)
    snf_core[0] += shutdown_core[0]
    snf_core[3] += shutdown_core[3]

    # --- Accumulate group totals ---
    for grp in ("snf", "noneq"):
        for lbl in ("YJ", "TS", "HM", "final", "osc"):
            ensp[f"{grp}_{lbl}"] = _zero_spectrum(ebins)

    core_records = []   # per-core event-rate summary (slide-7 format), Group C

    for i in range(n_cores):
        s_i = snf_core[i]
        n_i = noneq_core[i]
        rfis_i = ensp[f"rfis_core{i}"]

        s_osc = s_i.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
        )
        n_osc = n_i.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
        )
        ensp["snf_osc"] +=  s_osc
        ensp["noneq_osc"] += n_osc

        final_snf_i   = _apply_full_energy_response(s_osc, args, ensp, resp_matrix, s_ibd)
        final_noneq_i = _apply_full_energy_response(n_osc, args, ensp, resp_matrix, s_ibd)

        ensp["snf_final"]   += final_snf_i
        ensp["noneq_final"] += final_noneq_i

        # keep per-core pieces for the per-core step-by-step prefit export
        ensp[f"snf_osc_core{i}"]     = s_osc
        ensp[f"noneq_osc_core{i}"]   = n_osc
        ensp[f"snf_final_core{i}"]   = final_snf_i
        ensp[f"noneq_final_core{i}"] = final_noneq_i

        group_label = _core_group(i)[0]   # "YJ", "TS", or "HM"
        ensp[f"snf_{group_label}"]   += final_snf_i
        ensp[f"noneq_{group_label}"] += final_noneq_i

        # --- per-core event rates (events/day) for the Group C summary CSV ---
        rosc_main_i = rfis_i.GetOscillated(
            L=[core_baselines[i]], core_powers=[core_powers[i]],
            me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
        )
        reco_main_i = _apply_full_energy_response(rosc_main_i, args, ensp, resp_matrix, s_ibd)

        shut_rate = shutdown_core[i].GetIntegral() / ndays if shutdown_core[i] is not None else 0.0
        snf_rate  = s_i.GetIntegral() / ndays - shut_rate   # s_i already includes shutdown
        base_hm   = rfis_i.GetIntegral() / ndays
        noneq_rate = n_i.GetIntegral() / ndays
        core_records.append({
            "core":       core_names[i] if core_names is not None else f"core{i}",
            "unosc":      base_hm + noneq_rate + snf_rate + shut_rate,
            "oscillated": rosc_main_i.GetIntegral() / ndays,
            "reco":       reco_main_i.GetIntegral() / ndays,
            "base_hm":    base_hm,
            "noneq":      noneq_rate,
            "snf":        snf_rate,
            "shutdown":   shut_rate,
        })

        print(f"{group_label}{i}  snf={s_i.GetIntegral()/ndays:.2f}  noneq={n_i.GetIntegral()/ndays:.2f}")

    ensp["core_records"] = core_records

    print(f"Final SNF Integral:   {ensp['snf_final'].GetIntegral()/ndays:.2f}")
    print(f"Final Noneq Integral: {ensp['noneq_final'].GetIntegral()/ndays:.2f}")


def _write_groupC_csvs(ensp, args, core_names, core_powers, core_fractions):
    """
    Write the three Group-C summary tables as CSVs, matching the formats on
    slides 5-7 of the Reactor-Nu PWG group-comparison deck:

      * <out>_powers.csv     – average reactor power per core      (slide 5)
      * <out>_fractions.csv  – average fission fractions per core  (slide 6)
      * <out>_event_rates.csv – predicted event rates per core     (slide 7)

    All event rates are in events/day; a TOTAL row is appended.
    """
    out_dir = getattr(args, "plots_folder", ".")
    _ensure_dir(out_dir)
    # The file_prefix belongs in the *filename*, not glued in front of the
    # absolute out_dir path (which produced a non-existent relative directory
    # like 'PREFIX_/home/.../Plots/' and crashed open()).
    prefix = f"{args.file_prefix}_" if getattr(args, "file_prefix", "") else ""
    base = os.path.join(out_dir, f"{prefix}")

    names = list(core_names) if core_names is not None else [f"core{i}" for i in range(len(core_powers))]

    # --- slide 5: reactor powers ---
    with open(f"{base}_powers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Core", "Power_GW"])
        for name, power in zip(names, core_powers):
            w.writerow([name, f"{power:.3f}"])

    # --- slide 6: average fission fractions ---
    isotopes = ["U235", "U238", "Pu239", "Pu241"]
    with open(f"{base}_fractions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Core"] + isotopes)
        for name, frac in zip(names, core_fractions):
            w.writerow([name] + [f"{frac[iso]:.3f}" for iso in isotopes])

    # --- slide 7: predicted event rates ---
    records = ensp.get("core_records", [])
    cols = ["unosc", "oscillated", "reco", "base_hm", "noneq", "snf", "shutdown"]
    header = ["Core", "Unosc", "Oscillated", "Reco", "Base(HM)", "NonEq", "SNF", "Shutdown"]

    def _write_rates(path, scale):
        totals = {c: 0.0 for c in cols}
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for rec in records:
                w.writerow([rec["core"]] + [f"{rec[c] * scale:.4f}" for c in cols])
                for c in cols:
                    totals[c] += rec[c] * scale
            w.writerow(["TOTAL"] + [f"{totals[c]:.4f}" for c in cols])

    # as built (includes IBD detection efficiency)
    _write_rates(f"{base}_event_rates.csv", 1.0)

    # efficiency-corrected: divide out detector_efficiency -> 100% IBD efficiency
    eff = getattr(args, "detector_efficiency", 1.0) or 1.0
    _write_rates(f"{base}_event_rates_eff100.csv", 1.0 / eff)

    print(f"# Wrote Group C CSVs: {base}_powers.csv, {base}_fractions.csv, "
          f"{base}_event_rates.csv, {base}_event_rates_eff100.csv (eff={eff:.4f})")


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

    if detector == "TAO":
        # Suppression of the TAO non-eq correction (fit_options tao_noneq_scale,
        # code default 1.0). Standard: 0.15 = final recommendation, the
        # livetime-weighted suppression (26d*(0.96*0.03+0.03*0.9)+5d*0.75)/31d;
        # 0.75 applies only to the last 5 reactor-off days, when the fuel was
        # moved 43 m -> ~80 m to the SNF pool. Same value as ORSA.
        tao_noneq_scale = float(getattr(args, "tao_noneq_scale", 1.0))
        if tao_noneq_scale != 1.0:
            print(f"[TAO] scaling non-eq correction by {tao_noneq_scale}")
            ensp["noneq"] = ensp["noneq"].GetScaledFit(tao_noneq_scale)

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

        if detector == "TAO" and getattr(args, "nl_in_leak", False):
            # NL already carried by the energy-leakage matrix — no separate curve.
            ensp["snf_final"]   = ensp["snf_final_temp"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            ensp["noneq_final"] = ensp["noneq_final_temp"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        else:
            ensp["snf_final"] = _apply_nonlinearity(ensp["snf_final_temp"],   args, ensp).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            ensp["noneq_final"] =  _apply_nonlinearity(ensp["noneq_final_temp"],   args, ensp).ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

# def _build_snf_noneq_corrections_per_core(ensp, args,ndays, ebins, core_fractions,
#                                           core_baselines, core_powers,
#                                           resp_matrix, s_ibd, dm2_31):
#     """
#     Per-core SNF / Non-Eq corrections using the COMMON flux-ratio histograms
#     (SNF_FluxRatio / NonEq_FluxRatio from args.input_data_file) for every core.
#
#     Each core's correction is the global ratio weighted by that core's
#     IBD-folded flux (rfis_core{i}), oscillated with the core's own
#     baseline/power, then run through the full energy-response chain.
#     Group totals (YJ / TS / HM), the summed final spectra, and the
#     pre-response oscillated sums are accumulated in-place on *ensp*.
#
#     NOTE: core_fractions is accepted for call-site compatibility but no
#     longer used — all cores share the common ratio.
#     """
#     n_cores = len(core_powers)
#
#     # --- common SNF / Non-eq flux ratios (same histogram for all cores) ---
#     for key, hist in [("snf0", "SNF_FluxRatio"), ("noneq0", "NonEq_FluxRatio")]:
#         ensp[key] = GetSpectrumFromROOT(args.input_data_file, hist)
#         ensp[key].Rebin(ebins, mode="spline-not-keep-norm")
#
#     # --- per-core, weighted by each core's flux (already on ebins → no interp) ---
#     snf_core   = [ensp["snf0"].GetWeightedWithSpectrum(ensp[f"rfis_core{i}"])
#                   for i in range(n_cores)]
#     noneq_core = [ensp["noneq0"].GetWeightedWithSpectrum(ensp[f"rfis_core{i}"])
#                   for i in range(n_cores)]
#
#     del ensp["snf0"], ensp["noneq0"]
#
#     # --- zero the accumulators ---
#     for grp in ("snf", "noneq"):
#         for lbl in ("YJ", "TS", "HM", "final", "osc"):
#             ensp[f"{grp}_{lbl}"] = _zero_spectrum(ebins)
#
#     for i in range(n_cores):
#
#
#         s_osc = snf_core[i].GetOscillated(
#             L=[core_baselines[i]], core_powers=[core_powers[i]],
#             me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args)
#         n_osc = noneq_core[i].GetOscillated(
#             L=[core_baselines[i]], core_powers=[core_powers[i]],
#             me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args)
#
#         ensp["snf_osc"]   += s_osc
#         ensp["noneq_osc"] += n_osc
#
#         final_snf_i   = _apply_full_energy_response(s_osc, args, ensp, resp_matrix, s_ibd)
#         final_noneq_i = _apply_full_energy_response(n_osc, args, ensp, resp_matrix, s_ibd)
#
#         ensp["snf_final"]   += final_snf_i
#         ensp["noneq_final"] += final_noneq_i
#
#         grp = _CORE_GROUP[i][0]   # "YJ", "TS", or "HM"
#         ensp[f"snf_{grp}"]   += final_snf_i
#         ensp[f"noneq_{grp}"] += final_noneq_i
#
#         print(f"{grp}{i}  snf={snf_core[i].GetIntegral()/ndays:.2f}  "
#               f"noneq={noneq_core[i].GetIntegral()/ndays:.2f}")
#
#     print(f"Final SNF Integral:   {ensp['snf_final'].GetIntegral()/ndays:.2f}")
#     print(f"Final Noneq Integral: {ensp['noneq_final'].GetIntegral()/ndays:.2f}")
# ---------------------------------------------------------------------------
# Reactor flux builders
# ---------------------------------------------------------------------------

def _build_reactor_flux_per_core(ensp, args, ebins, s_ibd, s_bump_lin, ndays,
                                  core_names, core_powers, core_fractions, core_baselines, dm2_31):
    """Build per-core IBD-weighted reactor fluxes. Modifies *ensp* in-place."""
    rfis_sum  = None
    rfis0_sum = None
    rosc_sum = None
    osc_per_core = getattr(args, "osc_per_core", False)

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
        if osc_per_core:
            rosc_i = rfis_i.GetOscillated(
                L=[baseline], core_powers=[power],
                me_rho=args.me_rho, ene_mode="true", args=args,
                dm2_31=dm2_31
            )
            rosc_sum = rosc_i if rosc_sum is None else rosc_sum + rosc_i
    ensp["rfis0"] = rfis0_sum
    ensp["rfis"]  = rfis_sum
    if osc_per_core:
        ensp['rosc_pcore'] = rosc_sum


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

    # normalise with the same fission fractions used to build the flux,
    # not the common args.alpha (they differ for TAO/DYB)
    extrafactors = compute_extrafactors(args, alpha=list(fracs.values()))
    print(f"extrafactors: {extrafactors}")
    ensp["rfis"].GetScaled(extrafactors)
    print("DYB expectation scaling: ", extrafactors*ndays)
