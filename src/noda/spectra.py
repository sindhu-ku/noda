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
from . import backgrounds as bkgdef


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
# OMILREC / 0.6 MeV threshold correction  (Check 5-2, DocDB-17104 p10)
# ---------------------------------------------------------------------------
# The reactor "eps_eff_reactor" curve models the low-energy IBD efficiency loss
# caused by the OMILREC 0.6 MeV threshold. It is applied to *all IBD* components
# (reactor IBDs, geo-neutrinos, world reactors) at the reconstructed-energy
# level, so those components end up with fewer expected events in the low-energy
# bins — mimicking the data. Backgrounds (accidentals, etc.) are NOT corrected.

_OEC_CACHE = {}

# IBD component -> preferred correction column (new per-component CSV format).
# Only IBD components are ever corrected (Check 5-2); backgrounds never call
# _apply_oec_correction, so no bg columns are mapped here.
_OEC_COL = {
    "reactor": "reactor_efficiency_correction_factor",
    "geo":     "geoneutrino_efficiency_correction_factor",
    "wr":      "wr_efficiency_correction_factor",
}


def _load_oec_correction(path):
    """Load the OMILREC efficiency table; cache by path.

    Returns (e_low, e_high, factors) where `factors` maps each IBD component
    ('reactor', 'geo', 'wr') to its per-bin correction array. Two CSV layouts
    are accepted, keyed off the header row:
      * per-component: E_low,E_high,E_center,<comp>_efficiency_correction_factor…
        each IBD component uses its OWN column;
      * legacy single-column: E_low,E_high,E_center,eps_eff_reactor,
        eps_eff_accidental — the reactor column is used for every IBD component.
    """
    if path not in _OEC_CACHE:
        with open(path) as fh:
            header = [h.strip() for h in fh.readline().strip().split(",")]
        # genfromtxt (not loadtxt) so empty cells parse: some component columns
        # are blank where that component has no events (e.g. geo above its
        # endpoint). Blank -> NaN -> 1.0 (no correction; harmless as those bins
        # carry no events anyway).
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        data = np.nan_to_num(data, nan=1.0)
        hmap = {h: i for i, h in enumerate(header)}
        factors = {}
        for comp, col in _OEC_COL.items():
            if col in hmap:                       # per-component file
                factors[comp] = data[:, hmap[col]]
            elif "eps_eff_reactor" in hmap:       # legacy single-column file
                factors[comp] = data[:, hmap["eps_eff_reactor"]]
            else:                                 # positional fallback (4th col)
                factors[comp] = data[:, 3]
        _OEC_CACHE[path] = (data[:, 0], data[:, 1], factors)
    return _OEC_CACHE[path]


def _oec_eff_at(energies, e_low, e_high, eff):
    """Reactor efficiency evaluated at each energy (MeV).

    Below the first bin -> first value (no IBD signal below the fit threshold
    anyway); above the last bin -> 1.0 (no correction).
    """
    energies = np.asarray(energies, dtype=float)
    out = np.ones_like(energies)
    for lo, hi, ef in zip(e_low, e_high, eff):
        out[(energies >= lo) & (energies < hi)] = ef
    out[energies < e_low[0]] = eff[0]
    out[energies >= e_high[-1]] = 1.0
    return out


def _apply_oec_correction(spec, args, label="", component=None):
    """Multiply a reconstructed-energy IBD spectrum in place by the OMILREC
    per-component efficiency. No-op unless ``apply_oec_correction`` is set.
    Applied ONLY to IBD components (reactor / geo / world-reactors), each with
    its own correction column; backgrounds are never passed here.

    `component` ('reactor'|'geo'|'wr') selects the column; if omitted it is
    inferred from `label` (geo*, reactor*, wr*). Must be applied AFTER rate
    normalisation and BEFORE the non-uniform rebin. Returns *spec*."""
    if not getattr(args, "apply_oec_correction", False):
        return spec
    if component is None:
        lab = (label or "").lower()
        component = ("geo"     if lab.startswith("geo")
                     else "wr" if lab.startswith("wr")
                     else "reactor")
    e_low, e_high, factors = _load_oec_correction(args.oec_correction_file)
    eff = factors.get(component, factors["reactor"])
    fac = _oec_eff_at(spec.GetBinCenters(), e_low, e_high, eff)
    before = spec.GetIntegral()
    spec.bin_cont = np.asarray(spec.bin_cont, dtype=float) * fac
    if label:
        print(f"  OEC correction [{label}] (col={component}): {before:.2f} -> "
              f"{spec.GetIntegral():.2f} events")
    return spec


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
            pebins=ebins
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

    # For TAO with NL baked into the energy-leakage matrix there is no separate
    # NL curve (the positronNL/positronNLpull histograms are gone from the
    # new-style inputs), so nothing to load.
    if detector == "TAO" and getattr(args, "nl_in_leak", False):
        return

    if detector == "TAO":
        np=5
    else:
        np=4
    ensp["scintNL"] = GetSpectrumFromROOT(args.NL_data_file, args.nl_hist_name)
    ensp["NL_pull"] = [
        GetSpectrumFromROOT(args.NL_data_file, f"{args.nl_pull_hist_name.replace('key', str(k))}")
        for k in range(np)
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
        # OMILREC 0.6 MeV threshold correction: geo-nu are IBDs (Check 5-2)
        _apply_oec_correction(reco, args, label=f"geo-{name}")
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

        # Per-background efficiency options from the yaml (`efficiency:`);
        # defaults reproduce the reactor analysis (see noda/backgrounds.py).
        eff_flags = bkgdef.efficiency_flags(key, info)
        if "detector" in eff_flags:
            ensp[key].GetScaled(args.detector_efficiency)
        # OMILREC 0.6 MeV threshold correction (default: wr + geo, the IBD
        # backgrounds; Check 5-2). Applied after rate normalisation, before
        # rebinning. The analytic geo path rebuilds/replaces "geo" in
        # _build_geo_spectra where the same correction is re-applied.
        if "oec" in eff_flags:
            _apply_oec_correction(ensp[key], args, label=key,
                                  component=key if key in ("geo", "wr") else "reactor")
        if args.rebin_nonuniform:
            ensp[key].Rebin_nonuniform(args.bins_nonuniform)

        eff_print = args.detector_efficiency if "detector" in eff_flags else 1.0
        print(f"  {key}: {ensp[key].GetIntegral():.2f}  (rate={info['rate']*eff_print})")
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
    extrafactors = compute_extrafactors(
        args, alpha=[args.U235_scale, args.U238_scale,
                     args.Pu239_scale, args.Pu241_scale])
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
        print("DYB data scaling 1e-43: ", extrafactors*ndays)
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
    ensp["bckg"].GetScaled(ndays*args.bckg_rate/ensp['bckg'].GetIntegral())
    # reactor-on/off DAQ livetime ratio (Group A convention); 1.0 if absent
    ensp["bckg"].GetScaled(getattr(args, "bckg_scale", 1.0))
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

def _assemble_total_spectrum_juno(ensp, args, backgrounds):
    """Event summary + (spectra, labels, colors) for the JUNO prefit plot.
    Loops over the backgrounds listed in the yaml (any keys)."""
    bkgs = backgrounds["JUNO"]
    events = {
        "rtot":   ensp["rtot"].GetIntegral(),
        "rdet":   ensp["rdet"].GetIntegral(),
        "rbckg":  ensp["bckg"].GetIntegral() if ensp.get("bckg") is not None else 0.0,
        "ribd":   ensp["ribd"].GetIntegral(),
        "rosc":   ensp["rosc"].GetIntegral(),
        "bkgs":   [(key, ensp[key].GetIntegral()) for key in bkgs],
    }
    _print_juno_event_summary(events)

    extra = [ensp[k] for k in bkgs]
    labels = ["Reactor antineutrinos"] + [bkgdef.display_name(k, i) for k, i in bkgs.items()]
    palette = ["darkred", "green", "navy", "orange", "magenta", "lightblue",
               "yellow", "lightgreen", "purple", "grey", "teal", "olive",
               "pink", "brown", "cyan"]
    colors = ["black"] + [palette[n % len(palette)] for n in range(len(bkgs))]

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
    for key, n in ev["bkgs"]:
        print(f"    {key + ':':<18} {n:>10.2f}")
    print("=" * 46 + "\n")


# ---------------------------------------------------------------------------
# Optional write-out
# ---------------------------------------------------------------------------

def _build_nl_graph(ensp, args, kB=None, fC=None, kI=None, LY_shared=None,
                    NL_pulls=None):
    """Positron NL curve (Evis/Edep vs Edep), exported as
    <det>/response/<stage>/nl_positron: Geant4 model anchored to the
    2.224589 MeV gamma.  Parameters default to the pre-fit nominals.

    In pull mode, *NL_pulls* = [p0, …, p_{k-1}, LY_pull] gives the post-fit
    curve: nominal + Σ pᵢ·(pullᵢ − nominal), scaled by (1 + LY_pull) — the
    same construction as build_spectrum().  The number of shape pulls k is
    inferred from len(NL_pulls)−1, so this handles both JUNO (4 shape + LY)
    and TAO (5 shape + LY); the last element is always the LY pull."""
    if args.NL == "pull":
        e = 0.5 * (ensp["scintNL"].bins[:-1] + ensp["scintNL"].bins[1:])
        nom = ensp["scintNL"].bin_cont
        if NL_pulls is not None:
            n_shape = len(NL_pulls) - 1          # last element is the LY pull
            ratio = (nom + sum(NL_pulls[i] * (ensp["NL_pull"][i].bin_cont - nom)
                               for i in range(n_shape))) * (1 + NL_pulls[-1])
        else:
            ratio = nom
    else:
        kB = args.kB if kB is None else kB
        fC = args.fC if fC is None else fC
        kI = args.kI if kI is None else kI
        LY_shared = args.LY_shared if LY_shared is None else LY_shared
        # 1.022 MeV = minimum deposited energy of an IBD positron (physics, not a range choice)
        e = np.arange(1.022, analysis_range(args)[1], 0.001)
        ratio = (G4model.eval(e, kB, fC=fC, kI=kI) * LY_shared
                 / G4model_gamma.eval(2.224589, kB, fC=fC, kI=kI))
    graph = ROOT.TGraph(len(e),
                        np.ascontiguousarray(e, dtype=float),
                        np.ascontiguousarray(ratio, dtype=float))
    graph.SetName("nl_positron")
    return graph


def analysis_range(args):
    """(E_lo, E_hi) of the analysis in MeV, from the yaml: first/last edge of
    bins_nonuniform when rebin_nonuniform is set, else min_ene/max_ene."""
    bn = getattr(args, 'bins_nonuniform', None)
    if getattr(args, 'rebin_nonuniform', False) and bn:
        return float(bn[0][0]), float(bn[-1][1])
    return float(args.min_ene), float(args.max_ene)


def _build_resolution_graph(a, b, c, emin=0.8, emax=12.0):
    """Relative energy resolution σ_E/E vs Evis, same parametrisation as
    CalcRespMatrix_abc: sqrt(a²/E + b² + c²/E²)."""
    e = np.arange(max(emin, 0.1), emax, 0.001)   # sigma/E diverges at E -> 0
    res = np.sqrt(a**2 / e + b**2 + c**2 / e**2)
    graph = ROOT.TGraph(len(e),
                        np.ascontiguousarray(e, dtype=float),
                        np.ascontiguousarray(res, dtype=float))
    graph.SetName("resolution")
    return graph


# Map NODA shell labels to the top-level JUNO directory name in the output
# ROOT file. Output names and layout follow the ORSA export so the two
# fitters' files compare directly. Extend this when new shells are added.
_SHELL_DIR = {'S-15': 'juno_inner', 'S-15-172': 'juno_outer'}


def juno_output_dirname(args):
    """Top-level JUNO ROOT directory for this dataset.

    A single-shell run keeps the legacy bare ``juno``.  A multi-shell run maps
    each shell to its own self-contained top-level folder
    (``juno_inner`` / ``juno_outer``), falling back to
    ``juno_<shell>`` for shell labels not in the map.  The multi-shell signal
    is the per-dataset ``root_suffix`` carrying a shell component (``__s``)."""
    sfx = getattr(args, 'root_suffix', '')
    if '__s' not in sfx:
        return 'juno'
    shell = str(getattr(args, 'shell', ''))
    return _SHELL_DIR.get(shell, f"juno_{shell}")


def _write_juno_output(ensp, ndays, args, backgrounds, s_ibd=None):
    from .fit.minuit import write_param_dbl
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    jdir = juno_output_dirname(args)            # 'juno' | 'juno_inner' | 'juno_outer'
    output_file = ROOT.TFile(tag, "update")

    # Each shell writes its OWN top-level juno_<shell> directory
    # with bare object names. get-or-create: in "update" mode the dir may
    # already exist from a previous write, and TDirectory.mkdir() returns null
    # for an existing directory (which would crash on .cd()).
    juno_dir = output_file.GetDirectory(jdir) or output_file.mkdir(jdir)
    juno_dir.cd()
    write_param_dbl("LivetimeDays", ndays)

    dir_pre = juno_dir.GetDirectory("prefit") or juno_dir.mkdir("prefit")
    dir_pre.cd()
    write_param_dbl("Efficiency", args.detector_efficiency)
    if s_ibd is not None:
        enu = np.arange(0.0, max(12.0, float(args.max_ene)) + 0.001, 0.001)
        xsec_vals = s_ibd(enu)
        xsec_graph = ROOT.TGraph(len(enu),
                                 np.ascontiguousarray(enu, dtype=float),
                                 np.ascontiguousarray(xsec_vals, dtype=float))
        xsec_graph.Write("xsec", ROOT.TObject.kOverwrite)

    dir_resp = juno_dir.GetDirectory("response") or juno_dir.mkdir("response")
    dir_resp.cd()
    dir_resp_prefit = dir_resp.GetDirectory("prefit") or dir_resp.mkdir('prefit')
    dir_resp_prefit.cd()
    g_nl = _build_nl_graph(ensp, args);  g_nl.SetName("nl_positron")
    g_nl.Write("", ROOT.TObject.kOverwrite)
    g_res = _build_resolution_graph(args.a, args.b, args.c, *analysis_range(args))
    g_res.SetName("resolution");   g_res.Write("", ROOT.TObject.kOverwrite)
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["ribd"].WritetoROOT("unosc", tag, f"{jdir}/prefit")
    ensp["rosc"].WritetoROOT("osc", tag, f"{jdir}/prefit")
    if "rosc_wcorr" in ensp:
        ensp["rosc_wcorr"].WritetoROOT("osc_wcorr", tag, f"{jdir}/prefit")
    ensp["rvis_nonl_temp"].WritetoROOT("unosc_dep", tag, f"{jdir}/prefit")
    ensp["rvis_nonl"].WritetoROOT("dep", tag, f"{jdir}/prefit")
    ensp["rvis"].WritetoROOT("vis", tag, f"{jdir}/prefit")
    ensp["rdet_fine"].WritetoROOT("rec", tag, f"{jdir}/prefit")
    ensp["rdet"].WritetoROOT("reactor", tag, f"{jdir}/prefit")
    for core, steps in ensp.get("per_core_steps", {}).items():
        for step, spec in steps.items():
            spec.WritetoROOT(step, tag, f"{jdir}/prefit/per_core/{core}")
    ensp["rtot"].WritetoROOT("totalpred", tag, f"{jdir}/prefit")
    for key in backgrounds["JUNO"]:
        ensp[key].WritetoROOT(f"{key}", tag, f"{jdir}/prefit")
    if args.is_data: ensp["data"].WritetoROOT("data", tag, jdir)



def _write_params(tag, directory, params):
    """Write TParameter<double>s into <directory> of the output ROOT file."""
    from .fit.minuit import write_param_dbl
    f = ROOT.TFile(tag, "update")
    d = f.GetDirectory(directory) or f.mkdir(directory)
    d.cd()
    for k, v in params.items():
        write_param_dbl(k, float(v))
    force_vector_streamers(f)
    f.Close()


def _write_dyb_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    if not output_file.GetDirectory("dyb"):
        output_file.mkdir("dyb")
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["rdet"].WritetoROOT("totalpred", tag, "dyb/prefit")
    ensp["rdet"].WritetoROOT("reactor", tag, "dyb/prefit")   # reactor only (no bkg)
    _write_params(tag, "dyb/prefit", {"Efficiency": 1.0})
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "dyb")


def _write_tao_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    tao_dir = output_file.GetDirectory("tao") or output_file.mkdir("tao")
    dir_resp = tao_dir.GetDirectory("response") or tao_dir.mkdir("response")
    dir_resp_prefit = dir_resp.GetDirectory("prefit") or dir_resp.mkdir("prefit")
    dir_resp_prefit.cd()
    # NL baked into the leakage matrix (nl_in_leak): no separate NL curve exists,
    # so skip the pre-fit NL graph and write only the resolution curve.
    if not getattr(args, "nl_in_leak", False):
        _build_nl_graph(ensp, args).Write("", ROOT.TObject.kOverwrite)
    _build_resolution_graph(args.a, args.b, args.c,
                            *analysis_range(args)).Write("", ROOT.TObject.kOverwrite)
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["rtot"].WritetoROOT("totalpred", tag, "tao/prefit")
    ensp["bckg"].WritetoROOT("bkg", tag, "tao/prefit")       # (was "bckg")
    ensp["rdet"].WritetoROOT("reactor", tag, "tao/prefit")
    _write_params(tag, "tao/prefit", {"Efficiency": args.detector_efficiency})
    _write_params(tag, "tao", {"LivetimeDays": float(str(args.stat_opt).replace("days", ""))})
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
            dm2_31
        )
        _build_snf_noneq_corrections_per_core(
            ensp, args,ndays, ebins, core_fractions, core_baselines, core_powers, resp_matrix,s_ibd, dm2_31,
            core_names=core_names,
        )
        if args.write_output: _write_groupC_csvs(ensp, args, core_names, core_powers, core_fractions)
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
        if getattr(args, "osc_per_core", False):
            ensp["rosc"] = ensp["rosc_pcore"]
        else:
            ensp["rosc"] = ensp["ribd"].GetOscillated(
                L=list(core_baselines), core_powers=list(core_powers),
                me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
            )
    else:
        ensp["rosc"] = ensp["ribd"]

    # Oscillated spectrum with SNF + Non-Eq corrections added (true-energy level).
    # snf_osc / noneq_osc are only built in the per-core (Group C) path.
    if "snf_osc" in ensp and "noneq_osc" in ensp:
        ensp["rosc_wcorr"] = ensp["rosc"] + ensp["snf_osc"] + ensp["noneq_osc"]

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

        tao_nl_in_leak = (detector == "TAO" and getattr(args, "nl_in_leak", False))
        if detector != "DYB" and not tao_nl_in_leak:
            ensp["rvis"]      = _apply_nonlinearity(ensp["rvis_nonl_leak"],      args, ensp)
            ensp["rvis_temp"] = _apply_nonlinearity(ensp["rvis_nonl_temp_leak"], args, ensp)
            #ensp["rvis_temp_wcorr"] = _apply_nonlinearity(ensp["rvis_nonl_temp_wcorr"], args, ensp)
        elif tao_nl_in_leak:
            # NL already carried by the energy-leakage matrix — no separate curve.
            ensp["rvis"]      = ensp["rvis_nonl_leak"]
            ensp["rvis_temp"] = ensp["rvis_nonl_temp_leak"]
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
    # OMILREC 0.6 MeV threshold correction on reactor IBDs (Check 5-2).
    # Applied on the fine reco spectrum, before rebinning.
    if detector == "JUNO":
        _apply_oec_correction(ensp["rdet"], args, label="reactor")
    ensp['rdet_fine'] = ensp['rdet']
    if args.rebin_nonuniform:
        ensp["rdet"].Rebin_nonuniform(args.bins_nonuniform)

    # Per-core step-by-step reactor spectra (JUNO, per-core reactor path) for
    # bin-by-bin comparisons. Written by _write_juno_output under
    # <shell>/prefit/per_core/<core>/. Toggle with fit_options write_per_core.
    if (detector == "JUNO" and "snf_final_core0" in ensp
            and getattr(args, "write_output", False)
            and bool(getattr(args, "write_per_core", True))):
        ensp["per_core_steps"] = _build_per_core_steps(
            ensp, args, resp_matrix, s_ibd, core_names, core_baselines,
            core_powers, dm2_31)


    # ------------------------------------------------------------------
    # 7. Backgrounds + total spectrum
    # ------------------------------------------------------------------
    print("# Assembling total spectrum …")

    if detector == "JUNO":
        ensp["bckg"] = _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays, s_ibd=s_ibd)
        if ensp["bckg"] is None:      # empty background list in the yaml
            ensp["bckg"] = Spectrum(bins=ensp["rdet"].bins,
                                    bin_cont=np.zeros(len(ensp["rdet"].bin_cont)))
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
            elif "geo" in backgrounds["JUNO"]:
                ensp["geo_ana"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "geo",
                    float(args.geo_tnu_rate) * float(args.tnu_to_cpd), args.Th_U_ratio,
                )
                ensp["bckg"] = ensp["bckg"] - ensp["geo"] + ensp["geo_ana"]
                ensp["geo"]  = ensp["geo_ana"]
                print(f"  geo: {ensp['geo'].GetIntegral():.2f}")
        ensp["rtot"] = ensp["rdet"] + ensp["bckg"]
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_juno(ensp, args, backgrounds)

        if args.write_output:
            _write_juno_output(ensp, ndays, args, backgrounds, s_ibd=s_ibd)

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
            xmin=analysis_range(args)[0], xmax=analysis_range(args)[1], ymin=1e-3, log_scale=False,
        )

    if detector != "DYB":
        ensp['rvis_nonl'].Plot(f"{args.plots_folder}/stages_{detector}.png", \
        extra_spectra = [ensp['rvis_nonl_leak'],ensp['rvis']], colors=['black', 'blue', 'orange'],\
        leg_labels = ['rvis_nonl', 'rvis_nonl_leak', 'rvis'], xmin=analysis_range(args)[0], xmax=2.0)  # low-E zoom
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
