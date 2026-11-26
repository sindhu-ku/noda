"""IBD detection chain shared by every IBD component (reactor, geo, world
reactors, future true-level signals): positron energy, non-linearity,
detector response / leakage, OMILREC threshold efficiency (OEC);
plus small spectrum helpers used across model/.

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
