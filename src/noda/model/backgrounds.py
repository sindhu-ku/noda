"""Generic handling of the JUNO background list from the backgrounds yaml.

Every background is described only by its yaml entry; no code path requires a
particular set of keys. Two pieces of per-background behaviour used to be
hard-wired on the key name and are now optional yaml settings:

``efficiency``: list, subset of
    * ``detector`` - scale the prefit spectrum by ``detector_efficiency``
    * ``eff_nuis`` - the fit's detection-efficiency nuisance multiplies it
    * ``oec``      - OMILREC threshold correction (only if apply_oec_correction)
  Default (reactor analysis, unchanged behaviour):
    geo, wr -> [detector, eff_nuis, oec];  atm -> [detector];  others -> [].

Output ROOT histograms are always named by the yaml key; ``name`` is the
  display name (legend). ``register()`` remembers the
  name -> key mapping so post-fit code (which carries display names) can find
  the key without a hard-coded table.

The physics models of ``geo`` (U/Th, ThU_ratio nuisance) and ``wr`` (positron /
NL / response chain) remain tied to those keys.
"""

_EFF_OPTIONS = ("detector", "eff_nuis", "oec")

_DEFAULT_EFF = {
    "geo": ("detector", "eff_nuis", "oec"),
    "wr":  ("detector", "eff_nuis", "oec"),
    "atm": ("detector",),
}

# Display names of the standard backgrounds (fallback when a yaml has not been
# registered, e.g. scripts that call the plotting helpers directly).
_LEGACY_NAME_TO_KEY = {
    "Accidentals": "acc", "Geoneutrinos": "geo",
    r"$^{9}$Li / $^{8}$He": "lihe", "Fast neutrons": "fastn",
    "Double neutrons": "doublen",
    r"$^{13}$C ($\alpha$, n) $^{16}$O": "alphan",
    "Atmospheric neutrinos": "atm", "World reactor neutrinos": "wr",
    r"$^{214}$Bi": "bipo",
}

_NAME_TO_KEY = {}


def efficiency_flags(key, info):
    """Set of efficiency options for one background (see module doc)."""
    eff = info.get("efficiency") if info else None
    if eff is None:
        return frozenset(_DEFAULT_EFF.get(key, ()))
    if isinstance(eff, str):
        eff = [eff]
    bad = [e for e in eff if e not in _EFF_OPTIONS]
    if bad:
        raise ValueError(f"background '{key}': unknown efficiency option(s) {bad}; "
                         f"allowed: {list(_EFF_OPTIONS)}")
    return frozenset(eff)


def display_name(key, info):
    return (info or {}).get("name", key)


def register(backgrounds, detector="JUNO"):
    """Record display-name -> key for every background of *detector*."""
    for key, info in (backgrounds or {}).get(detector, {}).items():
        _NAME_TO_KEY[display_name(key, info)] = key
    return backgrounds


def name_to_key(name):
    """Short key for a display name; None if unknown."""
    return _NAME_TO_KEY.get(name, _LEGACY_NAME_TO_KEY.get(name))


# ===========================================================================
# Background spectra (JUNO prefit build), moved from spectra.py
# ===========================================================================
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
from .detection import _apply_nonlinearity, _apply_oec_correction, _apply_positron_energy

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
        eff_flags = efficiency_flags(key, info)
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
