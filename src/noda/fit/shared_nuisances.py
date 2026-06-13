#!/usr/bin/env python3
"""
shared_nuisances.py

Parameters shared across ALL detectors:
  - Oscillation parameters : sin2_th12, sin2_th13, dm2_21, dm2_31
  - me_rho                 : matter density (affects oscillation in every detector)
  - Shape nuisances        : bin-by-bin segment weights (PMOP_combined only)

These sit at the front of the Minuit parameter vector, before any
detector-specific nuisances.

Layout inside the shared block:
  index 0 : sin2_th12
  index 1 : sin2_th13
  index 2 : dm2_21
  index 3 : dm2_31
  index 4 : me_rho
"""

import numpy as np
from .. import nuosc as nuosc

# ── names ────────────────────────────────────────────────────────────────────
OSC_PARAM_NAMES    = ["sin2_th12", "sin2_th13", "dm2_21", "dm2_31"]
SHARED_PARAM_NAMES = OSC_PARAM_NAMES + ["me_rho"]   # 5 params total before any detector block
N_SHARED           = len(SHARED_PARAM_NAMES)        # 5


def get_shared_start_values(args_juno, dm2_31 = None):
    """Nominal start values for the shared parameter block (same order as SHARED_PARAM_NAMES)."""
    return [
        nuosc.op_nom["sin2_th12"],
        nuosc.op_nom["sin2_th13"],
        nuosc.op_nom["dm2_21"] * 1e5,   # internally stored as ×1e-5 eV²
        dm2_31*1e3 if dm2_31 is not None else nuosc.op_nom["dm2_31"] * 1e3,   # internally stored as ×1e-3 eV²
        args_juno.me_rho,
    ]


def get_shape_nuis_names(n_bins):
    return [f"nuis{i}" for i in range(n_bins)]


def get_shape_nuis_start_values(n_bins):
    return list(np.zeros(n_bins))


def apply_shared_limits(m, args_juno, n_shape_bins, nmo_opt=None):
    """Set Minuit limits on the shared parameters (me_rho + shape nuisances).

    `dm2_31` is confined to its ordering's sign domain so a fit cannot hop
    across dm2_31=0 into the other ordering's basin — which would flip both
    the oscillation and the LBL-pull branch (f_lbl) and corrupt the NMO
    significance. Bounds mirror profile_dm2_31 (Minuit units, ×1e-3 eV²).
    `nmo_opt` selects the ordering (defaults to args_juno.NMO_opt); pass
    `not args_juno.NMO_opt` for the opposite-ordering significance fit.
    """
    m.limits["me_rho"] = [0.0, None]
    if nmo_opt is None:
        nmo_opt = bool(args_juno.NMO_opt)
    m.limits["dm2_31"] = (1.5, 3.5) if nmo_opt else (-3.5, -1.5)
    for i in range(n_shape_bins):
        m.limits[f"nuis{i}"] = [-1, None]
