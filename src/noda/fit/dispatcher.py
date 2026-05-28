#!/usr/bin/env python3
"""
dispatcher.py — multi-(phase, shell) glue for the existing run_minuit.

CHANGES vs first version:
  * pull_dedup_correction() — subtracts the (N-1) over-count when a pull
    is emitted by multiple bundles that share the SAME Minuit parameter
  * build_chi2() now wires the correction in automatically
  * ER (energy-response) 2D covariance penalty is also de-duplicated
    when all 7 of (a, b, c, kB, fC, kI, LY_shared) resolve to the same
    Minuit names across multiple JUNO bundles

Each detector class must expose a get_pull_defs() method returning
    { base_name: callable(minuit_value) -> (pull/sigma)**2 }
See detector_pull_defs.py for ready-to-paste implementations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import inspect
import numpy as np
from iminuit import Minuit

from .param_policy import ParamPolicy
from .shared_nuisances import (SHARED_PARAM_NAMES, get_shared_start_values,
                               get_shape_nuis_names, get_shape_nuis_start_values,
                               apply_shared_limits)


# ─────────────────────────────────────────────────────────────────────────────
# DetectorBundle — one dataset
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DetectorBundle:
    phase: str
    shell: str
    kind:  str                 # 'juno' | 'tao' | 'dyb'
    args:  Any
    detector: Any              # JUNODetector / TAODetector / DYBDetector
    backgrounds: dict
    cm: dict
    unc_key: str
    ensp_nom: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Name mangling
# ─────────────────────────────────────────────────────────────────────────────
def mangled_param_table(bundles: List[DetectorBundle],
                        policy: ParamPolicy
                        ) -> tuple[list[str], list[float], dict]:
    names: list[str] = []
    starts: list[float] = []
    seen: dict[str, int] = {}
    routing: dict[int, dict[str, str]] = {}

    for b in bundles:
        base_names  = b.detector.get_param_names()
        base_starts = b.detector.get_start_values()
        local: dict[str, str] = {}
        for base, start in zip(base_names, base_starts):
            mangled = policy.resolve(base, b.phase, b.shell)
            if mangled not in seen:
                seen[mangled] = len(names)
                names.append(mangled)
                starts.append(start)
            local[base] = mangled
        routing[id(b)] = local
    return names, starts, routing


# ─────────────────────────────────────────────────────────────────────────────
# Minuit limits & first-fix — defer to each detector's existing apply_limits
# via a shim that rewrites assignments through the mangling table
# ─────────────────────────────────────────────────────────────────────────────
class _Forwarder:
    def __init__(self, target, mapping, attr):
        self._t = target; self._m = mapping; self._a = attr
    def __setitem__(self, base_name, value):
        mangled = self._m.get(base_name, base_name)
        getattr(self._t, self._a)[mangled] = value

class _LimitShim:
    def __init__(self, target: Minuit, mapping: dict[str, str]):
        self.limits = _Forwarder(target, mapping, 'limits')
        self.fixed  = _Forwarder(target, mapping, 'fixed')


def apply_all_limits(m: Minuit, bundles: List[DetectorBundle], routing: dict) -> None:
    for b in bundles:
        b.detector.apply_limits(_LimitShim(m, routing[id(b)]))


def collect_first_fix(bundles: List[DetectorBundle], routing: dict) -> list[str]:
    out = []
    for b in bundles:
        for base in b.detector.get_first_fix():
            out.append(routing[id(b)][base])
    return list(dict.fromkeys(out))   # dedup preserving order


# ─────────────────────────────────────────────────────────────────────────────
# Per-bundle chi² adapter — picks values out of the flat dict by base name
# ─────────────────────────────────────────────────────────────────────────────
def call_detector_chi2(b: DetectorBundle, mapping: dict[str, str],
                       values_by_mangled: dict[str, float],
                       shared_kwargs: dict,
                       E_segments) -> float:
    """Re-builds the kwargs each detector's chi2_contribution expects."""
    base_names = b.detector.get_param_names()
    def g(base):       return values_by_mangled[mapping[base]]
    def vals(prefix):  return [values_by_mangled[mapping[n]] for n in base_names
                                if n.startswith(prefix)]

    if b.kind == 'juno':
        # Match the field renames the original run_minuit does when it
        # first builds `bckgs`: rate_unc → sigma, b2b_unc → b2b.
        # build_spectrum reads info['sigma'], info['b2b'], info['b2b_scale'],
        # info['name'], info['pull'] — NOT the raw YAML key names.
        bckgs_for_chi2 = {}
        for k, raw in b.backgrounds['JUNO'].items():
            bckgs_for_chi2[k] = {
                'pull':      g(f'bkg_{k}'),
                'sigma':     raw['rate_unc'],
                'b2b':       raw['b2b_unc'],
                'name':      raw['name'],
                'b2b_scale': raw['b2b_scale'],
            }
        out = b.detector.chi2_contribution(
            **shared_kwargs,
            Nrea     = g('Nrea'),
            NL       = vals('NL'),
            abc      = [g('abc0'), g('abc1'), g('abc2')],
            bckgs_for_chi2 = bckgs_for_chi2,
            snf_noneq_norm = vals('snf_noneq_norm'),
            eff_nuis = g('eff_nuis'),
            ThU_ratio= g('ThU_ratio'),
            core_powers_delta = vals('core_power'),
            efission = vals('efission'),
            unc=b.unc_key)
        return float(out[0]) + float(out[2])     # chi2 + chi2_penalty_er

    if b.kind == 'tao':
        return float(b.detector.chi2_contribution(
            Nrea_tao = g('Nrea_tao'),
            NL_tao   = [g(f'NL_tao{i}') for i in range(5)],
            abc_tao  = [g('abc_tao0'), g('abc_tao1'), g('abc_tao2')],
            snf_noneq_norm_tao = [g(f'snf_noneq_norm_tao{i}') for i in range(2)],
            core_power_tao = g('core_power_tao'),
            eff_nuis_tao = g('eff_nuis_tao'),
            bckg_rate = g('bckg_rate'),
            unc=b.unc_key, nuisance=shared_kwargs['nuisance'])[0])

    if b.kind == 'dyb':
        # snf_noneq_norm_dyb is a REQUIRED positional arg in DYB's
        # chi2_contribution signature. get_param_names() only includes it
        # in CPC-non-dagflow mode; in DAGflow or non-CPC mode it isn't
        # a free Minuit param, but we still must pass a value. The original
        # run_minuit hard-codes 0.0 in those cases — match that.
        kwargs = dict(nuisance_coarse=np.array(shared_kwargs['nuisance']),
                      E_segments=E_segments,
                      snf_noneq_norm_dyb=0.0)
        if 'snf_noneq_norm_dyb' in base_names:
            kwargs['snf_noneq_norm_dyb'] = g('snf_noneq_norm_dyb')
        return float(b.detector.chi2_contribution(**kwargs)[0])

    raise ValueError(f"unknown bundle kind: {b.kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Pull deduplication
# ─────────────────────────────────────────────────────────────────────────────
def pull_dedup_correction(values_by_mangled: dict[str, float],
                          bundles: List[DetectorBundle],
                          routing: dict) -> float:
    """
    For each Minuit parameter shared by N>1 bundles, each bundle's
    chi2_contribution emits one (pull/sigma)**2 term — total of N copies.
    We want exactly 1. Return the (N-1)*pull^2 over-count to SUBTRACT.

    Same logic for the JUNO ER 2D covariance penalty when ALL 7 of its
    parameters resolve to the same mangled names across multiple JUNO bundles.
    """
    by_mangled: dict[str, list] = {}
    for b in bundles:
        if not hasattr(b.detector, 'get_pull_defs'):
            # Single-bundle detector (e.g. TAO / DYB never split by phase):
            # nothing to dedup for it. Skip and keep going so JUNO bundles
            # still get their shared pulls deduplicated correctly.
            continue
        for base, fn in b.detector.get_pull_defs().items():
            mangled = routing[id(b)].get(base)
            if mangled is None: continue
            by_mangled.setdefault(mangled, []).append((b, base, fn))

    correction = 0.0
    for mangled, owners in by_mangled.items():
        if len(owners) <= 1: continue
        val = values_by_mangled[mangled]
        pull_sq = float(owners[0][2](val))     # all owners give same value
        correction += (len(owners) - 1) * pull_sq

    # ── ER 2D covariance dedup (JUNO only, NL!=pull mode) ─────────────────
    # The ER penalty is only over-counted when EVERY piece is identical
    # across the JUNO bundles: the 7 mangled param names AND the covariance
    # AND the reference point theta0. If covariance_er differs per phase
    # (Case B), each penalty is a distinct constraint that must ADD — so we
    # must NOT dedup in that case.
    juno_bundles = [b for b in bundles if b.kind == 'juno'
                    and getattr(b.args, 'NL', 'G4') != 'pull']
    if len(juno_bundles) > 1:
        er_bases = ['abc0', 'abc1', 'abc2', 'NL0', 'NL1', 'NL2', 'NL3']
        try:
            mangled_sets = [tuple(routing[id(b)][n] for n in er_bases)
                            for b in juno_bundles]
        except KeyError:
            mangled_sets = []
        same_names = bool(mangled_sets) and all(ms == mangled_sets[0]
                                                for ms in mangled_sets)
        same_cov = False
        if same_names:
            b0   = juno_bundles[0]
            cov0 = getattr(b0.detector, 'covariance_er', None)
            th0  = getattr(b0.detector, 'theta0_er', None)
            same_cov = True
            for b in juno_bundles[1:]:
                covb = getattr(b.detector, 'covariance_er', None)
                thb  = getattr(b.detector, 'theta0_er', None)
                if not (cov0 is not None and covb is not None
                        and np.array_equal(np.asarray(cov0), np.asarray(covb))
                        and th0 is not None and thb is not None
                        and np.array_equal(np.asarray(th0), np.asarray(thb))):
                    same_cov = False
                    break
        if same_names and same_cov:
            b0 = juno_bundles[0]
            er_val = b0.detector._get_chi2_penalty_er(
                *[values_by_mangled[routing[id(b0)][n]] for n in er_bases])
            correction += (len(juno_bundles) - 1) * float(er_val)
    return correction


# ─────────────────────────────────────────────────────────────────────────────
# Build chi² closure for Minuit
# ─────────────────────────────────────────────────────────────────────────────
def build_chi2(bundles: List[DetectorBundle], policy: ParamPolicy,
               args_juno, *, opp: bool = False
               ) -> tuple[Callable, list[str], list[float], dict]:
    det_names, det_starts, routing = mangled_param_table(bundles, policy)

    n_shape = len(args_juno.seg_bins_nonuniform)
    E_segments = np.array(args_juno.seg_bins_nonuniform)
    shared_names  = SHARED_PARAM_NAMES + get_shape_nuis_names(n_shape)
    shared_starts = (get_shared_start_values(args_juno)
                     + get_shape_nuis_start_values(n_shape))

    names  = shared_names + det_names
    starts = shared_starts + det_starts

    def chi2(*flat):
        values = dict(zip(names, flat))
        shared_kwargs = dict(
            sin2_12 = values['sin2_th12'],
            sin2_13 = values['sin2_th13'],
            dm2_21  = values['dm2_21'] * 1e-5,
            dm2_31  = values['dm2_31'] * 1e-3,
            me_rho  = values['me_rho'],
            nuisance= [values[f'nuis{i}'] for i in range(n_shape)],
            opp=opp)
        c = 0.0
        for b in bundles:
            c += call_detector_chi2(b, routing[id(b)], values,
                                    shared_kwargs, E_segments)
        c -= pull_dedup_correction(values, bundles, routing)
        return c

    chi2.__signature__ = inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for n in names])
    return chi2, names, starts, routing