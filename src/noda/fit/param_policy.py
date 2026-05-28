#!/usr/bin/env python3
"""
param_policy.py

Decides, for each parameter base name, whether to give it ONE Minuit
parameter shared across all datasets or to suffix it per phase/shell.

The rule is symmetric — for any base name we ask:
    correlate_phase ?    correlate_shell ?
and produce:
    (True,  True ) → 'Nrea'                  # one parameter
    (False, True ) → 'Nrea__pP25C'           # one per phase
    (True,  False) → 'bkg_geo__sFV1'         # one per shell
    (False, False) → 'eff_nuis__pP25C_sFV1'  # one per (phase, shell)

Where the policy comes from
───────────────────────────
1. For backgrounds: read straight from backgrounds_*.yaml.  Each entry
   gains two optional keys `correlate_phase` / `correlate_shell`
   (both default to False — independent unless you say otherwise).
2. For other systematics (NL, abc, Nrea, snf, eff_nuis, …): a small
   hard-coded default table below, overridable from the manifest YAML
   under a top-level `systematics:` block.

Default policy choices encode the physics:
  - NL, abc, ThU_ratio, efission, shape_nuis → fully shared
  - Nrea, core_power, snf            → per phase  (reactor varies in time)
  - eff_nuis                         → per (phase, shell)

Bracketed base names are matched by family — 'NL0' is the 'NL' family,
'core_power3' is 'core_power'. The stripper is just rstrip-digits.
"""

from __future__ import annotations
import re
from typing import Dict, Tuple

CorrelationPair = Tuple[bool, bool]   # (correlate_phase, correlate_shell)


DEFAULT_SYSTEMATICS: Dict[str, CorrelationPair] = {
    # detector-intrinsic
    'NL':         (False,  False),
    'abc':        (False,  False),
    'ThU_ratio':  (True,  True),
    'efission':   (True,  True),
    'nuis':       (True,  True),   # shape nuisances
    # reactor — phase-dependent
    'Nrea':       (True, True),
    'core_power': (True, True),
    'snf_noneq_norm': (True, True),
    # detection efficiency — (phase, shell)
    'eff_nuis':   (False, False),
}


def _family(base_name: str) -> str:
    """'core_power3' → 'core_power',  'NL0' → 'NL',  'eff_nuis' → 'eff_nuis'."""
    return re.sub(r'\d+$', '', base_name)


class ParamPolicy:
    """Pure function: (base_name, phase, shell) → resolved Minuit name."""

    def __init__(self,
                 backgrounds_corr: Dict[str, CorrelationPair],
                 systematics_corr: Dict[str, CorrelationPair] | None = None):
        self.bkg = backgrounds_corr
        self.sys = {**DEFAULT_SYSTEMATICS, **(systematics_corr or {})}

    # ─── classification ──────────────────────────────────────────────────
    def correlation(self, base_name: str) -> CorrelationPair:
        if base_name.startswith('bkg_'):
            return self.bkg.get(base_name[4:], (False, False))
        return self.sys.get(_family(base_name), (True, True))   # default: shared

    # ─── name resolution ─────────────────────────────────────────────────
    def resolve(self, base_name: str, phase: str, shell: str) -> str:
        cp, cs = self.correlation(base_name)
        suffix = ''
        if not cp: suffix += f'__p{phase}'
        if not cs: suffix += f'__s{shell}'
        return base_name + suffix

    # ─── helpers for the dispatcher ──────────────────────────────────────
    def mangle_list(self, base_names, phase: str, shell: str):
        return [self.resolve(b, phase, shell) for b in base_names]


def policy_from_backgrounds(bg_dict: dict,
                            sys_override: dict | None = None) -> ParamPolicy:
    """
    Build a ParamPolicy by reading correlate_phase / correlate_shell from
    each background entry. Missing keys default to False (independent).

    `bg_dict` is the JUNO sub-dict of backgrounds_*.yaml.
    `sys_override` may be the optional `systematics:` block from the manifest.
    """
    bg = {}
    for key, info in bg_dict.items():
        bg[key] = (bool(info.get('correlate_phase', False)),
                   bool(info.get('correlate_shell', False)))
    return ParamPolicy(backgrounds_corr=bg, systematics_corr=sys_override)