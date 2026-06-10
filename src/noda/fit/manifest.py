#!/usr/bin/env python3
"""
manifest.py

A "manifest" is a top-level YAML that lists datasets by (phase[, shell]).
Three accepted shapes:

  # 1) phase only — whole radius fit, one full config per phase
  P25C: config/P25C_PRL2025.yaml
  P25D: config/P25D.yaml

  # 2) phase + shell
  P25C:
    FV1: config/P25C_FV1.yaml
    FV2: config/P25C_FV2.yaml
  P25D:
    FV1: config/P25D_FV1.yaml

  # 3) common base config + per-phase overrides — ONE full config shared
  #    by all phases; each phase block lists ONLY the keys that change
  #    (livetime, efficiency, backgrounds, reactor power, detector
  #    response, …). Keys are looked up in the base config's sections
  #    (juno_inputs, then fit_options, then common_det_inputs) and
  #    replaced in place; nest under an explicit section name
  #    (e.g. `tao_inputs: {…}`) to target a specific section.
  #
  #    common: config/240-day_FV165/AllP_Jun2026.yaml
  #    phases:
  #      groupA:
  #        stat_opt: 197.98095days
  #        detector_efficiency: 0.69985553
  #        eff_unc: 0.02098609788
  #        backgrounds_yaml: config/240-day_FV165/backgrounds_groupA.yaml
  #        core_powers: [...]
  #        det_resp_file: data/...
  #      phase1:
  #        ...
  #
  #    A shell can be encoded in the phase name as "phase/shell"
  #    (e.g. "P25C/FV1"); otherwise shell defaults to 'ALL'.

The loader normalises all shapes to a flat list of
`(phase, shell, cfg_path, overrides)` tuples. For shapes 1 and 2 the
overrides dict is empty. When no shells are given, shell is 'ALL'.

A single-file legacy run (just one per-phase YAML, no manifest) also
works — `load()` detects it and returns a one-element list with
phase='p0', shell='ALL'. So existing scripts keep running.
"""

from __future__ import annotations
import os
import yaml
from typing import Dict, List, Tuple

Dataset = Tuple[str, str, str, dict]   # (phase, shell, cfg_path, overrides)

# Reserved top-level manifest keys that are not phase names
RESERVED_KEYS = {'common', 'phases', 'systematics'}


def load(path: str) -> List[Dataset]:
    """Load a manifest or a plain per-phase config; return normalised list."""
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)

    # Heuristic: a manifest has top-level keys mapping to paths or to
    # {shell: path} dicts. A regular per-phase config has keys like
    # 'fit_options', 'juno_inputs', etc. Distinguish by the presence
    # of any known per-phase section.
    PHASE_CFG_MARKERS = {'fit_options', 'juno_inputs', 'tao_inputs',
                         'dyb_inputs', 'common_det_inputs'}
    if isinstance(doc, dict) and set(doc.keys()) & PHASE_CFG_MARKERS:
        return [('p0', 'ALL', path, {})]

    out: List[Dataset] = []

    # ── shape 3: common base config + per-phase override blocks ─────────
    if isinstance(doc, dict) and 'common' in doc:
        base = doc['common']
        if not isinstance(base, str):
            raise ValueError("manifest 'common' must be a config path string")
        phases = doc.get('phases')
        if not isinstance(phases, dict) or not phases:
            raise ValueError(
                "manifest with 'common' needs a non-empty 'phases' mapping "
                "of {phase_name: {key: value, ...}}")
        for name, ov in phases.items():
            if ov is None:
                ov = {}
            if not isinstance(ov, dict):
                raise ValueError(
                    f"manifest phase '{name}' must map to a dict of "
                    f"override keys (or be empty)")
            phase, _, shell = str(name).partition('/')
            out.append((phase, shell or 'ALL', base, dict(ov)))
        return out

    # ── shapes 1 and 2: phase → full config path (optionally per shell) ──
    for phase, val in doc.items():
        if phase in RESERVED_KEYS:
            continue
        if isinstance(val, str):
            out.append((str(phase), 'ALL', val, {}))
        elif isinstance(val, dict):
            for shell, cfg in val.items():
                if not isinstance(cfg, str):
                    raise ValueError(
                        f"manifest entry {phase}/{shell} must be a config path string")
                out.append((str(phase), str(shell), cfg, {}))
        else:
            raise ValueError(
                f"manifest entry {phase} must be a path or a {{shell: path}} dict")
    return out
