#!/usr/bin/env python3
"""
manifest.py

A "manifest" is a top-level YAML that lists datasets by (phase[, shell]),
each pointing at an existing per-phase config file. Two accepted shapes:

  # phase only — whole radius fit
  P25C: config/P25C_PRL2025.yaml
  P25D: config/P25D.yaml

  # phase + shell
  P25C:
    FV1: config/P25C_FV1.yaml
    FV2: config/P25C_FV2.yaml
  P25D:
    FV1: config/P25D_FV1.yaml

The loader normalises both to a flat list of `(phase, shell, cfg_path)`
tuples. When no shells are given, shell defaults to 'ALL'.

A single-file legacy run (just one per-phase YAML, no manifest) also
works — `load()` detects it and returns a one-element list with
phase='p0', shell='ALL'. So existing scripts keep running.
"""

from __future__ import annotations
import os
import yaml
from typing import List, Tuple

Dataset = Tuple[str, str, str]   # (phase, shell, cfg_path)


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
        return [('p0', 'ALL', path)]

    out: List[Dataset] = []
    for phase, val in doc.items():
        if isinstance(val, str):
            out.append((str(phase), 'ALL', val))
        elif isinstance(val, dict):
            for shell, cfg in val.items():
                if not isinstance(cfg, str):
                    raise ValueError(
                        f"manifest entry {phase}/{shell} must be a config path string")
                out.append((str(phase), str(shell), cfg))
        else:
            raise ValueError(
                f"manifest entry {phase} must be a path or a {{shell: path}} dict")
    return out