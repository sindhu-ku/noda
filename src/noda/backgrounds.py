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
