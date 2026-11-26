"""Reactor flux normalisation, survival probability, reactor inputs.

Split out of noda.py (pure move); noda.noda re-exports everything."""
from . import nuosc as nuosc
import numpy as np
import yaml
from .constants import *
from ..core.binning import centers_to_edges
from ..core.spectrum import Spectrum

def GetSpectrumFromDAT(filename, scale_U235=1.0, scale_U238=1.0, scale_Pu239=1.0, scale_Pu241=1.0):
    """
    Read neutrino flux data from a .dat file with columns:
    E_nu [keV], flux_235, flux_239, flux_241, flux_238 (#nu / fission / keV)
    Convert energy to meV and return Spectrum (same class as GetSpectrumFromROOT).
    """
    data = np.loadtxt(filename)

    # Columns: E(keV), 235, 239, 241, 238
    E_keV = data[:, 0]
    E_meV = E_keV / 1000.0  # convert keV → meV
    E_edges = centers_to_edges(E_meV)
    keV_to_meV = 1000.0

    total_flux = keV_to_meV * (
        data[:, 1] * scale_U235 +
        data[:, 2] * scale_Pu239 +
        data[:, 3] * scale_Pu241 +
        data[:, 4] * scale_U238
    )
    # Create Spectrum
    spec = Spectrum(
        bin_cont=np.array(total_flux),
        bins=np.array(E_edges)
    )
    return spec

def compute_extrafactors(args, efission=None, core_baselines=None, core_powers=None, alpha=None, eff=0.0):
    """
    Compute normalization scaling factor for reactor flux at given baselines.

    Handles 0 m baseline by removing 1/L^2 dependence (useful for near/fake detectors).
    """

    power_conv = 6.24e21*60*60*24
    if alpha is None:
        alpha_arr = np.array(args.alpha)
    else:
        alpha_arr = alpha

    if efission is None:
        efission_arr = np.array(args.efission)
    else:
        efission_arr = np.array(efission)

    if core_powers is None:
        Pth_arr = np.array(args.core_powers)*power_conv
    else:
        Pth_arr = np.array(core_powers)*power_conv

    if core_baselines is None:
        L_arr = np.array(args.core_baselines)*100. #m to cm
    else:
        L_arr = np.array(core_baselines)*100.

    if eff:
        efficiency = eff
    else:
        efficiency = args.detector_efficiency
    prefactor = efficiency * args.veto * args.Np / (4 * np.pi)

    # Compute normalization denominator (total weighted fission energy)
    norm = np.sum(alpha_arr * efission_arr)

    # Handle zero-baseline case safely
    if np.any(L_arr == 0):
        geo_term = np.sum(Pth_arr)  # no 1/L^2 factor
    else:
        geo_term = np.sum(Pth_arr / (L_arr * L_arr))

    extrafactor = prefactor * geo_term / norm
    return extrafactor

def compute_extrafactors_vec(args, efission=None, core_baselines=None,
                              core_powers=None, alpha=None, eff=0.0):
    """
    Vectorized per-core extrafactor computation.

    Returns an array of shape (n_cores,), where entry i is
        prefactor * (P_i / L_i^2) / sum_j(alpha[i,j] * efission[j])

    Parameters
    ----------
    efission       : (n_iso,) fission energies, shared across cores
    core_baselines : (n_cores,) baselines in metres
    core_powers    : (n_cores,) thermal powers
    alpha          : (n_cores, n_iso) fission fractions per core,
                     or (n_iso,) if shared across all cores
    eff            : optional efficiency override
    """
    power_conv = 6.24e21 * 60 * 60 * 24

    alpha_arr = np.asarray(args.alpha if alpha is None else alpha,
                            dtype=float)
    if alpha_arr.ndim == 1:                       # broadcast shared α
        alpha_arr = alpha_arr[np.newaxis, :]

    efission_arr = np.asarray(args.efission if efission is None else efission,
                               dtype=float)        # (n_iso,)
    L_arr   = np.asarray(args.core_baselines if core_baselines is None
                          else core_baselines, dtype=float) * 100.0   # m → cm
    Pth_arr = np.asarray(args.core_powers if core_powers is None
                          else core_powers, dtype=float) * power_conv

    efficiency = eff if eff else args.detector_efficiency
    prefactor  = efficiency * args.veto * args.Np / (4.0 * np.pi)

    # per-core normalisation  norm[i] = Σ_j α[i,j] · e_fission[j]
    norm = alpha_arr @ efission_arr               # (n_cores,)

    # per-core geometric term  P_i / L_i² (preserve original L=0 fallback)
    with np.errstate(divide='ignore', invalid='ignore'):
        geo = np.where(L_arr == 0.0,
                       Pth_arr,
                       Pth_arr / (L_arr * L_arr))

    return prefactor * geo / norm                 # (n_cores,)

def compute_extrafactors_per_core(args, core_powers, core_baselines, fission_fracs, eff=0.0):
    """
    Compute normalization extrafactor for each core separately,
    based on power, baseline, and fission fractions from YAML.
    """
    extrafactors = []
    for i, frac in enumerate(fission_fracs):
        ffs = [
            frac['U235'],
            frac['U238'],
            frac['Pu239'],
            frac['Pu241']
        ]

        # single-core extrafactor
        ef = compute_extrafactors(
            args,
            core_baselines=[core_baselines[i]],
            core_powers=[core_powers[i]],
            alpha=ffs,
            eff=eff
        )
        extrafactors.append(ef)
    return np.array(extrafactors)

def load_reactor_summary(yaml_path="data/reactor_data.yaml"):
    with open(yaml_path, 'r') as f:
        summary = yaml.safe_load(f)

    core_names, core_powers, core_fractions, core_baselines = [], [], [], []

    for npp, entry in summary.items():
        # Case 1: NPP with individual cores (YJ, TS)
        if isinstance(entry, dict) and all(isinstance(v, dict) for v in entry.values()):
            for core, vals in entry.items():
                core_names.append(core)
                core_powers.append(vals['power'])
                core_fractions.append(vals['fission_fraction'])
                core_baselines.append(vals.get('baseline', 0.0))  # baseline in meters
        # Case 2: Combined NPP (DYB, FC)
        else:
            core_names.append(npp)
            core_powers.append(entry['power'])
            core_fractions.append(entry['fission_fraction'])
            core_baselines.append(entry.get('baseline', 0.0))

    return core_names, np.array(core_powers), core_fractions, np.array(core_baselines)


def GetPee(Enu_array, L, core_powers=None,
           sin2_th12=None, sin2_th13=None,
           dm2_21=None, dm2_31=None, dm2_32=None, dm2_ee=None,
           me_rho=0., opp=False, args=None, osc_formula=nuosc.AntiNueSurvProb):
    """
    Returns the weighted survival probability Pee for an arbitrary array of neutrino energies.
    Enu_array: numpy array of neutrino energies (can be 1D, 2D, or 3D).
    L: baseline or list of baselines.
    """
    # 1. Handle Oscillation Parameter Defaults (same logic as your GetOscillated)
    if sin2_th12 is None:
        nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=(not args.NMO_opt if opp else args.NMO_opt))
        sin2_th12 = nuosc.op_nom["sin2_th12"]
        sin2_th13 = nuosc.op_nom["sin2_th13"]
        dm2_21 = nuosc.op_nom["dm2_21"]
        dm2_31 = nuosc.op_nom["dm2_31"]
        dm2_ee = nuosc.op_nom["dm2_ee"]
        dm2_32 = dm2_31 - dm2_21 if dm2_32 is None else dm2_32

    # 2. Handle Baselines and Core Shares
    if not isinstance(L, list):
        L = [L]

    if core_powers is None:
        core_shares = np.full(len(L), 1.0 / len(L))
    else:
        # Weight by Power/L^2
        core_shares = np.array([p / (l * l) for p, l in zip(core_powers, L)])
        core_shares = core_shares / np.sum(core_shares)

    # 3. Calculate Weighted Probability
    # We initialize with zeros matching the shape of the input energy array
    total_P = np.zeros_like(Enu_array)

    for l, share in zip(L, core_shares):
        if osc_formula == nuosc.AntiNueSurvProb:
            P_core = osc_formula(Enu_array, l,
                                 sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                                 dm2_21=dm2_21, dm2_31=dm2_31, dm2_32=dm2_32,
                                 me_rho=me_rho)
        elif osc_formula == nuosc.AntiNueSurvProbYB:
            P_core = osc_formula(Enu_array, l,
                                 sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                                 dm2_21=dm2_21, dm2_ee=dm2_ee,
                                 me_rho=me_rho)

        total_P += P_core * share

    return total_P
