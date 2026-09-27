#!/usr/bin/env python3
import numpy as np
from .constants import G_F, N_A, hbarc, prob_factor

phase_factor = 1e-3 * prob_factor / 4  # ≈ 1.2669e-3  [1/(m/MeV * eV^2)]

def potential(rho, E):
    """
    Calculates the matter potential for neutrino oscillations.

    Args:
        rho (float): Matter density in g/cm^3.
        E (float): Neutrino energy in MeV.

    Returns:
        float: The potential 'A' in units of eV^2.
    """
    Ye = 0.5  # Electron fraction (approx. 0.5 for Earth's crust/mantle)

    # G_F [MeV^-2] -> eV^-2: * 1e-12
    # hbarc [MeV·fm] -> eV·cm: * 1e-13 (fm->cm) * 1e6 (MeV->eV)
    # N_A * rho [g/cm^3] -> number density [cm^-3]
    # E [MeV] -> eV: * 1e6
    hbarc_eV_cm = hbarc * 1e-13 * 1e6  # eV·cm
    G_F_eV2 = G_F * 1e-12              # eV^-2

    factor = 2 * np.sqrt(2) * G_F_eV2 * N_A * hbarc_eV_cm**3 * 1e6

    return -factor * Ye * rho * E  # eV^2


def sin2_nb(x, dm2):
    """
    Calculates sin^2(phase) for neutrino oscillations.

    Args:
        x (float): baseline / energy in m/MeV.
        dm2 (float): mass squared difference in eV^2.

    Returns:
        float: sin^2(phase_factor * dm2 * x)
    """
    phase = phase_factor * dm2 * x  # x in [m/MeV]
    return np.sin(phase)**2

op_nom = {}

def SetOscillationParameters(opt, NO, silent=True):
  # NO -- "normal ordering" (True for NO, False for IO)
  global op_nom
  #
  if opt == 'YB':  # Yellow Book, p.30
    op_nom["sin2_th12"] = 0.307
    op_nom["sin2_th13"] = 0.024
    op_nom["dm2_21"] = 7.54e-5
    op_nom["dm2_atm"] = 2.43e-3  # (dm2_31 + dm2_32)/2.
    op_nom["dm2_31"] = op_nom["dm2_atm"] + 0.5*op_nom["dm2_21"]
    op_nom["dm2_32"] = op_nom["dm2_atm"] - 0.5*op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  elif opt == 'PDG2020': # https://pdg.lbl.gov/2020/tables/rpp2020-sum-leptons.pdf
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0218
    op_nom['dm2_21'] = 7.53e-5
    if NO:
      op_nom['sin2_th23'] = 0.545
      op_nom['dm2_32'] = 2.453e-3
    else:
      op_nom['sin2_th23'] = 0.547
      op_nom['dm2_32'] = -2.546e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
  #
  elif opt == "PDG2022": # https://pdg.lbl.gov/2022/tables/rpp2022-sum-leptons.pdf
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0220
    op_nom['sin2_th12_err'] = 0.013
    op_nom['sin2_th13_err'] = 0.07e-2
    op_nom['dm2_21'] = 7.53e-5
    op_nom['dm2_21_err'] = 0.18e-5
    if NO:
      op_nom['sin2_th23'] = 0.546
      op_nom['sin2_th23_err'] = 0.022
      op_nom['dm2_32'] = 2.453e-3
      op_nom['dm2_32_err'] = 0.033e-3
    else:
      op_nom['sin2_th23'] = 0.539
      op_nom['sin2_th23_err'] = 0.021
      op_nom['dm2_32'] = -2.536e-3
      op_nom['dm2_32_err'] = 0.034e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom['dm2_31_err'] = np.sqrt(pow(op_nom["dm2_32_err"], 2) + pow(op_nom["dm2_21_err"], 2))
  elif opt == 'PDG2024': # https://pdg.lbl.gov/2024/tables/contents_tables.html
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0219
    op_nom['sin2_th13_err'] = 0.07e-2
    op_nom['dm2_21'] = 7.53e-5
    if NO:
      op_nom['sin2_th23'] = 0.558
      op_nom['dm2_32'] = 2.455e-3
    else:
      op_nom['sin2_th23'] = 0.553
      op_nom['dm2_32'] = -2.529e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
  #
  elif opt == 'PDG2025': # https://pdg.lbl.gov/2025/tables/contents_tables.html
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02169
    op_nom['sin2_th13_err'] = 0.00058
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.451e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.527e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]

  elif opt == 'xchecks': # https://pdg.lbl.gov/2025/tables/contents_tables.html
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02160
    op_nom['sin2_th13_err'] = 0.00058
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.451e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.527e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]

  elif opt == 'DYB':
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02175
    op_nom['sin2_th13_err'] = 0.063e-2
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.466e-3
      op_nom['dm2_31_dyb']     = 2.466e-3 + op_nom['dm2_21']   # DYB, NO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.571e-3
      op_nom['dm2_31_dyb']     = -2.571e-3 + op_nom['dm2_21']  # DYB, IO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom['dm2_31_err'] = 0.06e-3
  elif opt == 'NT2K':
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02175
    op_nom['sin2_th13_err'] = 0.063e-2
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.429e-03
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.477e-03
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom['dm2_31_err'] = 0.035e-3
  elif opt == 'NuFitLBL':
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02175
    op_nom['sin2_th13_err'] = 0.063e-2
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.442e-03
      op_nom['dm2_31_err'] = 0.0265e-3
      # LBL projection (NuFit IC19 w/o SK atmospheric, NO best fit):
      # theta23/delta_cp feed the f_lbl |Dm2| conversion, ordering-dependent.
      op_nom['theta23'] = 48.5    # [deg]
      op_nom['dcp']     = 177.0   # [deg]
      # Daya Bay external Dm2_31 constraint (pull_dm2_31): Dm2_32 + Dm2_21.
      op_nom['dm2_31_dyb']     = 2.466e-3 + op_nom['dm2_21']   # DYB, NO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.483e-03
      op_nom['dm2_31_err'] = 0.0275e-3
      # LBL projection (NuFit IC19 w/o SK atmospheric, IO best fit):
      op_nom['theta23'] = 48.6    # [deg]
      op_nom['dcp']     = 285.0   # [deg]
      op_nom['dm2_31_dyb']     = -2.571e-3 + op_nom['dm2_21']  # DYB, IO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
  elif opt == 'JUNO2026':
    # June 2026 unblinded 240-day data best fit (2-shell JUNO+DYB+LBL fit,
    # config/240-day_FV165/manifest_phases.yaml) — centrals for Asimov
    # studies AT the observed best-fit point. External constraints stay as
    # in NuFitLBL: DYB dm2_31 pull central/width, LBL theta23/dcp for the
    # f_lbl projection, th13 pull width.
    op_nom['sin2_th13_err'] = 0.063e-2
    if NO:
      op_nom['sin2_th12'] = 0.304215
      op_nom['sin2_th13'] = 0.021793
      op_nom['dm2_21']    = 7.36541e-5
      op_nom['dm2_31']    = 2.50819e-3
      op_nom['dm2_31_err'] = 0.0265e-3
      op_nom['sin2_th23'] = 0.534
      op_nom['theta23'] = 48.5    # [deg]  (NuFitLBL, NO)
      op_nom['dcp']     = 177.0   # [deg]
      op_nom['dm2_31_dyb']     = 2.466e-3 + 7.50e-5   # DYB external, NO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    else:
      op_nom['sin2_th12'] = 0.304216
      op_nom['sin2_th13'] = 0.021784
      op_nom['dm2_21']    = 7.36460e-5
      op_nom['dm2_31']    = -2.45168e-3
      op_nom['dm2_31_err'] = 0.0275e-3
      op_nom['sin2_th23'] = 0.537
      op_nom['theta23'] = 48.6    # [deg]  (NuFitLBL, IO)
      op_nom['dcp']     = 285.0   # [deg]
      op_nom['dm2_31_dyb']     = -2.571e-3 + 7.50e-5  # DYB external, IO
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    op_nom['dm2_32'] = op_nom['dm2_31'] - op_nom['dm2_21']
  elif opt == 'JUNO2026aux':
    # JUNO2026 truth point, but with the DYB dm2_31 pull centred ON the
    # truth instead of the external measurement — the median-toy
    # configuration of the FC NMO toys (sample_aux_dm2_31 draws the DYB
    # pseudo-measurement around the data best fit; its IO image is
    # -meas + 2 sin2_th12 dm2_21). Pair with dm2_31_lbl set to the truth
    # and NO direct dm2_31_lbl_IO in the config to mimic the toys' f_lbl
    # projection. For Asimov cross-checks of the toy median only.
    op_nom['sin2_th13_err'] = 0.063e-2
    if NO:
      op_nom['sin2_th12'] = 0.304215
      op_nom['sin2_th13'] = 0.021793
      op_nom['dm2_21']    = 7.36541e-5
      op_nom['dm2_31']    = 2.50819e-3
      op_nom['dm2_31_err'] = 0.0265e-3
      op_nom['sin2_th23'] = 0.534
      op_nom['theta23'] = 48.5
      op_nom['dcp']     = 177.0
      op_nom['dm2_31_dyb']     = 2.50819e-3            # = truth
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    else:
      op_nom['sin2_th12'] = 0.304216
      op_nom['sin2_th13'] = 0.021784
      op_nom['dm2_21']    = 7.36460e-5
      op_nom['dm2_31']    = -2.45168e-3
      op_nom['dm2_31_err'] = 0.0275e-3
      op_nom['sin2_th23'] = 0.537
      op_nom['theta23'] = 48.6
      op_nom['dcp']     = 285.0
      op_nom['dm2_31_dyb']     = -2.46337e-3           # -truth + 2 s12 dm21
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    op_nom['dm2_32'] = op_nom['dm2_31'] - op_nom['dm2_21']
  elif opt == 'oldtoymed':
    # Median-toy configuration of the PRE-July-6 NMO toy production
    # (nmo_plots_172m): spectrum truth = NuFitLBL centrals, but the aux
    # dm2_31 pulls centred on the DATA best fit (old run_one_toy_nmo drew
    # them around initial_fits' free fit while generating the Asimov at
    # op_nom). True-NO ensemble only. Pair with dm2_31_lbl = 2.50819e-3
    # and no direct dm2_31_lbl_IO. For decomposing the old-vs-new toy
    # median shift; not for physics fits.
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.02175
    op_nom['sin2_th13_err'] = 0.063e-2
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.442e-03
      op_nom['dm2_31_err'] = 0.0265e-3
      op_nom['theta23'] = 48.5
      op_nom['dcp']     = 177.0
      op_nom['dm2_31_dyb']     = 2.50819e-3            # data best fit
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.483e-03
      op_nom['dm2_31_err'] = 0.0275e-3
      op_nom['theta23'] = 48.6
      op_nom['dcp']     = 285.0
      op_nom['dm2_31_dyb']     = -2.46337e-3           # IO image of data bf
      op_nom['dm2_31_dyb_err'] = 0.06e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
  else:
    print(f"No such option \"{opt}\"")
    return
  #
  op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if not silent:
    print(f"Oscillation parameters are set to \"{opt}\":", op_nom)



def AntiNueSurvProb(E, baseline, dm2_21=None, dm2_31=None, dm2_32=None, sin2_th12=None, sin2_th13=None, me_rho=0.0, component="both"):
    """
    Calculates the neutrino survival probability in matter.

    Args:
        E (float or ndarray): Neutrino energy in MeV.
        baseline (float): Baseline in km.
        dm2_21 (float): Mass-squared difference (2-1) in eV^2.
        dm2_31 (float): Mass-squared difference (3-1) in eV^2.
        sin2_th12 (float): sin^2(theta_12).
        sin2_th13 (float): sin^2(theta_13).
        rho (float): Matter density in g/cm^3.
        component (str, optional): Which component to calculate ('both', 'slow', 'fast'). Defaults to 'both'.

    Returns:
        float or ndarray: Survival probability.

    Raises:
        Exception: If an invalid component is specified.
    """
    if sin2_th12 == None: sin2_th12 = op_nom["sin2_th12"]
    if sin2_th13 == None: sin2_th13 = op_nom["sin2_th13"]
    if dm2_21 == None:    dm2_21=op_nom["dm2_21"]
    if dm2_31 == None:    dm2_31=op_nom["dm2_31"]
    if dm2_32 == None:
        if dm2_31 == None or dm2_21 == None:
            print("this should not ever happen")
            dm2_32=op_nom["dm2_32"]
        else: dm2_32 = dm2_31 - dm2_21
    rho=me_rho
    x = baseline  / E

    deltam_ee = dm2_31 - sin2_th12 * dm2_21

    c2_12  = 1.0 - sin2_th12
    c2_13  = 1.0 - sin2_th13
    c_2_12 = 1.0 - 2.0 * sin2_th12

    pot = potential(rho, E)
    appo_12 = c2_13 * pot / dm2_21

    sin2_th12_m = sin2_th12 * (
        1.0 + 2.0 * c2_12 * appo_12 + 3.0 * c2_12 * c_2_12 * appo_12**2
    )
    dm2_21_m = dm2_21 * (
        1.0 - c_2_12 * appo_12 + 2.0 * sin2_th12 * c2_12 * appo_12**2
    )
    sin2_th13_m = sin2_th13 * (1.0 + 2.0 * c2_13 * pot / deltam_ee)
    dm2_31_m = dm2_31 * (
        1.0 - pot / dm2_31 * (c2_12 * c2_13 - sin2_th13 - sin2_th12 * c2_12 * c2_13 * appo_12)
    )
    dm2_32_m = dm2_32 * (
        1.0 - pot / dm2_32 * (sin2_th12 * c2_13 - sin2_th13 + sin2_th12 * c2_12 * c2_13 * appo_12)
    )

    aa = (1.0 - sin2_th13_m)**2 * 4.0 * sin2_th12_m * (1.0 - sin2_th12_m)
    bb = (1.0 - sin2_th12_m) * 4 * sin2_th13_m * (1.0 - sin2_th13_m)
    cc = sin2_th12_m * 4 * sin2_th13_m * (1.0 - sin2_th13_m)

    slow_comp = aa * sin2_nb(x, dm2_21_m)
    fast_comp = bb * sin2_nb(x, dm2_31_m) + cc * sin2_nb(x, dm2_32_m)  # FIX: dm2_32_m not dm2_31_m

    matter_prob = 1.0 - slow_comp - fast_comp

    if component == "both":
        return matter_prob
    elif component == "slow":
        return 1 - slow_comp
    elif component == "fast":
        return 1 - fast_comp
    else:
        raise Exception("Choose between slow, fast, both")
def AntiNueSurvProbYB(E, L,
                      sin2_th12=None,
                      sin2_th13=None,
                      dm2_21=None,
                      dm2_ee=None,
                      me_rho=0.0,
                      NO = True):
  if sin2_th12 == None: sin2_th12 = op_nom["sin2_th12"]
  if sin2_th13 == None: sin2_th13 = op_nom["sin2_th13"]
  if dm2_21 == None:    dm2_21=op_nom["dm2_21"]
  if dm2_ee == None:    dm2_ee=op_nom["dm2_ee"]
  #
  if me_rho != 0.:
    mat_eff_corr = 0.005*E/4.*me_rho/3.  # YB eq.3.8, p.45
    sin2_th12 = sin2_th12 * (1 + 2*mat_eff_corr)   # YB eq.3.6, p.45
    dm2_21 = dm2_21 * (1 - mat_eff_corr)
  #
  cos2_th12 = 1. - sin2_th12
  sin2_2th12 = 4.*sin2_th12*cos2_th12
  cos2_th13 = 1. - sin2_th13
  sin2_2th13 = 4.*sin2_th13*cos2_th13
  cos4_th13 = np.power(cos2_th13, 2)
  #
  d_21 = 1.267 * dm2_21 * L / E
  d_ee = 1.267 * dm2_ee * L / E
  #
  sqrt = np.sqrt(1. - sin2_2th12*np.power(np.sin(d_21),2))
  phi = np.arcsin( (cos2_th12 * np.sin(2.*sin2_th12*d_21) - sin2_th12 * np.sin(2.*cos2_th12*d_21)) / sqrt  )
  #
  if NO:  t1 = 0.5*sin2_2th13 * (1. - sqrt * np.cos(2*d_ee + phi))
  else:   t1 = 0.5*sin2_2th13 * (1. - sqrt * np.cos(2*d_ee - phi))
  t2 = sin2_2th12 * cos4_th13 * np.power(np.sin(d_21), 2)
  return 1. - t1 - t2





# def AntiNueSurvProb(E, L,
#                     sin2_th12=None,
#                     sin2_th13=None,
#                     dm2_21=None,
#                     dm2_31=None,
#                     dm2_32=None,
#                     me_rho=0.0): # (g/cm3)

#   if sin2_th12 == None: sin2_th12 = op_nom["sin2_th12"]
#   if sin2_th13 == None: sin2_th13 = op_nom["sin2_th13"]
#   if dm2_21 == None:    dm2_21=op_nom["dm2_21"]
#   if dm2_31 == None:    dm2_31=op_nom["dm2_31"]
#   if dm2_32 == None:
#       if dm2_31 == None or dm2_21 == None:
#           print("this should not ever happen")
#           dm2_32=op_nom["dm2_32"]
#       else: dm2_32 = dm2_31 - dm2_21
#   # # make sure that the mass splitting sum rule holds
#   # if dm2_32 == op_nom["dm2_32"]:
#   #     dm2_31 = np.abs(dm2_31)
#   #     dm2_32 = dm2_31 - dm2_21
#   # elif dm2_31 == op_nom["dm2_31"]:
#   #     dm2_32 = np.abs(dm2_32)
#   #     dm2_31 = dm2_32 + dm2_21

#   cos2_th12 = 1. - sin2_th12
#   cos_2th12 = cos2_th12 - sin2_th12
#   cos2_th13 = 1. - sin2_th13
#   cos_2th13 = cos2_th13 - sin2_th13
#   #
#   if me_rho != 0.:
#     Ye = 0.5
#     Acc = 1.52*1e-7*Ye*me_rho*E # eV^2/MeV from Jinnan
#     #
#     sin2_th12 = sin2_th12 * (1 - 2 * Acc * cos2_th12 / dm2_21)   # YB eq.3.6, p.45
#     dm2_21 = dm2_21 * (1 + Acc * cos_2th12 / dm2_21)
#     sin2_th13 = sin2_th13 * (1 - 2 * Acc * cos2_th13 / dm2_31)   # YB eq.3.6, p.45
#     dm2_31 = dm2_31 * (1 + Acc * cos_2th13 / dm2_31)
#     #
#     cos2_th12 = 1. - sin2_th12
#     cos_2th12 = cos2_th12 - sin2_th12
#     cos2_th13 = 1. - sin2_th13
#     cos_2th13 = cos2_th13 - sin2_th13
#   #
#   sin2_2th12 = 4.*sin2_th12*cos2_th12
#   sin2_2th13 = 4.*sin2_th13*cos2_th13
#   cos4_th13 = cos2_th13**2
#   dm2_32 = dm2_31 - dm2_21
#   #
#   t1 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * L / E)**2
#   t2 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * L / E)**2
#   t3 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * L / E)**2
#   return 1. - t1 - t2 - t3
