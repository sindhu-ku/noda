#!/usr/bin/env python3
import numpy as np
def potential(rho, E):
    
    """
    Calculates the matter potential for neutrino oscillations.
    
    Args:
        rho (float): Matter density in g/cm^3.
        E (float): Neutrino energy in MeV.
        
    Returns:
        float: The potential 'A' in units of eV^2.
    """
    # Physical Constants
    # G_F is Fermi constant, N_A is Avogadro's number
    # The factor ~7.6e-5 converts (g/cm^3 * MeV) to eV^2
    # For antineutrinos, the potential is technically negative, 
    # but often the absolute value is passed if the signs are 
    # handled later in the oscillation formulas.
    
    Ye = 0.5  # Electron fraction (approx. 0.5 for Earth's crust/mantle)

    a = -1.52*1e-7*Ye*rho*E
    return a
def sin2_nb(x, dm2):
    """
    x: baseline / energy (km / MeV)
    dm2: mass squared difference (eV^2)
    """
    # 1.267 is (G_natural / 4) converted to units of km, MeV, eV^2
    phase = 1.267 * dm2 * x
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
    op_nom['sin2_th13'] = 0.02190 
    op_nom['sin2_th13_err'] = 0.00058 
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.451e-3
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.527e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom['dm2_31_err'] = np.sqrt(pow(op_nom["dm2_32_err"], 2) + pow(op_nom["dm2_21_err"], 2))
      
  elif opt == 'DYB': 
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0217
    op_nom['sin2_th13_err'] = 0.063e-2
    op_nom['dm2_21'] = 7.50e-5
    if NO:
      op_nom['sin2_th23'] = 0.534
      op_nom['dm2_32'] = 2.466e-3 
    else:
      op_nom['sin2_th23'] = 0.537
      op_nom['dm2_32'] = -2.571e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom['dm2_31_err'] = 0.06e-3
  #
  else:
    print(f"No such option \"{opt}\"")
    return
  #
  op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if not silent:
    print(f"Oscillation parameters are set to \"{opt}\":", op_nom)





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

    c2_12 = 1.0 - sin2_th12
    c2_13 = 1.0 - sin2_th13
    c_2_12 = 1.0 - 2.0 * sin2_th12
    pot = potential(rho, E)
    appo_12 = c2_13 * pot / dm2_21

    sin2_th12_m = sin2_th12 * (
        1.0 + 2.0 * c2_12 * appo_12 + 3.0 * c2_12 * c_2_12 * appo_12 * appo_12
    )
    dm2_21_m = dm2_21 * (
        1.0 - c_2_12 * appo_12 + 2.0 * sin2_th12 * c2_12 * appo_12 * appo_12
    )
    sin2_th13_m = sin2_th13 * (1.0 + 2.0 * c2_13 * pot / deltam_ee)

    dm2_31_m = dm2_31 * (
        1.0 - pot / dm2_31 * (c2_12 * c2_13 - sin2_th13 - sin2_th12 * c2_12 * c2_13 * appo_12)
    )
    dm2_32_m = dm2_32 * (
        1.0 - pot / dm2_32 * (sin2_th12 * c2_13 - sin2_th13 + sin2_th12 * c2_12 * c2_13 * appo_12)
    )

    aa = np.power(1.0 - sin2_th13_m, 2) * 4.0 * sin2_th12_m * (1.0 - sin2_th12_m)
    bb = (1.0 - sin2_th12_m) * 4 * sin2_th13_m * (1.0 - sin2_th13_m)
    cc = sin2_th12_m * 4 * sin2_th13_m * (1.0 - sin2_th13_m)

    slow_comp = aa * sin2_nb(x, dm2_21_m)
    fast_comp = bb * sin2_nb(x, dm2_31_m) + cc * sin2_nb(x, dm2_32_m)

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

