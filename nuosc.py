import numpy as np

op_nom = {}

def SetOscillationParameters(opt, NMO):
  # NMO -- "normal mass ordering" (True for NO, False for IO)
  global op_nom
  if opt == 'YB':  # Yellow Book, p.30
    op_nom["sin2_th12"] = 0.307
    op_nom["sin2_th13"] = 0.024
    op_nom["dm2_21"] = 7.54e-5
    op_nom["dm2_atm"] = 2.43e-3  # (dm2_31 + dm2_32)/2.
    op_nom["dm2_31"] = op_nom["dm2_atm"] + 0.5*op_nom["dm2_21"]
    op_nom["dm2_32"] = op_nom["dm2_atm"] - 0.5*op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if opt == 'PDG2016':
    op_nom["sin2_th12"] = 0.304
    op_nom["sin2_th13"] = 0.0219
    op_nom["dm2_21"] = 7.53e-5    # in eV2
    if NMO: op_nom["dm2_32"] = 2.44e-3
    else:   op_nom["dm2_32"] = -2.51e-3
    op_nom["dm2_31"] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if opt == 'PDG2018':   # From PDG 2018
    op_nom["sin2_th12"] = 0.297
    op_nom["dm2_21"] = 7.37e-5
    if NMO:
      op_nom["sin2_th13"] = 0.0215
      op_nom["sin2_th23"] = 0.425
      op_nom["dm2_31"] = 2.56e-3
      op_nom["dm2_32"] = op_nom["dm2_31"] - op_nom["dm2_21"]
    else:
      op_nom["sin2_th13"] = 0.0216
      op_nom["sin2_th23"] = 0.589
      op_nom["dm2_32"] = -2.54e-3
      op_nom["dm2_31"] = op_nom["dm2_32"] + op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if opt == 'Jinnan2019':
    NH_nom = True
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0218
    op_nom['dm2_21'] = 7.53e-5
    op_nom['dm2_32'] = 2.444e-3
    if NMO:
      op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    else:
      op_nom['dm2_31'] = op_nom["dm2_32"] - op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  if opt == 'PDG2019':
    NH_nom = True
    op_nom['sin2_th12'] = 0.307
    op_nom['sin2_th13'] = 0.0218
    op_nom['dm2_21'] = 7.53e-5
    if NMO:
      op_nom['dm2_32'] = 2.453e-3
      op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
    else:
      op_nom['dm2_32'] = 2.546e-3
      op_nom['dm2_31'] = op_nom["dm2_32"] - op_nom["dm2_21"]
    op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]

SetOscillationParameters(opt="PDG2019", NMO=True) #WARNING TODO: change this


def AntiNueSurvProb(E, L,
                    sin2_th12=op_nom["sin2_th12"],
                    sin2_th13=op_nom["sin2_th13"],
                    dm2_21=op_nom["dm2_21"],
                    dm2_31=op_nom["dm2_31"],
                    dm2_32=op_nom["dm2_32"],
                    me_rho=0.0): # (g/cm3)
#  |dm2_31| = |dm2_32|+|dm2_21|
#  (sign of dm2_xy does not matter)
  #
  # make sure that the mass splitting sum rule holds
  if dm2_32 == op_nom["dm2_32"]:
      dm2_31 = np.abs(dm2_31)
      dm2_32 = dm2_31 - dm2_21
  elif dm2_31 == op_nom["dm2_31"]:
      dm2_32 = np.abs(dm2_32)
      dm2_31 = dm2_32 + dm2_21


  cos2_th12 = 1. - sin2_th12
  cos_2th12 = cos2_th12 - sin2_th12
  cos2_th13 = 1. - sin2_th13
  cos_2th13 = cos2_th13 - sin2_th13

  #
  if me_rho != 0.:
    Ye = 0.5
    Acc = 1.52*1e-7*Ye*me_rho*E # eV^2/MeV from Jinnan
    #mat_eff_corr = 0.005*E/4.*rho/3.  # YB eq.3.8, p.45
    sin2_th12 = sin2_th12 * (1 - 2 * Acc * cos2_th12 / dm2_21)   # YB eq.3.6, p.45
    dm2_21 = dm2_21 * (1 + Acc * cos_2th12 / dm2_21)
    sin2_th13 = sin2_th13 * (1 - 2 * Acc * cos2_th13 / dm2_31)   # YB eq.3.6, p.45
    dm2_31 = dm2_31 * (1 + Acc * cos_2th13 / dm2_31)

    cos2_th12 = 1. - sin2_th12
    cos_2th12 = cos2_th12 - sin2_th12
    cos2_th13 = 1. - sin2_th13
    cos_2th13 = cos2_th13 - sin2_th13

  #
  sin2_2th12 = 4.*sin2_th12*cos2_th12
  sin2_2th13 = 4.*sin2_th13*cos2_th13
  cos4_th13 = cos2_th13**2

  dm2_32 = dm2_31 - dm2_21

  t1 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * L / E)**2
  t2 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * L / E)**2
  t3 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * L / E)**2
  return 1. - t1 - t2 - t3




def AntiNueSurvProbYB(E, L,
                      sin2_th12=op_nom["sin2_th12"],
                      sin2_th13=op_nom["sin2_th13"],
                      dm2_21=op_nom["dm2_21"],
                      dm2_ee=op_nom["dm2_ee"],
                      matter_eff=False):
  #
  if matter_eff:
    rho = 5.51 # (g/cm3) from Google
    mat_eff_corr = 0.005*E/4.*rho/3.  # YB eq.3.8, p.45
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
#  print("sqrt: {},   sin(phi): {},   phi: {}".format( sqrt, (cos2_th12 * np.sin(2.*sin2_th12*d_21) - sin2_th12 * np.sin(2.*cos2_th12*d_21)) / sqrt, phi ))
  if NH:  t1 = 0.5*sin2_2th13 * (1. - sqrt * np.cos(2*d_ee + phi))
  else:   t1 = 0.5*sin2_2th13 * (1. - sqrt * np.cos(2*d_ee - phi))
  t2 = sin2_2th12 * cos4_th13 * np.power(np.sin(d_21), 2)
  return 1. - t1 - t2
