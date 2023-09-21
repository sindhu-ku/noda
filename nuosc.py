import numpy as np

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
    op_nom['dm2_21'] = 7.53e-5
    if NO:
      op_nom['sin2_th23'] = 0.546
      op_nom['dm2_32'] = 2.453e-3
    else:
      op_nom['sin2_th23'] = 0.539
      op_nom['dm2_32'] = -2.536e-3
    op_nom['dm2_31'] = op_nom["dm2_32"] + op_nom["dm2_21"]
  else:
    print(f"No such option \"{opt}\"")
    return
  #
  op_nom["dm2_ee"] = (1-op_nom["sin2_th12"])*op_nom["dm2_31"] + op_nom["sin2_th12"]*op_nom["dm2_32"]
  #
  if not silent:
    print(f"Oscillation parameters are set to \"{opt}\":", op_nom)





def AntiNueSurvProb(E, L,
                    sin2_th12=None,
                    sin2_th13=None,
                    dm2_21=None,
                    dm2_31=None,
                    dm2_32=None,
                    me_rho=0.0): # (g/cm3)

  if sin2_th12 == None: sin2_th12 = op_nom["sin2_th12"]
  if sin2_th13 == None: sin2_th13 = op_nom["sin2_th13"]
  if dm2_21 == None:    dm2_21=op_nom["dm2_21"]
  if dm2_31 == None:    dm2_31=op_nom["dm2_31"]
  if dm2_32 == None:    dm2_32=op_nom["dm2_32"]
  #
  # make sure that the mass splitting sum rule holds
  if dm2_32 == op_nom["dm2_32"]:
      dm2_31 = np.abs(dm2_31)
      dm2_32 = dm2_31 - dm2_21
  elif dm2_31 == op_nom["dm2_31"]:
      dm2_32 = np.abs(dm2_32)
      dm2_31 = dm2_32 + dm2_21
  #
  cos2_th12 = 1. - sin2_th12
  cos_2th12 = cos2_th12 - sin2_th12
  cos2_th13 = 1. - sin2_th13
  cos_2th13 = cos2_th13 - sin2_th13
  #
  if me_rho != 0.:
    Ye = 0.5
    Acc = 1.52*1e-7*Ye*me_rho*E # eV^2/MeV from Jinnan
    #
    sin2_th12 = sin2_th12 * (1 - 2 * Acc * cos2_th12 / dm2_21)   # YB eq.3.6, p.45
    dm2_21 = dm2_21 * (1 + Acc * cos_2th12 / dm2_21)
    sin2_th13 = sin2_th13 * (1 - 2 * Acc * cos2_th13 / dm2_31)   # YB eq.3.6, p.45
    dm2_31 = dm2_31 * (1 + Acc * cos_2th13 / dm2_31)
    #
    cos2_th12 = 1. - sin2_th12
    cos_2th12 = cos2_th12 - sin2_th12
    cos2_th13 = 1. - sin2_th13
    cos_2th13 = cos2_th13 - sin2_th13
  #
  sin2_2th12 = 4.*sin2_th12*cos2_th12
  sin2_2th13 = 4.*sin2_th13*cos2_th13
  cos4_th13 = cos2_th13**2
  dm2_32 = dm2_31 - dm2_21
  #
  t1 = sin2_2th12 * cos4_th13 * np.sin(1.267 * dm2_21 * L / E)**2
  t2 = sin2_2th13 * cos2_th12 * np.sin(1.267 * dm2_31 * L / E)**2
  t3 = sin2_2th13 * sin2_th12 * np.sin(1.267 * dm2_32 * L / E)**2
  return 1. - t1 - t2 - t3



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
