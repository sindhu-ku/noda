import numpy as np
from scipy import constants


class juno_inputs:

    core_baselines_1 = [52.76e3]
    core_powers_1 = [4.6]

    core_powers_12 = [2.9]*6 + [4.6]*4 + [17.4] + [17.4]
    core_baselines_12 = [52.75e3, 52.84e3, 52.42e3, 52.51e3, 52.12e3, 52.21e3] + [52.76e3, 52.63e3, 52.32e3, 52.20e3] + [215.e3] + [265.e3]

    core_powers_10 = [2.9]*6 + [4.6]*2 + [17.4] + [17.4]
    core_baselines_10 = [52.75e3, 52.84e3, 52.42e3, 52.51e3, 52.12e3, 52.21e3] + [52.76e3, 52.63e3] + [215.e3] + [265.e3]

    core_powers_9 = [2.9]*6 + [4.6]*2 + [17.4]
    core_baselines_9 = [52.74e3, 52.82e3, 52.41e3, 52.49e3, 52.11e3, 52.19e3] + [52.77e3, 52.64e3] + [215.e3]

    alpha = np.array([0.58, 0.07, 0.30, 0.05])
    efission = np.array([202.36, 205.99, 211.12, 214.26])


    #Pth = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 17.4, 17.4/2.]) #GW for 10
    Pth = np.array([2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 4.6, 4.6, 17.4]) #GW for 9
    #Pth = np.array([4.6]) #GW for 1
    #L = np.array([52.75e3, 52.84e3, 52.42e3, 52.51e3, 52.12e3, 52.21e3, 52.76e3, 52.63e3, 215.e3, 265.e3]) #m for 10
    L = np.array([52.75e3, 52.84e3, 52.42e3, 52.51e3, 52.12e3, 52.21e3, 52.76e3, 52.63e3, 215.e3]) #m for 9
    #L = np.array([52.76e3]) #m for 1
    Np = 20.*12.01/1.0079*1e7*constants.Avogadro

    detector_efficiency = 0.82

    me_rho = 2.45
    me_rho_scale = 0.15
    Pth2 = Pth*6.24e21*60*60*24 # MeV/day
    print("Pth per day")
    print(Pth2)
    Pth2_s = Pth*6.24e21 # MeV/s
    print("Pth per second")
    print(Pth2_s)
    L2 = L*1e2 #cm

    Mean_Ef = np.sum(alpha*efission)
    print("Mean energy per fission = {:.5e}".format(Mean_Ef))

    extrafactors_geo = detector_efficiency*Np/(4*np.pi)*1.#/(np.sum(alpha*efission))*np.sum((Pth2)/(L2*L2))
    extrafactors = detector_efficiency*Np/(4*np.pi)*1./(np.sum(alpha*efission))*np.sum((Pth2)/(L2*L2))

    print(" Extra factors")
    print(extrafactors)

    U235_scale=0.58
    U238_scale=0.07
    Pu239_scale=0.30
    Pu241_scale=0.05

    a, b, c = 0.0261, 0.0082, 0.0123
    a_err, b_err, c_err = 0.0002, 0.0001, 0.0004

    r2_unc=0.02
    eff_unc=0.01
    b2b_unc=0.0134
    snf_unc=0.3
    noneq_unc=0.3
    core_flux_unc=0.008
    acc_rate_unc = 0.01
    geo_rate_unc=0.3
    geo_b2b_unc=0.05
    lihe_rate_unc=0.2
    lihe_b2b_unc=0.1
    fneu_rate_unc=1.0
    fneu_b2b_unc=0.2
    aneu_rate_unc=0.5
    aneu_b2b_unc=0.5
    atm_rate_unc=0.5
    atm_b2b_unc=0.5
    rea300_rate_unc=0.02
    rea300_b2b_unc=0.05
