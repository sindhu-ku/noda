import numpy as np

mp = 938.27208816  # MeV
mn = 939.56542052  # MeV
me = 0.51099895000  # MeV

delta_np = mn - mp  # MeV
Enu_threshold = 0.5 * (mn**2 / (mp - me) - mp + me)  # IBD threshold [MeV]

G_F = 1.1663787e-11  # Fermi weak constant [MeV**-2]
G_F_GeV2 = G_F * 1e06  # Fermi constanf [GeV**-2]
N_A = 6.02214076e23  # Avogadro number
sin2_tw = 0.22290  # sin^2 of theta_weak angle from CODATA2018
alpha = 0.0072973525664  # fine structure constant
V_ud = 0.9741           # CKM matrix element
mu_p = 2.7928           # proton magnetic moment
mu_n = -1.9130          # neutron magnetic moment
M_V = 840.0             # vector mass in MeV
M_A = 1032.0            # axial mass in MeV
m_pi = 139.57           # pion mass in MeV
t_C = 13.02 * (2 * np.pi / 360)  # Cabibbo angle [rad]
c = 299792458  # [m/s]
me_kg = 9.10938356e-31  # [kg]
hbar = 1.054571800e-34  # [J*s]
G_F_J = 4.5437957e14 * np.power(hbar * c, 3)  # [J**-2]
r2 = 0  # neutrino charge radius
mu = 0  # neutrino magnetic moment
qe = 1.602176634e-19  # electron charge [C]
hbarc = 197.3269804  # MeV fm
x = np.sqrt(2) * np.pi * alpha * r2 / (3 * G_F)  # elastic scattering variable
AMU2MeV = 931.5  # 1 AMU = 931 MeV
n_lifetime = 880.3  # neutron life time [s] (PDG2014)
F_A0 = 0.711
F_M0 = 1.516e-3 # Mev^-1
F_T0 = F_A0*2.01e-3 # Mev^-1
b_exc = 1.881e-15 # m
rho_exc = 0.23 

M_unified_atomic_kg = (
    1.6605390666e-27  # unified atomic mass unit= 1.6605390666e-27 kg from PDG2020
)

phase_space = 1.71465
g = 1.2762 #1.2723 #1.2701  # axial-vector coupling constant (PDG 2022/SV)
f = 1.0  # vector coupling constant
f2 = 3.706  # anomalous nucleon isovector magnetic moment


# Conversion factors
MEV_PER_JOULE = 1.0 / (qe * 1e6)
GW_TO_J_PER_SEC = 1e9
SEC_PER_DAY = 86400.0
KM_TO_CM = 1e5

############## Conversion factors ################
MeV_to_J = 1.0e6 * qe
J_to_MeV = 1.0 / MeV_to_J
MeV_to_cm = (hbar * c * J_to_MeV) ** 2 * 1.0e4 # squared! Same as hbarc
prob_factor = 1e03 / (hbar * c * J_to_MeV * 1e12)
