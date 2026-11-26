"""Analytic TAO fast-neutron spectrum.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import numpy as np
from scipy.integrate import quad
from .constants import *
from ..core.spectrum import Spectrum

def MakeTAOFastNSpectrum(bins, A, B, C):
  bin_cont = np.zeros(len(bins) - 1)
  for i in range(len(bin_cont)):
    E_j = bins[i]
    E_j1 = bins[i + 1]
    bin_cont[i] = spectrum_integral(E_j, E_j1, A, B, C)
  print(np.sum(np.array(bin_cont)))
  fastn = Spectrum(bin_cont=bin_cont, bins=bins).GetScaled(1./np.sum(np.array(bin_cont)))
  return fastn

def spectrum_integral(E_j, E_j1, A, B, C):
  def integrand(E):
    return A * np.exp(B * E) + C
  integral, _ = quad(integrand, E_j, E_j1)
  return integral
