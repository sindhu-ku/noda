#!/usr/bin/env python3
import pickle
import csv
import sys
from .noda import *
from . import nuosc as nuosc
import numpy as np
import scipy as sp
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import gc
from joblib import Parallel, delayed
import uproot
import ROOT
from scipy.integrate import quad
from numpy.linalg import solve, slogdet
from array import array
from scipy.linalg import cho_solve, cho_factor
# from numba import njit4 
from .nl_geant4 import Geant4Model
import yaml
import hashlib
#np.set_printoptions(threshold=sys.maxsize)
#global settings:

G4model = Geant4Model(particle="e+")
G4model_gamma = Geant4Model(particle="gamma")

def GetNLcurve(bins, kB, fC):
    return G4model.eval(bins, kB, fC=fC)/Geant4Model(2.2, particle="gamma")
#TODO: Do this better in main
osc_formula_options = {'default': nuosc.AntiNueSurvProb,
                       'YB':      nuosc.AntiNueSurvProbYB}
osc_formula = osc_formula_options['default']
def SetOscFormula(opt='default'):
  if not opt in osc_formula_options:
    sys.err(" ### Error: no such oscillation formula option: '{}'".format(opt))
  global osc_formula
  osc_formula = osc_formula_options[opt]
  print(" # Oscillation formula set to '{}'".format(opt))

class Spectrum:

  def __init__(self, bin_cont=[0]*100, bins=np.linspace(0,10,101), xlabel="", ylabel=""):
    self.bin_cont = np.array(bin_cont)
    self.bins = bins
    self.xlabel = xlabel
    self.ylabel = ylabel

    for i, x in enumerate(self.bin_cont):  # make negative bins equal to one
      if x<0 : self.bin_cont[i] = 0
  def __add__(self, other):
# Should we check weather the bins are the same in two spectra? May be slow.
#    if self.bins != other.bins:
#      print("\n  ### Error: attempt to sum two spectra with different binning! \n")
#      return None
    bin_cont = [b1+b2 for b1, b2 in zip(self.bin_cont, other.bin_cont)]
    return Spectrum(bin_cont=bin_cont, bins=self.bins, xlabel=self.xlabel, ylabel=self.ylabel)

  def __sub__(self, other):
    bin_cont = [b1-b2 for b1, b2 in zip(self.bin_cont, other.bin_cont)]
    return Spectrum(bin_cont=bin_cont, bins=self.bins, xlabel=self.xlabel, ylabel=self.ylabel)

  def Copy(self, scale=1.):
    if scale != 1.:
      bin_cont = [b*scale for b in self.bin_cont]
    else:
      bin_cont = self.bin_cont
    return Spectrum(bin_cont=bin_cont, bins=self.bins, xlabel=self.xlabel, ylabel=self.ylabel)

  def GetAtEnergy(self, ene):
    return self.bin_cont[self.FindBin(ene)]

  def GetNBins(self):
    return len(self.bin_cont)

  def GetBinWidth(self, i=None):
    if i==None:  # assuming equal-siezed bins
      return (self.bins[-1]-self.bins[0])/self.GetNBins()
    else:
      return (self.bins[i+1]-self.bins[i])
        
  def ScaleByBinWidth(self):
    widths = np.array([self.GetBinWidth(i) for i in range(self.GetNBins())])
    self.bin_cont = np.array(self.bin_cont, dtype=float) * widths
      
  def GetBinCenters(self):
    return 0.5*(self.bins[:-1]+self.bins[1:])

  def GetIntegral(self, mode='per_bin', left_edge=None, right_edge=None):
    # mode: 'per_bin', 'per_MeV'
    i1 = 0
    i2 = len(self.bin_cont)
    if left_edge != None:  i1 = self.FindBin(left_edge)
    if right_edge != None: i2 = self.FindBin(right_edge)
    if i1<0: i1=0
    if mode == 'per_bin':
      return sum(self.bin_cont[i1:i2])
    if mode == 'per_MeV':
      bw = [b2-b1 for b1,b2 in zip(self.bins[i1:i2-1], self.bins[i1+1:i2])]
      return sum([w*c for w,c in zip(bw, self.bin_cont[i1:i2])])

 # def FindBin(self, val):
 #   # too low value:  -1
 #   # inside the range: 0, 1, .., nebins-1
 #   # too high value: nebins
 #   if val < self.bins[0]: return -1
 #   return next( (i for i,b in enumerate(self.bins) if b>val), len(self.bins)) - 1
  def FindBin(self, val):
    # too low value:  -1
    # inside the range: 0, 1, .., nebins-1
    # too high value: nebins
    if val < self.bins[0]:
        return -1
    if val >= self.bins[-1]:
        return len(self.bin_cont) - 1
    left = 0
    right = len(self.bins) - 2
    while left <= right:
        mid = (left + right) // 2
        if val < self.bins[mid]:
            right = mid - 1
        elif val >= self.bins[mid+1]:
            left = mid + 1
        else:
            return mid

  def AddToBin(self, i, val):
    if i >= 0 and i < len(self.bin_cont):
      self.bin_cont[i] += val

  def Normalize(self, normTo=1., mode="per_bin", left_edge=None, right_edge=None):
    # mode options: "per_MeV", "per_bin"
    old_int = self.GetIntegral(mode=mode, left_edge=left_edge, right_edge=right_edge)
    scale_factor = normTo/old_int
    self.bin_cont *= scale_factor
    return scale_factor

  def GetNormalized(self, normTo=1., mode="per_bin", left_edge=None, right_edge=None):
    old_int = self.GetIntegral(mode=mode, left_edge=left_edge, right_edge=right_edge)
    scale_factor = normTo/old_int
    self.bin_cont *= scale_factor
    return self

  def GetScaled(self, scale_factor):
    self.bin_cont *= scale_factor
    return self

  def GetScaledFit(self, scale_factor):
    new_bin_cont = self.bin_cont*scale_factor
    return Spectrum(bins=self.bins, bin_cont=new_bin_cont)

  def Rebin(self, new_bins, mode='simple'):
    if mode == 'simple':
      self.Rebin_simple(new_bins)
    if mode == 'spline':
      self.Rebin_spline(new_bins, keep_norm=True)
    if mode == 'spline-not-keep-norm':
      self.Rebin_spline(new_bins, keep_norm=False)
    return self

  def Rebin_simple(self, new_bins):   # Better for similar binning
    s_new = Spectrum(np.zeros(len(new_bins)-1), new_bins, xlabel=self.xlabel, ylabel=self.ylabel)
    for be1, be2, val in zip(self.bins[:-1], self.bins[1:], self.bin_cont):
      i1 = s_new.FindBin(be1)
      i2 = s_new.FindBin(be2)
      if i1 == i2: # if it is inside one bin
        s_new.AddToBin(i1, val)
      else:
        bw = be2-be1

        s_new.AddToBin(i1, val * (s_new.bins[i1+1]- be1) / bw)  # first bin

        for i in range(i1+1, i2):                             # intermediate bins
          s_new.AddToBin(i, val * (s_new.bins[i+1] - s_new.bins[i]) / bw)
        s_new.AddToBin(i2, val * (be2 - s_new.bins[i2]) / bw)  # last bin
    #print(s_new.bins)
    #print(s_new.bin_cont)
    self.bin_cont = s_new.bin_cont
    self.bins = s_new.bins
    del s_new

  def Rebin_spline(self, new_bins, keep_norm=True):   # Better for binning with much more bins
    s_new = Spectrum(np.zeros(len(new_bins)-1), new_bins, xlabel=self.xlabel, ylabel=self.ylabel)
    #x = [(b1+b2)/2. for b1,b2 in zip(self.bins[:-1],self.bins[1:])]  # center of bins
    #y = values
    x = self.bins
    y = [(v1+v2)/2. for v1,v2 in zip([0]+list(self.bin_cont), list(self.bin_cont)+[0])]
    spline = sp.interpolate.interp1d(x, y, kind='slinear', bounds_error=False, fill_value=(0,0))
    #new_x = [(b1+b2)/2. for b1,b2 in zip(s_new.bins[:-1],s_new.bins[1:])]  # center of bins
    for i, (x1, x2) in enumerate(zip(s_new.bins[:-1], s_new.bins[1:])):
      bw_new = x2-x1
      s_new.AddToBin(i, spline((x1+x2)/2.))
    if keep_norm:
      s_new.Normalize(self.GetIntegral(left_edge=new_bins[0], right_edge=new_bins[-1], mode='per_bin'))
    self.bin_cont = s_new.bin_cont
    self.bins = s_new.bins
    del s_new

  def Rebin_nonuniform(self, bins_nonuniform):
    new_bins = []
    for start, end, num_bins in bins_nonuniform:
      bin_edges = np.linspace(start, end, num_bins + 1)
      new_bins.extend(bin_edges[:-1])  # Exclude the last bin edge for overlap
    new_bins.append(bins_nonuniform[-1][1])  # Append the last bin edge

    new_bin_cont = np.zeros(len(new_bins) - 1)
   
    for i in range(len(new_bins) - 1):
      bin_start, bin_end = new_bins[i], new_bins[i+1]
      bin_mask = (self.bins[:-1] < bin_end) & (self.bins[1:] > bin_start)
      for j, in_bin in enumerate(bin_mask):
        if in_bin:
          overlap_start = max(bin_start, self.bins[j])
          overlap_end = min(bin_end, self.bins[j+1])
          overlap_fraction = (overlap_end - overlap_start) / (self.bins[j+1] - self.bins[j])
          new_bin_cont[i] += self.bin_cont[j] * overlap_fraction
    self.bins = np.array(new_bins)
    self.bin_cont = new_bin_cont
    return Spectrum(bin_cont=new_bin_cont, bins=np.array(new_bins), xlabel=self.xlabel, ylabel=self.ylabel)


  def GetOscillated(self, L, points_per_bin=1,
                core_powers=None,
                sin2_th12=None,
                sin2_th13=None,
                dm2_21=None,
                dm2_31=None,
                dm2_32=None,
                dm2_ee=None,
                me_rho=0.,
                ene_mode='vis', opp=False, args=''):

    if sin2_th12 is None:
      if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
      else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
      sin2_th12 = nuosc.op_nom["sin2_th12"]
    if sin2_th13 is None:
      if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
      else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
      sin2_th13 = nuosc.op_nom["sin2_th13"]
    if dm2_21 is None:
      if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
      else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
      dm2_21 = nuosc.op_nom["dm2_21"]
    if dm2_31 is None:
      if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
      else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
      dm2_31 = nuosc.op_nom["dm2_31"]
    if dm2_32 is None:
      if dm2_31 is None or dm2_21 is None:
        if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
        else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
        dm2_32 = nuosc.op_nom["dm2_32"]
      else: dm2_32 = dm2_31 - dm2_21
    if dm2_ee is None:
      if opp: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=not args.NMO_opt) #WARNING TODO: change this
      else: nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #WARNING TODO: change this
      dm2_ee = nuosc.op_nom["dm2_ee"]

    spec = self.Copy(0.)

    if type(L) != list:
      L = [L]
    if core_powers == None:
      core_shares = np.full(len(L), 1./len(L))
    else:
      core_shares = np.array([p/(l*l) for p, l in zip(core_powers, L)])
      core_shares = core_shares / np.sum(core_shares)

    eshift = 0
    if ene_mode == 'vis': 
      eshift = -0.511+1.293

    # --- VECTORIZED MATH STARTS HERE ---
    
    # 1. Prepare bin arrays
    b1 = self.bins[:-1]
    b2 = self.bins[1:]
    
    # 2. Setup the evaluation energies depending on points_per_bin
    if points_per_bin == 1:
        evis_array = 0.5 * (b1 + b2) + eshift
    else:
        # Create a 2D array of sub-bin points: shape (n_bins, points_per_bin)
        w = (b2 - b1) / points_per_bin
        evis_array = np.array([b1 + (i + 0.5) * w for i in range(points_per_bin)]).T + eshift

    # 3. Create an empty array to accumulate the total spectrum
    total_val = np.zeros_like(self.bin_cont)

    # 4. Loop over baselines, calculate probabilities for ALL bins at once
    for l, share in zip(L, core_shares):
      if osc_formula == nuosc.AntiNueSurvProb:
        # osc_formula takes the whole array and returns an array of probabilities
        P_array = osc_formula(evis_array, l,
                              sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                              dm2_21=dm2_21, dm2_31=dm2_31, dm2_32=dm2_32,
                              me_rho=me_rho)
      elif osc_formula == nuosc.AntiNueSurvProbYB:
        P_array = osc_formula(evis_array, l,
                              sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                              dm2_21=dm2_21, dm2_ee=dm2_ee,
                              me_rho=me_rho)
      
      # If using multiple points per bin, average the probabilities across axis 1
      if points_per_bin > 1:
          P_array = np.mean(P_array, axis=1)

      # Multiply the base spectrum by the probabilities and share, and add to total
      total_val += self.bin_cont * P_array * share

    # 5. Return exactly ONE Spectrum object at the very end
    return self.Copy(0.) + Spectrum(total_val, bins=self.bins, xlabel=self.xlabel)
      
  def ApplyDetResp(self, respMat, pecrop=None, norm=False):
    # DITCH THE .copy() - just reference the data
    data = respMat.data

    # WARNING: If norm is True, you are modifying the matrix IN PLACE
    # and doing heavy math. Pre-normalize respMat OUTSIDE the minuit loop if possible!
    if norm:
        col_sums = data.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        data /= col_sums

    if pecrop is None:
        return Spectrum(data.dot(self.bin_cont), bins=respMat.pebins, xlabel=r"$E_{\rm vis}$ (MeV)")

    pebins_full = respMat.pebins
    pebw = (pebins_full[-1]-pebins_full[0])/(len(pebins_full)-1)
    
    # Calculate indices
    peb1 = int(np.around((pecrop[0]-pebins_full[0])/pebw))
    peb2 = int(np.around((pecrop[1]-pebins_full[0])/pebw))
    pebins_crop = pebins_full[peb1:peb2+1]

    # Create a view of the matrix (fast) and do the dot product
    # Note: If respMat is scipy.sparse, dot() is even faster!
    matrix_slice = data[:, peb1:peb2]
    smeared_cont = self.bin_cont.dot(matrix_slice)

    return Spectrum(smeared_cont, bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (MeV)")

  # def ApplyDetResp(self, respMat, pecrop=None, norm=False):
  #   data = respMat.data.copy()

  #   if norm:
  #       col_sums = data.sum(axis=0, keepdims=True)
  #       col_sums[col_sums == 0] = 1.0
  #       data /= col_sums
  #       #respMat = RespMatrix(data=data, ebins=respMat.ebins, pebins=respMat.pebins)

  #   if pecrop is None:
  #       return Spectrum(data.dot(self.bin_cont), bins=respMat.pebins, xlabel=r"$E_{\rm vis}$ (MeV)")

  #   # if pecrop == None:
  #   #   return Spectrum(self.bin_cont.dot(respMat.data), bins=respMat.pebins)
  #   pebins_full = respMat.pebins
  #   pebw = (pebins_full[-1]-pebins_full[0])/(len(pebins_full)-1)
  #   peb1 = int(np.around((pecrop[0]-pebins_full[0])/pebw))
  #   peb2 = int(np.around((pecrop[1]-pebins_full[0])/pebw))
  #   pebins_crop = pebins_full[peb1:peb2+1]
  #   #return Spectrum(self.bin_cont.dot(respMat.data[peb1:peb2]), bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (MeV)")
  #   return Spectrum(self.bin_cont.dot(respMat.data[:, peb1:peb2]), bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (MeV)")

  def GetWithShiftedEnergy(self, eshift):
    Enu = self.bins
    Epos = Enu + eshift
    return Spectrum(self.bin_cont, bins=self.bins+eshift).Rebin(self.bins)

  def GetWithPositronEnergy(self, type='ana', inputfile=None, tf2name=None):
    Enu = self.bins
    Mn = 939.56536 #MeV
    Mp = 938.27203 #MeV
    Me = 0.51099893 #MeV
    if type=='TF2':
        Epos = np.zeros_like(Enu)
        file = ROOT.TFile(inputfile)
        func = file.Get(tf2name)
        for i, energy in enumerate(Enu):
            Epos_temp=0.
            for cos_theta in range(-1, 1, 100):
                Epos_temp += func.Eval(energy, cos_theta)

            Epos[i] = Epos_temp
    elif type=='ana':
        Deltanp = Mn - Mp
        Mdiff = -Enu + Deltanp + (Deltanp*Deltanp - Me*Me)/(2.0*Mp)
        Tn = 2 * Enu**2 / (Mn + Mp)
        Epos = (-Mn + np.sqrt(Mn*Mn - 4.0*Mp*Mdiff))/2.0
    else:
        RaiseValueError(f"Unknown type value for positron energy shift {type}")

    Evis = Epos + Me
    return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)

  # def GetWithPositronEnergy(self, inputfile=None, tf2name=None):
  #   Enu = self.bins  # Neutrino energy
  #   cos_theta_vals = np.linspace(-1, 1, 100)  # Discretize cos(θ) from -1 to 1
  #
  #   # Constants
  #   Mn = 939.56536  # Neutron mass (MeV)
  #   Mp = 938.27203  # Proton mass (MeV)
  #   Me = 0.51099893  # Electron mass (MeV)
  #   delta = (Mn**2 - Mp**2 - Me**2) / (2.0 * Mp)  # Exact substitution for δ
  #
  #   if inputfile and tf2name:
  #       Epos = np.zeros_like(Enu)
  #       file = ROOT.TFile(inputfile)
  #       func = file.Get(tf2name)
  #       for i, energy in enumerate(Enu):
  #           Epos_temp = 0.0
  #           for cos_theta in cos_theta_vals:
  #               Epos_temp += func.Eval(energy, cos_theta)
  #           Epos[i] = Epos_temp  # Averaging over cos(θ)
  #   else:
  #       # Using exact equations for Ee and Jacobian
  #       Epos = []
  #       for Enu_i in Enu:
  #           Epos_temp = 0.0
  #           epsilon_nu = Enu_i / Mp
  #           for cos_theta in cos_theta_vals:
  #               kappa = (1 + epsilon_nu)**2 - epsilon_nu**2 * cos_theta**2
  #               if kappa <= 0:
  #                   continue  # Avoid invalid square root
  #
  #               # Exact substitution for Ee (Equation 11)
  #               Epos_i = ((Enu_i - delta) * (1 + epsilon_nu) +
  #                         epsilon_nu * cos_theta *
  #                         np.sqrt((Enu_i - delta)**2 + kappa * Me**2)) / kappa
  #
  #               EPSILON = 1e-10
  #
  #               # Exact substitution for ve (pe/Ee)
  #               ve = np.sqrt(max(0, Epos_i**2 - Me**2)) / Epos_i if Epos_i > Me else EPSILON
  #
  #               # Adjust Jacobian to handle ve -> 0 safely
  #               if abs(ve) < EPSILON:
  #                   jacobian = 1.0  # Default value when ve is negligible
  #               else:
  #                   jacobian = ((1 + Enu_i / Mp * (1 - 1 / ve * cos_theta)) /
  #                             (1 - Epos_i / Mp * (1 - ve * cos_theta)))
  #               print(Epos_i, jacobian)
  #               Epos_temp += jacobian * Epos_i
  #           Epos.append(Epos_temp / len(cos_theta_vals))  # Averaging
  #
  #       Epos = np.array(Epos)
  #   Evis = Epos + 0.511  # Add 0.511 MeV for visible energy
  #   return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)
  #
  # def GetWithPositronEnergy(self, inputfile=None, tf2name=None):
  #   Enu = self.bins  # Neutrino energy
  #   cos_theta_vals = np.linspace(-1, 1, 100)  # Discretize cos(θ) from -1 to 1
  #
  #   # Constants
  #   Mn = 939.56536  # Neutron mass (MeV)
  #   Mp = 938.27203  # Proton mass (MeV)
  #   Me = 0.51099893  # Electron mass (MeV)
  #   delta = (Mn**2 - Mp**2 - Me**2) / (2.0 * Mp)  # Exact substitution for δ
  #
  #   # Using exact equations for Ee and Jacobian
  #   Epos = []
  #   for Enu_i in Enu:
  #       Epos_temp = 0.0
  #       epsilon_nu = Enu_i / Mp
  #       for cos_theta in cos_theta_vals:
  #           kappa = (1 + epsilon_nu)**2 - epsilon_nu**2 * cos_theta**2
  #           if kappa <= 0:
  #               continue  # Avoid invalid square root
  #
  #           # Exact substitution for Ee (Equation 11)
  #           Epos_i = ((Enu_i - delta) * (1 + epsilon_nu) +
  #                     epsilon_nu * cos_theta *
  #                     np.sqrt((Enu_i - delta)**2 + kappa * Me**2)) / kappa
  #
  #           EPSILON = 1e-10
  #           #Tn_i = 2 * Enu_i**2 / (Mn + Mp) * (1 - cos_theta)
  #           # Exact substitution for ve (pe/Ee)
  #           ve = np.sqrt(max(0, Epos_i**2 - Me**2)) / Epos_i if Epos_i > Me else EPSILON
  #
  #           # Adjust Jacobian to handle ve -> 0 safely
  #           if abs(ve) < EPSILON:
  #               jacobian = 1.0  # Default value when ve is negligible
  #           else:
  #               jacobian = ((1 + Enu_i / Mp * (1 - 1 / ve * cos_theta)) /
  #                         (1 - Epos_i / Mp * (1 - ve * cos_theta)))
  #
  #
  #           #print(Epos_i, jacobian)
  #           Epos_temp += (jacobian) * Epos_i
  #       Epos.append((Epos_temp / len(cos_theta_vals)))  # Averaging
  #
  #   Epos = np.array(Epos)
  #   Evis = Epos + Me  # Add 0.511 MeV for visible energy
  #   return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)
  #
  #

  def ShiftEnergy(self, eshift):
    old_bins = self.bins
    self.bins = self.bins + eshift
    self.Rebin(old_bins)

  def GetWithModifiedEnergy(self, **kwargs):
    s_new = self.Copy()
    s_new.ModifyEnergy(**kwargs)
    return s_new

  def ModifyEnergy(self, mode="lin", scale=1., spline=None, spectrum=None, kB=0.0, fC=0.0, kI=0.0, LY_shared=0.0,fit_bins=None):
    if mode == 'lin':
      bins_new = self.bins*scale
    elif mode == 'spline': # TODO vectorize it
      bins_new = []
      for e in self.bins[:]:
        if e==0.: e=1.e-10 # to avoid the divide-by-zero problem
        e *= scale*spline(e)
        bins_new.append(e)
    elif mode == "spectrum":
      bins_new = self.bins*np.array([spectrum.GetAtEnergy(e) for e in self.bins])*(1+LY_shared)
    elif mode == "G4":  
      bins_new = self.bins*G4model.eval(self.bins, kB, fC=fC, kI=kI)*LY_shared/G4model_gamma.eval(2.2, kB, fC=fC, kI=kI)
    elif mode == "fit":
      bins_new = self.bins*fit_bins
    old_bins = self.bins
    self.bins = bins_new
    self.Rebin(old_bins, mode='simple')


  #def GetWithModifiedEnergyBulk(self, mode='lin', scale=1., splines=None, spectra=None):
  #  get_sp_bins = lambda sp: np.array([sp.GetAtEnergy(e) for e in  self.bins])*self.bins
  #  if mode == 'spline':
  #    N = len(splines)
  #    bins_new = np.array([scale*s(self.bins) for s in splines])*self.bins
  #  elif mode == 'spectrum': # TODO: test it!
  #    bins_new = Parallel(n_jobs=-1)(delayed(get_sp_bins)(sp) for sp in spectra)
  #  else:
  #    sys.exit(" ### ERROR: modify energy is currently implemented only for 'spline' and 'spectrum' modes")
  #  output_spectra = [self.Copy()]*len(spectra)
  #  for i, b in enumerate(bins_new):
  #    output_spectra[i].bin_cont = self.bin_cont
  #    output_spectra[i].bins = b
  #  output_spectra = Parallel(n_jobs=-1)(delayed(lambda x:x.Rebin(self.bins, mode='simple'))(out_sp) for out_sp in output_spectra)
  #  return output_spectra
  def GetWithModifiedEnergyBulk(self, mode='lin', scale=1., splines=None, spectra=None):
    if mode == 'spline':
      N = len(splines)
      bins_new = np.array([scale*s(self.bins) for s in splines])*self.bins
    elif mode == 'spectrum': # TODO: test it!
      pass
      '''
      a = np.array([spectrum.GetAtEnergy(e) for e in self.bins])
      print(a.shape)
      b = self.bins
      bins_new = np.tensordot(a , b, axes=1) # sum of products of 2nd dimension of a and 1st dimension of b
      '''
    else:
      sys.exit(" ### ERROR: modify energy is currently implemented only for 'spline' mode")
    output_spectra = [self.Copy()]*N
    for i, b in enumerate(bins_new):
      if i%10 == 0:
        print(f"         {i} / {N} \r", end='')
      output_spectra[i].bin_cont = self.bin_cont
      output_spectra[i].bins = b
      output_spectra[i].Rebin(self.bins, mode='simple')
    return output_spectra

  def GetWeightedWithSpectrum(self, spec):
    s_new = self.Copy()
    s_new.WeightWithSpectrum(spec)
    return s_new

  def WeightWithSpectrum(self, spec):
    if not np.array_equal(self.bins, spec.bins):
      sys.exit(" ### ERROR: attempt to weight spectrum with another one of different binning ")
    self.bin_cont *= spec.bin_cont

  def GetWeightedWithFunction(self, f):
    s_new = self.Copy()
    s_new.WeightWithFunction(f)
    return s_new

  def WeightWithFunction(self, f):
    values = []
    bin_centers = 0.5*(self.bins[1:]+self.bins[:-1])
    for bin_c, val in zip(bin_centers, self.bin_cont):
      values.append(val*f(bin_c))
    self.bin_cont = np.array(values)

  def Trim(self, crop):
    lower_edge, higher_edge = crop
    bins_full = self.bins
    binw = (bins_full[-1]-bins_full[0])/(len(bins_full)-1)
    bin1 = int(np.around((lower_edge-bins_full[0])/binw))
    bin2 = int(np.around((higher_edge-bins_full[0])/binw))
    self.bin_cont = self.bin_cont[bin1:bin2]
    self.bins = bins_full[bin1:bin2+1]

  def GetTrimmed(self, crop):
    s_new = self.Copy()
    s_new.Trim(crop)
    return s_new

  def GetCovMatrixFromRandSample(self, s_rnd):
    nb = self.GetNBins()
    data = np.zeros(shape=(nb,nb))
    for i,s in enumerate(s_rnd):
      data += np.outer(self.bin_cont - s.bin_cont, self.bin_cont - s.bin_cont)
    return CovMatrix(data/len(s_rnd), bins=self.bins, axis_label=self.GetXlabel())

  def GetStatCovMatrix(self):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,x in enumerate(self.bin_cont):
      data[i][i] = x
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetRateCovMatrix(self, sigma):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,x in enumerate(self.bin_cont):
      for j,y in enumerate(self.bin_cont):
        data[i][j] = sigma*sigma*x*y
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetCNPStatCovMatrix(self,exp):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,x in enumerate(self.bin_cont):
      #if(x==0): x = min(exp.bin_cont)
      if x > 0:
          data[i][i] = 3./((1./x) + (2./exp.bin_cont[i]))
      else:
          data[i][i] = 3./((1./1e-12) + (2./exp.bin_cont[i]))
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  # def GetB2BCovMatrix(self, sigma):
  #   nb = len(self.bin_cont)
  #   data = np.zeros(shape=(nb,nb))
  #   for i,x in enumerate(self.bin_cont):
  #     data[i][i] = x*x*sigma*sigma
  #   return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)
  # def GetB2BCovMatrix(self, sigma, w_source):
  #   """
  #   Construct bin-to-bin covariance matrix for background spectrum with non-uniform bins.

  #   Parameters
  #   ----------
  #   sigma : float
  #       Relative bin-to-bin uncertainty (defined for source bin width w_source).
  #   w_source : float
  #       Bin width (in MeV) for which sigma is defined.

  #   Returns
  #   -------
  #   CovMatrix
  #       Background covariance matrix accounting for target bin widths.
  #   """
  #   nb = len(self.bin_cont)
  #   data = np.zeros((nb, nb))

  #   # compute widths of current bins
  #   w_target = np.diff(self.bins)

  #   # rescaling factor for each bin: s = w_source / w_target
  #   s = w_source / w_target

  #   # diagonal covariance elements: (s * sigma * μ_b)^2
  #   var = (s * sigma * self.bin_cont)**2
  #   np.fill_diagonal(data, var)

  #   return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetB2BCovMatrix(self, sigma, w_source, pred=None):
    """
    Construct bin-to-bin covariance matrix for a background spectrum with non-uniform bins.

    Parameters
    ----------
    sigma : float
        Relative bin-to-bin uncertainty (defined for source bin width w_source).
    w_source : float
        Bin width (in MeV) for which sigma is defined.
    mu_pred : array-like, optional
        Predicted counts per bin for this background. If None, uses self.bin_cont.

    Returns
    -------
    CovMatrix
        Background covariance matrix accounting for target bin widths and background prediction.
    """
    nb = len(self.bin_cont)
    data = np.zeros((nb, nb))

    # compute widths of current bins
    w_target = np.diff(self.bins)

    # rescaling factor: s = w_source / w_target
    #s = 1.0
    s = w_source / w_target

    # use provided prediction if available
    if pred is None:
        mu = self.bin_cont
    else:
        mu = np.array(pred)

    # diagonal covariance elements: (s * sigma * mu_b)^2
    var =  s*( sigma * mu)**2

    np.fill_diagonal(data, var)

    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)
      
  def GetVariedB2BCovMatrix(self, sigmas):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,(x,sigma) in enumerate(zip(self.bin_cont, sigmas)):
      data[i][i] = x*x*sigma*sigma
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

 # def GetVariedB2BCovMatrix(self, b2b_spectra):
 #   sigmas = np.zeros(len(self.bin_cont))
 #   for ene, sigma in zip(b2b_spectra.bins[:-1], b2b_spectra.bin_cont):
 #     ibin = self.FindBin(ene)
 #     if ibin >= 0 and ibin < len(sigmas):
 #       sigmas[ibin] = 5#sigma
 #   nb = len(self.bin_cont)
 #   data = np.zeros(shape=(nb,nb))
 #   for i,(x,sigma) in enumerate(zip(self.bin_cont, sigmas)):
##      print(x, sigma)
 #     data[i][i] = x*x*sigma*sigma
 ##     print(data[i][i])
 #   return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  #def GetVariedB2BCovMatrixFromROOT(self, fname, hname):
  #  rf = uproot.open(fname)
  #  hist = rf[hname]
  #  edges = hist.axis().edges()
  #  sigmas = np.zeros(len(self.bin_cont))
  #  for ene, sigma in zip(hist.axis().edges()[:-1], hist.values()):
  #    ibin = self.FindBin(ene)
  #    if ibin >= 0 and ibin < len(sigmas):
  #      sigmas[ibin] = sigma
  #  if len(sigmas) != len(self.bin_cont):
  #    print(f" ### WARNING: number of b2b bins is {len(sigmas)}; number of spectrum bins is {len(self.bin_cont)}.")
  #  if edges[1]-edges[0] != self.bins[1]-self.bins[0]:
  #    print(f" ### WARNING: bin width for b2b is {edges[1]-edges[0]}; spectrum bin width is {self.bins[1]-self.bins[0]}.")
  #    #print(f"     Doing rebinning for b2b uncertainties")
  #  return self.GetVariedB2BCovMatrix(sigmas)

  def GetVariedB2BCovMatrixFromROOT(self, fname, hname):
    rf = uproot.open(fname)
    hist = rf[hname]
    sigmas_inv_var = np.zeros(len(self.bin_cont))
    hist_edges = hist.axis().edges()
    hist_values = hist.values()

    max_energy = self.bins[-1]
    bin_width = hist_edges[-1] - hist_edges[-2]

    while hist_edges[-1] < max_energy:
        hist_edges = np.append(hist_edges, hist_edges[-1] + bin_width)
        hist_values = np.append(hist_values, hist_values[-1])
    for i in range(len(self.bins)-1):
        for j in range(len(hist_edges) - 1):
            if hist_edges[j] > self.bins[i+1]: break
            if hist_edges[j] < self.bins[i]: continue
            sigma = hist_values[j]
            inv_var = 1. / sigma**2
            sigmas_inv_var[i] += inv_var

    sigmas = 1. / np.sqrt(sigmas_inv_var)
    return self.GetVariedB2BCovMatrix(sigmas)
  
  def WritetoROOT(self, histname, outfile):
    output_file = ROOT.TFile(outfile, "update")
    bins = self.bins
    bin_cont = self.bin_cont
    hist = ROOT.TH1D(histname, histname, len(bins) - 1, array('d', bins))
    hist.Sumw2()
    for i in range(len(bin_cont)):
        hist.SetBinContent(i + 1, bin_cont[i])
    hist.Write()
    output_file.Close()

  def SetXlabel(self, label):
    self.xlabel = label

  def GetXlabel(self):
    return self.xlabel

  def SetYlabel(self, label):
    self.ylabel = label

  def Plot(self, fname, **kwargs):
    self.Print(fname, **kwargs)

  def Print(self, fname, xlabel=None, ylabel=None, leg_labels=None, colors=None, extra_spectra=[], xmin=None, xmax=None, log_scale=False, ymin=None, ymax=None, yinterval=None, title=None):
    if xlabel != None: self.xlabel = xlabel
    if ylabel != None: self.ylabel = ylabel
    if leg_labels == None:
      leg_labels = [None] + [None]*len(extra_spectra)
    elif type(leg_labels) == 'str':
      leg_labels = [leg_labels] + [None]*len(extra_spectra)
    if colors == None:
      colors = ['darkred'] + ['lightsteelblue']*len(extra_spectra)
    elif type(colors) == 'str':
      colors = [colors] + ['lightsteelblue']*len(extra_spectra)
    plt.figure(figsize=(8,6))

    weights = self.bin_cont #/np.sum(self.bin_cont)
    plt.hist(self.bins[:-1],bins=self.bins, weights=weights, fill=False, histtype='step', linewidth=2, color=colors[0], label=leg_labels[0], log=log_scale)
    for i,s in enumerate(extra_spectra):
      weights_extra = s.bin_cont #/ np.sum(s.bins)
      plt.hist(s.bins[:-1], bins=s.bins, weights=weights_extra, fill=False, histtype='step', linewidth=2, color=colors[i+1], label=leg_labels[i+1], log=log_scale)
  
    if xmin is not None: plt.xlim(left=xmin)
    if xmax is not None: plt.xlim(right=xmax)
    if ymin is not None: plt.ylim(bottom=ymin)
    if ymax is not None: plt.ylim(top=ymax)
    if yinterval: plt.yticks(np.arange(ymin, ymax, yinterval))
    plt.xlabel(self.xlabel, fontsize=18)
    plt.ylabel(self.ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    from matplotlib.lines import Line2D
    if all(leg_labels):
        legend_handles = []
        for i, lab in enumerate(leg_labels):
            if lab is not None:
                legend_handles.append(
                    Line2D([0], [0], color=colors[i], linewidth=2, label=lab)
                )
        plt.legend(handles=legend_handles, fontsize=16)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

  def Dump(self, fname, bin_format="", value_format="", delimiter=' '):
    bin_format = "{"+bin_format+"}"
    value_format = "{"+value_format+"}"
    with open(fname, 'w') as csv_file:
      #for b, v in zip(self.bins[:-1], self.bin_cont):
      for b, v in zip(0.5*(self.bins[:-1]+self.bins[1:]), self.bin_cont):
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerow([bin_format.format(b),
                         value_format.format(v)])



def determinant(matrix):
    """
    Calculates the determinant of a square matrix using LU decomposition.

    Args:
        matrix (ndarray): The input square matrix.

    Returns:
        float: The determinant of the matrix.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Perform eigenvalue decomposition
    eigvals = np.linalg.eigvals(matrix)

    # Calculate the logarithm of the absolute values of the eigenvalues
    log_abs_eigvals = np.log(np.abs(eigvals))

    # Calculate the sum of the logarithms
    log_det = np.sum(log_abs_eigvals)


    return log_det


    return det_value

class CovMatrix:

  def __init__(self, data, bins=[], axis_label=""):
    self.data = np.array(data)
    if self.IsInvertible():    # keep pre-calculated inverted matrix too
      self.data_inv = np.linalg.inv(self.data)
    else:
      self.data_inv = None
    if len(bins)==0: bins = np.linspace(0, data.shape[0]+1, data.shape[0])
    self.bins = bins
    self.axis_label = axis_label

  def __add__(self, o):
    return CovMatrix(self.data + o.data, bins=self.bins, axis_label=self.axis_label)

  def Add(self, cm):  # to be removed
    return CovMatrix(self.data + cm.data, bins=self.bins, axis_label=self.axis_label)
  
  def Rebin(self, new_bins):
    fine_bins = np.asarray(self.bins)
    new_bins  = np.asarray(new_bins)

    assert np.all(np.diff(fine_bins) > 0)
    assert np.all(np.diff(new_bins) > 0)

    n_coarse = len(new_bins) - 1
    n_fine   = len(fine_bins) - 1

    R = np.zeros((n_coarse, n_fine))

    for i in range(n_coarse):
        c_low, c_high = new_bins[i], new_bins[i+1]
        for j in range(n_fine):
            f_low, f_high = fine_bins[j], fine_bins[j+1]

            overlap = max(0.0, min(c_high, f_high) - max(c_low, f_low))
            width   = f_high - f_low
            if overlap > 0:
                R[i, j] = overlap / width

    rebinned_data = R @ self.data @ R.T
    rebinned_data = 0.5 * (rebinned_data + rebinned_data.T)

    return CovMatrix(rebinned_data, bins=new_bins, axis_label=self.axis_label)
  def Extend(self, other):
    A = self.data.shape[0]
    B = other.data.shape[0]
    extended_data = np.zeros((A + B, A + B))
    extended_data[:A, :A] = self.data
    extended_data[A:, A:] = other.data
    new_bins = np.concatenate([self.bins, other.bins])
    return CovMatrix(data=extended_data, bins=new_bins)

  def Scale(self, scale_factor):
    sf2 = scale_factor*scale_factor
    sf2_inv = 1./sf2
    self.data *= sf2
    if self.IsInvertible():
      self.data_inv *= sf2_inv

  def GetScaled(self, scale_factor):
    sf2 = scale_factor*scale_factor
    sf2_inv = 1./sf2
    self.data *= sf2
    if self.IsInvertible():
      return CovMatrix(data=self.data_inv * sf2_inv, bins=self.bins)
        
  def IsInvertible(self):
    rank = np.linalg.matrix_rank(self.data)
    size = self.data.shape[0]
    return rank == size

  def SetXlabel(self, label):
    self.xlabel = label

  def SetYlabel(self, label):
    self.ylabel = label

  def Plot(self, fname, **kwargs):
    self.Print(fname, **kwargs)

  def Print(self, fname, bin_min=0, bin_max=None, vmin=None, vmax=None, log_scale=False):
    x,y,val = [],[],[]
    if bin_max == None: bin_max = self.data.shape[0]
    for index, v in np.ndenumerate(self.data):
      if index[0] < bin_min: continue
      if index[1] < bin_min: continue
      if index[0] > bin_max: continue
      if index[1] > bin_max: continue
      x.append(self.bins[index[0]])
      y.append(self.bins[index[1]])
      val.append(v)
    fig, ax = plt.subplots()
    if vmin==None: vmin=min(val)
    if vmax==None: vmax=max(val)
    if not log_scale:
      norm = cm.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
      norm = cm.colors.LogNorm(vmin=1, vmax=vmax)
    counts, xedges, yedges, im = plt.hist2d(x, y, bins=[self.bins[bin_min:bin_max+1],
                                            self.bins[bin_min:bin_max+1]], weights=val,
                                            cmap=plt.get_cmap("jet"),
                                            norm=norm)
    plt.colorbar(im, ax=ax)
    plt.xlabel(self.axis_label)
    plt.ylabel(self.axis_label)
    plt.savefig(fname)
    plt.close()

  def PlotProfile(self, fname):
    x = self.bins[:-1]
    height = np.sum(self.data, axis=1)
    plt.bar(x, height, align='edge')
    plt.xlabel(self.axis_label)
    plt.ylabel("Absolute uncertainty")
    plt.savefig(fname)
    plt.close()

  def Save(self, fname):
    f = open(fname, "wb")
    pickle.dump(self, f, 2)
    f.close()

  def Dump(self, fname, delimiter=' '):
    with open(fname, 'w') as csv_file:
      for row in self.data:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerow(row)



class RespMatrix():

  def __init__(self, data, ebins, pebins):
    self.data = data
    self.ebins = ebins
    self.pebins = pebins

  def Plot(self, fname, **kwargs):
    self.Print(fname, **kwargs)

  def Print(self, fname, pecrop=None):
    # 'pecrop' must be a tuple or None
    if pecrop == None:
      pe_min, pe_max = self.pebins[0], self.pebins[-1]
    x,y,val = [],[],[]
    for index, v in np.ndenumerate(self.data):
      x.append(self.ebins[index[0]])
      y.append(self.pebins[index[1]])
      val.append(v)
    fig, ax = plt.subplots()
    counts, xedges, yedges, im = plt.hist2d(x,y,bins=[self.ebins, self.pebins], weights=val, cmap=plt.get_cmap("jet"))
    plt.colorbar(im, ax=ax)
    plt.xlabel("True energy (MeV)")
    plt.ylabel("Visual energy (MeV)")
    plt.ylim(pecrop)
    plt.savefig(fname)
    plt.close()

  def Save(self, fname):
    f = open(fname, "wb")
    pickle.dump(self, f, 2)
    f.close()

  def Dump(self, fname, delimiter=' '):
    with open(fname, 'w') as csv_file:
      for row in self.data:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerow(row)


def SaveObject(obj, fname):
  with open(fname, "wb") as f:
    pickle.dump(obj, f, 2)

def LoadObject(fname):
  with open(fname, "rb") as f:
    return pickle.load(f)


def LoadCovMatrix(fname):
  f = open(fname, "rb")
  m = pickle.load(f)
  f.close()
  return m


def LoadRespMatrix(fname):
  f = open(fname, "rb")
  m = pickle.load(f)
  f.close()
  return m
    
def CalcRespMatrix_abc(a, b, c, ebins, pebins, escale=1., eshift=0., norm_mode="per_MeV", verbose=True):
    evis = 0.5*(ebins[1:] + ebins[:-1]) + eshift
    rel_res = np.sqrt(a**2 / evis + b**2 + c**2 / evis**2)  # in fraction

    loc = escale * evis                     # shape (N_e,)
    scale = escale * rel_res * evis         # shape (N_e,)
    
    pe = 0.5*(pebins[1:] + pebins[:-1])    # shape (N_pe,)
    pe_bin_width = pebins[1] - pebins[0]

    # Vectorized PDF calculation using broadcasting
    # pe shape -> (1, N_pe), loc & scale -> (N_e, 1)
    pe_grid = pe[np.newaxis, :]             # shape (1, N_pe)
    loc_grid = loc[:, np.newaxis]           # shape (N_e, 1)
    scale_grid = scale[:, np.newaxis]       # shape (N_e, 1)

    rm_data = (1 / (np.sqrt(2*np.pi) * scale_grid)) * \
              np.exp(-0.5 * ((pe_grid - loc_grid)/scale_grid)**2)
    
    # multiply by bin width
    rm_data *= pe_bin_width

    return RespMatrix(rm_data, ebins=ebins, pebins=pebins)
# def CalcRespMatrix_abc(a, b, c, ebins, pebins, escale=1., eshift=0., norm_mode="per_MeV", verbose=True):
#   evis = 0.5*(ebins[1:]+ebins[:-1]) + eshift


#   rel_res = 100.*(np.sqrt(a*a/evis + b*b + c*c/evis/evis))
#   if type(rel_res) == float:
#     rel_res = rel_res*np.sqrt(evis)

#   pe = 0.5*(pebins[1:]+pebins[:-1])
#   pe_bin_width = pebins[1]-pebins[0] # assuming equally sized bins

#   rm_data = np.zeros(shape=(len(ebins)-1,len(pebins)-1))
#   #gaussian_distribution = [*map(lambda a, b: sp.stats.norm(loc=escale*a, scale=escale*b/100.*a), evis, rel_res)]
#   #rm_data = [*map(lambda a: a.pdf(pe)*pe_bin_width, gaussian_distribution)]

#   gaussian_distribution = []
#   for i in range(len(evis)):
#     loc = escale * evis[i]
#     scale = escale * rel_res[i] / 100. * evis[i]
#     gaussian_distribution.append(sp.stats.norm(loc=loc, scale=scale))

#   for i in range(len(evis)):
#       rm_data[i,:] = gaussian_distribution[i].pdf(pe) * pe_bin_width

#   return RespMatrix(rm_data, ebins=ebins, pebins=pebins)
# import numpy as np
# import scipy.stats as sp

# def CalcRespMatrix_abc(a, b, c, ebins, pebins, escale=1., eshift=0.,
#                        norm_mode="per_MeV", verbose=True, method="cdf"):
#     """
#     Construct detector response matrix with Gaussian energy resolution.
    
#     Parameters
#     ----------
#     a, b, c : floats
#         Resolution parameters (a/sqrt(E) ⊕ b ⊕ c/E).
#     ebins : array
#         True energy bin edges.
#     pebins : array
#         Measured energy (PE) bin edges (can be nonuniform).
#     escale : float
#         Energy scale factor.
#     eshift : float
#         Energy shift (MeV).
#     norm_mode : str
#         If "per_MeV", normalize by true bin width.
#     method : str
#         "cdf" (exact, recommended) or "pdf" (approx).
#     """

#     # True visible energy centers
#     evis = 0.5*(ebins[1:] + ebins[:-1]) + eshift

#     # Relative resolution [%]
#     rel_res = 100. * np.sqrt(a*a/evis + b*b + c*c/(evis*evis))
#     if np.isscalar(rel_res):
#         rel_res = rel_res * np.sqrt(evis)

#     nE = len(ebins) - 1
#     nPE = len(pebins) - 1
#     rm_data = np.zeros((nE, nPE))

#     if method == "cdf":
#         # exact probability per bin using CDF differences
#         for i, e in enumerate(evis):
#             loc = escale * e
#             scale = escale * rel_res[i] / 100. * e
#             dist = sp.stats.norm(loc=loc, scale=scale)
#             rm_data[i, :] = dist.cdf(pebins[1:]) - dist.cdf(pebins[:-1])

#     elif method == "pdf":
#         # approximate: pdf(center)*bin_width (works if bins are small vs sigma)
#         pe_centers = 0.5*(pebins[1:] + pebins[:-1])
#         pe_widths = np.diff(pebins)
#         for i, e in enumerate(evis):
#             loc = escale * e
#             scale = escale * rel_res[i] / 100. * e
#             dist = sp.stats.norm(loc=loc, scale=scale)
#             rm_data[i, :] = dist.pdf(pe_centers) * pe_widths
#     else:
#         raise ValueError(f"Unknown method={method}")

#     # Optional normalization
#     if norm_mode == "per_MeV":
#         bin_widths = np.diff(ebins)
#         rm_data = rm_data / bin_widths[:, None]

#     return RespMatrix(rm_data, ebins=ebins, pebins=pebins)


def CalcRespMatrix_abc_flu(a, b, c, unc, ebins, pebins, escale=1., eshift=0., norm_mode='per_MeV', sample_size=100):
  print(" # Fluctuating (a,b,c) parameters...")
  a_err, b_err, c_err = unc
  a_flu = np.random.normal(loc=a, scale=a_err, size=sample_size)
  b_flu = np.random.normal(loc=b, scale=b_err, size=sample_size)
  c_flu = np.random.normal(loc=c, scale=c_err, size=sample_size)

  CalcRM = lambda a,b,c: CalcRespMatrix_abc( a, b, c, ebins, pebins, escale=escale,  eshift=eshift, norm_mode=norm_mode, verbose=False)
  spectra = Parallel(n_jobs=-1)(delayed(CalcRM)(a_flu[i], b_flu[i], c_flu[i]) for i in range(len(a_flu)))
  gc.collect()
  return spectra

def Rebin2D(data, xedges, yedges, new_xedges, new_yedges):
  rebinned_data = np.zeros((len(new_xedges) - 1, len(new_yedges) - 1))
  for i in range(len(new_xedges) - 1):
    for j in range(len(new_yedges) - 1):
      mask_x = (xedges[:-1] >= new_xedges[i]) & (xedges[:-1] < new_xedges[i + 1])
      mask_y = (yedges[:-1] >= new_yedges[j]) & (yedges[:-1] < new_yedges[j + 1])
      rebinned_data[i, j] = data[mask_x][:, mask_y].sum()
  return rebinned_data

def CalcEnergyLeak(rootfile, histname, ebins, pebins):
  with uproot.open(rootfile) as file:
    hist = file[histname].to_numpy()
  data, ebins_ori, pebins_ori = hist

  ebin_width_ori = np.diff(ebins_ori)
  pebin_width_ori = np.diff(pebins_ori)
  ebin_width = np.diff(ebins)
  pebin_width = np.diff(pebins)

  if (ebin_width_ori[0] < ebin_width[0]) or (pebin_width_ori[0] < pebin_width[0]):
    data = Rebin2D(data, ebins_ori, pebins_ori, ebins, pebins)

  respMat = RespMatrix(data, ebins, pebins)
  row_sums = respMat.data.sum(axis=1, keepdims=True)
  row_sums[row_sums == 0] = 1

  respMat.data = respMat.data / row_sums  # Normalize to probabilities
  respMat.data = np.nan_to_num(respMat.data, nan=0.0)

  return respMat

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

def GetSpectrumFromROOT(fname, hname, toy=False, xlabel="Energy (MeV)", scale=1., eshift=0):
  rf = uproot.open(fname)
  hist = rf[hname]
  bin_cont = hist.values()*scale
  #if toy: bin_cont[bin_cont == 0] = 0.1
  bins = hist.axis().edges()+eshift
  return Spectrum(bin_cont, bins, xlabel=xlabel)

# def Chi2(cm, tot_obs, tot_exp, rea_obs, rea_exp, unc=' ', stat_meth=' ', pulls=None, pull_unc=None, add_logdet=False):
#     penalty = 0.
#     if pulls and pull_unc:
#         for p, u in zip(pulls, pull_unc):
#             penalty += (p/u)**2

#     # --- choose stat covariance
#     if stat_meth == "NorP":
#         stat_cov_matrix = rea_obs.GetStatCovMatrix().data
#     else:
#         stat_cov_matrix = rea_obs.GetCNPStatCovMatrix(rea_exp).data

#     # --- choose total covariance
#     if unc == "stat":
#         cov_matrix = stat_cov_matrix
#     else:
#         cov_matrix = stat_cov_matrix + cm.data
#     obs = tot_obs.bin_cont
#     exp = tot_exp.bin_cont 
    
#     alpha = np.zeros_like(exp)  # pull values (could later be fit params)
#     exp = exp * (1 + alpha)     # distorted spectrum
#     penalty += np.sum((alpha)**2)
    

#     valid = (obs > 0) & (exp > 0)
#     diff = obs - exp
#     cov_matrix = cov_matrix[np.ix_(valid, valid)]
#     diff = diff[valid]
   

#     # --- quadratic form
#     chi2_term = diff.T @ np.linalg.solve(cov_matrix, diff)

#     # --- logdet term (optional)
#     logdet_term = 0.0
#     if add_logdet:
#         # use slogdet for numerical stability
#         sign, logdet = np.linalg.slogdet(cov_matrix)
#         if sign <= 0:
#             raise ValueError("Covariance matrix not positive definite")
#         logdet_term = logdet

#     chi2 = chi2_term + logdet_term + penalty
#     return chi2

#@njit
# def chi2_val(obs, exp, cov_matrix, penalty):
#   diff = obs - exp
#   chi2 = diff.T @ np.linalg.solve(cov_matrix, diff) + penalty

#   return chi2

# def Chi2(cm, tot_obs, tot_exp, rea_obs, rea_exp, unc=' ', stat_meth=' ', pulls=None, pull_unc=None):
#   penalty = 0.
#   if pulls and pull_unc:
#       for p, u in zip(pulls, pull_unc): penalty += (p/u)**2
#   chi2 = 0.0
#   if stat_meth == "NorP":
#       stat_cov_matrix = rea_obs.GetStatCovMatrix().data
#   else:
#       stat_cov_matrix = rea_obs.GetCNPStatCovMatrix(rea_exp).data
#   if unc == "stat":
#       cov_matrix = stat_cov_matrix
#   elif unc == "no stat":
#       cov_matrix = cm.data
#   else:
#       cov_matrix = stat_cov_matrix + cm.data
  
#   obs = tot_obs.bin_cont
#   exp = tot_exp.bin_cont
#   valid = (obs > 0) &(exp > 0)
  
#   cov_matrix = cov_matrix[np.ix_(valid, valid)]
#   #chi2 = diff.T @ np.linalg.solve(cov_matrix, diff) + penalty

#   chi2 = chi2_val(obs, exp,cov_matrix, penalty)
#   return chi2

def chi2_val(obs, exp, cov_matrix, penalty=0.0):
    """Compute chi2 = (obs - exp)^T V^-1 (obs - exp) + penalty."""
    diff = obs - exp
    chi2 = diff.T @ np.linalg.solve(cov_matrix, diff) + penalty
    return chi2


def Chi2(cm, tot_obs, tot_exp, rea_obs, rea_exp, unc=' ', stat_meth=' ', pulls=None, pull_unc=None, include_logV=False):
    penalty = 0.0
    if pulls and pull_unc:
        for p, u in zip(pulls, pull_unc):
            penalty += (p / u) ** 2

    if stat_meth == "Neyman":
        stat_cov_matrix = rea_obs.GetStatCovMatrix().data
    elif stat_meth == "Pearson" or include_logV:
        stat_cov_matrix = rea_exp.GetStatCovMatrix().data
    else:
        stat_cov_matrix = rea_obs.GetCNPStatCovMatrix(rea_exp).data

    if unc == 'stat':
        cov_matrix = stat_cov_matrix 
    else:
        cov_matrix = stat_cov_matrix + cm.data

    obs = tot_obs.bin_cont
    exp = tot_exp.bin_cont
    valid = (obs > 0) & (exp > 0)
    cov_matrix = cov_matrix[np.ix_(valid, valid)]
    
    obs = obs[valid]
    exp = exp[valid]


    logdet = 0
    if include_logV:
        sign, logdet = np.linalg.slogdet(cov_matrix)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix not positive definite")

    chi2 = chi2_val(obs, exp, cov_matrix, penalty) 

    return chi2, chi2 + logdet
    
def Combined_Chi2(cm1, cm2, corr_cm, tot_obs1, tot_exp1, tot_obs2, tot_exp2, rea_obs1, rea_exp1, rea_obs2, rea_exp2, unc=' ', stat_meth=' ', pulls=None, pull_unc=None, nd="TAO"):
  penalty = 0.0
  if pulls and pull_unc:
      for p, u in zip(pulls, pull_unc): penalty += (p / u) ** 2


  if stat_meth == "NorP":
      stat_cm1 = rea_obs1.GetStatCovMatrix()
      stat_cm2 = rea_obs2.GetStatCovMatrix()
  else:
      stat_cm1 = rea_obs1.GetCNPStatCovMatrix(rea_exp1)
      stat_cm2 = rea_obs2.GetCNPStatCovMatrix(rea_exp2)

  stat_cm1_mod = stat_cm1.Extend(CovMatrix(data=np.zeros((len(tot_obs1.bins)-1, len(tot_obs1.bins)-1)), bins=tot_obs1.bins))
  cm_stat_nd_z = CovMatrix(data=np.zeros((len(tot_obs2.bins)-1, len(tot_obs2.bins)-1)), bins=tot_obs2.bins)
  stat_cm2_mod = cm_stat_nd_z.Extend(stat_cm2)

  cm1_mod = cm1.Extend(CovMatrix(data=np.zeros((len(tot_obs1.bins)-1, len(tot_obs1.bins)-1)), bins=tot_obs1.bins))
  cm_nd_z = CovMatrix(data=np.zeros((len(tot_obs2.bins)-1, len(tot_obs2.bins)-1)), bins=tot_obs2.bins)
  cm2_mod = cm_nd_z.Extend(cm2)

  if unc == "stat":
      total_cov_matrix = stat_cm1_mod.data + stat_cm2_mod.data

  else:
      total_cov_matrix = stat_cm1_mod.data +  stat_cm2_mod.data + cm1_mod.data + cm2_mod.data + corr_cm.data
  

  diff1 = tot_obs1.bin_cont - tot_exp1.bin_cont
  diff2 = tot_obs2.bin_cont - tot_exp2.bin_cont
  
  obs_total = np.concatenate((tot_obs1.bin_cont, tot_obs2.bin_cont))
  exp_total = np.concatenate((tot_exp1.bin_cont, tot_exp2.bin_cont))

  valid_mask = (obs_total > 0) & (exp_total > 0)

  diff_total = np.concatenate((diff1, diff2))
  total_cov_matrix = total_cov_matrix[np.ix_(valid, valid)]
  #c_factor = cho_factor(cov_matrix)
  #chi2_term = cho_solve(c_factor, diff[valid])
  #chi2 = np.einsum('i,i->', diff[valid], chi2_term) + penalty 
  chi2 = diff_total[valid].T @ np.linalg.solve(total_cov_matrix, diff_total[valid]) + penalty
    
  return chi2

def Chi2DYB(rel_cm, obs_sp, exp_sp):
    obs = obs_sp.bin_cont
    exp = exp_sp.bin_cont
    diff = (obs/exp) - 1
    chi2 =  diff.T @ np.linalg.solve(rel_cm.data, diff)
    return chi2


def GetRelCM(V, d, bins=None, axis_label=""):
    if V.data.shape[0] != len(d.bin_cont):
        raise ValueError("Length of d must match size of covariance matrix")

    obs = d.bin_cont
    denom = np.outer(obs, obs)
    Q = V.data/denom
    
    return CovMatrix(Q, bins=d.bins, axis_label=axis_label)
# def GetFluctuatedSpectraNL(ensp_nonl, ensp_nl_nom, ensp_nl_pull_curve, sample_size=10000):
#   nc = sp.interpolate.interp1d(ensp_nl_nom.GetBinCenters(), ensp_nl_nom.bin_cont,
#                                          kind='slinear', bounds_error=False,
#                                          fill_value=(ensp_nl_nom.bin_cont[0],
#                                                      ensp_nl_nom.bin_cont[-1]))
#   pc = sp.interpolate.interp1d(ensp_nl_pull_curve.GetBinCenters(), ensp_nl_pull_curve.bin_cont,
#                                        kind='slinear', bounds_error=False,
#                                        fill_value=(ensp_nl_pull_curve.bin_cont[0],
#                                                    ensp_nl_pull_curve.bin_cont[-1]) )
#   weights = np.random.normal(loc=0., scale=1., size=sample_size)
#   fNL_flu = [lambda e: nc(e) + w*(pc(e)-nc(e)) for w in weights]
# #  return [ensp_nonl.GetWithModifiedEnergy(mode='spline', spline=f) for f in fNL_flu]
#   return ensp_nonl.GetWithModifiedEnergyBulk(mode='spline', splines=fNL_flu)

#def get_new_nonl(ensp_nl_nom, ensp_nl_pull_curve, sample_size, w):
#  new_nonl =  Spectrum(bins = ensp_nl_nom.bins, bin_cont=np.zeros(len(ensp_nl_nom.bin_cont)))
#  new_nonl.bin_cont =ensp_nl_nom.bin_cont + w*(ensp_nl_pull_curve.bin_cont - ensp_nl_nom.bin_cont)
#  return new_nonl
#
#def GetFluNL_old(ensp_nonl, ensp_nl_nom, ensp_nl_pull_curve, sample_size = 10000):
#  weights = np.random.normal(loc=0., scale=1., size=sample_size)
#  new_nonl_spectra = Parallel(n_jobs=-1)(delayed(get_new_nonl)(ensp_nl_nom, ensp_nl_pull_curve, sample_size,  w) for w in weights)
#  output = ensp_nonl.GetWithModifiedEnergyBulk(mode='spectrum', spectra = new_nonl_spectra)
#  return output
#

def make_nonuniform_bins(bins_nonuniform):
    edges = []
    for start, end, n in bins_nonuniform:
        part = np.linspace(start, end, n + 1)
        if edges:  # avoid duplicating the shared boundary
            part = part[1:]
        edges.extend(part)
    return np.array(edges)
import numpy as np

# def expand_nuisances_to_fine(coarse_edges, fine_edges, nuisance_coarse):
#     fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
#     interp_weights = np.interp(
#         fine_centers, 
#         coarse_edges, 
#         nuisance_coarse,
#         left=0.0,
#         right=0.0
#     )

#     return interp_weights

def expand_nuisances_to_fine(coarse_edges, fine_edges, nuisance_coarse):
    fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    
    # Shift to multiplicative space (0 becomes 1, 0.1 becomes 1.1)
    multiplicative_coarse = 1.0 + nuisance_coarse
    
    # Interpolate in log-space
    interp_log = np.interp(
        fine_centers, 
        coarse_edges, 
        np.log(multiplicative_coarse)
    )
    
    # Transform back and subtract the 1 to return to "shift" space
    return np.exp(interp_log) - 1.0

def GetSpectrumFromDAT(filename, scale_U235=1.0, scale_U238=1.0, scale_Pu239=1.0, scale_Pu241=1.0):
    """
    Read neutrino flux data from a .dat file with columns:
    E_nu [keV], flux_235, flux_239, flux_241, flux_238 (#nu / fission / keV)
    Convert energy to MeV and return Spectrum (same class as GetSpectrumFromROOT).
    """
    data = np.loadtxt(filename)

    # Columns: E(keV), 235, 239, 241, 238
    E_keV = data[:, 0]
    E_MeV = E_keV / 1000.0  # convert keV → MeV
    E_edges = centers_to_edges(E_MeV)
    keV_to_MeV = 1000.0

    total_flux = keV_to_MeV * (
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
def centers_to_edges(centers):
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # extrapolate first and last edges
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

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

    prefactor = (1+eff)*args.detector_efficiency * args.veto * args.Np / (4 * np.pi)

    # Compute normalization denominator (total weighted fission energy)
    norm = np.sum(alpha_arr * efission_arr)

    # Handle zero-baseline case safely
    if np.any(L_arr == 0):
        geo_term = np.sum(Pth_arr)  # no 1/L^2 factor
    else:
        geo_term = np.sum(Pth_arr / (L_arr * L_arr))

    extrafactor = prefactor * geo_term / norm
    return extrafactor

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

def rebin_matrix(mat, old_row_bins, old_col_bins, new_row_bins, new_col_bins):
    """
    Rebin a 2D matrix along both rows and columns.

    Parameters
    ----------
    mat : np.ndarray
        Original matrix of shape (n_old_rows, n_old_cols)
    old_row_bins : array-like
        Original row (true-energy) bin edges
    old_col_bins : array-like
        Original column (visible-energy) bin edges
    new_row_bins : array-like
        Desired new row bin edges
    new_col_bins : array-like
        Desired new column bin edges

    Returns
    -------
    mat_new : np.ndarray
        Re-binned matrix of shape (len(new_row_bins)-1, len(new_col_bins)-1)
    """
    # Step 1: Rebin rows (true-energy axis)
    old_row_centers = 0.5 * (old_row_bins[:-1] + old_row_bins[1:])
    new_row_centers = 0.5 * (new_row_bins[:-1] + new_row_bins[1:])
    temp_mat = np.zeros((len(new_row_centers), mat.shape[1]))

    for j in range(mat.shape[1]):  # iterate over columns
        f_row = interpolate.interp1d(old_row_centers, mat[:, j], kind='linear', fill_value='extrapolate')
        temp_mat[:, j] = f_row(new_row_centers)

    # Step 2: Rebin columns (visible-energy axis)
    old_col_centers = 0.5 * (old_col_bins[:-1] + old_col_bins[1:])
    new_col_centers = 0.5 * (new_col_bins[:-1] + new_col_bins[1:])
    mat_new = np.zeros((len(new_row_centers), len(new_col_centers)))

    for i in range(temp_mat.shape[0]):  # iterate over rows
        f_col = interpolate.interp1d(old_col_centers, temp_mat[i, :], kind='linear', fill_value='extrapolate')
        mat_new[i, :] = f_col(new_col_centers)

    return mat_new

