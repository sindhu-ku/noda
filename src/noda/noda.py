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
#np.set_printoptions(threshold=sys.maxsize)
#global settings:

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
      if x<0: self.bin_cont[i] = 0

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
        return len(self.bins) - 1
    left = 0
    right = len(self.bins) - 1
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
      core_shares = [1./len(L)]*len(L)
    else:
      core_shares = [p/(l*l) for p, l in zip(core_powers, L)]
      core_shares = [p/sum(core_shares) for p in core_shares]
    eshift = 0
    if ene_mode == 'vis':
      eshift = -0.511+1.293
    #print(sin2_th12, sin2_th13, dm2_21, dm2_32)
    for l, share in zip(L, core_shares):
      if osc_formula == nuosc.AntiNueSurvProb:
        P = lambda evis: osc_formula(evis+eshift, l,
                                     sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                                     dm2_21=dm2_21, dm2_31=dm2_31, dm2_32=dm2_32,
                                     me_rho=me_rho)
      elif osc_formula == nuosc.AntiNueSurvProbYB:
        P = lambda evis: osc_formula(evis+eshift, l,
                                     sin2_th12=sin2_th12, sin2_th13=sin2_th13,
                                     dm2_21=dm2_21, dm2_ee=dm2_ee,
                                     me_rho=me_rho)
      val = []
      for b1, b2, x in zip(self.bins[:-1], self.bins[1:], self.bin_cont):
        if points_per_bin == 1:
          val.append(x*P(0.5*(b1+b2))*share)
        else:
          v = 0
          for i in range(points_per_bin):
            w = (b2-b1)/points_per_bin
            v += P(b1+(i+0.5)*w)/points_per_bin
          val.append(x*v*share)
      spec += Spectrum(val, bins=self.bins, xlabel=self.xlabel)
    return spec

  def ApplyDetResp(self, respMat, pecrop=None):
    if pecrop == None:
      return Spectrum(self.bin_cont.dot(respMat.data), bins=respMat.pebins)
    pebins_full = respMat.pebins
    pebw = (pebins_full[-1]-pebins_full[0])/(len(pebins_full)-1)
    peb1 = int(np.around((pecrop[0]-pebins_full[0])/pebw))
    peb2 = int(np.around((pecrop[1]-pebins_full[0])/pebw))
    pebins_crop = pebins_full[peb1:peb2+1]
    return Spectrum(self.bin_cont.dot(respMat.data[peb1:peb2]), bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (MeV)")

  def GetWithShiftedEnergy(self, eshift):
    Enu = self.bins
    Epos = Enu + eshift
    return Spectrum(self.bin_cont, bins=self.bins+eshift).Rebin(self.bins)
  #
  # def GetWithPositronEnergy(self):
  #   # #print("entering pos E")
  #   Enu = self.bins
  #   Mn = 939.56536 #MeV
  #   Mp = 938.27203 #MeV
  #   Me = 0.51099893 #MeV
  #   Deltanp = Mn - Mp
  #   Mdiff = -Enu + Deltanp + (Deltanp*Deltanp - Me*Me)/(2.0*Mp)
  #   #print ("Any Mdiff is < 0", (Mdiff<0.).any())
  #   Epos = (-Mn + np.sqrt(Mn*Mn - 4.0*Mp*Mdiff))/2.0
  #   #Epos = (Enu - Deltanp) / 2.0 + np.sqrt((Enu - Deltanp)**2 - Me**2) / 2.0
  #
  #   Evis = Epos + 0.511
  #   return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)
  def GetWithPositronEnergy(self):
    # Constants
    Enu = self.bins  # Antineutrino energy bins
    Epos = np.zeros_like(Enu)
    file = ROOT.TFile("data/JUNOInputs2022_05_08.root")
    Epositron_Enu_cos_StrumiaVissani = file.Get("Epositron_Enu_cos_StrumiaVissani")
    for i, energy in enumerate(Enu):
        Epos_temp =0.
        for cos_theta in range(-1, 1, 100):  # Random angle for each energy
            Epos_temp += -1.*Epositron_Enu_cos_StrumiaVissani.Eval(energy, cos_theta)
        Epos[i] = -1e2*Epos_temp/100.
    # Visible energy includes the positron mass contribution
    Evis = Epos + 0.511  # Adding the rest mass of the positron
    return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)

  def ShiftEnergy(self, eshift):
    old_bins = self.bins
    self.bins = self.bins + eshift
    self.Rebin(old_bins)

  def GetWithModifiedEnergy(self, **kwargs):
    s_new = self.Copy()
    s_new.ModifyEnergy(**kwargs)
    return s_new

  def ModifyEnergy(self, mode="lin", scale=1., spline=None, spectrum=None):
    if mode == 'lin':
      bins_new = self.bins*scale
    elif mode == 'spline': # TODO vectorize it
      bins_new = []
      for e in self.bins[:]:
        if e==0.: e=1.e-10 # to avoid the divide-by-zero problem
        e *= scale*spline(e)
        bins_new.append(e)
    elif mode == "spectrum":
      bins_new = self.bins*np.array([spectrum.GetAtEnergy(e) for e in self.bins])
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
    self.bin_cont = values

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
      data[i][i] = 3./((1./x) + (2./exp.bin_cont[i]))
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetB2BCovMatrix(self, sigma):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,x in enumerate(self.bin_cont):
      data[i][i] = x*x*sigma*sigma
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetVariedB2BCovMatrix(self, sigmas):
    nb = len(self.bin_cont)
    data = np.zeros(shape=(nb,nb))
    for i,(x,sigma) in enumerate(zip(self.bin_cont, sigmas)):
      data[i][i] = x*x*sigma*sigma
    return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

 # def GetVariedB2BCovMatrix(self, b2b_spectra):
 #   sigmas = np.zeros(len(self.bin_cont))
 #   for ene, sigma in zip(b2b_spectra.bins, b2b_spectra.bin_cont):
 #     ibin = self.FindBin(ene)
 #     if ibin >= 0 and ibin < len(sigmas):
 #       sigmas[ibin] = 5#sigma
 #   nb = len(self.bin_cont)
 #   data = np.zeros(shape=(nb,nb))
 #   for i,(x,sigma) in enumerate(zip(self.bin_cont, sigmas)):
##      print(x, sigma)
 #     data[i][i] = x*x*sigma*sigma
 #     print(data[i][i])
 #   return CovMatrix(data, bins=self.bins, axis_label=self.xlabel)

  def GetVariedB2BCovMatrixFromROOT(self, fname, hname):
    rf = uproot.open(fname)
    hist = rf[hname]
    edges = hist.axis().edges()
    sigmas = np.zeros(len(self.bin_cont))
    for ene, sigma in zip(hist.axis().edges()[:-1], hist.values()):
      ibin = self.FindBin(ene)
      if ibin >= 0 and ibin < len(sigmas):
        sigmas[ibin] = sigma
    if len(sigmas) != len(self.bin_cont):
      print(f" ### WARNING: number of b2b bins is {len(sigmas)}; number of spectrum bins is {len(self.bin_cont)}.")
    if edges[1]-edges[0] != self.bins[1]-self.bins[0]:
      print(f" ### WARNING: bin width for b2b is {edges[1]-edges[0]}; spectrum bin width is {self.bins[1]-self.bins[0]}.")
      #print(f"     Doing rebinning for b2b uncertainties")
    return self.GetVariedB2BCovMatrix(sigmas)

  def SetXlabel(self, label):
    self.xlabel = label

  def GetXlabel(self):
    return self.xlabel

  def SetYlabel(self, label):
    self.ylabel = label

  def Plot(self, fname, **kwargs):
    self.Print(fname, **kwargs)

  def Print(self, fname, xlabel=None, ylabel=None, leg_labels=None, colors=None, extra_spectra=[], xmin=None, xmax=None, log_scale=False, ymin=None, ymax=None, yinterval=None):
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
    plt.figure(figsize=(10,6))
    plt.hist(self.bins[:-1], weights=self.bin_cont, bins=self.bins, fill=False, histtype='step', linewidth=1, color=colors[0], label=leg_labels[0], log=log_scale)
    for i,s in enumerate(extra_spectra):
      plt.hist(s.bins[:-1], weights=s.bin_cont, bins=s.bins, fill=False, histtype='step', linewidth=1, color=colors[i+1], label=leg_labels[i+1], log=log_scale)
    plt.hist(self.bins[:-1], weights=self.bin_cont, bins=self.bins, fill=False, histtype='step', linewidth=1, color=colors[0])
    if xmin is not None: plt.xlim(left=xmin)
    if xmax is not None: plt.xlim(right=xmax)
    if ymin is not None: plt.ylim(bottom=ymin)
    if ymax is not None: plt.ylim(top=ymax)
    if yinterval: plt.yticks(np.arange(ymin, ymax, yinterval))
    plt.xlabel(self.xlabel)
    plt.ylabel(self.ylabel)
    if all(leg_labels): plt.legend()
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

  def Scale(self, scale_factor):
    sf2 = scale_factor*scale_factor
    sf2_inv = 1./sf2
    self.data *= sf2
    if self.IsInvertible():
      self.data_inv *= sf2_inv

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
  evis = 0.5*(ebins[1:]+ebins[:-1]) + eshift


  rel_res = 100.*(np.sqrt(a*a/evis + b*b + c*c/evis/evis))
  if type(rel_res) == float:
    rel_res = rel_res*np.sqrt(evis)

  pe = 0.5*(pebins[1:]+pebins[:-1])
  pe_bin_width = pebins[1]-pebins[0] # assuming equally sized bins

  rm_data = np.zeros(shape=(len(ebins)-1,len(pebins)-1))
  #gaussian_distribution = [*map(lambda a, b: sp.stats.norm(loc=escale*a, scale=escale*b/100.*a), evis, rel_res)]
  #rm_data = [*map(lambda a: a.pdf(pe)*pe_bin_width, gaussian_distribution)]

  gaussian_distribution = []
  for i in range(len(evis)):
    loc = escale * evis[i]
    scale = escale * rel_res[i] / 100. * evis[i]
    gaussian_distribution.append(sp.stats.norm(loc=loc, scale=scale))

  for i in range(len(evis)):
      rm_data[i,:] = gaussian_distribution[i].pdf(pe) * pe_bin_width

  return RespMatrix(rm_data, ebins=ebins, pebins=pebins)


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

def GetSpectrumFromROOT(fname, hname, xlabel="Energy (MeV)", scale=1., eshift=0):
  rf = uproot.open(fname)
  hist = rf[hname]
  bin_cont = hist.values()*scale
  bins = hist.axis().edges()+eshift
  return Spectrum(bin_cont, bins, xlabel=xlabel)

def Chi2(cm, s1, s2, unc=' ', stat_meth=' '):
  diff = s1.bin_cont - s2.bin_cont
  chi2 = 0.0
  norp_stat_cm = s1.GetStatCovMatrix()
  if stat_meth == "NorP":
    if unc == "stat": chi2 = diff.T @ norp_stat_cm.data_inv @ diff
    else: chi2 = diff.T @ np.linalg.inv(norp_stat_cm.data + cm.data) @ diff
  else:
    cnp_stat_cm = s1.GetCNPStatCovMatrix(s2)
    if unc == "stat": chi2 = diff.T @ cnp_stat_cm.data_inv @ diff
    else: chi2 = diff.T @ np.linalg.inv(cnp_stat_cm.data + cm.data) @ diff
  return chi2

def Chi2_p(cm, s1, s2, unc=' ', stat_meth=' ', pulls=[], pull_unc=[]):
  penalty = 0.
  for p, u in zip(pulls, pull_unc):
    penalty += (p/u)**2
  diff = s1.bin_cont - s2.bin_cont
  chi2 = 0.0
  norp_stat_cm = s1.GetStatCovMatrix()
  if stat_meth == "NorP":
    if unc == "stat": chi2 = diff.T @ norp_stat_cm.data_inv @ diff + penalty
    else: chi2 = diff.T @ np.linalg.inv(norp_stat_cm.data + cm.data) @ diff + penalty
  else:
    cnp_stat_cm = s1.GetCNPStatCovMatrix(s2)
    if unc == "stat": chi2 = diff.T @ cnp_stat_cm.data_inv @ diff + penalty
    else: chi2 = diff.T @ np.linalg.inv(cnp_stat_cm.data + cm.data) @ diff + penalty
  return chi2

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
