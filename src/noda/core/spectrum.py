"""Spectrum class: binned spectra, oscillation, energy models, plotting.

The oscillation formula switch (SetOscFormula / osc_formula) lives here
because Spectrum reads the module-level osc_formula.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import csv
import sys
from ..physics import nuosc as nuosc
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import uproot
import ROOT
from array import array
from ..physics.nl_geant4 import Geant4Model
from ..physics.constants import *
from .root_utils import force_vector_streamers
from .binning import _build_nonuniform_bins, _build_rebin_matrix
from .covmatrix import CovMatrix


G4model = Geant4Model(particle="e+")
G4model_gamma = Geant4Model(particle="gamma")
_PE_2D_CACHE = {}
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
    # FindBin returns -1 below range, which would wrap to the LAST bin
    # (e.g. NL(12 MeV) instead of NL(threshold) for the TAO leakage tail)
    i = self.FindBin(ene)
    if i < 0: i = 0
    return self.bin_cont[i]

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
    # mode: 'per_bin', 'per_meV'
    i1 = 0
    i2 = len(self.bin_cont)
    if left_edge != None:  i1 = self.FindBin(left_edge)
    if right_edge != None: i2 = self.FindBin(right_edge)
    if i1<0: i1=0
    if mode == 'per_bin':
      return sum(self.bin_cont[i1:i2])
    if mode == 'per_meV':
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
    # mode options: "per_meV", "per_bin"
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
    new_bins = np.ascontiguousarray(new_bins, dtype=float)
    W = _build_rebin_matrix(self.bins, new_bins)
    self.bin_cont = W @ self.bin_cont
    self.bins = new_bins

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
    new_bins   = _build_nonuniform_bins(bins_nonuniform)
    W          = _build_rebin_matrix(self.bins, new_bins)
    new_bin_cont = W @ self.bin_cont
    self.bins     = new_bins
    self.bin_cont = new_bin_cont
    return Spectrum(bin_cont=new_bin_cont, bins=new_bins,
                    xlabel=self.xlabel, ylabel=self.ylabel)

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
      share_sum = np.sum(core_shares)
      # A zero-power core (e.g. a reactor off for the whole phase) has zero
      # total share; leave the shares at zero so it contributes a zero
      # spectrum instead of dividing 0/0 -> NaN.
      core_shares = core_shares / share_sum if share_sum > 0 else core_shares

    eshift = 0
    if ene_mode == 'vis':
      eshift = -0.511+1.293

    # --- VECTORIZED MATH STARTS HERE ---

    # 1. Prepare bin arrays
    b1 = self.bins[:-1]
    b2 = self.bins[1:]

    # 2. Setup the evaluation energies depending on points_per_bin.
    #    A caller that leaves the default (1) gets the detector's config value
    #    `osc_points_per_bin` (juno_inputs); averaging P over sub-points of each
    #    true-energy bin removes the far-core (DYB 215 km, FC 411 km) aliasing
    #    that bin-centre sampling causes at 20 keV.
    if points_per_bin == 1:
      points_per_bin = int(getattr(args, 'osc_points_per_bin', 1) or 1)
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
        return Spectrum(self.bin_cont.dot(data), bins=respMat.pebins,
                    xlabel=r"$E_{\rm vis}$ (meV)")
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

    return Spectrum(smeared_cont, bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (meV)")

  # def ApplyDetResp(self, respMat, pecrop=None, norm=False):
  #   data = respMat.data.copy()

  #   if norm:
  #       col_sums = data.sum(axis=0, keepdims=True)
  #       col_sums[col_sums == 0] = 1.0
  #       data /= col_sums
  #       #respMat = RespMatrix(data=data, ebins=respMat.ebins, pebins=respMat.pebins)

  #   if pecrop is None:
  #       return Spectrum(data.dot(self.bin_cont), bins=respMat.pebins, xlabel=r"$E_{\rm vis}$ (meV)")

  #   # if pecrop == None:
  #   #   return Spectrum(self.bin_cont.dot(respMat.data), bins=respMat.pebins)
  #   pebins_full = respMat.pebins
  #   pebw = (pebins_full[-1]-pebins_full[0])/(len(pebins_full)-1)
  #   peb1 = int(np.around((pecrop[0]-pebins_full[0])/pebw))
  #   peb2 = int(np.around((pecrop[1]-pebins_full[0])/pebw))
  #   pebins_crop = pebins_full[peb1:peb2+1]
  #   #return Spectrum(self.bin_cont.dot(respMat.data[peb1:peb2]), bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (meV)")
  #   return Spectrum(self.bin_cont.dot(respMat.data[:, peb1:peb2]), bins=pebins_crop, xlabel=r"$E_{\rm vis}$ (meV)")

  def GetWithShiftedEnergy(self, eshift):
    Enu = self.bins
    Epos = Enu + eshift
    return Spectrum(self.bin_cont, bins=self.bins+eshift).Rebin(self.bins)

  # def GetWithPositronEnergy(self, type='ana', inputfile=None, tf2name=None):
  #   Enu = self.bins
  #   mn = 939.56536 #meV
  #   mp = 938.27203 #meV
  #   me = 0.51099893 #meV
  #   if type=='TF2':
  #       Epos = np.zeros_like(Enu)
  #       file = ROOT.TFile(inputfile)
  #       func = file.Get(tf2name)
  #       for i, energy in enumerate(Enu):
  #           Epos_temp=0.
  #           for cos_theta in range(-1, 1, 100):
  #               Epos_temp += func.Eval(energy, cos_theta)

  #           Epos[i] = Epos_temp
  #   elif type=='ana':
  #       Deltanp = mn - mp
  #       Mdiff = -Enu + Deltanp + (Deltanp*Deltanp - me*me)/(2.0*mp)
  #       Tn = 2 * Enu**2 / (mn + mp)
  #       Epos = (-mn + np.sqrt(mn*mn - 4.0*mp*Mdiff))/2.0
  #   else:
  #       RaiseValueError(f"Unknown type value for positron energy shift {type}")

  #   Evis = Epos + me
  #   return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)

  def GetWithPositronEnergy(self, type='2d', s_ibd=None, inputfile=None, tf2name=None,
                          n_cos_theta=49, n_Ee_per_bin=12):#, core_baselines=None, core_powers=None, args=None):
    """
    GNA-equivalent IBD positron-energy mapping:
        * (Ee, ctheta) -> Enu       [IbdFirstOrder::calc_Enu]
        * Jacobian dEnu/dEe         [IbdFirstOrder::calc_dEnu_wrt_Ee]
        * full first-order Vogel-Beacom dsigma/dctheta   [IbdFirstOrder::Xsec]
        * Gauss-Legendre quadrature in ctheta   AND   in Ee within each output bin
          (the latter mirrors GNA's GaussLegendreHist)
    """
    E_nu  = 0.5 * (self.bins[:-1] + self.bins[1:])
    flux  = self.bin_cont
    dE_in = E_nu[1] - E_nu[0]                          # uniform input bin width

    if type == 'TF2':
        Epos = np.zeros_like(Enu)
        file = ROOT.TFile(inputfile)
        func = file.Get(tf2name)
        for i, energy in enumerate(Enu):
            Epos_temp = 0.
            for cos_theta in range(-1, 1, 100):
                Epos_temp += func.Eval(energy, cos_theta)
            Epos[i] = Epos_temp
        Evis = Epos + me
        return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)
    elif type == 'ana':
        # Exact replication of nu2dep(E_nu, has_recoil=True):
        #   delta = (mf^2 - mi^2 - me^2) / (2*mi)  with mi=mp, mf=mn
        delta   = (mn**2 - mp**2 - me**2) / (2.0 * mp)
        s       = 2.0 * mp * E_nu + mp**2
        sqrt_s  = np.sqrt(s)
        E_nu_CM = (s - mp**2) / (2.0 * sqrt_s)
        E_e_CM  = (s - mn**2 + me**2) / (2.0 * sqrt_s)
        coeff   = (s - (mn - me)**2) * (s - (mn + me)**2)
        p_e_CM = np.where(coeff > 0, np.sqrt(np.maximum(coeff, 0.0)) / (2.0 * sqrt_s), 0.0)
        # Average of the two kinematic extremes — matches nu2dep exactly
        E1   = E_nu - delta - (E_nu_CM / mp) * (E_e_CM + p_e_CM)  # max recoil (backward)
        E2   = E_nu - delta - (E_nu_CM / mp) * (E_e_CM - p_e_CM)  # min recoil (forward)
        E_e  = 0.5 * (E1 + E2)
        Evis = E_e + me                                            # T_e+ + 2me = E_e + me
        Evis = np.where(Evis < 2.0 * me, 2.0 * me, Evis)
        return Spectrum(self.bin_cont, bins=Evis).Rebin(self.bins)
    elif type == '2d':
        # ── one-time precomputation, keyed by (bins, s_ibd, quadrature) ─────
        cache_key = (self.bins.tobytes(), id(s_ibd), n_cos_theta, n_Ee_per_bin)
        cached = _PE_2D_CACHE.get(cache_key)
        if cached is None:
            Delta    = mn - mp
            delta_np = (mn**2 - mp**2 - me**2) / (2.0 * mp)
            M_nuc    = 0.5 * (mn + mp)
            y2       = 0.5 * (Delta**2 - me**2)
            fsq, gsq = f*f, g*g

            def _dsig_dct(Enu, ct):
                Enu = np.asarray(Enu, dtype=float); ct = np.asarray(ct, dtype=float)
                Ee0  = Enu - Delta
                ok0  = Ee0 > me
                sEe0 = np.where(ok0, Ee0, 1.0)
                pe0  = np.sqrt(np.maximum(0.0, sEe0*sEe0 - me*me))
                ve0  = pe0 / sEe0
                Ee1  = sEe0 * (1.0 - Enu/M_nuc * (1.0 - ve0*ct)) - y2/M_nuc
                ok1  = Ee1 > me
                sEe1 = np.where(ok1, Ee1, 1.0)
                pe1  = np.sqrt(np.maximum(0.0, sEe1*sEe1 - me*me))
                ve1  = pe1 / sEe1
                sigma1a = 0.5 * ((fsq + 3.0*gsq) + (fsq - gsq) * ve1 * ct) * Ee1 * pe1
                sve0 = np.where(ve0 > 0.0, ve0, 1.0)
                g1 = 2.0*g*(f + f2) * ((2.0*sEe0 + Delta)*(1.0 - ve0*ct) - me*me/sEe0)
                g2 = (fsq + gsq) * (Delta*(1.0 + ve0*ct) + me*me/sEe0)
                A  = (sEe0 + Delta)*(1.0 - ct/sve0) - Delta
                g3 = (fsq + 3.0*gsq) * A
                g4 = (fsq -      gsq) * A * ve0 * ct
                sigma1b = -0.5 * sEe0 * pe0 * (g1 + g2 + g3 + g4) / M_nuc
                hbar_meV  = 1.0e-6 * hbar / qe
                meV_to_cm = (hbar * c * (1.0 / (1.0e6 * qe)))**2 * 1.0e4
                sigma0 = (2.0 * np.pi**2 /
                          (phase_space * (fsq + 3.0*gsq) * (me**5) * n_lifetime / hbar_meV))
                return np.where(ok0 & ok1,
                                (sigma1a + sigma1b) * sigma0 * meV_to_cm, 0.0)

            E_nu  = 0.5 * (self.bins[:-1] + self.bins[1:])
            dE_in = E_nu[1] - E_nu[0]
            sig_in = np.asarray(s_ibd(E_nu), dtype=float)
            Evis_lo = self.bins[:-1]; Evis_hi = self.bins[1:]
            Evis_mid  = 0.5*(Evis_lo + Evis_hi)
            Evis_half = 0.5*(Evis_hi - Evis_lo)
            ct_nodes,    ct_w     = np.polynomial.legendre.leggauss(n_cos_theta)
            Ee_sub_nodes, Ee_sub_w = np.polynomial.legendre.leggauss(n_Ee_per_bin)

            Ee_2d = (Evis_mid[:, None] + Evis_half[:, None] * Ee_sub_nodes[None, :]) - me
            Ee_3d = Ee_2d[:, :, None]
            ct_3d = ct_nodes[None, None, :]
            ok_Ee = Ee_3d > me
            Ve_3d = np.sqrt(np.maximum(0.0, 1.0 - me*me /
                                       np.where(ok_Ee, Ee_3d*Ee_3d, 1.0)))
            denom = 1.0 - (Ee_3d/mp) * (1.0 - Ve_3d * ct_3d)
            sdenom = np.where(denom > 0.0, denom, 1.0)
            Enu_3d = (Ee_3d + delta_np) / sdenom
            in_range = ok_Ee & (denom > 0.0) & (Enu_3d >= E_nu[0]) & (Enu_3d <= E_nu[-1])
            sVe_3d = np.where(Ve_3d > 0.0, Ve_3d, 1.0)
            jac = (mp + (1.0 - ct_3d/sVe_3d) * Enu_3d) \
                  / (mp - (1.0 - Ve_3d*ct_3d) * Ee_3d)
            dsig = _dsig_dct(Enu_3d, np.broadcast_to(ct_3d, Enu_3d.shape))
            # Precombine all the static factors that multiply h_at:
            kernel = np.where(in_range, dsig * jac, 0.0)
            # ctheta quadrature weights already baked in
            ct_kernel = ct_w[None, None, :] * kernel
            cached = {
                'E_nu': E_nu, 'sig_in': sig_in, 'Enu_3d': Enu_3d,
                'ct_kernel': ct_kernel, 'in_range': in_range,
                'Ee_sub_w': Ee_sub_w, 'Evis_half': Evis_half, 'dE_in': dE_in,
            }
            _PE_2D_CACHE[cache_key] = cached

        # ── per-call: only the flux-dependent work ──────────────────────────
        E_nu   = cached['E_nu']
        sig_in = cached['sig_in']
        with np.errstate(divide='ignore', invalid='ignore'):
            h_in = np.where(sig_in > 0.0,
                            self.bin_cont / np.where(sig_in > 0.0, sig_in, 1.0), 0.0)
        from scipy.interpolate import PchipInterpolator
        h_cs = PchipInterpolator(E_nu, h_in)
        h_at = h_cs(cached['Enu_3d'])
        h_at = np.where(np.isfinite(h_at) & cached['in_range'], h_at, 0.0)
        inner = (cached['ct_kernel'] * h_at).sum(axis=2)
        new_counts = (cached['Evis_half'] / cached['dE_in']) \
                     * (cached['Ee_sub_w'][None, :] * inner).sum(axis=1)
        return Spectrum(new_counts, bins=self.bins)
    else:
        raise ValueError(f"Unknown type for positron energy shift: {type!r}")


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
      bins_new = self.bins*G4model.eval(self.bins, kB, fC=fC, kI=kI)*LY_shared/G4model_gamma.eval(2.224589, kB, fC=fC, kI=kI)
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
        data[i][i] = 3./((1./x) + (2./exp.bin_cont[i]))
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
  #       Bin width (in meV) for which sigma is defined.

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
        Bin width (in meV) for which sigma is defined.
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
    if w_source != 0:
        s = w_source / w_target
    else:
        s = 1.0

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

  def WritetoROOT(self, histname, outfile, directory=None):
    # Open the file in update mode
    output_file = ROOT.TFile(outfile, "UPDATE")

    # If a directory is specified, navigate into it or create it
    if directory:
        if not output_file.GetDirectory(directory):
            output_file.mkdir(directory)
        output_file.cd(directory)

    bins = self.bins
    bin_cont = self.bin_cont

    # Create the histogram
    # Note: ROOT.gDirectory will now point to your target folder
    hist = ROOT.TH1D(histname, histname, len(bins) - 1, array('d', bins))
    hist.Sumw2()

    for i in range(len(bin_cont)):
        hist.SetBinContent(i + 1, bin_cont[i])

    # Write the histogram to the current directory (the one we cd'd into)
    hist.Write("", ROOT.TObject.kOverwrite)

    force_vector_streamers(output_file)
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
