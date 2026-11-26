"""Detector response (resolution) matrices and energy leakage.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import gc
from joblib import Parallel
from joblib import delayed
import uproot
from ..physics.constants import *
from .binning import Rebin2D



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
    plt.xlabel("True energy (meV)")
    plt.ylabel("Visual energy (meV)")
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

def CalcRespMatrix_abc(a, b, c, ebins, pebins, escale=1., eshift=0., norm_mode="per_meV", verbose=True,
                       escale_on_vis=False):
    evis = 0.5*(ebins[1:] + ebins[:-1]) + eshift
    # escale_on_vis: scale the visible energy itself (E_vis -> escale*E_vis), so
    # the resolution is evaluated at the scaled energy, sigma(escale*E_vis).
    # Default: stretch the reco axis, width escale*sigma(E_vis).
    e_res = escale * evis if escale_on_vis else evis
    rel_res = np.sqrt(a**2 / e_res + b**2 + c**2 / e_res**2)  # in fraction

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


def CalcRespMatrix_abc_flu(a, b, c, unc, ebins, pebins, escale=1., eshift=0., norm_mode='per_meV', sample_size=100):
  print(" # Fluctuating (a,b,c) parameters...")
  a_err, b_err, c_err = unc
  a_flu = np.random.normal(loc=a, scale=a_err, size=sample_size)
  b_flu = np.random.normal(loc=b, scale=b_err, size=sample_size)
  c_flu = np.random.normal(loc=c, scale=c_err, size=sample_size)

  CalcRM = lambda a,b,c: CalcRespMatrix_abc( a, b, c, ebins, pebins, escale=escale,  eshift=eshift, norm_mode=norm_mode, verbose=False)
  spectra = Parallel(n_jobs=-1)(delayed(CalcRM)(a_flu[i], b_flu[i], c_flu[i]) for i in range(len(a_flu)))
  gc.collect()
  return spectra

def CalcEnergyLeak(rootfile, histname, ebins, pebins, ecrop=None):
    with uproot.open(rootfile) as file:
        hist = file[histname].to_numpy()
    data, ebins_ori, pebins_ori = hist
    ebin_width_ori = np.diff(ebins_ori)
    pebin_width_ori = np.diff(pebins_ori)
    ebin_width = np.diff(ebins)
    pebin_width = np.diff(pebins)
    if (ebin_width_ori[0] < ebin_width[0]) or (pebin_width_ori[0] < pebin_width[0]):
        data = Rebin2D(data, ebins_ori, pebins_ori, ebins, pebins)

    data = np.array(data, dtype=float)  # ensure writable

    # Zero observed-energy (pebins / column) bins below Ecrop
    if ecrop is not None:
        pe_centers = 0.5 * (pebins[:-1] + pebins[1:])
        data[:, pe_centers < ecrop] = 0.0

    respMat = RespMatrix(data, ebins, pebins)
    row_sums = respMat.data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    respMat.data = respMat.data / row_sums
    respMat.data = np.nan_to_num(respMat.data, nan=0.0)
    return respMat
