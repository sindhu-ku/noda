"""Covariance matrix class and helpers.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from ..physics.constants import *



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
    self._data_inv     = None      # lazy
    self._invertibility = None     # lazy
    if len(bins) == 0:
        bins = np.linspace(0, data.shape[0]+1, data.shape[0])
    self.bins = bins
    self.axis_label = axis_label

  @property
  def data_inv(self):
    if self._data_inv is None and self.IsInvertible():
        self._data_inv = np.linalg.inv(self.data)
    return self._data_inv

  def IsInvertible(self):
    # was: matrix_rank == size (does a full SVD — O(n^3), very slow).
    # Try-inv is also O(n^3) but with a much smaller constant; for our
    # well-conditioned bin-scale covariances it succeeds in microseconds.
    # Memoised so we only pay this once per matrix.
    if self._invertibility is None:
        try:
            self._data_inv = np.linalg.inv(self.data)
            self._invertibility = True
        except np.linalg.LinAlgError:
            self._invertibility = False
    return self._invertibility

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
    sf2 = scale_factor * scale_factor
    self.data *= sf2
    # invalidate the cached inverse — recompute on next access
    self._data_inv = None
    self._invertibility = None

  def GetScaled(self, scale_factor):
    sf2 = scale_factor * scale_factor
    return CovMatrix(data=self.data * sf2, bins=self.bins)

  def ScaleByBinWidth(self):
    """Scale in-place by bin widths (outer product), mirroring Spectrum.ScaleByBinWidth."""
    widths = np.diff(self.bins)          # bin edges → widths, shape (n,)
    self.data = self.data * np.outer(widths, widths)
    self._data_inv = None                # invalidate cached inverse
    self._invertibility = None

  def GetScaledByBinWidth(self):
    """Return a new CovMatrix scaled by bin widths, mirroring Spectrum.GetScaled."""
    widths = np.diff(self.bins)
    return CovMatrix(
        self.data * np.outer(widths, widths),
        bins=self.bins,
        axis_label=self.axis_label
    )
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


def GetRelCM(V, d, bins=None, axis_label=""):
    if V.data.shape[0] != len(d.bin_cont):
        raise ValueError("Length of d must match size of covariance matrix")

    obs = d.bin_cont
    denom = np.outer(obs, obs)
    Q = V.data/denom

    return CovMatrix(Q, bins=d.bins, axis_label=axis_label)
