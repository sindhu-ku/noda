"""Compatibility facade: noda.noda re-exports the split modules.

All code that did `from .noda import *` (and pickled CovMatrix/RespMatrix
caches that refer to noda.noda.<Class>) keeps working. New code should
import from noda.core directly:

  root_utils     ROOT helpers shared by several modules.
  binning        Bin-edge construction and rebinning helpers (with bounded caches).
  covmatrix      Covariance matrix class and helpers.
  spectrum       Spectrum class: binned spectra, oscillation, energy models, plotting.
  response       Detector response (resolution) matrices and energy leakage.
  likelihood     Chi2 / likelihood functions.
  physics.reactor        Reactor flux normalisation, survival probability, reactor inputs.
  fileio         Reading/writing spectra, matrices and pickled objects; livetime.
  physics.tao_bkg        Analytic TAO fast-neutron spectrum.
"""
from .core.root_utils import *
from .core.binning import *
from .core.covmatrix import *
from .core.spectrum import *
from .core.response import *
from .core.likelihood import *
from .physics.reactor import *
from .core.fileio import *
from .physics.tao_bkg import *
from .core.binning import _build_rebin_matrix, _build_nonuniform_bins, _overlap_matrix, _REBIN_MATRIX_CACHE, _REBIN_MATRIX_CACHE_MAX, _NONUNIFORM_BINS_CACHE
from .core.spectrum import _PE_2D_CACHE

# Original import header, re-bound last so `from .noda import *` exports
# exactly the same module names as before the split.
import pickle
import csv
import sys
from .physics import nuosc as nuosc
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
from .physics.nl_geant4 import Geant4Model
import yaml
import hashlib
from .physics.constants import *
