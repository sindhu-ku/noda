"""Reading/writing spectra, matrices and pickled objects; livetime.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import pickle
import numpy as np
import uproot
from ..physics.constants import *
from .covmatrix import CovMatrix
from .spectrum import Spectrum


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


def GetSpectrumFromROOT(fname, hname, scale=1., eshift=0, xlabel="Energy (meV)"):
    rf = uproot.open(fname)
    obj = rf[hname]

    classname = getattr(obj, "classname", "")

    # 1. Histogram (TH1 family) -- has the high-level behavior interface
    if hasattr(obj, "axis"):
        bin_cont = obj.values() * scale
        bins = obj.axis().edges() + eshift

    # 2. TGraph / TGraphErrors -- read raw members, do NOT rely on .values()
    elif "TGraph" in classname:
        n = int(obj.member("fNpoints"))
        x_pts = np.asarray(obj.member("fX"), dtype=float)[:n] + eshift
        y_pts = np.asarray(obj.member("fY"), dtype=float)[:n] * scale

        # y-errors only exist on TGraphErrors / TGraphAsymmErrors
        y_err = None
        if "fEY" in obj.members:
            y_err = np.asarray(obj.member("fEY"), dtype=float)[:n] * scale

        # sort by x in case the graph points aren't monotonic
        order = np.argsort(x_pts)
        x_pts, y_pts = x_pts[order], y_pts[order]
        if y_err is not None:
            y_err = y_err[order]

        # Build contiguous bin edges from discrete points via midpoints
        if len(x_pts) > 1:
            midpoints = (x_pts[:-1] + x_pts[1:]) / 2
            left_edge = x_pts[0] - (midpoints[0] - x_pts[0])
            right_edge = x_pts[-1] + (x_pts[-1] - midpoints[-1])
            bins = np.concatenate(([left_edge], midpoints, [right_edge]))
        else:
            bins = np.array([x_pts[0] - 0.5, x_pts[0] + 0.5])

        bin_cont = y_pts

    else:
        raise TypeError(
            f"Object {hname} ({classname!r}) is neither a Histogram nor a TGraph/TGraphErrors."
        )

    return Spectrum(bin_cont, bins, xlabel=xlabel)

def calculate_ndays(args_obj):
    opt = args_obj.stat_opt
    duty_cycle = args_obj.duty_cycle

    if opt.endswith("days"):
        val = float(opt[:-4])
        return val * duty_cycle
    elif opt.endswith("year"):
        val = float(opt[:-4])
        return 365.25 * val * duty_cycle
    elif opt.endswith("years"):
        val = float(opt[:-5])
        return 365.25 * val * duty_cycle
    elif opt.endswith("months"):
        val = float(opt[:-6])
        return 30.436875 * val * duty_cycle
    else:
        raise ValueError(f"Unsupported time format in '{opt}'. Use days, months, or years.")

def GetCovMatrixFromROOT(file_path, hist_name):

    with uproot.open(file_path) as f:
        # Access the histogram
        hist = f[hist_name]
        data = hist.values()
        bins = hist.axes[0].edges()

        # Get the title or axis label
        label = hist.title if hist.title else axis_label

    return CovMatrix(data=data, bins=bins, axis_label=label)
