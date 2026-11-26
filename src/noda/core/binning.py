"""Bin-edge construction and rebinning helpers (with bounded caches).

Split out of noda.py (pure move); noda.noda re-exports everything."""
import numpy as np
from ..physics.constants import *
from .covmatrix import CovMatrix

# ─── module-level rebin caches ───────────────────────────────────────────
# Bounded LRU: energy-scale/NL nuisances shift the input bin edges, so in a
# fit/MCMC almost every eval produces NEW keys (~4 x 1.4 MB matrices/eval);
# unbounded this leaks ~5 MB per chi2 eval.
from collections import OrderedDict
_REBIN_MATRIX_CACHE = OrderedDict()
_REBIN_MATRIX_CACHE_MAX = 64
_NONUNIFORM_BINS_CACHE = {}

def _build_rebin_matrix(in_bins, out_bins):
    """
    W[i, j] = overlap(out_i, in_j) / width(in_j)
    so that:  out_bin_cont = W @ in_bin_cont
    matches Rebin_simple's "count-per-bin" semantics exactly.
    Cached by the raw bytes of (in_bins, out_bins).
    """
    in_bins  = np.ascontiguousarray(in_bins,  dtype=float)
    out_bins = np.ascontiguousarray(out_bins, dtype=float)
    key = (in_bins.tobytes(), out_bins.tobytes())
    W = _REBIN_MATRIX_CACHE.get(key)
    if W is None:
        in_lo,  in_hi  = in_bins[:-1],  in_bins[1:]
        out_lo, out_hi = out_bins[:-1], out_bins[1:]
        out_lo = out_bins[:-1]
        in_w = in_hi - in_lo
        # rows = output bins, cols = input bins; broadcast
        overlap_lo = np.maximum(out_lo[:, None], in_lo[None, :])
        overlap_hi = np.minimum(out_hi[:, None], in_hi[None, :])
        overlap    = np.clip(overlap_hi - overlap_lo, 0.0, None)
        W = overlap / in_w[None, :]
        _REBIN_MATRIX_CACHE[key] = W
        while len(_REBIN_MATRIX_CACHE) > _REBIN_MATRIX_CACHE_MAX:
            _REBIN_MATRIX_CACHE.popitem(last=False)
    else:
        _REBIN_MATRIX_CACHE.move_to_end(key)
    return W


def _build_nonuniform_bins(bins_nonuniform):
    """Cached construction of the concatenated bin-edge array from a
    list of (start, end, num_bins) triples — same logic as the original
    Rebin_nonuniform but built once per unique spec."""
    spec_key = tuple((float(s), float(e), int(n)) for s, e, n in bins_nonuniform)
    arr = _NONUNIFORM_BINS_CACHE.get(spec_key)
    if arr is None:
        pieces = []
        for start, end, num_bins in bins_nonuniform:
            edges = np.linspace(start, end, num_bins + 1)
            pieces.extend(edges[:-1])
        pieces.append(bins_nonuniform[-1][1])
        arr = np.asarray(pieces, dtype=float)
        _NONUNIFORM_BINS_CACHE[spec_key] = arr
    return arr

def _overlap_matrix(old_edges, new_edges):
    """W[i,k] = fraction of old bin k falling into new bin i (counts-conserving)."""
    old_edges = np.asarray(old_edges, float)
    new_edges = np.asarray(new_edges, float)
    lo = np.maximum(new_edges[:-1, None], old_edges[None, :-1])
    hi = np.minimum(new_edges[1:,  None], old_edges[None, 1:])
    overlap = np.clip(hi - lo, 0.0, None)
    old_w = np.diff(old_edges)[None, :]
    return overlap / old_w

def Rebin2D(data, xedges, yedges, new_xedges, new_yedges):
    Wx = _overlap_matrix(xedges, new_xedges)   # (nx_new, nx_old)
    Wy = _overlap_matrix(yedges, new_yedges)   # (ny_new, ny_old)
    return Wx @ np.asarray(data, float) @ Wy.T


def make_nonuniform_bins(bins_nonuniform):
    edges = []
    for start, end, n in bins_nonuniform:
        part = np.linspace(start, end, n + 1)
        if edges:  # avoid duplicating the shared boundary
            part = part[1:]
        edges.extend(part)
    return np.array(edges)


def expand_nuisances_to_fine(coarse_edges, fine_edges, nuisance_coarse):
    fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    expanded_nuisance = np.zeros_like(fine_centers)

    indices = np.digitize(fine_centers, coarse_edges) - 1

    n_bins = len(coarse_edges) -1         # number of coarse segments
    valid_mask = (indices >= 0) & (indices < n_bins)

    expanded_nuisance[valid_mask] = np.asarray(nuisance_coarse)[indices[valid_mask]]
    return expanded_nuisance
def centers_to_edges(centers):
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    # extrapolate first and last edges
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges

def rebin_cov_matrix(cov, target_bins):
    def expand_edges(bins):
        bins = np.asarray(bins)
        if bins.ndim == 1:
            return bins  # already flat edges
        edges = []
        for lo, hi, n in bins:
            edges.extend(np.linspace(lo, hi, int(n) + 1).tolist())
        deduped = [edges[0]]
        for e in edges[1:]:
            if not np.isclose(e, deduped[-1]):
                deduped.append(e)
        return np.array(deduped)

    orig_edges = expand_edges(cov.bins)
    tgt_edges  = expand_edges(target_bins)

    assert cov.data.shape[0] == len(orig_edges) - 1, \
        f"Matrix size {cov.data.shape[0]} doesn't match orig bins ({len(orig_edges)-1} sub-bins)"

    N_orig = len(orig_edges) - 1
    N_tgt  = len(tgt_edges)  - 1

    T = np.zeros((N_tgt, N_orig))
    for j in range(N_orig):
        o_lo, o_hi = orig_edges[j], orig_edges[j + 1]
        o_width = o_hi - o_lo
        for i in range(N_tgt):
            t_lo, t_hi = tgt_edges[i], tgt_edges[i + 1]
            overlap = max(0.0, min(o_hi, t_hi) - max(o_lo, t_lo))
            if overlap > 0:
                T[i, j] = overlap / o_width

    new_data = T @ cov.data @ T.T
    return CovMatrix(new_data, bins=target_bins, axis_label=cov.axis_label)
