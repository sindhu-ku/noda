"""ROOT helpers shared by several modules.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import ROOT


def force_vector_streamers(root_file):
    """Persist the TVectorT<double> streamer info into an open `root_file`.

    ROOT's TFile(...,"update") only (re)writes the streamer infos for classes
    *touched in the current session*, clobbering the ones already on disk. Our
    postfit file is reopened many times after the TVectorD nuisances are
    written (write_param_dbl, postfit spectra, dm2_31 profile, …), so its
    TVectorT<double> streamer gets dropped — after which uproot can no longer
    decode the nuisance vectors (PyROOT still can, having the class compiled
    in). Force-writing the streamer just before every Close() keeps it in the
    file. Idempotent and harmless for files that hold no vectors.

    Call immediately before `root_file.Close()`.
    """
    try:
        cl = ROOT.TClass.GetClass("TVectorT<double>")
        info = cl.GetStreamerInfo() if cl else None
        if info:
            info.ForceWriteInfo(root_file, True)
    except Exception:
        pass
