from dayabay_model_official import model_dayabay
from .noda import make_nonuniform_bins

def make_dyb_model(args):
    print(" ")
    print("# Building custom Dagflow Daya Bay model")
    print(" ")
    override_indices = { 'detector_selected' : ("AD11", "AD12", "AD21", "AD22") }
    ebins_nd = make_nonuniform_bins(args.bins_nonuniform)
    dayabay_edges_fine = ebins_nd
    dyb_model = model_dayabay(path_data=args.dagflow_data, override_indices=override_indices,final_erec_bin_edges=dayabay_edges_fine, \
                              override_cfg_files={"antineutrino_spectrum_segment_edges": args.custom_bin_edges})
    
    return dyb_model
