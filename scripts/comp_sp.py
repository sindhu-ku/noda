# import ROOT
# from array import array
# output_file = ROOT.TFile("Asimov_NO.root", "RECREATE")
# bins = ensp_nom_juno[''].bins
# bin_cont = ensp_nom_juno[''].bin_cont
#  # Create a TH1D histogram
# hist = ROOT.TH1D("Asimov NO", "Asimov NO", len(bins) - 1, array('d', bins))
# hist.Sumw2()
#  #Fill the histogram with the bin contents
# for i in range(len(bin_cont)):
#    hist.SetBinContent(i + 1, bin_cont[i])
#  # Save the histogram in the ROOT file
# hist.Write()
# output_file.Close()
#print("Scanning ", dm2_31)
# def get_hist_from_root(filename, histname):
#     root_file = ROOT.TFile.Open(filename)
#     histogram = root_file.Get(histname)
#     bins = np.array([histogram.GetBinLowEdge(i) for i in range(1, histogram.GetNbinsX() + 2)])
#     bin_cont = np.array([histogram.GetBinContent(i) for i in range(1, histogram.GetNbinsX() + 1)])
#     root_file.Close()
#     return Spectrum(bin_cont=bin_cont, bins=bins)

# ensp_nom_juno["rdet"] = ensp_nom_juno['rdet'].Rebin_nonuniform(args_juno.bins_nonuniform)
# ensp_nom_tao["rdet"] = ensp_nom_tao['rdet'].Rebin_nonuniform(args_juno.bins_nonuniform)
