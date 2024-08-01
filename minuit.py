import sys
import nuosc
from  noda import *
import noda
import numpy as np
import os
import gc
from datetime import datetime
from iminuit import Minuit
import matplotlib.pyplot as plt

def write_results(m, filename, unc): #writes out fit results m for a given filename and uncertainty
    values = []
    for i in range(len(m.values)):
        values.append(m.values[i]) #central values
        values.append(m.errors[i]) #hesse errors
        values.append(m.merrors[i].lower) #minos neg errors
        values.append(m.merrors[i].upper) #minos pos errors
    fileo = open(filename, "a")
    fileo.write(unc+" ")
    fileo.write(" ".join(map(str, values)))
    fileo.write("\n")
    fileo.close()

def round_errors(param, neg_err, pos_err): #rounds errors for different parameters differently. Used inside plot_profile. But rounding is only for the plot title so as to look pretty
    m_err = neg_err
    p_err = pos_err
    if(param=='sin2_12'):
        m_err = round(neg_err, 5)
        p_err = round(pos_err, 5)
    elif(param=='sin2_13'):
        m_err = round(neg_err, 5)
        p_err = round(pos_err, 5)
    elif(param=='dm2_21'):
        m_err = round(neg_err, 7)
        p_err = round(pos_err, 7)
    elif(param=='dm2_31'):
        m_err = round(neg_err, 9)
        p_err = round(pos_err, 9)
    return m_err, p_err

def plot_profile(m, i, param, plotname): #plots the chi2 profiles for a given parameter i in the parameter list called param (see run_minuit)
    x, y, blah = m.mnprofile(param)
    plt.plot(x, y)
    plt.axvline(x= m.values[i], color='black', linestyle='dashed')
    #plt.axvspan(m.values[i]+m.merrors[i].lower,m.values[i]+m.merrors[i].upper, color='gray', alpha=0.3, label='Vertical Band')
    #m_err, p_err = round_errors(param, m.merrors[i].lower, m.merrors[i].upper)
    #plt.title(f'{param} {m.values[i]} {m_err} + {p_err}')
    plt.ylabel('FCN')
    plt.xlabel(param)
    plt.savefig(plotname)
    plt.close()

def run_minuit(ensp_nom = {}, unc='',baselines = [], powers=[], rm= [], cm ={}, args=''):#, dm2_31=0.):
    import ROOT
    #from array import array
    #output_file = ROOT.TFile("Asimov_NO.root", "RECREATE")
    #bins = ensp_nom['rtot'].bins
    #bin_cont = ensp_nom['rtot'].bin_cont
    #  # Create a TH1D histogram
    #hist = ROOT.TH1D("Asimov NO", "Asimov NO", len(bins) - 1, array('d', bins))
    #hist.Sumw2()
      # Fill the histogram with the bin contents
    #for i in range(len(bin_cont)):
    #    hist.SetBinContent(i + 1, bin_cont[i])
    #  # Save the histogram in the ROOT file
    #hist.Write()
    #output_file.Close()
    #print("Scanning ", dm2_31)
    def get_hist_from_root(filename, histname):
        root_file = ROOT.TFile.Open(filename)
        histogram = root_file.Get(histname)
        bins = np.array([histogram.GetBinLowEdge(i) for i in range(1, histogram.GetNbinsX() + 2)])
        bin_cont = np.array([histogram.GetBinContent(i) for i in range(1, histogram.GetNbinsX() + 1)])
        root_file.Close()
        return Spectrum(bin_cont=bin_cont, bins=bins)
    
    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #Vals for osc parameters and NMO
    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0): #chi2 definition
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true', args=args)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop) #apply energy resolution
        s_tot = s  + ensp_nom['acc'] + ensp_nom['fneu'] + ensp_nom['lihe'] + ensp_nom['aneu'] + ensp_nom['geo'] + ensp_nom['atm'] + ensp_nom['rea300']
        chi2 = 1e+6
        #steven = get_hist_from_root("control_histos_NO.root", "h_tot")
        if args.sin2_th13_opt== "pull":
            chi2 = cm[unc].Chi2_p(ensp_nom['rtot'], s_tot, unc, args.stat_method_opt, pulls=[sin2_13-nuosc.op_nom['sin2_th13']], pull_unc=[args.sin2_th13_pull_unc*nuosc.op_nom['sin2_th13']])  
        if args.sin2_th13_opt== "free":
            chi2 = cm[unc].Chi2(ensp_nom['rtot'],s_tot, unc, args.stat_method_opt) #calculate chi2 using covariance matrix
        #filet = open(f"chi2_{args.stat_opt}_{args.sin2_th13_opt}.txt", "a")
        #filet.write(str(sin2_12)+" "+str(sin2_13)+" "+str(dm2_21)+" "+str(dm2_31)+" "+str(chi2)+"\n")
        #filet.close()
        #print(sin2_12, sin2_13, dm2_21, dm2_31, chi2)
        #print("NO",sin2_12, sin2_13, dm2_21, dm2_31, chi2)
        return chi2

    def chi22(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0): #chi2 definition
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true', opp=True, args=args)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop) #apply energy resolution
        s_tot = s  + ensp_nom['acc'] + ensp_nom['fneu'] + ensp_nom['lihe'] + ensp_nom['aneu'] + ensp_nom['geo'] + ensp_nom['atm'] + ensp_nom['rea300']
        chi2 = 1e+6
        #steven = get_hist_from_root("control_histos_NO.root", "h_tot")
        if args.sin2_th13_opt== "pull":
            chi2 = cm[unc].Chi2_p(ensp_nom['rtot'], s_tot, unc, args.stat_method_opt, pulls=[sin2_13-nuosc.op_nom['sin2_th13']], pull_unc=[args.sin2_th13_pull_unc*nuosc.op_nom['sin2_th13']])  
        if args.sin2_th13_opt== "free":
            chi2 = cm[unc].Chi2(ensp_nom['rtot'],s_tot, unc, args.stat_method_opt) #calculate chi2 using covariance matrix
        
        #chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, unc, args.stat_method_opt) #calculate chi2 using covariance matrix
  #      print(chi2)
        #print("IO", sin2_12, sin2_13, dm2_21, dm2_31, chi2)
        return chi2
   
    def get_spectrum(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, opp=False):
        print("now asimov")
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true', opp=opp, args=args)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop) #apply energy resolution
        return s

   #fitting stuff
    print(nuosc.op_nom['dm2_32'], nuosc.op_nom['dm2_31'])
    m = Minuit(chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"]) #define minuit
    #m.limits['sin2_12'] = (nuosc.op_nom["sin2_th12"] - nuosc.op_nom["sin2_th12"]*0.1, nuosc.op_nom["sin2_th12"] + nuosc.op_nom["sin2_th12"]*0.1)
    #m.limits['dm2_21'] = (nuosc.op_nom["dm2_21"] - nuosc.op_nom["dm2_21"]*0.1, nuosc.op_nom["dm2_21"] + nuosc.op_nom["dm2_21"]*0.1)
    #m.limits['sin2_13'] = (nuosc.op_nom["sin2_th13"] - nuosc.op_nom["sin2_th13"]*3., nuosc.op_nom["sin2_th13"] + nuosc.op_nom["sin2_th13"]*0.3)
    #m.scan(ncall=100)
    m.migrad() #fit
    m.hesse() #get errors
    m.minos() #get minos errors
    print(chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3]))

    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO= not args.NMO_opt) #Vals for osc parameters and NMO
    print(nuosc.op_nom['dm2_32'], nuosc.op_nom['dm2_31'])
    m1 = Minuit(chi22, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"]) #define minuit
    #m.scan(ncall=100)
    m1.migrad() #fit
    m1.hesse() #get errors
    m1.minos() #get minos errors
#  
    print(chi22(sin2_12=m1.values[0], sin2_13=m1.values[1], dm2_21=m1.values[2], dm2_31=m1.values[3]))
    unc_new = unc
    if(args.stat_method_opt == "CNP" and unc != 'stat'): unc_new = 'stat+'+unc
    print("Uncertainty: ", unc_new)
    print(m)
   # print(m1)
    print("chi2", abs(chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3]) - chi22(sin2_12=m1.values[0], sin2_13=m1.values[1], dm2_21=m1.values[2], dm2_31=m1.values[3])))

 #plot IO and NO


#    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #Vals for osc parameters and NMO
    #NO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=False)
#    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO= not args.NMO_opt) #Vals for osc parameters and NMO
   # IO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=True)
#    NO_sp_fit= get_spectrum(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
#    IO_sp_fit = get_spectrum(sin2_12=m1.values[0], sin2_13=m1.values[1], dm2_21=m1.values[2], dm2_31=m1.values[3], opp=True)
#    ensp_nom['rdet'].Plot(f"{args.plots_folder}/NO_vs_IO_vs_data.png",
#                   xlabel="Neutrino energy (MeV)",
#                   ylabel=f"Events per 20 keV",
#                   extra_spectra=[NO_sp, IO_sp],
#                   leg_labels=['NO data', 'NO curve', 'IO curve'],
#                   colors=['black', 'darkred', 'steelblue'],
#                   xmin=0, xmax=10,
#                   ymin=0, ymax=None, log_scale=False)
   # NO_sp.Plot("NO_vs_IO.png",
   #                xlabel="Neutrino energy (MeV)",
   #                ylabel=f"Events per 20 keV",
   #                extra_spectra=[IO_sp],
   #                leg_labels=['NO curve', 'IO curve'],
   #                colors=['darkred', 'steelblue'],
   #                xmin=0, xmax=10,
   #                ymin=0, ymax=None, log_scale=False)
#    NO_sp_fit.Plot(f"{args.plots_folder}/NO_vs_IO_fit.png",
#                   xlabel="Neutrino energy (MeV)",
#                   ylabel=f"Events per 20 keV",
#                   extra_spectra=[IO_sp_fit],
#                   leg_labels=['NO fit', 'IO fit'],
#                   colors=['darkred', 'steelblue'],
#                   xmin=0, xmax=10,
#                   ymin=0, ymax=None, log_scale=False)
    #writing results
    #filename = f"{args.main_data_folder}/fit_results_{args.stat_method_opt}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins_minuit.txt"
    #if unc_new==args.unc_list[0]:
    #    fileo = open(filename, "w")
    #    fileo.write("unc sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr\n")
    #    fileo.close()
    #write_results(m, filename, unc_new) #write results into a textfile
   #fancy stuff
    if(args.plot_minuit_matrix or args.plot_minuit_profiles): #make plots folders
        if not os.path.exists(f"{args.plots_folder}/Chi2_profiles"): os.mkdir(f"{args.plots_folder}/Chi2_profiles")
        if not os.path.exists(f"{args.plots_folder}/Chi2_profiles/Minuit"): os.mkdir(f"{args.plots_folder}/Chi2_profiles/Minuit")

    if(args.plot_minuit_profiles): #create chi2 profiles
        print("Plotting chi2 profiles")
        param_list = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]
        for i in range(len(param_list)): #there is something weird with draw_mnprofile in minuit, so I have to do this from scratch inside plot_profile
            if (i!=3): continue
            plotname = f"{args.plots_folder}/Chi2_profiles/Minuit/chi2_{args.stat_opt}_{param_list[i]}_{unc_new}.png"
            plot_profile(m, i, param_list[i], plotname)

    if(args.print_minuit_correlation):
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())

    if (args.plot_minuit_matrix):
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1,2,3], size=1000, experimental=True)
        plt.savefig(f"{args.plots_folder}/Chi2_profiles/Minuit/matrix_{args.stat_opt}_{unc_new}.png")
