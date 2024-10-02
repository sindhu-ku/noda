#!/usr/bin/env python3
import sys
from . import nuosc as nuosc
from .noda import *
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
    print(f"Results written to {filename}")

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
    elif(param=='dm2_32'):
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

def run_minuit(ensp_nom_juno={}, ensp_nom_tao={},  unc_juno='', unc_tao='', unc_corr='', rm= [], ene_leak_tao =[], cm_juno ={}, cm_tao={}, cm_corr='', args_juno='', args_tao=''):#, dm2_31=0.):

    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt) #Vals for osc parameters and NMO

    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Ngeo=1.0, NMO_fit=False, opp=False):
        # Get the oscillated spectrum
        s_juno = ensp_nom_juno['ribd'].GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s_juno = s_juno.GetWithPositronEnergy()  # Shift to positron energy

        s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
        s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution
        s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                ensp_nom_juno['aneu'] + ensp_nom_juno['geo'].GetScaledFit(Ngeo) + ensp_nom_juno['atm'] + \
                ensp_nom_juno['rea300']

        chi2 = 1e+6
        if NMO_fit:
            s_tao = ensp_nom_tao['ribd'].GetOscillated(L=args_tao.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_tao.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_tao)
            s_tao = s_tao.GetWithPositronEnergy()  # Shift to positron energy
            s_tao = s_tao.ApplyDetResp(ene_leak_tao, pecrop=args_juno.ene_crop)
            s_tao = s_tao.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
            s_tao = s_tao.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution
            s_tot_tao = s_tao + ensp_nom_tao['acc'] + ensp_nom_tao['fneu'] + ensp_nom_tao['lihe']

            if unc_juno == 'stat' and unc_tao== 'stat' and unc_corr=='stat': unc='stat'
            else: unc='syst'

            if args_juno.sin2_th13_opt == "pull":
                chi2 = Combined_Chi2(cm_juno[unc_juno], cm_tao[unc_tao], cm_corr[unc_corr], ensp_nom_juno["rtot"],
                      s_tot_juno, ensp_nom_tao["rtot"], s_tot_tao, ensp_nom_juno["rdet"], s_juno,
                      ensp_nom_tao["rdet"], s_tao,  unc=unc,
                      stat_meth=args_juno.stat_method_opt, pulls=[sin2_13 - nuosc.op_nom['sin2_th13']],
                      pull_unc=[args_juno.sin2_th13_pull_unc * nuosc.op_nom['sin2_th13']])
            else:
                chi2 = Combined_Chi2(cm_juno[unc_juno], cm_tao[unc_tao], cm_corr[unc_corr], ensp_nom_juno["rtot"],
                      s_tot_juno, ensp_nom_tao["rtot"], s_tot_tao, ensp_nom_juno["rdet"], s_juno,
                      ensp_nom_tao["rdet"], s_tao,  unc=unc,
                      stat_meth=args_juno.stat_method_opt, pulls=[sin2_13 - nuosc.op_nom['sin2_th13']],
                      pull_unc=[args_juno.sin2_th13_pull_unc * nuosc.op_nom['sin2_th13']])
        else:
            if args_juno.sin2_th13_opt == "pull":
                chi2 = Chi2(cm_juno[unc_juno], ensp_nom_juno["rtot"], s_tot_juno, ensp_nom_juno["rdet"], s_juno, unc_juno,
                      args_juno.stat_method_opt, pulls=[sin2_13 - nuosc.op_nom['sin2_th13']],
                      pull_unc=[args_juno.sin2_th13_pull_unc * nuosc.op_nom['sin2_th13']])
            else:
                chi2 = Chi2(cm_juno[unc_juno], ensp_nom_juno["rtot"], s_tot_juno, ensp_nom_juno["rdet"], s_juno, unc_juno,
                       args_juno.stat_method_opt)


        return chi2

    #For drawing
    def get_spectrum(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, opp=False):
        s = ensp_nom_juno['ribd'].GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers, me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args_juno.ene_crop) #apply energy resolution
        return s

   #fitting stuff
    def chi2_pmop(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea=1.0, Ngeo=1.0, NMO_fit=False, opp=False)

    def combined_chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea=1.0, Ngeo=1.0, NMO_fit=True, opp=False)
    def combined_chi2_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea=1.0, Ngeo=1.0, NMO_fit=True, opp=True)

    def chi2_geo(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Ngeo=1.0):
        return chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea, Ngeo, MO_fit=False, opp=False)

    if args_juno.geo_fit:
        m = Minuit(chi2_geo, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0, Ngeo=1.0)
    else:
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])

    m.migrad() #fit
    m.hesse() #get errors
    m.minos() #get minos errors

    unc_new_juno = unc_juno
    if(unc_juno != 'stat'): unc_new_juno = 'stat+'+unc_juno
    print("Uncertainty JUNO: ", unc_new_juno)

    unc_new_tao = unc_tao
    if(unc_tao != 'stat'): unc_new_tao = 'stat+'+unc_tao
    print("Uncertainty TAO: ", unc_new_tao)

    print("Measurement of oscillation parameters: ")
    print(m)

    if args_juno.NMO_fit:
        print("Determining NMO: ")
        m_comb = Minuit(combined_chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m_comb.migrad()

        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
        m_comb_opp = Minuit(combined_chi2_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m_comb_opp.migrad()

        chi2_min = combined_chi2(sin2_12=m_comb.values[0], sin2_13=m_comb.values[1], dm2_21=m_comb.values[2], dm2_31=m_comb.values[3])
        chi2_min_opp = combined_chi2_opp(sin2_12=m_comb_opp.values[0], sin2_13=m_comb_opp.values[1], dm2_21=m_comb_opp.values[2], dm2_31=m_comb_opp.values[3])
        dchi2 = abs(chi2_min-chi2_min_opp)
        print(f"JUNO+TAO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")



 #plot IO and NO

#     nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt) #Vals for osc parameters and NMO
    NO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=False)
#     nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
    IO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=True)
#     NO_sp_fit= get_spectrum(sin2_12=m_tao.values[0], sin2_13=m_tao.values[1], dm2_21=m_tao.values[2], dm2_31=m_tao.values[3])
#     IO_sp_fit = get_spectrum(sin2_12=m1_tao.values[0], sin2_13=m1_tao.values[1], dm2_21=m1_tao.values[2], dm2_31=m1_tao.values[3], opp=True)
#
# #    ensp_nom_juno['rdet'].Plot(f"{args_juno.plots_folder}/NO_vs_IO_vs_data.png",
# #                   xlabel="Neutrino energy (MeV)",
# #                   ylabel=f"Events per 20 keV",
# #                   extra_spectra=[NO_sp, IO_sp],
# #                   leg_labels=['NO data', 'NO curve', 'IO curve'],
# #                   colors=['black', 'darkred', 'steelblue'],
# #                   xmin=0, xmax=10,
# #                   ymin=0, ymax=None, log_scale=False)
#
    NO_sp.Plot(f"{args_juno.plots_folder}/NO_vs_IO.png",
            xlabel="Neutrino energy (MeV)",
            ylabel=f"Events per 20 keV",
            extra_spectra=[IO_sp],
            leg_labels=['NO curve', 'IO curve'],
            colors=['darkred', 'steelblue'],
            xmin=0, xmax=10,
            ymin=0, ymax=None, log_scale=False)
#
#     NO_sp_fit.Plot(f"{args_juno.plots_folder}/NO_vs_IO_fit.png",
#                   xlabel="Neutrino energy (MeV)",
#                   ylabel=f"Events per 20 keV",
#                   extra_spectra=[IO_sp_fit],
#                   leg_labels=['NO fit', 'IO fit'],
#                   colors=['darkred', 'steelblue'],
#                   xmin=0, xmax=10,
#                   ymin=0, ymax=None, log_scale=False)
    #writing results
    if args_juno.write_results:
        if args_juno.NMO_fit:
            filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.stat_method_opt}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.txt"
            fileo = open(filename, "w")
            fileo.write("unc sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr\n")
            fileo.close()
            write_results(m_comb, filename, 'stat+all')
            filename_nmo = f"{args_juno.main_data_folder}/fit_results_NMO_{args_juno.stat_method_opt}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.txt"
            filenmo = open(filename_nmo, "w")
            filenmo.write("unc_juno unc_tao unc_corr chi2_no chi2_io dchi2 sigma\n")
            filenmo.write(args_juno.unc_list_juno[0]+' '+args_tao.unc_list_tao[0]+' '+args_juno.unc_corr_ind+args_juno.unc_corr_dep+' '+str(chi2_min)+' '\
                          +str(chi2_min_opp)+' '+str(dchi2)+' '+str(np.sqrt(dchi2))+'\n')
            filenmo.close()
        else:
            filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.stat_method_opt}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.txt"
            if unc_new_juno==args_juno.unc_list_juno[0]:
                fileo = open(filename, "w")
                if args_juno.geo_fit: fileo.write("unc sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr Nrea Nrea_err Nrea_merr Nrea_perr Ngeo Ngeo_err Ngeo_merr Ngeo_perr\n")
                else: fileo.write("unc sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr\n")
                fileo.close()
            write_results(m, filename, unc_new_juno) #write results into a textfile

   #fancy stuff
    if(args_juno.plot_minuit_matrix or args_juno.plot_minuit_profiles): #make plots folders
        if not os.path.exists(f"{args_juno.plots_folder}/Chi2_profiles"): os.mkdir(f"{args_juno.plots_folder}/Chi2_profiles")
        if not os.path.exists(f"{args_juno.plots_folder}/Chi2_profiles/Minuit"): os.mkdir(f"{args_juno.plots_folder}/Chi2_profiles/Minuit")

    if(args_juno.plot_minuit_profiles): #create chi2 profiles
        print("Plotting chi2 profiles")
        param_list = ["sin2_12", "sin2_13", "dm2_21", "dm2_32"]
        for i in range(len(param_list)): #there is something weird with draw_mnprofile in minuit, so I have to do this from scratch inside plot_profile
            plotname = f"{args_juno.plots_folder}/Chi2_profiles/Minuit/chi2_{args_juno.stat_opt}_{param_list[i]}_{unc_new}.png"
            plot_profile(m, i, param_list[i], plotname)

    if(args_juno.print_minuit_correlation):
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())

    if (args_juno.plot_minuit_matrix):
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1,2,3], size=1000, experimental=True)
        plt.savefig(f"{args_juno.plots_folder}/Chi2_profiles/Minuit/matrix_{args_juno.stat_opt}_{unc_new}.png")
