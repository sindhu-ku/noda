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
    for i in range(4):
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
    m_err = 0.
    p_err = 0.
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
    plt.axvspan(m.values[i]+m.merrors[i].lower,m.values[i]+m.merrors[i].upper, color='gray', alpha=0.3, label='Vertical Band')
    m_err, p_err = round_errors(param, m.merrors[i].lower, m.merrors[i].upper)
    plt.title(f'{param} {m.values[i]} {m_err} + {p_err}')
    plt.ylabel('FCN')
    plt.xlabel(param)
    plt.savefig(plotname)
    plt.close()

def run_minuit(ensp_nom = {}, unc='',baselines = [], powers=[], rm= [], cm ={}, args='', fileout=''):

    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0): #chi2 definition
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true', args=args)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop) #apply energy resolution
        chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, unc, args.stat_method_opt) #calculate chi2 using covariance matrix
        return chi2

   #fitting stuff
    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #Vals for osc parameters and NMO
    m = Minuit(chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"]) #define minuit
    m.migrad() #fit
    m.hesse() #get errors
    m.minos() #get minos errors

    #print results
    unc_new = unc
    if(args.stat_method_opt == "CNP" and unc != 'stat'): unc_new = 'stat+'+unc
    print("Uncertainty: ", unc_new)
    print(m)

    print(unc, abs(chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3]) - cm[unc].Chi2(ensp_nom["rdet"],ensp_nom["unosc"], unc, args.stat_method_opt)))

    #writing results
    filename = f"{args.main_data_folder}/fit_results_{args.stat_method_opt}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins_minuit.txt"
    if unc_new==args.unc_list[0]:
        fileo = open(filename, "w")
        fileo.write("unc sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr\n")
        fileo.close()
    write_results(m, filename, unc_new) #write results into a textfile

   #fancy stuff
    if(args.plot_minuit_matrix or args.plot_minuit_profiles): #make plots folders
        if not os.path.exists(f"{args.plots_folder}/Chi2_profiles"): os.mkdir(f"{args.plots_folder}/Chi2_profiles")
        if not os.path.exists(f"{args.plots_folder}/Chi2_profiles/Minuit"): os.mkdir(f"{args.plots_folder}/Chi2_profiles/Minuit")

    if(args.plot_minuit_profiles): #create chi2 profiles
        print("Plotting chi2 profiles")
        param_list = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]
        for i in range(len(param_list)): #there is something weird with draw_mnprofile in minuit, so I have to do this from scratch inside plot_profile
            plotname = f"{args.plots_folder}/Chi2_profiles/Minuit/chi2_{args.stat_opt}_{param_list[i]}_{unc_new}.png"
            plot_profile(m, i, param_list[i], plotname)

    if(args.print_minuit_correlation):
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())

    if (args.plot_minuit_matrix):
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1,2,3])
        plt.savefig(f"{args.plots_folder}/Chi2_profiles/Minuit/matrix_{args.stat_opt}_{unc_new}.png")
