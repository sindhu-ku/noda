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

def write_results(m, fileout, unc):
    values = []
    for i in range(4):
        values.append(m.values[i])
        values.append(m.errors[i])
        values.append(m.merrors[i].lower)
        values.append(m.merrors[i].upper)
    fileo = open(fileout, "a")
    fileo.write(unc+" ")
    fileo.write(" ".join(map(str, values)))
    fileo.write("\n")
    fileo.close()

#def plot_profiles(m, plots_folder, stat_opt, unc):



def run_minuit(ensp_nom = {}, unc='',baselines = [], powers=[], rm= [], cm ={}, args='', fileout=''):
    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true')
        s = s.GetWithPositronEnergy()
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop)
        chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, unc, args.stat_method_opt)
        return chi2
    m = Minuit(chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
    m.migrad()
    m.hesse()
    m.minos()
    # print("Plotting chi2 profiles")
    # param_list = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]
    # if not os.path.exists(f"{args.plots_folder}/Chi2_profiles"): os.mkdir(f"{args.plots_folder}/Chi2_profiles")
    # if not os.path.exists(f"{args.plots_folder}/Chi2_profiles/Minuit"): os.mkdir(f"{args.plots_folder}/Chi2_profiles/Minuit")
    # for i in range(len(param_list)):
    #      print(i)
    #      plt.plot(m.draw_profile(param_list[i]))
    #      plt.show()
        #prof = m.mnprofile(param_list[i])
        #plt.plot(prof)
        #plt.savefig(f"{args.plots_folder}/Chi2_profiles/Minuit/chi2_{args.stat_opt}_{param_list[i]}_{unc}.png")
    if(args.stat_method_opt == "CNP" and unc != 'stat'): unc = 'stat+'+unc
    print("Uncertainty: ", unc)
    print(m)
    write_results(m, fileout, unc)
    #print(m.covariance.correlation())
    if (args.show_minuit_posteriors):
        if(unc== args.unc_list[len(unc_list)-1]): m.draw_mnmatrix(cl=[1,2,3])
