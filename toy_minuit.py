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

def run_minuit(ensp_nom = {}, baselines = [], powers=[], rm= [], cm ={}, args=''):

    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0): #chi2 definition
        s = ensp_nom['ribd'].GetOscillated(L=baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=powers, me_rho=args.me_rho, ene_mode='true', args=args)
        s = s.GetWithPositronEnergy() #shift to positron energy
        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL']) #apply non-linearity
        s = s.ApplyDetResp(rm, pecrop=args.ene_crop) #apply energy resolution
        chi2 = cm[args.unc].Chi2(ensp_nom["toy"],s, args.unc, args.stat_method_opt) #calculate chi2 using covariance matrix
        return chi2

   #fitting stuff
    nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #Vals for osc parameters and NMO
    m = Minuit(chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"]) #define minuit
    print("# Chi2 function defined, performing migrad minimization")
    m.migrad()#fit
    if(m.valid):
        #print("# Getting hesse errors")
        #m.hesse() #get errors
        #print("# Getting minos errors")
        #m.minos() #get minos errors

        #print results
        unc_new = args.unc
        if(args.stat_method_opt == "CNP" and args.unc != 'stat'): unc_new = 'stat+'+args.unc
        print("# Uncertainty used: ", unc_new)
        print(m)

        delta_chi2 = abs(chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3]) - cm[args.unc].Chi2(ensp_nom["rdet"],ensp_nom["unosc"], args.unc, args.stat_method_opt))


        return m.values[0], m.values[1], m.values[2], m.values[3], delta_chi2
    else:
        print("Failed to find minimum")
        return -111, -111, -111, -111, -111
