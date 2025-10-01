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
import time

def write_results(m, filename, extra, merrors=True, silent=False): #writes out fit results m for a given filename and uncertainty
    values = []
    for i in range(len(m.values)):
        values.append(m.values[i]) #central values
        values.append(m.errors[i]) #hesse errors
        if merrors:
            values.append(m.merrors[i].lower) #minos neg errors
            values.append(m.merrors[i].upper) #minos pos errors
    fileo = open(filename, "a")
    fileo.write(" ".join(map(str, values)))
    fileo.write(" ")
    fileo.write(" ".join(map(str, extra)))
    fileo.write("\n")
    fileo.close()

    #fileo.write("\n")
    fileo.close()
    if not silent: print(f"Results written to {filename}")

def get_toy_results(m, extra, merrors): #writes out fit results m for a given filename and uncertainty
    params = []
    for i in range(len(m.values)):
        params.append(m.values[i]) #central values

        if merrors:
            params.append(max(abs(m.merrors[i].upper), abs(m.merrors[i].lower)))
        else:
            params.append(m.errors[i]) #hesse errors

    params.extend(extra)
    del m
    return params

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
    plt.axvspan(m.values[i]+m.merrors[i].lower,m.values[i]+m.merrors[i].upper, color='gray', alpha=0.3, label='Vertical Band')
    m_err, p_err = round_errors(param, m.merrors[i].lower, m.merrors[i].upper)
    plt.title(f'{param} {m.values[i]} {m_err} + {p_err}')
    plt.ylabel('FCN')
    plt.xlabel(param)
    plt.savefig(plotname)
    plt.close()

def run_minuit(ensp_nom_juno={}, ensp_nom_nd={},  unc_juno='', unc_nd='', unc_corr='', rm= [], rm_nd=[], ene_leak_nd =[], cm_juno ={}, cm_nd={}, cm_corr='', dyb_rel_cm = [], args_juno='', args_nd=''):#, dm2_31=0.):
   # a_nom, b_nom, c_nom =args_juno.a, args_juno.b, args_juno.c
#    crust = np.random.normal(loc=0., scale=0.1)

    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Ngeo=1.0, Nrea=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False): #, a1=0.0, a2=0.0, a3=0.0, a4=0.0, me_rho=args_juno.me_rho):
        pulls=[]
        pull_unc = []
        if args_juno.sin2_th13_opt=="pull":
            sin2_th13_pull_unc = nuosc.op_nom["sin2_th13_err"]/nuosc.op_nom["sin2_th13"]
            pulls.append(sin2_13 - nuosc.op_nom['sin2_th13'])
            pull_unc.append(sin2_th13_pull_unc * nuosc.op_nom['sin2_th13'])

        s_juno = ensp_nom_juno['ribd'].GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
        s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
        s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution

        if args_juno.rebin_nonuniform: s_juno.Rebin_nonuniform(args_juno.bins_nonuniform)
 
        sp_obs_tot = ensp_nom_juno["rtot"]
        if args_juno.fit_type == 'geo':
            if args_juno.geo_fit_type == 'UThfree':
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geou"]+ensp_nom_juno["geoth"]
                sp_exp = s_juno.GetScaledFit(Nrea) + ensp_nom_juno["geou"].GetScaledFit(NU)+ensp_nom_juno["geoth"].GetScaledFit(NTh)
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                             ensp_nom_juno['aneu'] + ensp_nom_juno['geou'].GetScaledFit(NU) + ensp_nom_juno['geoth'].GetScaledFit(NTh) + ensp_nom_juno['atm'] + \
                             ensp_nom_juno['rea300']
            elif args_juno.geo_fit_type == 'mantle':
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geomantle"]
                sp_exp = s_juno.GetScaledFit(Nrea) + ensp_nom_juno["geomantle"].GetScaledFit(Nmantle)
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                             ensp_nom_juno['aneu'] + ensp_nom_juno['geocrust'] + ensp_nom_juno['geomantle'].GetScaledFit(Nmantle) + ensp_nom_juno['atm'] + \
                             ensp_nom_juno['rea300']
            else:
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geo"]
                sp_exp = s_juno.GetScaledFit(Nrea) + ensp_nom_juno["geo"].GetScaledFit(Ngeo)
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                             ensp_nom_juno['aneu'] + ensp_nom_juno['geo'].GetScaledFit(Ngeo) + ensp_nom_juno['atm'] + \
                             ensp_nom_juno['rea300']

        else:
            sp_obs = ensp_nom_juno["rdet"]
            sp_exp = s_juno.GetScaledFit(Nrea)
            s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                         ensp_nom_juno['aneu'] + ensp_nom_juno['geo'] + ensp_nom_juno['atm'] + \
                         ensp_nom_juno['rea300'] #+ ensp_nom_juno['nn']

        if bool(args_juno.toymc):
            sp_obs_tot = ensp_nom_juno["rtot_toy"]
        if args_juno.is_data:
            sp_obs_tot = ensp_nom_juno['data']
            
        chi2 = 1e+6

        if args_juno.fit_type == "PMOP_combined":
            s_nd = ensp_nom_nd['ribd'].GetOscillated(L=args_nd.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_nd.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_nd)
            s_nd = s_nd.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
            if args_juno.nd == "TAO": s_nd = s_nd.ApplyDetResp(ene_leak_nd, pecrop=args_juno.ene_crop)
            s_nd = s_nd.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_nd['scintNL'])  # Apply non-linearity
            s_nd = s_nd.ApplyDetResp(rm_nd, pecrop=args_nd.ene_crop)  # Apply energy resolution
            if args_nd.rebin_nonuniform: s_nd.Rebin_nonuniform(args_nd.bins_nonuniform)

            s_tot_nd = s_nd 
            if args_juno.nd == "tao": s_tot_nd = s_tot_nd + ensp_nom_nd['acc'] + ensp_nom_nd['fneu'] + ensp_nom_nd['lihe']

            
            sp_obs_nd = ensp_nom_nd["rtot"]
            
            if unc_juno == 'stat' and unc_nd== 'stat' and unc_corr=='stat':
                unc='stat'
                cm_corr[unc_corr] = []
            else:
                unc='syst'

            if args_juno.nd == "TAO":
                chi2 = PMOP_combined_Chi2(cm_juno[unc_juno], cm_nd[unc_nd], cm_corr[unc_corr], sp_obs_tot,
                      s_tot_juno, sp_obs_nd, s_tot_nd, sp_obs_tot, s_tot_juno,
                      s_obs_nd, s_tot_nd,  unc=unc,
                      stat_meth=args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc, nd=args_juno.nd)
            else:
                chi2_juno = Chi2(cm_juno[unc_juno], sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, unc_juno, args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc, add_logdet=False)
                chi2_dyb = Chi2DYB(dyb_rel_cm, sp_obs_nd, s_tot_nd)
                #print(chi2_juno, chi2_dyb)
                chi2 = chi2_juno + chi2_dyb
            
        else:
            chi2 = Chi2(cm_juno[unc_juno], sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, unc_juno, args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc) 
            
        return chi2


   #fitting stuff
    def chi2_pmop(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31,Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)

    def chi2_pmop_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0,  opp=True)

    def PMOP_combined_chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)
    def PMOP_combined_chi2_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh =1.0,Nmantle=1.0, opp=True)

    def chi2_geo(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Ngeo=1.0): #, a1=0.0, a2=0.0, a3=0.0, a4=0.0):
       return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=Ngeo, Nrea=Nrea,  NU=1.0, NTh=1.0,Nmantle=1.0, opp=False) #eff=float(args_juno.detector_efficiency), me_rho=args_juno.me_rho, a1=0.0, a2=0.0, a3=0.0, a4=0.0)
    def chi2_mantle(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Nmantle=1.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=1.0, Nrea=Nrea,  NU=1.0, NTh=1.0,Nmantle=Nmantle, opp=False)
    def chi2_nomantle(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=1.0, Nrea=Nrea,NU=1.0, NTh=1.0,Nmantle=0.0, opp=False)

    def chi2_UTh(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, NU=1.0, NTh=1.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=Nrea, NU=NU, NTh=NTh,Nmantle=1.0, Ngeo=1.0, opp=False)

    if not args_juno.silent_print:
        print("************Fit configuration************")
        print(f"Fit type: {args_juno.fit_type}")
        if args_juno.fit_type == "geo" :
            print(f"Th/U free: {args_juno.geo_fit_type == 'UThfree'}")
            print(f"Mantle fit: {args_juno.geo_fit_type == 'mantle'}")
            print(f"OPfixed: {args_juno.geo_OPfixed}")
        if args_juno.fit_type == "PMOP_combined" :
            print(f"detector type: {args_juno.nd}")
            
    unc_new_juno = unc_juno
    if(unc_juno != 'stat'): unc_new_juno = 'stat+'+unc_juno
    if not args_juno.silent_print: print("Uncertainty JUNO: ", unc_new_juno)

    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt) #Vals for osc parameters and NMO
    osc_params = ['sin2_12', 'sin2_13', 'dm2_21', 'dm2_31']
    if args_juno.fit_type == "geo":
        if args_juno.geo_fit_type == 'UThfree':
            m = Minuit(chi2_UTh, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], NU=1.0, NTh=1.0, Nrea=1.0)
            m.limits["NU"] = [0, None]
            m.limits["NTh"] = [0, None]
        elif args_juno.geo_fit_type == 'mantle':
            m = Minuit(chi2_mantle, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0, Nmantle=1.0)
            m0 = Minuit(chi2_nomantle, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0)
            m.limits["Nmantle"] = [0, None]
            if args_juno.geo_OPfixed:
                for param in osc_params:
                    m.fixed[param]=True
            m0.migrad()
        else:
            m = Minuit(chi2_geo, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0, Ngeo=1.0)
            m.limits["Ngeo"] = [0, None]
        if args_juno.geo_OPfixed:
            for param in osc_params:
                m.fixed[param] =True

    elif args_juno.fit_type == "PMOP_combined":
        if args_juno.nd == "TAO":
            unc_new_nd = unc_nd
            if(unc_nd != 'stat'): unc_new_nd = 'stat+'+unc_nd
            print("Uncertainty nd: ", unc_new_nd)
            print("Correlated uncertainties: ", unc_corr)


        m = Minuit(PMOP_combined_chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
        m_opp = Minuit(PMOP_combined_chi2_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m_opp.migrad()



    else:
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
        m_opp = Minuit(chi2_pmop_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m_opp.migrad()
        
    start_time = time.time()
    max_time = args_juno.minuit_timeout
    m.migrad()   
    merrors=bool(args_juno.calc_minuit_errors)
    if merrors:
        m.hesse() #get errors
        if not m.fmin.is_valid:
            return None
        m.minos() #get minos errors

    if args_juno.fit_type == "geo"  and args_juno.geo_fit_type == 'mantle':
        chi2_min = chi2_mantle(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3], Nrea=m.values[4],Nmantle=m.values[5])
        chi2_mantle0 = chi2_nomantle(sin2_12=m0.values[0], sin2_13=m0.values[1], dm2_21=m0.values[2], dm2_31=m0.values[3], Nrea=m0.values[4])
        # chi2_min = chi2_mantle_OPfixed(Nrea=m.values[0], Nmantle=m.values[1])
        #chi2_mantle0 = chi2_nomantle_OPfixed(Nrea=m0.values[0])
        dchi2 = chi2_mantle0 - chi2_min
        if not args_juno.silent_print: print(f"Mantle discovery potential: delta chi2 between mantle and no mantle: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    else:
        if args_juno.fit_type == "PMOP_combined":
            chi2_min = PMOP_combined_chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
            chi2_min_opp = PMOP_combined_chi2_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])
        else:
            chi2_min = chi2_pmop(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
            chi2_min_opp = chi2_pmop_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])

        dchi2 = abs(chi2_min-chi2_min_opp)
        print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")
    
    if not args_juno.silent_print:
        print("************Fit results************")
        for i in range(len(m.values)):
            if merrors:
                err =  max(abs(m.merrors[i].upper), abs(m.merrors[i].lower))
            else:
                err = m.errors[i]
                
            print(f"{m.parameters[i]}: {m.values[i]} +/- {err}; rel unc: {err/m.values[i]}")
        #print(m)


    #chi2_min = chi2_geo(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3], Nrea=m.values[4], Ngeo=m.values[5])
    #print(chi2_min)
    if args_juno.write_results:
        extra = [] #[args_juno.geo_tnu_rate]
        if args_juno.fit_type == 'geo' and args_juno.geo_fit_type == 'mantle':
            extra.extend([dchi2, np.sqrt(dchi2), args_juno.mantle_model,  args_juno.crust_rate_tnu, args_juno.crust_rate_unc])
        if args_juno.fit_type == 'geo' and args_juno.geo_fit_type != 'mantle':
            extra.extend([args_juno.geo_tnu_rate])
        if args_juno.fit_type == "PMOP" or args_juno.fit_type == "PMOP_combined":
            extra.extend([dchi2, np.sqrt(dchi2)])
        if bool(args_juno.toymc):
            extra.extend([ensp_nom_juno['rtot_toy'].GetIntegral()])
            #minos_err = max(abs(m.merrors[5].upper), abs(m.merrors[5].lower))
            #hesse_err = m.errors[5]
            #print(abs(minos_err-hesse_err))#*100./minos_err)
            return get_toy_results(m, extra, merrors)
            
        extra_name = ''
        if not args_juno.geo_fit_type == 'total': extra_name = f'{args_juno.geo_fit_type}_'
        if args_juno.fit_type == "NMO" and args_juno.include_nd: extra_name = '_nd_'
        filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.fit_type}_{extra_name}{args_juno.stat_method_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.txt"

        write_results(m, filename, extra, merrors=merrors, silent=args_juno.silent_print) #write results into a textfile
   #fancy stuff
    if(args_juno.plot_minuit_matrix or args_juno.plot_minuit_profiles): #make plots folders
        if not os.path.exists(f"{args_juno.plots_folder}/Chi2_profiles"): os.mkdir(f"{args_juno.plots_folder}/Chi2_profiles")
        if not os.path.exists(f"{args_juno.plots_folder}/Chi2_profiles/Minuit"): os.mkdir(f"{args_juno.plots_folder}/Chi2_profiles/Minuit")

    if(args_juno.plot_minuit_profiles): #create chi2 profiles
        print("Plotting chi2 profiles")
        param_list = m.parameters
        for i in range(len(m.parameters)): #there is something weird with draw_mnprofile in minuit, so I have to do this from scratch inside plot_profile
            plotname = f"{args_juno.plots_folder}/Chi2_profiles/Minuit/chi2_{args_juno.stat_opt}_{param_list[i]}_{unc_juno}.png"
            plot_profile(m, i, param_list[i], plotname)

    if(args_juno.print_minuit_correlation):
        print("Correlation co-efficient between parameters")
        print(m.covariance.correlation())

    if (args_juno.plot_minuit_matrix):
        print("Plotting matrix")
        fig, ax = m.draw_mnmatrix(cl=[1,2,3])
        plt.show()
        #plt.savefig(f"{args_juno.plots_folder}/Chi2_profiles/Minuit/matrix_{args_juno.stat_opt}_{unc_juno}.png")
    del m

