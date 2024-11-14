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

def write_results(m, filename, extra, merrors=True): #writes out fit results m for a given filename and uncertainty
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
    #fileo.close()
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
    plt.axvspan(m.values[i]+m.merrors[i].lower,m.values[i]+m.merrors[i].upper, color='gray', alpha=0.3, label='Vertical Band')
    m_err, p_err = round_errors(param, m.merrors[i].lower, m.merrors[i].upper)
    plt.title(f'{param} {m.values[i]} {m_err} + {p_err}')
    plt.ylabel('FCN')
    plt.xlabel(param)
    plt.savefig(plotname)
    plt.close()

def run_minuit(ensp_nom_juno={}, ensp_nom_tao={},  unc_juno='', unc_tao='', unc_corr='', rm= [], ene_leak_tao =[], cm_juno ={}, cm_tao={}, cm_corr='', args_juno='', args_tao=''):#, dm2_31=0.):

    def chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Ngeo=1.0, Nrea=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False):#, a1=0.0, a2=0.0, a3=0.0, a4=0.0, eff=float(args_juno.detector_efficiency), me_rho=args_juno.me_rho):
        # Get the oscillated spectrum
        # ebins = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins)
        # s_juno = ensp_nom_juno['rfis'].GetScaledFit(eff/float(args_juno.detector_efficiency))
        # s_juno_snf = GetSpectrumFromROOT(args_juno.input_data_file, 'SNF_FluxRatio').Rebin(ebins, mode='spline-not-keep-norm').GetWeightedWithSpectrum(s_juno)
        # s_juno_noneq = GetSpectrumFromROOT(args_juno.input_data_file, 'NonEq_FluxRatio').Rebin(ebins, mode='spline-not-keep-norm').GetWeightedWithSpectrum(s_juno)
        # s_juno =  s_juno + s_juno_snf + s_juno_noneq
        # s_juno = s_juno.GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
        #                                   dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
        #                                   me_rho=me_rho, ene_mode='true', opp=opp, args=args_juno)

        s_juno = ensp_nom_juno['ribd'].GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy

        # new_nonl =  Spectrum(bins = ensp_nom_juno['scintNL'].bins, bin_cont=np.zeros(len(ensp_nom_juno['scintNL'].bin_cont)))
        # new_nonl.bin_cont = ensp_nom_juno['scintNL'].bin_cont + a1*(ensp_nom_juno['NL_pull'][0].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
        #                    +a2*(ensp_nom_juno['NL_pull'][1].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
        #                    +a3*(ensp_nom_juno['NL_pull'][2].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
        #                    +a4*(ensp_nom_juno['NL_pull'][3].bin_cont - ensp_nom_juno['scintNL'].bin_cont)
        s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
        s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution

        #s_geou = ensp_nom_juno['rfis0_geou'].GetScaledFit(eff/float(args_juno.detector_efficiency)).GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2).GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
        #s_geou = ensp_nom_juno['rvis_geou'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
        #s_geou = s_geou.ApplyDetResp(rm, pecrop=args_juno.ene_crop)
        #s_geoth = ensp_nom_juno['rfis0_geoth'].GetScaledFit(eff/float(args_juno.detector_efficiency)).GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2).GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
        #s_geoth = ensp_nom_juno['rvis_geoth'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
        #s_geoth = s_geoth.ApplyDetResp(rm, pecrop=args_juno.ene_crop)
        #s_geo = s_geou + s_geoth

        # print(ensp_nom_juno['rdet'].bin_cont*np.array(ratio))
        # print(ensp_nom_juno['rdet'].bin_cont)
        #s_juno = Spectrum(bins=s_juno.bins, bin_cont=s_juno.bin_cont*np.array(ratio))
        if args_juno.fit_type == 'geo':
            if args_juno.geo_fit_type == 'UThfree':
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                    ensp_nom_juno['aneu'] + ensp_nom_juno['geou'].GetScaledFit(NU) + ensp_nom_juno['geoth'].GetScaledFit(NTh) + ensp_nom_juno['atm'] + \
                    ensp_nom_juno['rea300']
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geou"]+ensp_nom_juno["geoth"]
                sp_exp = s_juno+ensp_nom_juno["geou"].GetScaledFit(NU)+ensp_nom_juno["geoth"].GetScaledFit(NTh)
            elif args_juno.geo_fit_type == 'mantle':
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                ensp_nom_juno['aneu'] + ensp_nom_juno['geocrust'] + ensp_nom_juno['geomantle'].GetScaledFit(Nmantle) + ensp_nom_juno['atm'] + \
                ensp_nom_juno['rea300']
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geomantle"]
                sp_exp = s_juno+ensp_nom_juno["geomantle"].GetScaledFit(Nmantle)
            else:
                s_tot_juno = s_juno.GetScaledFit(Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                ensp_nom_juno['aneu'] + ensp_nom_juno['geo'].GetScaledFit(Ngeo) + ensp_nom_juno['atm'] + \
                ensp_nom_juno['rea300']
                sp_obs = ensp_nom_juno["rdet"]+ensp_nom_juno["geo"]
                sp_exp = s_juno+ensp_nom_juno["geo"].GetScaledFit(Ngeo)
        else:
            s_tot_juno = s_juno + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                ensp_nom_juno['aneu'] + ensp_nom_juno['geo'] + ensp_nom_juno['atm'] + \
                ensp_nom_juno['rea300']
            sp_obs = ensp_nom_juno["rdet"]
            sp_exp = s_juno

        chi2 = 1e+6

        if args_juno.fit_type == "NMO" and args_juno.include_TAO:
            s_tao = ensp_nom_tao['ribd'].GetOscillated(L=args_tao.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_tao.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_tao)
            s_tao = s_tao.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
            s_tao = s_tao.ApplyDetResp(ene_leak_tao, pecrop=args_juno.ene_crop)
            s_tao = s_tao.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
            s_tao = s_tao.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution
            s_tot_tao = s_tao + ensp_nom_tao['acc'] + ensp_nom_tao['fneu'] + ensp_nom_tao['lihe']

            if unc_juno == 'stat' and unc_tao== 'stat' and unc_corr=='stat':
                unc='stat'
                cm_corr[unc_corr] = []
            else:
                unc='syst'

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
                chi2 = Chi2(cm_juno[unc_juno], ensp_nom_juno["rtot"], s_tot_juno,  sp_obs, sp_exp, unc_juno,
                      args_juno.stat_method_opt, pulls=[sin2_13 - nuosc.op_nom['sin2_th13']],
                      pull_unc=[args_juno.sin2_th13_pull_unc * nuosc.op_nom['sin2_th13']])
            else:
                chi2 = Chi2(cm_juno[unc_juno], ensp_nom_juno["rtot"], s_tot_juno, sp_obs, sp_exp, unc_juno, args_juno.stat_method_opt)
                #\, pulls=[eff-float(args_juno.detector_efficiency)],pull_unc=[args_juno.eff_unc*float(args_juno.detector_efficiency)] )#, pulls=[me_rho-args_juno.me_rho], pull_unc=[args_juno.me_rho_scale * args_juno.me_rho]) , puls=[a1, a2, a3, a4, a4], pulls_unc=[1.])

        return chi2


   #fitting stuff
    def chi2_pmop(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)

    def chi2_pmop_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=True)

    def combined_chi2(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)
    def combined_chi2_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh =1.0,Nmantle=1.0, opp=True)

    def chi2_geo(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Ngeo=1.0): #, a1=0.0, a2=0.0, a3=0.0, a4=0.0):
       return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=Ngeo, Nrea=Nrea,  NU=1.0, NTh=1.0,Nmantle=1.0, opp=False) #eff=float(args_juno.detector_efficiency), me_rho=args_juno.me_rho, a1=0.0, a2=0.0, a3=0.0, a4=0.0)

    def chi2_mantle(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, Nmantle=1.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=1.0, Nrea=Nrea,  NU=1.0, NTh=1.0,Nmantle=Nmantle, opp=False)
    def chi2_nomantle(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0): #, a1=0.0, a2=0.0, a3=0.0, a4=0.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Ngeo=1.0, Nrea=Nrea,  NU=1.0, NTh=1.0,Nmantle=0.0, opp=False)

    def chi2_UTh(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, Nrea=1.0, NU=1.0, NTh=1.0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=Nrea, NU=NU, NTh=NTh,Nmantle=1.0, Ngeo=1.0, opp=False)

    print("************Fit configuration************")
    print(f"Fit type: {args_juno.fit_type}")
    if args_juno.fit_type == "geo" :
        print(f"Th/U free: {args_juno.geo_fit_type == 'UThfree'}")
        print(f"Mantle fit: {args_juno.geo_fit_type == 'mantle'}")
        print(f"OPfixed: {args_juno.geo_OPfixed}")
    if args_juno.fit_type == "NMO" :
        print(f"TAO inclusion: {args_juno.include_TAO}")

    unc_new_juno = unc_juno
    if(unc_juno != 'stat'): unc_new_juno = 'stat+'+unc_juno
    print("Uncertainty JUNO: ", unc_new_juno)

    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt) #Vals for osc parameters and NMO
    osc_params = ['sin2_12', 'sin2_13', 'dm2_21', 'dm2_31']
    if args_juno.fit_type == "geo":
        if args_juno.geo_fit_type == 'UThfree':
            m = Minuit(chi2_UTh, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], NU=1.0, NTh=1.0, Nrea=1.0)
        elif args_juno.geo_fit_type == 'mantle':
            m = Minuit(chi2_mantle, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0, Nmantle=1.0)
            m0 = Minuit(chi2_nomantle, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0)
            if args_juno.geo_OPfixed:
                for param in osc_params:
                    m.fixed[param]=True
            m0.migrad()
        else:
            m = Minuit(chi2_geo, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], Nrea=1.0, Ngeo=1.0)
        if args_juno.geo_OPfixed:
            for param in osc_params:
                m.fixed[param] =True

    elif args_juno.fit_type == "NMO":
        if args_juno.include_TAO:
            unc_new_tao = unc_tao
            if(unc_tao != 'stat'): unc_new_tao = 'stat+'+unc_tao
            print("Uncertainty TAO: ", unc_new_tao)
            print("Correlated uncertainties: ", unc_corr)

            m = Minuit(combined_chi2, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
            nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
            m_opp = Minuit(combined_chi2_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
            m_opp.migrad()

        else:
            m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
            nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
            m_opp = Minuit(chi2_pmop_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
            m_opp.migrad()

    else:
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])

    m.migrad() #fit

    if args_juno.calc_minuit_errors:
        m.hesse() #get errors
        m.minos() #get minos errors

    if args_juno.fit_type == "geo"  and args_juno.geo_fit_type == 'mantle':
        chi2_min = chi2_mantle(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3], Nrea=m.values[4], Nmantle=m.values[5])
        chi2_mantle0 = chi2_nomantle(sin2_12=m0.values[0], sin2_13=m0.values[1], dm2_21=m0.values[2], dm2_31=m0.values[3], Nrea=m0.values[4])
        # chi2_min = chi2_mantle_OPfixed(Nrea=m.values[0], Nmantle=m.values[1])
        #chi2_mantle0 = chi2_nomantle_OPfixed(Nrea=m0.values[0])
        dchi2 = chi2_mantle0 - chi2_min
        print(f"Mantle discovery potential: delta chi2 between mantle and no mantle: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    if args_juno.fit_type == "NMO" :
        if args_juno.include_TAO:
            chi2_min = combined_chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
            chi2_min_opp = combined_chi2_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])
        else:
            chi2_min = chi2_pmop(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
            chi2_min_opp = chi2_pmop_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])

        dchi2 = abs(chi2_min-chi2_min_opp)
        print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    print("************Fit results************")
    for i in range(len(m.values)):
        print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")
    print(m)

    if args_juno.write_results:
        merrors=True
        extra = [unc_new_juno]
        extra_name = ''
        if not args_juno.geo_fit_type == 'total': extra_name = f'_args_juno.geo_fit_type_'
        if args_juno.fit_type == "NMO" and args_juno.include_TAO: extra_name = '_TAO_'
        filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.fit_type}{extra_name}{args_juno.stat_method_opt}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.txt"
        fileo = open(filename, "a")
        if unc_new_juno==args_juno.unc_list_juno[0]:
            if args_juno.fit_type == "NMO":
                fileo.write("sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr unc dchi2 sigma\n")
            elif args_juno.fit_type == 'geo':
                if args_juno.geo_fit_type == 'UThfree':
                    fileo.write("sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr Nrea Nrea_err Nrea_merr Nrea_perr NgeoU NgeoU_err NgeoU_merr NgeoU_perr NgeoTh NgeoTh_err NgeoTh_merr NgeoTh_perr unc\n")
                elif args_juno.geo_fit_type == 'mantle':
                    fileo.write("sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr Nrea Nrea_err Nrea_merr Nrea_perr Nmantle Nmantle_err Nmantle_merr Nmantle_perr unc dchi2 sigma model crust_rate crust_unc\n")
                else:
                    fileo.write("sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr Nrea Nrea_err Nrea_merr Nrea_perr Ngeo Ngeo_err Ngeo_merr Ngeo_perr unc\n")
            else:
                fileo.write("sin2_12 sin2_12_err sin2_12_merr sin2_12_perr sin2_13 sin2_13_err sin2_13_merr sin2_13_perr dm2_21 dm2_21_err dm2_21_merr dm2_21_perr dm2_31 dm2_31_err dm2_31_merr dm2_31_perr unc\n")

        if args_juno.fit_type == 'geo' and args_juno.geo_fit_type == 'mantle':
            extra.extend([dchi2, np.sqrt(dchi2), args_juno.mantle_model,  args_juno.crust_rate, args_juno.crust_rate_unc])
        if args_juno.fit_type == "NMO":
            extra.extend([dchi2, np.sqrt(dchi2)])
        fileo.close()
        write_results(m, filename, extra, merrors=merrors) #write results into a textfile

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

    # pts1 = m.mncontour("NU", "NTh", cl=0.68, size=10, interpolated=100)
    # pts2 = m.mncontour("NU", "NTh", cl=0.95, size=10, interpolated=100)
    # #pts3 = m.mncontour("NU", "NTh", cl=99.7, size=10, interpolated=100)
    #
    # x1, y2 = np.transpose(pts1)
    # x3, y4 = np.transpose(pts2)
    # #x5, y6 = np.transpose(pts3)
    # plt.plot(x1, y2, "-",)
    # plt.plot(x3, y4, "-",)
    # #plt.plot(x5, y6, "-",)
    # plt.xlabel("NU", fontsize=18)
    # plt.ylabel("NTh", fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # #fig, ax = m.draw_mncontour("NU", "NTh", cl=(0.68, 0.9, 0.99), size=20, interpolated=100);
    # plt.show()

        #For drawing
     #    def get_spectrum(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0, opp=False):
     #        s = ensp_nom_juno['ribd'].GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers, me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
     #        s = s.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2) #shift to positron energy
     #        s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL']) #apply non-linearity
     #        s = s.ApplyDetResp(rm, pecrop=args_juno.ene_crop) #apply energy resolution
     #        return s
     # #plot IO and NO

    #     nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt) #Vals for osc parameters and NMO
        #NO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=False)
    #     nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
        #IO_sp = get_spectrum(sin2_12=nuosc.op_nom["sin2_th12"], sin2_13=nuosc.op_nom["sin2_th13"], dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"], opp=True)
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
        #NO_sp.Plot(f"{args_juno.plots_folder}/NO_vs_IO.png",
               # xlabel="Neutrino energy (MeV)",
               # ylabel=f"Events per 20 keV",
               # extra_spectra=[IO_sp],
               # leg_labels=['NO curve', 'IO curve'],
               # colors=['darkred', 'steelblue'],
               # xmin=0, xmax=10,
               # ymin=0, ymax=None, log_scale=False)
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
