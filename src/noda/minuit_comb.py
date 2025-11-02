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
from numba import njit



def write_results(m, filename, extra, merrors=True, silent=False): #writes out fit results m for a given filename and uncertainty
    values = []
    for i in range(len(m.values)):
        values.append(m.values[i]) #central values
        values.append(m.errors[i]) #hesse errors
        if merrors and  i < 4:
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

def run_minuit(ndays,ndays_nd=0, ensp_nom_juno={}, ensp_nom_nd={},  unc_juno='', unc_nd='', unc_corr='', rm= [], rm_nd=[], ene_leak_nd =[], cm_juno ={}, cm_nd={}, cm_corr='', args_juno='', args_nd='', dyb_model=None):#, dm2_31=0.):
   # a_nom, b_nom, c_nom =args_juno.a, args_juno.b, args_juno.c
#    crust = np.random.normal(loc=0., scale=0.1)
    E_edges = make_nonuniform_bins(args_juno.bins_nonuniform)
   
    n_params = np.zeros(len(E_edges) -1)
    fine_bins = ensp_nom_juno["rfis0"].bins
    fine_contents = ensp_nom_juno["rfis0"].bin_cont
        
    s_ibd = sp.interpolate.interp1d(ensp_nom_juno['sibd'].GetBinCenters(), ensp_nom_juno['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['sibd'].bin_cont[0], ensp_nom_juno['sibd'].bin_cont[-1]))
    s_bump_lin = sp.interpolate.interp1d(ensp_nom_juno['bump_corr'].GetBinCenters(), ensp_nom_juno['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['bump_corr'].bin_cont[0], ensp_nom_juno['bump_corr'].bin_cont[-1]))

    if dyb_model: 
        DYB_edges = make_nonuniform_bins(args_nd.bins_nonuniform)
        from dag_modelling.tools.make_fcn import make_fcn
        flux_params = dyb_model.storage["parameters.free.neutrino_per_fission_factor"]
        chi2_dyb_func = make_fcn(dyb_model.storage[f"outputs.statistic.full.pull.chi2cnp"], parameters=[par for par in flux_params.walkvalues()], safe=False)

    
    def chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea, Nrea_nd, *nuisance, Ngeo=1.0,  NU=1.0, NTh=1.0, Nmantle=1.0, opp=False): #, a1=0.0, a2=0.0, a3=0.0, a4=0.0, me_rho=args_juno.me_rho):

        pulls=[]
        pull_unc = []
        if args_juno.sin2_th13_opt=="pull":
            sin2_th13_pull_unc = nuosc.op_nom["sin2_th13_err"]/nuosc.op_nom["sin2_th13"]
            pulls.append(sin2_13 - nuosc.op_nom['sin2_th13'])
            pull_unc.append(sin2_th13_pull_unc * nuosc.op_nom['sin2_th13'])
        pulls.append(Nrea)
        pull_unc.append(args_juno.r2_unc)

        for n in nuisance[0]:  
           pulls.append(n)
           pull_unc.append(3.0)#0.447)       

        nuisance_coarse = np.array(nuisance[0])  # len = len(coarse_edges)-1
        
        nuisance_fine = expand_nuisances_to_fine(fine_bins, E_edges, nuisance_coarse)


        rfis_param = Spectrum(bins=ensp_nom_juno["rfis0"].bins, bin_cont=ensp_nom_juno["rfis0"].bin_cont * (1+np.array(nuisance_fine)))

    
        # # Apply IBD weighting
        rfis_param.WeightWithFunction(s_ibd)

        # # Apply bump correction if needed
        rfis_bparam = rfis_param.GetWeightedWithFunction(s_bump_lin)
        rfis_bparam.GetScaled(ndays)

        rfis_bparam.GetScaled(compute_extrafactors(args_juno))
        

        ibd = rfis_bparam + ensp_nom_juno["snf"] + ensp_nom_juno["noneq"]

        s_juno =ibd.GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
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
            sp_exp = s_juno.GetScaledFit(1+Nrea)
            s_tot_juno = s_juno.GetScaledFit(1+Nrea) + ensp_nom_juno['acc'] + ensp_nom_juno['fneu'] + ensp_nom_juno['lihe'] + \
                         ensp_nom_juno['aneu'] + ensp_nom_juno['geo'] + ensp_nom_juno['atm'] + \
                         ensp_nom_juno['rea300'] #+ ensp_nom_juno['nn']

        if bool(args_juno.toymc):
            sp_obs_tot = ensp_nom_juno["rtot_toy"]
        if args_juno.is_data:
            sp_obs_tot = ensp_nom_juno['data']
            
        chi2 = 1e+6

        if args_juno.fit_type == "PMOP_combined":
           

            if not args_juno.use_dagflow:
                nuisance_nd = expand_nuisances_to_fine(ensp_nom_nd["rfis0"].bins, E_edges, nuisance_coarse)
                rfis_param_nd = Spectrum(bins=ensp_nom_nd["rfis0"].bins, bin_cont=ensp_nom_nd["rfis0"].bin_cont * (1+np.array(nuisance_nd)))
                rfis_param_nd.WeightWithFunction(s_ibd)
                rfis_bparam_nd = rfis_param_nd.GetWeightedWithFunction(s_bump_lin)
    
                rfis_bparam_nd.GetScaled(ndays_nd)
                rfis_bparam_nd.GetScaled(compute_extrafactors(args_nd))
                
                
                ibd_nd = rfis_bparam_nd + ensp_nom_nd["snf"] + ensp_nom_nd["noneq"]
    
                s_nd = ibd_nd
                if args_juno.nd == "TAO": s_nd = s_nd.GetOscillated(L=args_nd.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                              dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_nd.core_powers,
                                              me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_nd)
                s_nd = s_nd.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
             
                #s_nd = s_nd.ApplyDetResp(ene_leak_nd, pecrop=args_nd.ene_crop)
                if args_juno.nd == "TAO": s_nd = s_nd.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_nd['scintNL'])  # Apply non-linearity
                s_nd = s_nd.ApplyDetResp(rm_nd, pecrop=args_nd.ene_crop)  # Apply energy resolution
                if args_nd.rebin_nonuniform: s_nd.Rebin_nonuniform(args_nd.bins_nonuniform)

                s_tot_nd = s_nd.GetScaledFit(1+Nrea_nd)
                if args_juno.nd == "TAO": s_tot_nd = s_tot_nd + ensp_nom_nd['acc'] + ensp_nom_nd['fneu'] + ensp_nom_nd['lihe']

    
            
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
                
                chi2_juno = Chi2(cm_juno[unc_juno], sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, unc_juno, args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc)
               
                if not args_juno.use_dagflow:
                    chi2_dyb =  Chi2(cm_nd['all'], sp_obs_nd, s_tot_nd, sp_obs_nd, s_tot_nd, 'no stat', args_juno.stat_method_opt, pulls=[Nrea_nd], pull_unc=[args_nd.rea_rate_unc])
                else:
                    nuisance_nd = expand_nuisances_to_fine(DYB_edges, E_edges, nuisance_coarse)
   
                        
                    chi2_dyb = chi2_dyb_func(*nuisance_nd)[0]
                #print(chi2_juno, chi2_dyb)
                chi2 = chi2_juno + chi2_dyb
                
            
        else:
            chi2 = Chi2(cm_juno[unc_juno], sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, unc_juno, args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc) 
            
        return chi2
    
    def plot_fit_spectrum(sin2_12, sin2_13, dm2_21, dm2_31, Nrea, Nrea_nd, nuisance, opp=False):


        nuisance_coarse = np.array(nuisance)  # len = len(coarse_edges)-1
        
        nuisance_fine = expand_nuisances_to_fine(fine_bins, E_edges, nuisance_coarse)


        rfis_param = Spectrum(bins=ensp_nom_juno["rfis0"].bins, bin_cont=ensp_nom_juno["rfis0"].bin_cont * (1+np.array(nuisance_fine)))

    
        # # Apply IBD weighting
        rfis_param.WeightWithFunction(s_ibd)

        # # Apply bump correction if needed
        rfis_bparam = rfis_param.GetWeightedWithFunction(s_bump_lin)
        rfis_bparam.GetScaled(ndays)

        rfis_bparam.GetScaled(compute_extrafactors(args_juno))
        

        ibd = rfis_bparam + ensp_nom_juno["snf"] + ensp_nom_juno["noneq"]

        s_juno =ibd.GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
        s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])  # Apply non-linearity
 
        s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution

        if args_juno.rebin_nonuniform: s_juno.Rebin_nonuniform(args_juno.bins_nonuniform)

        s_juno.Plot(f"{args_juno.plots_folder}/fit_spectrum_JUNO+DYB_nom_scaled_500p.png",
                  xlabel="Reconstructed energy (MeV)",
                  ylabel=f"", #Events per {binw:0.1f} keV",
                  extra_spectra=[ensp_nom_juno['rdet'], ensp_nom_juno['rdet1']],
                  leg_labels=['Fitted spectrum', 'Observation (RenShape)', 'Expectation (Common Inputs)'],
                  colors=['black', 'peru', 'steelblue'],
                  xmin=0.8, xmax=12.0,
                  ymin=1e-3,log_scale=False)

   #fitting stuff
    def chi2_pmop(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31,Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)

    def chi2_pmop_opp(sin2_12=0, sin2_13=0, dm2_21=0, dm2_31=0):
        return chi2(sin2_12=sin2_12, sin2_13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31, Nrea=1.0, Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0,  opp=True)

    def PMOP_combined_chi2(*all_params):
        sin2_th12, sin2_th13, dm2_21, dm2_31, Nrea, Nrea_nd,*nuisance = all_params
        return chi2(sin2_th12, sin2_th13, dm2_21, dm2_31, Nrea, Nrea_nd, nuisance,  Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=False)
        
    def PMOP_combined_chi2_opp(*all_params):
        sin2_th12, sin2_th13, dm2_21, dm2_31, *nuisance = all_params
        return chi2(sin2_th12, sin2_th13, dm2_21, dm2_31, nuisance,  Ngeo=1.0, NU=1.0, NTh=1.0, Nmantle=1.0, opp=True)

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
            
        all_params = [
            nuosc.op_nom["sin2_th12"],
            nuosc.op_nom["sin2_th13"],
            nuosc.op_nom["dm2_21"],
            nuosc.op_nom["dm2_31"],
            0.0,
            0.0,
        ] + list(n_params)
        
        m = Minuit(PMOP_combined_chi2, *all_params)
        # new_names = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"] \
        #     + [f"n{i}" for i in range(len(m.parameters) - 4)]

        # m.parameters = new_names

        if args_juno.nd == "TAO":
            nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
            all_params = [
                nuosc.op_nom["sin2_th12"],
                nuosc.op_nom["sin2_th13"],
                nuosc.op_nom["dm2_21"],
                nuosc.op_nom["dm2_31"],
            ] + list(n_params)
            
            m_opp = Minuit(PMOP_combined_chi2_opp, *all_params)
            m_opp.migrad()
    
        for p in m.parameters[4:4+len(all_params)]:
            m.limits[p] = [-1, None]

        if args_juno.nd == "DYB" and args_juno.use_dagflow:
            m.fixed["x5"] = True
    # for p in m.parameters[5:5+len(n_params)]:
    #     m.fixed[p] = True


    else:
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m = Minuit(chi2_pmop, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO= not args_juno.NMO_opt) #Vals for osc parameters and NMO
        m_opp = Minuit(chi2_pmop_opp, sin2_12= nuosc.op_nom["sin2_th12"], sin2_13= nuosc.op_nom["sin2_th13"],  dm2_21=nuosc.op_nom["dm2_21"], dm2_31=nuosc.op_nom["dm2_31"])
        m_opp.migrad()
        
    start_time = time.time()
    max_time = args_juno.minuit_timeout
  
    for p in m.parameters[5:5+len(n_params)]:
        m.fixed[p] = True

    # m.migrad()

    # print(m.fmin.is_valid) 

    # for p in m.parameters[:4]:
    #     m.values[p] = m.values[p]
    # for p in m.parameters[5:5+len(n_params)]:
    #     m.fixed[p] = False
        
    # m.strategy = 0   
    # m.tol = 1e-2  
    m.migrad()
    

    


    #plot_fit_spectrum(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5], m.values[5:5+len(n_params)])


    print(m.fmin.is_valid)     
    print(m.fmin.has_reached_call_limit)
    print(m.fmin.is_above_max_edm)

    print(m)
    merrors=bool(args_juno.calc_minuit_errors)
    if merrors:
        #m.hesse() #get errors
        # if not m.fmin.is_valid:
        #     print("not valid")
        #     return None
        m.minos("x0", "x1", "x2", "x3") #(params=['x0', 'x1', 'x2', 'x3']) #m.minos() #get minos errors

    if args_juno.fit_type == "geo"  and args_juno.geo_fit_type == 'mantle':
        chi2_min = chi2_mantle(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3], Nrea=m.values[4],Nmantle=m.values[5])
        chi2_mantle0 = chi2_nomantle(sin2_12=m0.values[0], sin2_13=m0.values[1], dm2_21=m0.values[2], dm2_31=m0.values[3], Nrea=m0.values[4])
        # chi2_min = chi2_mantle_OPfixed(Nrea=m.values[0], Nmantle=m.values[1])
        #chi2_mantle0 = chi2_nomantle_OPfixed(Nrea=m0.values[0])
        dchi2 = chi2_mantle0 - chi2_min
        if not args_juno.silent_print: print(f"Mantle discovery potential: delta chi2 between mantle and no mantle: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    else:
        if args_juno.fit_type == "PMOP_combined":
            if args_juno.nd == "TAO":
                chi2_min = PMOP_combined_chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
                chi2_min_opp = PMOP_combined_chi2_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])        
                dchi2 = abs(chi2_min-chi2_min_opp)
                print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")
            
        else:
            chi2_min = chi2_pmop(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
            chi2_min_opp = chi2_pmop_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])

            dchi2 = abs(chi2_min-chi2_min_opp)
            print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")
        
    if not args_juno.silent_print:
        print("************Fit results************")
        for i in range(4): #len(m.values)):
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
        if args_juno.fit_type == "PMOP_combined" and args_juno.nd=="TAO":
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

