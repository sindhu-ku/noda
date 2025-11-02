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
import inspect

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_fit_residuals(s_tot_juno, s_obs, s_juno, bckgs, ndays):
    """
    Plot observed vs. fitted spectra with residuals in σ units, 
    including reactor and all background components.

    Parameters
    ----------
    s_tot_juno : Spectrum
        Total fitted spectrum (reactor + all backgrounds)
    s_obs : Spectrum
        Observed spectrum (data)
    s_juno : Spectrum
        Reactor-only fitted spectrum
    bckgs : dict[str, Spectrum]
        Dictionary of background spectra (key = name, value = Spectrum instance)
    ndays : float
        Number of live days (used for cpd display)
    """

    os.makedirs("Plots", exist_ok=True)

    bin_centers = 0.5 * (s_tot_juno.bins[:-1] + s_tot_juno.bins[1:])

    # --- Extract data ---
    obs_vals = s_obs.bin_cont
    obs_err = np.sqrt(np.maximum(obs_vals, 1e-12))  # Poisson uncertainties
    fit_vals = s_tot_juno.bin_cont
    fit_err = np.sqrt(np.maximum(fit_vals, 1e-12))  # approximate model uncertainty

    # --- Compute residuals (in σ units) ---
    residuals = (obs_vals - fit_vals) / np.maximum(obs_err, 1e-12)
    residual_err = np.ones_like(residuals)  # 1σ uncertainty (unit pulls)

    # --- Prepare figure ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # === Top panel: Data and Fit ===
    ax1.errorbar(bin_centers, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label='Observed data')

    # Total fit (reactor + backgrounds)
    ax1.step(bin_centers, fit_vals, where='mid', color='blue', lw=1.5, label='Total Fit')
    ax1.fill_between(bin_centers, fit_vals - fit_err, fit_vals + fit_err,
                     color='blue', alpha=0.25, label='Fit uncertainty ±1σ')

    # Reactor-only component
    ax1.step(bin_centers, s_juno.bin_cont, where='mid', color='red', lw=1.2,
             linestyle='-', label='Reactor expectation')

    # Background components
    colors = plt.cm.tab10.colors
    for i, (key, spec) in enumerate(bckgs.items()):
        cpd = spec.GetIntegral() / ndays
        ax1.step(bin_centers, spec.bin_cont, where='mid',
                 color=colors[i % len(colors)], linestyle='--', lw=1,
                 label=f'{key}: {cpd:.3f} cpd')

    # --- Labels and legend ---
    ax1.set_ylabel("Events")
    #ax1.set_title("Fit Results with Reactor and Background Components")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # === Bottom panel: Residuals (in σ) ===
    ax2.errorbar(bin_centers, residuals, yerr=residual_err,
                 fmt='o', color='blue', markersize=3, label='Residuals')
    ax2.axhline(0, color='k', linestyle='--', lw=1)

    # --- Blue σ bands (±1, ±2, ±3σ) ---
    sigma_colors = ['#1f77b4', '#5fa8d3', '#a6cee3']  # dark → light blue
    sigmas = [1, 2, 3]
    for nsigma, color, alpha in zip(sigmas, sigma_colors, [0.25, 0.2, 0.15]):
        ax2.fill_between(bin_centers, -nsigma, nsigma, color=color, alpha=alpha,
                         label=f'±{nsigma}σ')

    # --- Axes labels and styling ---
    ax2.set_xlabel("Reconstructed Energy [MeV]")
    ax2.set_ylabel("Residuals [σ]")
    #ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Plots/fit_residuals.png", dpi=200)
    plt.close(fig)



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

def run_minuit(ndays,ndays_nd=0, ensp_nom_juno={}, ensp_nom_nd={},  unc_juno='', unc_nd='', unc_corr='', rm= [], rm_nd=[], ene_leak_nd =[], cm_juno ={}, cm_nd={}, cm_corr='', args_juno='', args_nd='', dyb_model=None, backgrounds=None):

    
    E_edges = make_nonuniform_bins(args_juno.seg_bins_nonuniform)
    fine_bins = ensp_nom_juno["rfis0"].bins
        
    s_ibd = sp.interpolate.interp1d(ensp_nom_juno['sibd'].GetBinCenters(), ensp_nom_juno['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['sibd'].bin_cont[0], ensp_nom_juno['sibd'].bin_cont[-1]))
    s_bump_lin = sp.interpolate.interp1d(ensp_nom_juno['bump_corr'].GetBinCenters(), ensp_nom_juno['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['bump_corr'].bin_cont[0], ensp_nom_juno['bump_corr'].bin_cont[-1]))

    if dyb_model: 
        DYB_edges = make_nonuniform_bins(args_nd.bins_nonuniform)
        from dag_modelling.tools.make_fcn import make_fcn
        flux_params = dyb_model.storage["parameters.free.neutrino_per_fission_factor"]
        chi2_dyb_func = make_fcn(dyb_model.storage[f"outputs.statistic.full.covmat.chi2cnp"], parameters=[par for par in flux_params.walkvalues()], safe=False)

    resp_matrix_cache = {}
    def get_resp_matrix(a, b, c):
        key = (a,b,c)
        if key not in resp_matrix_cache:
            resp_matrix_cache[key] = CalcRespMatrix_abc(a,b,c, escale=1, ebins=fine_bins, pebins=fine_bins)
        return resp_matrix_cache[key]

    fit_bckgs_spectra = {}
    def chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea=0.0, Nrea_nd =0.0,\
             NL=None, abc=None, nuisance=None, bckgs=None,\
             rea_norm=None, eff_nuis=None,
             Ngeo=0.0, NU=0.0, NTh=0.0,Nmantle=0.0,\
             opp=False, plot_fit=False): #,  me_rho=args_juno.me_rho):
        pulls=[]
        pull_unc = []
        if args_juno.sin2_th13_opt=="pull":
            pulls.append(sin2_13 - nuosc.op_nom['sin2_th13'])
            pull_unc.append(nuosc.op_nom['sin2_th13_err'])


        if Nrea:
            pulls.append(Nrea)
            pull_unc.append(args_juno.r2_unc)

        if args_juno.fit_type == "PMOP_combined" and args_juno.pull_shape:
            for n in nuisance:  
                pulls.append(n)
                pull_unc.append(args_juno.pull_shape_unc)#0.447
                    
            nuisance_coarse = np.array(nuisance)  
            nuisance_fine = expand_nuisances_to_fine(fine_bins, E_edges, nuisance_coarse)
    
    
            rfis_param = Spectrum(bins=ensp_nom_juno["rfis0"].bins, bin_cont=ensp_nom_juno["rfis0"].bin_cont * (1+np.array(nuisance_fine)))
    
        
            # # Apply IBD weighting
            rfis_param.WeightWithFunction(s_ibd)
    
            # # Apply bump correction if needed
            #rfis_param = rfis_param.GetWeightedWithFunction(s_bump_lin)
            rfis_param.GetScaled(ndays)

            if eff_nuis:
                rfis_param.GetScaled(compute_extrafactors(args_juno, eff_nuis=eff_nuis))
                pulls.append(eff_nuis)
                pull_unc.append(args_juno.eff_unc)       
                
            else:
                rfis_param.GetScaled(compute_extrafactors(args_juno))
            
    
            ibd = rfis_param
            
        else:
            ibd = ensp_nom_juno["rfis"]

        if rea_norm:
            Nsnf=rea_norm[0], Nnoneq=rea_norm[1]
            ibd_tot = ibd + ensp_nom_juno["snf"].GetScaledFit(1+Nsnf) + ensp_nom_juno["noneq"].GetScaledFit(1+Nnoneq)
            pulls.extend([Nsnf, Nnoneq])
            pull_unc.extend([args_juno.snf_unc, args_juno.noneq_unc])

        else:
            ibd_tot = ibd + ensp_nom_juno["snf"] + ensp_nom_juno["noneq"]

        s_juno =ibd_tot.GetOscillated(L=args_juno.core_baselines, sin2_th12=sin2_12, sin2_th13=sin2_13,
                                          dm2_21=dm2_21, dm2_31=dm2_31, core_powers=args_juno.core_powers,
                                          me_rho=args_juno.me_rho, ene_mode='true', opp=opp, args=args_juno)
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy

        #print(NL)
        if NL:
            # new_nonl =  Spectrum(bins = ensp_nom_juno['scintNL'].bins, bin_cont=np.zeros(len(ensp_nom_juno['scintNL'].bin_cont)))
            # new_nonl.bin_cont = ensp_nom_juno['scintNL'].bin_cont\
            #                     +NL[0]*(ensp_nom_juno['NL_pull'][0].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
            #                     +NL[1]*(ensp_nom_juno['NL_pull'][1].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
            #                     +NL[2]*(ensp_nom_juno['NL_pull'][2].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
            #                     +NL[3]*(ensp_nom_juno['NL_pull'][3].bin_cont - ensp_nom_juno['scintNL'].bin_cont)
            # s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl) 
            # pulls.extend([NL[0], NL[1], NL[2], NL[3]])
            # pull_unc.extend([args_juno.NL_sd]*4)
            kB = NL[0]
            fC = NL[1]
            s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC) 
            pulls.extend([kB-args_juno.kB, fC-args_juno.fC])
            pull_unc.extend([args_juno.kB_err, args_juno.fC_err])
        else:
            #s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])
            s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=args_juno.kB, fC=args_juno.fC)

        if abc:
            resp_matrix = get_resp_matrix(abc[0], abc[1], abc[2])
            s_juno = s_juno.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
            pulls.extend([abc[0]-args_juno.a, abc[1]-args_juno.b, abc[2]-args_juno.c])
            pull_unc.extend([args_juno.a_err, args_juno.b_err, args_juno.c_err])
        else:
            s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution

        if args_juno.rebin_nonuniform: s_juno.Rebin_nonuniform(args_juno.bins_nonuniform)


        s_tot_juno = s_juno.GetScaledFit(1+Nrea) 

        for key, info in bckgs.items():
            if args_juno.fit_type == "geo" and key=="geo": continue
            # if key=="geo" and NL:
            #     bckgU = ensp_nom_juno[f'rvis_geou'].GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC) 
            #     bckgTh = ensp_nom_juno[f'rvis_geoth'].GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC) 
            #     bckgU.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
            #     bckgTh.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
            #     bckgU.Rebin_nonuniform(args_juno.bins_nonuniform)
            #     bckgTh.Rebin_nonuniform(args_juno.bins_nonuniform)
            #     bckg = bckgU+bckgTh
            #     s_tot_juno += bckg.GetScaledFit(1+info["pull"])
            # else:
            s_tot_juno += ensp_nom_juno[key].GetScaledFit(1+info["pull"])
            if info["sigma"] != 0 and bckgs[key]:
                pulls.append(info["pull"])
                pull_unc.append(info["sigma"])
            if plot_fit: 
                # if key == "geo":
                #     fit_bckgs_spectra[key] = bckg.GetScaledFit(1+info["pull"])
                # else:
                fit_bckgs_spectra[key] = ensp_nom_juno[key].GetScaledFit(1+info["pull"])
         
        if args_juno.fit_type == 'geo':
            if args_juno.geo_fit_type == 'UThfree':
                geo_sp =  ensp_nom_juno["geou"].GetScaledFit(1+NU)+ensp_nom_juno["geoth"].GetScaledFit(1+NTh)
            elif args_juno.geo_fit_type == 'mantle':
                geo_sp =  ensp_nom_juno["geomantle"].GetScaledFit(1+Nmantle)
            else:
                geo_sp = ensp_nom_juno["geo"].GetScaledFit(1+Ngeo)
            s_tot += geo_sp

        if bool(args_juno.toymc):
            sp_obs_tot = ensp_nom_juno["rtot_toy"]
        if args_juno.is_data:
            sp_obs_tot = ensp_nom_juno['data']
        else:
            sp_obs_tot = ensp_nom_juno["rtot"]
            
            
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
                chi2 = chi2_juno + chi2_dyb           

        if plot_fit:
            plot_fit_residuals(s_tot_juno, sp_obs_tot, s_juno, fit_bckgs_spectra, ndays)
            
        else:
            chi2 = Chi2(cm_juno[unc_juno], sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, unc_juno, args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc) 
            
        return chi2
    


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

    
    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=args_juno.NMO_opt)

    bckgs = {}
    for key, info in backgrounds["JUNO"].items():
        if args_juno.pull_all: bckgs[key] = {"pull": 0.0, "sigma": info["rate_unc"]}
        else: bckgs[key] = {"pull": 0.0, "sigma": 0.0}

    bckgs_keys = [k for k, v in bckgs.items()]# if v["sigma"] != 0.0]

    NL = [args_juno.kB, args_juno.fC] if args_juno.pull_all else []

    abc = [args_juno.a, args_juno.b, args_juno.c] if args_juno.pull_all else []

    rea_norm = [0.0]*2
    eff_nuis = 0.0
    nuisance = [0.0] * (len(E_edges) - 1) if args_juno.fit_type == "PMOP_combined" else []

    n_NL = len(NL)
    n_abc = len(abc)
    n_bckgs = len(bckgs_keys)
    
    # def chi2_wrapper(sin2_12, sin2_13, dm2_21, dm2_31, Nrea, *all_nuisance_params):

    #     n_nuis = len(all_nuisance_params) - n_NL -n_abc - n_bckgs
    #     NL_vals = all_nuisance_params[:n_NL]
    #     abc_vals = all_nuisance_params[n_NL:n_NL + n_abc]
    #     bckgs_vals = all_nuisance_params[n_NL + n_abc:n_NL + n_abc + n_bckgs]
    #     nuisance_vals = all_nuisance_params[n_NL + n_abc + n_bckgs:]

    #     for i, key in enumerate(bckgs_keys):
    #         bckgs[key]["pull"] = bckgs_vals[i]
    
    #     return chi2(
    #         sin2_12=sin2_12,
    #         sin2_13=sin2_13,
    #         dm2_21=dm2_21,
    #         dm2_31=dm2_31,
    #         Nrea=Nrea,
    #         NL=NL_vals,
    #         abc=abc_vals,
    #         nuisance=nuisance_vals,
    #         bckgs=bckgs,
    #     )
    def chi2_wrapper(sin2_12, sin2_13, dm2_21, dm2_31, Nrea, *all_nuisance_params, plot_fit=False):
        n_nuis = len(all_nuisance_params) - n_NL -n_abc - n_bckgs
        NL_vals = all_nuisance_params[:n_NL]
        abc_vals = all_nuisance_params[n_NL:n_NL + n_abc]
        bckgs_vals = all_nuisance_params[n_NL + n_abc:n_NL + n_abc + n_bckgs]
        nuisance_vals = all_nuisance_params[n_NL + n_abc + n_bckgs:]

        for i, key in enumerate(bckgs_keys):
            bckgs[key]["pull"] = bckgs_vals[i]
        return chi2(
        sin2_12=sin2_12,
        sin2_13=sin2_13,
        dm2_21=dm2_21,
        dm2_31=dm2_31,
        Nrea=Nrea,
        NL=NL_vals,
        abc=abc_vals,
        nuisance=nuisance_vals,
        bckgs=bckgs,
        plot_fit=plot_fit,   # pass this flag down
        )
    def make_chi2_func(n_fixed=5, n_total=None):
        def f(*params):
            return chi2_wrapper(*params[:n_fixed], *params[n_fixed:])
        f.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(f"p{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for i in range(n_total)
            ]
        )
        return f
    
    start_values = [
        nuosc.op_nom["sin2_th12"],
        nuosc.op_nom["sin2_th13"],
        nuosc.op_nom["dm2_21"],
        nuosc.op_nom["dm2_31"],
        0.0,  # Nrea
    ] + NL + abc + [bckgs[k]["pull"] for k in bckgs_keys] + nuisance

    print("Parameters length: ", len(start_values))
    param_names = (
        ["sin2_th12", "sin2_th13", "dm2_21", "dm2_31", "Nrea"]
        + [f"NL{i}" for i in range(len(NL))]
        + [f"abc{i}" for i in range(len(abc))]
        + [f"bkg_{k}" for k in bckgs_keys]
        + [f"nuis{i}" for i in range(len(nuisance))]
    )
    
    chi2_func = make_chi2_func(n_fixed=5, n_total=len(start_values))
    m = Minuit(chi2_func, *start_values, name=param_names)


    m.migrad()  
    # m.tol = 1e-4
    # m.migrad()
    if not args_juno.silent_print: print(m)
    #plot_fit_spectrum(m.values[0], m.values[1], m.values[2], m.values[3], m.values[4], m.values[5], m.values[5:5+len(n_params)])
 
    print("Function minimum value: ", m.fval)
    print("chi2/ndf: ", m.fval/(66-4))
    print("-----Good things, should be True-----")
    print("FMin valid?             ", m.fmin.is_valid)
    print("Has covariance?         ", m.fmin.has_covariance)
    print("Covariance accurate?    ", m.fmin.has_accurate_covar)
    print("Minos valid?            ", m.fmin.is_valid)

    print("-----Bad things, should be False-----")
    print("Above EDM threshold?    ", m.fmin.is_above_max_edm)
    print("Has reached call limit? ", m.fmin.has_reached_call_limit)
    print("Hesse failed?           ", m.fmin.hesse_failed)

    

    best_params = [m.values[name] for name in m.parameters]
    chi2_wrapper(*best_params, plot_fit=True) 
    merrors=bool(args_juno.calc_minuit_errors)
    if merrors:
        m.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31") #(params=['x0', 'x1', 'x2', 'x3']) #m.minos() #get minos errors

    # if args_juno.fit_type == "geo"  and args_juno.geo_fit_type == 'mantle':
    #     chi2_min = chi2_mantle(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3], Nrea=m.values[4],Nmantle=m.values[5])
    #     chi2_mantle0 = chi2_nomantle(sin2_12=m0.values[0], sin2_13=m0.values[1], dm2_21=m0.values[2], dm2_31=m0.values[3], Nrea=m0.values[4])
    #     # chi2_min = chi2_mantle_OPfixed(Nrea=m.values[0], Nmantle=m.values[1])
    #     #chi2_mantle0 = chi2_nomantle_OPfixed(Nrea=m0.values[0])
    #     dchi2 = chi2_mantle0 - chi2_min
    #     if not args_juno.silent_print: print(f"Mantle discovery potential: delta chi2 between mantle and no mantle: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    # else:
    #     if args_juno.fit_type == "PMOP_combined":
    #         if args_juno.nd == "TAO":
    #             chi2_min = PMOP_combined_chi2(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
    #             chi2_min_opp = PMOP_combined_chi2_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])        
    #             dchi2 = abs(chi2_min-chi2_min_opp)
    #             print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")
            
    #     else:
    #         chi2_min = chi2_pmop(sin2_12=m.values[0], sin2_13=m.values[1], dm2_21=m.values[2], dm2_31=m.values[3])
    #         chi2_min_opp = chi2_pmop_opp(sin2_12=m_opp.values[0], sin2_13=m_opp.values[1], dm2_21=m_opp.values[2], dm2_31=m_opp.values[3])

    #         dchi2 = abs(chi2_min-chi2_min_opp)
    #         print(f"NMO: delta chi2 between NO and IO assuming {args_juno.NMO_opt}: {dchi2} and corresponding significance: {np.sqrt(dchi2)}")

    if not args_juno.silent_print:
        print("************Oscillation parameter fit results************")
        for i in range(4): #len(m.values)):
            if merrors:
                err =  max(abs(m.merrors[i].upper), abs(m.merrors[i].lower))
            else:
                err = m.errors[i]
                
            print(f"{m.parameters[i]}: {m.values[i]} +/- {err}; rel unc: {err/m.values[i]}")
    print("************Other parameter fit results************")
    for i in range(4, len(m.values)): 
        print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")
  
    print("************Background rates************")
        
    for j, key in enumerate(bckgs_keys):
        i = n_NL + n_abc + j              
        nuis = m.values[i]
        nuis_err = m.errors[i]
        start_rate = backgrounds["JUNO"][key]["rate"]
        rate_fit = start_rate * (1 + nuis)
        rate_err = start_rate * nuis_err
        N_fit = rate_fit * ndays
        N_err = rate_err * ndays
    
        print(f"{key}: {rate_fit:.4e} ± {rate_err:.4e} cpd"
              f"({N_fit:.4e} ± {N_err:.4e} events)")

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

