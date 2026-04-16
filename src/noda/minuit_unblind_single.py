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
from .energy_response_cov import *
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import scipy as scipy
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.linalg import pinvh


FULL_CHI2 = 0
DYB_BINS = 0
FREE_PARAM = 0

def f_lbl(dm2_31, sin2_12, sin2_13, dm2_21, theta23, dcp, ordering):
    if ordering == "NO":
        return dm2_31
    elif ordering == "IO":
        # Ensure values are within physical bounds to avoid NaNs
        safe_sin2_12 = np.clip(sin2_12, 1e-10, 1.0)
        safe_sin2_13 = np.clip(sin2_13, 1e-10, 1.0)
        
        c12 = np.sqrt(1 - safe_sin2_12)
        s13 = np.sqrt(safe_sin2_13)

        # Correct the formula (note: you had tan*cos + tan*cos, likely meant to use args)
        # Using the simplified term_brk logic we discussed:
        term = dm2_21 * (2*c12**2 - np.sin(2*np.arcsin(np.sqrt(safe_sin2_12))) * s13 * (np.tan(np.radians(theta23))*np.cos(np.radians(dcp)) * 2)) # Assuming symmetric NO/IO values for now

        return -dm2_31 + term

def write_parameter(name, val, err):
    vec = ROOT.TVectorD(2)
    vec[0] = val
    vec[1] = err
    vec.Write(name)

def write_osc_params(name, val, err):
    value = ROOT.TParameter("double")(f"{name}", val)
    error = ROOT.TParameter("double")(f"{name}_err", err)
    value.Write("", ROOT.TObject.kOverwrite)
    error.Write("", ROOT.TObject.kOverwrite)

def profile_dm2_31(m, args_juno, NMO_opt=True, write=False):

    from scipy.interpolate import CubicSpline
    print("Profiling dm2_31 using built-in mnprofile...")
    # for p in m.parameters:
    #     m.fixed[p] = True
    # m.fixed["dm2_31"] = False
    # 1. Run the built-in profile scan
    # 'bins=10' matches your original scan density
    # 'bound=(1.0e-3, 3.0e-3)' sets the physical range
    if NMO_opt:
        res = m.mnprofile("dm2_31",size=int(args_juno.profile_points), bound=(1.5, 3.5))
    else:
        res = m.mnprofile("dm2_31",size=int(args_juno.profile_points),bound=(-3.5, -1.5))
    
    # mnprofile returns: (values, fvals, success)
    dm2_scan_vals = res[0]
    chi2_vals = res[1]
    delta_chi2 = chi2_vals - np.min(chi2_vals)
    # --- Convert to Δχ² ---
    
    dm2_fine = np.linspace(dm2_scan_vals.min(), dm2_scan_vals.max(), 500)

    # 3. Fit the spline
    cs = CubicSpline(dm2_scan_vals, chi2_vals)
    chi2_smooth = cs(dm2_fine)
    delta_chi2_smooth = chi2_smooth - np.min(chi2_smooth)
# 4. Plot the smooth version

    # --- Plotting (same as your original) ---
    plt.figure(figsize=(7, 5))
    plt.plot(dm2_fine, delta_chi2_smooth, label="Smoothed Profile", lw=2)
    plt.xlabel(r"$\Delta m^2_{31}$ [10$^{-3}$ eV$^2$]", fontsize=14)
    plt.ylabel(r"$\Delta \chi^2$", fontsize=14)
    #plt.title(r"Profile Likelihood Scan ($\mathtt{mnprofile}$)", fontsize=15)
    plt.axhline(1.0, color="green", ls="--", label=r"1$\sigma$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"Plots/profile_dm2_31_NO_{NMO_opt}_{args_juno.file_prefix}.png")

    # --- Extract Results ---
    best_index = np.argmin(delta_chi2)
    best_fit_val = dm2_scan_vals[best_index]


    if write:
        # n = number of points, x_array, y_array
        n_points = len(dm2_fine)
        graph = ROOT.TGraph(n_points, dm2_fine.astype(float), delta_chi2_smooth.astype(float))
        
        # 2. Set metadata (optional but helpful for ROOT users)
        graph.SetName("dm2_31")
        graph.SetMarkerStyle(2)
        
        # 3. Save to file
        out_file = ROOT.TFile(f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_prof = out_file.mkdir("profiles")
        dir_prof.cd()
        graph.Write()
        out_file.Close()
        
    best_dm2_from_scan = dm2_fine[np.argmin(chi2_smooth)]
    print(best_dm2_from_scan)
    m.values["dm2_31"] = best_dm2_from_scan
    print(m.values["dm2_31"])
    # for p in m.parameters:
    #     m.fixed[p] = False
    # m.reset()
    m.migrad()    

def plot_fit_residuals(s_tot_juno, s_obs, s_juno, bckgs, ndays, args_juno,write_output=False):
    """
    Plot observed vs fitted spectra with residuals and local/cumulative χ²/ndf.
    The cumulative χ²/ndf is shown on a right-hand y-axis in histogram style.
    Also generates a secondary plot showing cumulative χ² and global χ²/ndf vs energy.
    """

    if write_output:
        output_file = f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}_dm2_31_pull_{args_juno.pull_dm2_31}.root"
        s_juno.WritetoROOT("reactor", f"{output_file}", directory="postfit")
        #s_tot_juno.WritetoROOT("totalpred", f"Postfit_{args_juno.file_prefix}.root")
        #s_obs.WritetoROOT("total", f"Postfit_{args_juno.file_prefix}.root")

    os.makedirs("Plots", exist_ok=True)
    bin_centers = s_tot_juno.bins
    bin_centers2 = 0.5 * (s_tot_juno.bins[:-1] + s_tot_juno.bins[1:])

    # --- Extract data ---
    obs_vals = s_obs.bin_cont
    obs_err = np.sqrt(np.maximum(obs_vals, 1e-12))
    fit_vals = s_tot_juno.bin_cont
    fit_err = np.sqrt(np.maximum(fit_vals, 1e-12))

    # --- Residuals and χ² ---
    residuals = (obs_vals - fit_vals) / np.maximum(obs_err, 1e-12)
    chi2_bin = (obs_vals - fit_vals) ** 2 / np.maximum(obs_err, 1e-12) ** 2

    total_chi2 = np.sum(chi2_bin)
    ndf = len(chi2_bin) + DYB_BINS - FREE_PARAM
    chi2_ndf = FULL_CHI2 / ndf

    # --- Cumulative χ²/ndf (high → low energy) ---
    chi2_bin_rev = chi2_bin[::-1]
    cum_chi2_rev = np.cumsum(chi2_bin_rev)
    ndf_bins = np.arange(1, len(cum_chi2_rev) + 1)
    cum_chi2_ndf_rev = cum_chi2_rev / ndf_bins
    cum_chi2_ndf = cum_chi2_ndf_rev[::-1]
    cum_chi2_global_ndf_rev = cum_chi2_rev / (len(chi2_bin)-3)
    cum_chi2_global_ndf = cum_chi2_global_ndf_rev[::-1]

    # --- Main 3-panel figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8),
        gridspec_kw={'height_ratios': [3, 1, 1]}
    )

    # === TOP PANEL: Data vs Fit ===
    ax1.errorbar(bin_centers2, obs_vals, yerr=obs_err, fmt='o', color='black',
                 markersize=3, label='Observed data')
    ax1.step(bin_centers, np.append(fit_vals, fit_vals[-1]), where='mid', color='blue', lw=1.5, label='Total Fit')
    ax1.step(bin_centers, np.append(s_juno.bin_cont, s_juno.bin_cont[-1]), where='mid', color='red', lw=1.2,
             linestyle='-', label=f'Reactor $\\bar{{\\nu}}_e$ expectation: {s_juno.GetIntegral()/ndays:.2f} cpd') # ± {np.sqrt(s_juno.GetIntegral())/ndays:.2f} cpd')

    colors = plt.cm.tab10.colors
    name_to_key = {
        "Accidentals": "acc",
        "Geoneutrinos": "geo",
        r"$^{9}$Li / $^{8}$He": "lihe",
        "Fast neutrons": "fastn",
        "Double neutrons": "doublen",
        r"$^{13}$C ($\alpha$, n) $^{16}$O": "alphan",
        "Atmospheric neutrinos": "atm",
        "World reactor neutrinos": "wr",
        r"$^{214}$Bi": "bipo"
    }

    for i, (key, spec) in enumerate(bckgs.items()):
        cpd = spec.GetIntegral() / ndays
        ax1.step(bin_centers, np.append(spec.bin_cont, spec.bin_cont[-1]), where='mid',
                 color=colors[i % len(colors)], lw=1,
                 label=f'{key}: {cpd:.2f} cpd')
        if write_output:
            spec.WritetoROOT(f"{name_to_key.get(key)}" , f"{output_file}", "postfit")

    ax1.set_ylabel("Events")
    leg = ax1.legend(fontsize=9, loc="upper right", frameon=True, edgecolor="black")
    leg.get_frame().set_linewidth(1.0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.7, 12.0)

    # === MIDDLE PANEL: Residuals ===
    ax2.errorbar(bin_centers2, residuals, yerr=np.ones_like(residuals),
                 fmt='o', color='blue', markersize=3)
    ax2.axhline(0, color='k', lw=1)
    for nsigma, alpha in zip([1, 2, 3], [0.25, 0.15, 0.1]):
        ax2.fill_between(bin_centers, -nsigma, nsigma, color='blue', alpha=alpha)
    ax2.set_ylabel("Residuals [σ]")
    ax2.set_ylim(-5.0, 3.0)
    ax2.grid(True, alpha=0.3)

    # === BOTTOM PANEL: χ² distributions ===
    ax3.step(bin_centers, np.append(chi2_bin, chi2_bin[-1]), where='mid', color='steelblue', lw=1.5,
             label='Local χ²/ndf')
    ax3.axhline(1.0, color='gray', ls='--', lw=1)
    ax3.set_ylabel("Local χ²/ndf", color='steelblue')
    ax3.tick_params(axis='y', colors='steelblue')
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    ax3b.step(bin_centers, np.append(cum_chi2_ndf, cum_chi2_ndf[-1]), where='mid', color='brown', lw=1.5,
              label='Cumulative χ²/ndf (high→low)')
    ax3b.axhline(chi2_ndf, color='darkred', ls='--', lw=1)
    ax3b.set_ylabel("Cumulative χ²/ndf", color='brown')
    ax3b.tick_params(axis='y', colors='brown')
    ax3.set_ylim(0, 5)
    ax3b.set_ylim(0,5)
    # --- Toy χ² distribution for p-value ---
    n_toys = 1000
    chi2_toys = []
    for _ in range(n_toys):
        toy_data = np.random.poisson(np.maximum(fit_vals, 1e-12))
        chi2_toy = np.sum((toy_data - fit_vals) ** 2 / np.maximum(obs_err, 1e-12) ** 2)
        chi2_toys.append(chi2_toy)
    chi2_toys = np.array(chi2_toys)
    p_val = np.sum(chi2_toys > total_chi2) / len(chi2_toys)

    print(f"\n--- Toy-based goodness-of-fit test ---")
    print(f"Observed χ² = {total_chi2:.2f} for {ndf} dof → χ²/ndf = {chi2_ndf:.3f}")
    print(f"Toy mean χ² = {chi2_toys.mean():.2f} ± {chi2_toys.std():.2f}")
    print(f"p-value = {p_val:.3f}\n")
    
    textstr = (
        f"Livetime = {ndays:.2f} days\n"
        f"χ²/ndf = {FULL_CHI2:.2f}/{ndf} = {chi2_ndf:.2f}"
        #f"p-value (from toys) = {p_val:.3f}\n"
        #f"Reactor $\\bar{{\\nu}}_e$ rate = {s_juno.GetIntegral()/ndays:.2f} ± {np.sqrt(s_juno.GetIntegral())/ndays:.2f} cpd"
    )
    ax1.text(
        0.02, 0.9, textstr,                   # (x=0.9 → right side, y=0.5 → vertical center)
        transform=ax1.transAxes,
        fontsize=9,
        va='center', ha='left',             # vertically centered, right-aligned
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )


    ax3.set_xlabel("Reconstructed Energy [MeV]")
    plt.tight_layout()
    plt.savefig(f"Plots/fit_residuals_{args_juno.file_prefix}.png", dpi=200)
    plt.close(fig)


    # === NEW: Additional figure for cumulative χ² ===
    fig2, (ax_cum1, ax_cum2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    ax_cum1.step(bin_centers, np.append(cum_chi2_ndf, cum_chi2_ndf[-1]),
                 where='mid', color='purple', lw=1.5, label='Cumulative χ² (high→low)')
    ax_cum1.set_ylabel("Cumulative χ²/ ndf")
    ax_cum1.legend(loc="upper left", fontsize=9)
    ax_cum1.grid(True, alpha=0.3)

    ax_cum2.step(bin_centers, np.append(cum_chi2_global_ndf, cum_chi2_global_ndf[-1]),
                 where='mid', color='darkgreen', lw=1.5, label='Global cumulative χ²/ndf')
    #ax_cum2.axhline(chi2_ndf, color='gray', ls='--', lw=1)
    ax_cum2.set_xlabel("Reconstructed Energy [MeV]")
    ax_cum2.set_ylabel("Cummulative χ²/global ndf")
    ax_cum2.legend(loc="upper right", fontsize=9)
    ax_cum2.grid(True, alpha=0.3)
    ax_cum1.set_xlim(0.7, 12.0)
    ax_cum2.set_xlim(0.7, 12.0)
    
    plt.tight_layout()
    plt.savefig("Plots/cumulative_chi2.png", dpi=200)
    plt.close(fig2)

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

def run_minuit(ndays,ndays_dyb=0, ndays_tao=0, ensp_nom_juno={}, ensp_nom_dyb={}, ensp_nom_tao={}, unc_juno='', unc_dyb='', unc_tao='', unc_corr='', rm= [], rm_dyb=[], rm_tao=[], ene_leak_tao =[], cm_juno ={}, cm_dyb={}, cm_tao={}, cm_corr='', args_juno='', args_dyb='', args_tao='', dyb_model=None, backgrounds=None, toy=False, fix_param=None):

    E_segments = np.array(args_juno.seg_bins_nonuniform)
    print("DYB segments: ", E_segments)
    fine_bins = ensp_nom_juno["rfis0"].bins
    fission_fracs = []

    core_names, core_powers_std, core_fractions, core_baselines = load_reactor_summary(args_juno.reactor_summary_yaml)


    for i, frac in enumerate(core_fractions):
        fission_fracs.append([
            frac['U235'],
            frac['U238'],
            frac['Pu239'],
            frac['Pu241']
        ])

    # s_ibd = sp.interpolate.interp1d(ensp_nom_juno['sibd'].GetBinCenters(), ensp_nom_juno['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['sibd'].bin_cont[0], ensp_nom_juno['sibd'].bin_cont[-1]))
    s_ibd = sp.interpolate.interp1d(
        ensp_nom_juno['sibd'].GetBinCenters(), 
        ensp_nom_juno['sibd'].bin_cont, 
        kind='slinear', 
        bounds_error=False, 
        fill_value=(0.0, 0.0)  # <-- Force cross-section to 0 below threshold!
    )
    s_bump_lin = sp.interpolate.interp1d(ensp_nom_juno['bump_corr'].GetBinCenters(), ensp_nom_juno['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp_nom_juno['bump_corr'].bin_cont[0], ensp_nom_juno['bump_corr'].bin_cont[-1]))
    global DYB_BINS;
    if dyb_model: 
        DYB_edges = make_nonuniform_bins(args_dyb.bins_nonuniform)
        from dag_modelling.tools.make_fcn import make_fcn
        flux_params = dyb_model.storage["parameters.free.neutrino_per_fission_factor"]
        dyb_model.storage["parameters.all.survival_probability.DeltaMSq32"].value = 2.5413e-3
        dyb_model.storage["parameters.all.survival_probability.SinSq2Theta13"].value = 0.0856
        dyb_model.storage["parameters.all.survival_probability.DeltaMSq21"].value = 7.53e-05
        dyb_model.storage["parameters.all.survival_probability.SinSq2Theta12"].value = 0.851

        DYB_BINS = dyb_model.storage["outputs.eventscount.final.concatenated.detector"].data.shape[0]
        print("DYB bins", DYB_BINS)
        chi2_dyb_func = make_fcn(dyb_model.storage[f"outputs.statistic.full.covmat.chi2cnp"], parameters=[par for par in flux_params.walkvalues()], safe=False)
    elif args_juno.fit_type == "PMOP_combined":
        DYB_BINS = len(ensp_nom_dyb['rdet'].bins)
        print("DYB bins", DYB_BINS)
    else:
        DYB_bins = 0

    resp_matrix_cache = {}
    def get_resp_matrix(a, b, c):
        key = (a,b,c)
        if key not in resp_matrix_cache:
            resp_matrix_cache[key] = CalcRespMatrix_abc(a,b,c, escale=1, ebins=fine_bins, pebins=fine_bins)
        return resp_matrix_cache[key]

    fit_bckgs_spectra = {}

    chi2_cache = {}
    def get_chi2_dyb(nuisance_coarse):
        """Return cached chi2_dyb if nuisance_coarse already computed."""
        key = tuple(nuisance_coarse)
        if key in chi2_cache:
            return chi2_cache[key]

        nuisance_dyb = nuisance_coarse
        chi2_dyb = chi2_dyb_func(*nuisance_dyb)[0]

        chi2_cache[key] = chi2_dyb
        return chi2_dyb

    nuisance_fine_cache= {}
    def get_nuisance_fine(nuisance_coarse):
        key = tuple(nuisance_coarse)
        if key in nuisance_fine_cache:
            return nuisance_fine_cache[key]

        nuisance_fine_cache[key] = expand_nuisances_to_fine(E_segments, fine_bins, nuisance_coarse)
        return nuisance_fine_cache[key]

    if bool(args_juno.toymc):
        sp_obs_tot = ensp_nom_juno["rtot_toy"]
    elif args_juno.is_data:
        sp_obs_tot = ensp_nom_juno['data']
    else:
        sp_obs_tot = ensp_nom_juno["rtot"]
    theta0 = np.array([
        args_juno.a,
        args_juno.b, 
        args_juno.c,
        args_juno.kB,
        args_juno.fC,
        args_juno.kI,
        args_juno.LY_shared
        ])

    chi2_penalty_er_cache = {}

   

    def get_chi2_penalty_er(a,b,c, kB, fC, kI, LY_shared): 
        key = (a,b,c,kB,fC,kI,LY_shared) 
        if key in chi2_penalty_er_cache:
            return chi2_penalty_er_cache[key]
            
        theta = np.array([a, b, c, kB, fC, kI, LY_shared])

        diff_er = theta - theta0 

        inv_cov_er = np.linalg.inv(covariance_er)

        chi2_penalty_er = diff_er @ inv_cov_er @ diff_er 
        chi2_penalty_er_cache[key] = chi2_penalty_er 

        return chi2_penalty_er



    geo_spectrum_cache = {}
    def get_geo_spectrum(ThU_ratio, resp_matrix, a, b, c, kB, fC, kI, LY_shared):
        key = (ThU_ratio, a, b, c, kB, fC, kI, LY_shared)
        if key in geo_spectrum_cache:
            return geo_spectrum_cache[key]
            
        rate = float(args_juno.geo_tnu_rate)*float(args_juno.tnu_to_cpd)
        scale = rate*ndays / args_juno.duty_cycle
        Th = ensp_nom_juno['rfis0_geoth'].GetScaledFit(ThU_ratio*scale / ((1 + ThU_ratio) * ensp_nom_juno['rfis0_geoth'].GetIntegral()))
        U  = ensp_nom_juno['rfis0_geou'].GetScaledFit(scale / ((1 + ThU_ratio) * ensp_nom_juno['rfis0_geou'].GetIntegral()))
                
        U = U.GetWithPositronEnergy(type=args_juno.posE_shift, inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)
        Th = Th.GetWithPositronEnergy(type=args_juno.posE_shift, inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)
        
        U = U.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
        Th = Th.GetWithModifiedEnergy(mode='G4',  kB=kB, fC=fC, kI=kI, LY_shared=LY_shared) 
                
        U.ApplyDetResp(rm, pecrop=args_juno.ene_crop)
        Th.ApplyDetResp(rm, pecrop=args_juno.ene_crop)
                
        U.Rebin_nonuniform(args_juno.bins_nonuniform)
        Th.Rebin_nonuniform(args_juno.bins_nonuniform)

        bckg = Th + U
        geo_spectrum_cache[key] = bckg
        return bckg
    

    wr_spectrum_cache = {}
    def get_wr_spectrum(resp_matrix, a, b, c, kB, fC, kI, LY_shared):
        key = (a, b, c, kB, fC, kI, LY_shared)
        if key in wr_spectrum_cache:
            return wr_spectrum_cache[key]
            
        key = (a, b, c, kB, fC, kI, LY_shared)
        bckg = ensp_nom_juno[f'rfis_wr_pos'].GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared) 
        bckg.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
    
        rate = backgrounds['JUNO']['wr']['rate'] #args_juno.wr_rate
        scale = rate*ndays / (args_juno.duty_cycle*bckg.GetIntegral())
        bckg = bckg.GetScaledFit(scale)
        
        bckg.Rebin_nonuniform(args_juno.bins_nonuniform)
        
        wr_spectrum_cache[key] = bckg
        return bckg

    snf_noneq_cache = {}
    def get_snf_noneq(snf_noneq_norm):
        key = tuple(snf_noneq_norm)
        if key in snf_noneq_cache:
            return snf_noneq_cache[key]
        if not args_juno.CI_powers:
            YJ = ensp_nom_juno['snf_YJ'].GetScaledFit(1+snf_noneq_norm[0]) + ensp_nom_juno['noneq_YJ'].GetScaledFit(1+snf_noneq_norm[3])
            TS = ensp_nom_juno['snf_TS'].GetScaledFit(1+snf_noneq_norm[1]) + ensp_nom_juno['noneq_TS'].GetScaledFit(1+snf_noneq_norm[4])
            HM = ensp_nom_juno['snf_HM'].GetScaledFit(1+snf_noneq_norm[2]) + ensp_nom_juno['noneq_HM'].GetScaledFit(1+snf_noneq_norm[5])

            tot = YJ+TS+HM
        else:
            tot = ensp_nom_juno['snf_final'].GetScaledFit(1+snf_noneq_norm[0]) + ensp_nom_juno['noneq_final'].GetScaledFit(1+snf_noneq_norm[1])
        snf_noneq_cache[key] = tot
        return tot
   
    dyb_spectrum_cache = {}
    def get_dyb_exp(rfis_param_dyb, nuisance_coarse):
        
        key = tuple(nuisance_coarse)
        if key in dyb_spectrum_cache:
            return dyb_spectrum_cache[key]
            
        nuisance_dyb =expand_nuisances_to_fine(E_segments, ensp_nom_dyb["rfis0"].bins, nuisance_coarse)

        rfis_param_dyb = Spectrum(bins=rfis_param_dyb.bins, bin_cont=rfis_param_dyb.bin_cont * (1+np.array(nuisance_dyb)))
        
        ibd_dyb = rfis_param_dyb

        s_dyb = ibd_dyb

        s_dyb = s_dyb.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
     
        s_dyb = s_dyb.ApplyDetResp(rm_dyb, pecrop=args_dyb.ene_crop) 
        dyb_spectrum_cache[key] = s_dyb
        return s_dyb   
        
    n_cores = len(core_baselines)
    
    if args_juno.CI_powers:
        dummy_sp = ensp_nom_juno["rfis0"]

    else:
        dummy_sp = ensp_nom_juno["rfis0_core0"]

        
    fission_arrays = np.zeros((n_cores, len(dummy_sp.bin_cont)))  
    for i in range(n_cores):
        if args_juno.CI_powers:
            dummy_rfis = Spectrum(bins=ensp_nom_juno[f"rfis0"].bins, 
                              bin_cont=ensp_nom_juno[f"rfis0"].bin_cont)
        
        else:
            dummy_rfis = Spectrum(bins=ensp_nom_juno[f"rfis0_core{i}"].bins, 
                              bin_cont=ensp_nom_juno[f"rfis0_core{i}"].bin_cont)
        dummy_rfis.WeightWithFunction(s_ibd)
        dummy_rfis.ScaleByBinWidth()
        if args_juno.apply_bump: 
            dummy_rfis.WeightWithFunction(s_bump_lin)
        dummy_rfis.GetScaled(ndays)

        fission_arrays[i] = dummy_rfis.bin_cont


    def chi2(sin2_12, sin2_13, dm2_21, dm2_31, Nrea=0.0,\
             NL=None, abc=None, nuisance=None, bckgs=None,\
             snf_noneq_norm=None, eff_nuis=0.0,me_rho=args_juno.me_rho,
             core_powers_delta=[], efission=[],
             Ngeo=0.0, NU=0.0, NTh=0.0,Nmantle=0.0,\
             opp=False, plot_fit=False, ThU_ratio=args_juno.Th_U_ratio): #,  me_rho=args_juno.me_rho):
        chi2 = 1e+6
        chi2_penalty_er = 0.0
        
        dm2_31 = dm2_31*1e-3
        dm2_21 = dm2_21*1e-5
        
        pulls=[]
        pull_unc = []
        if args_juno.sin2_th13_opt=="pull":
            pulls.append(sin2_13 - nuosc.op_nom['sin2_th13'])
            pull_unc.append(nuosc.op_nom['sin2_th13_err'])
        if bool(args_juno.pull_dm2_31):
            pulls.append(dm2_31 -   nuosc.op_nom['dm2_31'])
            pull_unc.append(nuosc.op_nom['dm2_31_err'])

        if args_juno.pull_dm2_31_lbl:
            # choose ordering
            ordering = "NO" if dm2_31 >0 else "IO"
            
            # expected LBL value (or measured input)
            dm2_31_lbl = f_lbl(
                args_juno.dm2_31_lbl,
                sin2_12,
                sin2_13,
                dm2_21,
                args_juno.theta23,
                args_juno.dcp,
                ordering
            )
            #if dm2_31 <0: print(dm2_31_lbl)
            
            pulls.append(dm2_31 - dm2_31_lbl)
            pull_unc.append(args_juno.dm2_31_lbl_err)

        fission_fraction_spectra  = []

        if len(core_powers_delta):
            core_powers_new = list(np.array(core_powers_std)*(1+np.array(core_powers_delta)))
            pulls.extend(core_powers_delta)
            pull_unc.extend([args_juno.core_flux_unc] * 10)
        else:
            core_powers_new = args_juno.core_powers

        if len(efission):
            efission = np.array(efission)*1e2
            efission = list(efission)
            pulls.extend(np.array(efission) - np.array(args_juno.efission))
            pull_unc.extend(args_juno.efission_err)
        else:
            efission = args_juno.efission

        if args_juno.fit_type == "PMOP_combined":   
            nuisance_coarse = np.array(nuisance) 
            nuisance_fine = get_nuisance_fine(nuisance_coarse)
            nuisance_multiplier = 1.0 + nuisance_fine
        else:
            nuisance_multiplier = 1.0

        # 1. Compute scaling factors dynamically (since efission/powers might float)
        core_scales = np.zeros(n_cores)
        for i in range(n_cores):
            core_scales[i] = compute_extrafactors(
                args_juno, 
                efission=efission, 
                alpha=fission_fracs[i], 
                core_baselines=[core_baselines[i]], 
                core_powers=[core_powers_new[i]]
            )


        raw_ibd_tot = np.sum(fission_arrays * core_scales[:, np.newaxis], axis=0)
        raw_ibd_tot = raw_ibd_tot * nuisance_multiplier

        rfis_param_post = Spectrum(bins=dummy_sp.bins, bin_cont=raw_ibd_tot)
        #print("before osc", rfis_param_post.GetIntegral())
        if me_rho != 0.0:
            pulls.append(me_rho-args_juno.me_rho)
            pull_unc.append(args_juno.me_rho_scale)
            rho=me_rho
        else:
            rho=args_juno.me_rho
            
        s_juno = rfis_param_post.GetOscillated(
                L=list(core_baselines),
                sin2_th12=sin2_12,
                sin2_th13=sin2_13,
                dm2_21=dm2_21,
                dm2_31=dm2_31,
                core_powers=list(core_powers_new),
                me_rho=rho,
                ene_mode='true',
                opp=opp,
                args=args_juno
            )

        #print("after osc", s_juno.GetIntegral())
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)  # Shift to positron energy
        #print("after det resp", s_juno.GetIntegral())
        if NL:
            if args_juno.NL == "pull":
                new_nonl =  Spectrum(bins = ensp_nom_juno['scintNL'].bins, bin_cont=np.zeros(len(ensp_nom_juno['scintNL'].bin_cont)))
                new_nonl.bin_cont = ensp_nom_juno['scintNL'].bin_cont\
                                    +NL[0]*(ensp_nom_juno['NL_pull'][0].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
                                    +NL[1]*(ensp_nom_juno['NL_pull'][1].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
                                    +NL[2]*(ensp_nom_juno['NL_pull'][2].bin_cont - ensp_nom_juno['scintNL'].bin_cont)\
                                    +NL[3]*(ensp_nom_juno['NL_pull'][3].bin_cont - ensp_nom_juno['scintNL'].bin_cont)
                s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl, LY_shared=NL[4]) 
                pulls.extend([NL[0], NL[1], NL[2], NL[3], NL[4]])
                pull_unc.extend([args_juno.NL_sd]*4)
                pull_unc.append(0.005)
            else:
                kB = NL[0]
                fC = NL[1]
                kI = NL[2]
                LY_shared = NL[3]
           
                s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)  #GetWithModifiedEnergy(mode='fit', fit_bins=get_NL(s_juno.bins, kB=kB, fC=fC))        
         
                chi2_penalty_er = get_chi2_penalty_er(abc[0],abc[1],abc[2], kB, fC, kI, LY_shared)
      
                # pulls.extend([kB-args_juno.kB, fC-args_juno.fC, kI-args_juno.kI, LY_shared-args_juno.LY_shared])
                # pull_unc.extend([args_juno.kB_err, args_juno.fC_err, args_juno.kI_err, args_juno.LY_shared_err])


        else:
            if args_juno.NL == "pull":
                s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom_juno['scintNL'])
            else:
                s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=args_juno.kB, fC=args_juno.fC, kI=args_juno.kI, LY_shared=args_juno.LY_shared)
        #print("after NL", s_juno.GetIntegral())
        if abc:
            resp_matrix = get_resp_matrix(abc[0], abc[1], abc[2])
            s_juno = s_juno.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
            if args_juno.NL == "pull":
                pulls.extend([abc[0]-args_juno.a, abc[1]-args_juno.b, abc[2]-args_juno.c])
                pull_unc.extend([args_juno.a_err, args_juno.b_err, args_juno.c_err])
        else:
            s_juno = s_juno.ApplyDetResp(rm, pecrop=args_juno.ene_crop)  # Apply energy resolution


        if snf_noneq_norm:
            pulls.extend(snf_noneq_norm)
            pull_unc.extend([args_juno.snf_unc]*3)
            pull_unc.extend([args_juno.noneq_unc]*4)
            s_juno = s_juno + get_snf_noneq(snf_noneq_norm) 
        
        else:
            s_juno = s_juno + ensp_nom_juno['snf_final'] + ensp_nom_juno['noneq_final']

        if eff_nuis:
            pulls.append(eff_nuis)
            pull_unc.append(args_juno.eff_unc)
            s_juno.GetScaled(1+eff_nuis)     
            


        if Nrea:
            pulls.append(Nrea)
            pull_unc.append(args_juno.r2_unc)
            s_juno = s_juno.GetScaledFit(1+Nrea)
            
        if args_juno.rebin_nonuniform: s_juno.Rebin_nonuniform(args_juno.bins_nonuniform)

        s_tot_juno = s_juno 
        cm_bckg = None
        
        for key, info in bckgs.items():
            if key=="geo":
                pulls.append(ThU_ratio-args_juno.Th_U_ratio)
                pull_unc.append(args_juno.Th_U_ratio_err)
            
                if args_juno.NL == "pull":
                    rate = float(args_juno.geo_tnu_rate)*float(args_juno.tnu_to_cpd)
                    scale = rate*ndays / args_juno.duty_cycle
                    Th = ensp_nom_juno['rfis0_geoth'].GetScaledFit(ThU_ratio*scale / ((1 + ThU_ratio) * ensp_nom_juno['rfis0_geoth'].GetIntegral()))
                    U  = ensp_nom_juno['rfis0_geou'].GetScaledFit(scale / ((1 + ThU_ratio) * ensp_nom_juno['rfis0_geou'].GetIntegral()))
                            
                    U_pos = U.GetWithPositronEnergy(type=args_juno.posE_shift, inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)
                    Th_pos = Th.GetWithPositronEnergy(type=args_juno.posE_shift, inputfile=args_juno.input_data_file, tf2name=args_juno.pos_ene_TF2)
                    
                    U_nonl =  U_pos.GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl, LY_shared=NL[4])
                    Th_nonl = Th_pos.GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl, LY_shared=NL[4])
                    
                    U_final = U_nonl.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
                    Th_final = Th_nonl.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)

                    if args_juno.rebin_nonuniform:
                        U_final.Rebin_nonuniform(args_juno.bins_nonuniform)
                        Th_final.Rebin_nonuniform(args_juno.bins_nonuniform)
                    bckg = U_final+Th_final
                    
                else:
                    bckg = get_geo_spectrum(ThU_ratio, resp_matrix, abc[0], abc[1], abc[2], kB, fC, kI, LY_shared)

            elif key=="wr":
                if args_juno.NL == "pull":
                    bckg = ensp_nom_juno[f'rfis_wr_pos'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl, LY_shared=NL[4])
                    bckg.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop)
                    rate = backgrounds['JUNO']['wr']['rate'] #args_juno.wr_rate
                    scale = rate*ndays / (args_juno.duty_cycle*bckg.GetIntegral())
                    bckg = bckg.GetScaledFit(scale)
                    if args_juno.rebin_nonuniform: bckg.Rebin_nonuniform(args_juno.bins_nonuniform)
                else:
                    bckg = get_wr_spectrum(resp_matrix, abc[0], abc[1], abc[2], kB, fC, kI, LY_shared)

            else:
                bckg = ensp_nom_juno[key]

            # print(key, bckg.GetIntegral())
            if info["sigma"] > 0.0:
                pulls.append(info["pull"])
                pull_unc.append(info["sigma"])
                
            bckg = bckg.GetScaledFit(1+info["pull"])
            
            if info["b2b"] >0: # and key != 'acc':
                # if key == 'acc':
                #     b2bunc = float(args_juno.acc_unc)
                # else:
                b2bunc = info["b2b"]
                if cm_bckg is None:
                    cm_bckg = bckg.GetB2BCovMatrix(b2bunc, 0.1)

                else:
                    cm_bckg += bckg.GetB2BCovMatrix(b2bunc, 0.1)

            s_tot_juno += bckg
            
            
            if plot_fit: 
                fit_bckgs_spectra[info["name"]] = bckg

        # print(s_tot_juno.GetIntegral())

        chi2_dyb=0
        if unc_juno != 'stat':
            cm_j = cm_juno[unc_juno]
        else:
            ebins = ensp_nom_juno['rdet'].bins
            nbins = len(ebins)
            cm_j = CovMatrix(data=np.zeros((nbins-1, nbins-1)), bins=ebins)
            
        if args_juno.pull_all:
            cm_j = cm_j + cm_bckg
        
        if args_juno.fit_type == "PMOP_combined":
            if unc_juno == 'stat' and unc_dyb== 'stat' and unc_corr=='stat':
                unc='stat'
                cm_corr[unc_corr] = []
            else:
                unc='syst'

            chi2_nolog, chi2_juno = Chi2(cm_j, sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, 'all', args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc, include_logV=False)
               
            if not args_juno.use_dagflow:
                rfis_param_dyb = ensp_nom_dyb['ribd']
                #core_powers_dyb = np.array(args_dyb.core_powers)*(1+core_powers_delta[8])
                #extrafactors_dyb = compute_extrafactors(args_dyb, efission=efission, core_powers=list(core_powers_dyb))
                #rfis_param_dyb = rfis_param_dyb.GetScaledFit(extrafactors_dyb)
                s_tot_dyb = get_dyb_exp(rfis_param_dyb, nuisance_coarse) 
                noneq = ensp_nom_dyb["noneq_final"].GetScaledFit(1+snf_noneq_norm[6])
                s_tot_dyb = s_tot_dyb + noneq
                if args_dyb.rebin_nonuniform: s_tot_dyb.Rebin_nonuniform(args_dyb.bins_nonuniform)
                sp_obs_dyb = ensp_nom_dyb["rtot"]
                #s_tot_dyb = s_tot_dyb.GetScaledFit(1./s_tot_dyb.GetIntegral())
                cm_dyb_new = CovMatrix(data=cm_dyb['all'].data * np.outer(s_tot_dyb.bin_cont, s_tot_dyb.bin_cont), bins=cm_dyb['all'].bins)
                chi2_dyb_nolog, chi2_dyb =  Chi2(cm_dyb_new, sp_obs_dyb, s_tot_dyb, sp_obs_dyb, s_tot_dyb, 'all', args_juno.stat_method_opt)
                
            else:
                chi2_dyb = get_chi2_dyb(nuisance_coarse)
            #print(chi2_juno, chi2_dyb)


            chi2 = chi2_juno + chi2_dyb   

        else:
            chi2_nolog, chi2 = Chi2(cm_j, sp_obs_tot, s_tot_juno, sp_obs_tot, s_tot_juno, 'bckg', args_juno.stat_method_opt, pulls=pulls, pull_unc=pull_unc, include_logV=False) 
            
        if plot_fit:
            #s_tot_nd.WritetoROOT(f"postfit", f"DYB_spectra_Mar30_2025.root")
            global FULL_CHI2;
            FULL_CHI2 = chi2_nolog + chi2_dyb + chi2_penalty_er
            plot_fit_residuals(s_tot_juno, sp_obs_tot, s_juno, fit_bckgs_spectra, ndays, args_juno,write_output=args_juno.write_output)
            
            ensp_nom_juno['ribd'].Plot(f"{args_juno.plots_folder}/rfis_posfit.png",
                  xlabel="Antineutrino energy (MeV)",
                  ylabel="Events/ 23 keV",
                  extra_spectra=[rfis_param_post],
                  leg_labels=['pre-fit', 'post-fit'],
                  colors=['steelblue', 'brown'],
                  xmin=0.7, xmax=12.0,
                  ymin=1e-3,log_scale=False, title=f"CPC 2021, {len(E_segments)}")
            #rfis_param.Rebin_nonuniform(args_juno.bins_nonuniform)
            #if args_juno.write_output: rfis_param.WritetoROOT(f"unosc_{args_juno.prefix}", "Unosc_Jan21.root")

        #print(chi2, chi2_penalty_er, chi2_juno, chi2_dyb)
        return chi2 + chi2_penalty_er
    


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

    # --- Set oscillation parameters ---
    nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=bool(args_juno.NMO_opt))
    
    # --- Background setup ---
    bckgs = {}
    for key, info in backgrounds["JUNO"].items():
        bckgs[key] = {
            "pull": 0.0,
            "sigma": info["rate_unc"],
            "b2b": info["b2b_unc"],
            "name": info["name"],
        }
    
    bckgs_keys = list(bckgs.keys())
    
    # --- Parameter initialization ---
    if args_juno.NL =="pull":
        NL = [0.0]*5
    else:
        NL = [args_juno.kB, args_juno.fC, args_juno.kI, args_juno.LY_shared]
    abc = [args_juno.a, args_juno.b, args_juno.c]
    if args_juno.CI_powers:
        snf_noneq_norm = [0.0]*2
    else:
        snf_noneq_norm = [0.0]*7
    eff_nuis = 0.0
    me_rho = args_juno.me_rho
    ThU_ratio=args_juno.Th_U_ratio
    core_powers_delta = list(np.zeros(10))

    efission = list(np.array(args_juno.efission)*1e-2)

    if args_juno.fit_type == "PMOP_combined":
        # if args_juno.use_dagflow:
        #nuisance = (s_bump_lin(E_segments)).tolist()
        # else:
        nuisance = list(np.zeros(len(E_segments)))

        print("DYB segment weights start values :", nuisance)
    else:
        nuisance = []
    # --- Length bookkeeping ---
    n_NL = len(NL)
    n_abc = len(abc)
    n_bckgs = len(bckgs_keys)
    n_rea = len(snf_noneq_norm)
    n_nuis = len(nuisance)
    n_core = len(core_powers_delta)
    n_efis = len(efission)
    
    # --- Chi2 wrapper ---
    def chi2_wrapper(sin2_12, sin2_13, dm2_21, dm2_31, Nrea,
                     *all_nuisance_params, opp=False, plot_fit=False):
    
        # Slice nuisance parameters
        offset = 0
        NL_vals = all_nuisance_params[offset : offset + n_NL]
        offset += n_NL
        
        abc_vals = all_nuisance_params[offset : offset + n_abc]
        offset += n_abc
        
        bckgs_vals = all_nuisance_params[offset : offset + n_bckgs]
        offset += n_bckgs
        
        rea_vals = all_nuisance_params[offset : offset + n_rea]
        offset += n_rea
        
        eff_val = all_nuisance_params[offset]; offset += 1
        me_rho_val = all_nuisance_params[offset]; offset += 1
        ThU_ratio_val = all_nuisance_params[offset]; offset += 1
        
        nuisance_vals = all_nuisance_params[offset : offset + n_nuis]
        offset += n_nuis
        
        core_vals = all_nuisance_params[offset : offset + n_core]
        offset += n_core
        
        efis_vals = all_nuisance_params[offset : offset + n_efis]
                # # Update background pulls
        # for i, key in enumerate(bckgs_keys):
        #     bckgs[key]["pull"] = bckgs_vals[i]
        bckgs_for_chi2 = {}
        for i, key in enumerate(bckgs_keys):
            info = dict(bckgs[key])           # shallow copy of info dict
            info["pull"] = bckgs_vals[i]
            bckgs_for_chi2[key] = info
      
        return chi2(
            sin2_12=sin2_12,
            sin2_13=sin2_13,
            dm2_21=dm2_21,
            dm2_31=dm2_31,
            Nrea=Nrea,
            NL=NL_vals,
            abc=abc_vals,
            nuisance=nuisance_vals,
            bckgs=bckgs_for_chi2,
            snf_noneq_norm=rea_vals,
            eff_nuis=eff_val,
            me_rho=me_rho_val,
            ThU_ratio=ThU_ratio_val,
            core_powers_delta=core_vals,
            efission=efis_vals,
            opp=opp,
            plot_fit=plot_fit,
        )
    
    # --- Minuit chi2 function ---
    def make_chi2_func(n_fixed=5, n_total=None,**extra_kwargs):
        def f(*params):
            return chi2_wrapper(*params[:n_fixed], *params[n_fixed:], **extra_kwargs)
        f.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(f"p{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for i in range(n_total)
            ]
        )
        return f
    
    # --- Start values ---
    start_values = (
        [
            nuosc.op_nom["sin2_th12"],
            nuosc.op_nom["sin2_th13"],
            nuosc.op_nom["dm2_21"]*1e5,
            nuosc.op_nom["dm2_31"]*1e3,
            0.0,# Nrea
        ]
        + NL
        + abc
        + [bckgs[k]["pull"] for k in bckgs_keys]
        + snf_noneq_norm
        + [eff_nuis]
        + [me_rho]
        + [ThU_ratio]
        + nuisance
        + core_powers_delta
        + efission
    )
    
    # --- Parameter names ---
    param_names = (
        ["sin2_th12", "sin2_th13", "dm2_21", "dm2_31", "Nrea"]
        + [f"NL{i}" for i in range(len(NL))]
        + [f"abc{i}" for i in range(len(abc))]
        + [f"bkg_{k}" for k in bckgs_keys]
        + [f"snf_noneq_norm{i}" for i in range(len(snf_noneq_norm))]
        + ["eff_nuis"]
        + ["me_rho"]
        + ["ThU_ratio"]
        + [f"nuis{i}" for i in range(len(nuisance))]
        + [f"core_power{i}" for i in range(len(core_powers_delta))]
        + [f"efission{i}" for i in range(len(efission))]
    )

        
    chi2_func = make_chi2_func(n_fixed=5, n_total=len(start_values))
    m = Minuit(chi2_func, *start_values, name=param_names)
    # for p in m.parameters:
    #     if "bkg" in p:
    #         m.errors[p] = 0.05
    #     if "core_power" in p:
    #         m.errors[p] = 0.1

    # m.errors['abc2'] = 1e-3

    # --- Limits ---
    nuis_limit_minus_names = (
        ["Nrea"] 
        + [f"snf_noneq_norm{i}" for i in range(len(snf_noneq_norm))]
        + [f"bkg_{k}" for k in bckgs_keys]
        + [f"nuis{i}" for i in range(len(nuisance))]
        + ["eff_nuis"]
    )
    
        
    for p in nuis_limit_minus_names:
        m.limits[p] = [-1, None]

    for i in range(len(NL)):
        if args_juno.NL == "pull":
            m.limits[f"NL{i}"] = [-1, None] 
        else:
            m.limits[f"NL{i}"] = [0, None] 
    
    # for i in range(len(abc)):
    #     m.limits[f"abc{i}"] = [0, None] 
        
  
    for i in range(len(core_powers_delta)):
        m.limits[f"core_power{i}"] = [-1.0, None]

    for i in range(len(efission)):
        m.limits[f"efission{i}"] = [0.0, None]  
    m.limits["me_rho"] = [0.0, None]
    m.limits["ThU_ratio"] = [0.0, None]
    
    if not args_juno.silent_print: 
        print("-----Parameter limits-----")
        for p in m.parameters:
            print(p, m.limits[p])
 
    first_fix = (
        ["Nrea"]
        + [f"NL{i}" for i in range(len(NL))]
        + [f"abc{i}" for i in range(len(abc))]
        + [f"snf_noneq_norm{i}" for i in range(len(snf_noneq_norm))]
        + [f"bkg_{k}" for k in bckgs_keys]
        + ["eff_nuis"]
        + ["me_rho"]
        + ["ThU_ratio"]
        + [f"core_power{i}" for i in range(len(core_powers_delta))]
        + [f"efission{i}" for i in range(len(efission))]
    )
    # w, V = np.linalg.eigh(covariance_er)
    # print("eigenvalues:", w)
    # print("cond number:", w.max()/w.min())
    
    # show eigenvectors for the 3 smallest eigenvalues
    # for i in range(3):
    #     idx = i
    #     print("eig", i, "val", w[idx], "vec:", V[:, idx])
    shn_start = n_NL + n_abc + n_bckgs
    for p in m.parameters[:4]:
         #if not args_juno.silent_print: print(f"Fixing {p}")
         m.fixed[p] = True
        
    for p in first_fix:
         #if not args_juno.silent_print: print(f"Fixing {p}")
         m.fixed[p] = True
    
    #m.print_level = 2
    
    m.migrad()
   # print(m)



    print("freeing all osc params")
    for p in m.parameters[:4]:
         #if not args_juno.silent_print: print(f"Freeing {p}")
         m.fixed[p] = False
   
    m.migrad()
   # print(m)
    #m.tol=100
    m.strategy=2
    # print("freeing detector response")
    # for i in range(len(NL)):
    #     m.fixed[f'NL{i}'] = False
    # for i in range(len(abc)):
    #    m.fixed[f'abc{i}'] = False   
    #m.migrad()

    # print("freeing backgrounds")
    # for k in bckgs_keys:
    #     m.fixed[f'bkg_{k}'] = False
    # m.fixed['ThU_ratio'] = False
    #m.migrad()


    print("freeing everything else")
    for p in first_fix:
        if m.fixed[p] == False: continue
        #if not args_juno.silent_print: print(f"Freeing {p}")
        m.fixed[p] = False
        
    # for i in range(len(NL)):
    #     m.fixed[f'NL{i}'] = True
    # for i in range(len(abc)):
    #    m.fixed[f'abc{i}'] = True 
    # for i in range(len(core_powers)):
    #    m.fixed[f'core_power{i}'] = True
   # print(m)


    m.migrad()

    m.hesse()

    # m.hesse()
    #if not args_juno.silent_print and not args_juno.is_data: print(m)

    if not args_juno.silent_print:
        print("Function minimum value: ", m.fval)
        nbins = len(ensp_nom_juno["rdet"].bins) -1
        if args_juno.fit_type == "PMOP_combined" and args_juno.use_dagflow:
            nbins +=  DYB_BINS
        global FREE_PARAM;
        FREE_PARAM = 3+len(nuisance)
        ndf = nbins-FREE_PARAM
        print("Free parameters", FREE_PARAM)
        print("Nbins", nbins)
        print("chi2/ndf: ", m.fval/ndf)
        print("-----Good things, should be True-----")
        print("FMin valid?             ", m.fmin.is_valid)
        print("Has covariance?         ", m.fmin.has_covariance)
        print("Covariance accurate?    ", m.fmin.has_accurate_covar)
        print("Minos valid?            ", m.fmin.is_valid)
        print("-----Bad things, should be False-----")
        print("Above EDM threshold?    ", m.fmin.is_above_max_edm)
        print("Has reached call limit? ", m.fmin.has_reached_call_limit)
        print("Hesse failed?           ", m.fmin.hesse_failed)
    
        

    #if args_juno.is_data:
    if args_juno.write_output:
        output_file = ROOT.TFile(f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_nuis = output_file.mkdir("nuisances")
        dir_post = output_file.mkdir("postfit")
        
    if not args_juno.silent_print:
        print("************Other parameter fit results************")
        for i in range(4, len(m.values)): 
            print(f"{m.parameters[i]}: {m.values[i]} +/- {m.errors[i]}")
      
        print("************Background rates************")
        for j, key in enumerate(bckgs_keys):
            i = 6 + n_NL + n_abc + j
            nuis = m.values[i]
            nuis_err = m.errors[i]
            if key == "geo": 
                start_rate = float(args_juno.geo_tnu_rate)*float(args_juno.tnu_to_cpd)
            else:
                start_rate = backgrounds["JUNO"][key]["rate"]
            rate_unc = backgrounds["JUNO"][key].get("rate_unc", None)  # optional if not always defined
            rate_fit = start_rate * (1 + nuis)
            rate_err = start_rate * nuis_err
        
            # Compute deviation in sigma if rate_unc is available
            if rate_unc:
                deviation_sigma = (rate_fit - start_rate) / (start_rate * rate_unc)
            else:
                deviation_sigma = float('nan')

            print(f"{key}: {rate_fit:.4f} ± {rate_err:.4f} cpd "
                  f"(rel unc: {nuis_err:.3%}; deviation: {deviation_sigma:.2f} σ)")    
            if args_juno.write_output:
                dir_nuis.cd()
                write_parameter(f"{key}", rate_fit, rate_err)

        print("************Normalization parameters************")
        print(f"Rflux: {1.0*(1+m.values['Nrea'])} +/- {1.0*m.errors['Nrea']}") 
        print(f"Eff: {args_juno.detector_efficiency*(1+m.values['eff_nuis'])} +/- {args_juno.detector_efficiency*m.errors['eff_nuis']}")

        print("************Detector response************")
        print(f"a: {m.values['abc0']} +/- {m.errors['abc0']}")
        print(f"b: {m.values['abc1']} +/- {m.errors['abc1']}")
        print(f"c: {m.values['abc2']} +/- {m.errors['abc2']}")
        print(f"kB: {m.values['NL0']} +/- {m.errors['NL0']}")
        print(f"fC: {m.values['NL1']} +/- {m.errors['NL1']}")
        print(f"kI: {m.values['NL2']} +/- {m.errors['NL2']}")
        print(f"LY: {m.values['NL3']} +/- {m.errors['NL3']}")

        if args_juno.write_output:
            dir_nuis.cd()
            write_parameter("Rflux", 1.0*(1+m.values['Nrea']), 1.0*m.errors['Nrea'])
            write_parameter("Eff", args_juno.detector_efficiency*(1+m.values['eff_nuis']), args_juno.detector_efficiency*m.errors['eff_nuis'])
            write_parameter("a", m.values['abc0'], m.errors['abc0'])
            write_parameter("b", m.values['abc1'], m.errors['abc1'])
            write_parameter("c", m.values['abc2'], m.errors['abc2'])
            write_parameter("kB", m.values['NL0'], m.errors['NL0'])
            write_parameter("fC", m.values['NL1'], m.errors['NL1'])
            write_parameter("kI", m.values['NL2'], m.errors['NL2'])
            write_parameter("LY", m.values['NL3'], m.errors['NL3'])

    if args_juno.write_output:
        output_file.Close()
            
                            
    if args_juno.plot_correlation:
    # get correlation matrix
        corr = m.covariance.correlation()
        param_names = m.parameters
        
        import seaborn as sns
        plt.figure(figsize=(30,30))
        sns.heatmap(
            corr,
            annot=True, fmt=".2f",
            xticklabels=param_names,
            yticklabels=param_names,
            cmap="coolwarm", vmin=-1, vmax=1,
            cbar=False  # make colorbar narrower
        )
        plt.tight_layout()
        
        plt.savefig("Plots/corr.pdf")
    
    if args_juno.profile_dm2_31:
        profile_dm2_31(m, args_juno, NMO_opt=bool(args_juno.NMO_opt), write=args_juno.write_output)
    # You can still use your interpolation logic here to find the 1-sigma crossings
    # OR use m.minos() right after for the most precise 1-sigma values
    merrors=bool(args_juno.calc_minuit_errors)
    
    if merrors:
        m.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31") #(params=['x0', 'x1', 'x2', 'x3']) #m.minos() #get minos errors

    osc_params =  [nuosc.op_nom["sin2_th12"], nuosc.op_nom["sin2_th13"], nuosc.op_nom["dm2_21"]*1e5, nuosc.op_nom["dm2_31"]*1e3]

    if args_juno.write_output:
        output_file = ROOT.TFile(f"Postfit_{args_juno.file_prefix}_NO_{args_juno.NMO_opt}_dm2_31_pull_{args_juno.pull_dm2_31}.root", "update")
        dir_osc = output_file.mkdir("oscillation_parameters")
        dir_osc.cd()  
    if not args_juno.silent_print and not args_juno.is_data: 
        print("************Oscillation parameter fit results************")
    for i in range(4): #len(m.values)):
        if merrors:
            err =  max(abs(m.merrors[i].upper), abs(m.merrors[i].lower))
        else:
            err = m.errors[i]
        val = m.values[i]
        exp_val = osc_params[i]
        rel_unc = err / val if val != 0 else float('nan')
        deviation_sigma = (val - exp_val) / err if err != 0 else float('nan')

        #if not args_juno.silent_print and not args_juno.is_data: 
        print(f"{m.parameters[i]}: {val:.6g} +/- {err:.6g}; "
                   f"rel unc: {rel_unc:.3%}; "
                   f"deviation: {deviation_sigma:.2f} σ")
        if args_juno.write_output:
            write_osc_params(str(m.parameters[i]), val, err)
   
    if args_juno.write_output:
        output_file.Close()

    best_params = [m.values[name] for name in m.parameters]
    chi2_NO = chi2_wrapper(*best_params, plot_fit=True) 
    if args_juno.solar_contour:
            x1, y1 = zip(*contour_1sigma)
            x2, y2 = zip(*contour_2sigma)
            
            bf_th12 = m.values["sin2_th12"]
            bf_dm21 = m.values["dm2_21"]
            
            plt.figure(figsize=(7,6))
            plt.plot(x1, y1, 'b-', label='68% CL (1σ)')
            plt.plot(x2, y2, 'r--', label='95% CL (2σ)')
            plt.plot(bf_th12, bf_dm21, 'ko', label='Best fit')
            plt.xlabel(r'$\sin^2\theta_{12}$')
            plt.ylabel(r'$\Delta m^2_{21}$ [eV$^2$]')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig("Plots/solar_contour.png")

    if args_juno.NMO_sig:
        nuosc.SetOscillationParameters(opt=args_juno.PDG_opt, NO=not bool(args_juno.NMO_opt))
        m.values['dm2_31'] = nuosc.op_nom['dm2_31']
        chi2_func_opp = make_chi2_func(n_fixed=5, n_total=len(start_values), opp=True)
        m_opp = Minuit(chi2_func_opp, *m.values, name=param_names)
        m_opp.migrad()
        print(m_opp)
        if merrors:
            m_opp.minos("sin2_th12", "sin2_th13", "dm2_21", "dm2_31")
        best_params_IO = [m_opp.values[name] for name in m_opp.parameters]
        chi2_IO = chi2_wrapper(*best_params_IO)
        if args_juno.profile_dm2_31: profile_dm2_31(m_opp, args_juno,NMO_opt=not bool(args_juno.NMO_opt), write=False)
        print("NMO significance: ", chi2_NO, chi2_IO)



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
    return m

