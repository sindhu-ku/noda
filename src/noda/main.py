#!/usr/bin/env python3
import sys
import numpy as np
import os
import gc
import pandas as pd
from datetime import datetime
import argparse
import yaml
from . import spectra_new_single as spec
from . import matrices as mat
from . import scan as scan
from .noda import *

from .bayesian import bayesian as bayes
from .bayesian import bayesian_results as bayes_res
from . import grid_scan_results as scan_res
from . import minuit as minuit
from . import minuit_pull as minuit_pull
from . import minuit_comb as minuit_comb
from .fit import combined_fit as minuit_new
from .toymc import run_toy, save_batch_results
from .constants import *

def _build_ibd_interpolator(ebins=None):

    hbar_MeV = 1.0e-6 * hbar / qe
    MeV_to_cm = (hbar * c * (1.0 / (1.0e6 * qe)))**2 * 1.0e4 

    me2, fsq, gsq = me**2, f**2, g**2
    y2 = (delta_np**2 - me2) * 0.5
    M = (mn + mp) * 0.5

    sigma0 = (2.0 * np.pi**2 / (phase_space * (fsq + 3.0*gsq) * (me**5) * n_lifetime / hbar_MeV))
    n_points = int((12.0- 0.0) / 0.001) + 1
    enu_array = np.linspace(0.0, 12.0, n_points)
    
    cos_theta_arr = np.linspace(-1.0, 1.0, 100)
    dc = 2.0 / (len(cos_theta_arr) - 1)
    xsec = np.zeros_like(enu_array)
    for i, Enu in enumerate(enu_array):
        Ee0 = Enu - delta_np
        if Ee0 <= me: continue
        pe0 = np.sqrt(Ee0**2 - me2)
        ve0 = pe0 / Ee0
        sum_sigma_0 = 0.0
        sum_sigma_1 = 0.0
        for j, ctheta in enumerate(cos_theta_arr):
            # Calculate Ee1 regardless
            Ee1 = Ee0 * (1.0 - Enu / M * (1.0 - ve0 * ctheta)) - y2 / M

            # Use an if/else to keep the point in the sum
            if Ee1 > me:
                pe1 = np.sqrt(Ee1**2 - me2)
                ve1 = pe1 / Ee1

                t0 = (sigma0 * 0.5 * ((fsq + 3.0*gsq) + (fsq - gsq) * ve1 * ctheta) * Ee1 * pe1)

                g1 = 2.0 * g * (f + f2) * ((2.0 * Ee0 + delta_np) * (1.0 - ve0 * ctheta) - me2 / Ee0)
                g2 = (fsq + gsq) * (delta_np * (1.0 + ve0 * ctheta) + me2 / Ee0)
                A_val = (Ee0 + delta_np) * (1.0 - ctheta / ve0) - delta_np
                g3 = (fsq + 3.0 * gsq) * A_val
                g4 = (fsq - gsq) * A_val * ve0 * ctheta

                t1 = (-0.5 * sigma0 * Ee0 * pe0 * (g1 + g2 + g3 + g4) / M)
            else:
                t0 = 0.0
                t1 = 0.0

            w = 0.5 if (j == 0 or j == len(cos_theta_arr) - 1) else 1.0
            sum_sigma_0 += w * t0
            sum_sigma_1 += w * t1
        xsec[i] = (sum_sigma_0 + sum_sigma_1) * dc * MeV_to_cm
    return sp.interpolate.interp1d(
        enu_array, xsec, kind='cubic', bounds_error=False, fill_value=(0.0, 0.0)
    )

def _build_ibd_interpolator_root(input_data_file):
    """Return (spectrum, interpolator) for the IBD cross-section."""
    sibd_spec = GetSpectrumFromROOT(input_data_file, "IBDXsec_VogelBeacom_DYB")
    s_ibd = sp.interpolate.interp1d(
        sibd_spec.GetBinCenters(), sibd_spec.bin_cont,
        kind="slinear", bounds_error=False, fill_value=(0.0, 0.0),
    )
    return s_ibd

def _build_bump_interpolator(input_data_file):
    bump = GetSpectrumFromROOT(input_data_file, "DYBFluxBump_ratio")
    return sp.interpolate.interp1d(
        bump.GetBinCenters(), bump.bin_cont,
        kind="slinear", bounds_error=False,
        fill_value=(bump.bin_cont[0], bump.bin_cont[-1]),
    )

def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  start_sp_time = datetime.now()
  # --- STEP 1: Pre-parser to get the config file ---
  conf_parser = argparse.ArgumentParser(add_help=False) # No help yet, or it will trigger early
  conf_parser.add_argument("--config", default="config/groupA_paperconfig.yaml", help="Path to config file")
  
  # Use parse_known_args to get only the --config flag
  args, remaining_argv = conf_parser.parse_known_args()
  
  # --- STEP 2: Load the YAML ---
  with open(args.config, "r") as file:
      config = yaml.safe_load(file)
  
  # --- STEP 3: Full Parser setup ---
  # Include parents=[conf_parser] so --config still shows up in -h
  parser_juno = argparse.ArgumentParser(parents=[conf_parser])
  parser_dyb = argparse.ArgumentParser(parents=[conf_parser])
  parser_tao = argparse.ArgumentParser(parents=[conf_parser])
  
  # Dynamically add your YAML arguments as defaults
  sections = {
      "fit_options": [parser_juno, parser_dyb, parser_tao],
      "common_det_inputs": [parser_juno, parser_dyb, parser_tao],
      "juno_inputs": [parser_juno],
      "dyb_inputs": [parser_dyb],
      "tao_inputs": [parser_tao]
  }
  
  for section, parsers in sections.items():
      if section in config:
          for option, value in config[section].items():
              for p in parsers:
                  # Add as argument so it can be overridden by command line
                  p.add_argument(f"--{option}", default=value)
  
  # --- STEP 4: Final Parse ---
  # Now parse the 'remaining_argv' (everything except the --config part)
  # Example for JUNO:
  args_juno = parser_juno.parse_args(remaining_argv)
  args_dyb = parser_dyb.parse_args()
  args_tao = parser_tao.parse_args()
    
  if not os.path.exists(f"{args_juno.main_data_folder}"):
    os.mkdir(f"{args_juno.main_data_folder}")
  
  ndays = calculate_ndays(args_juno)
  ndays_dyb  = calculate_ndays(args_dyb)
  ndays_tao  = calculate_ndays(args_tao)

  print("JUNO livetime: ", ndays, "days")
  print("DYB livetime: ", ndays_dyb, "days")
  print("TAO livetime: ", ndays_tao, "days")
    
  juno_baselines = args_juno.core_baselines #which reactor baselines and cores
  juno_powers = args_juno.core_powers


  ebins = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins) #energy distributions/pdf binning
  ebins_dyb = np.linspace(args_dyb.min_ene, args_dyb.max_ene, args_dyb.bins) #energy distributions/pdf binning
  ebins_tao = np.linspace(args_tao.min_ene, args_tao.max_ene, args_tao.bins) #energy distributions/pdf binning

  min_temp = args_juno.max_ene + ((args_juno.max_ene- args_juno.min_ene)/args_juno.bins-1) #for combined CM calculation
  max_temp = min_temp + (args_juno.max_ene- args_juno.min_ene)
  ebins_temp = np.linspace(min_temp,max_temp, args_juno.bins)
  #create matrix directory
  if not os.path.exists(f"{args_juno.data_matrix_folder}"):
      os.mkdir(f"{args_juno.data_matrix_folder}")

  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}")
 
  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv_{args_dyb.stat_opt}"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv_{args_dyb.stat_opt}")

    
  print(" # Run configuration:")
  print("   Statistics:   {} days".format(ndays) )
  print("   NMO opt NO: {}".format(args_juno.NMO_opt))


      #Create reactor spectra and get backgrounds spectra, function inside spectra.py
  s_ibd  = _build_ibd_interpolator(ebins)
 # s_ibd = _build_ibd_interpolator_root(args_juno.input_data_file)

  s_bump_lin = _build_bump_interpolator(args_juno.input_data_file)

  
  with open(args_juno.backgrounds_yaml, "r") as f:
      backgrounds = yaml.safe_load(f)
      
  ensp_nom_juno, resp_matrix  = spec.CreateSpectra(ndays=ndays,
                                    ebins=ebins,
                                    detector="JUNO",
                                    backgrounds=backgrounds,
                                    s_ibd=s_ibd, s_bump_lin=s_bump_lin,
                                    args=args_juno)
  

  
  if args_juno.fit_type == 'PMOP_combined':
      if not args_juno.use_dagflow:
          ensp_nom_dyb, resp_matrix_dyb  = spec.CreateSpectra(ndays=ndays_dyb,
                                    ebins=ebins_dyb,
                                    detector="DYB",
                                    backgrounds=backgrounds,
                                    s_ibd=s_ibd, s_bump_lin=s_bump_lin,                
                                    args=args_dyb) 
      ensp_nom_tao, resp_matrix_tao, ene_leak_tao  = spec.CreateSpectra(ndays=ndays_tao,
                                    ebins=ebins_tao,
                                    detector="TAO",
                                    backgrounds=backgrounds,
                                    s_ibd=s_ibd, s_bump_lin=s_bump_lin,                    
                                    args=args_tao)     


  end_sp_time = datetime.now()
  print("Spectra production time: ", end_sp_time - start_sp_time)
  
  #Create covariance matrices and energy response matrix, function inside matrices.py
  start_cm_time = datetime.now()

  unc_max_juno = max(args_juno.unc_list_juno, key=len)
  if '+' in unc_max_juno:
      unc_list_juno = unc_max_juno.split('+')
  else:
      unc_list_juno = [unc_max_juno]

  cm_juno = {}

  for u in unc_list_juno:
      if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
          print(" # Loading covariance matriix", f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          cm_juno[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

      else:
          print(f" # Constructing covariance matrix {args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          cm_juno[u] = mat.GetCM(ensp = ensp_nom_juno,
            resp_matrix=resp_matrix,
            ndays=ndays,
            unc=u,
            detector="JUNO",
            args=args_juno,
            backgrounds=backgrounds)

      cm_juno[u].Dump(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
      if args_juno.PLOT_CM: cm_juno[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_JUNO_{u}.png")
      del cm_juno[u]

  # if args_juno.fit_type == 'PMOP_combined' and args_juno.nd == "TAO":
  #     unc_max_nd = max(args_nd.unc_list_nd, key=len)
  #     if '+' in unc_max_nd:
  #         unc_list_nd = unc_max_nd.split('+')
  #     else:
  #         unc_list_nd = [unc_max_nd]

  #     cm_nd = {}

  #     for u in unc_list_nd:
  #         if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
  #             print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #             cm_nd[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #         else:
  #             print(f" # Constructing covariance matrix {args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #             cm_nd[u] = mat.GetCM(ensp = ensp_nom_nd,
  #                resp_matrix=resp_matrix,
  #                ndays=ndays,
  #                unc=u,
  #                ene_leak_nd=ene_leak_nd,
  #                detector="nd",
  #                args=args_nd,
  #                backgrounds=backgrounds)
  #         cm_nd[u].Dump(f"{args_nd.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
  #         if args_juno.PLOT_CM: cm_nd[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_nd_{u}.png")
  #         del cm_nd[u]

  #     if args_juno.unc_corr_dep:
  #         cm_corr_dep = {}
  #         if '+' in args_juno.unc_corr_dep:
  #             unc_list_corr_dep = args_juno.unc_corr_dep.split('+')
  #         else:
  #             unc_list_corr_dep = [args_juno.unc_corr_dep]

  #         for u in unc_list_corr_dep:
  #             if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
  #                 print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #                 cm_corr_dep[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #             else:
  #                 print(f" # Constructing covariance matrices {args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #                 cm_corr_dep[u] = mat.GetCorrCM(ensp_juno = ensp_nom_juno,
  #                           ensp_nd = ensp_nom_nd,
  #                           resp_matrix=resp_matrix,
  #                           ndays=ndays,
  #                           unc=u,
  #                           ene_leak_nd=ene_leak_nd,
  #                           args_juno=args_juno,
  #                           args_nd=args_nd)
  #             cm_corr_dep[u].Dump(f"{args_nd.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_corr_{u}.csv")
  #             if args_juno.PLOT_CM:
  #                 cm_temp = CovMatrix(data=cm_corr_dep[u].data, bins=ebins+ebins_temp)
  #                 cm_temp.Plot(f"{args_juno.cov_matrix_plots_folder}/cm_corr_{u}.png")
  #             del cm_corr_dep[u]

  #     cm_corr = {}

  #     if args_juno.unc_corr_ind:
  #         unc_list_corr_ind = args_juno.unc_corr_ind.split('+')
  #         for new_unc in unc_list_corr_ind:
  #             cm_corr[new_unc] = LoadObject(f"{args_juno.data_matrix_folder}/cm_JUNO_{new_unc}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")\
  #                               .Extend(LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{new_unc}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat"))
  #             #cm_juno[new_unc].Extend(cm_nd[new_unc])
  

  end_cm_time = datetime.now()
  print("Covariance matrices production: ", end_cm_time - start_cm_time)
  start_scan_time = datetime.now()

  unc_list_new_juno = []
  #unc_list_new_nd = []
  #TODO: This is mainly for CNP when stat matrix is included by default, can be done better
  for unc in args_juno.unc_list_juno:
    unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
    unc_list_new_juno.append(unc.replace(args_juno.unc_corr_ind+'+', ''))

  for full_unc in unc_list_new_juno:
    single_unc_list = full_unc.split("+")
    cm_juno[full_unc] = CovMatrix(data=np.zeros((len(ensp_nom_juno['rdet'].bins)-1, len(ensp_nom_juno['rdet'].bins)-1)), bins=ebins)
    for u in single_unc_list:
      cm_juno[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_JUNO_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
      #print(u)
      cm_juno[full_unc] += cm_juno[u]


  # if args_juno.fit_type == 'PMOP_combined' and args_juno.nd == "TAO":
  #     for unc in args_nd.unc_list_nd:
  #         unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
  #         if args_juno.fit_type == 'NMO':
  #             unc_list_new_nd.append(unc.replace(args_juno.unc_corr_ind+'+', ''))
  #         else:
  #             unc_list_new_nd.append(unc)
  #     for full_unc in unc_list_new_nd:
  #         single_unc_list = full_unc.split("+")
  #         cm_nd[full_unc] = CovMatrix(data=np.zeros((args_juno.bins-1, args_juno.bins-1)), bins=ebins)
  #         for u in single_unc_list:
  #             cm_nd[full_unc] += LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #             #cm_nd[u]

  #     for key in unc_list_new_nd:
  #         if not key in cm_nd.keys():
  #             print(" ### WARNING: Covariance matrix '{}' is not available".format(key))
  #             continue
  #     if args_juno.unc_corr_ind == args_juno.unc_corr_dep == 'stat':
  #         unc_corr = 'stat'
  #     else:
  #         unc_corr = args_juno.unc_corr_ind+args_juno.unc_corr_dep
  #         corr_ind_list = args_juno.unc_corr_ind.split("+")
  #         corr_dep_list = args_juno.unc_corr_dep.split("+")
  #         cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] = cm_corr[corr_ind_list[0]]
  #         for u in corr_ind_list[1:]:
  #             cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] += cm_corr[u]
  #         for u in corr_dep_list:
  #              cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] += LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  #              #cm_corr_dep[u]

  #run bayesian, function inside bayesian.py and get_results inside bayesian_results.py
  if args_juno.stat_method_opt == 'bayesian':
       dm2_31_val = 2.583e-3
       dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 10)
       bayes_res.get_results(args=args_juno)

 #For frequentist, function inside scan.py
  else:
      if(args_juno.grid_scan):
          #Form grid for the gridscan
          grid ={'sin2_12': np.linspace(args_juno.grid_params['sin2_12'][0],args_juno.grid_params['sin2_12'][1],args_juno.grid_points),
          'sin2_13': np.linspace(args_juno.grid_params['sin2_13'][0],args_juno.grid_params['sin2_13'][1],args_juno.grid_points),
          'dm2_21': np.linspace(args_juno.grid_params['dm2_21'][0],args_juno.grid_params['dm2_21'][1],args_juno.grid_points),
          'dm2_31': np.linspace(args_juno.grid_params['dm2_31'][0],args_juno.grid_params['dm2_31'][1],args_juno.grid_points)}
          scan.scan_chi2(grid=grid, ensp_nom_juno =ensp_nom_juno, unc_list =unc_list_new,
                      baselines = baselines, powers=powers, rm=resp_matrix, cm=cm, args=args_juno)
          scan_res.get_results(args=args_juno)
      else:
          cm_tao = {}
          cm_tao['all'] = rebin_cov_matrix(GetCovMatrixFromROOT(args_tao.data_file, args_tao.FV_cov_matrix), args_tao.bins_nonuniform)
          
          if args_juno.fit_type == 'PMOP_combined':
              if not args_juno.use_dagflow:
                  dyb_V = np.loadtxt(args_dyb.dyb_cm)
                  if args_dyb.model == "PRL":
                      raw = dyb_V
                      cov_75 = raw.reshape(75, 75)
                      cov_total = cov_75[50:75, 50:75]
                      cov_total *= 1e-86  # if needed
                      dyb_V = cov_total
                  DYB_CM = CovMatrix(dyb_V, bins=ensp_nom_dyb['rdet'].bins)

                  if args_dyb.model == "CPC": 
                      DYB_spectrum = pd.read_csv(args_dyb.dyb_spectrum, sep='\s+', names=["Bin_Center", "Sigma", "Rel_Unc"])
                  else:
                      DYB_spectrum = pd.read_csv(args_dyb.dyb_spectrum, sep='\s+', names=["Bin_lowE", "Bin_highE", "Sigma", "Rel_Unc"])
                      
                  DYB_sp = Spectrum(bins=ensp_nom_dyb['rdet'].bins, bin_cont=DYB_spectrum["Sigma"].to_numpy())

                  DYB_rel_cm = GetRelCM(DYB_CM, DYB_sp)

                  obs = ensp_nom_dyb['rdet'].bin_cont
                  DYB_abs_CM = CovMatrix(data=DYB_rel_cm.data * np.outer(obs, obs), bins=DYB_CM.bins)

                  cm_dyb = {}

                  cm_dyb['all'] = DYB_rel_cm

                  minuit_new.run_minuit(ndays=ndays,ndays_dyb=ndays_dyb,ndays_tao=ndays_tao, s_ibd=s_ibd, s_bump_lin=s_bump_lin, ensp_nom_juno=ensp_nom_juno, ensp_nom_dyb=ensp_nom_dyb, ensp_nom_tao=ensp_nom_tao, unc_juno=unc_list_new_juno[0], unc_tao='all', rm = resp_matrix,rm_dyb = resp_matrix_dyb,rm_tao=resp_matrix_tao, cm_juno=cm_juno, cm_dyb=cm_dyb,cm_tao=cm_tao,args_juno=args_juno, args_dyb=args_dyb, args_tao=args_tao,ene_leak_tao=ene_leak_tao, backgrounds=backgrounds)
              else:    
                  from .DYB_model import make_dyb_model
                  dyb_model = make_dyb_model(args_dyb)
                  minuit_new.run_minuit(ndays=ndays,ndays_tao=ndays_tao, s_ibd=s_ibd, s_bump_lin=s_bump_lin, ensp_nom_juno=ensp_nom_juno, ensp_nom_tao=ensp_nom_tao, unc_juno=unc_list_new_juno[0], unc_tao='stat', rm = resp_matrix, rm_tao=resp_matrix_tao, cm_juno=cm_juno, cm_tao=cm_tao, args_juno=args_juno, args_dyb=args_dyb, args_tao=args_tao,dyb_model=dyb_model,ene_leak_tao=ene_leak_tao, backgrounds=backgrounds)
                      

                      
          else:
              if bool(args_juno.toymc):
                  for batch_start in range(0, int(args_juno.ntoys), args_juno.toy_batch_size):
                      batch_start_t = datetime.now()
                      batch_end = min(batch_start + args_juno.toy_batch_size, int(args_juno.ntoys))
                      batch_results = Parallel(n_jobs=-1, timeout=args_juno.toymc_timeout)(delayed(run_toy)(i=t, ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0],\
                           resp_matrix=resp_matrix, cm_juno=cm_juno, args_juno=args_juno) for t in range(batch_start, batch_end))

                      filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.fit_type}_UThfree_{args_juno.stat_method_opt}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit_{args_juno.toy_prefix}.hdf5"
                      save_batch_results(filename, batch_results)
                      del batch_results
                      if not args_juno.silent_print: print(f"Batch time: {datetime.now() - batch_start_t}")
              else:
                  for unc in unc_list_new_juno: 
                      minuit_new.run_minuit(ndays=ndays,ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0], rm = resp_matrix, cm_juno=cm_juno, args_juno=args_juno, backgrounds=backgrounds)
                     


  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
