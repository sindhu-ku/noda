#!/usr/bin/env python3
import sys
import numpy as np
import os
import gc
import pandas as pd
from datetime import datetime
import argparse
import yaml
from . import spectra as spec
from . import matrices as mat
from . import scan as scan
from .noda import *
from .bayesian import bayesian as bayes
from .bayesian import bayesian_results as bayes_res
from . import grid_scan_results as scan_res
from . import minuit as minuit
from . import minuit_pull as minuit_pull
from . import minuit_comb as minuit_comb
from . import minuit_new as minuit_new
from .toymc import run_toy, save_batch_results

def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  start_sp_time = datetime.now()

  #Create parser for config file
  parser_juno = argparse.ArgumentParser()
  parser_nd = argparse.ArgumentParser()
#  parser.add_argument('--config', help='Path to the YAML configuration file')
#  args_juno = parser.parse_args_juno()

  #create parser for yaml file
  with open("config/fit_configuration_inputs.yaml", "r") as file:
    config = yaml.safe_load(file)

     # Parse arguments from the YAML content
    for option, value in config["fit_options"].items():
      parser_juno.add_argument(f"--{option}", default=value)
      parser_nd.add_argument(f"--{option}", default=value)

    for option, value in config["common_det_inputs"].items():
      parser_juno.add_argument(f"--{option}", default=value)
      parser_nd.add_argument(f"--{option}", default=value)

    for option, value in config["juno_inputs"].items():
      parser_juno.add_argument(f"--{option}", default=value)

    for option, value in config["nd_inputs"].items():
      parser_nd.add_argument(f"--{option}", default=value)

  args_juno = parser_juno.parse_args()
  args_nd = parser_nd.parse_args()
  #This is where all the data comes and goes
  if not os.path.exists(f"{args_juno.main_data_folder}"):
    os.mkdir(f"{args_juno.main_data_folder}")

  #livetime calculation in number of days
  ndays =0
  if args_juno.stat_opt[-4:] == "days":
      ndays = float(args_juno.stat_opt[:-4])
  elif args_juno.stat_opt[-4:] == "year":
      ndays = 365.25*float(args_juno.stat_opt[:-4])*args_juno.duty_cycle
  elif args_juno.stat_opt[-5:] == "years":
      ndays = 365.25*float(args_juno.stat_opt[:-5])*args_juno.duty_cycle
  else:
      raise ValueError("only days or year(s) supported")
      
  ndays_nd =0
  if args_nd.stat_opt[-4:] == "days":
      ndays_nd = float(args_nd.stat_opt[:-4])
  elif args_nd.stat_opt[-4:] == "year":
      ndays_nd = 365.25*float(args_nd.stat_opt[:-4])*args_nd.duty_cycle
  elif args_nd.stat_opt[-5:] == "years":
      ndays_nd = 365.25*float(args_nd.stat_opt[:-5])*args_nd.duty_cycle
  else:
      raise ValueError("only days or year(s) supported")

  juno_baselines = args_juno.core_baselines #which reactor baselines and cores
  juno_powers = args_juno.core_powers


  ebins = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins) #energy distributions/pdf binning


  if args_nd.rebin_nonuniform:
      ebins_nd = make_nonuniform_bins(args_nd.bins_nonuniform)
  else:
      ebins_nd = np.linspace(args_nd.min_ene, args_nd.max_ene, args_nd.bins) #energy distributions/pdf binning


  min_temp = args_juno.max_ene + ((args_juno.max_ene- args_juno.min_ene)/args_juno.bins-1) #for combined CM calculation
  max_temp = min_temp + (args_juno.max_ene- args_juno.min_ene)
  ebins_temp = np.linspace(min_temp,max_temp, args_juno.bins)
  #create matrix directory
  if not os.path.exists(f"{args_juno.data_matrix_folder}"):
      os.mkdir(f"{args_juno.data_matrix_folder}")

  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}")
 
  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv_{args_nd.stat_opt}"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv_{args_nd.stat_opt}")

    
  print(" # Run configuration:")
  print("   Statistics:   {} days".format(ndays) )
  print("   NMO opt NO: {}".format(args_juno.NMO_opt))


  ene_leak_nd=None
  # if args_juno.fit_type == "PMOP_combined":
  # #     if os.path.isfile(f"{args_juno.data_matrix_folder}/energy_leak_{args_juno.nd}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_nd.FORCE_CALC_ENE_LEAK:
  # #         ene_leak_nd = LoadRespMatrix(f"{args_juno.data_matrix_folder}/energy_leak_{args_juno.nd}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  # #     else:
  #     if args_juno.nd == "TAO": ene_leak_nd = CalcEnergyLeak(rootfile=args_juno.input_data_file, histname="TAO_response_matrix_25", ebins=ebins, pebins=ebins)
  #     if args_juno.nd == "DYB":
  #         iav_data = np.loadtxt(args_nd.iav_matrix_file)  # shape should be (240, 240)
  #         Emin_crop, Emax_crop = args_nd.min_ene, args_nd.max_ene
  #         n_crop_bins = args_nd.bins

  #         ebins_crop = np.linspace(Emin_crop, Emax_crop, n_crop_bins)


  #         old_ebins = np.linspace(0, 12, 240 + 1)
  #         i_start = np.searchsorted(old_ebins, Emin_crop)
  #         i_stop  = np.searchsorted(old_ebins, Emax_crop)

  #         iav_data_crop = iav_data[i_start:i_stop, i_start:i_stop]
  #         iav_data_rebinned = rebin_matrix(iav_data_crop, old_row_bins=ebins_crop, old_col_bins=ebins_crop, new_row_bins=ebins_nd, new_col_bins=ebins_nd)
  #         ene_leak_nd = RespMatrix(data=iav_data_rebinned, ebins=ebins_nd, pebins=ebins_nd)
  #         ene_leak_nd.Save(f"{args_juno.data_matrix_folder}/energy_leak_{args_juno.nd}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")


      #Create reactor spectra and get backgrounds spectra, function inside spectra.py

  
  with open("config/backgrounds_groupB.yaml", "r") as f:
      backgrounds = yaml.safe_load(f)
      
  ensp_nom_juno, resp_matrix  = spec.CreateSpectra(ndays=ndays,
                                    ebins=ebins,
                                    detector="JUNO",
                                    backgrounds=backgrounds,
                                    args=args_juno)

  if args_juno.fit_type == 'PMOP_combined' and not args_juno.use_dagflow:
      ensp_nom_nd, resp_matrix_nd  = spec.CreateSpectra(ndays=ndays_nd,
                                    ebins=ebins_nd,
                                    detector=args_juno.nd,
                                    backgrounds=backgrounds,
                                    ene_leak_nd=ene_leak_nd,
                                    args=args_nd)

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

  if args_juno.fit_type == 'PMOP_combined' and args_juno.nd == "TAO":
      unc_max_nd = max(args_nd.unc_list_nd, key=len)
      if '+' in unc_max_nd:
          unc_list_nd = unc_max_nd.split('+')
      else:
          unc_list_nd = [unc_max_nd]

      cm_nd = {}

      for u in unc_list_nd:
          if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
              print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              cm_nd[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          else:
              print(f" # Constructing covariance matrix {args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              cm_nd[u] = mat.GetCM(ensp = ensp_nom_nd,
                 resp_matrix=resp_matrix,
                 ndays=ndays,
                 unc=u,
                 ene_leak_nd=ene_leak_nd,
                 detector="nd",
                 args=args_nd,
                 backgrounds=backgrounds)
          cm_nd[u].Dump(f"{args_nd.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
          if args_juno.PLOT_CM: cm_nd[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_nd_{u}.png")
          del cm_nd[u]

      if args_juno.unc_corr_dep:
          cm_corr_dep = {}
          if '+' in args_juno.unc_corr_dep:
              unc_list_corr_dep = args_juno.unc_corr_dep.split('+')
          else:
              unc_list_corr_dep = [args_juno.unc_corr_dep]

          for u in unc_list_corr_dep:
              if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
                  print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
                  cm_corr_dep[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              else:
                  print(f" # Constructing covariance matrices {args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
                  cm_corr_dep[u] = mat.GetCorrCM(ensp_juno = ensp_nom_juno,
                            ensp_nd = ensp_nom_nd,
                            resp_matrix=resp_matrix,
                            ndays=ndays,
                            unc=u,
                            ene_leak_nd=ene_leak_nd,
                            args_juno=args_juno,
                            args_nd=args_nd)
              cm_corr_dep[u].Dump(f"{args_nd.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_corr_{u}.csv")
              if args_juno.PLOT_CM:
                  cm_temp = CovMatrix(data=cm_corr_dep[u].data, bins=ebins+ebins_temp)
                  cm_temp.Plot(f"{args_juno.cov_matrix_plots_folder}/cm_corr_{u}.png")
              del cm_corr_dep[u]

      cm_corr = {}

      if args_juno.unc_corr_ind:
          unc_list_corr_ind = args_juno.unc_corr_ind.split('+')
          for new_unc in unc_list_corr_ind:
              cm_corr[new_unc] = LoadObject(f"{args_juno.data_matrix_folder}/cm_JUNO_{new_unc}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")\
                                .Extend(LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{new_unc}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat"))
              #cm_juno[new_unc].Extend(cm_nd[new_unc])
  
      
  end_cm_time = datetime.now()
  print("Covariance matrices production: ", end_cm_time - start_cm_time)
  start_scan_time = datetime.now()

  unc_list_new_juno = []
  unc_list_new_nd = []
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


  if args_juno.fit_type == 'PMOP_combined' and args_juno.nd == "TAO":
      for unc in args_nd.unc_list_nd:
          unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
          if args_juno.fit_type == 'NMO':
              unc_list_new_nd.append(unc.replace(args_juno.unc_corr_ind+'+', ''))
          else:
              unc_list_new_nd.append(unc)
      for full_unc in unc_list_new_nd:
          single_unc_list = full_unc.split("+")
          cm_nd[full_unc] = CovMatrix(data=np.zeros((args_juno.bins-1, args_juno.bins-1)), bins=ebins)
          for u in single_unc_list:
              cm_nd[full_unc] += LoadObject(f"{args_juno.data_matrix_folder}/cm_nd_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              #cm_nd[u]

      for key in unc_list_new_nd:
          if not key in cm_nd.keys():
              print(" ### WARNING: Covariance matrix '{}' is not available".format(key))
              continue
      if args_juno.unc_corr_ind == args_juno.unc_corr_dep == 'stat':
          unc_corr = 'stat'
      else:
          unc_corr = args_juno.unc_corr_ind+args_juno.unc_corr_dep
          corr_ind_list = args_juno.unc_corr_ind.split("+")
          corr_dep_list = args_juno.unc_corr_dep.split("+")
          cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] = cm_corr[corr_ind_list[0]]
          for u in corr_ind_list[1:]:
              cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] += cm_corr[u]
          for u in corr_dep_list:
               cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] += LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.geo_tnu_rate}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
               #cm_corr_dep[u]

  #run bayesian, function inside bayesian.py and get_results inside bayesian_results.py
  if args_juno.stat_method_opt == 'bayesian':
      # Parallel(n_jobs = -1)(delayed(bayes.run_emcee)(ensp_nom_juno =ensp_nom_juno, baselines = baselines, powers=powers, rm=resp_matrix, cm=cm, SEED=i, args=args_juno) for i in range (args_juno.bayes_seed_beg, args_juno.bayes_seed_beg+args_juno.bayes_nprocesses))
       dm2_31_val = 2.583e-3
       dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 10)
      # Parallel(n_jobs = -1)(delayed(bayes.run_emcee)(ensp_nom_juno =ensp_nom_juno, baselines = baselines, powers=powers, rm=resp_matrix, cm=cm, SEED=i, args=args_juno, dm2_31=m31) for i in range (args_juno.bayes_seed_beg, args_juno.bayes_seed_beg+args_juno.bayes_nprocesses) for m31 in dm2_31_list)
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
          #Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom_juno=ensp_nom_juno, ensp_nom_nd=ensp_nom_nd, unc=unc, rm=resp_matrix, ene_leak_nd=ene_leak_nd, cm_juno=cm_juno, cm_nd=cm_nd, args_juno=args_juno, args_nd=args_nd) for unc in unc_list_new_juno)
          if args_juno.fit_type == 'PMOP_combined':
              if args_juno.nd == "TAO":
                  if not args_juno.pull_all: minuit.run_minuit(ensp_nom_juno=ensp_nom_juno, ensp_nom_nd=ensp_nom_nd, unc_juno=unc_list_new_juno[0].replace(args_juno.unc_corr_ind+'+', ''), unc_nd=unc_list_new_nd[0].replace(args_juno.unc_corr_ind+'+', ''), unc_corr=unc_corr, rm=resp_matrix, rm_nd=resp_matrix_nd, ene_leak_nd=ene_leak_nd, cm_juno=cm_juno, cm_nd=cm_nd,cm_corr=cm_corr, args_juno=args_juno, args_nd=args_nd)
                  else: minuit_pull.run_minuit(ensp_nom_juno=ensp_nom_juno, ensp_nom_nd=ensp_nom_nd, unc_juno=unc_list_new_juno[0].replace(args_juno.unc_corr_ind+'+', ''), unc_nd=unc_list_new_nd[0].replace(args_juno.unc_corr_ind+'+', ''), unc_corr=unc_corr, rm=resp_matrix, rm_nd=resp_matrix_nd, ene_leak_nd=ene_leak_nd, cm_juno=cm_juno, cm_nd=cm_nd,cm_corr=cm_corr, args_juno=args_juno, args_nd=args_nd)
              
              else:
                  if not args_juno.use_dagflow:
                      dyb_V = np.loadtxt(args_nd.dyb_cm)
                      DYB_CM = CovMatrix(dyb_V, bins=ensp_nom_nd['rdet'].bins)
    
                      DYB_spectrum = pd.read_csv(args_nd.dyb_spectrum, delim_whitespace=True, names=["Bin_Center", "Sigma", "Rel_Unc"])
                      DYB_sp = Spectrum(bins=ensp_nom_nd['rdet'].bins, bin_cont=DYB_spectrum["Sigma"].to_numpy())
    
                      DYB_rel_cm = GetRelCM(DYB_CM, DYB_sp)
    
                      obs = ensp_nom_nd['rdet'].bin_cont
                      DYB_abs_CM = DYB_rel_cm.data * np.outer(obs, obs)
    
                      cm_nd = {}
    
                      cm_nd['all'] = DYB_abs_CM 
                  
                      minuit_comb.run_minuit(ndays=ndays, ndays_nd=ndays_nd,ensp_nom_juno=ensp_nom_juno, ensp_nom_nd=ensp_nom_nd, unc_juno=unc_list_new_juno[0], rm = resp_matrix, rm_nd = resp_matrix_nd, cm_juno=cm_juno,cm_nd=cm_nd, ene_leak_nd=ene_leak_nd, args_juno=args_juno, args_nd=args_nd)
                      #minuit_new.run_minuit(ndays=ndays,ensp_nom_juno=ensp_nom_juno, ensp_nom_nd=ensp_nom_nd, unc_juno=unc_list_new_juno[0], rm = resp_matrix,rm_nd = resp_matrix_nd, cm_juno=cm_juno, cm_nd=cm_nd,args_juno=args_juno, args_nd=args_nd, backgrounds=backgrounds)
                  else:    
                      from .DYB_model import make_dyb_model
                      dyb_model = make_dyb_model(args_nd)
                      minuit_new.run_minuit(ndays=ndays,ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0], rm = resp_matrix, cm_juno=cm_juno, args_juno=args_juno, args_nd=args_nd, dyb_model=dyb_model, backgrounds=backgrounds)
                      #minuit_comb.run_minuit(ndays=ndays,ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0], rm = resp_matrix, cm_juno=cm_juno, args_juno=args_juno, args_nd=args_nd, dyb_model=dyb_model)
                      

                      
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
                      # if not args_juno.pull_all:
                      #     minuit.run_minuit(ensp_nom_juno=ensp_nom_juno, unc_juno=unc, rm=resp_matrix, cm_juno=cm_juno, args_juno=args_juno)
                      # else:
                      #     minuit_pull.run_minuit(ensp_nom_juno=ensp_nom_juno, unc_juno=unc, rm=resp_matrix, cm_juno=cm_juno, args_juno=args_juno)
                      minuit_new.run_minuit(ndays=ndays,ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0], rm = resp_matrix, cm_juno=cm_juno, args_juno=args_juno, backgrounds=backgrounds)
                     
          #Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom=ensp_nom_juno, unc=unc_list_new[0], baselines=baselines, powers=powers, rm=resp_matrix, cm=cm, args=args_juno, dm2_31=m31) for m31 in dm2_31_list)

  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
