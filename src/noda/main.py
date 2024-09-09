#!/usr/bin/env python3
import sys
import numpy as np
import os
import gc
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

def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]

  start_sp_time = datetime.now()

  #Create parser for config file
  parser_juno = argparse.ArgumentParser()
  parser_tao = argparse.ArgumentParser()
#  parser.add_argument('--config', help='Path to the YAML configuration file')
#  args_juno = parser.parse_args_juno()

  #create parser for yaml file
  with open("config/fit_configuration_inputs.yaml", "r") as file:
    config = yaml.safe_load(file)

     # Parse arguments from the YAML content
    for option, value in config["fit_options"].items():
      parser_juno.add_argument(f"--{option}", default=value)
      parser_tao.add_argument(f"--{option}", default=value)

    for option, value in config["common_det_inputs"].items():
      parser_juno.add_argument(f"--{option}", default=value)
      parser_tao.add_argument(f"--{option}", default=value)

    for option, value in config["juno_inputs"].items():
      parser_juno.add_argument(f"--{option}", default=value)

    for option, value in config["tao_inputs"].items():
      parser_tao.add_argument(f"--{option}", default=value)

  args_juno = parser_juno.parse_args()
  args_tao = parser_tao.parse_args()
  #This is where all the data comes and goes
  if not os.path.exists(f"{args_juno.main_data_folder}"):
    os.mkdir(f"{args_juno.main_data_folder}")

  #livetime calculation in number of days
  ndays =0
  if args_juno.stat_opt[-4:] == "days":
      ndays = float(args_juno.stat_opt[:-4])
  elif args_juno.stat_opt[-4:] == "year":
      ndays = 365.25*float(args_juno.stat_opt[:-4])
  elif args_juno.stat_opt[-5:] == "years":
      ndays = 365.25*float(args_juno.stat_opt[:-5])
  else:
      raise ValueError("only days or year(s) supported")

  ndays *= args_juno.duty_cycle #for nuclear reactor livetime, effectively only 11 out of 12 months in a year

  juno_baselines = args_juno.core_baselines #which reactor baselines and cores
  juno_powers = args_juno.core_powers
  ebins = np.linspace(args_juno.min_ene, args_juno.max_ene, args_juno.bins) #energy distributions/pdf binning

  #create matrix directory
  if not os.path.exists(f"{args_juno.data_matrix_folder}"):
      os.mkdir(f"{args_juno.data_matrix_folder}")

  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv")

  print(" # Run configuration:")
  print("   Statistics:   {} days".format(ndays) )

  #   Energy resolution
  a, b, c = args_juno.a, args_juno.b, args_juno.c
  a_err, b_err, c_err =args_juno.a_err, args_juno.b_err, args_juno.c_err

  if os.path.isfile(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_RM:
    resp_matrix = LoadRespMatrix(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  else:
    resp_matrix = CalcRespMatrix_abc(a, b, c, escale=1, ebins=ebins, pebins=ebins)
    resp_matrix.Save(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

  if args_juno.FORCE_CALC_RM:
    resp_matrix.Dump(f"{args_juno.data_matrix_folder}/csv/resp_matrix.csv")

  if os.path.isfile(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_RM:
    ene_leak_tao = LoadRespMatrix(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  else:
    ene_leak_tao = CalcEnergyLeak(rootfile=args_juno.input_data_file, histname="TAO_response_matrix_25", ebins=ebins, pebins=ebins)
    ene_leak_tao.Save(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

  if args_tao.FORCE_CALC_ENE_LEAK:
    ene_leak_tao.Dump(f"{args_juno.data_matrix_folder}/csv/ene_leak_tao_matrix.csv")
  #Create reactor spectra and get backgrounds spectra, function inside spectra.py
  ensp_nom_juno  = spec.CreateSpectra(ndays=ndays,
                                    ebins=ebins,
                                    detector="juno",
                                    resp_matrix=resp_matrix,
                                    args=args_juno)

  ensp_nom_tao  = spec.CreateSpectra(ndays=ndays,
                                    ebins=ebins,
                                    detector="tao",
                                    resp_matrix=resp_matrix,
                                    ene_leak_tao=ene_leak_tao,
                                    args=args_tao)

  end_sp_time = datetime.now()
  print("Spectra production time: ", end_sp_time - start_sp_time)
  #Create covariance matrices and energy response matrix, function inside matrices.py
  start_cm_time = datetime.now()

  if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
      print(" # Loading covariance matrices", f"{args_juno.data_matrix_folder}/cm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
      cm = LoadObject(f"{args_juno.data_matrix_folder}/cm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

  else:
      cm = {}
      print(f" # Constructing covariance matrices {args_juno.data_matrix_folder}/cm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
      cm = mat.GetCM(ensp = ensp_nom_juno,
            core_baselines=juno_baselines,
            core_powers=juno_powers,
            resp_matrix=resp_matrix,
            ndays=ndays,
            args=args_juno)
  if args_juno.PLOT_CM:
    if not os.path.exists(f"{args_juno.cov_matrix_plots_folder}"):
      os.makedirs(f"{args_juno.cov_matrix_plots_folder}")
    for key, cov_mat in cm.items():
      cov_mat.Plot(f"{args_juno.cov_matrix_plots_folder}/cm_{key}.png")

  end_cm_time = datetime.now()
  print("Covariance matrices production: ", end_cm_time - start_cm_time)
  start_scan_time = datetime.now()

  unc_list_new = []
  #TODO: This is mainly for CNP when stat matrix is included by default, can be done better
  print("Uncertainty list for ", args_juno.stat_method_opt)
  for unc in args_juno.unc_list:
    unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
    print(unc)
    unc_list_new.append(unc)

  for full_unc in unc_list_new:
    single_unc_list = full_unc.split("+")
    cm[full_unc] = cm[single_unc_list[0]]
    for u in single_unc_list[1:]:
      cm[full_unc] += cm[u]

  for key in unc_list_new:
    if not key in cm.keys():
      print(" ### WARNING: Covariance matrix '{}' is not available".format(key))
      continue
    #
    # if not cm[key].IsInvertible():
    #   print(" ### WARNING: Covariance matrix for '{}' is not invertible and can not be used to calculate Chi2".format(key))
    #   del cm[key]
    #   continue


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
          Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom_juno=ensp_nom_juno, ensp_nom_tao=ensp_nom_tao, unc=unc, rm=resp_matrix, cm_juno=cm, cm_tao=cm, args_juno=args_juno, args_tao=args_tao) for unc in unc_list_new)
         # dm2_31_val = 2.5283e-3
         # dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 100 )
          #Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom=ensp_nom_juno, unc=unc_list_new[0], baselines=baselines, powers=powers, rm=resp_matrix, cm=cm, args=args_juno, dm2_31=m31) for m31 in dm2_31_list)

  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
