import sys
#from fitter_options import *
import spectra as spec
import matrices as mat
import scan as scan
from noda import *
import bayesian as bayesian
import numpy as np
import os
import gc
from datetime import datetime
import argparse
import yaml

def main(argv):

  start_sp_time = datetime.now()

  # Create parser for config file
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', help='Path to the YAML configuration file')
  args = parser.parse_args()

  #create parser for yaml file
  with open(args.config, "r") as file:
    config = yaml.safe_load(file)

     # Parse arguments from the YAML content
    for option, value in config["fit_options"].items():
      parser.add_argument(f"--{option}", default=value)
    for option, value in config["juno_inputs"].items():
      parser.add_argument(f"--{option}", default=value)
  args = parser.parse_args()
  # Parse command-line input
  #livetime calculation in number of days
  ndays =0
  if args.stat_opt[-4:] == "days":
      ndays = int(args.stat_opt[:-4])
  elif args.stat_opt[-4:] == "year":
      ndays = 365.25*int(args.stat_opt[:-4])
  elif args.stat_opt[-5:] == "years":
      ndays = 365.25*int(args.stat_opt[:-5])
      ndays *= (11/12) #for nuclear reactor livetime, effectively only 11 out of 12 months in a year

  livetime = ndays
  baselines = args.core_baselines_9 #which reactor baselines and cores
  powers = args.core_powers_9
  ebins = np.linspace(args.min_ene, args.max_ene, args.bins) #energy distributions/pdf binning

  #create matrix directory
  if not os.path.exists(f"{args.data_matrix_folder}"):
      os.mkdir(f"{args.data_matrix_folder}")

  if not os.path.exists(f"{args.data_matrix_folder}/csv"):
      os.mkdir(f"{args.data_matrix_folder}/csv")

  print(" # Run configuration:")
  print("   Statistics:   {} days".format(ndays) )
  print("   PMT system:   {}".format(args.pmt_opt) )

  ensp_nom, rm, opt = spec.Initialize(ndays=ndays,
                                    core_baselines=baselines,
                                    core_powers=powers,
                                    ebins=ebins,
                                    args=args)


  end_sp_time = datetime.now()
  print("Spectra production time: ", end_sp_time - start_sp_time)

  start_cm_time = datetime.now()
  if os.path.isfile(f"{args.data_matrix_folder}/cm_{args.pmt_opt}_{ndays:.0f}days.dat") and not args.FORCE_CALC_CM:
      print(" # Loading covariance matrices")
      cm = LoadObject(f"{args.data_matrix_folder}/cm_{args.pmt_opt}_{ndays:.0f}days.dat")
  else:
      cm = {}
      print(" # Constructing covariance matrices")
      cm = mat.GetCM(ensp = ensp_nom,
            core_baselines=baselines,
            core_powers=powers,
            resp_matrix=rm,
            ndays=ndays,
            args=args)

  if args.PLOT_CM:
    if not os.path.exists(f"{args.cov_matrix_plots_folder}"):
      os.makedirs(f"{args.cov_matrix_plots_folder}")
    for key, cov_mat in cm.items():
      cov_mat.Plot(f"{args.cov_matrix_plots_folder}/cm_{key}.png")

  end_cm_time = datetime.now()
  print("Covariance matrices production: ", end_cm_time - start_cm_time)
  start_scan_time = datetime.now()

  unc_list_new = []
  #TODO: This is mainly for CNP when stat matrix is included by default, can be done better
  for unc in args.unc_list:
    if((args.stat_method_opt == 'CNP' or args.bayes_chi2 == 'CNP') and unc != 'stat'):
        unc = unc.replace('stat+', "")
    unc_list_new.append(unc)

  for full_unc in unc_list_new:
    print(full_unc)
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

  grid ={'sin2_12': np.linspace(args.grid_params['sin2_12'][0],args.grid_params['sin2_12'][1],args.grid_points),
  'sin2_13': np.linspace(args.grid_params['sin2_13'][0],args.grid_params['sin2_13'][1],args.grid_points),
  'dm2_21': np.linspace(args.grid_params['dm2_21'][0],args.grid_params['dm2_21'][1],args.grid_points),
  'dm2_31': np.linspace(args.grid_params['dm2_31'][0],args.grid_params['dm2_31'][1],args.grid_points)}


  if args.stat_method_opt == 'bayesian': #have to paraleelize this still
      for i in range (args.bayes_seed_beg, args.bayes_seed_beg+args.bayes_nprocesses):
          bayesian.run_emcee(ensp_nom =ensp_nom, baselines = baselines, powers=powers, rm=rm, cm=cm, SEED=i, args=args)
      bayesian.get_results(args=args)
  else:
      scan.scan_chi2(grid=grid, ensp_nom =ensp_nom, unc_list =unc_list_new,
                     baselines = baselines, powers=powers, rm=rm, cm=cm, args=args)

  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
