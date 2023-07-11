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

def fit_parser(yaml_file):
    parser = argparse.ArgumentParser()

    # Load configuration from YAML file
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    # Parse arguments from the YAML content
    for option, value in config["fit_options"].items():
        parser.add_argument(f"--{option}", default=value)
    for option, value in config["juno_inputs"].items():
        parser.add_argument(f"--{option}", default=value)

    return parser

def main(argv):
  start_sp_time = datetime.now()
  # Create parser
  subparser = fit_parser("fit_options.yaml")

  # Parse command-line input
  args = subparser.parse_args()
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
                                    reac_model=args.reactor_model,
                                    core_baselines=baselines,
                                    core_powers=powers,
                                    me_rho=args.me_rho,
                                    ebins=ebins,
                                    ene_crop =args.ene_crop,
                                    ene_crop2 =args.ene_crop2,
                                    pmt_opt=args.pmt_opt,
                                    data_file =args.input_data_file,
                                    FORCE_CALC_RM=args.FORCE_CALC_RM,
                                    plots_folder=args.plots_folder,
                                    data_matrix_folder=args.data_matrix_folder, args=args)


  end_sp_time = datetime.now()
  print("Spectra production time: ", end_sp_time - start_sp_time)

  start_cm_time = datetime.now()
  if os.path.isfile(f"{args.data_matrix_folder}/cm_{args.pmt_opt}_{ndays:.0f}days.dat") and not args.FORCE_CALC_CM:
      print(" # Loading covariance matrices")
      cm = LoadObject(f"{args.data_matrix_folder}/cm_{args.pmt_opt}_{ndays:.0f}days.dat")
  else:
      cm = {}
      print(" # Constructing covariance matrices")
      cm = mat.GetCM(ensp = ensp_nom, sample_size_me=args.sample_size_me,
            sample_size_core=args.sample_size_core,
            sample_size_nonl=args.sample_size_nonl,
            sample_size_resp=args.sample_size_resp,
            data_matrix_folder=args.data_matrix_folder,
            pmt_opt=args.pmt_opt,
            data_file=args.input_data_file,
            me_rho=args.me_rho,
            core_baselines=baselines,
            core_powers=powers,
            resp_matrix=rm,
            ene_crop = args.ene_crop,
            ndays=ndays,
            plots_folder = args.plots_folder,
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


  if args.stat_method_opt == 'bayesian':
      for i in range (args.bayes_seed_beg, args.bayes_seed_end):
          bayesian.run_emcee(PDG_opt = args.PDG_opt, NMO_opt=args.NMO_opt, pmt_opt=args.pmt_opt,stat_opt=args.stat_opt,
                       osc_formula_opt=args.osc_formula_opt, bins = args.bins, ensp_nom =ensp_nom,me_rho = args.me_rho,
                       baselines = baselines, powers=powers, rm=rm, cm=cm, ene_crop=args.ene_crop, SEED=i, unc=args.unc_list[len(args.unc_list)-1], stat_meth=args.bayes_chi2)
  else:
      scan.scan_chi2(sin2_th13_opt = args.sin2_th13_opt, PDG_opt = args.PDG_opt, NMO_opt=args.NMO_opt, pmt_opt=args.pmt_opt,stat_opt=args.stat_opt,
                 osc_formula_opt=args.osc_formula_opt, bins = args.bins, grid=grid, ensp_nom =ensp_nom, unc_list =unc_list_new,
                 me_rho = args.me_rho, baselines = baselines, powers=powers, rm=rm, cm=cm, ene_crop=args.ene_crop, stat_meth=args.stat_method_opt)

  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
