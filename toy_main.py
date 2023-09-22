import sys
import numpy as np
import os
import gc
from datetime import datetime
import argparse
import yaml
import toy_spectra as spec
import matrices as mat
import scan as scan
from noda import *
import bayesian as bayes
import bayesian_results as bayes_res
import frequentist_results as freq_res
import toy_minuit as minuit


def toy_mc(sin2_12_init=0., sin2_13_init=0., dm2_21_init=0., dm2_31_init=0., ndays=0,
                                       core_baselines=[],
                                       core_powers=[],
                                       ebins=None,
                                       args=''):
     #Create reactor spectra and get backgrounds spectra, function inside spectra.py
     ensp_nom, rm, opt = spec.Initialize(ndays=ndays,
                                       core_baselines=core_baselines,
                                       core_powers=core_powers,
                                       ebins=ebins, sin2_th12=sin2_12_init,
                                       sin2_th13=sin2_13_init,
                                       dm2_21=dm2_21_init,
                                       dm2_31=dm2_31_init,
                                       args=args)



     #Create covariance matrices and energy response matrix, function inside matrices.py
     start_cm_time = datetime.now()
     if os.path.isfile(f"{args.data_matrix_folder}/cm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat") and not args.FORCE_CALC_CM:
         print(" # Loading covariance matrices")
         cm = LoadObject(f"{args.data_matrix_folder}/cm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
     else:
         cm = {}
         print(" # Constructing covariance matrices")
         cm = mat.GetCM(ensp = ensp_nom,
               core_baselines=core_baselines,
               core_powers=core_powers,
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


     #run bayesian, function inside bayesian.py and get_results inside bayesian_results.py
     sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2 = minuit.run_minuit(ensp_nom=ensp_nom, baselines=core_baselines, powers=core_powers, rm=rm, cm=cm, args=args)

     return sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2



def main(argv):
  if (len(sys.argv) <  2):
      print("ERROR: Please give the config file using the option --config_file=<filename>")

  start_sp_time = datetime.now()

  #Create parser for config file
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

  #This is where all the data comes and goes
  if not os.path.exists(f"{args.main_data_folder}"):
    os.mkdir(f"{args.main_data_folder}")

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
  start_toy_time = datetime.now()
  nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt)

  num_experiments = 100

  # Define the parameters and their errors as arrays
  mean_values = np.array([nuosc.op_nom['sin2_th12'], nuosc.op_nom['sin2_th13'], nuosc.op_nom['dm2_21'], nuosc.op_nom['dm2_31']])
  sigma_values = np.array([nuosc.op_nom['sin2_th12_err'], nuosc.op_nom['sin2_th13_err'], nuosc.op_nom['dm2_21_err'], nuosc.op_nom['dm2_31_err']])

  # Generate random Gaussian values for all parameters at once
  random_values = np.random.normal(mean_values, sigma_values, (num_experiments, len(mean_values)))

  # Initialize empty arrays for the results
  sin2_12_arr, sin2_13_arr, dm2_21_arr, dm2_31_arr, delta_chi2_arr = [], [], [], [], []
  def run_experiment(i):
      print("Toy experiment {}".format(i))
      print("Toy parameters {}".format(random_values[i]))
      sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2 = toy_mc(sin2_12_init=random_values[i, 0], sin2_13_init=random_values[i, 1],
                                             dm2_21_init=random_values[i, 2], dm2_31_init=random_values[i, 3],
                                             ndays=ndays,
                                             core_baselines=baselines,
                                             core_powers=powers,
                                             ebins=ebins,
                                             args=args)

      return sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2

  results = Parallel(n_jobs=-1)(delayed(run_experiment)(i) for i in range(num_experiments))

  # Unpack the results
  for result in results:
      sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2 = result
      sin2_12_arr.append(sin2_12)
      sin2_13_arr.append(sin2_13)
      dm2_21_arr.append(dm2_21)
      dm2_31_arr.append(dm2_31)
      delta_chi2_arr.append(delta_chi2)


  end_toy_time = datetime.now()
  print("Toy time {}".format(end_toy_time - start_toy_time))
  print("Total time {}".format(end_toy_time - start_sp_time))
  plt.hist(sin2_12_arr)
  plt.show()
  plt.hist(sin2_13_arr)
  plt.show()
  plt.hist(dm2_21_arr)
  plt.show()
  plt.hist(dm2_21_arr)
  plt.show()
  plt.hist(delta_chi2_arr)
  plt.show()


if __name__ == "__main__":
  main(sys.argv[1:])
