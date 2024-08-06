import sys
import numpy as np
import os
import gc
from datetime import datetime
import argparse
import yaml
import toy_spectra as spec
import scan as scan
from noda import *
import bayesian as bayes
import bayesian_results as bayes_res
import frequentist_results as freq_res
import toy_minuit as minuit
import toy_plot_results as plot_res
import itertools

def toy_mc(sin2_12_init=0., sin2_13_init=0., dm2_21_init=0., dm2_31_init=0., ndays=0,
                                       core_baselines=[],
                                       core_powers=[],
                                       ebins=None,
                                       args=''):
     return sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2


def main(argv):
  if (len(sys.argv) <  2):
      print("ERROR: Please give the config file using the option --config_file=<filename>")

  start_time = datetime.now()

  #Create parser for config file
  parser = argparse.ArgumentParser()
#  parser.add_argument('--config', help='Path to the YAML configuration file')
#  args = parser.parse_args()

  #create parser for yaml file
  with open("toy_fit_configuration_inputs.yaml", "r") as file:
    config = yaml.safe_load(file)

     # Parse arguments from the YAML content
    for option, value in config["toy"].items():
      parser.add_argument(f"--{option}", default=value)
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

  nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt)

  num_experiments = args.nexp

  
  #Create reactor spectra and get backgrounds spectra, function inside spectra.py
  start_sp_time = datetime.now()
  ensp_nom, rm, opt = spec.Initialize(ndays=ndays,
                                    core_baselines=baselines,
                                    core_powers=powers,
                                    ebins=ebins,
                                    args=args)


  end_sp_time = datetime.now()
  print("Spectra production time: ", end_sp_time - start_sp_time)
  #Create covariance matrices and energy response matrix, function inside matrices.py
  
  cm ={}


  #ensp_nom['rtot'].Plot(f"rtot.png",
  #                 xlabel="Reconstructed energy (MeV)",
  #                 xmin=0, xmax=10, log_scale=False)
  #cm['stat'] = ensp_nom['rtot'].GetStatCovMatrix()
  def generate_poisson_fluctuated_spectrum(original_spectrum):
    l_poisson = []
    for i in range(len(original_spectrum.bin_cont)):
        fluc_arr = np.random.poisson(original_spectrum.bin_cont[i])#, 10).mean()
        #if(fluc_arr == 0): fluc_arr = min(original_spectrum.bin_cont)
        l_poisson.append(fluc_arr)
        del fluc_arr
    #erged = list(itertools.chain.from_iterable(l_poisson))
    modified_bin_cont = np.array(l_poisson)
    new_spectrum = Spectrum(bin_cont=modified_bin_cont, bins=original_spectrum.bins,
                            xlabel=original_spectrum.xlabel, ylabel=original_spectrum.ylabel)
    return new_spectrum

  dm2_31_val = 2.5283e-3
  dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 100 )
  def run_experiment(i, m2_31):
      print("Toy experiment {}".format(i))
      ensp = {}
      #ensp['rosc'] = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,dm2_31=m2_31, me_rho=args.me_rho, ene_mode='true', args=args)
      #ensp['rvis_nonl'] = ensp['rosc'].GetWithPositronEnergy()
      #ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])

      #ensp['rdet'] = ensp['rvis'].ApplyDetResp(rm, pecrop=args.ene_crop)
      #ensp['rtot'] = ensp['rdet'] + ensp_nom['acc'] + ensp_nom['fneu'] + ensp_nom['lihe'] + ensp_nom['aneu'] + ensp_nom['geo'] + ensp_nom['atm'] + ensp_nom['rea300']

      new_spectrum = generate_poisson_fluctuated_spectrum(ensp_nom['rtot'])
      ensp_nom['rdet_toy'] = new_spectrum
      #ensp_nom['rdet_toy'].Plot(f"toy_{i}_rtot.png",
      #             xlabel="Reconstructed energy (MeV)",
      #             xmin=0, xmax=10, log_scale=False)

      cm['stat'] = ensp_nom['rdet_toy'].GetStatCovMatrix()
      minuit.run_minuit(ensp_nom=ensp_nom, baselines=baselines, powers=powers, rm=rm, cm=cm, args=args, i=i, m2_31=m2_31)   
      del cm['stat']
      #return sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2

  #results = 
  Parallel(n_jobs=-1)(delayed(run_experiment)(i, m31) for i in range(num_experiments) for m31 in dm2_31_list)
  #Parallel(n_jobs=-1)(delayed(run_experiment)(i) for i in range(num_experiments))
  #
 # sin2_12_arr, sin2_13_arr, dm2_21_arr, dm2_31_arr, delta_chi2_arr = [], [], [], [], []

##   Unpack the results
 # for result in results:
 #       sin2_12, sin2_13, dm2_21, dm2_31, delta_chi2 = result
 #       if(delta_chi2==-111): continue
 #       sin2_12_arr.append(sin2_12)
 #       sin2_13_arr.append(sin2_13)
 #       dm2_21_arr.append(dm2_21)
 #       dm2_31_arr.append(dm2_31)
 #       delta_chi2_arr.append(delta_chi2)

 # 
 # 
 # np.savez(f"toy_results_{args.stat_opt}_1k.npz",
 #        sin2_12_arr=sin2_12_arr,
 #        sin2_13_arr=sin2_13_arr,
 #        dm2_21_arr=dm2_21_arr,
 #        dm2_31_arr=dm2_31_arr,
 #        delta_chi2_arr=delta_chi2_arr)

 # plot_res.plot(f"toy_results_{args.stat_opt}_1k.npz", args=args)
  end_time = datetime.now()
  print("Total time: ", end_time - start_time)


if __name__ == "__main__":
  main(sys.argv[1:])
