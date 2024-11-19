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
import h5py

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
  min_temp = args_juno.max_ene + ((args_juno.max_ene- args_juno.min_ene)/args_juno.bins-1) #for combined CM calculation
  max_temp = min_temp + (args_juno.max_ene- args_juno.min_ene)
  ebins_temp = np.linspace(min_temp,max_temp, args_juno.bins)
  #create matrix directory
  if not os.path.exists(f"{args_juno.data_matrix_folder}"):
      os.mkdir(f"{args_juno.data_matrix_folder}")

  if not os.path.exists(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}"):
      os.mkdir(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}")

  print(" # Run configuration:")
  print("   Statistics:   {} days".format(ndays) )
  print("   NMO opt NO: {}".format(args_juno.NMO_opt))

  #   Energy resolution
  a, b, c = args_juno.a, args_juno.b, args_juno.c
  a_err, b_err, c_err =args_juno.a_err, args_juno.b_err, args_juno.c_err

  if os.path.isfile(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_RM:
    resp_matrix = LoadRespMatrix(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  else:
    resp_matrix = CalcRespMatrix_abc(a, b, c, escale=1, ebins=ebins, pebins=ebins)
    resp_matrix.Save(f"{args_juno.data_matrix_folder}/rm_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

  if args_juno.FORCE_CALC_RM:
    resp_matrix.Dump(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}/resp_matrix.csv")

  if os.path.isfile(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_tao.FORCE_CALC_ENE_LEAK:
    ene_leak_tao = LoadRespMatrix(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
  else:
    ene_leak_tao = CalcEnergyLeak(rootfile=args_juno.input_data_file, histname="TAO_response_matrix_25", ebins=ebins, pebins=ebins)
    ene_leak_tao.Save(f"{args_juno.data_matrix_folder}/energy_leak_tao_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

  if args_tao.FORCE_CALC_ENE_LEAK:
    ene_leak_tao.Dump(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}/ene_leak_tao_matrix.csv")
  #Create reactor spectra and get backgrounds spectra, function inside spectra.py
  ensp_nom_juno  = spec.CreateSpectra(ndays=ndays,
                                    ebins=ebins,
                                    detector="juno",
                                    resp_matrix=resp_matrix,
                                    args=args_juno)
  # ensp_nom_juno['rfis'].WritetoROOT("Enu_noRC", "Sindhu_spectra_Oct25.root")
  # ensp_nom_juno['ribd'].WritetoROOT("Enu_wRC", "Sindhu_spectra_Oct25.root")
  #ensp_nom_juno['rdet'].WritetoROOT("rosc_newpos_DYB", "Sindhu_spectra_Oct28.root")
  # ensp_nom_juno['geo'].WritetoROOT("geo", "Sindhu_Nov11.root")
  # ensp_nom_juno['geou'].WritetoROOT("geoU", "Sindhu_Nov11.root")
  # ensp_nom_juno['geoth'].WritetoROOT("geoTh", "Sindhu_Nov11.root")
  #ensp_nom_juno['rdet'].WritetoROOT("rea_osc_noFT2", "Sindhu_Nov11.root")
  # ensp_nom_juno['ribd'].WritetoROOT("rea_unosc_noRC_noME", "Sindhu_Nov11.root")
  if args_juno.fit_type == 'NMO' and args_juno.include_TAO:
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

  unc_max_juno = max(args_juno.unc_list_juno, key=len)
  if '+' in unc_max_juno:
      unc_list_juno = unc_max_juno.split('+')
  else:
      unc_list_juno = [unc_max_juno]

  cm_juno = {}

  for u in unc_list_juno:
      if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_juno_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
          print(" # Loading covariance matriix", f"{args_juno.data_matrix_folder}/cm_juno_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          cm_juno[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_juno_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")

      else:
          print(f" # Constructing covariance matrix {args_juno.data_matrix_folder}/cm_juno_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          cm_juno[u] = mat.GetCM(ensp = ensp_nom_juno,
            resp_matrix=resp_matrix,
            ndays=ndays,
            unc=u,
            detector="juno",
            args=args_juno)

      cm_juno[u].Dump(f"{args_juno.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
      if args_juno.PLOT_CM: cm_juno[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_juno_{u}.png")
      del cm_juno[u]

  if args_juno.fit_type == 'NMO'and args_juno.include_TAO:
      unc_max_tao = max(args_tao.unc_list_tao, key=len)
      if '+' in unc_max_tao:
          unc_list_tao = unc_max_tao.split('+')
      else:
          unc_list_tao = [unc_max_tao]

      cm_tao = {}

      for u in unc_list_tao:
          if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_tao_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
              print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_tao_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              cm_tao[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_tao_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          else:
              print(f" # Constructing covariance matrix {args_juno.data_matrix_folder}/cm_tao_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              cm_tao[u] = mat.GetCM(ensp = ensp_nom_tao,
                 resp_matrix=resp_matrix,
                 ndays=ndays,
                 unc=u,
                 ene_leak_tao=ene_leak_tao,
                 detector="tao",
                 args=args_tao)
          cm_tao[u].Dump(f"{args_tao.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_juno_{u}.csv")
          if args_juno.PLOT_CM: cm_tao[u].Plot(f"{args_juno.cov_matrix_plots_folder}/cm_tao_{u}.png")
          del cm_tao[u]

      if args_juno.unc_corr_dep:
          cm_corr_dep = {}
          if '+' in args_juno.unc_corr_dep:
              unc_list_corr_dep = args_juno.unc_corr_dep.split('+')
          else:
              unc_list_corr_dep = [args_juno.unc_corr_dep]

          for u in unc_list_corr_dep:
              if os.path.isfile(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat") and not args_juno.FORCE_CALC_CM:
                  print(" # Loading covariance matrix", f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
                  cm_corr_dep[u] = LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              else:
                  print(f" # Constructing covariance matrices {args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
                  cm_corr_dep[u] = mat.GetCorrCM(ensp_juno = ensp_nom_juno,
                            ensp_tao = ensp_nom_tao,
                            resp_matrix=resp_matrix,
                            ndays=ndays,
                            unc=u,
                            ene_leak_tao=ene_leak_tao,
                            args_juno=args_juno,
                            args_tao=args_tao)
              cm_corr_dep[u].Dump(f"{args_tao.data_matrix_folder}/csv_{args_juno.stat_opt}/cov_mat_corr_{u}.csv")
              if args_juno.PLOT_CM:
                  cm_temp = CovMatrix(data=cm_corr_dep[u].data, bins=ebins+ebins_temp)
                  cm_temp.Plot(f"{args_juno.cov_matrix_plots_folder}/cm_corr_{u}.png")
              del cm_corr_dep[u]

      cm_corr = {}

      if args_juno.unc_corr_ind:
          unc_list_corr_ind = args_juno.unc_corr_ind.split('+')
          for new_unc in unc_list_corr_ind:
              cm_corr[new_unc] = LoadObject(f"{args_juno.data_matrix_folder}/cm_juno_{new_unc}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")\
                                .Extend(LoadObject(f"{args_juno.data_matrix_folder}/cm_tao_{new_unc}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat"))
              #cm_juno[new_unc].Extend(cm_tao[new_unc])


  end_cm_time = datetime.now()
  print("Covariance matrices production: ", end_cm_time - start_cm_time)
  start_scan_time = datetime.now()

  unc_list_new_juno = []
  unc_list_new_tao = []
  #TODO: This is mainly for CNP when stat matrix is included by default, can be done better
  for unc in args_juno.unc_list_juno:
    unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
    unc_list_new_juno.append(unc.replace(args_juno.unc_corr_ind+'+', ''))

  for full_unc in unc_list_new_juno:
    single_unc_list = full_unc.split("+")
    cm_juno[full_unc] = CovMatrix(data=np.zeros((args_juno.bins-1, args_juno.bins-1)), bins=ebins)
    for u in single_unc_list:
      cm_juno[full_unc] += LoadObject(f"{args_juno.data_matrix_folder}/cm_juno_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
      #cm_juno[u]


  if args_juno.fit_type == 'NMO' and args_juno.include_TAO:
      for unc in args_tao.unc_list_tao:
          unc = unc.replace('stat+', "") #stat is always directly calculated inside chi2 function
          if args_juno.fit_type == 'NMO':
              unc_list_new_tao.append(unc.replace(args_juno.unc_corr_ind+'+', ''))
          else:
              unc_list_new_tao.append(unc)
      for full_unc in unc_list_new_tao:
          single_unc_list = full_unc.split("+")
          cm_tao[full_unc] = CovMatrix(data=np.zeros((args_juno.bins-1, args_juno.bins-1)), bins=ebins)
          for u in single_unc_list:
              cm_tao[full_unc] += LoadObject(f"{args_juno.data_matrix_folder}/cm_tao_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
              #cm_tao[u]

      for key in unc_list_new_tao:
          if not key in cm_tao.keys():
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
               cm_corr[args_juno.unc_corr_ind+args_juno.unc_corr_dep] += LoadObject(f"{args_juno.data_matrix_folder}/cm_correlated_{u}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
               #cm_corr_dep[u]
  np.random.seed(42)
  def generate_toy_spectrum(spectrum, cov_matrix):
      L = np.linalg.cholesky(cov_matrix.data)
      y = np.random.normal(0, 1, len(spectrum.bin_cont))
      S_fluc = np.array(spectrum.bin_cont) + L @ y
      S_fluc = np.maximum(S_fluc, 0)
      S_fluc_poisson = np.random.poisson(S_fluc)
      return S_fluc_poisson

  def run_toy(i):
      if i%100 == 0: print(f"Toys: {i}/{args_juno.ntoys}")
      ensp_nom_juno['toy'] = Spectrum(bins=ensp_nom_juno['rdet'].bins, bin_cont=generate_toy_spectrum(ensp_nom_juno['geo'] + ensp_nom_juno['rdet'], cm_juno[unc_list_new_juno[0]]))
      nan_mask = np.isnan(ensp_nom_juno['toy'].bin_cont)
      if len(ensp_nom_juno['toy'].bin_cont[nan_mask] !=0): print("WARNING: NaN values found!")
      ensp_nom_juno['rtot_toy'] = ensp_nom_juno['rtot'] - ensp_nom_juno['geo'] - ensp_nom_juno['rdet'] + ensp_nom_juno['toy']
      try:
          results = minuit.run_minuit(ensp_nom_juno=ensp_nom_juno, unc_juno=unc_list_new_juno[0], rm=resp_matrix, cm_juno=cm_juno, args_juno=args_juno)
          return results
      except Exception as e:
          print(f"WARNING: Minuit failed")
          return None 

  def save_batch_results(filename, batch_results):
    filtered_results = [row for row in batch_results if all(value is not None for value in row)]
    if not filtered_results:
        print("WARNING: No valid data to save. Skipping...")
        return
    new_data = np.array(filtered_results, dtype='S64')
    dataset_name ='geo'
    with h5py.File(filename, "a") as hdf:
        if dataset_name in hdf:
            dset = hdf[dataset_name]
            dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
            dset[-new_data.shape[0]:] = new_data
        else:
            dset = hdf.create_dataset(
                dataset_name,
                data=new_data,
                maxshape=(None, new_data.shape[1]),  # Unlimited rows, fixed columns
                compression="gzip",
            )

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
          #Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom_juno=ensp_nom_juno, ensp_nom_tao=ensp_nom_tao, unc=unc, rm=resp_matrix, ene_leak_tao=ene_leak_tao, cm_juno=cm_juno, cm_tao=cm_tao, args_juno=args_juno, args_tao=args_tao) for unc in unc_list_new_juno)
          if args_juno.fit_type == 'NMO' and args_juno.include_TAO:
              minuit.run_minuit(ensp_nom_juno=ensp_nom_juno, ensp_nom_tao=ensp_nom_tao, unc_juno=unc_list_new_juno[0].replace(args_juno.unc_corr_ind+'+', ''), unc_tao=unc_list_new_tao[0].replace(args_juno.unc_corr_ind+'+', ''), unc_corr=unc_corr, rm=resp_matrix, ene_leak_tao=ene_leak_tao, cm_juno=cm_juno, cm_tao=cm_tao,cm_corr=cm_corr, args_juno=args_juno, args_tao=args_tao)
          else:
              if args_juno.toymc:
                  for batch_start in range(0, args_juno.ntoys, args_juno.toy_batch_size):
                      print(f"Batch {int(batch_start/args_juno.toy_batch_size)}")
                      batch_start_t = datetime.now()
                      batch_end = min(batch_start + args_juno.toy_batch_size, args_juno.ntoys)
                      batch_results = Parallel(n_jobs=-1, verbose=10)(delayed(run_toy)(i=t) for t in range(batch_start, batch_end))
                      filename = f"{args_juno.main_data_folder}/fit_results_{args_juno.fit_type}_{args_juno.stat_method_opt}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins_minuit.hdf5"
                      save_batch_results(filename, batch_results)
                      del batch_results
              else:
                  for unc in unc_list_new_juno: minuit.run_minuit(ensp_nom_juno=ensp_nom_juno, unc_juno=unc, rm=resp_matrix, cm_juno=cm_juno, args_juno=args_juno)
         # dm2_31_val = 2.5283e-3
         # dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 100 )
          #Parallel(n_jobs =-1)(delayed(minuit.run_minuit)(ensp_nom=ensp_nom_juno, unc=unc_list_new[0], baselines=baselines, powers=powers, rm=resp_matrix, cm=cm, args=args_juno, dm2_31=m31) for m31 in dm2_31_list)

  end_scan_time = datetime.now()
  print("Scanning time", end_scan_time-start_scan_time)

if __name__ == "__main__":
  main(sys.argv[1:])
