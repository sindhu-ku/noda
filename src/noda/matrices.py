#!/usr/bin/env python3
import sys, os
from .noda import *
import csv
from datetime import datetime

def GetCM(ensp = {},
      core_baselines=[],
      core_powers=[],
      resp_matrix=[],
      ndays=10,
      args=''):

 #take the longest string in the unc list to calculate CMs
  unc_max = max(args.unc_list, key=len)
  if '+' in unc_max:
      unc = unc_max.split('+')
  else:
      unc = [unc_max]
  # Define a dictionary mapping the strings to their corresponding function calls
  unc_map = {
    'stat': lambda: ensp['rtot'].GetStatCovMatrix(),
    'r2': lambda: ensp['rdet'].GetRateCovMatrix(args.r2_unc),
    'eff': lambda: ensp['rdet'].GetRateCovMatrix(args.eff_unc),
    'b2b_DYB': lambda: ensp['rdet'].GetVariedB2BCovMatrixFromROOT(args.input_data_file, "DYBUncertainty"),
    'b2b_TAO': lambda: ensp['rdet'].GetVariedB2BCovMatrixFromROOT(args.input_data_file, "TAOUncertainty"),
    'snf': lambda: ensp['snf_final'].GetRateCovMatrix(args.snf_unc),
    'noneq': lambda: ensp['noneq_final'].GetRateCovMatrix(args.noneq_unc),
    'me': lambda: get_ME_CM(),
    'nl': lambda: get_NL_CM(),
    'abc': lambda: get_abc_CM(),
    'crel': lambda: get_core_flux_CM(),
    'bg': lambda: get_bckg_CM()
  }


  def mat_flu(me_rho_flu):
      ensp['rosc_me_flu'] = ensp['ribd'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=me_rho_flu, ene_mode='true', args=args)
      ensp['rvis_me_flu_0'] = ensp['rosc_me_flu'].GetWithPositronEnergy()
      ensp['rvis_me_flu'] = ensp['rvis_me_flu_0'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      del ensp['rosc_me_flu'], ensp['rvis_me_flu_0']
      return ensp['rvis_me_flu']

  def get_ME_CM():
      print ("Matter effect fluctuated spectra")
      time_start_me = datetime.now()
      me_rho_flu = np.random.normal(loc=args.me_rho, scale=args.me_rho_scale, size=args.sample_size_me)
      ensp['rvis_me_flu'] = Parallel(n_jobs=-1)(delayed(mat_flu)(val) for val in me_rho_flu)
      ensp['rdet_me_flu'] = [s.ApplyDetResp(resp_matrix, pecrop=args.ene_crop) for s in ensp['rvis_me_flu']]
      del ensp['rvis_me_flu']
      #ensp['rdet'].Plot(f"{args.plots_folder}/rdet_me_flu.pdf", extra_spectra=ensp['rdet_me_flu'], ylabel="Events per bin")
      time_end_me = datetime.now()
      print ("ME flu time", time_end_me - time_start_me)
      return ensp['rdet'].GetCovMatrixFromRandSample(ensp['rdet_me_flu'])

  def new_NL_curve(pull_num, w):
      new_nonl =  Spectrum(bins = ensp['scintNL'].bins, bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
      for i in range(len(new_nonl.bins)-1):
        new_nonl.bin_cont[i] = ensp['scintNL'].bin_cont[i] + w*(ensp['NL_pull'][pull_num].bin_cont[i] - ensp['scintNL'].bin_cont[i])
      output = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
      del new_nonl
      return output

  def GetFluNL(pull_num):
      weights = np.random.normal(loc=0., scale=1., size=args.sample_size_nonl)
      # output_spectra = [*map(lambda w: new_NL_curve(ensp_nonl, ensp_nl_nom, ensp_nl_pull_curve, w), weights)]
      output_spectra = Parallel(n_jobs=-1)(delayed(new_NL_curve)(pull_num, w) for w in weights)
      return output_spectra

  def get_NL_CM():
      print ("NL fluctuated spectra")
      start_time_nl = datetime.now()

      for i in range(4):
          print("   NL pull curve {}".format(i))
          print("     getting rvis spectra")
          ensp['rvis_nl_flu'+f'_{i}'] = GetFluNL(i)
          print("     getting rdet spectra")
          ensp['rdet_nl_flu'+f'_{i}'] = [s.ApplyDetResp(resp_matrix, pecrop=args.ene_crop) for s in ensp['rvis_nl_flu'+f'_{i}']]
          del ensp['rvis_nl_flu'+f'_{i}']
          print("     constructing cov. matrix")
          cm['nl'+f'_{i}'] = ensp['rdet'].GetCovMatrixFromRandSample(ensp['rdet_nl_flu'+f'_{i}'])
      end_time_nl = datetime.now()
      del ensp['rvis_nonl'], ensp['NL_pull']
      for i in range(4):
          del ensp['rdet_nl_flu'+f'_{i}']
      print ("NL flu time", end_time_nl - start_time_nl)
      print("   Summing nl matrices")
      return cm['nl_0']+cm['nl_1']+cm['nl_2']+cm['nl_3']
  #ensp['rdet'].Plot(f"{args.plots_folder}/det_nl_flu.png",
  #             xlabel="Reconstructed energy (MeV)",
  #             ylabel=f"Events",
  #             extra_spectra=ensp['rdet_nl_flu_0']+ensp['rdet_nl_flu_1']+ensp['rdet_nl_flu_2']+ensp['rdet_nl_flu_3'],
  #             xmin=0, xmax=10,
  #             ymin=0, ymax=None, log_scale=False)



  def CalcRespMatrix_abc_flu(ebins, pebins, escale=1., eshift=0., norm_mode='per_MeV'):
      print(" # Fluctuating (a,b,c) parameters...")
      a, b, c = args.a, args.b, args.c
      a_err, b_err, c_err =args.a_err, args.b_err, args.c_err
      a_flu = np.random.normal(loc=a, scale=a_err, size=args.sample_size_resp)
      b_flu = np.random.normal(loc=b, scale=b_err, size=args.sample_size_resp)
      c_flu = np.random.normal(loc=c, scale=c_err, size=args.sample_size_resp)

      CalcRM = lambda a,b,c: CalcRespMatrix_abc( a, b, c, ebins, pebins, escale=escale,  eshift=eshift, norm_mode=norm_mode, verbose=False)
      spectra = Parallel(n_jobs=-1)(delayed(CalcRM)(a_flu[i], b_flu[i], c_flu[i]) for i in range(len(a_flu)))
      return spectra

  def get_abc_CM():
      ebins = ensp['ribd'].bins
      print ("Response matrix fluctuated spectra")
      start_time_resp = datetime.now()
      resp_mat_flu = CalcRespMatrix_abc_flu(escale=1, ebins=ebins, pebins=ebins)
      ensp['rdet_abc_flu'] = [*map(lambda x :  ensp['rvis'].ApplyDetResp(x, pecrop=args.ene_crop), resp_mat_flu)]
      del resp_mat_flu
      end_time_resp = datetime.now()
      del ensp['rvis']
      print("RM flu time", end_time_resp - start_time_resp)
      return ensp['rdet'].GetCovMatrixFromRandSample(ensp['rdet_abc_flu'])


  def get_core_flu(i):
      if i%1000 == 0: print (f"{i}/{args.sample_size_core}")
      deviations = np.random.normal(loc=1., scale=args.core_flux_unc, size=len(core_powers))
      flu_powers = [dev*p for dev, p in zip(deviations, core_powers)]
      flu_powers2 = np.array([dev*p for dev, p in zip(deviations, core_powers)])
      flu_powers22 = flu_powers2*6.24e21*60*60*24 # MeV/day
      alpha_arr = np.array(args.alpha)
      efission_arr = np.array(args.efission)
      Pth_arr = np.array(args.Pth)
      L_arr = np.array(args.L)
      extrafactors = args.detector_efficiency*args.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr/(L_arr*L_arr))
      extrafactors2 = args.detector_efficiency*args.Np/(4*np.pi)*1./np.sum(alpha_arr*efission_arr)*np.sum(flu_powers22/(L_arr*L_arr))
      ensp['ribd_crel'] = ensp['ribd'].Copy()
      ensp['ribd_crel'].GetScaled(1./extrafactors)
      ensp['ribd_crel'].GetScaled(extrafactors2)
      ensp['rosc_crel_flu'] = ensp['ribd_crel'].GetOscillated(L=core_baselines, core_powers=flu_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      del ensp['ribd_crel']
      ensp['rvis_crel_flu_nonl'] = ensp['rosc_crel_flu'].GetWithPositronEnergy()
      ensp['rvis_crel_flu'] = ensp['rvis_crel_flu_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      del ensp['rvis_crel_flu_nonl']
      return ensp['rvis_crel_flu']

  def get_core_flux_CM():
      print(" # Fluctuating relative core fluxes ")
      start_time_core = datetime.now()
      ensp['rvis_crel_flu'] = Parallel(n_jobs=-1)(delayed(get_core_flu)(i) for i in range(args.sample_size_core))
      ensp["rdet_crel_flu"] = [s.ApplyDetResp(resp_matrix, pecrop=args.ene_crop) for s in ensp['rvis_crel_flu']]
      del ensp['rvis_crel_flu']
      end_time_core = datetime.now()
      print("Core flu time", end_time_core - start_time_core)
      return ensp["rdet"].GetCovMatrixFromRandSample(ensp["rdet_crel_flu"])

  def get_bckg_CM():
    print("Background CM")
    cm['acc'] = ensp['acc'].GetRateCovMatrix(args.acc_rate_unc) #+ ensp['acc'].GetStatCovMatrix()
    cm['geo'] = ensp['geo'].GetRateCovMatrix(args.geo_rate_unc) + ensp['geo'].GetB2BCovMatrix(args.geo_b2b_unc) #+ ensp['geo'].GetStatCovMatrix()
    cm['lihe'] = ensp['lihe'].GetRateCovMatrix(args.lihe_rate_unc) + ensp['lihe'].GetB2BCovMatrix(args.lihe_b2b_unc) #+ ensp['lihe'].GetStatCovMatrix()
    cm['fneu'] = ensp['fneu'].GetRateCovMatrix(args.fneu_rate_unc) + ensp['fneu'].GetB2BCovMatrix(args.fneu_b2b_unc) #+ ensp['fneu'].GetStatCovMatrix()
    cm['aneu'] = ensp['aneu'].GetRateCovMatrix(args.aneu_rate_unc) + ensp['aneu'].GetB2BCovMatrix(args.aneu_b2b_unc) #+ ensp['aneu'].GetStatCovMatrix()
    cm['atm'] = ensp['atm'].GetRateCovMatrix(args.atm_rate_unc) + ensp['atm'].GetB2BCovMatrix(args.atm_b2b_unc) #+ ensp['atm'].GetStatCovMatrix()
    cm['rea300'] = ensp['rea300'].GetRateCovMatrix(args.rea300_rate_unc) + ensp['rea300'].GetB2BCovMatrix(args.rea300_b2b_unc) #+ ensp['rea300'].GetStatCovMatrix()

    del ensp['rtot_noenecrop'], ensp['rdet_noenecrop'], ensp['acc_noenecrop'], ensp['fneu_noenecrop'], ensp['lihe_noenecrop'], ensp['aneu_noenecrop'], ensp['geo_noenecrop'], ensp['geou_noenecrop'], ensp['geoth_noenecrop']
    return cm['acc'] + cm['geo'] + cm['lihe'] + cm['fneu'] + cm['aneu'] + cm['atm'] +cm['rea300']


 # Initialize the cm dictionary
  cm = {}

# Iterate over the unc list and use the map to call the appropriate function
  for u in unc:
      if u in unc_map:
          cm[u] = unc_map[u]()
      else:
          raise ValueError(f"unc {u} not found! Cannot calculate!")

  SaveObject(cm, f"{args.data_matrix_folder}/cm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
  for key in cm.keys():
      cm[key].Dump(f"{args.data_matrix_folder}/csv/cov_mat_{key}.csv")
  return cm
