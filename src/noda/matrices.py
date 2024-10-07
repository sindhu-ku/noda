#!/usr/bin/env python3
import sys, os
from .noda import *
from datetime import datetime

def GetCM(ensp = {},
      resp_matrix=[],
      ndays=10,
      unc='',
      detector="juno",
      ene_leak_tao=[],
      args=None):

 # #take the longest string in the unc list to calculate CMs
 #  unc_max = max(unc_list, key=len)
 #  if '+' in unc_max:
 #      unc = unc_max.split('+')
 #  else:
 #      unc = [unc_max]
  # Define a dictionary mapping the strings to their corresponding function calls
  unc_map = {
    'stat': lambda: ensp['rdet'].GetStatCovMatrix(),
    'r2': lambda: ensp['rdet'].GetRateCovMatrix(args.r2_unc),
    'eff': lambda: ensp['rdet'].GetRateCovMatrix(args.eff_unc),
    'b2b_DYB': lambda: ensp['rdet'].GetVariedB2BCovMatrixFromROOT(args.input_data_file, "DYBUncertainty"),
    'b2b_TAO': lambda: ensp['rdet'].GetVariedB2BCovMatrixFromROOT(args.input_data_file, "TAOUncertainty"),
    'snf': lambda: ensp['snf_final'].GetRateCovMatrix(args.snf_unc),
    'noneq': lambda: ensp['noneq_final'].GetRateCovMatrix(args.noneq_unc),
    'ene_scale': lambda: ensp['rdet'].GetRateCovMatrix(args.ene_scale_unc),
    'me': lambda: get_ME_CM(),
    'nl': lambda: get_NL_CM(),
    'abc': lambda: get_abc_CM(),
    'crel': lambda: get_core_flux_CM(),
    'bg': lambda: get_bckg_CM()
  }


  def mat_flu(me_rho_flu):
      ensp['rosc_me_flu'] = ensp['ribd'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=me_rho_flu, ene_mode='true', args=args)
      ensp['rvis_me_flu_0'] = ensp['rosc_me_flu'].GetWithPositronEnergy()
      if detector == "tao": ensp['rvis_me_flu_0'] = ensp['rvis_me_flu_0'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop)
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
      cm = ensp['rdet'].GetCovMatrixFromRandSample(ensp['rdet_me_flu'])
      del ensp['rdet_me_flu']
      return cm

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
          del ensp['rdet_nl_flu'+f'_{i}']
      end_time_nl = datetime.now()
      del ensp['rvis_nonl'], ensp['NL_pull']
      print ("NL flu time", end_time_nl - start_time_nl)
      print("   Summing nl matrices")
      cm_temp = cm['nl_0']+cm['nl_1']+cm['nl_2']+cm['nl_3']
      del cm['nl_0'], cm['nl_1'], cm['nl_2'], cm['nl_3']
      return cm_temp
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
      cm_temp = ensp['rdet'].GetCovMatrixFromRandSample(ensp['rdet_abc_flu'])
      del ensp['rdet_abc_flu']
      return cm_temp


  def get_core_flu(i):
      if i%1000 == 0: print (f"{i}/{args.sample_size_core}")
      deviations = np.random.normal(loc=1., scale=args.core_flux_unc, size=len(args.core_powers))
      flu_powers = [dev*p for dev, p in zip(deviations, args.core_powers)]
      flu_powers2 = np.array([dev*p for dev, p in zip(deviations, args.core_powers)])
      flu_powers22 = flu_powers2*6.24e21*60*60*24 # MeV/day
      alpha_arr = np.array(args.alpha)
      efission_arr = np.array(args.efission)
      Pth_arr = np.array(args.Pth)
      L_arr = np.array(args.L)
      extrafactors = args.detector_efficiency*args.veto*args.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr/(L_arr*L_arr))
      extrafactors2 = args.detector_efficiency*args.veto*args.Np/(4*np.pi)*1./np.sum(alpha_arr*efission_arr)*np.sum(flu_powers22/(L_arr*L_arr))
      ensp['ribd_crel'] = ensp['ribd'].Copy()
      ensp['ribd_crel'].GetScaled(1./extrafactors)
      ensp['ribd_crel'].GetScaled(extrafactors2)
      ensp['rosc_crel_flu'] = ensp['ribd_crel'].GetOscillated(L=args.core_baselines, core_powers=flu_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      del ensp['ribd_crel']
      ensp['rvis_crel_flu_nonl'] = ensp['rosc_crel_flu'].GetWithPositronEnergy()
      if detector == "tao": ensp['rvis_crel_flu_nonl'] = ensp['rvis_crel_flu_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop)
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
      cm_temp  = ensp["rdet"].GetCovMatrixFromRandSample(ensp["rdet_crel_flu"])
      del ensp['rdet_crel_flu']
      return cm_temp

  def get_bckg_CM():
    cm['acc'] = ensp['acc'].GetRateCovMatrix(args.acc_rate_unc) + ensp['acc'].GetStatCovMatrix()
    cm['lihe'] = ensp['lihe'].GetRateCovMatrix(args.lihe_rate_unc) + ensp['lihe'].GetB2BCovMatrix(args.lihe_b2b_unc) + ensp['lihe'].GetStatCovMatrix()
    cm['fneu'] = ensp['fneu'].GetB2BCovMatrix(args.fneu_b2b_unc) + ensp['fneu'].GetStatCovMatrix()
    if detector != "tao":
        cm['fneu'] += ensp['fneu'].GetRateCovMatrix(args.fneu_rate_unc)
        if not args.geo_fit: cm['geo'] = ensp['geo'].GetRateCovMatrix(args.geo_rate_unc) + ensp['geo'].GetB2BCovMatrix(args.geo_b2b_unc) + ensp['geo'].GetStatCovMatrix()
        else: cm['geo'] = ensp['geo'].GetB2BCovMatrix(args.geo_b2b_unc) + ensp['geo'].GetStatCovMatrix()
        cm['aneu'] = ensp['aneu'].GetRateCovMatrix(args.aneu_rate_unc) + ensp['aneu'].GetB2BCovMatrix(args.aneu_b2b_unc) + ensp['aneu'].GetStatCovMatrix()
        cm['atm'] = ensp['atm'].GetRateCovMatrix(args.atm_rate_unc) + ensp['atm'].GetB2BCovMatrix(args.atm_b2b_unc) + ensp['atm'].GetStatCovMatrix()
        cm['rea300'] = ensp['rea300'].GetRateCovMatrix(args.rea300_rate_unc) + ensp['rea300'].GetB2BCovMatrix(args.rea300_b2b_unc) + ensp['rea300'].GetStatCovMatrix()

    if detector != "tao": return cm['acc'] + cm['geo'] + cm['lihe'] + cm['fneu'] + cm['aneu'] + cm['atm'] +cm['rea300']
    else: return cm['acc'] + cm['lihe'] + cm['fneu']


 # # Initialize the cm dictionary
  cm = {}

# Iterate over the unc list and use the map to call the appropriate function
  #for u in unc:
  if unc in unc_map:
      cm[unc] = unc_map[unc]()
      SaveObject(cm[unc], f"{args.data_matrix_folder}/cm_{detector}_{unc}_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
      return cm[unc]
  else:
      raise ValueError(f"unc {unc} not found! Cannot calculate!")

def GetCorrCM(ensp_juno = {},
      ensp_tao = {},
      resp_matrix=[],
      ndays=10,
      unc='',
      ene_leak_tao=[],
      args_juno=None,
      args_tao=None):
  cm_juno = {}
  cm_tao = {}
 #take the longest string in the unc list to calculate CMs
  # unc_max = max(unc_list, key=len)
  # if '+' in unc_max:
  #     unc = unc_max.split('+')
  # else:
  #     unc = [unc_max]
  # Define a dictionary mapping the strings to their corresponding function calls
  unc_map = {
    'nl': lambda: get_NL_CM(),
    'crel': lambda: get_core_flux_CM(),
  }

  def new_NL_curve(pull_num, w, ensp_juno, ensp_tao):
      new_nonl =  Spectrum(bins = ensp_juno['scintNL'].bins, bin_cont=np.zeros(len(ensp_juno['scintNL'].bin_cont)))
      new_nonl.bin_cont = ensp_juno['scintNL'].bin_cont + w*(ensp_juno['NL_pull'][pull_num].bin_cont - ensp_juno['scintNL'].bin_cont)
      output_juno = ensp_juno['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
      output_tao = ensp_tao['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=new_nonl)
      del new_nonl
      return output_juno, output_tao

  def GetFluNL(pull_num, ensp_juno, ensp_tao):
      weights = np.random.normal(loc=0., scale=1., size=args_juno.sample_size_nonl)
      # output_spectra = [*map(lambda w: new_NL_curve(ensp_nonl, ensp_nl_nom, ensp_nl_pull_curve, w), weights)]
      output = Parallel(n_jobs=-1)(delayed(new_NL_curve)(pull_num, w, ensp_juno, ensp_tao) for w in weights)
      output_spectra_juno, output_spectra_tao = zip(*output)
      return output_spectra_juno, output_spectra_tao

  def get_NL_CM():
      print ("NL fluctuated spectra")
      start_time_nl = datetime.now()

      for i in range(4):
          print("   NL pull curve {}".format(i))
          print("     getting rvis spectra")
          ensp_juno['rvis_nl_flu'+f'_{i}'], ensp_tao['rvis_nl_flu'+f'_{i}'] = GetFluNL(i, ensp_juno, ensp_tao)
          print("     getting rdet spectra")
          ensp_juno['rdet_nl_flu'+f'_{i}'] = [s.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop) for s in ensp_juno['rvis_nl_flu'+f'_{i}']]
          ensp_tao['rdet_nl_flu'+f'_{i}'] = [s.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop) for s in ensp_tao['rvis_nl_flu'+f'_{i}']]
          del ensp_juno['rvis_nl_flu'+f'_{i}']
          del ensp_tao['rvis_nl_flu'+f'_{i}']
          print("     constructing cov. matrix")
          cm_juno['nl'+f'_{i}'] = ensp_juno['rdet'].GetCovMatrixFromRandSample(ensp_juno['rdet_nl_flu'+f'_{i}'])
          cm_tao['nl'+f'_{i}'] = ensp_tao['rdet'].GetCovMatrixFromRandSample(ensp_tao['rdet_nl_flu'+f'_{i}'])
          SaveObject(cm_juno['nl'+f'_{i}'], f"{args_juno.data_matrix_folder}/cm_correlated_juno_nl_{i}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          SaveObject(cm_tao['nl'+f'_{i}'], f"{args_juno.data_matrix_folder}/cm_correlated_tao_nl_{i}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
          del ensp_juno['rdet_nl_flu'+f'_{i}']
          del ensp_tao['rdet_nl_flu'+f'_{i}']

      end_time_nl = datetime.now()
      print ("NL flu time", end_time_nl - start_time_nl)
      print("   Summing nl matrices")
      cm_juno['nl'] =  cm_juno['nl_0']+cm_juno['nl_1']+cm_juno['nl_2']+cm_juno['nl_3']

      cm_tao['nl'] =  cm_tao['nl_0']+cm_tao['nl_1']+cm_tao['nl_2']+cm_tao['nl_3']
      cm_comb = cm_juno['nl'].Extend(cm_tao['nl'])
      return cm_comb

  def get_core_flu(i):
      if i%1000 == 0: print (f"{i}/{args_juno.sample_size_core}")
      deviations_juno = np.random.normal(loc=1., scale=args_juno.core_flux_unc, size=len(args_juno.core_powers))
      deviations_tao = np.array([deviations_juno[6], deviations_juno[7]])

      flu_powers_juno = [dev*p for dev, p in zip(deviations_juno, args_juno.core_powers)]
      flu_powers_tao = [dev*p for dev, p in zip(deviations_tao, args_tao.core_powers)]

      flu_powers2_juno = np.array([dev*p for dev, p in zip(deviations_juno, args_juno.core_powers)])
      flu_powers22_juno = flu_powers2_juno*6.24e21*60*60*24 # MeV/day
      flu_powers2_tao = np.array([dev*p for dev, p in zip(deviations_tao, args_tao.core_powers)])
      flu_powers22_tao = flu_powers2_tao*6.24e21*60*60*24 # MeV/day

      alpha_arr = np.array(args_juno.alpha)
      efission_arr = np.array(args_juno.efission)

      Pth_arr_juno = np.array(args_juno.Pth)
      L_arr_juno = np.array(args_juno.L)

      Pth_arr_tao = np.array(args_tao.Pth)
      L_arr_tao = np.array(args_tao.L)

      extrafactors_juno = args_juno.detector_efficiency*args_juno.veto*args_juno.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr_juno/(L_arr_juno*L_arr_juno))
      extrafactors2_juno = args_juno.detector_efficiency*args_juno.veto*args_juno.Np/(4*np.pi)*1./np.sum(alpha_arr*efission_arr)*np.sum(flu_powers22_juno/(L_arr_juno*L_arr_juno))

      extrafactors_tao = args_tao.detector_efficiency*args_tao.veto*args_tao.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr_tao/(L_arr_tao*L_arr_tao))
      extrafactors2_tao = args_tao.detector_efficiency*args_tao.veto*args_tao.Np/(4*np.pi)*1./np.sum(alpha_arr*efission_arr)*np.sum(flu_powers22_tao/(L_arr_tao*L_arr_tao))

      ensp_juno['ribd_crel'] = ensp_juno['ribd'].Copy()
      ensp_juno['ribd_crel'].GetScaled(1./extrafactors_juno)
      ensp_juno['ribd_crel'].GetScaled(extrafactors2_juno)
      ensp_juno['rosc_crel_flu'] = ensp_juno['ribd_crel'].GetOscillated(L=args_juno.core_baselines, core_powers=flu_powers_juno, me_rho=args_juno.me_rho, ene_mode='true', args=args_juno)
      del ensp_juno['ribd_crel']
      ensp_juno['rvis_crel_flu_nonl'] = ensp_juno['rosc_crel_flu'].GetWithPositronEnergy()
      ensp_juno['rvis_crel_flu'] = ensp_juno['rvis_crel_flu_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_juno['scintNL'])

      ensp_tao['ribd_crel'] = ensp_tao['ribd'].Copy()
      ensp_tao['ribd_crel'].GetScaled(1./extrafactors_tao)
      ensp_tao['ribd_crel'].GetScaled(extrafactors2_tao)
      ensp_tao['rosc_crel_flu'] = ensp_tao['ribd_crel'].GetOscillated(L=args_tao.core_baselines, core_powers=flu_powers_tao, me_rho=args_tao.me_rho, ene_mode='true', args=args_tao)
      del ensp_tao['ribd_crel']
      ensp_tao['rvis_crel_flu_nonl'] = ensp_tao['rosc_crel_flu'].GetWithPositronEnergy()
      ensp_tao['rvis_crel_flu_nonl'] =  ensp_tao['rvis_crel_flu_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args_tao.ene_crop)
      ensp_tao['rvis_crel_flu'] = ensp_tao['rvis_crel_flu_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_tao['scintNL'])

      del ensp_juno['rvis_crel_flu_nonl'], ensp_tao['rvis_crel_flu_nonl']
      return ensp_juno['rvis_crel_flu'], ensp_tao['rvis_crel_flu']

  def get_core_flux_CM():
      print(" # Fluctuating relative core fluxes ")
      start_time_core = datetime.now()
      results = Parallel(n_jobs=-1)(delayed(get_core_flu)(i) for i in range(args_juno.sample_size_core))
      ensp_juno['rvis_crel_flu'], ensp_tao['rvis_crel_flu'] = zip(*results)
      ensp_juno["rdet_crel_flu"] = [s.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop) for s in ensp_juno['rvis_crel_flu']]
      ensp_tao["rdet_crel_flu"] = [s.ApplyDetResp(resp_matrix, pecrop=args_juno.ene_crop) for s in ensp_tao['rvis_crel_flu']]
      del ensp_juno['rvis_crel_flu']
      del ensp_tao['rvis_crel_flu']
      end_time_core = datetime.now()
      print("Core flu time", end_time_core - start_time_core)
      cm_juno['crel'] = ensp_juno["rdet"].GetCovMatrixFromRandSample(ensp_juno["rdet_crel_flu"])
      cm_tao['crel'] = ensp_tao["rdet"].GetCovMatrixFromRandSample(ensp_tao["rdet_crel_flu"])
      cm_comb = cm_juno['crel'].Extend(cm_tao['crel'])
      del ensp_juno["rdet_crel_flu"]
      del ensp_tao["rdet_crel_flu"]
      return cm_comb

 # Initialize the cm dictionary
  cm = {}
# Iterate over the unc list and use the map to call the appropriate function
  #for u in unc:
  if unc in unc_map:
      cm[unc] = unc_map[unc]()
      SaveObject(cm[unc], f"{args_juno.data_matrix_folder}/cm_correlated_{unc}_{args_juno.bayes_chi2}_{args_juno.sin2_th13_opt}_NO-{args_juno.NMO_opt}_{args_juno.stat_opt}_{args_juno.bins}bins.dat")
      return cm[unc]
  else:
      raise ValueError(f"unc {unc} not found! Cannot calculate!")
