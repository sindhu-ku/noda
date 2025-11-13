#!/usr/bin/env python3
import sys, os
from .noda import *
import csv
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from scipy import constants

def CreateSpectra(ndays=10,
               ebins=None,
               detector = "JUNO",
               backgrounds =None,
               ene_leak_nd=None,
               args=None):


  opt = {'detector': detector, 'efficiency': args.detector_efficiency, 'ndays': ndays, 'core_baselines': args.core_baselines, 'core_powers': args.core_powers}
# --- split rfis per NPP using compute_extrafactors ---
  core_powers = np.array(args.core_powers)
  core_baselines = np.array(args.core_baselines)
  isotope_scales = {
    'U235': args.U235_scale,
    'U238': args.U238_scale,
    'Pu239': args.Pu239_scale,
    'Pu241': args.Pu241_scale
  }
  print(opt)

  # Nominal spectra
  ensp = {}   # energy spectra
  events = {} #events in each spectrum
  rebin_mode = 'spline'
  if not os.path.exists(args.plots_folder):
      os.makedirs(args.plots_folder)
  binw = (ebins[1]-ebins[0])*1000
  print(" # Constructing nominal spectra")

      # IBD xsection
  ensp['sibd'] = GetSpectrumFromROOT(args.input_data_file, 'IBDXsec_VogelBeacom_DYB')
  s_ibd = sp.interpolate.interp1d(ensp['sibd'].GetBinCenters(), ensp['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['sibd'].bin_cont[0],       ensp['sibd'].bin_cont[-1]))
  ensp['bump_corr'] = GetSpectrumFromROOT(args.input_data_file, 'DYBFluxBump_ratio')


  s_bump_lin = sp.interpolate.interp1d(ensp['bump_corr'].GetBinCenters(), ensp['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['bump_corr'].bin_cont[0], ensp['bump_corr'].bin_cont[-1]))

  #cm_suffix += "_{:}bins".format(int(0.5*(len(ebins)-1)))





  #   reactor x IBD
  #     From ROOT file
  ensp['rfis0'] = GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_U235', scale=args.U235_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_U238', scale=args.U238_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_Pu239', scale=args.Pu239_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_Pu241', scale=args.Pu241_scale)

  

  ensp['rfis0'].Rebin(ebins, mode='spline-not-keep-norm')

  ensp['rfis0'].ScaleByBinWidth()


  ensp['rfis1'] = GetSpectrumFromDAT(
    "data/RS_dummy.dat",
    scale_U235=args.U235_scale,
    scale_U238=args.U238_scale,
    scale_Pu239=args.Pu239_scale,
    scale_Pu241=args.Pu241_scale
  )

  # # Apply same processing as rfis0

  ensp['rfis1'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['rfis1'].ScaleByBinWidth()

  #ensp['rfis0'] = GetSpectrumFromROOT("ReactorPWG_check3_Nov8.root", "rea_flux") #+ ensp['bckg'] #
  ensp['rfis_b'] =  ensp['rfis0'].Copy()
  ensp['rfis_b1'] =  ensp['rfis1'].Copy()
  
  ensp['rfis_b'].WeightWithFunction(s_ibd)
  ensp['rfis_b1'].WeightWithFunction(s_ibd)

  if detector=="JUNO":
    if args.fit_type != "PMOP_combined":
        ensp['rfis_b'] = ensp['rfis_b'].GetWeightedWithFunction(s_bump_lin)

  #ensp['rfis_b1'] = ensp['rfis_b1'].GetWeightedWithFunction(s_bump_lin)
  del s_bump_lin
  #
  # Complete IBD flux
  ensp['rfis'] = ensp['rfis_b'].Copy()
  ensp['rfiss'] = ensp['rfis_b1'].Copy()


  ensp['rfis'].GetScaled(ndays)
  ensp['rfiss'].GetScaled(ndays)
  extrafactors = compute_extrafactors(args)


  print("extrafactors", extrafactors)


  print(ensp["rfis0"].GetIntegral())
  print(" ")
  print("NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral()))


   # --- Define core groups ---
  idx_YJ, idx_TS, idx_HM = range(0, 6), range(6, 8), range(8, 10)
  
  # --- Compute normalized per-core weights ---
  core_weights = core_powers / core_baselines**2
  core_weights /= core_weights.sum()
  
  # --- Create per-core rfis spectra ---
  rfis_core = []
  for i in range(10):
      spec = ensp['rfis'].Copy()   # unscaled spectrum
      spec.GetScaled(extrafactors * core_weights[i])  # apply fractional contribution
      rfis_core.append(spec)
      ensp[f'rfis_core{i}'] = spec  # optional: store individually for later plotting/debugging
  
  # --- Define interpolation helper ---
  def interp_to_target(source_spec, target_spec):
      src_x = source_spec.GetBinCenters()
      src_y = source_spec.bin_cont
      tgt_x = target_spec.GetBinCenters()
      f_interp = sp.interpolate.interp1d(
          src_x, src_y, kind='linear',
          bounds_error=False,
          fill_value=(src_y[0], src_y[-1])
      )
      new_spec = target_spec.Copy()
      new_spec.bin_cont = f_interp(tgt_x)
      return new_spec
  
  # --- Global SNF and Noneq base corrections ---
  ensp['snf0'] = GetSpectrumFromROOT(args.input_data_file, 'SNF_FluxRatio')
  ensp['snf0'].Rebin(ebins, mode='spline-not-keep-norm')
  snf0_interp0 = interp_to_target(ensp['snf0'], ensp['rfis_core8'])
  snf0_interp1 = interp_to_target(ensp['snf0'], ensp['rfis_core9'])
  ensp['snf_FC'] = snf0_interp0.GetWeightedWithSpectrum(ensp['rfis_core8'])
  ensp['snf_DB'] = snf0_interp1.GetWeightedWithSpectrum(ensp['rfis_core9'])

  snf_HM = ensp['snf_FC'] + ensp['snf_DB']
  del ensp['snf0']
  
  ensp['noneq0'] = GetSpectrumFromROOT(args.input_data_file, 'NonEq_FluxRatio')
  ensp['noneq0'].Rebin(ebins, mode='spline-not-keep-norm')
  noneq0_interp0 = interp_to_target(ensp['noneq0'], ensp['rfis_core8'])
  noneq0_interp1 = interp_to_target(ensp['noneq0'], ensp['rfis_core9'])
  ensp['noneq_FC'] = noneq0_interp0.GetWeightedWithSpectrum(ensp['rfis_core8'])
  ensp['noneq_DB'] = noneq0_interp1.GetWeightedWithSpectrum(ensp['rfis_core9'])
  del ensp['noneq0']
  
  noneq_HM = ensp['noneq_FC'] + ensp['noneq_DB']
 
  # --- SNF correction per core ---
  snf_core = []
  for i in range(10):
      if i in idx_YJ:
          prefix, core_idx = "YJ", i - idx_YJ.start + 1
      if i in idx_TS:
          prefix, core_idx = "TS", i - idx_TS.start + 1

      hname = f"Ratio{prefix}ReactorSNF{core_idx}"
      spec = GetSpectrumFromROOT(args.snf_file, hname)
      spec.Rebin(ebins, mode='spline-not-keep-norm')
      spec_interp = interp_to_target(spec, rfis_core[i])
      weighted = spec_interp.GetWeightedWithSpectrum(rfis_core[i])
      snf_core.append(weighted)
  
  # --- YJ shutdowns (applied only to YJ cores collectively) ---
  spec_off1 = GetSpectrumFromROOT(args.snf_file, "RatioYJReactor1Shutdown")
  spec_off1.Rebin(ebins, mode='spline-not-keep-norm')
  spec_off1_interp = interp_to_target(spec_off1, rfis_core[0])
  spec_off1_weighted = spec_off1_interp.GetWeightedWithSpectrum(rfis_core[0])
  
  spec_off2 = GetSpectrumFromROOT(args.snf_file, "RatioYJReactor4Shutdown")
  spec_off2.Rebin(ebins, mode='spline-not-keep-norm')
  spec_off2_interp = interp_to_target(spec_off2, rfis_core[3])
  spec_off2_weighted = spec_off2_interp.GetWeightedWithSpectrum(rfis_core[3])
  

  # --- Non-equilibrium corrections per core ---
  noneq_core = []
  for i in range(10):
      if i in idx_YJ:
          prefix, core_idx = "YJ", i - idx_YJ.start + 1
      if i in idx_TS:
          prefix, core_idx = "TS", i - idx_TS.start + 1

      rfis_i = rfis_core[i]
      group_noneq = None
      for iso, scale in isotope_scales.items():
          if iso == 'U238':
              continue
          hname = f"{prefix}{core_idx}_{iso}"
          spec = GetSpectrumFromROOT(args.noneq_file, hname)
          spec.Rebin(ebins, mode='spline-not-keep-norm')
          spec_scaled = spec.GetScaledFit(scale * 0.01)
          spec_interp = interp_to_target(spec_scaled, rfis_i)
          weighted = spec_interp.GetWeightedWithSpectrum(rfis_i)
          group_noneq = weighted.Copy() if group_noneq is None else group_noneq + weighted
      noneq_core.append(group_noneq)
  
  # --- Combine per-NPP ---
  snf_YJ = Spectrum(bins=ensp['rfis'].bins, bin_cont = ensp['rfis'].bin_cont*0.0)
  noneq_YJ = Spectrum(bins=ensp['rfis'].bins, bin_cont = ensp['rfis'].bin_cont*0.0)
  snf_TS = Spectrum(bins=ensp['rfis'].bins, bin_cont = ensp['rfis'].bin_cont*0.0)
  noneq_TS = Spectrum(bins=ensp['rfis'].bins, bin_cont = ensp['rfis'].bin_cont*0.0)
  for i in idx_YJ: 
      snf_YJ += snf_core[i]
      noneq_YJ += noneq_core[i]
  for i in idx_TS: 
      snf_TS += snf_core[i] 
      noneq_TS += noneq_core[i]

  snf_YJ = snf_YJ + spec_off1_weighted + spec_off2_weighted

  # --- Store results ---
  ensp['snf_YJ'], ensp['snf_TS'], ensp['snf_HM'] = snf_YJ, snf_TS, snf_HM
  ensp['noneq_YJ'], ensp['noneq_TS'], ensp['noneq_HM'] = noneq_YJ, noneq_TS, noneq_HM
  ensp['snf'] = snf_YJ + snf_TS + snf_HM
  ensp['noneq'] = noneq_YJ + noneq_TS + noneq_HM
  
  ensp['rfiss'].GetScaled(extrafactors)
  ensp['rfis'].GetScaled(extrafactors)
  ensp['ribd'] = ensp['rfis'] +  ensp['snf'] +  ensp['noneq'] 

  print("Before corr",ensp['rfis'].GetIntegral())
  # print("SNF ", ensp['snf_tot'].GetIntegral())
  # print("noneq ", ensp['noneq_tot'].GetIntegral())
  


  ensp['ribd1'] = ensp['rfiss'] + ensp['snf'] + ensp['noneq']
  events['ribd'] = ensp["ribd"].GetIntegral()
#  ensp['rfis0'].Plot(f"{args.plots_folder}/reac_spectrum.pdf",
#                   xlabel="Neutrino energy (MeV)",
#                   ylabel=f"Events per {binw:0.1f} keV",
#                   extra_spectra=[ensp['rfis'], ensp['ribd'], ensp['snf'], ensp['noneq']],
#                   leg_labels=['From HM', '+ bump fix', ' + SNF + NonEq', 'SNF', 'NonEq'],
#                   colors=['black', 'red', 'blue', 'green', 'magenta'],
#                   xmin=0, xmax=10,
#                   ymin=0, ymax=None, log_scale=False)
  #del ensp['rfis0']#, ensp['rfis']
  print(" ")
  print("NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral()))
  print(" ")
  print("2-6 MeV NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral(left_edge = 2, right_edge = 6)))


  # Oscillated spectrum
  if not detector == "DYB":
      ensp['rosc'] = ensp['ribd'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      ensp['rosc1'] = ensp['ribd1'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
  else:
      ensp['rosc'] = ensp['ribd']
      ensp['rosc1'] = ensp['ribd1']

  events['rosc'] = ensp["rosc"].GetIntegral()

  #ensp['ribd'].Plot(f"{args.plots_folder}/osc_spectrum.pdf",
  #                 xlabel="Neutrino energy (MeV)",
  #                 ylabel=f"Events per {binw:0.1f} keV",
  #                 extra_spectra=[ensp['rosc']],
  #                 leg_labels=['Reactor', 'Oscillated'],
  #                 colors=['black', 'darkred'],
  #                 xmin=0, xmax=10,
  #                 ymin=0, ymax=None, log_scale=False)
  #print(ensp['rosc'].GetIntegral(), ensp['rosc1'].GetIntegral())
 # ensp['rosc_nome'] = ensp['ribd'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=0.0, ene_mode='true', args=args)


  ensp['rosc'].Plot(f"{args.plots_folder}/osc_spectrum_{detector}.png",
                  xlabel="Neutrino energy (MeV)",
                  ylabel=f"Events per {binw:0.1f} keV",
                  extra_spectra=[ensp['ribd']],
                  leg_labels=['oscillated', 'unoscillated'],
                  colors=['black', 'darkred'],
                  xmin=0, xmax=10,
                  ymin=0, ymax=None, log_scale=False)
  #del ensp['rosc_nome']

  if not detector == "DYB":
      ensp['snf_osc'] = ensp['snf'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      #del ensp['snf']
      ensp['noneq_osc'] = ensp['noneq'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      #del ensp['noneq']
  else:
      ensp['snf_osc'] = ensp['snf']
      ensp['noneq_osc'] = ensp['noneq']

  ensp['rvis_nonl'] = ensp['rosc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
  ensp['rvis_nonl1'] = ensp['rosc1'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
 
  ensp['rvis_nonl_temp'] = ensp['ribd'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)

  ensp['snf_osc_nonl'] = ensp['snf_osc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
  # del ensp['snf_osc']
  ensp['noneq_osc_nonl'] = ensp['noneq_osc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
  # del ensp['noneq_osc']

  # if detector != "JUNO":
  #     #ensp['rosc_pos'] = ensp['rosc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2) #only for plotting
  #     #ensp['rosc_eneleak'] = ensp['rvis_nonl'].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop) #only for plotting
  #     ensp['rvis_nonl'] = ensp['rvis_nonl'].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)
  #     ensp['rvis_nonl1'] = ensp['rvis_nonl1'].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)
  #     ensp['snf_osc_nonl'] = ensp['snf_osc_nonl'].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)
  #     ensp['noneq_osc_nonl'] = ensp['noneq_osc_nonl'].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop) 
  #   Non-linearity
  ensp['scintNL'] = GetSpectrumFromROOT(args.input_data_file, args.nl_hist_name)
  ensp['NL_pull'] = [ GetSpectrumFromROOT(args.input_data_file, 'positronScintNLpull0'),
                      GetSpectrumFromROOT(args.input_data_file, 'positronScintNLpull1'),
                      GetSpectrumFromROOT(args.input_data_file, 'positronScintNLpull2'),
                      GetSpectrumFromROOT(args.input_data_file, 'positronScintNLpull3') ]

  if args.nl_hist_name == "J22rc0_positronScintNL":
      ensp['old_scintNL'] = GetSpectrumFromROOT(args.input_data_file, "positronScintNL")
      ratio = ensp['scintNL'].bin_cont/ensp['old_scintNL'].bin_cont
      for i in range(len(ensp['NL_pull'])):
          new_bin_cont = ensp['NL_pull'][i].bin_cont*ratio
          ensp['NL_pull'][i] = Spectrum(bins=ensp['NL_pull'][i].bins, bin_cont=new_bin_cont)


  print("applying non-linearity")
  if not detector == "DYB": 
      if args.NL == "NMO":
          ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
          ensp['rvis1'] = ensp['rvis_nonl1'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      else:
          ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
          ensp['rvis1'] = ensp['rvis_nonl1'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
  else:
      ensp['rvis'] = ensp['rvis_nonl']
      ensp['rvis1'] = ensp['rvis_nonl1']


  if not detector == "DYB": 
      if args.NL == "NMO":
          ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
          ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
          ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      else:   
          ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
          ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
          ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)

  else:
      ensp['rvis_temp'] = ensp['rvis_nonl_temp']
  #     ensp['snf_osc_vis'] = ensp['snf_osc_nonl']
  #     ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl']
  # del ensp['snf_osc_nonl'], ensp['noneq_osc_nonl']
  print("length of NL Pull:")
  print (len(ensp['NL_pull']))

  if args.plot_spectra:
      ensp['rvis_nonl'].Plot(f"{args.plots_folder}/vis_spectra.png",
                        xlabel="Visual energy (MeV)",
                        ylabel=f"Events per {binw:0.1f} keV",
                        extra_spectra=[ensp['rvis']],
                        leg_labels=['Before NL', 'After NL'],
                        colors=['darkred', 'green'],
                        xmin=0, xmax=10,
                        ymin=0, ymax=None, log_scale=False)

      ensp['scintNL'].Plot(f"{args.plots_folder}/non_linearity_old.png",
                  xlabel="Reconstructed energy (MeV)",
                  extra_spectra=ensp['NL_pull'],
                  xmin=0, xmax=10,
                  ymin=0.9, ymax=1.1, log_scale=False)


  a, b, c = args.a, args.b, args.c

  if os.path.isfile(f"{args.data_matrix_folder}/rm_{detector}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat") and not bool(args.FORCE_CALC_RM):
    resp_matrix = LoadRespMatrix(f"{args.data_matrix_folder}/rm_{detector}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
  else:
    resp_matrix = CalcRespMatrix_abc(args.a, args.b, args.c, escale=1, ebins=ebins, pebins=np.linspace(args.min_ene, args.max_ene, args.bins))# CalcRespMatrix_abc(a, b, c, escale=1, ebins=ebins, pebins=ebins)
    resp_matrix.Save(f"{args.data_matrix_folder}/rm_{detector}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")

  if bool(args.FORCE_CALC_RM):
    resp_matrix.Dump(f"{args.data_matrix_folder}/csv_{args.stat_opt}/{detector}_resp_matrix.csv")



  ensp['rdet'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop) 

  ensp['rdet1'] = ensp['rvis1'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['rdet_temp'] = ensp['rvis_temp'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['snf_final'] = ensp['snf_osc_vis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
  ensp['noneq_final'] = ensp['noneq_osc_vis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
  for l in ["YJ", "TS","HM"]:
      ensp[f'snf_{l}_osc'] = ensp[f'snf_{l}'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      ensp[f'noneq_{l}_osc'] = ensp[f'noneq_{l}'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
      
      ensp[f'snf_{l}_pos'] = ensp[f'snf_{l}_osc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      ensp[f'noneq_{l}_pos'] = ensp[f'noneq_{l}_osc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      
      ensp[f'snf_{l}_nonl'] = ensp[f'snf_{l}_pos'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
      ensp[f'noneq_{l}_nonl'] = ensp[f'noneq_{l}_pos'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
      
      ensp[f'snf_{l}_final'] = ensp[f'snf_{l}_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop) 
      ensp[f'noneq_{l}_final'] = ensp[f'noneq_{l}_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
      
  # ensp['noneq_YJ_final'].Plot(f"{args.plots_folder}/noneq_YJ.png")
  # ensp['noneq_TS_final'].Plot(f"{args.plots_folder}/noneq_TS.png")
  # ensp['noneq_HM_final'].Plot(f"{args.plots_folder}/noneq_HM.png")
  # ensp['snf_YJ_final'].Plot(f"{args.plots_folder}/snf_YJ.png")
  # ensp['snf_TS_final'].Plot(f"{args.plots_folder}/snf_TS.png")
  # ensp['snf_HM_final'].Plot(f"{args.plots_folder}/snf_HM.png")  
  # --- Final ribd per NPP ---
  #del ensp['snf_osc_vis'], ensp['noneq_osc_vis']

  extra_spectra = []
  leg_labels = []
  colors = []
  ensp['bckg'] = None
  if not detector == "DYB":
    for key, info in backgrounds[detector].items():
        ensp[key] = GetSpectrumFromROOT(info["rootfile"], info["histname"])
        if key == "rea300":
            ensp['rfis_rea300'] = ensp[key].Rebin(ebins, mode='spline-not-keep-norm')
            ensp['rfis_rea300_pos'] = ensp['rfis_rea300'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
            ensp['rvis_rea300'] = ensp['rfis_rea300_pos'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
            ensp[key] = ensp['rvis_rea300'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            
        scale = float(info["rate"])*ndays / (args.duty_cycle*ensp[key].GetIntegral())
        ensp[key].GetScaled(scale)
        print(key, ensp[key].GetIntegral(), info["rate"])
        ensp[key].Trim(args.ene_crop)
        # ensp[key].Rebin(ebins, mode="spline-not-keep-norm")
        if args.rebin_nonuniform: 
            ensp[key].Rebin_nonuniform(args.bins_nonuniform)
        if ensp['bckg'] is None:
            ensp['bckg'] = ensp[key]
        else:
            ensp['bckg'] += ensp[key]
        print(key, ensp[key].GetIntegral())
  
  def make_ana_geo_spectra(rate, ratio, name):
      ensp['rfis0_geou'] = GetSpectrumFromROOT(args.geo_file, 'geoU')
      ensp['rfis0_geoth'] = GetSpectrumFromROOT(args.geo_file, 'geoTh')
      ensp['rfis0_geou'].Rebin(ebins, mode='spline-not-keep-norm')
      
      ensp['rfis0_geoth'].Rebin(ebins, mode='spline-not-keep-norm')

      ensp['rfis0_geou'].ScaleByBinWidth()
      ensp['rfis0_geoth'].ScaleByBinWidth()

      
      ensp['rfis0_geou'].WeightWithFunction(s_ibd)
      ensp['rfis0_geoth'].WeightWithFunction(s_ibd)

      ensp[f'rfis_{name}u'] = ensp['rfis0_geou'].GetScaledFit(ndays*rate/((1+ratio)*args.duty_cycle*ensp['rfis0_geou'].GetIntegral()))
      ensp[f'rfis_{name}th'] = ensp['rfis0_geoth'].GetScaledFit(ndays*rate*ratio/((1+ratio)*args.duty_cycle*ensp['rfis0_geoth'].GetIntegral()))
      #ensp[f'rfis_{name}u'] = ensp['rfis0_geou'].GetScaledFit(1./((1+ratio)*ensp['rfis0_geou'].GetIntegral()))
      #ensp[f'rfis_{name}th'] = ensp['rfis0_geoth'].GetScaledFit(ratio/((1+ratio)*ensp['rfis0_geoth'].GetIntegral()))

      
      #ensp['rfis_geo'] =  ensp['rfis0_geou'] + ensp['rfis0_geoth']

      
      ensp[f'rvis_{name}u'] = ensp[f'rfis_{name}u'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      ensp[f'rvis_{name}th'] = ensp[f'rfis_{name}th'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      #ensp['rvis_geo'] = ensp['rfis_geo'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)

      ensp[f'rvis_{name}u_nonl'] = ensp[f'rvis_{name}u'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared) #GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp[f'rvis_{name}th_nonl'] = ensp[f'rvis_{name}th'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared) #GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      #ensp['rvis_geo_nonl'] = ensp['rvis_geo'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)

      ensp[f'{name}u'] = ensp[f'rvis_{name}u_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
      ensp[f'{name}th'] = ensp[f'rvis_{name}th_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

      #ensp['geo_full'] = ensp['rvis_geo_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
      
      if args.rebin_nonuniform: 
          ensp[f'{name}u'].Rebin_nonuniform(args.bins_nonuniform)
          ensp[f'{name}th'].Rebin_nonuniform(args.bins_nonuniform)
          #ensp['geo_full'].Rebin_nonuniform(args.bins_nonuniform)
          
      #del ensp['rfis0_geou'], ensp['rfis0_geoth'], ensp[f'rfis_{name}u'], ensp[f'rfis_{name}th']
      return ensp[f'{name}u'] + ensp[f'{name}th']

  if args.geo_spectra == 'ana' and detector == "JUNO":
      if args.geo_fit_type == 'mantle':
          ensp['geomantle'] = make_ana_geo_spectra(args.mantle_rate.get(args.mantle_model), args.mantle_Th_U_ratio.get(args.mantle_model), 'mantle')
          ensp['geocrust'] = make_ana_geo_spectra(float(args.crust_rate_tnu)*args.tnu_to_cpd, args.crust_Th_U_ratio, 'crust')
          ensp['geo_ana'] = ensp['geocrust'] + ensp['geomantle']
      else:
          ensp['geo_ana'] = make_ana_geo_spectra(float(args.geo_rate), args.Th_U_ratio, 'geo')
          #scale = float(args.geo_rate)*ndays / (args.duty_cycle*ensp['geo_ana'].GetIntegral())
          #ensp['geo_full'].GetScaled(scale)
          #ensp['geo_ana'].GetScaled(scale)
          
          ensp['bckg'] = ensp['bckg'] - ensp['geo'] + ensp['geo_ana']
          ensp['geo'] = ensp['geo_ana']
          print('geo', ensp['geo'].GetIntegral())

  ensp['rdet'] = ensp['rdet'] #+ ensp['snf_final'] + ensp['noneq_final']
  if args.rebin_nonuniform: 
      ensp['rdet'].Rebin_nonuniform(args.bins_nonuniform)
      ensp['rdet1'].Rebin_nonuniform(args.bins_nonuniform)
      #ensp['snf_final'].Rebin_nonuniform(args.bins_nonuniform)
      #ensp['noneq_final'].Rebin_nonuniform(args.bins_nonuniform)

  events['rdet'] = ensp['rdet'].GetIntegral()

  if detector=="DYB":
      #ensp['rtot'] =  ensp['rdet']
      # DYB_spectrum = pd.read_csv(args.dyb_spectrum, delim_whitespace=True, names=["Bin_Center", "Sigma", "Rel_Unc"])
      # ensp['rtot'] = Spectrum(bins=ensp['rdet'].bins, bin_cont=DYB_spectrum["Sigma"].to_numpy())
      # ensp['rtot'].WeightWithFunction(s_ibd)
      ensp['rtot'] = ensp['rdet']
      extra_spectra = []
      leg_labels = ['DYB expectation']
      colors = ['brown']
  else:
      ensp['rtot'] = ensp['rdet'] + ensp['bckg']

  events['rtot'] = ensp['rtot'].GetIntegral()
  if not detector == "DYB":
      events['rbckg'] = ensp['bckg'].GetIntegral()
      events['acc'] = ensp['acc'].GetIntegral()
      events['fneu'] = ensp['fneu'].GetIntegral()
      events['lihe'] = ensp['lihe'].GetIntegral()
      events['aneu'] = ensp['aneu'].GetIntegral()
      events['geo'] = ensp['geo'].GetIntegral()
      events['atm'] = ensp['atm'].GetIntegral()
      events['rea300'] = ensp['rea300'].GetIntegral()


  # if detector == "JUNO":

  #    ensp['rdet'].WritetoROOT("ci", "Rea_rfis.root")
  #    ensp['rdet1'].WritetoROOT("rs", "Rea_rfis.root")
  if args.plot_spectra:
      ensp['rdet'].Plot(f"{args.plots_folder}/rfis_spectra_{detector}.png",
              xlabel="Reconstructed energy (MeV)",
              ylabel=f"", #Events per {binw:0.1f} keV",
              extra_spectra=[ensp["rdet1"]],
              leg_labels=['Common Inputs', 'Summation model (RenShape)'],
              colors=['steelblue', 'brown'],
              xmin=0.8, xmax=12.0,
              ymin=1e-3,log_scale=False)
      
      if detector == "JUNO":
          extra_spectra=[ensp['acc'], ensp['geo'], ensp['lihe'], ensp['fneu'], ensp['aneu'], ensp['atm'], ensp['rea300'], ensp['doublen'], ensp['bipo']]
          leg_labels = ['Reactor antineutrinos', 'Accidentals', 'Geoneutrinos', 'Li9/He8','Fast neutrons', '(alpha, n)', 'Atmospheric', 'Reactors > 300 km', 'Double neutrons', 'BiPo214']
          colors=['black', 'darkred', 'green', 'navy', 'orange', 'magenta', 'lightblue', 'yellow', 'brown', 'purple', 'grey']
    

      if detector == "TAO":
          extra_spectra=[ensp['rosc_pos'], ensp['rosc_eneleak'], ensp['acc'], ensp['lihe'], ensp['fneu']]
          leg_labels = ['+ NL + energy res', '+ flux and cross-sec', '+ energy leak', 'Accidentals', 'Li9/He8', 'Fast neutrons']
          colors=['grey', 'magenta', 'darkred', 'green', 'navy', 'orange']

      
      ensp['rdet'].Plot(f"{args.plots_folder}/det_spectra_{detector}.png",
                  xlabel="Reconstructed energy (MeV)",
                  ylabel=f"", #Events per {binw:0.1f} keV",
                  extra_spectra=extra_spectra,
                  leg_labels=leg_labels,
                  colors=colors,
                  xmin=0.8, xmax=12.0,
                  ymin=1e-3,log_scale=False)
                  #ymin=0.0, ymax=14900, yinterval=2500, log_scale=False)



      resp_matrix.Plot(f"{args.plots_folder}/resp_mat.pdf")




  #
  print(" ")
  print(" ")
  print("   Reac x IBD:    {:.2f} events".format(events['ribd']))
  print("   Oscillated:    {:.2f} events".format(events['rosc']))
  print("   Detected:      {:.2f} events".format(events['rdet']))
  print("   IBD+BG:        {:.2f} events".format(events["rtot"]))

  if not detector == "DYB":
     print("   Backgrounds")
     print("   Total:        {:.2f} events".format(events["rbckg"]))
     print("      accidentals:    {:.2f} events".format(events["acc"]))
     print("      Li-9/He-8:      {:.2f} events".format(events["lihe"]))
     print("      fast n:         {:.2f} events".format(events["fneu"]))

  if detector == "JUNO":

    print("      (alpha,n):      {:.2f} events".format(events["aneu"]))
    print("      geo-nu:         {:.2f} events".format(events["geo"]))
    print("      atmospheric:         {:.2f} events".format(events["atm"]))
    print("      global reactors:         {:.2f} events".format(events["rea300"]))
  del events

 
  print(" # Initialization completed")
    #
  print(" # Dumping data")
  # for key in ensp.keys():
  #     if type(ensp[key]) != list:
  #         ensp[key].Dump(f"{args.data_matrix_folder}/csv_{args.stat_opt}/ensp_{detector}_{key}.csv")

  # for i in range(0, 100):
  #     ensp['toy'] = Spectrum(bins=ensp['rtot'].bins, bin_cont=np.random.poisson(ensp['rtot'].bin_cont))
  #     ensp['toy'].WritetoROOT(f"toy{i}", "chi2_test_toys.root")
  # ensp['rtot'].WritetoROOT("asimov", "ReactorPWG_check3_Nov8.root")
  #ensp['rfis'].WritetoROOT("rea_flux_xsec+bump", "ReactorPWG_check3_Nov8.root")
  if args.write_output:
      ensp['rdet'].WritetoROOT("rea_prefit", "Post_fit_NODA_JUNO+DYB_Nov12_reprod25C_blind.root")
      #ensp['geo_ana'].WritetoROOT("geo", "Geo_Nov4.root")
      #ensp['rea300'].WritetoROOT("rea300", "rea300_Nov4.root")
#  for key in bg_keys:
#      ensp[key].WritetoROOT(key, "Geo_test/Spectra_Sep2_25.root")
  #ensp['rtot'] = GetSpectrumFromROOT("ReactorPWG_check3_Nov8.root", "toy") #+ ensp['bckg'] #
  if args.is_data: 
      #ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, "GroupB/h_data_B")
      ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, "ibd")
      if args.rebin_nonuniform: 
          ensp["data"].Rebin_nonuniform(args.bins_nonuniform)
      #ensp['data'] += ensp['bckg']

  # ensp['rfis'].WritetoROOT("flux_xsec", "Reactor_xchecks_Nov11.root")
  # ensp['ribd'].WritetoROOT("unosc", "Reactor_xchecks_Nov11.root")
  # ensp['rosc'].WritetoROOT("osc", "Reactor_xchecks_Nov11.root")
  # ensp['rvis_nonl'].WritetoROOT("dep", "Reactor_xchecks_Nov11.root")
  # ensp['rvis'].WritetoROOT("vis", "Reactor_xchecks_Nov11.root")
  # ensp['rdet'].WritetoROOT("reco", "Reactor_xchecks_Nov11.root")
  # # ensp['geo'].WritetoROOT("geo", "Geo_Nov9.root")
  # ensp['rea300'].WritetoROOT("worldrea", "WR_Nov9.root")
  print("rea_precross-sec", ensp['rfis0'].GetIntegral())
  print("rea_precorr", ensp['rfis'].GetIntegral())
  print("snf", ensp['snf_final'].GetIntegral())
  print("noneq", ensp['noneq_final'].GetIntegral())
  print("rea_preosc", ensp['ribd'].GetIntegral())
  print("rea", ensp['rdet'].GetIntegral())
  print("total", ensp['rtot'].GetIntegral())  
  #ensp["data"] = ensp['rtot'] - ensp['rdet'] + ensp['data_rea']
  #rf = uproot.open(args.input_data_file) 
  return ensp, resp_matrix
