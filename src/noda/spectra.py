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
    
  ensp['rfis_b'] =  ensp['rfis0'].Copy()
  ensp['rfis_b1'] =  ensp['rfis1'].Copy()
  
  ensp['rfis_b'].WeightWithFunction(s_ibd)
  ensp['rfis_b1'].WeightWithFunction(s_ibd)


  if detector=="JUNO" and not args.fit_type == "PMOP_combined":
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
  ensp['rfis'].GetScaled(extrafactors) #correct normalization including fission fractions, mean energy per fission ... eq.13.5 YB
  ensp['rfiss'].GetScaled(extrafactors)
  #ensp['rfiss'].GetScaled(ensp['rfis'].GetIntegral()/ensp['rfiss'].GetIntegral())
 
  print(" ")
  print("NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral()))

  #
  ensp['snf0'] = GetSpectrumFromROOT(args.input_data_file, 'SNF_FluxRatio')     #spent nuclear fuel
  ensp['snf0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['snf'] = ensp['snf0'].GetWeightedWithSpectrum(ensp['rfis'])
  ensp['snf1'] = ensp['snf0'].GetWeightedWithSpectrum(ensp['rfiss'])
  del ensp['snf0']


  ensp['noneq0'] = GetSpectrumFromROOT(args.input_data_file, 'NonEq_FluxRatio')
  ensp['noneq0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['noneq'] = ensp['noneq0'].GetWeightedWithSpectrum(ensp['rfis']) #non-equilibrium ratio
  ensp['noneq1'] = ensp['noneq0'].GetWeightedWithSpectrum(ensp['rfiss']) #non-equilibrium ratio
  del ensp['noneq0']
    
  ensp['ribd'] = ensp['rfis'] + ensp['snf'] + ensp['noneq']
  ensp['ribd1'] = ensp['rfiss'] + ensp['snf1'] + ensp['noneq1']
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
  del ensp['snf_osc']
  ensp['noneq_osc_nonl'] = ensp['noneq_osc'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
  del ensp['noneq_osc']

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
      ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp['rvis1'] = ensp['rvis_nonl1'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      # ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)
      # ensp['rvis1'] = ensp['rvis_nonl1'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)
  else:
      ensp['rvis'] = ensp['rvis_nonl']
      ensp['rvis1'] = ensp['rvis_nonl1']

  print(ensp['rvis'].bin_cont.shape)      
  
  if not detector == "DYB": 
      ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
     
      # ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)
      # ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)
      # ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)

  else:
      ensp['rvis_temp'] = ensp['rvis_nonl_temp']
      ensp['snf_osc_vis'] = ensp['snf_osc_nonl']
      ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl']
  del ensp['snf_osc_nonl'], ensp['noneq_osc_nonl']
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


  del ensp['snf_osc_vis'], ensp['noneq_osc_vis']

  extra_spectra = []
  leg_labels = []
  colors = []
  ensp['bckg'] = None
  if not detector == "DYB":
    for key, info in backgrounds[detector].items():
        ensp[key] = GetSpectrumFromROOT(info["rootfile"], info["histname"])
        # if key == "rea300":
        #     ensp['rfis_rea300'] = ensp[key].Rebin(ebins, mode='spline-not-keep-norm')
        #     ensp['rfis_rea300_pos'] = ensp['rfis_rea300'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
        #     ensp['rvis_rea300'] = ensp['rfis_rea300_pos'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)
        #     ensp[key] = ensp['rvis_rea300'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        scale = float(info["rate"])*ndays / (args.duty_cycle*ensp[key].GetIntegral())
        ensp[key].GetScaled(scale)
        print(key, ensp[key].GetIntegral(), info["rate"])
        # ensp[key].Trim(args.ene_crop)
        # # ensp[key].Rebin(ebins, mode="spline-not-keep-norm")
        # if args.rebin_nonuniform: 
        #     ensp[key].Rebin_nonuniform(args.bins_nonuniform)
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

      # ensp[f'rfis_{name}u'] = ensp['rfis0_geou'].GetScaledFit(ndays*rate/((1+ratio)*args.duty_cycle*ensp['rfis0_geou'].GetIntegral()))
      # ensp[f'rfis_{name}th'] = ensp['rfis0_geoth'].GetScaledFit(ndays*rate*ratio/((1+ratio)*args.duty_cycle*ensp['rfis0_geoth'].GetIntegral()))
      ensp[f'rfis_{name}u'] = ensp['rfis0_geou'].GetScaledFit(1./((1+ratio)*ensp['rfis0_geou'].GetIntegral()))
      ensp[f'rfis_{name}th'] = ensp['rfis0_geoth'].GetScaledFit(ratio/((1+ratio)*ensp['rfis0_geoth'].GetIntegral()))

      ensp['rfis_geo'] =  ensp['rfis0_geou'] + ensp['rfis0_geoth']

      ensp[f'rvis_{name}u'] = ensp[f'rfis_{name}u'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      ensp[f'rvis_{name}th'] = ensp[f'rfis_{name}th'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
      ensp['rvis_geo'] = ensp['rfis_geo'].GetWithPositronEnergy(type=args.posE_shift, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)

      ensp[f'rvis_{name}u_nonl'] = ensp[f'rvis_{name}u'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC) #GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp[f'rvis_{name}th_nonl'] = ensp[f'rvis_{name}th'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC) #GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
      ensp['rvis_geo_nonl'] = ensp['rvis_geo'].GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC)

      ensp[f'{name}u'] = ensp[f'rvis_{name}u_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
      ensp[f'{name}th'] = ensp[f'rvis_{name}th_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

      ensp['geo_full'] = ensp['rvis_geo_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
      
      if args.rebin_nonuniform: 
          ensp[f'{name}u'].Rebin_nonuniform(args.bins_nonuniform)
          ensp[f'{name}th'].Rebin_nonuniform(args.bins_nonuniform)
          ensp['geo_full'].Rebin_nonuniform(args.bins_nonuniform)
          
      del ensp['rfis0_geou'], ensp['rfis0_geoth'], ensp[f'rfis_{name}u'], ensp[f'rfis_{name}th']
      return ensp[f'{name}u'] + ensp[f'{name}th']

  if args.geo_spectra == 'ana' and detector == "JUNO":
      if args.geo_fit_type == 'mantle':
          ensp['geomantle'] = make_ana_geo_spectra(args.mantle_rate.get(args.mantle_model), args.mantle_Th_U_ratio.get(args.mantle_model), 'mantle')
          ensp['geocrust'] = make_ana_geo_spectra(float(args.crust_rate_tnu)*args.tnu_to_cpd, args.crust_Th_U_ratio, 'crust')
          ensp['geo_ana'] = ensp['geocrust'] + ensp['geomantle']
      else:
          ensp['geo_ana'] = make_ana_geo_spectra(float(args.geo_rate), args.Th_U_ratio, 'geo')
          scale = float(args.geo_rate)*ndays / (args.duty_cycle*ensp['geo_full'].GetIntegral())
          ensp['geo_full'].GetScaled(scale)
          ensp['geo_ana'].GetScaled(scale)
          
          ensp['bckg'] = ensp['bckg'] - ensp['geo'] + ensp['geo_full']
          ensp['geo'] = ensp['geo_full']
          print('geo', ensp['geo'].GetIntegral())
  #geo_edep = ensp[f'rvis_geou'] +  ensp[f'rvis_geoth']

  #geo_edep.WritetoROOT("geo", "Geo_edep_Nov1.root")   
  if args.rebin_nonuniform: 
      ensp['rdet'].Rebin_nonuniform(args.bins_nonuniform)
      ensp['rdet1'].Rebin_nonuniform(args.bins_nonuniform)
      ensp['snf_final'].Rebin_nonuniform(args.bins_nonuniform)
      ensp['noneq_final'].Rebin_nonuniform(args.bins_nonuniform)

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
          extra_spectra=[ensp['rdet'], ensp['acc'], ensp['geo'], ensp['lihe'], ensp['fneu'], ensp['aneu'], ensp['atm'], ensp['rea300'], ensp['doublen']]
          leg_labels = ['JUNO expected', 'Reactor antinu', 'Accidentals', 'Geoneutrinos', 'Li9/He8','Fast neutrons', '(alpha, n)', 'Atmospheric', 'Reactors > 300 km', 'Double neutrons']
          colors=['black', 'darkred', 'green', 'navy', 'orange', 'magenta', 'lightblue', 'yellow', 'brown', 'purple']
    

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


  if args.write_output:
      ensp['rdet'].WritetoROOT("rea_prefit", "Pre_fit_NODA_GroupB_JUNO-only.root")
      #ensp['geo_ana'].WritetoROOT("geo", "Geo_Nov4.root")
      #ensp['rea300'].WritetoROOT("rea300", "rea300_Nov4.root")
#  for key in bg_keys:
#      ensp[key].WritetoROOT(key, "Geo_test/Spectra_Sep2_25.root")
  if args.is_data: 
      ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, "GroupB/h_data_B")
      if args.rebin_nonuniform: 
          ensp["data"].Rebin_nonuniform(args.bins_nonuniform)
  print("rea_precorr", ensp['rfis'].GetIntegral())
  print("snf", ensp['snf'].GetIntegral())
  print("noneq", ensp['noneq'].GetIntegral())
  print("rea_preosc", ensp['ribd'].GetIntegral())
  print("rea", ensp['rdet'].GetIntegral())
    
  #ensp["data"] = ensp['rtot'] - ensp['rdet'] + ensp['data_rea']
  #rf = uproot.open(args.input_data_file) 
  return ensp, resp_matrix
