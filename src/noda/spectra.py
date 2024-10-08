#!/usr/bin/env python3
import sys, os
from .noda import *
import csv
from datetime import datetime
from joblib import Parallel, delayed

from scipy import constants


def CreateSpectra(ndays=10,
               ebins=None,
               detector = "juno",
               resp_matrix=None,
               ene_leak_tao=None,
               args=None):


  opt = {'detector': detector, 'efficiency': args.detector_efficiency, 'ndays': ndays, 'core_baselines': args.core_baselines, 'core_powers': args.core_powers}

  print(opt)


  #cm_suffix += "_{:}bins".format(int(0.5*(len(ebins)-1)))
  #
  #
  # Nominal spectra
  ensp = {}   # energy spectra
  events = {} #events in each spectrum
  rebin_mode = 'spline'
  if not os.path.exists(args.plots_folder):
      os.makedirs(args.plots_folder)
  binw = (ebins[1]-ebins[0])*1000
  print(" # Constructing nominal spectra")
  #   reactor x IBD
  #     From ROOT file
  ensp['rfis0'] = GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_U235', scale=args.U235_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_U238', scale=args.U238_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_Pu239', scale=args.Pu239_scale) + \
                 GetSpectrumFromROOT(args.input_data_file, 'HuberMuellerFlux_Pu241', scale=args.Pu241_scale)
  ensp['rfis0_geou'] = GetSpectrumFromROOT(args.geo_file, 'geoU')
  ensp['rfis0_geoth'] = GetSpectrumFromROOT(args.geo_file, 'geoTh')
  # IBD xsection
  ensp['sibd'] = GetSpectrumFromROOT(args.input_data_file, 'IBDXsec_VogelBeacom_DYB')
  s_ibd = sp.interpolate.interp1d(ensp['sibd'].GetBinCenters(), ensp['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['sibd'].bin_cont[0], ensp['sibd'].bin_cont[-1]))
  #why s_ibd rebinned separately?
  del ensp['sibd']
  ensp['rfis0'].WeightWithFunction(s_ibd)
  ensp['rfis0_geou'].WeightWithFunction(s_ibd)
  ensp['rfis0_geoth'].WeightWithFunction(s_ibd)
  del s_ibd
  ensp['rfis0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['rfis0_geou'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['rfis0_geoth'].Rebin(ebins, mode='spline-not-keep-norm')
  bin_width = ensp['rfis0'].GetBinWidth()
  ensp['rfis0'].GetScaled(bin_width)
  ensp['rfis0_geou'].GetScaled(bin_width)
  ensp['rfis0_geoth'].GetScaled(bin_width)
  #  DYB Bump
  #  Previous spectrum (Oct2020) + extra bins + interpolation lin + getweightedwithfunction
  ensp['bump_corr'] = GetSpectrumFromROOT(args.input_data_file, 'DYBFluxBump_ratio')
#  bins_new_bump = [1.799]+list(ensp['bump_corr'].bins)+[11.999,12]
#  bin_cont_new_bump = [ensp['bump_corr'].bin_cont[0]]+list(ensp['bump_corr'].bin_cont)+[ensp['bump_corr'].bin_cont[-1]]*2
 # ensp['bump_corr'].bins = np.array(bins_new_bump)
 # ensp['bump_corr'].bin_cont = np.array(bin_cont_new_bump)
#  ensp['bump_corr'].Plot(f"{args.plots_folder}/bump_correction.pdf",
#                   xlabel="Neutrino energy (MeV)",
#                   xmin=0, xmax=10,
#                   ymin=0.8, ymax=1.1, log_scale=False)

  s_bump_lin = sp.interpolate.interp1d(ensp['bump_corr'].GetBinCenters(), ensp['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['bump_corr'].bin_cont[0], ensp['bump_corr'].bin_cont[-1]))
  del ensp['bump_corr']
  xlin = np.linspace(0.8, 12., 561)
  #xlin = np.linspace(1.5, 15., 2700)
  xlin_c = 0.5*(xlin[:-1]+xlin[1:])
  ylin = s_bump_lin(xlin_c)
  with open(f'{args.data_matrix_folder}/csv_{args.stat_opt}/s_bump_lin.csv', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(zip(xlin_c,ylin))
  f.close()
  ensp['rfis_b'] = ensp['rfis0'].GetWeightedWithFunction(s_bump_lin)
  del s_bump_lin
  #
  # Complete IBD flux
  ensp['rfis'] = ensp['rfis_b'].Copy()
  del ensp['rfis_b']
  ensp['rfis'].GetScaled(ndays)
  alpha_arr = np.array(args.alpha)
  efission_arr = np.array(args.efission)
  Pth_arr = np.array(args.Pth)
  L_arr = np.array(args.L)
  extrafactors = args.detector_efficiency*args.veto*args.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr/(L_arr*L_arr))
  print("extrafactors", extrafactors)
  ensp['rfis'].GetScaled(extrafactors) #correct normalization including fission fractions, mean energy per fission ... eq.13.5 YB
  print(" ")
  print("NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral()))
  print(" ")
  print("2-6 MeV NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral(left_edge = 2, right_edge = 6)))
  #
  ensp['snf0'] = GetSpectrumFromROOT(args.input_data_file, 'SNF_FluxRatio')     #spent nuclear fuel
  ensp['snf0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['snf'] = ensp['snf0'].GetWeightedWithSpectrum(ensp['rfis'])
  del ensp['snf0']

  ensp['rfis0_geou'].GetScaled(ndays*args.geo_rate/((1+args.Th_U_ratio)*args.duty_cycle*ensp['rfis0_geou'].GetIntegral()))
  ensp['rfis0_geoth'].GetScaled(ndays*args.geo_rate*args.Th_U_ratio/((1+args.Th_U_ratio)*args.duty_cycle*ensp['rfis0_geoth'].GetIntegral()))

  ensp['noneq0'] = GetSpectrumFromROOT(args.input_data_file, 'NonEq_FluxRatio')
  ensp['noneq0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['noneq'] = ensp['noneq0'].GetWeightedWithSpectrum(ensp['rfis']) #non-equilibrium ratio
  del ensp['noneq0']
  ensp['ribd'] = ensp['rfis'] + ensp['snf'] + ensp['noneq']
  events['ribd'] = ensp["ribd"].GetIntegral()
#  ensp['rfis0'].Plot(f"{args.plots_folder}/reac_spectrum.pdf",
#                   xlabel="Neutrino energy (MeV)",
#                   ylabel=f"Events per {binw:0.1f} keV",
#                   extra_spectra=[ensp['rfis'], ensp['ribd'], ensp['snf'], ensp['noneq']],
#                   leg_labels=['From HM', '+ bump fix', ' + SNF + NonEq', 'SNF', 'NonEq'],
#                   colors=['black', 'red', 'blue', 'green', 'magenta'],
#                   xmin=0, xmax=10,
#                   ymin=0, ymax=None, log_scale=False)
  del ensp['rfis0'], ensp['rfis']
  print(" ")
  print("NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral()))
  print(" ")
  print("2-6 MeV NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral(left_edge = 2, right_edge = 6)))

  # Oscillated spectrum
  ensp['rosc'] = ensp['ribd'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)

  events['rosc'] = ensp["rosc"].GetIntegral()

  #ensp['ribd'].Plot(f"{args.plots_folder}/osc_spectrum.pdf",
  #                 xlabel="Neutrino energy (MeV)",
  #                 ylabel=f"Events per {binw:0.1f} keV",
  #                 extra_spectra=[ensp['rosc']],
  #                 leg_labels=['Reactor', 'Oscillated'],
  #                 colors=['black', 'darkred'],
  #                 xmin=0, xmax=10,
  #                 ymin=0, ymax=None, log_scale=False)

  ensp['rosc_nome'] = ensp['ribd'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=0.0, ene_mode='true', args=args)


  #ensp['rosc'].Plot(f"{args.plots_folder}/osc_spectrum_mecomp.pdf",
  #                 xlabel="Neutrino energy (MeV)",
  #                 ylabel=f"Events per {binw:0.1f} keV",
  #                 extra_spectra=[ensp['rosc_nome']],
  #                 leg_labels=['With ME', 'Without ME'],
  #                 colors=['black', 'darkred'],
  #                 xmin=0, xmax=10,
  #                 ymin=0, ymax=None, log_scale=False)
  del ensp['rosc_nome']

  ensp['snf_osc'] = ensp['snf'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
  del ensp['snf']
  ensp['noneq_osc'] = ensp['noneq'].GetOscillated(L=args.core_baselines, core_powers=args.core_powers, me_rho=args.me_rho, ene_mode='true', args=args)
  del ensp['noneq']



  ensp['rvis_nonl'] = ensp['rosc'].GetWithPositronEnergy()
  ensp['rvis_nonl_temp'] = ensp['ribd'].GetWithPositronEnergy()

  ensp['rvis_geou'] = ensp['rfis0_geou'].GetWithPositronEnergy()
  ensp['rvis_geoth'] = ensp['rfis0_geoth'].GetWithPositronEnergy()

  ensp['snf_osc_nonl'] = ensp['snf_osc'].GetWithPositronEnergy()
  del ensp['snf_osc']
  ensp['noneq_osc_nonl'] = ensp['noneq_osc'].GetWithPositronEnergy()
  del ensp['noneq_osc']

  if detector=="tao":
      ensp['rosc_pos'] = ensp['rosc'].GetWithPositronEnergy() #only for plotting
      ensp['rosc_eneleak'] = ensp['rvis_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop) #only for plotting

      ensp['rvis_nonl'] = ensp['rvis_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop)
      ensp['snf_osc_nonl'] = ensp['snf_osc_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop)
      ensp['noneq_osc_nonl'] = ensp['noneq_osc_nonl'].ApplyDetResp(ene_leak_tao, pecrop=args.ene_crop)

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
  ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])

  ensp['rvis_geou_nonl'] = ensp['rvis_geou'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['rvis_geoth_nonl'] = ensp['rvis_geoth'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])

  ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
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


  ensp['rdet'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['geou_ana'] = ensp['rvis_geou_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
  ensp['geoth_ana'] = ensp['rvis_geoth_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['geo_ana'] = ensp['geou_ana'] + ensp['geoth_ana']

  ensp['snf_final'] = ensp['snf_osc_vis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
  ensp['noneq_final'] = ensp['noneq_osc_vis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['rdet_noenecrop'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop2)

  del ensp['snf_osc_vis'], ensp['noneq_osc_vis']

  events['rdet'] = ensp['rdet'].GetIntegral()
  events['rdet_noenecrop'] = ensp["rdet_noenecrop"].GetIntegral()

  print ("Backgrounds")
  bg_labels = ['AccBkgHistogramAD', 'FnBkgHistogramAD', 'Li9BkgHistogramAD', 'AlphaNBkgHistogramAD',  'GeoNuHistogramAD', 'GeoNuTh232', 'GeoNuU238', 'AtmosphericNeutrinoModelGENIE2', 'OtherReactorSpectrum_L300km']
  bg_keys = ['acc', 'fneu', 'lihe', 'aneu', 'geo', 'geoth', 'geou', 'atm', 'rea300']

  for key, label in zip(bg_keys, bg_labels):
    ensp[key] = GetSpectrumFromROOT(args.input_data_file, label)
    ensp[key].GetScaled(ndays/args.duty_cycle)
    ensp[key].Trim(args.ene_crop)
    if(ensp[key].bins[1] - ensp[key].bins[0] != ebins[1]-ebins[0]):
        print("different bins, rebinning ", key)
        ensp[key].Rebin(ebins, mode='spline-not-keep-norm')

  if detector == "tao":
      ensp['acc'].GetScaled(args.acc_scale)
      ensp['lihe'].GetScaled(args.lihe_scale)
      ensp['fneu'] = MakeTAOFastNSpectrum(bins=ebins, A=args.fneu_A, B=args.fneu_B, C=args.fneu_C)
      ensp['fneu'].GetScaled(args.fneu_rate*ndays/args.duty_cycle)

  #

  bg_keys2 = ['acc_noenecrop', 'fneu_noenecrop', 'lihe_noenecrop', 'aneu_noenecrop', 'geo_noenecrop', 'geoth_noenecrop', 'geou_noenecrop', 'atm_noenecrop', 'rea300_noenecrop']

  for key2, label in zip(bg_keys2, bg_labels):
    ensp[key2] = GetSpectrumFromROOT(args.input_data_file, label)
    ensp[key2].GetScaled(ndays*12/11)
    ensp[key2].Trim(args.ene_crop2)

  #del cm['acc'], cm['geo'], cm['lihe'], cm['fneu'], cm['aneu']

  if args.geo_ana_spectra:
      ensp['geo'] = ensp['geo_ana']
      ensp['geou'] = ensp['geou_ana']
      ensp['geoth'] = ensp['geoth_ana']

  if detector == "juno":
      if args.geo_uthfree: ensp['geo'] = ensp['geou'] + ensp['geoth']
      ensp['bckg'] = ensp['acc'] + ensp['fneu'] + ensp['lihe'] + ensp['aneu'] + ensp['geo'] + ensp['atm'] + ensp['rea300']
      ensp['bckg_noenecrop'] = ensp['acc_noenecrop'] + ensp['fneu_noenecrop'] + ensp['lihe_noenecrop'] + ensp['aneu_noenecrop'] + ensp['geo_noenecrop'] + ensp['atm_noenecrop'] + ensp['rea300_noenecrop']
      extra_spectra=[ensp['acc'], ensp['geo'], ensp['lihe'], ensp['fneu'], ensp['aneu'], ensp['atm'], ensp['rea300']]
      leg_labels = ['Reactor', 'Accidentals', 'Geoneutrinos', 'Li9/He8','Fast neutrons', '(alpha, n)', 'Atmospheric', 'Reactors > 300 km']
      colors=['darkred', 'green', 'navy', 'orange', 'magenta', 'lightblue', 'yellow', 'brown']
        #
      ensp['rtot_noenecrop'] = ensp['rdet_noenecrop'] + ensp['bckg_noenecrop']
      events['rbckg_noenecrop'] = ensp['bckg_noenecrop'].GetIntegral()
      events['rtot_noenecrop'] = ensp['rtot_noenecrop'].GetIntegral()
      events['acc_noenecrop'] = ensp['acc_noenecrop'].GetIntegral()
      events['fneu_noenecrop'] = ensp['fneu_noenecrop'].GetIntegral()
      events['lihe_noenecrop'] = ensp['lihe_noenecrop'].GetIntegral()
      events['aneu_noenecrop'] = ensp['aneu_noenecrop'].GetIntegral()
      events['geo_noenecrop'] = ensp['geo_noenecrop'].GetIntegral()

  if detector == "tao":
      ensp['bckg'] = ensp['acc'] + ensp['fneu'] + ensp['lihe']
      extra_spectra=[ensp['rosc_pos'], ensp['rosc_eneleak'], ensp['acc'], ensp['lihe'], ensp['fneu']]
      leg_labels = ['+ NL + energy res', '+ flux and cross-sec', '+ energy leak', 'Accidentals', 'Li9/He8', 'Fast neutrons']
      colors=['grey', 'magenta', 'darkred', 'green', 'navy', 'orange']

  ensp['rtot'] = ensp['rdet'] + ensp['bckg']

  events['rbckg'] = ensp['bckg'].GetIntegral()
  events['rtot'] = ensp['rtot'].GetIntegral()
  events['acc'] = ensp['acc'].GetIntegral()
  events['fneu'] = ensp['fneu'].GetIntegral()
  events['lihe'] = ensp['lihe'].GetIntegral()
  events['aneu'] = ensp['aneu'].GetIntegral()
  events['geo'] = ensp['geo'].GetIntegral()
  events['atm'] = ensp['atm'].GetIntegral()
  events['rea300'] = ensp['rea300'].GetIntegral()


  if args.plot_spectra:
      ensp['rdet'].Plot(f"{args.plots_folder}/det_spectra_{detector}.png",
                  xlabel="Reconstructed energy (MeV)",
                  ylabel=f"Events per {binw:0.1f} keV",
                  extra_spectra=extra_spectra,
                  leg_labels=leg_labels,
                  colors=colors,
                  xmin=0.8, xmax=9.0,
                  ymin=1e-3,log_scale=False)
                  #ymin=0.0, ymax=14900, yinterval=2500, log_scale=False)





#  resp_matrix.Plot(f"{args.plots_folder}/resp_mat.pdf")

  #
  print(" ")
  print(" ")
  print("   Reac x IBD:    {:.2f} events".format(events['ribd']))
  print("   Oscillated:    {:.2f} events".format(events['rosc']))
  print("   Detected:      {:.2f} events".format(events['rdet']))
  print("   IBD+BG:        {:.2f} events".format(events["rtot"]))

  print("   Backgrounds")
  print("   Total:        {:.2f} events".format(events["rbckg"]))
  print("      accidentals:    {:.2f} events".format(events["acc"]))
  print("      Li-9/He-8:      {:.2f} events".format(events["lihe"]))
  print("      fast n:         {:.2f} events".format(events["fneu"]))

  if detector == "juno":

    print("      (alpha,n):      {:.2f} events".format(events["aneu"]))
    print("      geo-nu:         {:.2f} events".format(events["geo"]))
    print("      atmospheric:         {:.2f} events".format(events["atm"]))
    print("      global reactors:         {:.2f} events".format(events["rea300"]))
    print("   Det NoEneC:    {:.2f} events".format(events['rdet_noenecrop']))
    print("   IBD+BG NoEneC: {:.2f} events".format(events["rtot_noenecrop"]))
    print("   BG NoEneC: {:.2f} events".format(events["rbckg_noenecrop"]))
    print("      accidentals NoEneC:    {:.2f} events".format(events["acc_noenecrop"]))
    print("      geo-nu NoEneC:         {:.2f} events".format(events["geo_noenecrop"]))
    print("      Li-9/He-8 NoEneC:      {:.2f} events".format(events["lihe_noenecrop"]))
    print("      fast n NoEneC:         {:.2f} events".format(events["fneu_noenecrop"]))
    print("      (alpha,n) NoEneC:      {:.2f} events".format(events["aneu_noenecrop"]))

  del events


  print(" # Initialization completed")
    #
  print(" # Dumping data")
  for key in ensp.keys():
      if type(ensp[key]) != list:
          ensp[key].Dump(f"{args.data_matrix_folder}/csv_{args.stat_opt}/ensp_{detector}_{key}.csv")


  #ensp["rtot_toy"] = GetSpectrumFromROOT(f"../fake_data/toy{args.ntoy}/fake_data_{args.ntoy}.root", "data_0")

  return ensp
