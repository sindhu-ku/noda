import sys, os
from noda import *
import csv
from datetime import datetime
from joblib import Parallel, delayed

from scipy import constants


def Initialize( ndays=10,
               reac_model="",
               core_baselines=[],
               core_powers=[],
               me_rho=0.8,
               ebins=None, ene_crop = (),
               ene_crop2 =(),
               pmt_opt="",
               data_file = "",
               FORCE_CALC_RM=False,
               plots_folder="",
               data_matrix_folder="", args=""):


  opt = {'ndays': ndays, 'pmt_opt': pmt_opt, "me_rho": me_rho,
         'core_baselines': core_baselines, 'core_powers': core_powers,
         'ebins': ebins }  # for output



  #cm_suffix += "_{:}bins".format(int(0.5*(len(ebins)-1)))
  #
  #
  # Nominal spectra
  ensp = {}   # energy spectra
  events = {} #events in each spectrum
  rebin_mode = 'spline'
  if not os.path.exists(plots_folder):
      os.makedirs(plots_folder)
  binw = (ebins[1]-ebins[0])*1000
  print(" # Constructing nominal spectra")
  #   reactor x IBD
  #     From ROOT file
  ensp['rfis0'] = GetSpectrumFromROOT(data_file, 'HuberMuellerFlux_U235', scale=args.U235_scale) + \
                 GetSpectrumFromROOT(data_file, 'HuberMuellerFlux_U238', scale=args.U238_scale) + \
                 GetSpectrumFromROOT(data_file, 'HuberMuellerFlux_Pu239', scale=args.Pu239_scale) + \
                 GetSpectrumFromROOT(data_file, 'HuberMuellerFlux_Pu241', scale=args.Pu241_scale)
  # IBD xsection
  ensp['sibd'] = GetSpectrumFromROOT(data_file, 'IBDXsec_VogelBeacom_DYB')
  s_ibd = sp.interpolate.interp1d(ensp['sibd'].GetBinCenters(), ensp['sibd'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['sibd'].bin_cont[0], ensp['sibd'].bin_cont[-1]))
  #why s_ibd rebinned separately?
  del ensp['sibd']
  ensp['rfis0'].WeightWithFunction(s_ibd)
  del s_ibd
  ensp['rfis0'].Rebin(ebins, mode='spline-not-keep-norm') #I don't understand rebinning and scaling here
  bin_width = ensp['rfis0'].GetBinWidth()
  ensp['rfis0'].GetScaled(bin_width)


  #  DYB Bump
  #  Previous spectrum (Oct2020) + extra bins + interpolation lin + getweightedwithfunction
  ensp['bump_corr'] = GetSpectrumFromROOT(data_file, 'DYBFluxBump_ratio')
#  bins_new_bump = [1.799]+list(ensp['bump_corr'].bins)+[11.999,12]
#  bin_cont_new_bump = [ensp['bump_corr'].bin_cont[0]]+list(ensp['bump_corr'].bin_cont)+[ensp['bump_corr'].bin_cont[-1]]*2
 # ensp['bump_corr'].bins = np.array(bins_new_bump)
 # ensp['bump_corr'].bin_cont = np.array(bin_cont_new_bump)
  ensp['bump_corr'].Plot(f"{plots_folder}/bump_correction.pdf",
                   xlabel="Neutrino energy (MeV)",
                   xmin=0, xmax=10,
                   ymin=0.8, ymax=1.1, log_scale=False)

  s_bump_lin = sp.interpolate.interp1d(ensp['bump_corr'].GetBinCenters(), ensp['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['bump_corr'].bin_cont[0], ensp['bump_corr'].bin_cont[-1]))
  del ensp['bump_corr']
  xlin = np.linspace(0.8, 12., 561)
  #xlin = np.linspace(1.5, 15., 2700)
  xlin_c = 0.5*(xlin[:-1]+xlin[1:])
  ylin = s_bump_lin(xlin_c)
  with open(f'{data_matrix_folder}/csv/s_bump_lin.csv', 'w') as f:
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
  Pth2 = [Pth*6.24e21*60*60*24 for Pth in args.Pth] # MeV/day
  L2 = [L*1e2 for L in args.L]
  alpha_efission = [args.alpha[i]*args.efission[i] for i in range(len(args.alpha))]
  Pth_L2 = [Pth2[i]/(L2[i]*L2[i]) for i in range(len(Pth2))]
  extrafactors = args.detector_efficiency*args.Np/(4*np.pi)*1./(np.sum(alpha_efission))*np.sum(Pth_L2)
  ensp['rfis'].GetScaled(extrafactors) #correct normalization including fission fractions, mean energy per fission ... eq.13.5 YB
  print(" ")
  print("NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral()))
  print(" ")
  print("2-6 MeV NUMBER OF IBD")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["rfis"].GetIntegral(left_edge = 2, right_edge = 6)))
  #
  ensp['snf0'] = GetSpectrumFromROOT(data_file, 'SNF_FluxRatio')     #spent nuclear fuel
  ensp['snf0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['snf'] = ensp['snf0'].GetWeightedWithSpectrum(ensp['rfis'])
  del ensp['snf0']

  ensp['noneq0'] = GetSpectrumFromROOT(data_file, 'NonEq_FluxRatio')
  ensp['noneq0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['noneq'] = ensp['noneq0'].GetWeightedWithSpectrum(ensp['rfis']) #non-equilibrium ratio
  del ensp['noneq0']

  ensp['ribd'] = ensp['rfis'] + ensp['snf'] + ensp['noneq']
  events['ribd'] = ensp["ribd"].GetIntegral()
  ensp['rfis0'].Plot(f"{plots_folder}/reac_spectrum.pdf",
                   xlabel="Neutrino energy (MeV)",
                   ylabel=f"Events per {binw:0.1f} keV",
                   extra_spectra=[ensp['rfis'], ensp['ribd'], ensp['snf'], ensp['noneq']],
                   leg_labels=['From HM', '+ bump fix', ' + SNF + NonEq', 'SNF', 'NonEq'],
                   colors=['black', 'red', 'blue', 'green', 'magenta'],
                   xmin=0, xmax=10,
                   ymin=0, ymax=None, log_scale=False)
  del ensp['rfis0'], ensp['rfis']
  print(" ")
  print("NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral()))
  print(" ")
  print("2-6 MeV NUMBER OF IBD + SNF + nEq")
  print(" Expected IBDs (no osc):     {:.2f} events".format(ensp["ribd"].GetIntegral(left_edge = 2, right_edge = 6)))

  # Oscillated spectrum
  ensp['rosc'] = ensp['ribd'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=me_rho, ene_mode='true')

  events['rosc'] = ensp["rosc"].GetIntegral()

  ensp['ribd'].Plot(f"{plots_folder}/osc_spectrum.pdf",
                   xlabel="Neutrino energy (MeV)",
                   ylabel=f"Events per {binw:0.1f} keV",
                   extra_spectra=[ensp['rosc']],
                   leg_labels=['Reactor', 'Oscillated'],
                   colors=['black', 'darkred'],
                   xmin=0, xmax=10,
                   ymin=0, ymax=None, log_scale=False)

  ensp['rosc_nome'] = ensp['ribd'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=0.0, ene_mode='true')


  ensp['rosc'].Plot(f"{plots_folder}/osc_spectrum_mecomp.pdf",
                   xlabel="Neutrino energy (MeV)",
                   ylabel=f"Events per {binw:0.1f} keV",
                   extra_spectra=[ensp['rosc_nome']],
                   leg_labels=['With ME', 'Without ME'],
                   colors=['black', 'darkred'],
                   xmin=0, xmax=10,
                   ymin=0, ymax=None, log_scale=False)
  del ensp['rosc_nome']

  ensp['snf_osc'] = ensp['snf'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=me_rho, ene_mode='true')
  del ensp['snf']
  ensp['noneq_osc'] = ensp['noneq'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=me_rho, ene_mode='true')
  del ensp['noneq']
  #   Non-linearity
  ensp['rvis_nonl'] = ensp['rosc'].GetWithPositronEnergy()
  ensp['rvis_nonl_temp'] = ensp['ribd'].GetWithPositronEnergy()
  del ensp['rosc']



  ensp['snf_osc_nonl'] = ensp['snf_osc'].GetWithPositronEnergy()
  del ensp['snf_osc']
  ensp['noneq_osc_nonl'] = ensp['noneq_osc'].GetWithPositronEnergy()
  del ensp['noneq_osc']
  ensp['scintNL'] = GetSpectrumFromROOT(data_file, 'positronScintNL')
  ensp['NL_pull'] = [ GetSpectrumFromROOT(data_file, 'positronScintNLpull0'),
                      GetSpectrumFromROOT(data_file, 'positronScintNLpull1'),
                      GetSpectrumFromROOT(data_file, 'positronScintNLpull2'),
                      GetSpectrumFromROOT(data_file, 'positronScintNLpull3') ]
  print("applying non-linearity")
  ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['rvis_temp'] = ensp['rvis_nonl_temp'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['snf_osc_vis'] = ensp['snf_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['noneq_osc_vis'] = ensp['noneq_osc_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  del ensp['snf_osc_nonl'], ensp['noneq_osc_nonl']
  print("length of NL Pull:")
  print (len(ensp['NL_pull']))
  ensp['rvis_nonl'].Plot(f"{plots_folder}/vis_spectra.pdf",
                   xlabel="Visual energy (MeV)",
                   ylabel=f"Events per {binw:0.1f} keV",
                   extra_spectra=[ensp['rvis']],
                   leg_labels=['Before NL', 'After NL'],
                   colors=['darkred', 'green'],
                   xmin=0, xmax=10,
                   ymin=0, ymax=None, log_scale=False)

  ensp['scintNL'].Plot(f"{plots_folder}/non_linearity.pdf",
                   xlabel="Reconstructed energy (MeV)",
                   extra_spectra=ensp['NL_pull'],
                   xmin=0, xmax=10,
                   ymin=0.9, ymax=1.1, log_scale=False)

  #   Energy resolution
  a, b, c = args.a, args.b, args.c
  a_err, b_err, c_err =args.a_err, args.b_err, args.c_err
  ebins = ensp['ribd'].bins

  if os.path.isfile(f"{data_matrix_folder}/rm_{pmt_opt}_{ndays:.0f}days.dat") and not FORCE_CALC_RM:
    resp_matrix = LoadRespMatrix(f"{data_matrix_folder}/rm_{pmt_opt}_{ndays:.0f}days.dat")
  else:
    resp_matrix = CalcRespMatrix_abc(a, b, c, escale=1, ebins=ebins, pebins=ebins)
    resp_matrix.Save(f"{data_matrix_folder}/rm_{pmt_opt}_{ndays:.0f}days.dat")
  ensp['rdet'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=ene_crop)
  ensp['rdet_temp'] = ensp['rvis_temp'].ApplyDetResp(resp_matrix, pecrop=ene_crop)
  events['rdet'] = ensp['rdet'].GetIntegral()

  ensp['snf_final'] = ensp['snf_osc_vis'].ApplyDetResp(resp_matrix, pecrop=ene_crop)
  ensp['noneq_final'] = ensp['noneq_osc_vis'].ApplyDetResp(resp_matrix, pecrop=ene_crop)

  ensp['rdet_noenecrop'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=ene_crop2)
  events['rdet_noenecrop'] = ensp["rdet_noenecrop"].GetIntegral()
  del ensp['snf_osc_vis'], ensp['noneq_osc_vis']


  #file_ibd = open("IBD_spec_noosc.txt", "a")
  #for i in range(0, len(ensp['rdet_temp'].bin_cont)):
  ##    file_ibd.write(str(ensp['rdet_temp'].bins[i])+' '+str(ensp['rdet_temp'].bin_cont[i])+'\n')
  #file_ibd.close()
  #file_osc = open("IBD_spec_osc_IH.txt", "a")
  #for i in range(0, len(ensp['rdet'].bin_cont)):
  #    file_osc.write(str(ensp['rdet'].bins[i])+' '+str(ensp['rdet'].bin_cont[i])+'\n')
  #file_osc.close()
 # print("B2B TAO spectrum")
 # ensp['b2b_tao'] = GetSpectrumFromROOT(data_file, "TAOUncertainty")
 # if(ensp['b2b_tao'].bins[1] - ensp['b2b_tao'].bins[0] != ebins[1]-ebins[0]):
 #     print("different bins, rebinning B2B_TAO")
 #     new_bins = int(1 + ((ensp['b2b_tao'].bins[1] - ensp['b2b_tao'].bins[0])*(len(ensp['b2b_tao'].bins) -1)/(ebins[1]-ebins[0])))
 #     print("new bins: ", new_bins)
 #     ensp['b2b_tao'].Rebin(ebins, mode='spline-not-keep-norm')

  print("textfiles done")
  #   backgrounds
  print ("Backgrounds")
  bg_labels = ['AccBkgHistogramAD', 'FnBkgHistogramAD', 'Li9BkgHistogramAD', 'AlphaNBkgHistogramAD', 'GeoNuHistogramAD', 'GeoNuTh232', 'GeoNuU238', 'AtmosphericNeutrinoModelGENIE2', 'OtherReactorSpectrum_L300km']
  bg_keys = ['acc', 'fneu', 'lihe', 'aneu', 'geo', 'geoth', 'geou', 'atm', 'rea300']
  for key, label in zip(bg_keys, bg_labels):
    ensp[key] = GetSpectrumFromROOT(data_file, label)
    ensp[key].GetScaled(ndays*12/11)
    ensp[key].Trim(ene_crop)
    if(ensp[key].bins[1] - ensp[key].bins[0] != ebins[1]-ebins[0]):
        print("different bins, rebinning ", key)
        ensp[key].Rebin(ebins, mode='spline-not-keep-norm')


  bg_keys2 = ['acc_noenecrop', 'fneu_noenecrop', 'lihe_noenecrop', 'aneu_noenecrop', 'geo_noenecrop', 'geoth_noenecrop', 'geou_noenecrop']
  for key2, label in zip(bg_keys2, bg_labels):
    ensp[key2] = GetSpectrumFromROOT(data_file, label)
    ensp[key2].GetScaled(ndays*12/11)
    ensp[key2].Trim(ene_crop2)


  #del cm['acc'], cm['geo'], cm['lihe'], cm['fneu'], cm['aneu']

  ensp['rtot'] = ensp['rdet'] + ensp['acc'] + ensp['fneu'] + ensp['lihe'] + ensp['aneu'] + ensp['geo'] + ensp['atm'] + ensp['rea300']
  events['rtot'] = ensp['rtot'].GetIntegral()
  events['acc'] = ensp['acc'].GetIntegral()
  events['fneu'] = ensp['fneu'].GetIntegral()
  events['lihe'] = ensp['lihe'].GetIntegral()
  events['aneu'] = ensp['aneu'].GetIntegral()
  events['geo'] = ensp['geo'].GetIntegral()
  ensp['rdet'].Plot(f"{plots_folder}/det_spectra.pdf",
                   xlabel="Reconstructed energy (MeV)",
                   ylabel=f"Events per {binw:0.1f} keV",
                   extra_spectra=[ensp['acc'], ensp['geo'], ensp['lihe'],
                                  ensp['fneu'], ensp['aneu'], ensp['atm'], ensp['rea300']],
                   leg_labels=['Reactor', 'Accidentals', 'Geo-neutrino', 'Li9/He8',
                               'Fast neutrons', '(alpha, n)', 'Atmospheric', 'Reactors > 300 km'],
                   colors=['darkred', 'green', 'navy', 'orange', 'magenta', 'lightblue', 'yellow', 'brown'],
                   xmin=0, xmax=10,
                   ymin=1e-3, ymax=None, log_scale=True)


  ensp['rtot_noenecrop'] = ensp['rdet_noenecrop'] + ensp['acc_noenecrop'] + ensp['fneu_noenecrop'] + ensp['lihe_noenecrop'] + ensp['aneu_noenecrop'] + ensp['geo_noenecrop']
  events['rtot_noenecrop'] = ensp['rtot_noenecrop'].GetIntegral()
  events['acc_noenecrop'] = ensp['acc_noenecrop'].GetIntegral()
  events['fneu_noenecrop'] = ensp['fneu_noenecrop'].GetIntegral()
  events['lihe_noenecrop'] = ensp['lihe_noenecrop'].GetIntegral()
  events['aneu_noenecrop'] = ensp['aneu_noenecrop'].GetIntegral()
  events['geo_noenecrop'] = ensp['geo_noenecrop'].GetIntegral()



  resp_matrix.Plot(f"{plots_folder}/resp_mat.pdf")

  #
  print(" ")
  print(" ")
  print("   Reac x IBD:    {:.2f} events".format(events['ribd']))
  print("   Oscillated:    {:.2f} events".format(events['rosc']))
  print("   Detected:      {:.2f} events".format(events['rdet']))
  print("   Det NoEneC:    {:.2f} events".format(events['rdet_noenecrop']))
  print("   IBD+BG:        {:.2f} events".format(events["rtot"]))
  print("   IBD+BG NoEneC: {:.2f} events".format(events["rtot_noenecrop"]))
  print("   Backgrounds")
  print("      accidentals:    {:.2f} events".format(events["acc"]))
  print("      geo-nu:         {:.2f} events".format(events["geo"]))
  print("      Li-9/He-8:      {:.2f} events".format(events["lihe"]))
  print("      fast n:         {:.2f} events".format(events["fneu"]))
  print("      (alpha,n):      {:.2f} events".format(events["aneu"]))
  print("      accidentals NoEneC:    {:.2f} events".format(events["acc_noenecrop"]))
  print("      geo-nu NoEneC:         {:.2f} events".format(events["geo_noenecrop"]))
  print("      Li-9/He-8 NoEneC:      {:.2f} events".format(events["lihe_noenecrop"]))
  print("      fast n NoEneC:         {:.2f} events".format(events["fneu_noenecrop"]))
  print("      (alpha,n) NoEneC:      {:.2f} events".format(events["aneu_noenecrop"]))

  del events
  #
  #
  if FORCE_CALC_RM:
      resp_matrix.Dump(f"{data_matrix_folder}/csv/resp_matrix.csv")
      print("Finished first loop")
      #
    #

  print(" # Initialization completed")
    #
  print(" # Dumping data")
  for key in ensp.keys():
      if type(ensp[key]) != list:
          ensp[key].Dump(f"{data_matrix_folder}/csv/ensp_{key}.csv")




  return ensp, resp_matrix, opt
