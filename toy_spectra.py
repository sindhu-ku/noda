import sys, os
from noda import *
import csv
from datetime import datetime
from joblib import Parallel, delayed

from scipy import constants


def Initialize( ndays=10,
               core_baselines=[],
               core_powers=[],
               ebins=None,
               args=""):


  opt = {'ndays': ndays, "me_rho": args.me_rho,
         'core_baselines': core_baselines, 'core_powers': core_powers,
         'ebins': ebins }  # for output



  #cm_suffix += "_{:}bins".format(int(0.5*(len(ebins)-1)))
  #
  #
  # Nominal spectra
  ensp = {}   # energy spectra

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
  # IBD xsection
  print("# Applying IBD cross-section")
  ensp['sibd'] = GetSpectrumFromROOT(args.input_data_file, 'IBDXsec_VogelBeacom_DYB')
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
  print("# Applying DYB 5 MeV bump")
  ensp['bump_corr'] = GetSpectrumFromROOT(args.input_data_file, 'DYBFluxBump_ratio')
  s_bump_lin = sp.interpolate.interp1d(ensp['bump_corr'].GetBinCenters(), ensp['bump_corr'].bin_cont, kind='slinear', bounds_error=False, fill_value=(ensp['bump_corr'].bin_cont[0], ensp['bump_corr'].bin_cont[-1]))
  del ensp['bump_corr']
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
  extrafactors = args.detector_efficiency*args.Np/(4*np.pi)*1./(np.sum(alpha_arr*efission_arr))*np.sum(Pth_arr/(L_arr*L_arr))
  print("# Extrafactors for scaling the spectra", extrafactors)
  ensp['rfis'].GetScaled(extrafactors) #correct normalization including fission fractions, mean energy per fission ... eq.13.5 YB

  #
  print("# Applying SNF and non-equilibrium corrections")
  ensp['snf0'] = GetSpectrumFromROOT(args.input_data_file, 'SNF_FluxRatio')     #spent nuclear fuel
  ensp['snf0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['snf'] = ensp['snf0'].GetWeightedWithSpectrum(ensp['rfis'])
  del ensp['snf0']

  ensp['noneq0'] = GetSpectrumFromROOT(args.input_data_file, 'NonEq_FluxRatio')
  ensp['noneq0'].Rebin(ebins, mode='spline-not-keep-norm')
  ensp['noneq'] = ensp['noneq0'].GetWeightedWithSpectrum(ensp['rfis']) #non-equilibrium ratio
  del ensp['noneq0']
  ensp['ribd'] = ensp['rfis'] + ensp['snf'] + ensp['noneq']


  print("# Applying Oscillations")
  # Oscillated spectrum
  ensp['rosc'] = ensp['ribd'].GetOscillated(L=core_baselines, core_powers=core_powers, me_rho=args.me_rho, ene_mode='true', args=args)

  #   Non-linearity
  ensp['rvis_nonl'] = ensp['rosc'].GetWithPositronEnergy()


  ensp['scintNL'] = GetSpectrumFromROOT(args.input_data_file, 'positronScintNL')

  print("# Applying non-linearity")
  ensp['rvis'] = ensp['rvis_nonl'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])


  #   Energy resolution
  print("# Applying energy resolution")
  a, b, c = args.a, args.b, args.c
  a_err, b_err, c_err =args.a_err, args.b_err, args.c_err
  ebins = ensp['ribd'].bins

  if os.path.isfile(f"{args.data_matrix_folder}/rm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat") and not args.FORCE_CALC_RM:
    resp_matrix = LoadRespMatrix(f"{args.data_matrix_folder}/rm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
  else:
    resp_matrix = CalcRespMatrix_abc(a, b, c, escale=1, ebins=ebins, pebins=ebins)
  #  resp_matrix.Save(f"{args.data_matrix_folder}/rm_{args.bayes_chi2}_{args.sin2_th13_opt}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins.dat")
  ensp['rdet'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  ensp['rdet_noenecrop'] = ensp['rvis'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop2)
    #for delta_chi2 calc
  ensp['unosc_pos'] = ensp['ribd'].GetWithPositronEnergy()
  ensp['unosc_nonl'] = ensp['unosc_pos'].GetWithModifiedEnergy(mode='spectrum', spectrum=ensp['scintNL'])
  ensp['unosc'] = ensp['unosc_nonl'].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

  print ("Creating background spectra")
  bg_labels = ['AccBkgHistogramAD', 'FnBkgHistogramAD', 'Li9BkgHistogramAD', 'AlphaNBkgHistogramAD', 'GeoNuHistogramAD', 'GeoNuTh232', 'GeoNuU238', 'AtmosphericNeutrinoModelGENIE2', 'OtherReactorSpectrum_L300km']
  bg_keys = ['acc', 'fneu', 'lihe', 'aneu', 'geo', 'geoth', 'geou', 'atm', 'rea300']
  for key, label in zip(bg_keys, bg_labels):
    ensp[key] = GetSpectrumFromROOT(args.input_data_file, label)
    ensp[key].GetScaled(ndays*12/11)
    ensp[key].Trim(args.ene_crop)
    if(ensp[key].bins[1] - ensp[key].bins[0] != ebins[1]-ebins[0]):
        print("different bins, rebinning ", key)
        ensp[key].Rebin(ebins, mode='spline-not-keep-norm')




  ensp['rtot'] = ensp['rdet'] + ensp['acc'] + ensp['fneu'] + ensp['lihe'] + ensp['aneu'] + ensp['geo'] + ensp['atm'] + ensp['rea300']

  #
  #
  if args.FORCE_CALC_RM:
      resp_matrix.Dump(f"{args.data_matrix_folder}/csv/resp_matrix.csv")
      print("Finished first loop")
      #
    #

  print(" # Initialization completed")
    #
  print(" # Dumping data")
  for key in ensp.keys():
      if type(ensp[key]) != list:
          ensp[key].Dump(f"{args.data_matrix_folder}/csv/ensp_{key}.csv")




  return ensp, resp_matrix, opt
