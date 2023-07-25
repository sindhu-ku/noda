import numpy as np
import matplotlib.pyplot as plt
import sys, os
from chi2 import *


def get_results(args=''):
  axis_labels = {'sin2_12': r"$\sin^2 \theta_{12}$",
               'sin2_13': r"$\sin^2 \theta_{13}$",
                'dm2_21':    r"$\Delta m^2_{21} (10^{-5} {\rm eV}^2)$",
                'dm2_31':    r"$\Delta m^2_{31} (10^{-3} {\rm eV}^2)$",
                'dm2_ee':    r"$\Delta m^2_{ee} (10^{-3} {\rm eV}^2)$"}
  chi2maps, grid = np.load(f"../Data/chi2maps_{args.stat_method_opt}_{args.sin2_th13_opt}_{args.stat_opt}_{args.bins}bins_{args.grid_points}gridpoints.npy", allow_pickle=True)
  file_out = open(f"../Data/fit_results_{args.stat_method_opt}_{args.sin2_th13_opt}_{args.stat_opt}_{args.bins}bins_{args.grid_points}gridpoints.txt", "a")
  file_out.write("unc sin2_12 sin2_12_err sin2_13 sin2_13_err dm2_21 dm2_21_err dm2_31 dm2_31_err\n")


  print("Unc.\tsin2_th12\tsin2_th13\tdm2_21\tdm2_31")
  chi2marg = {}
  #xval = {'sin2_th12': sin2_ar, 'dm2_21': dm2_ar, 'dm2_31': dm2_31_ar}
  op_list = ['sin2_12', 'sin2_13', 'dm2_21', 'dm2_31']
  gridfit = {'sin2_12': np.linspace(0.302,0.312,200),
          'sin2_13': np.linspace(0.0068,0.0368,200),
          'dm2_21': np.linspace(7.47e-5,7.59e-5,200),
          'dm2_31': np.linspace(2.515e-3,2.543e-3,200)}
  #
  #   Marginalization and fitting chi2 profiles
  sens = {}
  for unc in chi2maps.keys():
    #TODO: make this better
    if args.stat_method_opt == "CNP":
      if unc == "stat":
        file_out.write(f"{unc} ")
      elif unc.count("+") > 1:
        file_out.write("stat+all ")
      else:
        file_out.write(f"stat+{unc} ")
    else:
        file_out.write(f"{unc} ")
    sens[unc] = []
    chi2marg['sin2_12'] = [np.amin(chi2maps[unc][i,:,:,:]) for i in range(len(grid['sin2_12']))]
    chi2marg['sin2_13'] = [np.amin(chi2maps[unc][:,j,:,:]) for j in range(len(grid['sin2_13']))]
    chi2marg['dm2_21']    = [np.amin(chi2maps[unc][:,:,k,:]) for k in range(len(grid['dm2_21']))]
    chi2marg['dm2_31']    = [np.amin(chi2maps[unc][:,:,:,l]) for l in range(len(grid['dm2_31']))]
    print(unc, end='\t')
    if not os.path.exists(f"{args.plots_folder}/Chi2_profiles"): os.mkdir(f"{args.plots_folder}/Chi2_profiles")
    for op in op_list:
      if len(chi2marg[op])<2: continue
      _,a0,b0,c0 = FitChi2Pol2(grid[op], chi2marg[op], chi2_level=1.0, return_param = True)
      sigma,a,b,c = FitChi2Pol2_through_zero(grid[op], chi2marg[op], p0=[a0,b0], chi2_level=1.0, return_param = True)
      sens[unc].append(sigma)
      #_,a,b,c = noda.FitChi2Pol2_cross_zero(grid[op], chi2marg[op], chi2_level=1.0, return_param = True)
      file_out.write(f"{-1*b/(2*a)} {sens[unc][-1]}")
      if (op == 'dm2_31'): file_out.write("\n")
      else: file_out.write(" ")
      print("{:>1.6f}\t".format(sens[unc][-1]), end="")
      plt.figure(figsize=(6, 6))
   #   print(grid[op])
  #    print(chi2marg[op])
      plt.plot(grid[op], chi2marg[op], label = "data")
      plt.plot(gridfit[op], a*gridfit[op]*gridfit[op] + b*gridfit[op] + c,  label = "fit", color = 'r')
      plt.grid()
      plt.xlabel(axis_labels[op])
      plt.ylabel(r"$\Delta \chi^2$")
      plt.savefig(f"{args.plots_folder}/Chi2_profiles/chi2_{args.stat_opt}_{op}_{unc}.png")
  #    if unc=='eff+r2+crel+snf+noneq+b2b_TAO+abc+nl+bg+me':
  #    plt.show()
      plt.close()
    print("")

#  import pickle
 # with open(f"sens_{args.stat_opt}_sin2_th13_{args.sin2_th13_opt}_{args.stat_method_opt}_{args.bins}bins_grid5s.dat", "wb") as f:
  #  pickle.dump(sens, f)
