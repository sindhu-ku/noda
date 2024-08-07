#!/usr/bin/env python3
import numpy  as np
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import arviz  as az
import corner as cr
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

def get_results(args=''):

  if not os.path.exists(f"{args.bayes_plots_folder}"):
    os.makedirs(f"{args.bayes_plots_folder}")
  if not os.path.exists(f"{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange"):
    os.makedirs(f"{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange")

  labels = [r"$\Delta M^2_{SOL}$",
          r"$\Delta M^2_{ATM}$",
          r"$S^2\Theta_{12}$",
          r"$S^2\Theta_{13}$"]

  nfiles    = args.bayes_nprocesses*10
  nvars     = len(labels)
  nsteps    = args.bayes_events
  skipsteps = 50
  file_list = []
  nwalkers = args.bayes_nwalkers
  dm2_31_val = 2.583e-3
  dm2_31_list = np.linspace((dm2_31_val - dm2_31_val*0.2),(dm2_31_val + dm2_31_val*0.2), 10)
  for i in range(args.bayes_seed_beg, args.bayes_nprocesses+args.bayes_seed_beg):
      for j in dm2_31_list:
        file_list.append(f"{args.bayes_data_folder}/MCMC_Bayesian_1_{nsteps}_{i}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins_{args.bayes_chi2}_dm231_asimov_{j}.npz")


  chains = []

  for infile in file_list:
      arrays = np.load(infile)
      print(arrays["chain"].shape)
      chains.append(arrays["chain"])
      arrays.close()

  for c in chains:
      print(c[:-1,0,0].shape)
      #print(c)
      repetition = np.equal(c[:-1,0,0],c[1:,0,0])
      print(repetition.sum())


  npchains = np.stack(chains, axis=0)


  print(npchains.shape)
  print(len(labels))

  chainr  = range(nfiles)
  drawr = range(nsteps)
  walkerr = range(args.bayes_nwalkers)

  var0 = npchains[:,:,:,0]
  var1 = npchains[:,:,:,1]
  var2 = npchains[:,:,:,2]
  var3 = npchains[:,:,:,3]
  #var4 = npchains[:,:,:,4]

  xrdata = xr.Dataset(
          {
              labels[0]: (["chain","draw","walker"],var0),
              labels[1]: (["chain","draw","walker"],var1),
              labels[2]: (["chain","draw","walker"],var2),
              labels[3]: (["chain","draw","walker"],var3)
          #    labels[4]: (["chain","draw","walker"],var4)
         },
          coords={
              "chain":(["chain"],chainr),
              "draw":(["draw"],drawr),
              "walker":(["walker"],walkerr)
          }
      )
  dataset = az.InferenceData(posterior=xrdata)


  sample_size = az.ess(dataset)
  r_hat = az.rhat(dataset)

  print("Sample size:")
  ssa=sample_size.to_array()
  print(np.sum(ssa,axis=1))


  print("R_hat")
  rha = r_hat.to_array()
  print(rha)

  var0_0 = npchains[:,:,0,0]
  var1_0 = npchains[:,:,0,1]
  var2_0 = npchains[:,:,0,2]
  var3_0 = npchains[:,:,0,3]
  #var4_0 = npchains[:,:,0,4]

  xrdata_w0 = xr.Dataset(
          {
              labels[0]: (["chain","draw"],var0_0),
              labels[1]: (["chain","draw"],var1_0),
              labels[2]: (["chain","draw"],var2_0),
              labels[3]: (["chain","draw"],var3_0),
              #labels[4]: (["chain","draw"],var4_0)
          },
          coords={
              "chain":(["chain"],chainr),
              "draw":(["draw"],drawr)
          }
      )
  dataset_w0 = az.InferenceData(posterior=xrdata_w0)

  for i in range(nvars):
      print("Autocorrplot ",labels[i])
      az.plot_autocorr(dataset_w0, var_names=labels[i])
      plt.savefig(f"{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange/autocorrectplot.png")

  for i in range(nvars):
      fig, axes = plt.subplots(1, figsize = (10,7), sharex = True)
      for j in range(nfiles):
        for k in range(1):
           axes.plot(npchains[j,:,k,i], "k", alpha=0.3)
      axes.set_xlim(0,nsteps)
      axes.set_ylabel(labels[i])
      Axis.set_label_coords(axes.yaxis, -0.1, 0.5)
      axes.set_xlabel("step number")
      plt.savefig(f'{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange/stepnumber.png')



  plt.savefig(f'{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange/stepnumber-what.png')

  npchains_aw = npchains.reshape(nfiles*args.bayes_nwalkers,nsteps,nvars)

  chainr_aw  = range(nfiles*args.bayes_nwalkers)
  drawr_aw = range(nsteps-skipsteps)

  var0_aw = npchains_aw[:,skipsteps:,0]
  var1_aw = npchains_aw[:,skipsteps:,1]
  var2_aw = npchains_aw[:,skipsteps:,2]
  var3_aw = npchains_aw[:,skipsteps:,3]
  #var4_aw = npchains_aw[:,skipsteps:,4]

  xrdata_aw = xr.Dataset(
          {
              labels[0]: (["chain","draw"],var0_aw),
              labels[1]: (["chain","draw"],var1_aw),
              labels[2]: (["chain","draw"],var2_aw),
              labels[3]: (["chain","draw"],var3_aw),
              #labels[4]: (["chain","draw"],var4_aw)
         },
          coords={
              "chain":(["chain"],chainr_aw),
              "draw":(["draw"],drawr_aw)
          }
      )

  dataset_aw = az.InferenceData(posterior=xrdata_aw)
  print(az.summary(dataset_aw,round_to="none",fmt="long").to_string())
  az.plot_posterior(dataset_aw)

  plt.savefig(f'{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange/az.png')

  npchains_cut  = npchains[:,skipsteps:,:]
  npchains_full = npchains_cut.reshape(nfiles*args.bayes_nwalkers*(nsteps-skipsteps),nvars)
  figure = cr.corner( npchains_full , labels = labels )
  np.savez(f"fchain_{args.stat_opt}_asimov_fullrange.npz", npchains_full)
  plt.savefig(f'{args.bayes_plots_folder}/{args.stat_opt}_NO-{args.NMO_opt}_{args.bins}bins_asimov_fullrange/corner.png')


  means      = np.zeros((nfiles,nvars))
  variances  = np.zeros((nfiles,nvars))
  sdevs      = np.zeros((nfiles,nvars))

  for i in range(nvars):
     for j in range(nfiles):
         means[j,i] = np.mean(npchains_aw[j,:,i])
         variances[j,i] = np.var(npchains_aw[j,:,i])
         sdevs[j,i] = np.sqrt(variances[j,i])
     print(labels[i]," : ")
     print("Mean value : ")
     print("{:5.10E}".format(np.mean(means[:,i]))," pm {:5.10E}".format(np.std(means[:,i])))
     print("Variance : ")
     print("{:5.10E}".format(np.mean(variances[:,i]))," pm {:5.10E}".format(np.std(variances[:,i])))
     print("Sigma : ")
     print("{:5.10E}".format(np.mean(sdevs[:,i]))," pm {:5.10E}".format(np.std(sdevs[:,i])))
     print(">>> {:1.2E}".format(np.mean(means[:,i]))," pm {:1.2E}".format(np.mean(sdevs[:,i]))," ({:1.5}%)".format(np.mean(sdevs[:,i])/np.mean(means[:,i])))
     print(" ")
     print(" ")




  print("Means:")
  print(means)
  print("Variances:")
  print(variances)
