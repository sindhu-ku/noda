#!/usr/bin/python
import sys
import numpy as np
import emcee
from nuosc import *
from  noda import *
import noda
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import arviz  as az
import corner as cr
import os

#TO DO: This code needs more cleaning, mostly unedited from Pietro

def run_emcee(ensp_nom = {}, baselines = [],  powers=[], rm= [], cm ={}, SEED = 0, args=''):

  unc=args.unc_list[len(args.unc_list)-1]
  if(args.bayes_chi2 == 'CNP' and unc != 'stat'): unc = unc.replace('stat+', '')

  nuosc.SetOscillationParameters(opt=args.PDG_opt, NMO=args.NMO_opt) #Vals for osc parameters and NMO
  noda.SetOscFormula(args.osc_formula_opt) #Gets the antinu survival probability
  print(" # Oscillation parameters:")
  for k, val in nuosc.op_nom.items(): #print input values
      print(f"   {k:<12} {val}")
  np.random.seed(SEED)
  EVENTS = args.bayes_events
  # here set your estimates of central values
  mid_dm2sol = nuosc.op_nom["dm2_21"]
  mid_dm2atm = nuosc.op_nom["dm2_31"]
  mid_s2t12  = nuosc.op_nom["sin2_th12"]
  mid_s2t13  = nuosc.op_nom["sin2_th13"]
  mid_vals = np.array([mid_dm2sol, mid_dm2atm, mid_s2t12, mid_s2t13])
  print(" JB : ")
  print("Present dm2sol: ", mid_dm2sol)
  print("Present dm2atm: ", mid_dm2atm)
  print("Present s2t12: ", mid_s2t12)
  print("Present s2t13: ", mid_s2t13)
  # Simple uniform prior over definite range:
  # this is actually the  log of prior plus an arbitrary (irrelevant) constant
  # 1 for parameters within range
  # -inf for parameters out of range
  # Parameters: like in mid_vals
  def prior(x):
      p = 1
      if x[0] < mid_dm2sol/10.: p = -1.*float("inf")
      if x[0] > mid_dm2sol*10.: p = -1.*float("inf")
      if x[1] < mid_dm2atm/10.: p = -1.*float("inf")
      if x[1] > mid_dm2atm*10.: p = -1.*float("inf")
      if x[2] < mid_s2t12/10. : p = -1.*float("inf")
      if x[2] > mid_s2t12*10. : p = -1.*float("inf")
      if x[3] < mid_s2t13/10. : p = -1.*float("inf")
      if x[3] > mid_s2t13*10. : p = -1.*float("inf")
      return p
  def log_prob(x):
      p = prior(x)
      if p == -1.*float("inf") : return p, p, p

      s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
      sin2_th12=x[2], sin2_th13=x[3], dm2_21=x[0], dm2_31=x[1],
      me_rho=args.me_rho,
      ene_mode='true')
      s = s.GetWithPositronEnergy()
      s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
      s = s.ApplyDetResp(rm, pecrop=args.ene_crop)
      chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, args.bayes_chi2)
    #  print(chi2)

     # print("x", x)
      #print("chi2", chi2)

      ll = chi2
      return p + ll, p + ll, p
  # 5 parameters
  # the number fo walkers should be >> ndim
  ndim, nwalkers = len(mid_vals), args.bayes_nwalkers
  mid_vals.reshape(ndim,1)
  # this estimates comes from frequentist estimates
  # and are used to initialize the walkers
  sigma_estimate_list = [ 2.27e-7, 3.86e-6, 2.42e-3, 9.61e-3]
  print(" JB ")
  for i in range(len(sigma_estimate_list)):
      print(" Sigma estimate par ",i," = ", sigma_estimate_list[i])
  sigma_estimate = np.array(sigma_estimate_list)
  sigma2_estimate = sigma_estimate**2
  sigma_estimate.reshape(ndim,1)
  p0 = np.zeros((args.bayes_nwalkers, ndim))
  # Initialize walkers
  for  i in range(args.bayes_nwalkers):
      p0[i,:] = mid_vals + sigma_estimate * np.random.normal(size = ndim)
      # Check initial values are within range, otherwise set again
      while prior(p0[i,:]) == -1.*float("inf") : p0[i,:] = mid_vals + sigma_estimate * np.random.normal(size = ndim)
  print("p0 :")
  print(p0)
  # Now run the sampler
  sampler = emcee.EnsembleSampler(args.bayes_nwalkers, ndim, log_prob )
  sampler.run_mcmc(p0, EVENTS, skip_initial_state_check = True)
  chain = sampler.get_chain() # the chain of parameters
  blobs = sampler.get_blobs() # additional infos, maybe used later
  print(chain)
  if not os.path.exists(f"{args.bayes_data_folder}"):
    os.makedirs(f"{args.bayes_data_folder}")
  np.savez(f"{args.bayes_data_folder}/MCMC_Bayesian_1_{EVENTS}_{SEED}_{args.stat_opt}_{args.bayes_chi2}.npz",chain=chain,blobs=blobs)
  print("Ending JBF")

def get_results(args=''):

  if not os.path.exists(f"{args.bayes_plots_folder}"):
    os.makedirs(f"{args.bayes_plots_folder}")
  stat_opt = args.stat_opt

  labels = [r"$\Delta M^2_{SOL}$",
          r"$\Delta M^2_{ATM}$",
          r"$S^2\Theta_{12}$",
          r"$S^2\Theta_{13}$"]

  nfiles    = args.bayes_nprocesses
  nvars     = len(labels)
  nsteps    = args.bayes_events
  skipsteps = 50
  file_list = []
  nwalkers = args.bayes_nwalkers
  for i in range(args.bayes_seed_beg, args.bayes_nprocesses+args.bayes_seed_beg):
      file_list.append(f"{args.bayes_data_folder}/MCMC_Bayesian_1_{nsteps}_{i}_{stat_opt}.npz")


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
      plt.savefig(f"{bayes_plots_folder}/{stat_opt}/autocorrectplot.png")

  for i in range(nvars):
      fig, axes = plt.subplots(1, figsize = (10,7), sharex = True)
      for j in range(nfiles):
        for k in range(1):
           axes.plot(npchains[j,:,k,i], "k", alpha=0.3)
      axes.set_xlim(0,nsteps)
      axes.set_ylabel(labels[i])
      Axis.set_label_coords(axes.yaxis, -0.1, 0.5)
      axes.set_xlabel("step number")
      plt.savefig(f'{bayes_plots_folder}/{stat_opt}_{nsteps}events/stepnumber.png')



  plt.savefig(f'{bayes_plots_folder}/{stat_opt}_{nsteps}events/stepnumber-what.png')

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

  plt.savefig(f'{bayes_plots_folder}/{stat_opt}_{nsteps}events/az.png')

  npchains_cut  = npchains[:,skipsteps:,:]
  npchains_full = npchains_cut.reshape(nfiles*args.bayes_nwalkers*(nsteps-skipsteps),nvars)
  figure = cr.corner( npchains_full , labels = labels )

  plt.savefig(f'{bayes_plots_folder}/{stat_opt}_{nsteps}events/corner.png')


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
