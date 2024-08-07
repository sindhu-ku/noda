#!/usr/bin/python3
import sys
import numpy as np
import emcee
from .. import nuosc as nuosc
from .. import noda as noda
import os

#TO DO: This code needs more cleaning, mostly unedited from Pietro

def run_emcee(ensp_nom = {}, baselines = [],  powers=[], rm= [], cm ={}, SEED = 0, args='', dm2_atm=0):
  unc = args.bayes_unc

  if(args.bayes_chi2 == 'CNP' and unc != 'stat'): unc = unc.replace('stat+', '')

  nuosc.SetOscillationParameters(opt=args.PDG_opt, NO=args.NMO_opt) #Vals for osc parameters and NMO
  noda.SetOscFormula(args.osc_formula_opt) #Gets the antinu survival probability
  print(" # Oscillation parameters:")
  for k, val in nuosc.op_nom.items(): #print input values
      print(f"   {k:<12} {val}")
  np.random.seed(SEED)
  EVENTS = args.bayes_events
  # here set your estimates of central values
  mid_dm2sol = nuosc.op_nom["dm2_21"]
#  mid_dm2atm = nuosc.op_nom["dm2_32"]
  mid_s2t12  = nuosc.op_nom["sin2_th12"]
  mid_s2t13  = nuosc.op_nom["sin2_th13"]
  mid_vals = np.array([mid_dm2sol, mid_s2t12, mid_s2t13])
  print(" JB : ")
  print("Present dm2sol: ", mid_dm2sol)
#  print("Present dm2atm: ", mid_dm2atm)
  print("Present s2t12: ", mid_s2t12)
  print("Present s2t13: ", mid_s2t13)
  # Simple uniform prior over definite range:
  # this is actually the  log of prior plus an arbitrary (irrelevant) constant
  # 1 for parameters within range
  # -inf for parameters out of range
  # Parameters: like in mid_vals
  def prior(x):
      p = 1
      if x[0] < mid_dm2sol - mid_dm2sol*0.1 : p = -1.*float("inf")
      if x[0] > mid_dm2sol + mid_dm2sol*0.1 : p = -1.*float("inf")
#      if x[1] < mid_dm2atm - mid_dm2atm*0.2 : p = -1.*float("inf")
 #     if x[1] > mid_dm2atm + mid_dm2atm*0.2 : p = -1.*float("inf")
      if x[1] < mid_s2t12 - mid_s2t12*0.1 : p = -1.*float("inf")
      if x[2] > mid_s2t12 + mid_s2t12*0.1 : p = -1.*float("inf")
      if x[2] < mid_s2t13 - mid_s2t13*3. : p = -1.*float("inf")
      if x[2] > mid_s2t13 + mid_s2t13*3. : p = -1.*float("inf")
     # if x[0] < mid_dm2sol - mid_dm2sol*10. : p = -1.*float("inf")
     # if x[0] > mid_dm2sol + mid_dm2sol/10. : p = -1.*float("inf")
     # if x[1] < mid_dm2atm - mid_dm2atm*10. : p = -1.*float("inf")
     # if x[1] > mid_dm2atm + mid_dm2atm/10. : p = -1.*float("inf")
     # if x[2] < mid_s2t12 - mid_s2t12*10. : p = -1.*float("inf")
     # if x[2] > mid_s2t12 + mid_s2t12/10. : p = -1.*float("inf")
     # if x[3] < mid_s2t13 - mid_s2t13*10. : p = -1.*float("inf")
     # if x[3] > mid_s2t13 + mid_s2t13/10. : p = -1.*float("inf")
      return p
  def log_prob(x):
      p = prior(x)
      if p == -1.*float("inf") : return p, p, p

      s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
      sin2_th12=x[1], sin2_th13=x[2], dm2_21=x[0], dm2_32=dm2_atm,
      me_rho=args.me_rho,
      ene_mode='true', args=args)
      s = s.GetWithPositronEnergy()
      s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
      s = s.ApplyDetResp(rm, pecrop=args.ene_crop)
      chi2 = 1e+6
      if args.sin2_th13_opt== "pull":
        chi2 = cm[unc].Chi2_p(ensp_nom["rdet"], s, pulls=[x[2]-nuosc.op_nom['sin2_th13']], pull_unc=[args.sin2_th13_pull_unc*nuosc.op_nom['sin2_th13']], stat_meth=args.bayes_chi2)
      if args.sin2_th13_opt== "free":
        chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, unc, args.bayes_chi2)
      filet = open(f"chi2_{args.stat_opt}_pull_bayes.txt", "a")
      filet.write(str(x[1])+" "+str(x[2])+" "+str(x[0])+" "+str(dm2_atm)+" "+str(chi2)+"\n")
      filet.close()
    #  print(chi2)

     # print("x", x)
      #print("chi2", chi2)

      ll = -0.5*chi2
      return p + ll, p + ll, p
  # 5 parameters
  # the number fo walkers should be >> ndim
  ndim, nwalkers = len(mid_vals), args.bayes_nwalkers
  mid_vals.reshape(ndim,1)
  # this estimates comes from frequentist estimates
  # and are used to initialize the walkers
  sigma_estimate_list = [ 2.27e-7, 2.42e-3, 9.61e-3]
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
  np.savez(f"{args.bayes_data_folder}/MCMC_Bayesian_1_{EVENTS}_{SEED}_NO-{args.NMO_opt}_{args.stat_opt}_{args.bins}bins_{args.bayes_chi2}_fixatm.npz",chain=chain,blobs=blobs)
  print("Ending JBF")
