#!/usr/bin/python
import sys
import numpy as np
import emcee
from nuosc import *
from  noda import *
import noda

def run_emcee( PDG_opt = ' ', NMO_opt ='', pmt_opt = '', stat_opt='',
             osc_formula_opt = '', ensp_nom = {}, me_rho =0.0, baselines = [], bins=411, powers=[], rm= [], cm ={}, ene_crop=(), SEED = 0, unc =' ', stat_meth= ' '):
  print("unc", unc)
  if(stat_meth == 'CNP' and unc != 'stat'): unc = unc.replace('stat+', '')
  nuosc.SetOscillationParameters(opt=PDG_opt, NMO=NMO_opt) #Vals for osc parameters and NMO
  noda.SetOscFormula(osc_formula_opt) #Gets the antinu survival probability
  print(" # Oscillation parameters:")
  for k, val in nuosc.op_nom.items(): #print input values
      print(f"   {k:<12} {val}")
  np.random.seed(SEED)
  EVENTS = 200
  # here set your estimates of central values
  mid_dm2sol = nuosc.op_nom["dm2_21"]
  mid_dm2atm = nuosc.op_nom["dm2_31"]
  mid_s2t12  = nuosc.op_nom["sin2_th12"]
  mid_s2t13  = nuosc.op_nom["sin2_th13"]
  #mid_rho    = juno_inputs.me_rho
  mid_vals = np.array([mid_dm2sol, mid_dm2atm, mid_s2t12, mid_s2t13])#, mid_rho])
  print(" JB : ")
  print("Present dm2sol: ", mid_dm2sol)
  print("Present dm2atm: ", mid_dm2atm)
  print("Present s2t12: ", mid_s2t12)
  print("Present s2t13: ", mid_s2t13)
  #print("Present rho: ", mid_rho)
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
      # if x[4] < 0. : p = -1.*float("inf")
      # if x[4] > 10. : p = -1.*float("inf")
      return p
  def log_prob(x):
      p = prior(x)
      if p == -1.*float("inf") : return p, p, p

      s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
      sin2_th12=x[2], sin2_th13=x[3], dm2_21=x[0], dm2_31=x[1],
      me_rho=me_rho,
      ene_mode='true')
      s = s.GetWithPositronEnergy()
      s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
      s = s.ApplyDetResp(rm, pecrop=ene_crop)
      chi2 = cm[unc].Chi2(ensp_nom["rdet"],s, stat_meth)
    #  print(chi2)

     # print("x", x)
      #print("chi2", chi2)

      ll = chi2
      return p + ll, p + ll, p
  # 5 parameters
  # the number fo walkers should be >> ndim
  ndim, nwalkers = 4, 30
  mid_vals.reshape(ndim,1)
  # this estimates comes from frequentist estimates
  # and are used to initialize the walkers
  sigma_estimate_list = [ 2.27e-7, 3.86e-6, 2.42e-3, 9.61e-3]#, 0.144 ]
  #sigma_estimate_list = [5.41e-7, 1.45e-5, 4.08e-3, 7.4e-3]
  print(" JB ")
  for i in range(len(sigma_estimate_list)):
      print(" Sigma estimate par ",i," = ", sigma_estimate_list[i])
  sigma_estimate = np.array(sigma_estimate_list)
  sigma2_estimate = sigma_estimate**2
  sigma_estimate.reshape(ndim,1)
  p0 = np.zeros((nwalkers, ndim))
  # Initialize walkers
  for  i in range(nwalkers):
      p0[i,:] = mid_vals + sigma_estimate * np.random.normal(size = ndim)
      # Check initial values are within range, otherwise set again
      while prior(p0[i,:]) == -1.*float("inf") : p0[i,:] = mid_vals + sigma_estimate * np.random.normal(size = ndim)
  print("p0 :")
  print(p0)
  # Now run the sampler
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob )
  sampler.run_mcmc(p0, EVENTS, skip_initial_state_check = True)
  chain = sampler.get_chain() # the chain of parameters
  blobs = sampler.get_blobs() # additional infos, maybe used later
  print(chain)
  if not os.path.exists("../Data/npz_files"):
    os.makedirs("../Data/npz_files")
  np.savez(f"../Data/npz_files/MCMC_Bayesian_1_{EVENTS}_{SEED}_{stat_opt}_{stat_meth}.npz",chain=chain,blobs=blobs)
  print("Ending JBF")
