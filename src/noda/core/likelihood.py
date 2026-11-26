"""Chi2 / likelihood functions.

Split out of noda.py (pure move); noda.noda re-exports everything."""
import numpy as np
from ..physics.constants import *
from .covmatrix import CovMatrix


def chi2_val(obs, exp, cov_matrix, penalty=0.0):
    """Compute chi2 = (obs - exp)^T V^-1 (obs - exp) + penalty."""
    diff = obs - exp
    chi2 = diff.T @ np.linalg.solve(cov_matrix, diff) + penalty
    return chi2


def Chi2(cm, tot_obs, tot_exp, rea_obs, rea_exp, unc=' ', stat_meth=' ', pulls=None, pull_unc=None, include_logV=False, ext_cov=False):
    penalty = 0.0
    if pulls and pull_unc:
        for p, u in zip(pulls, pull_unc):
            penalty += (p / u) ** 2

    if stat_meth == "Neyman":
        stat_cov_matrix = rea_obs.GetStatCovMatrix().data
    elif stat_meth == "Pearson" or include_logV:
        stat_cov_matrix = rea_exp.GetStatCovMatrix().data
    else:
        epsilon = 1e-10
        rea_obs.bin_cont[:] = np.maximum(rea_obs.bin_cont, epsilon)
        rea_exp.bin_cont[:] = np.maximum(rea_exp.bin_cont, epsilon)
        stat_cov_matrix = rea_obs.GetCNPStatCovMatrix(rea_exp).data

    if unc == 'stat':
        cov_matrix = stat_cov_matrix

    elif ext_cov:
        cov_matrix = cm.data

    else:
        cov_matrix = stat_cov_matrix + cm.data

    obs = tot_obs.bin_cont
    exp = tot_exp.bin_cont
    valid = (obs > 0) & (exp > 0)
    cov_matrix = cov_matrix[np.ix_(valid, valid)]

    obs = obs[valid]
    exp = exp[valid]


    logdet = 0
    if include_logV:
        sign, logdet = np.linalg.slogdet(cov_matrix)
        if sign <= 0:
            raise np.linalg.LinAlgError("Covariance matrix not positive definite")

    chi2 = chi2_val(obs, exp, cov_matrix, penalty)

    return chi2, chi2 + logdet

def Combined_Chi2(cm1, cm2, corr_cm, tot_obs1, tot_exp1, tot_obs2, tot_exp2, rea_obs1, rea_exp1, rea_obs2, rea_exp2, unc=' ', stat_meth=' ', pulls=None, pull_unc=None, nd="TAO"):
  penalty = 0.0
  if pulls and pull_unc:
      for p, u in zip(pulls, pull_unc): penalty += (p / u) ** 2


  if stat_meth == "NorP":
      stat_cm1 = rea_obs1.GetStatCovMatrix()
      stat_cm2 = rea_obs2.GetStatCovMatrix()
  else:
      stat_cm1 = rea_obs1.GetCNPStatCovMatrix(rea_exp1)
      stat_cm2 = rea_obs2.GetCNPStatCovMatrix(rea_exp2)

  stat_cm1_mod = stat_cm1.Extend(CovMatrix(data=np.zeros((len(tot_obs1.bins)-1, len(tot_obs1.bins)-1)), bins=tot_obs1.bins))
  cm_stat_nd_z = CovMatrix(data=np.zeros((len(tot_obs2.bins)-1, len(tot_obs2.bins)-1)), bins=tot_obs2.bins)
  stat_cm2_mod = cm_stat_nd_z.Extend(stat_cm2)

  cm1_mod = cm1.Extend(CovMatrix(data=np.zeros((len(tot_obs1.bins)-1, len(tot_obs1.bins)-1)), bins=tot_obs1.bins))
  cm_nd_z = CovMatrix(data=np.zeros((len(tot_obs2.bins)-1, len(tot_obs2.bins)-1)), bins=tot_obs2.bins)
  cm2_mod = cm_nd_z.Extend(cm2)

  if unc == "stat":
      total_cov_matrix = stat_cm1_mod.data + stat_cm2_mod.data

  else:
      total_cov_matrix = stat_cm1_mod.data +  stat_cm2_mod.data + cm1_mod.data + cm2_mod.data + corr_cm.data


  diff1 = tot_obs1.bin_cont - tot_exp1.bin_cont
  diff2 = tot_obs2.bin_cont - tot_exp2.bin_cont

  obs_total = np.concatenate((tot_obs1.bin_cont, tot_obs2.bin_cont))
  exp_total = np.concatenate((tot_exp1.bin_cont, tot_exp2.bin_cont))

  valid_mask = (obs_total > 0) & (exp_total > 0)

  diff_total = np.concatenate((diff1, diff2))
  total_cov_matrix = total_cov_matrix[np.ix_(valid, valid)]
  #c_factor = cho_factor(cov_matrix)
  #chi2_term = cho_solve(c_factor, diff[valid])
  #chi2 = np.einsum('i,i->', diff[valid], chi2_term) + penalty
  chi2 = diff_total[valid].T @ np.linalg.solve(total_cov_matrix, diff_total[valid]) + penalty

  return chi2

def Chi2DYB(rel_cm, obs_sp, exp_sp):
    obs = obs_sp.bin_cont
    exp = exp_sp.bin_cont
    diff = (obs/exp) - 1
    chi2 =  diff.T @ np.linalg.solve(rel_cm.data, diff)
    return chi2
