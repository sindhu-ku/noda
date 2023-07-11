import sys
import nuosc
from  noda import *
import noda
import numpy as np
import os
import gc
from datetime import datetime
from joblib import Parallel, delayed
from collections import ChainMap
import pandas as pd

def scan_chi2(sin2_th13_opt = '', PDG_opt = ' ', NMO_opt ='', pmt_opt = '', stat_opt='',
             osc_formula_opt = '', grid={}, ensp_nom = {}, unc_list = [],
             me_rho =0.0, baselines = [], bins=411, powers=[], rm= [], cm ={}, ene_crop=(), stat_meth= ' '):
  nuosc.SetOscillationParameters(opt=PDG_opt, NMO=NMO_opt) #Vals for osc parameters and NMO
  noda.SetOscFormula(osc_formula_opt) #Gets the antinu survival probability
  print(" # Oscillation parameters:")
  for k, val in nuosc.op_nom.items(): #print input values
      print(f"   {k:<12} {val}")

  temp_file = f'../Data/chi2maps_temp_{stat_meth}_{stat_opt}_{bins}.txt'
  chi2maps = {}
  for key in unc_list:
    if sin2_th13_opt == 'fixed':
        chi2maps[key] = np.zeros((len(grid['sin2_12']), len(grid['dm2_21']), len(grid['dm2_31'])))
    else:
        chi2maps[key] = np.zeros((len(grid['sin2_12']), len(grid['sin2_13']), len(grid['dm2_21']), len(grid['dm2_31'])))

  fileoo = open((temp_file),"a")
  if sin2_th13_opt == 'fixed':
    fileoo.write('key i j k chi2 \n')
  else:
    fileoo.write('key i j k l chi2 \n')
  fileoo.close()

  def sin2_th13_fixed(i, j, k, sin2_12, dm2_21, dm2_31):
       print(i, j, k, sin2_12, dm2_21, dm2_31)
       s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
                                          sin2_th12=sin2_12, dm2_21=dm2_21, dm2_31=dm2_31,
                                          me_rho=me_rho,
                                          ene_mode='true')
       s = s.GetWithPositronEnergy()
       s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
       s = s.ApplyDetResp(rm, pecrop=ene_crop)
       for key in unc_list:
           chi2 = cm[key].Chi2(ensp_nom["rdet"],s, stat_meth)
           fileoo.write(str(key)+' '+str(i)+' '+str(j)+' '+str(k)+' '+str(chi2)+'\n')
           fileoo.close()
       del s

  def sin2_th13_free(i, j, k, l, sin2_12, sin2_13, dm2_21, dm2_31):
       print(i, j, k, l, sin2_12, sin2_13, dm2_21, dm2_31)
       s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
                                        sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31,
                                          me_rho=me_rho,
                                          ene_mode='true')
       s = s.GetWithPositronEnergy()
       s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
       s = s.ApplyDetResp(rm, pecrop=ene_crop)
       for key in unc_list:
           chi2 = cm[key].Chi2(ensp_nom["rdet"],s, stat_meth)
           print(key, chi2)
           with open(temp_file, 'a') as fileoo:
               data = [str(key), str(i), str(j), str(k), str(l), str(chi2)]
               line = ' '.join(data) + '\n'
               fileoo.write(line)
       del s

  def sin2_th13_pull(i, j, k, l, sin2_12, sin2_13, dm2_21, dm2_31, sin2_th13_nom):
       print(i, j, k, l, sin2_12, sin2_13, dm2_21, dm2_31)
       s = ensp_nom['ribd'].GetOscillated(L=baselines, core_powers=powers,
                                        sin2_th12=sin2_12, sin2_th13=sin2_13, dm2_21=dm2_21, dm2_31=dm2_31,
                                          me_rho=me_rho,
                                          ene_mode='true')
       s = s.GetWithPositronEnergy()
       s = s.GetWithModifiedEnergy(mode='spectrum', spectrum=ensp_nom['scintNL'])
       s = s.ApplyDetResp(rm, pecrop=ene_crop)
       for key in unc_list:
           chi2 = cm[key].Chi2_p(ensp_nom["rdet"], s,
                                             pulls=[sin2_13-sin2_th13_nom], pull_unc=[0.032*sin2_th13_nom]) # 3%
           fileoo.write(str(key)+' '+str(i)+' '+str(j)+' '+str(k)+' '+str(l)+' '+str(chi2)+'\n')
           fileoo.close()
       del s

  if sin2_th13_opt == 'fixed':
        Parallel(n_jobs = -1)(delayed(sin2_th13_fixed)(i, j, k, sin2_th12, dm2_21, dm2_31) for i, sin2_th12 in enumerate(grid['sin2_12']) for j, dm2_21 in enumerate(grid['dm2_21']) for k, dm2_31 in enumerate(grid['dm2_31']))
        df = pd.read_csv(temp_file, delimiter=' ')
        for index, row in df.iterrows():
          key = row['key']
          i = row['i']
          j = row['j']
          k = row['k']
          chi2 = row['chi2']
          chi2maps[key][i, j, k] = chi2

  elif sin2_th13_opt == 'pull':
       sin2_th13_nom = nuosc.op_nom['sin2_th13']
       Parallel(n_jobs = -1)(delayed(sin2_th13_pull)(i,j,k,l, sin2_th12, sin2_th13, dm2_21, dm2_31, sin2_th13_nom) for i, sin2_th12 in enumerate(grid['sin2_12']) for j, sin2_th13 in enumerate(grid['sin2_13']) for k, dm2_21 in enumerate(grid['dm2_21']) for l, dm2_31 in enumerate(grid['dm2_31']))
       df = pd.read_csv(temp_file, delimiter=' ')
       for index, row in df.iterrows():
         key = row['key']
         i = row['i']
         j = row['j']
         k = row['k']
         l = row['l']
         chi2 = row['chi2']
         chi2maps[key][i, j, k, l] = chi2

  else:
       Parallel(n_jobs = -1)(delayed(sin2_th13_free)(i,j,k,l, sin2_th12, sin2_th13, dm2_21, dm2_31) for i, sin2_th12 in enumerate(grid['sin2_12']) for j,sin2_th13 in enumerate(grid['sin2_13']) for k,dm2_21 in enumerate(grid['dm2_21']) for l,dm2_31 in enumerate(grid['dm2_31']))
       df = pd.read_csv(temp_file, delimiter=' ')
       for index, row in df.iterrows():
         key = row['key']
         i = row['i']
         j = row['j']
         k = row['k']
         l = row['l']
         chi2 = row['chi2']
         chi2maps[key][i, j, k, l] = chi2

  os.remove(temp_file)
  np.save(f"../Data/chi2maps_{stat_meth}_{sin2_th13_opt}_{pmt_opt}_{stat_opt}_{bins}.npy", (chi2maps, grid),  allow_pickle=True)
