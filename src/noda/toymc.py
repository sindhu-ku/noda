from .noda import Spectrum, CovMatrix, GetSpectrumFromROOT
from .minuit import run_minuit
import numpy as np
import h5py
import matplotlib.pyplot as plt

#np.random.seed(42)

def generate_toy_spectrum(spectrum_bin_cont, cov_matrix_data):
  L = np.linalg.cholesky(cov_matrix_data)
  y = np.random.normal(0, 1, len(spectrum_bin_cont))
  S_fluc = np.array(spectrum_bin_cont) + L @ y
  S_fluc = np.maximum(S_fluc, 0)

  S_fluc_poisson = np.random.poisson(S_fluc) #(spectrum_bin_cont).astype(float)   #(S_fluc)
  #S_fluc_poisson[S_fluc_poisson == 0] = 0.1
  #print(S_fluc_poisson)
  return S_fluc_poisson

def save_batch_results(filename, batch_results):
  if batch_results is None:
       print("WARNING: No valid data to save. Skipping...")
       return

  filtered_results = [row for row in batch_results if row is not None]
  if filtered_results is None:
       print("WARNING: No valid data to save. Skipping...")
       return
  new_data = np.array(filtered_results, dtype='S64')
  dataset_name ='geo'
  with h5py.File(filename, "a") as hdf:
       if dataset_name in hdf:
           dset = hdf[dataset_name]
           dset.resize(dset.shape[0] + new_data.shape[0], axis=0)
           dset[-new_data.shape[0]:] = new_data
       else:
           dset = hdf.create_dataset(
               dataset_name,
               data=new_data,
               maxshape=(None, new_data.shape[1]),  # Unlimited rows, fixed columns
               compression="gzip",
           )

def run_toy(i, ensp_nom_juno={}, unc_juno='', resp_matrix=None, cm_juno=None, args_juno=''):
     if i%1000 == 0: print(f"Toys: {i}/{args_juno.ntoys}")
     det_sp = ensp_nom_juno['rdet']
     if args_juno.fit_type == 'geo':
         if args_juno.geo_fit_type == 'total':
             det_sp += ensp_nom_juno['geo']
         elif args_juno.geo_fit_type == 'UThfree':
             det_sp += ensp_nom_juno['geou'] + ensp_nom_juno['geoth']
         else:
             det_sp += ensp_nom_juno['geomantle']

     ensp_nom_juno['rtot_toy'] = Spectrum(bins=ensp_nom_juno['rtot'].bins, bin_cont=generate_toy_spectrum(ensp_nom_juno['rtot'].bin_cont, cm_juno[unc_juno].data))
     nan_mask = np.isnan(ensp_nom_juno['rtot_toy'].bin_cont)
     if len(ensp_nom_juno['rtot_toy'].bin_cont[nan_mask] !=0): print("WARNING: NaN values found!")
     results = run_minuit(ensp_nom_juno=ensp_nom_juno, unc_juno=unc_juno, rm=resp_matrix, cm_juno=cm_juno, args_juno=args_juno)
     return results

