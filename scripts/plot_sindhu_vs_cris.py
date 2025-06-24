import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to the original and new HDF5 files
# File paths
hdf5_path1 ='toy_test/June3/fit_results_geo_covmatrix_toy_stat-only_10.0TNU_CNP_free_NO-True_1years_561bins_minuit.hdf5'
#hdf5_path ='toy_test/fit_results_geo_10.0TNU_stat-only_CNP_free_NO-True_1years_561bins_minuit.hdf5'
hdf5_path = 'June6/fit_results_from1_OPfixed_geo_CNP_10.0_NO-True_1years_561bins_minuit.hdf5'#'toy_test/June3/fit_results_geo_nocovmatrix_stat-only_10.0TNU_CNP_free_NO-True_1years_561bins_minuit.hdf5'
csv_path = '~/Downloads/fit_results_1year_10TNU_minos_cnp_shindhu_opfixed_12NN_from1_bkgfix.csv' #fit_results_1year_10TNU_minos_cnp_12NN_from1.csv'

# Load CSV
csv_df = pd.read_csv(csv_path)
geo = csv_df['geo_ratio'].values
geo_err = csv_df['geo_err'].values

# Load HDF5 (assumes one main dataset inside)
with h5py.File(hdf5_path, 'r') as f:
    # Print dataset names to find the correct one if needed
    dataset = list(f.values())[0]  # Assuming only one dataset
    data = dataset[...]
    print(data[0])

    # Columns 10 and 11 (indexing from 0)
    ngeo = np.array([float(val.decode('utf-8')) if isinstance(val, bytes) else float(val)
                     for val in data[:, 10]])

    ngeo_err = np.array([float(val.decode('utf-8')) if isinstance(val, bytes) else float(val)
                         for val in data[:, 11]])

with h5py.File(hdf5_path1, 'r') as f1:
    # Print dataset names to find the correct one if needed
    dataset1 = list(f1.values())[0]  # Assuming only one dataset
    data1 = dataset1[...]

    # Columns 10 and 11 (indexing from 0)
    ngeo1 = np.array([float(val.decode('utf-8')) if isinstance(val, bytes) else float(val)
                     for val in data1[:, 10]])

    ngeo_err1 = np.array([float(val.decode('utf-8')) if isinstance(val, bytes) else float(val)
                         for val in data1[:, 11]])

fig = plt.figure(figsize=(5,5))
plt.plot(ngeo, abs(geo-ngeo)*100./ngeo, 'o')
x = np.linspace(-0, 3, 100)  # 100 points from -10 to 10
y = x
#plt.plot(x, y, label='y = x', color='red', linestyle=':')
plt.xlabel('Sindhu Ngeo')
plt.ylabel('Rel. diff %') #'Cris Ngeo')
# plt.xlim(0, 3)
# plt.ylim(0, 3)
plt.show()

plt.hist(geo, histtype='step', label='Cris')
print(np.nanmedian(geo*118), np.nanmedian(ngeo*118.))
plt.hist(ngeo, histtype='step', label='Sindhu')
plt.legend()
plt.show()
#plt.show()
# Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# First canvas: Ngeo vs geo
axs[0].plot(ngeo, geo/0.77, 'o')
axs[0].set_xlabel('Ngeo (Sindhu)')
axs[0].set_ylabel('Ngeo (Cris)')
axs[0].set_title('Ngeo comparison, stat only')
axs[0].set_xlim(0, 2.5)
axs[0].set_ylim(0, 2.5)


# Second canvas: Ngeo_err vs geo_err
axs[1].plot(ngeo_err, geo_err/0.77, 'o', color='r')
axs[1].set_xlabel('Ngeo_err (Sindhu)')
axs[1].set_ylabel('Ngeo_err (Cris)')
axs[1].set_title('Ngeo error comparison, stat only')
axs[1].set_xlim(0, 3)
axs[1].set_ylim(0, 3)

axs[2].plot(ngeo_err/ngeo, geo_err/geo, 'o', color='r')
axs[2].set_xlabel('Ngeo_err/Ngeo (Sindhu)')
axs[2].set_ylabel('Ngeo_err/Ngeo (Cris)')
axs[2].set_title('Ngeo relative error comparison, stat only')
axs[2].set_xlim(0, 3)
axs[2].set_ylim(0, 3)

plt.tight_layout()
plt.show()
