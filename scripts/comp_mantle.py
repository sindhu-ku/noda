import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('fit_results_geo_mantleGD_test_NorP_free_NO-True_6years_411bins_minuit.hdf5', 'r') as hdf:
    dataset = list(hdf.keys())[0]  # Assuming the first dataset is the relevant one
    data = hdf[dataset][:]         # Load data as a NumPy array

    # Extract 13th and 14th columns (Nmantle and Nmantle_err)
    nmantle_hdf5 = data[:, 12].astype('float')  # 0-based index
    nmantle_err_hdf5 = data[:, 13].astype('float')

# Load CSV data
df = pd.read_csv('fits_mantle.csv')
nmantle_csv = df['mantle'].values
nmantle_err_csv = df['mantle_err'].values

# Compare values
diff_mantle = np.abs(nmantle_hdf5 - nmantle_csv)
diff_mantle_err = np.abs(nmantle_err_hdf5 - nmantle_err_csv)

plt.plot(diff_mantle*100./nmantle_hdf5, linestyle='', marker='.', markersize=10)
plt.xlabel('Experiment #')
plt.ylabel('Rel. diff %')
plt.title('Nmantle')
plt.show()


plt.plot(diff_mantle_err*100./nmantle_err_hdf5, linestyle='', marker='.', markersize=10)
plt.xlabel('Experiment #')
plt.ylabel('Rel. diff %')
plt.title('Nmantle err')
plt.show()
