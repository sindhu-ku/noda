import ROOT
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

def get_xy(filename):
    df = pd.read_csv(filename, delimiter=' ', header=None) 
    df[3] = df[3].round(decimals=5)
    # Get unique values in the 4th column
    unique_values_4th_column = sorted(df[3].unique())
    
    # Initialize lists to store unique values in the 4th column and their corresponding minimum values in the 5th column
    unique_4th_values = []
    min_5th_values = []
    
    # Loop over unique values in the 4th column
    for value in unique_values_4th_column:
        # Filter the DataFrame for the current unique value
        subset_df = df[df[3] == value]
        # Find the minimum value in the 5th column for the subset
        min_value = subset_df[4].min()
        # Append the unique value and its corresponding minimum value to the lists
        unique_4th_values.append(value)
        min_5th_values.append(min_value)
    
    return unique_4th_values, min_5th_values
def get_array(filename):
    # Load the data from the npz file
    data = np.load(filename)
    ndarray_data  = []    
    # Assuming there is only one array in the npz file
    if len(data.files) == 1:
        # Access the array
        ndarray_data = data[list(data.files)[0]]
    
    else:
        print(f'The npz file contains more than one array. Please specify the key to access the desired array.')

    return ndarray_data

def main(argv):
        
    ndarray_data_fr = get_array("fchain_100days_asimov_fullrange.npz")
    ndarray_data_osv = get_array("fchain_100days_asimov_onestartval.npz")
    unique_4th_values_100, min_5th_values_100 = get_xy("../../noda_freq/noda/chi2_100days_pull.txt")
#    ndarray_22 = get_array("fchain_100days_toy_12_0.0022.npz")
#    ndarray_29 = get_array("fchain_100days_toy_12_0.0029.npz")
    # Create histograms for each column (0, 1, 2, 3)
 #   print(ndarray_data_osv[:, 1].size, ndarray_data_fr[:, 1].size)
    plt.hist(ndarray_data_fr[:, 1], bins=60,alpha=0.5,weights=max(min_5th_values_100)*np.ones_like(ndarray_data_fr[:, 1])/1680000., label='start value range +/- 20% of PDG 2022 value 0.002583')
   
    plt.hist(ndarray_data_osv[:, 1], bins=60,alpha=0.5,weights=max(min_5th_values_100)*np.ones_like(ndarray_data_osv[:, 1])/168000., label='one start value = PDG 2022, 0.002583')
   # plt.plot(unique_4th_values_100, min_5th_values_100, linestyle='-', label='Frequentist $\chi^{2}$')# marker='.', markersize=10)
 #   plt.hist(ndarray_22[:, 1], bins=60, label='start value = 0.0022')
  #  plt.hist(ndarray_29[:, 1], bins=60, label='start_value = 0.0029')
    
    plt.xlabel('dm2_31')
    #plt.yscale('log')
    plt.title('Asimov spectrum')
    plt.legend()
    plt.show()
    # Create histograms for each column (0, 1, 2, 3)
    #root_file = ROOT.TFile(f'output_bayes_posterior_toy_12_change_start_values.root', 'recreate')
    
    #hist = ROOT.TH1F(f'dm2_31','dm2_31 posterior for toy 12, start values range +/- 20% of PDG 2022 value', 60, np.min(ndarray_data[:, 1]), np.max(ndarray_data[:,1]))
    #for value in ndarray_data[:, 1]:
    #    hist.Fill(value)
    
    #hist.Write()
    
    #root_file.Close()

if __name__ == "__main__":
  main(sys.argv[1:])

