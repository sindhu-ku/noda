import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
from scipy.interpolate import CubicSpline

def get_xy(filename):
    df = pd.read_csv(filename, delimiter=' ', header=None) 
    df[3] = df[3].round(decimals=5)  # Get unique values in the 4th column
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

def main(argv):

    hist_m231=[]
    x=[]
    y=[]
    for i in range(1):
        unique_4th_values_100, min_5th_values_100 = get_xy(f"chi2_txt/chi2_6years_{i}_pull_100p.txt")
        print("length ", len(unique_4th_values_100))
        # Find the index of the minimum value in min_5th_values_100
        min_index = min_5th_values_100.index(min(min_5th_values_100))
        #x.append(unique_4th_values_100)
        #y.append(min_5th_values_100)
        # Retrieve the corresponding value from unique_4th_values_100
        #delta_chi2_values_100 = min_5th_values_100 - min_5th_values_100[min_index]
        #hist_m231.append( unique_4th_values_100[min_index])
        # Assuming unique_4th_values_100 and min_5th_values_100 are your x, y data
        #x = np.array(unique_4th_values_100)
        #y = np.array(min_5th_values_100)
        
        # Create an interpolation function
        #spl = CubicSpline(x, y)
        
        # Generate new x values for interpolation
        #new_x_values = np.linspace(min(x), max(x), 10000)  # Adjust the number of points as needed
        
        # Use the interpolation function to get corresponding y values
        #new_y_values = spl(new_x_values)
        plt.plot(unique_4th_values_100, min_5th_values_100, linestyle='-', label={i})# marker='.', markersize=10)
        del unique_4th_values_100, min_5th_values_100#, x, y, new_x_values, new_y_values#, delta_chi2_values_100
   # merged_x = list(itertools.chain.from_iterable(x))    
   # merged_y = list(itertools.chain.from_iterable(y))    
   # plt.hist2d(merged_x, merged_y, bins=20)
    #plt.colorbar()
    plt.xlabel('dm2_atm')
    #plt.ylim(0,30)
    plt.ylabel('chi2_CNP')
    #plt.legend()
    #plt.axhline(9, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
    #plt.axhline(16, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
    #plt.axhline(25, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
    plt.show()
    #plt.hist(hist_m231, bins=100)
    #plt.show()


if __name__ == "__main__":
    main(None)
