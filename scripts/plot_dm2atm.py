import matplotlib.pyplot as plt
import pandas as pd
# <<<<<<< develop/toy_mc
# import itertools
# import numpy as np
# from scipy.interpolate import CubicSpline

# def get_xy(filename):
#     df = pd.read_csv(filename, delimiter=' ', header=None) 
#     df[3] = df[3].round(decimals=5)  # Get unique values in the 4th column
# =======

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

def main(argv):

# <<<<<<< develop/toy_mc
#     hist_m231=[]
#     x=[]
#     y=[]
#     for i in range(1):
#         unique_4th_values_100, min_5th_values_100 = get_xy(f"chi2_txt/chi2_6years_{i}_pull_100p.txt")
#         print("length ", len(unique_4th_values_100))
#         # Find the index of the minimum value in min_5th_values_100
#         min_index = min_5th_values_100.index(min(min_5th_values_100))
#         #x.append(unique_4th_values_100)
#         #y.append(min_5th_values_100)
#         # Retrieve the corresponding value from unique_4th_values_100
#         #delta_chi2_values_100 = min_5th_values_100 - min_5th_values_100[min_index]
#         #hist_m231.append( unique_4th_values_100[min_index])
#         # Assuming unique_4th_values_100 and min_5th_values_100 are your x, y data
#         #x = np.array(unique_4th_values_100)
#         #y = np.array(min_5th_values_100)
        
#         # Create an interpolation function
#         #spl = CubicSpline(x, y)
        
#         # Generate new x values for interpolation
#         #new_x_values = np.linspace(min(x), max(x), 10000)  # Adjust the number of points as needed
        
#         # Use the interpolation function to get corresponding y values
#         #new_y_values = spl(new_x_values)
#         plt.plot(unique_4th_values_100, min_5th_values_100, linestyle='-', label={i})# marker='.', markersize=10)
#         del unique_4th_values_100, min_5th_values_100#, x, y, new_x_values, new_y_values#, delta_chi2_values_100
#    # merged_x = list(itertools.chain.from_iterable(x))    
#    # merged_y = list(itertools.chain.from_iterable(y))    
#    # plt.hist2d(merged_x, merged_y, bins=20)
#     #plt.colorbar()
#     plt.xlabel('dm2_atm')
#     #plt.ylim(0,30)
#     plt.ylabel('chi2_CNP')
#     #plt.legend()
#     #plt.axhline(9, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
#     #plt.axhline(16, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
#     #plt.axhline(25, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
#     plt.show()
#     #plt.hist(hist_m231, bins=100)
#     #plt.show()
# =======
    #unique_4th_values_100, min_5th_values_100 = get_xy("chi2_100days_pull_bayes.txt")
    #unique_4th_values_200, min_5th_values_200 = get_xy("chi2_200days_pull_bayes.txt")
    #unique_4th_values_300, min_5th_values_300 = get_xy("chi2_300days_pull_bayes.txt")
    #unique_4th_values_400, min_5th_values_400 = get_xy("chi2_400days_pull_bayes.txt")
    #unique_4th_values_500, min_5th_values_500 = get_xy("chi2_500days_pull_bayes.txt")
    #unique_4th_values_1000, min_5th_values_1000 = get_xy("chi2_1000days_pull_bayes.txt")
   # unique_4th_values_2000, min_5th_values_2000 = get_xy("chi2_2000days_bayes.txt")

    unique_4th_values_100, min_5th_values_100 = get_xy("chi2_100days_sin2_th13_free.txt")
    unique_4th_values_200, min_5th_values_200 = get_xy("chi2_200days_sin2_th13_free.txt")
    unique_4th_values_300, min_5th_values_300 = get_xy("chi2_300days_sin2_th13_free.txt")
    unique_4th_values_400, min_5th_values_400 = get_xy("chi2_400days_sin2_th13_free.txt")
    unique_4th_values_500, min_5th_values_500 = get_xy("chi2_500days_sin2_th13_free.txt")
    unique_4th_values_600, min_5th_values_600 = get_xy("chi2_600days_sin2_th13_free.txt")
    unique_4th_values_700, min_5th_values_700 = get_xy("chi2_700days_sin2_th13_free.txt")
    unique_4th_values_800, min_5th_values_800 = get_xy("chi2_800days_sin2_th13_free.txt")
    unique_4th_values_900, min_5th_values_900 = get_xy("chi2_900days_sin2_th13_free.txt")
    unique_4th_values_1000, min_5th_values_1000 = get_xy("chi2_1000days_sin2_th13_free.txt")

    plt.plot(unique_4th_values_100, min_5th_values_100, linestyle='-', label='100d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_200, min_5th_values_200, linestyle='-', label='200d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_300, min_5th_values_300, linestyle='-', label='300d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_400, min_5th_values_400, linestyle='-', label='400d')# marker='.', markersize=10)

    #plt.plot(unique_4th_values_500, min_5th_values_500, linestyle='-', label='500d')# marker='.', markersize=10)
    #plt.plot(unique_4th_values_1000, min_5th_values_1000, linestyle='-', label='1000d')# marker='.', markersize=10)
    #plt.plot(unique_4th_values_2000, min_5th_values_2000, linestyle='-', label='2000d')# marker='.', markersize=10)
    #plt.xlabel('dm2_atm')
    #plt.ylabel('chi2_CNP')
    #plt.legend()

    plt.plot(unique_4th_values_500, min_5th_values_500, linestyle='-', label='500d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_600, min_5th_values_600, linestyle='-', label='600d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_700, min_5th_values_700, linestyle='-', label='700d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_800, min_5th_values_800, linestyle='-', label='800d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_900, min_5th_values_900, linestyle='-', label='900d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_1000, min_5th_values_1000, linestyle='-', label='1000d')# marker='.', markersize=10)
    plt.xlabel('$\Delta m^{2}_{31}$')
    plt.ylabel('$\Delta \chi^{2}_{CNP}$')
    plt.legend()
    plt.axhline(9, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
    plt.axhline(16, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 
    plt.axhline(25, color='black', linestyle='--', linewidth=0.8, alpha=0.5) 

    plt.show()


if __name__ == "__main__":
    main(None)
