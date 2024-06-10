import matplotlib.pyplot as plt
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

def main(argv):

    unique_4th_values_100, min_5th_values_100 = get_xy("chi2_100days_pull_bayes.txt")
    unique_4th_values_200, min_5th_values_200 = get_xy("chi2_200days_pull_bayes.txt")
    unique_4th_values_300, min_5th_values_300 = get_xy("chi2_300days_pull_bayes.txt")
    unique_4th_values_400, min_5th_values_400 = get_xy("chi2_400days_pull_bayes.txt")
    #unique_4th_values_500, min_5th_values_500 = get_xy("chi2_500days_pull_bayes.txt")
    #unique_4th_values_1000, min_5th_values_1000 = get_xy("chi2_1000days_pull_bayes.txt")
   # unique_4th_values_2000, min_5th_values_2000 = get_xy("chi2_2000days_bayes.txt")
    plt.plot(unique_4th_values_100, min_5th_values_100, linestyle='-', label='100d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_200, min_5th_values_200, linestyle='-', label='200d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_300, min_5th_values_300, linestyle='-', label='300d')# marker='.', markersize=10)
    plt.plot(unique_4th_values_400, min_5th_values_400, linestyle='-', label='400d')# marker='.', markersize=10)
    #plt.plot(unique_4th_values_500, min_5th_values_500, linestyle='-', label='500d')# marker='.', markersize=10)
    #plt.plot(unique_4th_values_1000, min_5th_values_1000, linestyle='-', label='1000d')# marker='.', markersize=10)
    #plt.plot(unique_4th_values_2000, min_5th_values_2000, linestyle='-', label='2000d')# marker='.', markersize=10)
    plt.xlabel('dm2_atm')
    plt.ylabel('chi2_CNP')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(None)
