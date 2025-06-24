import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
def load_data(filename):
    df = pd.read_csv(filename, sep=' ')  # Adjust delimiter if necessary
    stat = df.iloc[0]['Ngeo_err']  # Extract stat from the first row
    print(stat)
    df = df.iloc[1:].copy()  # Remove the first row


    # Compute modified uncertainty
    df['modified_unc'] = np.sqrt(df['Ngeo_err']**2 - stat**2)
    df['modified_unc'] = df['modified_unc'].fillna(0)
    print(df['modified_unc'])
    df['unc'] = df['unc'].astype(str).str.replace('stat+', '', regex=False)
    selected_uncertainties = ['bg', 'rea_shape', 'lsnl']

    # Filter the DataFrame
    df = df[df['unc'].isin(selected_uncertainties)]

    return df

# Plot function
def plot_data(df):
    plt.figure(figsize=(16, 10))
    plt.plot(df['unc'], df['modified_unc']*100., marker='o', markersize=10, linestyle='-')
    plt.xlabel('Systematic uncertainty', fontsize=28)
    plt.ylabel('Contribution to geoneutrino precision [%]', fontsize=28)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22, rotation=45)

    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    filename = "Mar15/fit_results_geo_NorP_free_NO-True_6years_411bins_minuit.txt"  # Change to your actual file
    df = load_data(filename)
    plot_data(df)
