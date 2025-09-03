import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_geo_sens():

    def U_Th_err(U_err, Th_err):
        coeff = -0.7
        return np.sqrt(U_err**2 + Th_err**2 + 2.*coeff*U_err*Th_err)

    years = [1, 3, 6, 10]
    data = {}
    data_j = {}
    #colorblind friendly
    colors = {
    1: '#0072B2',  # Sky Blue
    3: '#E69F00',  # Orange
    6: '#009E73',  # Bluish Green
    10: '#D55E00', # Yellow
    }

    df_j = pd.read_csv(f"Nov22_final_results/ScangeoU.txt", sep=' ', header=None)
    df_j2 = pd.read_csv(f"Nov22_final_results/ScangeoTh.txt", sep=' ', header=None)
    data_j[1] = (U_Th_err(df_j.iloc[:,1], df_j2.iloc[:, 1]),  df_j.iloc[:,0])
    data_j[3] = (U_Th_err(df_j.iloc[:,2], df_j2.iloc[:, 2]),  df_j.iloc[:,0])
    data_j[6] = (U_Th_err(df_j.iloc[:,3], df_j2.iloc[:, 3]),  df_j.iloc[:,0])
    data_j[10] = (U_Th_err(df_j.iloc[:,4], df_j2.iloc[:, 4]), df_j.iloc[:,0])
    for year in years:
        try:
            df = pd.read_csv(f"Nov22_final_results/fit_results_geo_UThfree_NorP_free_NO-True_{year}years_411bins_minuit.txt", sep=' ')
            uncertainty =U_Th_err(df['NgeoU_err'],df['NgeoTh_err'])*100.
            tnu = df['geo_tnu']
            data[year] = (uncertainty, tnu)
        except Exception as e:
            print(f"Error reading file for {year} years: {e}")

    plt.figure(figsize=(10, 8))
    for year, (uncertainty, tnu) in data_j.items():
        plt.plot(tnu, uncertainty, label=f"{year} years", color=colors[year], marker='o')
    for year, (uncertainty, tnu) in data.items():
        plt.plot(tnu, uncertainty, marker='o', color=colors[year],linestyle=':')
    plt.axvspan(25, 45, color='lightgoldenrodyellow', alpha=0.6, label="Expected signal at JUNO")
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='J\u00fclich/GSI'),
        Line2D([0], [0], color='black', linestyle=':', label='Irvine')
    ]
    handles, labels = plt.gca().get_legend_handles_labels()

    combined_handles = handles + custom_lines
    combined_labels = labels + [line.get_label() for line in custom_lines]

    plt.legend(combined_handles, combined_labels, framealpha=1, fontsize=16, loc='upper right')

    plt.ylabel("Uncertainty [%]", fontsize=18)
    plt.xlabel("Total geoneutrino [$^{238}$U + $^{232}$Th] rate [TNU]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title('Uncertainty on $^{238}$U  + $^{232}$Th (Th/U free)', fontsize=18)
    plt.tight_layout()
    #plt.savefig('geoTh_total_precision.png')
    plt.show()


if __name__ == '__main__':
    #plot_fit_results()
    #plot_mantle2D()
    plot_geo_sens()
