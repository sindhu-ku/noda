import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def pmop():
    filenames = ["iminuit", "iminuit_nuosc"]
    df = []
    for i, name in enumerate(filenames):
        print(i, name)
        df.append(pd.read_csv(f"fit_results_{name}.txt", delimiter = ' '))


        #
        parameters = ['sin2_12_err', 'sin2_13_err', 'dm2_21_err', 'dm2_31_err']
        # #
        df[0] = df[0][df[0]['unc'] == 'stat+all']
        df[1] = df[1][df[1]['unc'] == 'stat+all']
        y = []
        param = []
        for p in parameters:
            param.append(p.strip('_err'))
            #    g = (df[0][p].values[0]*df[0][p.strip('_err')].values[0])/100.

            g = df[0][p].values[0]
            m = df[1][p].values[0]
            y.append(abs(g-m)*100./g)
            plt.plot(param,  y, marker ='o')
            plt.title('PDG2022 update changes, 20 years')
            plt.ylabel('Rel. change in precision % [%]')
            #axs[i].set_xlabel('Changes')
            plt.axhline(y=1, linestyle=':')

            plt.show()

def single(filename):
    df = pd.read_csv(f"{filename}.txt", delimiter = ' ')
    unc_list = []
    change = []
    for i in range(len(df['unc'])):
        if i ==0:
            unc_list.append(df['unc'][i])
            change.append(0)
        else:
            unc = df['unc'][i].replace(df['unc'][i-1], '')
            if unc == '+me': unc = '+mat-eff'
            if unc == '+r2': unc = '+rea-uncorr'
            if unc == '+crel': unc = '+rea-corr'
            unc_list.append(unc)
            change.append((df['Ngeo_err'][i] - df['Ngeo_err'][i-1])*100./df['Ngeo_err'][i-1])
    return unc_list, df['Ngeo_err'], change

def nmo():
    NO_stat = []
    NO_syst = []
    IO_stat = []
    IO_syst = []
    JUNO_NO_DYB = []
    JUNO_NO_TAO = []
    years = []
    for i in range(1, 21):
        NO_stat.append(pd.read_csv(f"NMO_data_Oct3/fit_results_NMO_NorP_pull_NO-True_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        NO_syst.append(pd.read_csv(f"NMO_data_Oct3/syst/fit_results_NMO_NorP_pull_NO-True_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        IO_stat.append(pd.read_csv(f"NMO_data_Oct3/fit_results_NMO_NorP_pull_NO-False_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        IO_syst.append(pd.read_csv(f"NMO_data_Oct3/syst/fit_results_NMO_NorP_pull_NO-False_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        JUNO_NO_TAO.append(pd.read_csv(f"NMO_JUNO-only/fit_results_TAO_NorP_pull_NO-True_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        JUNO_NO_DYB.append(pd.read_csv(f"NMO_JUNO-only/fit_results_NorP_pull_NO-True_{i}years_411bins_minuit.txt", delimiter = ' ')['sigma'][0])
        years.append(i)
    print(JUNO_NO_DYB)
    fig, ax = plt.subplots()

    ##ax.plot(years, NO_stat, color='red', linestyle=':', marker='o', markersize=3, label='NO stat. only')
    ax.plot(years, NO_syst, color='red', marker='o', markersize=3, label='JUNO + TAO')
    ax.plot(years, JUNO_NO_TAO, color='black', marker='o', markersize=3, label='JUNO only TAO unc')
    ax.plot(years, JUNO_NO_DYB, color='blue', marker='o', markersize=3, label='JUNO only DYB unc')

    # ax.plot(years, IO_stat, color='blue', linestyle=':', marker='o', markersize=3, label='IO stat. only')
    # ax.plot(years, IO_syst, color='blue', marker='o', markersize=3, label='IO stat.+syst.')

    ax.set_xlabel('Years', fontsize=18)
    ax.set_ylabel('$\sigma$', fontsize=18)
    ax.grid(True, which='major', linestyle='-', linewidth=1)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5)

    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_tick_params(right=True, which='both')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)

    ax.legend(fontsize=14)
    #plt.ylim(0, 6)
    plt.xlim(0, 20)
    plt.title('Assuming NO (stat+syst)', fontsize=18)
    plt.show()


def geo():
    df1 = []
    df2 = []
    df3 = []
    df4 = []
    values1=[]
    values2=[]
    values3=[]
    values4=[]
    years=[]
    err=[]
    values2 = []
    for y in range(1,11,1):
        df1.append(pd.read_csv(f"Geo_zhao/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt", delimiter = r'\s+'))
        df2.append(pd.read_csv(f"Geo_Juelich/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt", delimiter = r'\s+'))
        df3.append(pd.read_csv(f"Geo_MC_zhao/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt", delimiter = r'\s+'))
        df4.append(pd.read_csv(f"Geo_withSNFnoneq/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt", delimiter = r'\s+'))
        if y == 1 or y ==3 or y ==6 or y ==10:
            years.append(y)
            values1.append(df1[y-1]['Ngeo_err'].iloc[-1]*100.)
            values2.append(df2[y-1]['Ngeo_err'].iloc[-1]*100.)
            values3.append(df3[y-1]['Ngeo_err'].iloc[-1]*100.)
            values4.append(df4[y-1]['Ngeo_err'].iloc[-1]*100.)
    z_values = [22.08, 13.29, 9.93, 8.19]
    j_values = [22.84331412860796, 13.801369523627526, 10.324075341056187, 8.51217810101987]
    #j_values = [21.65, 13.22, 10.01, 8.36]
    j_values_toy = [22.58, 13.65, 10.22, 8.56]


    z_diff_noda_ana = list((np.array(z_values) - np.array(values1))*100./np.array(z_values))
    z_diff_zhao_ana = list((np.array(z_values) - np.array(values2))*100./np.array(z_values))
    z_diff_mc = list((np.array(z_values) - np.array(values3))*100./np.array(z_values))
    z_diff_ci = list((np.array(z_values) - np.array(values4))*100./np.array(z_values))
    j_diff = list((np.array(j_values) - np.array(values2))*100./np.array(j_values))
    j_diff_toy = list((np.array(j_values_toy) - np.array(values2))*100./np.array(j_values_toy))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # First plot: Square-like aspect ratio
#    ax1.plot(years, values4, marker='o', label='Irvine common inputs')
    #ax1.plot(years, values1, marker='o', label='Irvine NODA analytical')
    ax1.plot(years, values2, marker='o', label='Irvine')
    #ax1.plot(years, values3, marker='o', label='Irvine MC')
    ax1.plot(years, z_values, marker='o', label='Zhao')
    #ax1.plot(years, j_values, marker='o', label='Juelich Asimov')
    #ax1.plot(years, j_values_toy, marker='o', label='Juelich toyMC')

    ax1.set_ylabel('Precision %', fontsize=18)
    ax1.grid()
    ax1.legend(fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_xticks(years)
    ax1.tick_params(axis='x', labelsize=16)

    # Second plot: Rectangular aspect ratio
#    ax2.plot(years, z_diff_ci, marker='o', label="Irvine common inputs vs Zhao")
    ax2.plot(years, z_diff_noda_ana, marker='o', label="Irvine vs Zhao")
    #ax2.plot(years, z_diff_zhao_ana, marker='o', label="Irvine Zhao analytical vs Zhao")
    #ax2.plot(years, z_diff_mc, marker='o', label="Irvine MC vs Zhao")
    #ax2.plot(years, j_diff, marker='o', label="Irvine Asimov vs Juelich Asimov")
    #ax2.plot(years, j_diff_toy, marker='o', label="Irvine Asimov vs Juelich toyMC")

    ax2.set_ylabel('Rel. difference %', fontsize=18)
    ax2.set_xlabel('Time (years)', fontsize=18)
    ax2.grid()
    ax2.legend(fontsize=16)
    ax2.tick_params(axis='both', labelsize=16)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    #plt.savefig('Juelich_comp.png')
def main():
    # unc_list, geo, change = single('fit_results_NorP_free_NO-True_6years_411bins_minuit_NL_Oct14')
    # unc_list_new, geo_new, change_new = single('fit_results_NorP_free_NO-True_6years_411bins_minuit_newNL')
    #
    # plt.figure(figsize=(8,20))
    #
    # plt.plot(unc_list, geo*100., marker ='o', label='old NL')
    # #plt.plot(unc_list_new, geo_new*100., marker ='o', label='new J22 NL')
    # plt.ylabel('Precision %', fontsize=18)
    # plt.xlabel('Uncertainty', fontsize=18)
    # plt.xticks(rotation=90, fontsize=16)
    # plt.yticks(fontsize=16)
    # #plt.legend(fontsize=18)
    # #plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    # plt.show()
    #
    # plt.figure(figsize=(8,20))
    #
    # plt.plot(unc_list, change, marker ='o', label='old NL')
    # #plt.plot(unc_list_new, change_new, marker ='o', label='new J22 NL')
    # plt.ylabel('Rel. change in precision % [%]', fontsize=18)
    # plt.xlabel('Uncertainty', fontsize=18)
    # plt.xticks(rotation=90, fontsize=16)
    # plt.yticks(fontsize=16)
    # #plt.legend(fontsize=18)
    # #plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2)
    # plt.show()

    geo()
if __name__ == "__main__":
  main()
