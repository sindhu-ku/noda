import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

folder_labels = {
    "Geo_Juelich": "C.I. (w/o rea corr)", #"Technote conditions w/o SNF and non-equilibrium",
    "Geo_withSNFnoneq": "Reactor corrections", #"with SNF and non-equilibrium",
    "Geo_PDG2022": "PDG 2022",
    "Geo_withnewNL": "J22 NL curve",
    "Geo_ana": "NODA geo analytical",
    "Geo_MC": "MC geo"
}

def U_Th_err(U_err, Th_err):
    U_err = U_err*0.924
    Th_err = Th_err*0.276
    coeff = -0.7
    return np.sqrt(U_err**2 + Th_err**2 + 2.*coeff*U_err*Th_err)
tnu_to_cpd = 0.032387707
def plot_fit_results():
    labels = []
    results = []
    plt.figure(figsize=(10, 6))
    for folder, label in folder_labels.items():
        files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        year_fit_data = []
        for file in sorted(files):

            year = int(file.split('_')[5].replace('years', ''))
            if year != 6: continue
            labels.append(label)
            df = pd.read_csv(os.path.join(folder, file), delimiter=' ')
            fit_result = df['Ngeo_err'].iloc[-1]*100
            results.append(fit_result)

            year_fit_data.append((year, fit_result))
            year_fit_data.sort(key=lambda x: x[0])

        years, fit_results = zip(*year_fit_data)
        #plt.plot(years, fit_results, label=label, marker='o')
    results_new = [0]*len(labels)
    for i, r in enumerate(results):
        if i == 0:
            results_new[i] = 0
        else:
            results_new[i] = (results[i] - results[0])*100./results[0]

    plt.plot(labels, results_new, marker='o')
    #plt.xlabel('Years', fontsize=18)
    plt.ylabel('Relative change in precision %', fontsize=18)
#    plt.legend(fontsize=18)
    plt.xticks(rotation =30, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #plt.savefig('tot_geo_pres.png')
def plot_hesse_minos():
    years =[1, 3, 6, 10]
    colors = {
    1: '#0072B2',  # Sky Blue
    3: '#E69F00',  # Orange
    6: '#009E73',  # Bluish Green
    10: '#D55E00', # Yellow
    }
    data = {}
    for year in years:
        try:
            df = pd.read_csv(f"June11/fit_results_geo_CNP_free_NO-True_{year}years_561bins_minuit.txt", sep=' ')

            tnu = df['geo_tnu']
            mask = tnu % 10 == 0
            tnu = tnu[mask]
            #print(tnu)
            unc_h = df['Ngeo_err'][mask]
            unc = df[['Ngeo_merr', 'Ngeo_perr']][mask].abs().max(axis=1)
            plt.plot(tnu, (unc-unc_h)*100., color = colors[year], marker='o', label=f'{year} years')
        except Exception as e:
            print(f"Error reading file for {year} years: {e}")

    plt.legend()
    plt.xlabel("Geo rate (TNU)", fontsize=18)
    plt.ylabel("Minos - Hesse %", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.show()

    plt.legend()
    plt.ylabel("Uncertainty [%]", fontsize=18)
    plt.xlabel("Total geoneutrino [$^{238}$U + $^{238}$Th] rate [TNU]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title('Uncertainty on [$^{238}$U + $^{238}$Th] (Th/U fixed)', fontsize=18)
    plt.tight_layout()
    plt.ylim(0, 300)
    #plt.savefig('geoTh_total_precision.png')
    plt.show()
def plot_toy_asimov():
    years =[1, 3, 6, 10]
    colors = {
    1: '#0072B2',  # Sky Blue
    3: '#E69F00',  # Orange
    6: '#009E73',  # Bluish Green
    10: '#D55E00', # Yellow
    }
    data = {}
    plt.figure(figsize=(10, 8))
    for year in years:
        try:
            df = pd.read_csv(f"June11/fit_results_geo_CNP_free_NO-True_{year}years_561bins_minuit.txt", sep=' ')
            df_toy = pd.read_csv(f"June11/geo_minos_toy_error_{year}years.txt", sep=' ')
            #df = df[df['unc'] != 'stat']
            tnu = df['geo_tnu']
            mask = tnu % 10 == 0
            #mask = tnu < 40 == 0
            tnu = tnu[mask]
            #print(tnu)
            #print(tnu)
            #unc = df['Ngeo_err'][mask]
            unc = df[['Ngeo_merr', 'Ngeo_perr']][mask].abs().max(axis=1)
            #unc =df['Ngeo_err'][mask] # U_Th_err(df['NgeoU_err'][mask], df['NgeoTh_err'][mask])*100./(tnu*tnu_to_cpd)
            unc_toy = df_toy['median'] #U_Th_err(df_toy['medianU'], df_toy['medianTh'])*100./(df_toy['crust_tnu']*tnu_to_cpd)
            unc_toy_sp = df_toy['sigma_p'] # U_Th_err(df_toy['sigma_pU'], df_toy['sigma_pTh'])*100./(df_toy['crust_tnu']*tnu_to_cpd)
            unc_toy_sm = df_toy['sigma_m'] #U_Th_err(df_toy['sigma_mU'], df_toy['sigma_mTh'])*100./(df_toy['crust_tnu']*tnu_to_cpd)
            data[year] = (unc, unc_toy, unc_toy_sp, unc_toy_sm, tnu)
            # unc = df['Ngeo_err']*100.
            # unc_toy = df_toy['median']*100.
            # unc_toy_sp = df_toy['sigma_p']*100.
            # unc_toy_sm = df_toy['sigma_m']*100.
            # tnu = df['geo_tnu']
            # data[year] = (unc, unc_toy, unc_toy_sp, unc_toy_sm, tnu)
        except Exception as e:
            print(f"Error reading file for {year} years: {e}")
    # plt.plot(tnu, (unc-unc_h)*100., color = colors[year], linestyle=':', marker='o', label='Asimov hesse')
    # plt.xlabel("Geo rate (TNU)", fontsize=18)
    # plt.ylabel("Minos - Hesse %", fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.grid()
    # plt.show()
    for year, (unc, unc_toy, unc_toy_sp, unc_toy_sm, tnu) in data.items():
        print(unc, tnu, unc_toy)#*100./(tnu*tnu_to_cpd))
        #plt.plot(tnu, unc_h*100., color = colors[year], linestyle=':', marker='o', label='Asimov hesse') #label=f"{year} years")
        plt.plot(tnu, unc*100., color = colors[year], linestyle='-', marker='o', label=f"{year} years")
        #plt.plot(tnu, unc_toy*100., color = colors[year], linestyle='--', marker='o', label='Toy hesse')
        plt.errorbar(tnu, unc_toy*100., yerr=[unc_toy_sm*100., unc_toy_sp*100.], marker='o', fill_style=None, color=colors[year],  capsize=3, linestyle='--')
    plt.axvspan(35, 53, color='lightgoldenrodyellow', alpha=0.6, label="Expected signal at JUNO")
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='Asimov'),
        Line2D([0], [0], color='black', linestyle='--', label='Toy')
    ]
    handles, labels = plt.gca().get_legend_handles_labels()

    combined_handles = handles + custom_lines
    combined_labels = labels + [line.get_label() for line in custom_lines]

    plt.legend(combined_handles, combined_labels, framealpha=1, fontsize=16, loc='upper right')

    #plt.legend()
    plt.ylabel("Uncertainty [%]", fontsize=18)
    plt.xlabel("Total geoneutrino [$^{238}$U + $^{238}$Th] rate [TNU]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title('Uncertainty on [$^{238}$U + $^{238}$Th] (Th/U fixed)', fontsize=18)
    plt.tight_layout()
    plt.ylim(0, 300)
    #plt.savefig('geoTh_total_precision.png')
    plt.show()

def plot_toy_asimov_new():
    years =[1, 3, 6, 10]
    colors = {
    1: '#0072B2',  # Sky Blue
    3: '#E69F00',  # Orange
    6: '#009E73',  # Bluish Green
    10: '#D55E00', # Yellow
    }
    data = {}
    plt.figure(figsize=(10, 8))
    for year in years:
        try:
            df = pd.read_csv(f"Nov22_final_results/fit_results_geoNorP_free_NO-True_{year}years_411bins_minuit.txt", sep=' ')
            df_toy = pd.read_csv(f"Nov22_final_results/geo_toy_error_{year}years_new.txt", sep=' ')

            tnu = df['geo_tnu']
            mask = tnu % 10 == 0
            tnu = tnu[mask]
            #print(tnu)
            unc =df['Ngeo_err'][mask] # U_Th_err(df['NgeoU_err'][mask], df['NgeoTh_err'][mask])*100./(tnu*tnu_to_cpd)
            unc_toy = df_toy['median'] #U_Th_err(df_toy['medianU'], df_toy['medianTh'])*100./(df_toy['crust_tnu']*tnu_to_cpd)
            unc_toy_new = df_toy['sigma'] # U_Th_err(df_toy['sigma_pU'], df_toy['sigma_pTh'])*100./(df_toy['crust_tnu']*tnu_to_cpd)
            data[year] = (unc, unc_toy, unc_toy_new, tnu)
            # unc = df['Ngeo_err']*100.
            # unc_toy = df_toy['median']*100.
            # unc_toy_sp = df_toy['sigma_p']*100.
            # unc_toy_sm = df_toy['sigma_m']*100.
            # tnu = df['geo_tnu']
            # data[year] = (unc, unc_toy, unc_toy_sp, unc_toy_sm, tnu)
        except Exception as e:
            print(f"Error reading file for {year} years: {e}")
    for year, (unc, unc_toy, unc_toy_new, tnu) in data.items():
        print(unc, tnu, unc_toy, unc_toy_new)#*100./(tnu*tnu_to_cpd))
        plt.plot(tnu, unc, color = colors[year], linestyle='-', marker='o', label=f"{year} years")
        plt.errorbar(tnu, unc_toy, yerr=unc_toy_new, marker='o', fill_style=None, color=colors[year],  capsize=3, linestyle='--')
    plt.axvspan(25, 45, color='lightgoldenrodyellow', alpha=0.6, label="Expected signal at JUNO")
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='Asimov'),
        Line2D([0], [0], color='black', linestyle='--', label='Toy')
    ]
    handles, labels = plt.gca().get_legend_handles_labels()

    combined_handles = handles + custom_lines
    combined_labels = labels + [line.get_label() for line in custom_lines]

    plt.legend(combined_handles, combined_labels, framealpha=1, fontsize=16, loc='upper right')

    plt.ylabel("Uncertainty [%]", fontsize=18)
    plt.xlabel("Total geoneutrino [$^{238}$U + $^{238}$Th] rate [TNU]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.title('Uncertainty on [$^{238}$U + $^{238}$Th] (Th/U fixed)', fontsize=18)
    plt.tight_layout()
    #plt.ylim(0, 300)
    #plt.savefig('geoTh_total_precision.png')
    plt.show()
def plot_geo_sens():

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

    df_j = pd.read_csv(f"Nov22_final_results/Scangeo.txt", sep=' ', header=None)
    for i, y in enumerate(years):
        data_j[y] = (df_j.iloc[:,i],  df_j.iloc[:,0])
    for year in years:
        try:
            df = pd.read_csv(f"Nov22_final_results/fit_results_geoNorP_free_NO-True_{year}years_411bins_minuit.txt", sep=' ')
            uncertainty =df['Ngeo_err']*100.
            tnu = df['geo_tnu']
            data[year] = (uncertainty/(tnu*tnu_to_cpd), tnu)
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
    plt.title('Uncertainty on $^{238}$U  + $^{232}$Th (Th/U fixed)', fontsize=18)
    plt.tight_layout()
    #plt.savefig('geoTh_total_precision.png')
    plt.show()

def plot_mantle2D():
    tnu_conv = 31.113683 #30.875912
    df = pd.read_csv('Nov22_final_results/fit_results_geo_mantleGC_Jan14_NorP_free_NO-True_6years_411bins_minuit.txt', header=None, delimiter=' ')
    plt.figure(figsize=(9, 8))
    plt.tricontourf(df[16]*tnu_conv, df[17]*100., df[14], levels=50, cmap='viridis')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Discovery potential ($\sigma$)', fontsize=18)
    sigma_levels = [1.0, 1.5] #[1.5, 2.0,2.5, 3.0, 3.5]  # Set your desired sigma levels here
    contour_lines = plt.tricontour(df[16]*tnu_conv,  df[17] * 100, df[14], levels=sigma_levels, colors='black', linestyles='--')
    manual_positions = [(35, 11), (40, 17)]
    #manual_positions = [(45, 21), (40, 17), (35, 14), (31, 12),(25,8)]
    plt.clabel(contour_lines, inline=True, fontsize=14, fmt='%1.1f$\sigma$', manual=manual_positions)  # Label the contours
    plt.xlim(25, 50)
    plt.ylim(5, 25)

    x, y = 42.3, 10
    plt.scatter(x, y, color='red', label='JULOC-I', zorder=5)  # Plot the point
    plt.text(x-1, y-1, "JULOC-I", fontsize=14, color='red', zorder=5)  # Add the label

    plt.xlabel("Crust rate [TNU]", fontsize=18)
    plt.ylabel("Crust uncertainty %", fontsize=18)
    plt.title("Geochemical Mantle (8 TNU), 6 years", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

def mantle_2D_comp():
    tnu_conv = 31.113683
    dfs = pd.read_csv('Nov22_final_results/fit_results_geo_mantleGC_Jan14_NorP_free_NO-True_6years_411bins_minuit.txt', header=None, delimiter=' ')
    dfc = pd.read_csv('~/Downloads/discovery_potential_mantle.txt', header=None, delimiter=' ')
    #dfc.sort_values(by=[dfc.columns[1], dfc.columns[0]], key=pd.to_numeric, ascending=False)
    dfc = dfc.set_index(dfc.columns[1], append=True).sort_index(level=1).reset_index(level=1)
    #dfc.sort_index(axis=1, inplace=True)
    print(list(dfs[14]))
    print(list(dfc[2]))
    diff = (np.array(dfs[14]) - np.array(dfc[2]))*100./np.array(dfs[14])
    print(diff)
    # plt.figure(figsize=(9, 8))
    # plt.tricontourf(dfs[16]*tnu_conv, dfs[17]*100., diff, levels=50, cmap='viridis')
    # #print(dfs[14], dfc[2], list(abs(dfs[14]-dfc[2])))
    #
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=16)
    # cbar.set_label('Rel. diff. in discovery potential (%)', fontsize=18)

    plt.hist(diff, color='steelblue', bins=25)
    #
    # plt.xlim(25, 50)
    # plt.ylim(5, 25)

    plt.xlabel("$\Delta \sigma$/$\sigma$ (%)", fontsize=18)
    plt.ylabel("#", fontsize=18)
    plt.title("Relative difference between Irvine and Juelich [%]", fontsize=18)
    # plt.xlabel("Crust rate [TNU]", fontsize=18)
    # plt.ylabel("Crust uncertainty %", fontsize=18)
    #plt.title("Geochemical Mantle (8 TNU), 6 years", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

def plot_mantle_all():
    CC = []
    GC = []
    GD = []
    CC_f = []
    GC_f = []
    GD_f = []
    years = np.arange(1, 11, 1)
    for y in years:
        df = pd.read_csv(f'Geo_mantle/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt', header=None, delimiter=' ')
        df1 = pd.read_csv(f'Geo_mantle_OPfixed/fit_results_NorP_free_NO-True_{y}years_411bins_minuit.txt', header=None, delimiter=' ')
        CC.extend(df[14][df[15] == 'CC'].tolist())
        GC.extend(df[14][df[15] == 'GC'].tolist())
        GD.extend(df[14][df[15] == 'GD'].tolist())
        CC_f.extend(df1[6][df1[7] == 'CC'].tolist())
        GC_f.extend(df1[6][df1[7] == 'GC'].tolist())
        GD_f.extend(df1[6][df1[7] == 'GD'].tolist())

    plt.figure(figsize=(10,11))
    plt.plot(years, CC, marker='o', color='blue', label='Cosmochemical')
    plt.plot(years, CC_f, marker='o',color='blue', linestyle=':', label='Cosmochemical OP fixed')
    plt.plot(years, GC,marker='o', color='red', label='Geochemical ')
    plt.plot(years, GC_f, marker='o',color='red', linestyle=':', label='Geochemical OP fixed')
    plt.plot(years, GD, marker='o', color='magenta',label='Geodynamical')
    plt.plot(years, GD_f, marker='o',color='magenta', linestyle=':', label='Geodynamical OP fixed')
    plt.title('Mantle discovery potential (JULOC-I crust)', fontsize=18)
    plt.xlabel('Time [y]', fontsize=18)
    plt.ylabel('$\sigma$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=16, loc='center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #plot_fit_results()
    #plot_mantle2D()
    #mantle_2D_comp()
    #plot_geo_sens()
    #plot_hesse_minos()
    plot_toy_asimov()
