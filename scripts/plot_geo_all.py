import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder_labels = {
    "Geo_Juelich": "C.I. (w/o rea corr)", #"Technote conditions w/o SNF and non-equilibrium",
    "Geo_withSNFnoneq": "Reactor corrections", #"with SNF and non-equilibrium",
    "Geo_PDG2022": "PDG 2022",
    "Geo_withnewNL": "J22 NL curve",
    "Geo_ana": "NODA geo analytical",
    "Geo_MC": "MC geo"
}

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

def plot_geo_sens():
    years = [1, 3, 6, 10]
    data = {} 

    for year in years:
        try:
            df = pd.read_csv(f"Geo_Nov20/fit_results_geo_UThfree_NorP_free_NO-True_{year}years_411bins_minuit.txt", sep=' ', header=None)
            uncertainty = df.iloc[:, 25]*100.
            tnu = df.iloc[:, 29]
            data[year] = (uncertainty, tnu)
        except Exception as e:
            print(f"Error reading file for {year} years: {e}")

    plt.figure(figsize=(10, 6))
    
    for year, (uncertainty, tnu) in data.items():
        plt.plot(tnu, uncertainty, label=f"{year} years", marker='.')
    plt.axvspan(25, 45, color='forestgreen', alpha=0.3, label="Expected TNU at JUNO") 
    plt.ylabel("Uncertainty on $^{232}$Th rate [%]", fontsize=18)
    plt.xlabel("Total geoneutrino ($^{238}$U + $^{232}$Th) rate [TNU]", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig('geoTh_total_precision.png')
    #plt.show() 
        
def plot_mantle2D():
    tnu_conv = 30.875912
    df = pd.read_csv('Geo_mantle_2D/fit_results_NorP_free_NO-True_6years_411bins_minuit.txt', header=None, delimiter=' ')
    plt.figure(figsize=(8, 10))
    plt.tricontourf(df[16]*tnu_conv, df[17]*100., df[14], levels=50, cmap='viridis')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Discovery potential ($\sigma$)', fontsize=18)
    sigma_levels = [1.0, 1.4, 1.8, 2.2]  # Set your desired sigma levels here
    contour_lines = plt.tricontour(df[16]*tnu_conv,  df[17] * 100, df[14], levels=sigma_levels, colors='black', linestyles='--')
    #manual_positions = [(26, 5.5), (35, 10), (40, 20)]
    manual_positions = [(27, 6), (33, 11), (40, 14),(48,18)]
    plt.clabel(contour_lines, inline=True, fontsize=14, fmt='%1.1f$\sigma$', manual=manual_positions)  # Label the contours
    plt.xlim(25, 50)
    plt.ylim(5, 25)

    plt.xlabel("Crust rate [TNU]", fontsize=18)
    plt.ylabel("Crust uncertainty %", fontsize=18)
    plt.title("GC Mantle (8.8 TNU), 6 years, OP free", fontsize=18)
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
    plot_geo_sens()
