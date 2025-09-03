import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import norm
import os
import re
# Asimov values
asimov_values = {
    "sin2_12": 0.307,
    "sin2_13": 0.022,
    "dm2_21": 7.53e-5,
    "dm2_31":  0.0025283000,
    "Nrea": 1.0,
    "Ngeo": 1.0
}

def create_pair_plot_PMOP(h5_file):
    # Read HDF5 data
    with h5py.File(h5_file, "r") as hdf:
        print(hdf['geo'])
        geo_data = np.array(hdf["geo"][:, :-4], dtype=float)
        print(geo_data[0])
        #print(h5_file.dtype)
        # Define column names corresponding to data structure
        columns = [
            "sin2_12", "sin2_12_err",
            "sin2_13", "sin2_13_err",
            "dm2_21", "dm2_21_err",
            "dm2_31", "dm2_31_err"
            # "inj_events", "unc",
            # "dchi2", "sigma"
        ]
        # columns = [
        #         "sin2_12", "sin2_12_err","sin2_12_merr",
        #         "sin2_13", "sin2_13_err","sin2_13_merr",
        #         "dm2_21", "dm2_21_err", "dm2_21_merr",
        #         "dm2_31", "dm2_31_err","dm2_31_merr",
        #         "Nrea", "Nrea_err", "Nrea_merr",
        #         "Ngeo", "Ngeo_err", "Ngeo_merr",
        # ]
        # Create a pandas DataFrame
        df = pd.DataFrame(geo_data, columns=columns)
    #df["Ngeo"] = df["Ngeo"].clip(lower=0)
    # Select variables of interest for the plot
    variables = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]

    # Filter the DataFrame to include only the selected variables
    plot_data = df[variables]

    print(plot_data)
    # Create pairplot
    pair_plot = sns.pairplot(
        plot_data,
        kind="hist",
        diag_kind="hist",
        diag_kws = {'bins':20},
        corner=True,  # Only show lower triangle of scatter plots
        height=3  # Increase figure size for better visibility
    )

    for i, var in enumerate(variables):
        # Compute the median and the 16th/84th percentiles for spread
        median_val = np.median(df[var])
        sigma_plus = np.percentile(df[var], 84) - median_val
        sigma_minus = median_val - np.percentile(df[var], 16)
        print(f"{var}: Median = {median_val}, +34% = {sigma_plus}, -34% = {sigma_minus}")
        digits = 3

        # Add median and spread text to diagonal plots
        ax = pair_plot.diag_axes[i]
        ax.annotate(f"Median: {median_val:.{digits}e}\n+34%: {sigma_plus:.{digits}e}\n-34%: {sigma_minus:.{digits}e}",
                    xy=(0.85, 0.95), xycoords="axes fraction", fontsize=10,
                    ha="center", va="center", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))

        # Overlay Asimov values as a vertical line and add explicit label
        if var in asimov_values:
            asimov_value = asimov_values[var]
            ax.axvline(asimov_value, color="red", linestyle="--", linewidth=1.5)

            # Calculate position slightly left of the Asimov line
            text_x = asimov_value * 1 if asimov_value > 0 else asimov_value * 1.02
            text_y = ax.get_ylim()[1] * 0.5  # Move text down a bit to avoid overlap

            ax.text(text_x, text_y, f"Bias w.r.t. Asimov: {abs(asimov_value - median_val) * 100. / asimov_value:.2f}%",
                    color="red", fontsize=10, ha="right", bbox=dict(facecolor="white", alpha=0.8))

    pair_plot.fig.suptitle("Pair Plot with Median, Spread, and Asimov Values", y=1.02)

    plt.show()
    with h5py.File(h5_file, "r") as hdf:
        dchi2 = np.sqrt(np.array(hdf["geo"][:,10], dtype=float))
        sin2_12 = np.array(hdf["geo"][:,0], dtype=float)
        sin2_13 = np.array(hdf["geo"][:,2], dtype=float)
        dm2_31 = np.array(hdf["geo"][:,6], dtype=float)
        dm2_21 = np.array(hdf["geo"][:,4], dtype=float)

        df = pd.DataFrame({
            'sin2_12': sin2_12,
            'sin2_13': sin2_13,
            'dm2_21': dm2_21,
            'dm2_32': dm2_31-dm2_21,
            '$\sigma$': dchi2
        })

        plt.hist(dchi2)
        plt.show()

        sns.set_context("notebook", font_scale=1.5)  # Options: "paper", "notebook", "talk", "poster"


        # Pairplot (color as discrete bins or with gradient)
        g = sns.pairplot(df, vars=['sin2_12', 'sin2_13', 'dm2_21', 'dm2_32'],
                     hue='$\sigma$', palette='viridis',
                     plot_kws={'alpha': 1, 's': 40, 'edgecolor': 'none'},height=3.5, aspect=1.2)

        sns.move_legend(g, "upper left", bbox_to_anchor=(0.2, 1))
        # g._legend.set_frame_on(False)
        # handles = g._legend_data.values()
        # labels = g._legend_data.keys()
        #
        # g.fig.legend(handles=handles, labels=labels, loc='upper left')
        # g.fig.subplots_adjust(top=1.5, bottom=0.08)
        #plt.suptitle("Pairwise Parameter Plots Colored by Normalized $\Delta \chi^2$", y=1.02)
        plt.tight_layout()
        plt.show()



        plt.hist2d(sin2_13, dchi2, bins =50)
        plt.ylabel('NMO $\sigma$')
        plt.xlabel('sin2_13 value')
        plt.show()


        plt.figure(figsize=(8, 6))

        sc = plt.scatter(sin2_13, dm2_31-dm2_21, c=dchi2, cmap='viridis', s=40)
        plt.xlabel('sin2_13')
        plt.ylabel('dm2_32')
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\Delta \chi^2$')

        plt.grid(True)
        plt.tight_layout()
        plt.show()
    print(f"dchi2: {np.sqrt(np.nanmedian(dchi2))}")

def create_pair_plot(h5_file):
    # Read HDF5 data
    with h5py.File(h5_file, "r") as hdf:
        print(hdf['geo'])
        geo_data = np.array(hdf["geo"][:, :-3], dtype=float)
        print(geo_data[0])
        #print(h5_file.dtype)
        # Define column names corresponding to data structure
        columns = [
            "sin2_12", "sin2_12_err",
            "sin2_13", "sin2_13_err",
            "dm2_21", "dm2_21_err",
            "dm2_31", "dm2_31_err",
            "Nrea", "Nrea_err",
            "Ngeo", "Ngeo_err"
        ]
        # columns = [
        #         "sin2_12", "sin2_12_err","sin2_12_merr",
        #         "sin2_13", "sin2_13_err","sin2_13_merr",
        #         "dm2_21", "dm2_21_err", "dm2_21_merr",
        #         "dm2_31", "dm2_31_err","dm2_31_merr",
        #         "Nrea", "Nrea_err", "Nrea_merr",
        #         "Ngeo", "Ngeo_err", "Ngeo_merr",
        # ]
        # Create a pandas DataFrame
        df = pd.DataFrame(geo_data, columns=columns)
    #df["Ngeo"] = df["Ngeo"].clip(lower=0)
    # Select variables of interest for the plot
    variables = ["sin2_12", "sin2_13", "dm2_21", "dm2_31", "Ngeo", "Nrea"]

    # Filter the DataFrame to include only the selected variables
    plot_data = df[variables]

    print(plot_data)
    # Create pairplot
    pair_plot = sns.pairplot(
        plot_data,
        kind="hist",
        diag_kind="hist",
        diag_kws = {'bins':10},
        corner=True,  # Only show lower triangle of scatter plots
        height=3  # Increase figure size for better visibility
    )

    for i, var in enumerate(variables):
        # Compute the median and the 16th/84th percentiles for spread
        median_val = np.median(df[var])
        sigma_plus = np.percentile(df[var], 84) - median_val
        sigma_minus = median_val - np.percentile(df[var], 16)
        print(f"{var}: Median = {median_val}, +34% = {sigma_plus}, -34% = {sigma_minus}")
        digits = 3

        # Add median and spread text to diagonal plots
        ax = pair_plot.diag_axes[i]
        ax.annotate(f"Median: {median_val:.{digits}e}\n+34%: {sigma_plus:.{digits}e}\n-34%: {sigma_minus:.{digits}e}",
                    xy=(0.85, 0.95), xycoords="axes fraction", fontsize=10,
                    ha="center", va="center", bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"))

        # Overlay Asimov values as a vertical line and add explicit label
        if var in asimov_values:
            asimov_value = asimov_values[var]
            ax.axvline(asimov_value, color="red", linestyle="--", linewidth=1.5)

            # Calculate position slightly left of the Asimov line
            text_x = asimov_value * 1 if asimov_value > 0 else asimov_value * 1.02
            text_y = ax.get_ylim()[1] * 0.5  # Move text down a bit to avoid overlap

            ax.text(text_x, text_y, f"Bias w.r.t. Asimov: {abs(asimov_value - median_val) * 100. / asimov_value:.2f}%",
                    color="red", fontsize=10, ha="right", bbox=dict(facecolor="white", alpha=0.8))

    pair_plot.fig.suptitle("Pair Plot with Median, Spread, and Asimov Values", y=1.02)

    plt.show()

def plot_ngeo_relative_error(stat, h5_file):
    # Read HDF5 data
    with h5py.File(h5_file, "r") as hdf:
        geo_data = np.array(hdf["geo"][:, :-3], dtype=float)
        # Define column names corresponding to data structure
        columns = [
            "sin2_12", "sin2_12_err",
            "sin2_13", "sin2_13_err",
            "dm2_21", "dm2_21_err",
            "dm2_31", "dm2_31_err",
            "Nrea", "Nrea_err",
            "Ngeo", "Ngeo_err"
        ]



        print(len(geo_data))

        # Create a pandas DataFrame
        df = pd.DataFrame(geo_data, columns=columns)
    with h5py.File("June11/fit_results_geo_CNP_10_NO-True_1years_561bins_minuit.hdf5", "r") as hdf:
        geo_data = np.array(hdf["geo"][:, :-3], dtype=float)
        # Define column names corresponding to data structure
        columns = [
            "sin2_12", "sin2_12_err",
            "sin2_13", "sin2_13_err",
            "dm2_21", "dm2_21_err",
            "dm2_31", "dm2_31_err",
            "Nrea", "Nrea_err",
            "Ngeo", "Ngeo_err"
        ]


        # Create a pandas DataFrame
        df1 = pd.DataFrame(geo_data, columns=columns)


    #df = df[df["Nmantle"] !=0]
    # Calculate Ngeo relative error
    ngeo_relative_error = abs(df["Ngeo_err"] / df["Ngeo"])
    ngeo_relative_error1 = abs(df1["Ngeo_err"] / df1["Ngeo"])
    mean_err = np.mean(df['Ngeo_err'])
    std_err = np.std(df['Ngeo_err'], ddof=1)
    print(std_err, mean_err)
    print((std_err / (0.005 * mean_err))**2)
    #ngeo_relative_error = ngeo_relative_error[ngeo_relative_error > 0]
    median = np.nanmedian(ngeo_relative_error)
    median1 = np.nanmedian(ngeo_relative_error1)
    sigma_plus = np.nanpercentile(ngeo_relative_error, 84) - median
    sigma_minus = median - np.nanpercentile(ngeo_relative_error, 16)
    # #
    filename = os.path.basename(h5_file)
    crust_rate_match = re.search(r"geo_CNP_(\d+)", filename)
    crust_rate = crust_rate_match.group(1) if crust_rate_match else "Unknown"
    output_file = f"June11/geo_minos_toy_error_{stat}years.txt"
    header = "crust_tnu median sigma_p sigma_m\n"
    row = f"{crust_rate} {median:.3f} {sigma_plus:.3f} {sigma_minus:.3f}\n"
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)
    with open(output_file, "a") as f:
        f.write(row)

    # plt.hist(df['inj_events'], density=True, alpha=0.3, label="Sindhu")
    # plt.hist(df1['inj_events'],density=True,  alpha=0.3, label="Cristobal" )
    # plt.axvline(x=16912, linestyle=':', label='Asimov', color='red')
    # plt.xlabel('Injected events')
    # plt.legend()
    # plt.show()
    print(f"Ngeo Relative Error: Median = {median:.3f}, +34% Sigma = {sigma_plus:.3f}, -34% Sigma = {sigma_minus:.3f}")
    #
    #Plot Ngeo relative error
    plt.figure(figsize=(8, 6))
    plt.hist(df1['Ngeo_err']*100, histtype='step', bins=1000, density=True, label='Hesse')
    plt.hist(df['Ngeo_err']*100, histtype='step', bins=50, density=True, label='Minos')

    plt.legend()
    plt.title('Ngeo err %')
    plt.yscale('log')
    plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.hist(df['Nrea'], bins=50, density=True, alpha=0.3)
    # plt.title('Ngeo')
    # #plt.yscale('log')
    # plt.show()
    # # #
    # plt.figure(figsize=(8, 6))
    # plt.hist(df['dm2_31'], bins=50, density=True, alpha=0.3, label="Asimov")
    # plt.hist(df1['dm2_31'], bins=50, density=True,  alpha=0.3, label="Toy")
    # plt.legend()
    # plt.title('dm2_31')
    # plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(ngeo_relative_error[ngeo_relative_error < 100] * 100., bins=100,histtype='step',alpha=0.7, label="Ngeo Relative Error Minos")
    plt.hist(ngeo_relative_error1[ngeo_relative_error1 < 100] * 100., bins=100, histtype='step', alpha=0.7, label="Ngeo Relative Error Hesse")
    plt.axvline(median * 100., color="orange", linestyle="--", label=f"Median Minos= {median * 100.:.1f}")
    plt.axvline(median1 * 100., color="red", linestyle="--", label=f"Median Hesse = {median1 * 100.:.1f}")
    #plt.axvline((median + sigma_plus) * 100., color="green", linestyle="--", label=f"+34% = {sigma_plus * 100.:.1f}")
    #plt.axvline((median - sigma_minus) * 100., color="red", linestyle="--", label=f"-34% = {sigma_minus * 100.:.1f}")
    plt.yscale('log')
    plt.xlabel("Ngeo_err / Ngeo [%]")
    plt.ylabel("No. of toys")
    plt.legend()
    plt.grid()
    plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.hist(abs(df3["Ngeo_merr"] - df3["Ngeo_err"])*100., bins=100)
    # plt.xlabel('abs(minos-hesse)[%]')
    # plt.show()
    #
    # plt.figure(figsize=(8, 6))
    # plt.hist(ngeo_relative_error[ngeo_relative_error < 100] * 100., bins=100, color='black', alpha=0.3,density=True,label="Sindhu")
    # plt.hist(ngeo_relative_error1[ngeo_relative_error1 < 100] * 100., bins=100, color='green', alpha=0.3, density=True, label="Cristobal")
    # #plt.hist(ngeo_relative_error1[ngeo_relative_error1 < 100] * 100., bins=1000, color='blue', alpha=0.3, density=True, label="N/P hesse errors")
    # #plt.hist(ngeo_relative_error2[ngeo_relative_error2 < 100] * 100., bins=1000, color='red', alpha=0.3, density=True, label="N/P minos errors")
    # plt.axvline(np.nanmedian(ngeo_relative_error)* 100., linestyle="--", color='black', label=f"Sindhu = {np.nanmedian(ngeo_relative_error)* 100:.1f}")
    # plt.axvline(np.nanmedian(ngeo_relative_error1)* 100., linestyle="--", color='green',label=f"Cristobal = { np.nanmedian(ngeo_relative_error1)* 100.:.1f}")
    # #plt.axvline(np.nanmedian(ngeo_relative_error1)* 100., linestyle="--", color='blue',label=f"Median N/P hesse = {np.nanmedian(ngeo_relative_error1)* 100:.1f}")
    # #plt.axvline(np.nanmedian(ngeo_relative_error2)* 100., linestyle="--", color='red',label=f"Median N/P minos = { np.nanmedian(ngeo_relative_error2)* 100.:.1f}")
    # plt.yscale('log')
    # plt.xlabel("Ngeo_err / Ngeo [%]")
    # plt.ylabel("No. of toys")
    # plt.legend()
    # plt.grid()
    # plt.show()
def plot_ngeo_relative_error_free(stat, h5_file):
    # Read HDF5 data
    with h5py.File(h5_file, "r") as hdf:
        geo_data = np.array(hdf["geo"][:, :-1], dtype=float)
        # Define column names corresponding to data structure
        columns = [
            "sin2_12", "sin2_12_err",
            "sin2_13", "sin2_13_err",
            "dm2_21", "dm2_21_err",
            "dm2_31", "dm2_31_err",
            "Nrea", "Nrea_err",
            "NU", "NU_err",
            "NTh", "NTh_err",
        ]

        # Create a pandas DataFrame
        df = pd.DataFrame(geo_data, columns=columns)
    #df = df[df["Nmantle"] !=0]
    # Calculate Ngeo relative error
    nU_relative_error = abs(df["NU_err"] / df["NU"])
    nU_relative_error = nU_relative_error[nU_relative_error <= 1]
    median_U = np.nanmedian(nU_relative_error)
    sigma_plus_U = np.nanpercentile(nU_relative_error, 84) - median_U
    sigma_minus_U = median_U - np.nanpercentile(nU_relative_error, 16)

    nTh_relative_error = abs(df["NTh_err"] / df["NTh"])
    median_Th = np.nanmedian(nTh_relative_error)
    sigma_plus_Th = np.nanpercentile(nTh_relative_error, 84) - median_Th
    sigma_minus_Th = median_Th - np.nanpercentile(nTh_relative_error, 16)

    filename = os.path.basename(h5_file)
    crust_rate_match = re.search(r"geo_UThfree_(\d+)TNU_", filename)
    crust_rate = crust_rate_match.group(2) if crust_rate_match else "Unknown"
    output_file = f"May22/UThfree/geo_free_toy_error_{stat}years.txt"
    header = "crust_tnu medianU sigma_pU sigma_mU medianTh sigma_pTh sigma_mTh\n"
    row = f"{crust_rate} {median_U:.3f} {sigma_plus_U:.3f} {sigma_minus_U:.3f} {median_Th:.3f} {sigma_plus_Th:.3f} {sigma_minus_Th:.3f}\n"
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)
    with open(output_file, "a") as f:
        f.write(row)

    print(f"NU Relative Error: Median = {median_U:.3f}, +34% Sigma = {sigma_plus_U:.3f}, -34% Sigma = {sigma_minus_U:.3f}")
    print(f"NTh Relative Error: Median = {median_Th:.3f}, +34% Sigma = {sigma_plus_Th:.3f}, -34% Sigma = {sigma_minus_Th:.3f}")
    # Plot Ngeo relative error
    plt.figure(figsize=(8, 6))
    plt.hist(nU_relative_error * 100., bins=30, color="skyblue", alpha=0.7, label="NU Relative Error")
    plt.axvline(median_U * 100., color="orange", linestyle="--", label=f"Median = {median_U * 100.:.1f}")
    plt.axvline((median_U + sigma_plus_U) * 100., color="green", linestyle="--", label=f"+34% = {sigma_plus_U * 100.:.1f}")
    plt.axvline((median_U - sigma_minus_U) * 100., color="red", linestyle="--", label=f"-34% = {sigma_minus_U * 100.:.1f}")
    plt.xlabel("NU_err / NU [%]")
    plt.ylabel("No. of toys")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(nTh_relative_error * 100., bins=30, color="skyblue", alpha=0.7, label="NTh Relative Error")
    plt.axvline(median_Th * 100., color="orange", linestyle="--", label=f"Median = {median_Th * 100.:.1f}")
    plt.axvline((median_Th + sigma_plus_Th) * 100., color="green", linestyle="--", label=f"+34% = {sigma_plus_Th * 100.:.1f}")
    plt.axvline((median_Th - sigma_minus_Th) * 100., color="red", linestyle="--", label=f"-34% = {sigma_minus_Th * 100.:.1f}")
    plt.xlabel("NTh_err / NTh [%]")
    plt.ylabel("No. of toys")
    plt.legend()
    plt.grid()
    plt.show()


def plot_dchi2(h5_fileCC, h5_fileGC, h5_fileGD, h5_file0):
    # Read only the second-to-last column (dchi2)
    with h5py.File(h5_fileCC, "r") as hdf:
        dchi2CC = np.array(hdf["geo"][:, -3], dtype=float)  # Extract second-to-last column

    with h5py.File(h5_fileGC, "r") as hdf:
        dchi2GC = np.array(hdf["geo"][:, -3], dtype=float)  # Extract second-to-last column

    with h5py.File(h5_fileGD, "r") as hdf:
        dchi2GD = np.array(hdf["geo"][330:10329][:, -3], dtype=float)  # Extract second-to-last column

    with h5py.File(h5_file0, "r") as hdf:
        dchi20 = np.array(hdf["geo"][:, -3], dtype=float)  # Extract second-to-last column

    # Calculate the median of dchi2
    medianCC = np.median(dchi2CC)
    medianGC = np.median(dchi2GC)
    medianGD = np.median(dchi2GD)
    # Calculate p-value: fraction of dchi20 >= median
    p_valueCC = np.sum(dchi20 >= medianCC) / len(dchi20)
    p_valueGC = np.sum(dchi20 >= medianGC) / len(dchi20)
    print(np.sum(dchi20 >= medianGD) )
    p_valueGD = np.sum(dchi20 >= medianGD) / len(dchi20)
    # Calculate significance (Z-score) from the p-value
    significanceCC = norm.isf(p_valueCC)  # Inverse survival function (1-CDF)
    significanceGC = norm.isf(p_valueGC)  # Inverse survival function (1-CDF)
    significanceGD = norm.isf(p_valueGD)  # Inverse survival function (1-CDF)

    # Print results
    print(f"Δχ²: Median CC = {medianCC:.3f}")
    print(f"p-value CC = {p_valueCC:.3e}")
    print(f"Significance CC = {significanceCC:.3f}σ")

    print(f"Δχ²: Median GC = {medianGC:.3f}")
    print(f"p-value GC = {p_valueGC:.3e}")
    print(f"Significance GC = {significanceGC:.3f}σ")

    print(f"Δχ²: Median GD = {medianGD:.3f}")
    print(f"p-value GD = {p_valueGD:.3e}")
    print(f"Significance GD = {significanceGD:.3f}σ")


    # Plot Δχ² distributions
    plt.figure(figsize=(20, 18))
    plt.hist(dchi2GD, bins=50, color="#0072B2",histtype='step', linewidth=4,label="GD Δχ² Distribution", log=True)
    plt.hist(dchi2GC, bins=50, color="#E69F00",histtype='step',linewidth=4, label="GC Δχ² Distribution", log=True)
    plt.hist(dchi2CC, bins=50, color="#D55E00", histtype='step',linewidth=4,label="CC Δχ² Distribution", log=True)
    plt.hist(dchi20, bins=50, color="gray",histtype='step',linewidth=4, label="No mantle Δχ² Distribution", log=True)
    plt.axvline(medianGD, color="#0072B2", linestyle="--", linewidth=2,label=f"GD median = {medianGD:.3f}, p-value = {p_valueGD:.3e}, $\sigma$ = {significanceGD:.3f}")
    plt.axvline(medianGC, color="#E69F00", linestyle="--", linewidth=2,label=f"GC median = {medianGC:.3f}, p-value = {p_valueGC:.3e}, $\sigma$ = {significanceGC:.3f}")
    plt.axvline(medianCC, color="#D55E00", linestyle="--",linewidth=2, label=f"CC median = {medianCC:.3f}, p-value = {p_valueCC:.3e}, $\sigma$ = {significanceCC:.3f}")

    plt.xlabel("Δχ²", fontsize=20)
    plt.ylabel("No. of psuedo experiments", fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.title("Mantle discovery potential (JULOC-I crust, 6 years)", fontsize=20)
    plt.show()

# Specify the HDF5 file names
h5_file = "Nov22_final_results/fit_results_geo_37.0_NorP_free_NO-True_100years_411bins_minuit.hdf5"
h5_file0 = "fit_results_geo_mantle0_NorP_free_NO-True_6years_411bins_minuit.hdf5"
#h5_file0 = "Nov22_final_results/fit_results_geomantle0_NorP_free_NO-True_6years_411bins_minuit.hdf5" #fit_results_geoCC_low0_NorP_free_NO-True_6years_411bins_minuit.hdf5"

h5_fileCC = "Mar15/fit_results_geo_mantleCC_NorP_free_NO-True_6years_411bins_minuit.hdf5"
h5_fileGC = "Mar15/fit_results_geo_mantleGC_NorP_free_NO-True_6years_411bins_minuit.hdf5"
h5_fileGD = "Mar15/fit_results_geo_mantleGD_NorP_free_NO-True_6years_411bins_minuit.hdf5"
#h5_fileGD = "Nov22_final_results/fit_results_geoGD_NorP_free_NO-True_6years_411bins_minuit.hdf5"
# Run the function
#plot_dchi2(h5_fileCC, h5_fileGC, h5_fileGD, h5_file0)
for i in range(10,80, 10):
# #      #create_pair_plot(f"Apr28/fit_results_geo_{i}_CNP_free_NO-True_3years_561bins_minuit.hdf5")
      plot_ngeo_relative_error(10, f"June11/fit_results_minos_geo_CNP_{i}_NO-True_10years_561bins_minuit.hdf5")
      #plot_ngeo_relative_error(3, f"June4/fit_results_geo_{i}TNU_CNP_free_NO-True_3years_561bins_minuit.hdf5")
      #plot_ngeo_relative_error(6, f"June4/fit_results_geo_{i}TNU_CNP_free_NO-True_6years_561bins_minuit.hdf5")
      #plot_ngeo_relative_error(10, f"June4/fit_results_geo_{i}TNU_CNP_free_NO-True_10years_561bins_minuit.hdf5")
#create_pair_plot_PMOP("May11/fit_results_NMO_CNP_pull_NO-True_2400days_561bins_minuit.hdf5")
# Generate the pair plot
#h5_file = "toy_test/June3/fit_results_geo_nocovmatrix_stat-only_10.0TNU_CNP_free_NO-True_3years_561bins_minuit.hdf5"
# # create_pair_plot(h5_file)
# years=[1]#,3,6,10]
# for y in years:
#     h5_file = f"June11/fit_results_geo_CNP_10_NO-True_3years_561bins_minuit.hdf5"
#     #create_pair_plot(h5_file)
#     plot_ngeo_relative_error(1, h5_file)
#     with h5py.File(h5_file, "r") as hdf:
#         nzbins = np.array(hdf["geo"][:, -1], dtype=float)  # -1 selects the last column
#     #     ngeo1 = np.array(hdf["geo"][0:1000, 11], dtype=float)
#     #     ngeo2 = np.array(hdf["geo"][1000:2000, 11], dtype=float)
#     #     ngeo3 = np.array(hdf["geo"][2000:3000, 11], dtype=float)
#     # print(nzbins)
#     # plt.hist(ngeo1, histtype='step')
#     # plt.hist(ngeo2, histtype='step')
#     # plt.hist(ngeo3, histtype='step')
#     # plt.yscale('log')
#     # plt.show()
#     #plt.xticks([0, 1, 2])
#     plt.hist(nzbins, bins=int(np.max(nzbins) - np.min(nzbins)), histtype='step', label=f'{y} years')
# plt.xlabel("# of bins with 0 bin content")
# plt.ylabel("#toys")
# plt.title("0 bins vs statistics")
# plt.legend()
# plt.show()
# # Plot the Ngeo relative error
# #create_pair_plot("Apr2/fit_results_geo_CNP_Poisson12MeV_wsyst_free_NO-True_6years_561bins_minuit.hdf5")
#plot_ngeo_relative_error(3, h5_file)
#plot_dchi2(h5_file)
