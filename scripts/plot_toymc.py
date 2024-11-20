import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def create_pair_plot(h5_file):
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
            "Ngeo", "Ngeo_err"
        ]

        # Create a pandas DataFrame
        df = pd.DataFrame(geo_data, columns=columns)

    # Select variables of interest for the plot
    variables = ["sin2_12", "sin2_13", "dm2_21", "dm2_31", "Ngeo", "Nrea"]

    # Filter the DataFrame to include only the selected variables
    plot_data = df[variables]

    # Create pairplot
    pair_plot = sns.pairplot(
        plot_data,
        kind="hist",  # Use KDE for the histograms
        diag_kind="kde",
        #plot_kws={"s": 5, "alpha": 0.6},  # Increase scatter size and transparency
        corner=True,  # Only show lower triangle of scatter plots
        height=3  # Increase figure size for better visibility
    )
    for var in variables:
        # Fit the data to a normal distribution
        mu, std = norm.fit(df[var])
        print(var, mu, std)
        #ax = pair_plot.axes[0, 0]  # Get the diagonal axis (histogram for the first variable)
        #ax.annotate(f'Mean: {mu:.3f}\nSigma: {std:.3f}',
        #            xy=(0.75, 0.75), xycoords="axes fraction", fontsize=12,
        #            ha="center", va="center", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1.0'))

    pair_plot.fig.suptitle("Pair Plot with Histograms and Scatter Plots", y=1.02)

    # Save the figure
    #plt.savefig("pairplot_with_Ngeo_Nrea.png", dpi=300, bbox_inches="tight")
    plt.show()

# Specify the HDF5 file name
h5_file = "Geo_toy/fit_results_geo_NorP_free_NO-True_6years_411bins_minuit.hdf5"  # Replace with your file's path
create_pair_plot(h5_file)
