import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Asimov values
asimov_values = {
    "sin2_12": 0.307,
    "sin2_13": 0.022,
    "dm2_21": 7.53e-5,
    "dm2_31":  0.0025283000,
    "Nrea": 1.0,
    "Ngeo": 1.0
}

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
        kind="hist",
        diag_kind="hist",
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

def plot_ngeo_relative_error(h5_file):
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

    # Calculate Ngeo relative error
    ngeo_relative_error = df["Ngeo_err"] / df["Ngeo"]
    median = np.median(ngeo_relative_error)
    sigma_plus = np.percentile(ngeo_relative_error, 84) - median
    sigma_minus = median - np.percentile(ngeo_relative_error, 16)

    print(f"Ngeo Relative Error: Median = {median:.3f}, +34% Sigma = {sigma_plus:.3f}, -34% Sigma = {sigma_minus:.3f}")

    # Plot Ngeo relative error
    plt.figure(figsize=(8, 6))
    plt.hist(ngeo_relative_error * 100., bins=30, color="skyblue", alpha=0.7, label="Ngeo Relative Error")
    plt.axvline(median * 100., color="orange", linestyle="--", label=f"Median = {median * 100.:.1f}")
    plt.axvline((median + sigma_plus) * 100., color="green", linestyle="--", label=f"+34% = {sigma_plus * 100.:.1f}")
    plt.axvline((median - sigma_minus) * 100., color="red", linestyle="--", label=f"-34% = {sigma_minus * 100.:.1f}")
    plt.xlabel("Ngeo_err / Ngeo [%]")
    plt.ylabel("No. of toys")
    plt.legend()
    plt.grid()
    plt.show()

# Specify the HDF5 file name
h5_file = "Nov22_final_results/fit_results_geo_NorP_free_NO-True_6years_411bins_minuit.hdf5"

# Generate the pair plot
create_pair_plot(h5_file)

# Plot the Ngeo relative error
plot_ngeo_relative_error(h5_file)
