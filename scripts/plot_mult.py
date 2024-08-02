import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mean, sigma):
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma)**2)

def plot(files=[]):
    all_results = {param_name: [] for param_name in ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]}
    max_bin_diff = 0

    for file in files:
        loaded_data = np.load(file, allow_pickle=True)

        sin2_12_arr = loaded_data["sin2_12_arr"]
        sin2_13_arr = loaded_data["sin2_13_arr"]
        dm2_21_arr = loaded_data["dm2_21_arr"]
        dm2_31_arr = loaded_data["dm2_31_arr"]

        parameter_arrays = [sin2_12_arr, sin2_13_arr, dm2_21_arr, dm2_31_arr]
        parameter_names = ["sin2_12", "sin2_13", "dm2_21", "dm2_31"]

        for param_arr, param_name in zip(parameter_arrays, parameter_names):
            param_arr = np.array(param_arr)

            hist, bin_edges = np.histogram(param_arr, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[param_arr.mean(), param_arr.std()])

            all_results[param_name].append((bin_centers, hist, popt))

            # Calculate the maximum difference in bin edges
            bin_diff = bin_edges[-1] - bin_edges[0]
            max_bin_diff = max(max_bin_diff, bin_diff)

    # Use the bin_centers from the file with the largest bin difference
    common_bin_centers = np.linspace(bin_edges[0], bin_edges[-1], num=50)

    return all_results, common_bin_centers

def main():
    files = ["toy_results_100days.npz"]#, "toy_results_200days.npz", "toy_results_2000days.npz"]
    all_results, common_bin_centers = plot(files=files)

    for param_name, results_per_file in all_results.items():
        plt.figure()
        for i, (bin_centers, hist, popt_values) in enumerate(results_per_file):
            label = f'File {i+1}'
            plt.hist(common_bin_centers, bins=100, density=False, alpha=0.5, label=label, color=f'C{i}')

        plt.xlabel(param_name)
        plt.ylabel('Frequency')

        plt.legend(loc='best')
#        plt.savefig(f"{param_name}_histogram_combined.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()

