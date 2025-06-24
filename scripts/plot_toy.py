import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the data from a text file into a pandas DataFrame
data = pd.read_csv('Geo_toy/fit_results_geoNorP_free_NO-True_6years_411bins_minuit.txt', delim_whitespace=True)

# Ensure the 'Ngeo' column exists and convert it to numeric, forcing errors to NaN
if 'Ngeo' not in data.columns:
    raise ValueError("The file does not contain an 'Ngeo' column.")

data['Ngeo'] = pd.to_numeric(data['Ngeo'], errors='coerce')
ngeo_data = data['Ngeo'].dropna()

# Define the Gaussian function for fitting
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Plot the histogram directly from the 'Ngeo' column
counts, bins, _ = plt.hist(ngeo_data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# Midpoints of bins for fitting
bin_centers = (bins[:-1] + bins[1:]) / 2

# Initial guesses for the fit parameters: amplitude, mean (mu), and standard deviation (sigma)
initial_guess = [1, ngeo_data.mean(), ngeo_data.std()]

# Fit the Gaussian to the histogram data
popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
amp, mu, sigma = popt

# Overlay the fitted Gaussian curve
x = np.linspace(bins[0], bins[-1], 100)
plt.plot(x, gaussian(x, *popt), 'r--', linewidth=2, label=f'Gaussian fit\nMean = {mu:.2f}\nSigma = {sigma:.2f}')

# Customize the plot
plt.xlabel('Ngeo')
plt.ylabel('Density')
plt.title('Histogram of Ngeo with Gaussian Fit using curve_fit')
plt.legend()

# Show the plot
plt.show()
