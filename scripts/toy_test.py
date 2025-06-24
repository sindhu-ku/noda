import numpy as np
import matplotlib.pyplot as plt

# Define the function to generate smooth Gaussian spectrum
def generate_gaussian_spectrum(mean, std, num_bins, num_events):
    # Create a range of bin centers
    bin_centers = np.linspace(mean - 5*std, mean + 5*std, num_bins)

    # Generate a Gaussian distribution
    spectrum = np.exp(-(bin_centers - mean)**2 / (2 * std**2))

    # Normalize the spectrum to the total number of events
    spectrum = spectrum / np.sum(spectrum) * num_events

    return bin_centers, spectrum

# Parameters
num_bins = 100
num_toys = 10000

# Create two smooth Gaussian spectra
mean1, std1, num_events1 = 0, 1, 1000  # First Gaussian with 1000 events
mean2, std2, num_events2 = 0, 1, 100  # Second Gaussian with 100 events

bin_centers1, spectrum1 = generate_gaussian_spectrum(mean1, std1, num_bins, num_events1)
bin_centers2, spectrum2 = generate_gaussian_spectrum(mean2, std2, num_bins, num_events2)

plt.hist(spectrum1)
plt.hist(spectrum2)
plt.show()
# Sum of the two spectra
summed_spectrum = spectrum1 + spectrum2

# Generate 10k Poisson fluctuated toys for the summed spectrum
summed_spectrum_toys = np.random.poisson(summed_spectrum, size=(num_toys, num_bins))

# Sum the toys
summed_spectrum_sum = np.sum(summed_spectrum_toys, axis=0)

# Generate 10k Poisson fluctuated toys for each individual Gaussian
spectrum1_toys = np.random.poisson(spectrum1, size=(num_toys, num_bins))
spectrum2_toys = np.random.poisson(spectrum2, size=(num_toys, num_bins))

# Sum the toys for each Gaussian
spectrum1_sum = np.sum(spectrum1_toys, axis=0)
spectrum2_sum = np.sum(spectrum2_toys, axis=0)

# Compare the sums
comparison_sum = (summed_spectrum_sum - (spectrum1_sum + spectrum2_sum))*1000./summed_spectrum_sum

# Chi-squared calculation
def chi_squared(nominal, observed):
    return np.sum(((observed - nominal) ** 2) / nominal)

# Chi-squared for summed spectrum
chi_squared_sum = np.array([chi_squared(summed_spectrum, summed_spectrum_toys[i]) for i in range(num_toys)])

# Chi-squared for individual spectra sum
chi_squared_individual = np.array([chi_squared(summed_spectrum, spectrum1_toys[i] + spectrum2_toys[i]) for i in range(num_toys)])

# Plotting
plt.figure(figsize=(12, 6))

# Plot chi-squared distribution for summed spectrum
plt.subplot(1, 2, 1)
plt.hist(chi_squared_sum, bins=50, alpha=0.6, label="Summed spectrum")
plt.hist(chi_squared_individual, bins=50, alpha=0.6, label="Individual spectra sum")
plt.legend()
plt.title("Chi-squared Distribution Comparison")
plt.xlabel("Chi-squared")
plt.ylabel("Frequency")

# Plot comparison of summed spectra
plt.subplot(1, 2, 2)
plt.plot(bin_centers1, comparison_sum, label="Difference in sums", color="red")
plt.xlabel("Bin Center")
plt.ylabel("Difference")
plt.title("Comparison of Summed Spectra")

plt.tight_layout()
plt.show()
