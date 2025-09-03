import uproot
import numpy as np
import matplotlib.pyplot as plt

def poisson_zero_prob(lmbda):
    """Returns the probability of observing 0 counts in a Poisson distribution with mean lmbda."""
    return np.exp(-lmbda)

def main(root_file_path, hist_name):
    # Load histogram from ROOT file
    with uproot.open(root_file_path) as file:
        hist = file[hist_name]

        # Get bin centers and bin contents
        bin_edges = hist.axes[0].edges()
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_contents = hist.values()

    # Compute probability of 0 observed events (Poisson zero probability)
    zero_probs = poisson_zero_prob(bin_contents)

    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(bin_centers, zero_probs, marker='o', linestyle='-')
    plt.xlabel("Reco energy (MeV)")
    plt.ylabel("P(0 observed | μ)")
    plt.title("Probability of Observing 0 Events per Bin")
    plt.grid(True)
    #plt.ylim(0, 0.0003)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: update with your ROOT file and histogram name
    root_file = "June6/Spectra_1year_10TNU_June6.root"
    histogram_name = "Total"
    main(root_file, histogram_name)
