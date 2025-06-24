import uproot
import numpy as np
import matplotlib.pyplot as plt

# Paths to ROOT files
poisson2_path = "Apr2/toy_spectra_1year_10TNU_poisson_12MeV_wsyst.root"
poisson3_path = "Apr2/toys_1year_10TNU.root"

# Load ROOT files
file2 = uproot.open(poisson2_path)
file3 = uproot.open(poisson3_path)

# Get toy histograms
toy_hists2 = [obj for key, obj in file2.items() if key.startswith("toy_")]
toy_hists3 = [obj for key, obj in file3.items() if key.startswith("toy")]

if not toy_hists2 or not toy_hists3:
    raise RuntimeError("No toy_* histograms found in one or both files.")

# Extract bin edges and centers
bin_edges2 = toy_hists2[0].axes[0].edges()
bin_centers2 = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])
bin_edges3 = toy_hists3[0].axes[0].edges()
bin_centers3 = 0.5 * (bin_edges3[:-1] + bin_edges3[1:])
# --- Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Overlay Poisson2 toys
for hist in toy_hists2:
    values = hist.values(flow=False)
    ax1.step(bin_centers2, values, where="mid", alpha=0.2)
ax1.set_title("Overlay of All Toys - Sindhu")
ax1.set_ylabel("Counts")
ax1.grid(True)

# Overlay Poisson3 toys
for hist in toy_hists3:
    values = hist.values(flow=False)
    ax2.step(bin_centers3, values, where="mid", alpha=0.2)
ax2.set_title("Overlay of All Toys - Cristobal")
ax2.set_ylabel("Counts")
ax2.set_xlabel("Reco energy [MeV]")
ax2.grid(True)

plt.tight_layout()
plt.show()
