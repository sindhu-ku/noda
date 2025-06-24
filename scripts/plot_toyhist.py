import uproot
import numpy as np
import matplotlib.pyplot as plt

# --- File paths
poisson2_path = "~/Downloads/toys_1year_10TNU_from1.root" #"June6/toy_spectra_1years_10TNU_stat-only.root" #"Apr2/toy_spectra_1year_10TNU_poisson3.root"
#poisson3_path = "Apr2/toy_spectra_1year_10TNU_cholesky.root" #"Apr2/toys_1year_10TNU.root"
asimov_path = "June6/Spectra_1year_10TNU_June13.root"#"Apr2/toy_spectra_1year_10TNU_poisson.root"
# --- Load toy histograms from both files
file2 = uproot.open(poisson2_path)
#file3 = uproot.open(poisson3_path)
file = uproot.open(asimov_path)

toy_hists2 = [obj for key, obj in file2.items() if key.startswith("toy")]
#toy_hists3 = [obj for key, obj in file3.items() if key.startswith("toy")]

if not toy_hists2:# or not toy_hists3:
    raise RuntimeError("No toy_* histograms found in one or both files.")

# --- Binning
bin_edges = toy_hists2[0].axes[0].edges()
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# --- Sum toy histograms for Poisson2 and Poisson3
toy_sum2 = np.sum([h.values(flow=False) for h in toy_hists2], axis=0)/len(toy_hists2)
#toy_sum3 = np.sum([h.values(flow=False) for h in toy_hists3], axis=0)/len(toy_hists3)

# --- Load Asimov histogram
asimov_hist = file["Total"]
asimov_vals = asimov_hist.values(flow=False)

# --- Relative difference for Poisson3 vs Asimov
asimov_vals_new = asimov_vals[1:]
with np.errstate(divide='ignore', invalid='ignore'):
    rel_diff2 = np.where(asimov_vals_new != 0, (toy_sum2 - asimov_vals_new)*100. / asimov_vals_new, 0)
    #rel_diff3 = np.where(asimov_vals_new != 0, (toy_sum3 - asimov_vals_new)*100. / asimov_vals_new, 0)
print(np.sum(asimov_vals_new), np.sum(toy_sum2))
# --- Plot: Asimov vs toy sums (Poisson2 and Poisson3)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
ax1.step(bin_centers, asimov_vals_new, where="mid", label="Asimov", color="black")
ax1.step(bin_centers, toy_sum2, where="mid", label="Sum of Toys (Poisson)", color="green", alpha=0.7)
#ax1.step(bin_centers, toy_sum3, where="mid", label="Sum of Toys (Poisson + Cholesky)", color="red", alpha=0.7)
ax1.legend()
ax1.grid(True)
ax1.set_ylabel("Counts")

# Relative difference (Poisson3 vs Asimov)
ax2.axhline(0, color="black", linestyle="--")
ax2.step(bin_centers, rel_diff2, where="mid", color="green")
#ax2.step(bin_centers, rel_diff3, where="mid", color="red")
ax2.set_ylabel("Rel. Diff [%]")
ax2.set_ylim(-10, 10)
ax2.set_xlabel("Reco energy [MeV]")
ax2.grid(True)

plt.tight_layout()
plt.show()

# # --- Total event count comparison (with vline at Asimov total)
# totals2 = [np.sum(h.values(flow=False)) for h in toy_hists2]
# totals3 = [np.sum(h.values(flow=False)) for h in toy_hists3]
#
# plt.figure(figsize=(10, 5))
# bins = np.linspace(min(min(totals2), min(totals3)), max(max(totals2), max(totals3)), 40)
# plt.hist(totals2, bins=bins, alpha=0.3, label="Poisson", color="green")
# plt.hist(totals3, bins=bins, alpha=0.3, label="Poisson + Cholesky", color="red")
# plt.axvline(16912, color="black", linestyle="--", label="Asimov (16912)")
# plt.xlabel("Total Events per Toy")
# plt.ylabel("Number of Toys")
# plt.title("Total Event Count Comparison")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
