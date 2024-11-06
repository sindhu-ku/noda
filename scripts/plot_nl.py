import ROOT
import numpy as np
import matplotlib.pyplot as plt
# Open the ROOT file
imgtype = 'all'
type = 'sp'
if type == 'NL': histtype = 'new_nonl'
else: histtype = 'NL'
if imgtype == 'all':
    fname = 'Oct22'
    num_curves  = 1
    num_exps = 5000
else:
    fname = 'Oct17_4x'
    num_curves =4
    num_exps=1000
file = ROOT.TFile.Open(f"NL_fluc_{fname}.root")
file1 = ROOT.TFile.Open("NL_fluc_Oct16.root")
# Retrieve the nominal curve (replace 'nominal' with actual name if needed)
nominal_curve = file1.Get(f"nominal_{type}")


# Convert ROOT histogram to numpy arrays for easier manipulation
def root_hist_to_numpy(hist):
    nbins = hist.GetNbinsX()
    bin_centers = np.array([hist.GetBinCenter(i) for i in range(1, nbins+1)])
    bin_contents = np.array([hist.GetBinContent(i) for i in range(1, nbins+1)])
    return bin_centers, bin_contents

# Get nominal histogram data
nominal_x, nominal_y = root_hist_to_numpy(nominal_curve)
# pull0_x,pull0_y = root_hist_to_numpy(file.Get(f"pull_{0}"))
# pull1_x,pull1_y = root_hist_to_numpy(file.Get(f"pull_{1}"))
# pull2_x,pull2_y = root_hist_to_numpy(file.Get(f"pull_{2}"))
# pull3_x,pull3_y = root_hist_to_numpy(file.Get(f"pull_{3}"))

# Prepare the figure for two subplots: one for spectra and one for differences
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
integral = np.zeros(410)
# Loop over the histograms NL_{i}_{j}, i from 0 to 3 and j from 0 to 24
add_diff = np.zeros(410)
one_hist_x = []
one_hist_y  = []
for i in range(num_curves):
    for j in range(num_exps):
        # Get the histogram name
        hist_name = f"{histtype}_{i}_{j}"
        hist = file.Get(hist_name)
        #print(nominal_curve.Integral() - hist.Integral())
        if hist:  # Ensure the histogram exists
            #hist.Scale(1./4.)
            #hist.Scale(nominal_curve.Integral()/hist.Integral())
            hist_x, hist_y = root_hist_to_numpy(hist)
            #integral.append((nominal_curve.Integral()-hist.Integral())*100./nominal_curve.Integral())
            #print(np.sum(hist_y) - np.sum(nominal_y))
            # Plot all histograms on the first subplot (spectra)
            axs[0].plot(hist_x, hist_y, label=f'NL_{i}_{j}', color='grey', alpha=0.5)

            # Calculate the difference between histogram and nominal curve
            diff_y = (nominal_y - hist_y)*100./nominal_y
            #integral += np.array(diff_y)
            #add_diff = add_diff + hist_y
            # Plot all differences on the second subplot
            axs[1].plot(hist_x, diff_y, label=f'Diff {i}_{j}', color='grey', alpha=0.2)


#print(integral/1000.)
# Plot the nominal curve in black for the first subplot (spectra)
axs[0].plot(nominal_x, nominal_y, label='Nominal (Rea+Geo)', color='red', linestyle='--', linewidth=2)
axs[0].set_title('Fluctuated NL curves', fontsize=18)
# axs[0].set_xlabel('Positron deposity energy (MeV)',fontsize=18 )
# #axs[0].set_xlabel('Nominal - Fluctuated', fontsize=18)

#axs[0].legend(ncol=5, fontsize='small', loc='upper right')

# Plot settings for differences subplot
axs[1].axhline(0, color='red', linestyle='--')  # Zero line
axs[1].set_xlabel('Positron deposited energy (MeV)', fontsize=18)
axs[1].set_ylabel('(nominal - fluctuated)/nominal [%]', fontsize=18)
#axs[1].set_ylim(-10,10)
# #axs[1].legend(ncol=5, fontsize='small', loc='upper right')
axs[0].tick_params(axis='both', which='major', labelsize=16)  # Set tick label size for top plot
axs[1].tick_params(axis='both', which='major', labelsize=16)  # Set tick label size for bottom plot
#
# # Adjust layout
plt.tight_layout()
#
# # Save the plot as a file if necessary
#plt.savefig(f"{imgtype}_{type}_ylim.png")
#
# # Show the plot
# #
plt.show()
# # plt.plot(nominal_x, add_diff)
# plt.show()
#
# plt.hist(integral, bins=100)
# plt.xlabel('Total nominal - total fluctuated/total nominal [%]', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
# plt.plot(nominal_x, nominal_y, label='nominal', linewidth=2)
# plt.plot(pull0_x, pull0_y, label='pull 0', linewidth=1)
# plt.plot(pull1_x, pull1_y, label='pull 1', linewidth=1)
# plt.plot(pull2_x, pull2_y, label='pull 2', linewidth=1)
# plt.plot(pull3_x, pull3_y, label='pull 3', linewidth=1)
# plt.xlabel('Positron deposited energy (MeV)', fontsize=18)
# plt.legend(fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
# plt.plot(nominal_x, (nominal_y-one_hist_y)*100/nominal_y, label='relative diff %')
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()

# Close the ROOT file
file.Close()
