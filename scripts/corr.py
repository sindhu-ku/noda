import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import sys
# Load the data
days = sys.argv[1]
file = f"toy_results_{days}_Feb21.npz"
loaded_data = np.load(file, allow_pickle=True)

# Extract the arrays
sin2_12_arr = loaded_data["sin2_12_arr"]
sin2_13_arr = loaded_data["sin2_13_arr"]
dm2_21_arr = loaded_data["dm2_21_arr"]
dm2_31_arr = loaded_data["dm2_31_arr"]
delta_chi2_arr = loaded_data["delta_chi2_arr"]

# Create a pandas DataFrame
df = pd.DataFrame({
    "sin2_12": sin2_12_arr,
    "sin2_13": sin2_13_arr,
    "dm2_21": dm2_21_arr,
    "dm2_31": dm2_31_arr
#    "delta_chi2": delta_chi2_arr
})
#df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
#df = df[df['dm2_31'] > 0.000]
#df = df[df['dm2_31'] < 0.004]

# Create a pair plot with individual distribution plots
g = sns.pairplot(df, kind = "hist", diag_kind = "hist", 
             diag_kws = {'alpha':0.55, 'bins':50}, corner=True)

# Access the diagonal axes
axes = g.diag_axes

# Iterate through the diagonal subplots and add annotations
for i, ax in enumerate(axes):
    # Extract the Gaussian fit parameters
    mean, cov = stats.norm.fit(df.iloc[:, i])
    cov = np.atleast_2d(cov)
    # Calculate standard deviation
    std_dev = np.sqrt(np.diag(cov))
    
    # Annotate the plot with mean and standard deviation
    ax.annotate(f"{mean:.9f} +/- {std_dev[0]:.9f}", xy=(0.5, 0.75), xycoords="axes fraction",
            ha="center", va="center", bbox=dict(boxstyle="round", alpha=0.1), fontsize=10)


plt.show()
#lt.savefig(f'/home/sindhu/matrix_{days}.png')
