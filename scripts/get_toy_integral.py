import uproot
import numpy as np

def get_histogram_integrals_mean(root_file_path):
    with uproot.open(root_file_path) as file:
        integrals = []

        for key in file.keys():
            obj = file[key]

            # Check if it's a TH1 or TH2 histogram (modern uproot way)
            if obj.classname.startswith("TH1") or obj.classname.startswith("TH2"):
                values = obj.values(flow=False)
                integral = np.sum(values)
                integrals.append(integral)

        if integrals:
            mean_integral = np.mean(integrals)
            print(f"Found {len(integrals)} histograms.")
            print(f"Mean of integrals: {mean_integral}")
            return mean_integral
        else:
            print("No histograms found in the file.")
            return None

# Example usage
root_file = "~/Downloads/toys_1year_10TNU_from1.root"  # Replace with your filename
get_histogram_integrals_mean(root_file)
