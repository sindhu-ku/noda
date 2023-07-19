Early version of cleaned-up and improved NODA code, used for the precision measurement of oscillation parameters with the JUNO experiment

NODA- Neutrino Oscillation Data Analysis
Python version = 3.10.6

usage: python main.py --config=<config_yaml_file> --input_data_file=<data_file_name>

There is much more optimization to-do. Next in list is to make directories for the scripts based on function
spectra.py: produces all the spectra necessary for the fit and the energy response matrix
matrices.py: produces covariance matrices for all the uncertainties
scan.py: runs the grid scan and stores the data in a npy file
bayesian.py: bayesian fitter code, data saved in npz file
frequentist_results.py: plots final chi2 profiles and print final results for the frequentist fit
bayesian_results.py: plots final corner plots and prints final results for the Bayesian fit
noda.py: Contains all classes and helper functions
nuosc.py: Everything related to oscillation parameters and formulae
chi2.py: contains chi2 polynomial functions needed for scan.py