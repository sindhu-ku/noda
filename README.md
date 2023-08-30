Early version of cleaned-up and improved NODA code, used for the precision measurement of oscillation parameters with the JUNO experiment (cross-checked, v2.0.0 should give you results as in docdb #10250)

NODA- Neutrino Oscillation Data Analysis

Python version currently used: 3.10.6

usage: python main.py --config=<config_yaml_file> 

Change the path of your input file in the yaml file or do --input_data_file=<data_file_name>

(add more parsers as you want, look at yaml file for the required parameter)

spectra.py: produces all the spectra necessary for the fit and the energy response matrix

matrices.py: produces covariance matrices for all the uncertainties

scan.py: runs the grid scan and stores the data in a npy file

bayesian.py: bayesian fitter code, data saved in npz file

frequentist_results.py: plots final chi2 profiles and print final results for the frequentist fit

bayesian_results.py: plots final corner plots and prints final results for the Bayesian fit

noda.py: Contains all classes and helper functions

nuosc.py: Everything related to oscillation parameters and formulae

chi2.py: contains chi2 polynomial functions needed for scan.py
