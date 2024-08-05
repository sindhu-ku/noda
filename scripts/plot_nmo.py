import re
import matplotlib.pyplot as plt
import numpy as np

def get_years_and_sigmas(filenames):
    years = []
    sigma = []
    dchi2 = []

    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('delta chi2 '):
                    # dchi2_value = float(line.split()[1])
                    # dchi2.append(dchi2_value)
                    sigma.append(float(line.split()[-1]))
                    # Extract the period from the filename
                    period = filename.split('_')[0]
                    if period.endswith('d'):
                        years.append(int(period[:-1]) / 365)
                    elif period.endswith('yr'):
                        years.append(int(period[:-2]))

    return years, sigma

def main():
    # Define the filenames
    years = [1, 2, 4, 6 ,8, 10, 15, 20, 25]
    filenames1 = []
    filenames2 = []
    filenames3 = []

    for y in years:
        filenames1.append(f'{y}yr_newabc.out')
        filenames2.append(f'{y}yr_nofact.out')

    # Get years and sigmas
    years1, sigma1 = get_years_and_sigmas(filenames1)
    years2, sigma2 = get_years_and_sigmas(filenames2)
    years3, sigma3 = get_years_and_sigmas(filenames3)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(years1, sigma1, marker='o', label='without 11/12 factor for livetime')
    plt.plot(years2, sigma2, marker='o', label='with 11/12 factor for livetime')
    plt.xlabel('Years')
    plt.ylabel('$\sigma$')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
