import re
import matplotlib.pyplot as plt
import numpy as np

def get_years_and_sigmas(filenames, prompt):
    years = []
    sigma = []
    dchi2 = []

    for filename in filenames:
        with open(f'out/{filename}', 'r') as file:
            for line in file:
                if line.startswith(prompt):
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
    years1= [1, 2, 4, 6 ,8, 10, 15, 20, 25]
    years2 = [4, 6, 20]
    filenames1 = []
    filenames2 = []
    filenames3 = []
    steven_years = [1, 6]
    steven_sigma = [np.sqrt(1.470), np.sqrt(8.795)]

    # for y in years1:
    #     filenames1.append(f'{y}yr_newabc.out')
    # for y in years2:
    #     filenames2.append(f'{y}yr_nmo_syst.out')
    for y in years1:
        filenames1.append(f'{y}yr_JUNO-TAO.out')
    # Get years and sigmas
    years1, sigma1 = get_years_and_sigmas(filenames1, prompt='JUNO: delta chi2 ')
    years2, sigma2 = get_years_and_sigmas(filenames1, prompt='TAO: delta chi2 ')
    years3, sigma3 = get_years_and_sigmas(filenames1, prompt='JUNO+TAO: delta chi2 ')

    #print(sigma1)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(years1, sigma1, marker='o', label='JUNO stat. only')
    #plt.plot(steven_years, steven_sigma, marker='o', label='Steven')
    plt.plot(years2, sigma2, marker='o', label='TAO stat. only')
    plt.plot(years3, sigma3, marker='o', label='JUNO+TAO stat. only')
    plt.xlabel('Years')
    plt.ylabel('$\sigma$')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
