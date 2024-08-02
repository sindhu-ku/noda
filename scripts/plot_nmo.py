#import re
#import matplotlib.pyplot as plt
#import numpy as np
#
## Dictionary to store the data
#years =[]
#sigma =[]
#dchi2 =[]
#
## Define the filenames
#filenames = [ "1yr_abc.out", "2yr_abc.out",  "4yr_abc.out", "6yr_abc.out", "8yr_abc.out",  "10yr_abc.out", "15yr_abc.out", "20yr_abc.out", "25yr_abc.out", "30yr_abc.out"]
#
## Read and process each file
#for filename in filenames:
#    with open(filename, 'r') as file:
#        for line in file:
#            if line.startswith('chi2 '):
#                dchi2.append(line.split()[1])
#                sigma.append(np.sqrt(float(line.split()[1])))
#                # Extract the period from the filename
#                period = filename.split('_')[0]#.split('.')[0]
#                print(period)
#                if period.endswith('d'):
#                    years.append(int(period[:-1]) / 365)
#                elif period.endswith('yr'):
#                    years.append(int(period[:-2]))
#
#print(years)
#print(dchi2)
#print(sigma)
## Plotting
#plt.figure(figsize=(10, 6))
#plt.plot(years, sigma, marker='o')
#plt.xlabel('Years')
#plt.ylabel('$\sigma$')
#plt.grid(True)
#plt.show()

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
                if line.startswith('chi2 '):
                    dchi2_value = float(line.split()[1])
                    dchi2.append(dchi2_value)
                    sigma.append(np.sqrt(dchi2_value))
                    # Extract the period from the filename
                    period = filename.split('_')[0]
                    if period.endswith('d'):
                        years.append(int(period[:-1]) / 365)
                    elif period.endswith('yr'):
                        years.append(int(period[:-2]))

    return years, sigma

def main():
    # Define the filenames
    years = [1, 2, 4, 6 ,8, 10, 15, 20, 25, 30]
    filenames1 = []
    filenames2 = []
    filenames3 = []

    for y in years:
        filenames1.append(f'{y}yr_abc.out')
        filenames2.append(f'{y}yr_CNP.out')
        filenames3.append(f'{y}yr_NorP.out')

    # Get years and sigmas
    years1, sigma1 = get_years_and_sigmas(filenames1)
    years2, sigma2 = get_years_and_sigmas(filenames2)
    years3, sigma3 = get_years_and_sigmas(filenames3)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(years2, sigma2, marker='o', label='old res CNP')
    plt.plot(years3, sigma3, marker='o', label='old res NorP')
    plt.plot(years1, sigma1, marker='o', label='new res NorP')
    plt.xlabel('Years')
    plt.ylabel('$\sigma$')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
