#import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fontsize=18
labelsize=16

#matrix_rea = np.loadtxt('cov_mat_nl_rea.csv', delimiter=' ')
#matrix_geo = np.loadtxt('cov_mat_nl_reageo.csv', delimiter=' ')

matrix = np.loadtxt('cov_mat_juno_nl.csv', delimiter=' ')
matrix1 = np.loadtxt('cov_mat_juno_nl1.csv', delimiter=' ')

fig = plt.figure(figsize=(10, 10))

im = plt.imshow(matrix-matrix1, cmap='viridis', interpolation='nearest')#,vmin=0, vmax=10 )
plt.ylim(0,314)
plt.xlabel('Reco energy bins', fontsize=fontsize)
plt.ylabel('Reco energy bins', fontsize=fontsize)
#plt.ylim(0, 410)
x_ticks = plt.gca().get_xticks()
y_ticks = plt.gca().get_yticks()

# Apply the formula: 0.8 + (i - 1) * 0.02 to each tick value
#new_x_labels = [0.8 + (i * 0.02) for i in x_ticks]
#new_y_labels = [0.8 + (i * 0.02) for i in y_ticks]

# Set the new x and y tick labels
#plt.gca().set_xticklabels([f'{label:.2f}' for label in new_x_labels])
#plt.gca().set_yticklabels([f'{label:.2f}' for label in new_y_labels])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
cbar = plt.colorbar()
cbar.set_label('# events', fontsize=fontsize)
cbar.ax.tick_params(labelsize=labelsize)
#im.set_clim(-200, 450)
#plt.title("Reactor+Geo - Reactor", fontsize=20)
plt.show()
