import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Parameter names and label-friendly versions
params = ['sin2_12', 'dm2_21', 'dm2_31', 'sin2_13', 'a1', 'a2', 'a3', 'a4', 'Nrea', 'me_rho']
labels = [r'$\sin^2\theta_{12}$',
          r'$\Delta m^2_{21}$', r'$\Delta m^2_{31}$',  r'$\sin^2\theta_{13}$',
          'NL_a1', 'NL_a2', 'NL_a3', 'NL_a4', 'Nrea', r'$\rho_{ME}$']

# Reference (nominal) values
ref_vals = {
    'sin2_12': 0.307, 'dm2_21': 7.53e-5, 'dm2_31': 2.5303e-3, 'sin2_13': 0.0219,
    'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'a4': 0.0, 'Nrea': 1.0, 'me_rho': 2.45
}

# w NL (cov)
cov_vals = {
    'sin2_12': (0.303, 0.0016),
    'dm2_21': (7.48e-5, 2.4e-7),
    'dm2_31': (2.5223e-3, 0.0043e-3),
    'sin2_13': (0.0219, 0.0007)
}

# w NL (pull) + Nrea free — UPDATED values
pull_vals = {
    'sin2_12': (0.30435997632082645, 0.001614177451472531),
    'dm2_21': (7.49075900463439e-05, 2.2720134248477693e-07),
    'dm2_31': (0.002523076367237844, 3.8241491639155165e-06),
    'sin2_13': (0.022014560049634747, 0.0006712435118733423),
    'a1': (0.09532993820224062, 0.9197431933524876),
    'a2': (0.9527196835111867, 0.8095433381232073),
    'a3': (-0.04780992416799111, 0.8021328950060946),
    'a4': (-0.9094624814476301, 0.7565569684454537),
    'Nrea': (1.0623581282836545, 0.02322932711823502),
    'me_rho': (2.435078244535548, 0.3673408382088078)
}

# Create plot
fig, axes = plt.subplots(5, 2, figsize=(10, 10))
axes = axes.flatten()

for i, param in enumerate(params):
    ax = axes[i]
    y_ticks = []
    y_labels = []

    # Dashed reference line
    ref_val = ref_vals[param]
    ax.axvline(ref_val, color='green', linestyle='--', label='Nominal')

    # Covariance matrix values
    if param in cov_vals:
        val, err = cov_vals[param]
        ax.errorbar(val, 1, xerr=err, fmt='o', color='steelblue')
        y_ticks.append(1)
        y_labels.append('Cov. matrix approach')

    # Pull terms values
    if param in pull_vals:
        val, err = pull_vals[param]
        ax.errorbar(val, 0, xerr=err, fmt='o', color='brown')
        y_ticks.append(0)
        y_labels.append(r'NL & $\rho_{ME}$ pull terms+' + '\nNrea free')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title(labels[i])
    ax.grid(True)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
