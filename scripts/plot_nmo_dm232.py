import matplotlib.pyplot as plt

# Data
dm2_32_IO = [2.527, 2.529, 2.531, 2.533, 2.535, 2.537]  # in units of x10^-3
sigma = [9.3, 8.2, 8.2, 2.9, 2.9, 2.9]

# Plot
plt.figure(figsize=(5, 3))
plt.plot(dm2_32_IO, sigma, marker='o', linestyle='-', color='b', label='σ vs Δm²₃₂')
plt.xlabel(r'$\Delta m^2_{32}$ [$\times 10^{-3}$ eV$^2$, IO]')
plt.ylabel(r'NMO $\sigma$')
#plt.title(r'$\sigma$ vs $\Delta m^2_{32}$ (IO)')
plt.grid(True)
plt.tight_layout()
plt.show()
