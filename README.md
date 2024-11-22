NODA package (Neutrino Oscillation Data Analysis), a fitting framework for JUNO

Usage:
- `run_noda`
- You can also add options like these (see `config/fit_configuration_inputs.yaml`): `run_noda --stat_opt=1year`.
- Make sure to update the relevant data inputs path in the config.

Capabilities:

- Precision measurement of oscillation parameters (PMOP, Asimov)
- Neutrino Mass Ordering with and without TAI (NMO, Asimov)
- Geoneutrino measurement (Asimov)
- Toy MC for above under development (works for geo, albeit slow)
- Older version of Bayesian MCMC for PMOP (pending further development)
