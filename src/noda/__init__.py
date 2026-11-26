"""NODA reactor-neutrino fitter.

Layout:
  core/      Spectrum, CovMatrix, response matrices, binning, likelihood, I/O
  physics/   oscillation (nuosc), Geant4 NL, constants, reactor flux, TAO fast-n
  model/     assemble (CreateSpectra), reactor, detection chain, backgrounds,
             covariance matrices, DYB model
  fit/ fc/ bayesian/   fitting, Feldman-Cousins, MCMC
  legacy/    superseded code, not used by the fitter
  noda.py    compatibility facade (`from noda.noda import *`)
  main.py    run_noda entry point

`noda.nuosc` and `noda.constants` are aliases of the SAME module objects in
noda.physics (nuosc keeps module-level state such as op_nom), so old imports
like `from noda import nuosc` keep working.
"""
import sys as _sys

from .physics import nuosc, constants

_sys.modules.setdefault(__name__ + ".nuosc", nuosc)
_sys.modules.setdefault(__name__ + ".constants", constants)
