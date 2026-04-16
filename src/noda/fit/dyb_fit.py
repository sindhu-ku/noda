#!/usr/bin/env python3
"""
dyb_fit.py

Encapsulates everything DYB-specific:
  - Spectrum building via DAGflow or direct PMOP path
  - DYB-specific nuisance parameter names, start values, limits
  - chi2 contribution

Shared quantities received from outside:
  - shape nuisances (nuisance) -- list/array  (same object as JUNO's)
"""

import numpy as np
from ..noda import *
from .. import nuosc as nuosc


class DYBDetector:
    """
    Handles the DYB near-detector chi2 contribution.

    Two operating modes (matching original logic):
      1. use_dagflow=True  : chi2 comes from dag_modelling make_fcn
      2. use_dagflow=False : chi2 computed directly from PMOP spectra
    """

    def __init__(self, ensp_nom_dyb, args_dyb, args_juno, cm_dyb,
                 rm_dyb,
                 dyb_model=None):
        self.ensp_nom   = ensp_nom_dyb
        self.args_dyb    = args_dyb
        self.args_juno  = args_juno
        self.cm_dyb      = cm_dyb
        self.rm_dyb      = rm_dyb
        self.dyb_model  = dyb_model

        # ── DAGflow setup (mirrors original run_minuit logic) ───────────────
        self._chi2_dyb_func  = None
        self._flux_params    = None
        self.n_dyb_bins      = 0

        if dyb_model is not None:
            from dag_modelling.tools.make_fcn import make_fcn
            flux_params = dyb_model.storage["parameters.free.neutrino_per_fission_factor"]
            dyb_model.storage["parameters.all.survival_probability.DeltaMSq32"].value   = 2.5413e-3
            dyb_model.storage["parameters.all.survival_probability.SinSq2Theta13"].value = 0.0856
            dyb_model.storage["parameters.all.survival_probability.DeltaMSq21"].value   = 7.53e-05
            dyb_model.storage["parameters.all.survival_probability.SinSq2Theta12"].value = 0.851
            self.n_dyb_bins      = dyb_model.storage[
                "outputs.eventscount.final.concatenated.detector"].data.shape[0]
            self._chi2_dyb_func  = make_fcn(
                dyb_model.storage["outputs.statistic.full.covmat.chi2cnp"],
                parameters=[par for par in flux_params.walkvalues()],
                safe=False)
            self._flux_params = flux_params
            print("DYB bins (dagflow):", self.n_dyb_bins)
        else:
            # PMOP direct path
            if args_juno.fit_type == "PMOP_combined":
                self.n_dyb_bins = len(ensp_nom_dyb['rdet'].bins)
                print("DYB bins (direct):", self.n_dyb_bins)

        # ── per-call cache ───────────────────────────────────────────────────
        self._chi2_cache  = {}
        self._dyb_spec_cache = {}

    # ════════════════════════════════════════════════════════════════════════
    #  Nuisance parameter interface
    # ════════════════════════════════════════════════════════════════════════

    def get_param_names(self):
        return ["snf_noneq_norm_dyb"]  

    def get_start_values(self):
        return [0.0]

    def apply_limits(self, m):
        m.limits["snf_noneq_norm_dyb"] = [-1, None]
        
    def get_first_fix(self):
        """Parameters to fix in the first Minuit stage."""
        return (
            ["snf_noneq_norm_dyb"]  
        )
    # ════════════════════════════════════════════════════════════════════════
    #  Internal cached helpers
    # ════════════════════════════════════════════════════════════════════════

    def _get_chi2_dyb_dagflow(self, nuisance_coarse):
        key = tuple(nuisance_coarse)
        if key in self._chi2_cache:
            return self._chi2_cache[key]
        chi2_dyb = self._chi2_dyb_func(*nuisance_coarse)[0]
        self._chi2_cache[key] = chi2_dyb
        return chi2_dyb

    def _get_dyb_spectrum(self, nuisance_coarse, E_segments):
        key = tuple(nuisance_coarse)
        if key in self._dyb_spec_cache:
            return self._dyb_spec_cache[key]
        rfis_param_dyb = self.ensp_nom['ribd']
        nuisance_fine = expand_nuisances_to_fine(E_segments, rfis_param_dyb.bins, nuisance_coarse)
        rfis_param_dyb = Spectrum(bins=rfis_param_dyb.bins,
                                 bin_cont=rfis_param_dyb.bin_cont * (1 + np.array(nuisance_fine)))
        s_dyb = rfis_param_dyb.GetWithPositronEnergy(inputfile=self.args_juno.input_data_file, tf2name=self.args_juno.pos_ene_TF2)
        s_dyb = s_dyb.ApplyDetResp(self.rm_dyb, pecrop=self.args_dyb.ene_crop) 
        self._dyb_spec_cache[key] = s_dyb
        return s_dyb

    # ════════════════════════════════════════════════════════════════════════
    #  chi2 contribution
    # ════════════════════════════════════════════════════════════════════════

    def chi2_contribution(self, snf_noneq_norm_dyb, nuisance_coarse, E_segments):
        """
        Returns scalar chi2_dyb.

        nuisance_coarse : shared shape nuisances (from combined_fit)
        snf_noneq_norm  : list; index [6] is the DYB noneq norm (PMOP direct path)
        E_segments      : from args_juno (shared binning)
        fine_bins       : fine energy bins
        """
        args_juno = self.args_juno
        args_dyb   = self.args_dyb

        if args_juno.use_dagflow:
            return self._get_chi2_dyb_dagflow(nuisance_coarse)

        # ── direct PMOP path ─────────────────────────────────────────────────
        s_dyb = self._get_dyb_spectrum(nuisance_coarse, E_segments)

        noneq = self.ensp_nom["noneq_final"].GetScaledFit(1 + snf_noneq_norm_dyb)
        s_tot_dyb = s_dyb + noneq
        
        if args_dyb.rebin_nonuniform:
            s_tot_dyb.Rebin_nonuniform(args_dyb.bins_nonuniform)

        sp_obs_dyb = self.ensp_nom["rtot"]

        cm_dyb_new = CovMatrix(
            data=self.cm_dyb['all'].data * np.outer(s_tot_dyb.bin_cont, s_tot_dyb.bin_cont),
            bins=self.cm_dyb['all'].bins)

        _, chi2_dyb = Chi2(cm_dyb_new, sp_obs_dyb, s_tot_dyb, sp_obs_dyb, s_tot_dyb,
                           'all', args_juno.stat_method_opt,
                           pulls=[snf_noneq_norm_dyb], pull_unc=[args_dyb.noneq_unc])
        return chi2_dyb, sp_obs_dyb, s_tot_dyb
