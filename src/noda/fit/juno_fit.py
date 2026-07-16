#!/usr/bin/env python3
"""
juno_fit.py

Encapsulates everything JUNO-specific:
  - Spectrum building (oscillation, NL, detector response, backgrounds, SNF/non-eq)
  - JUNO-specific nuisance parameter names, start values, limits
  - chi2 contribution (calls shared Chi2 utility)

The only shared quantities this module *receives* from outside are:
  - osc params  (sin2_12, sin2_13, dm2_21, dm2_31)  -- floats
  - shape nuisances (nuisance)                        -- list/array
"""

import numpy as np
import scipy as sp
from ..noda import *
from .. import nuosc as nuosc


def _aux_dm2_31(toy, which):
    """Per-pull toy dm²₃₁ central. `toy` may be None, a float (legacy: the same
    value for whichever pull asks), or a dict {'dyb':…, 'lbl':…} so DYB and LBL
    constraints can each carry their own sampled auxiliary in combined toys."""
    if toy is None:
        return None
    if isinstance(toy, dict):
        return toy.get(which)
    return toy


class JUNODetector:
    """
    Builds the JUNO predicted spectrum and returns a chi2 contribution.

    Parameters passed at construction time are things that never change
    between Minuit calls (spectra templates, response matrices, config).
    Parameters passed at call time are the floating nuisances.
    """

    def __init__(self, ndays, ensp_nom, rm, cm, args, backgrounds,
                 core_baselines, core_powers_std,
                 E_segments, s_ibd, s_bump_lin, core_fractions=None,
                 core_names=None, covariance_er=None):
        self.ndays          = ndays
        self.ensp_nom       = ensp_nom
        self.rm             = rm
        self.cm             = cm
        self.args           = args
        self.backgrounds    = backgrounds
        self.core_baselines = core_baselines
        self.core_powers_std = core_powers_std
        self.core_fractions = core_fractions
        self.core_names     = core_names
        self.fine_bins      = ensp_nom['ribd'].bins
        self.E_segments     = E_segments
        self.s_ibd          = s_ibd
        self.s_bump_lin     = s_bump_lin
        self._linearize_resp = args.lin_resp
        self._osc_per_core   = getattr(args, "osc_per_core", False)
        self._external_cov = None
        # ── pre-build fission arrays (expensive, do once) ──────────────────
        args_juno = args
        n_cores = len(core_baselines)
        if args_juno.CI_powers:
            dummy_sp = ensp_nom["rfis0"]
        else:
            dummy_sp = ensp_nom["rfis0_core0"]
        self.dummy_sp = dummy_sp

        fission_fracs = []
        if not args.CI_powers:
            for frac in core_fractions:
                fission_fracs.append([frac['U235'], frac['U238'],
                                       frac['Pu239'], frac['Pu241']])

        else:
            for power in core_powers_std:
                fission_fracs.append([self.args.U235_scale, self.args.U238_scale,
                                   self.args.Pu239_scale, self.args.Pu241_scale])


        self.fission_fracs = fission_fracs
        self.fission_fracs = np.asarray(fission_fracs, dtype=float)

        fission_arrays = np.zeros((n_cores, len(dummy_sp.bin_cont)))
        for i in range(n_cores):
            if args_juno.CI_powers:
                dummy_rfis = Spectrum(bins=ensp_nom["rfis0"].bins,
                                      bin_cont=ensp_nom["rfis0"].bin_cont)
            else:
                dummy_rfis = Spectrum(bins=ensp_nom[f"rfis0_core{i}"].bins,
                                      bin_cont=ensp_nom[f"rfis0_core{i}"].bin_cont)
            dummy_rfis.WeightWithFunction(s_ibd)
            dummy_rfis.ScaleByBinWidth()
            if args_juno.apply_bump:
                dummy_rfis.WeightWithFunction(s_bump_lin)
            dummy_rfis.GetScaled(ndays)
            fission_arrays[i] = dummy_rfis.bin_cont
        self.fission_arrays = fission_arrays

        self.n_cores = n_cores

        # ── per-call caches ─────────────────────────────────────────────────
        self._resp_matrix_cache    = {}
        self._osc_cache            = {}   # post-positron flux, keyed by osc_key
        self._nl_cache             = {}   # post-NL flux, keyed by (osc_key, NL)
        self._nuisance_fine_cache  = {}
        self._chi2_penalty_er_cache = {}
        self._geo_spectrum_cache   = {}
        self._wr_spectrum_cache    = {}
        self._snf_noneq_cache      = {}

        # ── energy response covariance reference point ──────────────────────

        self.theta0_er = np.array([
            args.a, args.b, args.c,
            args.kB, args.fC, args.kI, args.LY_shared
        ])

        # Per-instance energy-response covariance. Falls back to the module
        # global when not supplied, so single-phase runs are unchanged.

        self.covariance_er = np.asarray(covariance_er)
        self._inv_cov_er   = np.linalg.inv(self.covariance_er)

        if args.NL == "pull" and 'rfis0_geou' in ensp_nom and 'rfis0_geoth' in ensp_nom:
            self._geo_U_pos = ensp_nom['rfis0_geou'].GetWithPositronEnergy(
                type=args.posE_shift, s_ibd=s_ibd,
                inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
            self._geo_Th_pos = ensp_nom['rfis0_geoth'].GetWithPositronEnergy(
                type=args.posE_shift, s_ibd=s_ibd,
                inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
            self._geo_U_pos_integral  = self._geo_U_pos.GetIntegral()
            self._geo_Th_pos_integral = self._geo_Th_pos.GetIntegral()
        else:
            self._geo_U_pos = self._geo_Th_pos = None

        # ── precompute linearized response matrix in (a, b, c) ────────────────────
        if self._linearize_resp:
            a0, b0, c0 = args.a, args.b, args.c
            # Finite-difference steps for the abc linearization come from the ER
            # covariance diagonal (theta0_er order: a, b, c, kB, fC, kI, LY),
            # i.e. each parameter's 1σ prior. In covariance mode (NL != "pull")
            # the abc prior IS covariance_er, so this needs no separate
            # a_err/b_err/c_err config keys.
            eps_a, eps_b, eps_c = np.sqrt(np.diag(self.covariance_er)[:3])

            def _R(a, b, c):
                return CalcRespMatrix_abc(a, b, c, escale=1,
                                          ebins=self.fine_bins,
                                          pebins=self.fine_bins)

            R0  = _R(a0, b0, c0)
            Rap = _R(a0 + eps_a, b0, c0); Ram = _R(a0 - eps_a, b0, c0)
            Rbp = _R(a0, b0 + eps_b, c0); Rbm = _R(a0, b0 - eps_b, c0)
            Rcp = _R(a0, b0, c0 + eps_c); Rcm = _R(a0, b0, c0 - eps_c)

            self._R0      = R0.data
            self._dRda    = (Rap.data - Ram.data) / (2.0 * eps_a)
            self._dRdb    = (Rbp.data - Rbm.data) / (2.0 * eps_b)
            self._dRdc    = (Rcp.data - Rcm.data) / (2.0 * eps_c)
            self._R_ebins = R0.ebins
            self._R_pebins = R0.pebins
            self._a0, self._b0, self._c0 = a0, b0, c0

    # ════════════════════════════════════════════════════════════════════════
    #  Nuisance parameter interface (consumed by combined_fit.py)
    # ════════════════════════════════════════════════════════════════════════
    def set_external_cov(self, C_reco):
        """Store propagated nuisance covariance from DYB+TAO pre-fit."""
        self._external_cov = C_reco

    def get_param_names(self):
        args = self.args
        if args.NL == "pull":
            NL_names = [f"NL{i}" for i in range(5)]
        else:
            NL_names = [f"NL{i}" for i in range(4)]

        bckgs_keys = list(self.backgrounds["JUNO"].keys())

        if args.CI_powers:
            snf_names = [f"snf_noneq_norm{i}" for i in range(2)]
        else:
            snf_names = [f"snf_noneq_norm{i}" for i in range(6)]

        efission_names = [f"efission{i}" for i in range(len(args.efission))]
        core_names     = [f"core_power{i}" for i in range(self.n_cores)]

        # NOTE: me_rho is a shared parameter (lives in combined_fit, not here)
        return (
            ["Nrea"]
            + NL_names
            + ["abc0", "abc1", "abc2"]
            + [f"bkg_{k}" for k in bckgs_keys]
            + snf_names
            + ["eff_nuis"]
            + ["ThU_ratio"]
            + core_names
            + efission_names
        )

    def get_start_values(self):
        args = self.args
        if args.NL == "pull":
            NL_start = [0.0] * 5
        else:
            NL_start = [args.kB, args.fC, args.kI, args.LY_shared]

        abc_start       = [args.a, args.b, args.c]
        bckgs_keys      = list(self.backgrounds["JUNO"].keys())
        bckgs_start     = [0.0] * len(bckgs_keys)

        if args.CI_powers:
            snf_start = [0.0] * 2
        else:
            snf_start = [0.0] * 6

        efission_start = list(np.array(args.efission) * 1e-2)
        core_start     = list(np.zeros(self.n_cores))

        return (
            [0.0]               # Nrea
            + NL_start
            + abc_start
            + bckgs_start
            + snf_start
            + [0.0]             # eff_nuis
            + [args.Th_U_ratio] # ThU_ratio
            + core_start
            + efission_start
        )

    def apply_limits(self, m):
        """Apply Minuit limits for all JUNO-specific parameters."""
        args = self.args
        bckgs_keys = list(self.backgrounds["JUNO"].keys())

        if args.NL == "pull":
            n_NL = 5
        else:
            n_NL = 4

        nuis_limit_minus = (
            ["Nrea"]
            + [f"snf_noneq_norm{i}" for i in range(2 if args.CI_powers else 6)]
            + [f"bkg_{k}" for k in bckgs_keys]
            + ["eff_nuis"]
        )
        for p in nuis_limit_minus:
            m.limits[p] = [-1, None]

        for i in range(n_NL):
            if args.NL == "pull":
                m.limits[f"NL{i}"] = [-1, None]
            else:
                m.limits[f"NL{i}"] = [0, None]

        for i in range(self.n_cores):
            m.limits[f"core_power{i}"] = [-1.0, None]

        for i in range(len(args.efission)):
            m.limits[f"efission{i}"] = [0.0, None]

        m.limits["ThU_ratio"] = [0.0, None]

    def get_first_fix(self):
        """Parameters to fix in the first Minuit stage."""
        args = self.args
        bckgs_keys = list(self.backgrounds["JUNO"].keys())
        n_NL = 5 if args.NL == "pull" else 4
        n_snf = 2 if args.CI_powers else 6
        return (
            ["Nrea"]
            + [f"NL{i}" for i in range(n_NL)]
            + ["abc0", "abc1", "abc2"]
            + [f"snf_noneq_norm{i}" for i in range(n_snf)]
            + [f"bkg_{k}" for k in bckgs_keys]
            + ["eff_nuis", "ThU_ratio"]
            + [f"core_power{i}" for i in range(self.n_cores)]
            + [f"efission{i}" for i in range(len(args.efission))]
        )

    def get_pull_defs(self):
        """
        Return {base_name: callable(value) -> (pull/sigma)**2} mirroring
        the pulls that build_spectrum() appends internally.

        Used by combined_fit.dispatcher.pull_dedup_correction to remove
        the (N-1) over-count when a Minuit parameter is shared across
        multiple bundles (phase × shell). For a single-bundle run, the
        correction is identically zero and behaviour is unchanged.
        """
        from .. import nuosc as nuosc                # local import
        args = self.args
        nom  = nuosc.op_nom
        defs = {}

        # ── oscillation parameters ────────────────────────────────────────
        if args.sin2_th13_opt == "pull":
            defs['sin2_th13'] = (
                lambda v, c=nom['sin2_th13'], s=nom['sin2_th13_err']:
                    ((v - c) / s) ** 2)
        if bool(args.pull_dm2_31):
            # Minuit holds dm2_31 in 1e-3 eV²; nuosc.op_nom['dm2_31'] is in eV².
            # pulls.append(dm2_31 - nuosc.op_nom['dm2_31']) — both in physical eV²
            # is what build_spectrum does (dm2_31 there is dm2_31_phys = Minuit*1e-3).
            # To compute the SAME pull from the raw Minuit value we use
            # physical-space central scaled back into Minuit units:
            defs['dm2_31'] = (
                lambda v, c=nom.get('dm2_31_dyb', nom['dm2_31']) * 1e3,
                          s=nom.get('dm2_31_dyb_err', nom['dm2_31_err']) * 1e3:
                    ((v - c) / s) ** 2)
        # me_rho — build_spectrum emits this pull whenever me_rho != 0.0
        defs['me_rho'] = (
            lambda v, c=args.me_rho, s=args.me_rho_scale:
                ((v - c) / s) ** 2 if v != 0.0 else 0.0)

        # ── core powers (one per reactor core) ────────────────────────────
        for i in range(self.n_cores):
            defs[f'core_power{i}'] = (
                lambda v, s=args.core_flux_unc: (v / s) ** 2)

        # ── efission (length len(args.efission)) ──────────────────────────
        # In build_spectrum: efission_scaled = efission * 100, pull =
        # efission_scaled - args.efission, sigma = args.efission_err.
        # In Minuit units (efission = args.efission/100), central is
        # args.efission/100 and sigma is args.efission_err/100.
        for i, (e_nom, e_err) in enumerate(zip(args.efission, args.efission_err)):
            defs[f'efission{i}'] = (
                lambda v, c=e_nom * 1e-2, s=e_err * 1e-2:
                    ((v - c) / s) ** 2)

        # ── NL / abc — pull mode only ─────────────────────────────────────
        if args.NL == "pull":
            for i in range(4):
                defs[f'NL{i}'] = (lambda v, s=args.NL_sd: (v / s) ** 2)
            defs['NL4'] = (lambda v, s=args.ene_scale_unc: (v / s) ** 2)
            defs['abc0'] = (
                lambda v, c=args.a, s=args.a_err: ((v - c) / s) ** 2)
            defs['abc1'] = (
                lambda v, c=args.b, s=args.b_err: ((v - c) / s) ** 2)
            defs['abc2'] = (
                lambda v, c=args.c, s=args.c_err: ((v - c) / s) ** 2)
        # (When NL != "pull", the joint ER 2D-covariance penalty is handled
        # SEPARATELY by pull_dedup_correction via _get_chi2_penalty_er().)

        # ── SNF / non-equilibrium ─────────────────────────────────────────
        if args.CI_powers:
            defs['snf_noneq_norm0'] = (lambda v, s=args.snf_unc:   (v / s) ** 2)
            defs['snf_noneq_norm1'] = (lambda v, s=args.noneq_unc: (v / s) ** 2)
        else:
            for i in range(6):
                sigma = args.snf_unc if i < 3 else args.noneq_unc
                defs[f'snf_noneq_norm{i}'] = (lambda v, s=sigma: (v / s) ** 2)

        # ── efficiency, Nrea ──────────────────────────────────────────────
        defs['eff_nuis'] = (lambda v, s=args.eff_unc: (v / s) ** 2)
        defs['Nrea']     = (lambda v, s=args.r2_unc:  (v / s) ** 2)

        # ── Th/U ratio (only meaningful when "geo" is in backgrounds) ─────
        bckgs_keys = list(self.backgrounds['JUNO'].keys())
        if 'geo' in bckgs_keys:
            defs['ThU_ratio'] = (
                lambda v, c=args.Th_U_ratio, s=args.Th_U_ratio_err:
                    ((v - c) / s) ** 2)

        # ── background rate pulls ─────────────────────────────────────────
        for k in bckgs_keys:
            sigma = self.backgrounds['JUNO'][k].get('rate_unc', 0.0)
            if sigma > 0:
                defs[f'bkg_{k}'] = (lambda v, s=sigma: (v / s) ** 2)
        return defs

    # ════════════════════════════════════════════════════════════════════════
    #  Internal cached helpers  (identical logic to original run_minuit)
    # ════════════════════════════════════════════════════════════════════════
    def _get_resp_matrix(self, a, b, c):
        # Cache by (a,b,c) for BOTH the linearized and full-recompute paths.
        # build_spectrum calls this on every chi2 eval, and migrad's gradient
        # holds (a,b,c) fixed while it perturbs the ~Nparam nuisances one-by-one,
        # so the same matrix is requested dozens of times per gradient step.
        # Even the "cheap" linear combination R0 + Σ Δ·dR materialises a full
        # fine×fine matrix, so without caching lin_resp=True is SLOWER than the
        # cached non-linear path. The cache is bounded to the working set
        # (current point + gradient neighbours) so it can't grow to multi-GB
        # over a fit or accumulate across FC toys on the persistent truth
        # bundles.
        key = (a, b, c)
        rm = self._resp_matrix_cache.get(key)
        if rm is None:
            if self._linearize_resp:
                R = (self._R0
                     + (a - self._a0) * self._dRda
                     + (b - self._b0) * self._dRdb
                     + (c - self._c0) * self._dRdc)
                rm = RespMatrix(data=R, ebins=self._R_ebins, pebins=self._R_pebins)
            else:
                rm = CalcRespMatrix_abc(
                    a, b, c, escale=1, ebins=self.fine_bins, pebins=self.fine_bins)
            self._resp_matrix_cache[key] = rm
            if len(self._resp_matrix_cache) > 6:
                for k in list(self._resp_matrix_cache.keys())[:3]:
                    if k != key:
                        del self._resp_matrix_cache[k]
        return rm
    def _get_nuisance_fine(self, nuisance_coarse):
        # Bounded like _resp_matrix_cache: Minuit's working set is one point
        # plus its gradient neighbours, so a few hundred entries preserves
        # every hit. Unbounded, an MCMC chain (every eval a fresh nuisance
        # vector) accumulates one fine array per eval in each pool worker.
        key = tuple(nuisance_coarse)
        if key not in self._nuisance_fine_cache:
            self._nuisance_fine_cache[key] = expand_nuisances_to_fine(
                self.E_segments, self.fine_bins, nuisance_coarse)
            if len(self._nuisance_fine_cache) > 256:
                for k in list(self._nuisance_fine_cache.keys())[:128]:
                    if k != key:
                        del self._nuisance_fine_cache[k]
        return self._nuisance_fine_cache[key]

    def _get_chi2_penalty_er(self, a, b, c, kB, fC, kI, LY_shared):
        key = (a, b, c, kB, fC, kI, LY_shared)
        if key not in self._chi2_penalty_er_cache:
            theta = np.array([a, b, c, kB, fC, kI, LY_shared])
            diff  = theta - self.theta0_er
            self._chi2_penalty_er_cache[key] = diff @ self._inv_cov_er @ diff
            if len(self._chi2_penalty_er_cache) > 256:
                for k in list(self._chi2_penalty_er_cache.keys())[:128]:
                    if k != key:
                        del self._chi2_penalty_er_cache[k]
        return self._chi2_penalty_er_cache[key]

    def _get_geo_spectrum(self, ThU_ratio, resp_matrix, a, b, c, kB, fC, kI, LY_shared):
        key = (ThU_ratio, a, b, c, kB, fC, kI, LY_shared)
        if key not in self._geo_spectrum_cache:
            args  = self.args
            ensp  = self.ensp_nom
            ndays = self.ndays
            rate  = float(args.geo_tnu_rate) * float(args.tnu_to_cpd)
            scale = rate * ndays / args.duty_cycle
            Th = ensp['rfis0_geoth'].GetScaledFit(
                ThU_ratio * scale / ((1 + ThU_ratio) * ensp['rfis0_geoth'].GetIntegral()))
            U  = ensp['rfis0_geou'].GetScaledFit(
                scale / ((1 + ThU_ratio) * ensp['rfis0_geou'].GetIntegral()))
            U  = U.GetWithPositronEnergy(type=args.posE_shift,s_ibd=self.s_ibd,
                                         inputfile=args.input_data_file,
                                         tf2name=args.pos_ene_TF2)
            Th = Th.GetWithPositronEnergy(type=args.posE_shift,s_ibd=self.s_ibd,
                                          inputfile=args.input_data_file,
                                          tf2name=args.pos_ene_TF2)
            U  = U.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            Th = Th.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            U = U.ApplyDetResp(self.rm, pecrop=args.ene_crop)
            Th = Th.ApplyDetResp(self.rm, pecrop=args.ene_crop)
            geo_sp = U + Th
            geo_sp.GetScaled(args.detector_efficiency)
            if args.rebin_nonuniform:
                geo_sp.Rebin_nonuniform(args.bins_nonuniform)
            self._geo_spectrum_cache[key] = geo_sp
        return self._geo_spectrum_cache[key]

    def _get_geo_spectrum_pull(self, ThU_ratio, NL, abc, new_nonl, resp_matrix):
            """
            Compute the geo background under NL='pull' mode, cached by (ThU_ratio, NL, abc).
            `resp_matrix` is derived from abc and `new_nonl` from NL, so the key captures
            all dependencies — when those values are unchanged the cached spectrum is reused.
            """
            args  = self.args
            ndays = self.ndays

            key = (round(ThU_ratio, 10),) \
                + tuple(round(x, 10) for x in NL) \
                + tuple(round(x, 10) for x in abc)

            if key in self._geo_spectrum_cache:
                cached = self._geo_spectrum_cache[key]
                # Hand out a copy — downstream may rebin / scale in place
                return Spectrum(bins=cached.bins, bin_cont=cached.bin_cont.copy())

            # ── cache miss: compute it ───────────────────────────────────────────
            rate  = float(args.geo_tnu_rate) * float(args.tnu_to_cpd)
            scale = rate * ndays / args.duty_cycle

            U_pos  = self._geo_U_pos.GetScaledFit(
                scale / ((1 + ThU_ratio) * self._geo_U_pos_integral))
            Th_pos = self._geo_Th_pos.GetScaledFit(
                ThU_ratio * scale / ((1 + ThU_ratio) * self._geo_Th_pos_integral))

            U_nonl  = U_pos.GetWithModifiedEnergy(mode='spectrum',
                                                  spectrum=new_nonl, LY_shared=NL[4])
            Th_nonl = Th_pos.GetWithModifiedEnergy(mode='spectrum',
                                                   spectrum=new_nonl, LY_shared=NL[4])
            U_final  = U_nonl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            Th_final = Th_nonl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

            bckg = U_final + Th_final
            bckg.GetScaled(args.detector_efficiency)
            if args.rebin_nonuniform:
                bckg.Rebin_nonuniform(args.bins_nonuniform)
            self._geo_spectrum_cache[key] = bckg

            # ── store (with FIFO eviction) ──────────────────────────────────────
            if len(self._geo_spectrum_cache) > 100:
                for k in list(self._geo_spectrum_cache.keys())[:50]:
                    del self._geo_spectrum_cache[k]

            # Store a copy so future hits aren't poisoned by anyone mutating `bckg`
            self._geo_spectrum_cache[key] = Spectrum(
                bins=bckg.bins, bin_cont=bckg.bin_cont.copy())
            return bckg

    def _get_wr_spectrum(self, resp_matrix, a, b, c, kB, fC, kI, LY_shared):
        key = (a, b, c, kB, fC, kI, LY_shared)
        if key not in self._wr_spectrum_cache:
            args  = self.args
            ensp  = self.ensp_nom
            ndays = self.ndays
            bckg  = ensp['rfis_wr_pos'].GetWithModifiedEnergy(
                mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            bckg = bckg.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            rate  = self.backgrounds['JUNO']['wr']['rate']
            scale = rate * ndays / (args.duty_cycle * bckg.GetIntegral())
            bckg  = bckg.GetScaledFit(scale*args.detector_efficiency)
            bckg.Rebin_nonuniform(args.bins_nonuniform)
            self._wr_spectrum_cache[key] = bckg
            if len(self._wr_spectrum_cache) > 64:
                for k in list(self._wr_spectrum_cache.keys())[:32]:
                    if k != key:
                        del self._wr_spectrum_cache[k]
        return self._wr_spectrum_cache[key]

    def _get_snf_noneq(self, snf_noneq_norm):
        key = tuple(snf_noneq_norm)
        if key not in self._snf_noneq_cache:
            args = self.args
            ensp = self.ensp_nom
            if not args.CI_powers:
                YJ = ensp['snf_YJ'].GetScaledFit(1 + snf_noneq_norm[0]) \
                   + ensp['noneq_YJ'].GetScaledFit(1 + snf_noneq_norm[3])
                TS = ensp['snf_TS'].GetScaledFit(1 + snf_noneq_norm[1]) \
                   + ensp['noneq_TS'].GetScaledFit(1 + snf_noneq_norm[4])
                HM = ensp['snf_HM'].GetScaledFit(1 + snf_noneq_norm[2]) \
                   + ensp['noneq_HM'].GetScaledFit(1 + snf_noneq_norm[5])
                tot = YJ + TS + HM
            else:
                tot = ensp['snf_final'].GetScaledFit(1 + snf_noneq_norm[0]) \
                    + ensp['noneq_final'].GetScaledFit(1 + snf_noneq_norm[1])
            self._snf_noneq_cache[key] = tot
            if len(self._snf_noneq_cache) > 64:
                for k in list(self._snf_noneq_cache.keys())[:32]:
                    if k != key:
                        del self._snf_noneq_cache[k]
        return self._snf_noneq_cache[key]

    # ════════════════════════════════════════════════════════════════════════
    #  Main spectrum builder
    # ════════════════════════════════════════════════════════════════════════

    def build_spectrum(self, sin2_12, sin2_13, dm2_21, dm2_31,me_rho,
                       Nrea, NL, abc, bckgs_for_chi2, snf_noneq_norm,
                       eff_nuis, ThU_ratio,
                       core_powers_delta, efission,nuisance,
                       opp=False, toy_dm2_31=None):
        """
        Builds the JUNO predicted spectrum.
        Returns (s_juno_reactor_only, s_tot_juno, pulls, pull_unc,
                 chi2_penalty_er, fit_bckgs_spectra, cm_bckg, resp_matrix).
        """

        args  = self.args
        ensp  = self.ensp_nom
        ndays = self.ndays

        pulls    = []
        pull_unc = []
        chi2_penalty_er = 0.0
        fit_bckgs_spectra = {}

        # ── oscillation parameter pulls ─────────────────────────────────────
        if args.sin2_th13_opt == "pull":
            pulls.append(sin2_13 - nuosc.op_nom['sin2_th13'])
            pull_unc.append(nuosc.op_nom['sin2_th13_err'])
        if bool(args.pull_dm2_31):
            # DYB reactor constraint. The toy carries the DYB central already
            # projected to each ordering (dyb_NO / dyb_IO, built in
            # sample_aux_dm2_31 with the -dm2_31 + 2 sin^2(th12) Dm2_21 reactor
            # offset); pick by this fit's ordering. Data path uses the
            # per-ordering op_nom['dm2_31_dyb'].
            t_dyb = _aux_dm2_31(toy_dm2_31, 'dyb_NO' if dm2_31 > 0 else 'dyb_IO')
            if t_dyb is not None:
                targ_dm2_31  = t_dyb
            else:
                targ_dm2_31  = nuosc.op_nom.get('dm2_31_dyb', nuosc.op_nom['dm2_31'])
            pulls.append(dm2_31 - targ_dm2_31)
            pull_unc.append(nuosc.op_nom.get('dm2_31_dyb_err',
                                             nuosc.op_nom['dm2_31_err']))
        if args.pull_dm2_31_lbl:
            ordering = "NO" if dm2_31 > 0 else "IO"
            # Toy: the LBL central is already projected to each ordering
            # (lbl_NO / lbl_IO, built in sample_aux_dm2_31 so the TRUE-ordering
            # fit gets the measurement and the wrong one the f_lbl projection).
            t_lbl = _aux_dm2_31(toy_dm2_31, 'lbl_NO' if dm2_31 > 0 else 'lbl_IO')
            if t_lbl is not None:
                lbl_err = args.dm2_31_lbl_err
                if ordering == "IO":
                    lbl_err = getattr(args, 'dm2_31_lbl_err_IO', None) or lbl_err
                dm2_31_lbl = t_lbl
            else:
                # Data path: direct IO measurement if configured, else project
                # the fixed LBL central to this fit ordering (see lbl_pull_target).
                from .minuit import lbl_pull_target
                dm2_31_lbl, lbl_err = lbl_pull_target(
                    args, ordering, sin2_12, sin2_13, dm2_21)

            pulls.append(dm2_31 - dm2_31_lbl)
            pull_unc.append(lbl_err)

        # ── core powers ─────────────────────────────────────────────────────
        if len(core_powers_delta):
            core_powers_new = list(
                np.array(self.core_powers_std) * (1 + np.array(core_powers_delta)))
            pulls.extend(core_powers_delta)
            pull_unc.extend([args.core_flux_unc] * len(core_powers_delta))
        else:
            core_powers_new = args.core_powers

        # ── efission ────────────────────────────────────────────────────────
        if len(efission):
            efission_scaled = list(np.array(efission) * 1e2)
            pulls.extend(np.array(efission_scaled) - np.array(args.efission))
            pull_unc.extend(args.efission_err)
        else:
            efission_scaled = args.efission

        # ── shape nuisances (shared) ─────────────────────────────────────────
        if args.fit_type == "PMOP_combined":
            nuisance_coarse    = np.array(nuisance)
            nuisance_fine      = self._get_nuisance_fine(nuisance_coarse)
            nuisance_multiplier = 1.0 + nuisance_fine
        else:
            nuisance_multiplier = 1.0

        # ── build raw IBD spectrum ───────────────────────────────────────────

        core_scales = compute_extrafactors_vec(
            args, efission=efission_scaled,
            alpha=self.fission_fracs,
            core_baselines=self.core_baselines,
            core_powers=core_powers_new)

        raw_ibd_tot    = np.sum(self.fission_arrays * core_scales[:, np.newaxis], axis=0)
        raw_ibd_tot    = raw_ibd_tot * nuisance_multiplier
        rfis_param_post = Spectrum(bins=self.dummy_sp.bins, bin_cont=raw_ibd_tot)

        #print("ibd", rfis_param_post.GetIntegral())
        # ── matter density ───────────────────────────────────────────────────
        if me_rho != 0.0:
            pulls.append(me_rho - args.me_rho)
            pull_unc.append(args.me_rho_scale)
            rho = me_rho
        else:
            rho = args.me_rho

        # ── oscillation ──────────────────────────────────────────────────────
        osc_key = (
            round(sin2_12, 12), round(sin2_13, 12),
            round(dm2_21, 14),  round(dm2_31, 14),
            round(rho, 8),
            tuple(round(p, 10) for p in core_powers_new),
            tuple(round(e, 10) for e in efission_scaled),
            tuple(round(n, 10) for n in nuisance) if nuisance is not None else (),
            bool(opp),
        )
        if osc_key in self._osc_cache:
            cached = self._osc_cache[osc_key]
            s_juno = Spectrum(bins=cached.bins, bin_cont=cached.bin_cont.copy())
        else:
            if self._osc_per_core:
                # Oscillate each core's flux at its own baseline, then sum,
                # instead of oscillating the summed flux with power/L^2 shares.
                raw_ibd_cores = (self.fission_arrays * core_scales[:, np.newaxis]
                                 * nuisance_multiplier)
                s_juno = None
                for i in range(self.n_cores):
                    sp_i = Spectrum(bins=self.dummy_sp.bins,
                                    bin_cont=raw_ibd_cores[i])
                    osc_i = sp_i.GetOscillated(
                            L=[self.core_baselines[i]],
                            sin2_th12=sin2_12, sin2_th13=sin2_13,
                            dm2_21=dm2_21, dm2_31=dm2_31,
                            core_powers=[core_powers_new[i]],
                            me_rho=rho, ene_mode='true', opp=opp, args=self.args)
                    s_juno = osc_i if s_juno is None else s_juno + osc_i
            else:
                s_juno = rfis_param_post.GetOscillated(
                        L=list(self.core_baselines),
                        sin2_th12=sin2_12, sin2_th13=sin2_13,
                        dm2_21=dm2_21, dm2_31=dm2_31,
                        core_powers=list(core_powers_new),
                        me_rho=rho, ene_mode='true', opp=opp, args=self.args)
            # Positron-energy mapping is a pure function of the oscillated flux,
            # so fold it INTO the osc cache (store the post-positron spectrum).
            # The scipy IBD positron mapping is the single biggest per-call cost
            # (~40%); caching it here skips it on every osc-cache hit — i.e. all
            # perturbations that leave osc/core/efission/segment params unchanged
            # (backgrounds, eff, Nrea, NL, abc, snf), which is most gradient evals.
            s_juno = s_juno.GetWithPositronEnergy(inputfile=args.input_data_file,
                                                  s_ibd=self.s_ibd,
                                                  tf2name=args.pos_ene_TF2)
            if len(self._osc_cache) > 200:
                # FIFO eviction; Python dicts keep insertion order
                for k in list(self._osc_cache.keys())[:100]:
                    del self._osc_cache[k]
            self._osc_cache[osc_key] = Spectrum(bins=s_juno.bins,
                                                bin_cont=s_juno.bin_cont.copy())
        #print("osc", s_juno.GetIntegral())
        # (positron-energy shift now folded into the osc cache above)

        # ── non-linearity ────────────────────────────────────────────────────
        # The energy-axis remap (GetWithModifiedEnergy) is ~29% of a call; it is
        # a pure function of (post-positron flux = osc_key, NL params), so cache
        # it by (osc_key, NL). new_nonl, the NL pulls and the ER penalty are
        # cheap and computed UNCONDITIONALLY — new_nonl feeds the geo-pull
        # background and the pulls/penalty feed the total chi2 — so only the
        # expensive remap itself is skipped on a hit (every perturbation that
        # leaves osc/core/efission/segment AND NL params unchanged: backgrounds,
        # eff, Nrea, snf, abc, ThU).
        if NL:
            if args.NL == "pull":
                new_nonl = Spectrum(bins=ensp['scintNL'].bins,
                                    bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
                new_nonl.bin_cont = (ensp['scintNL'].bin_cont
                    + NL[0] * (ensp['NL_pull'][0].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[1] * (ensp['NL_pull'][1].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[2] * (ensp['NL_pull'][2].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[3] * (ensp['NL_pull'][3].bin_cont - ensp['scintNL'].bin_cont))
                pulls.extend([NL[0], NL[1], NL[2], NL[3], NL[4]])
                pull_unc.extend([args.NL_sd] * 4)
                pull_unc.append(args.ene_scale_unc)
                # Exact floats (not rounded): a non-NL perturbation passes the
                # identical NL value, so exact equality still hits, and the
                # cached spectrum then reproduces the no-cache result bit-for-bit.
                nl_tag = ('pull', NL[0], NL[1], NL[2], NL[3], NL[4])
            else:
                kB = NL[0]; fC = NL[1]; kI = NL[2]; LY_shared = NL[3]
                chi2_penalty_er = self._get_chi2_penalty_er(
                    abc[0], abc[1], abc[2], kB, fC, kI, LY_shared)
                new_nonl = None
                nl_tag = ('g4', kB, fC, kI, LY_shared)
        else:
            new_nonl = None  # not used below in the NL=="pull" geo branch
            nl_tag = ('pull_nom',) if args.NL == "pull" else ('g4_nom',)

        nl_key = (osc_key, nl_tag)
        cached_nl = self._nl_cache.get(nl_key)
        if cached_nl is not None:
            s_juno = Spectrum(bins=cached_nl.bins, bin_cont=cached_nl.bin_cont.copy())
        else:
            if NL:
                if args.NL == "pull":
                    s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum',
                                                          spectrum=new_nonl, LY_shared=NL[4])
                else:
                    s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=NL[0], fC=NL[1],
                                                          kI=NL[2], LY_shared=NL[3])
            else:
                if args.NL == "pull":
                    s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum',
                                                          spectrum=ensp['scintNL'])
                else:
                    s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=args.kB, fC=args.fC,
                                                          kI=args.kI, LY_shared=args.LY_shared)
            if len(self._nl_cache) > 200:
                for k in list(self._nl_cache.keys())[:100]:
                    del self._nl_cache[k]
            self._nl_cache[nl_key] = Spectrum(bins=s_juno.bins,
                                              bin_cont=s_juno.bin_cont.copy())

        # ── detector response ────────────────────────────────────────────────
        if abc:
            resp_matrix = self._get_resp_matrix(abc[0], abc[1], abc[2])
            s_juno = s_juno.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            if args.NL == "pull":
                pulls.extend([abc[0] - args.a, abc[1] - args.b, abc[2] - args.c])
                pull_unc.extend([args.a_err, args.b_err, args.c_err])
        else:
            resp_matrix = self.rm
            s_juno = s_juno.ApplyDetResp(self.rm, pecrop=args.ene_crop)
        #print("det", s_juno.GetIntegral())
        # ── SNF + non-equilibrium ────────────────────────────────────────────
        if snf_noneq_norm:
            pulls.extend(snf_noneq_norm)
            pull_unc.extend([args.snf_unc] * 3)
            pull_unc.extend([args.noneq_unc] * 3)
            s_juno = s_juno + self._get_snf_noneq(snf_noneq_norm)
        else:
            s_juno = s_juno + ensp['snf_final'] + ensp['noneq_final']
        #print("final", s_juno.GetIntegral())
        # ── efficiency ───────────────────────────────────────────────────────
        if eff_nuis:
            pulls.append(eff_nuis)
            pull_unc.append(args.eff_unc)
            s_juno = s_juno.GetScaledFit(1 + eff_nuis)

        # ── reactor normalisation ────────────────────────────────────────────
        if Nrea:
            pulls.append(Nrea)
            pull_unc.append(args.r2_unc)
            s_juno = s_juno.GetScaledFit(1 + Nrea)

        if args.rebin_nonuniform:
            s_juno.Rebin_nonuniform(args.bins_nonuniform)

        s_tot_juno = s_juno
        cm_bckg    = None

        # ── backgrounds ─────────────────────────────────────────────────────
        for key, info in bckgs_for_chi2.items():
            if key == "geo":
                pulls.append(ThU_ratio - args.Th_U_ratio)
                pull_unc.append(args.Th_U_ratio_err)
                if args.NL == "pull":
                    bckg = self._get_geo_spectrum_pull(ThU_ratio, NL, abc, new_nonl, resp_matrix)
                else:
                    bckg = self._get_geo_spectrum(ThU_ratio, resp_matrix,
                                                  abc[0], abc[1], abc[2],
                                                  NL[0] if NL else args.kB,
                                                  NL[1] if NL else args.fC,
                                                  NL[2] if NL else args.kI,
                                                  NL[3] if NL else args.LY_shared)

            elif key == "wr":
                if args.NL == "pull":
                    bckg = ensp['rfis_wr_pos'].GetWithModifiedEnergy(
                        mode='spectrum', spectrum=new_nonl, LY_shared=NL[4])
                    bckg = bckg.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
                    rate  = self.backgrounds['JUNO']['wr']['rate']
                    scale = rate * ndays / (args.duty_cycle * bckg.GetIntegral())
                    bckg  = bckg.GetScaledFit(scale*args.detector_efficiency)
                    if args.rebin_nonuniform:
                        bckg.Rebin_nonuniform(args.bins_nonuniform)
                else:
                    bckg = self._get_wr_spectrum(resp_matrix,
                                                 abc[0], abc[1], abc[2],
                                                 NL[0] if NL else args.kB,
                                                 NL[1] if NL else args.fC,
                                                 NL[2] if NL else args.kI,
                                                 NL[3] if NL else args.LY_shared)
            else:
                bckg = ensp[key]

            if info["sigma"] > 0.0:
                pulls.append(info["pull"])
                pull_unc.append(info["sigma"])

            bckg = bckg.GetScaledFit(1 + info["pull"])

            if bckg == "geo" or bckg == "wr":
                bckg = bckg.GetScaledFit(1 + eff_nuis)

            if info["b2b"] > 0:
                b2bunc = info["b2b"]
                if cm_bckg is None:
                    cm_bckg = bckg.GetB2BCovMatrix(b2bunc, info['b2b_scale'])
                else:
                    cm_bckg += bckg.GetB2BCovMatrix(b2bunc, info['b2b_scale'])

            s_tot_juno += bckg
            fit_bckgs_spectra[info["name"]] = bckg
        return (s_juno, s_tot_juno, pulls, pull_unc,
                chi2_penalty_er, fit_bckgs_spectra, cm_bckg, resp_matrix, rfis_param_post)

    # ════════════════════════════════════════════════════════════════════════
    #  chi2 contribution
    # ════════════════════════════════════════════════════════════════════════

    def chi2_contribution(self, sin2_12, sin2_13, dm2_21, dm2_31,me_rho,
                          Nrea, NL, abc, bckgs_for_chi2, snf_noneq_norm,
                          eff_nuis, ThU_ratio,
                          core_powers_delta, efission, nuisance,
                          unc, opp=False, toy_dm2_31=None):
        """
        Returns (chi2_juno, chi2_penalty_er, s_juno, s_tot_juno,
                 fit_bckgs_spectra, rfis_param_post).
        Caller adds chi2_penalty_er to the total.
        """
        args  = self.args

        (s_juno, s_tot_juno, pulls, pull_unc,
         chi2_penalty_er, fit_bckgs_spectra, cm_bckg,
         resp_matrix, rfis_param_post) = self.build_spectrum(
            sin2_12, sin2_13, dm2_21, dm2_31,me_rho,
            Nrea, NL, abc, bckgs_for_chi2, snf_noneq_norm,
            eff_nuis, ThU_ratio,
            core_powers_delta, efission, nuisance, opp=opp, toy_dm2_31=toy_dm2_31)

        sp_obs = self._get_obs_spectrum()

        if unc != 'stat':
            cm_j = self.cm[unc]
        else:
            ebins = self.ensp_nom['rdet'].bins
            nbins = len(ebins)
            cm_j  = CovMatrix(data=np.zeros((nbins - 1, nbins - 1)), bins=ebins)

        cm_j = cm_j + cm_bckg
        if self._external_cov is not None:
            ebins = self.ensp_nom['rdet'].bins
            cm_j = cm_j + CovMatrix(data=self._external_cov, bins=ebins)

        chi2_nolog, chi2_juno = Chi2(
            cm_j, sp_obs, s_tot_juno, sp_obs, s_tot_juno,
            'bckg',
            args.stat_method_opt,
            pulls=pulls, pull_unc=pull_unc, include_logV=False)

        return (chi2_juno, chi2_nolog, chi2_penalty_er,
                s_juno, s_tot_juno, fit_bckgs_spectra, rfis_param_post, cm_j)

    def _get_obs_spectrum(self):
        args = self.args
        ensp = self.ensp_nom
        if bool(args.toymc):
            return ensp["rtot_toy"]
        elif args.is_data:
            return ensp['data']
        else:
            return ensp["rtot"]
