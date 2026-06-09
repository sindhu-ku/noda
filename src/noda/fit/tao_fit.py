#!/usr/bin/env python3
"""
tao_fit.py

TAO detector module.

Key differences from JUNO:
  - NL is always "pull" style (no G4 kB/fC/kI mode)
  - abc (a, b, c) has its own independent pull penalties using args_tao errors
    (no 2-D covariance matrix penalty)
  - Single background: ensp_nom['bckg'], pull sigma = sqrt(bckg.GetIntegral())
  - One core-power nuisance: core_power_tao
  - me_rho is shared (received from outside, not owned here)

Shared quantities received at call time:
  - sin2_12, sin2_13, dm2_21, dm2_31  -- oscillation params
  - me_rho                             -- shared matter density
  - nuisance                           -- shared shape nuisances (may be empty)
"""

import numpy as np
import scipy as sp
from ..noda import *
from .. import nuosc as nuosc


class TAODetector:
    """
    Builds the TAO predicted spectrum and returns a chi2 contribution.
    """

    def __init__(self, ndays, ensp_nom, rm, cm, args,
                 core_baselines, core_powers_std,
                 s_ibd, ene_leak_tao, E_segments):
        self.ndays           = ndays
        self.ensp_nom        = ensp_nom
        self.rm              = rm
        self.cm              = cm
        self.args            = args          # args_tao
        self.core_baselines  = core_baselines
        self.core_powers_std = core_powers_std
        self.fine_bins       = ensp_nom['ribd'].bins
        self.s_ibd           = s_ibd
        self.ene_leak        = ene_leak_tao
        self.E_segments      = E_segments

        # ── pre-build fission arrays ─────────────────────────────────────────
        n_cores  = len(core_baselines)
        dummy_sp = ensp_nom["rfis0"]
        self.dummy_sp = dummy_sp
        self.n_cores  = n_cores

        fission_fracs = []
        for power in core_powers_std:
            fission_fracs.append([self.args.U235_scale, self.args.U238_scale,
                                   self.args.Pu239_scale, self.args.Pu241_scale])
        self.fission_fracs = fission_fracs

        fission_arrays = np.zeros((n_cores, len(dummy_sp.bin_cont)))
        for i in range(n_cores):
            dummy_rfis = Spectrum(bins=ensp_nom["rfis0"].bins,
                                  bin_cont=ensp_nom["rfis0"].bin_cont)
            dummy_rfis.WeightWithFunction(s_ibd)
            dummy_rfis.ScaleByBinWidth()
            dummy_rfis.GetScaled(ndays)
            fission_arrays[i] = dummy_rfis.bin_cont
        self.fission_arrays = fission_arrays

        # ── per-call caches ──────────────────────────────────────────────────
        self._resp_matrix_cache = {}
        self._osc_cache         = {}
        self._nuisance_fine_cache  = {}
        self._linearize_resp = args.lin_resp
        if self._linearize_resp:
            a0, b0, c0 = args.a, args.b, args.c
            eps_a, eps_b, eps_c = args.a_err, args.b_err, args.c_err

            def _M(a, b, c):
                R = CalcRespMatrix_abc(a, b, c, escale=1,
                                       ebins=self.fine_bins, pebins=self.fine_bins)
                return R.data

            M0  = _M(a0, b0, c0)
            Map = _M(a0 + eps_a, b0, c0); Mam = _M(a0 - eps_a, b0, c0)
            Mbp = _M(a0, b0 + eps_b, c0); Mbm = _M(a0, b0 - eps_b, c0)
            Mcp = _M(a0, b0, c0 + eps_c); Mcm = _M(a0, b0, c0 - eps_c)

            self._M0   = M0
            self._dMda = (Map - Mam) / (2.0 * eps_a)
            self._dMdb = (Mbp - Mbm) / (2.0 * eps_b)
            self._dMdc = (Mcp - Mcm) / (2.0 * eps_c)
            self._a0, self._b0, self._c0 = a0, b0, c0

    # ════════════════════════════════════════════════════════════════════════
    #  Nuisance parameter interface
    # ════════════════════════════════════════════════════════════════════════

    def get_param_names(self):
        """TAO-specific Minuit parameter names."""
        # NL pull: 5 parameters (same as JUNO pull mode: NL0-NL3 + LY)
        NL_names  = [f"NL_tao{i}" for i in range(5)]
        abc_names = ["abc_tao0", "abc_tao1", "abc_tao2"]
        return (
            ["Nrea_tao"]
            + NL_names
            + abc_names
            + [f"snf_noneq_norm_tao{i}" for i in range(2)]
            + ["core_power_tao"]
            + ["eff_nuis_tao"]
            + ["bckg_rate"]

        )

    def get_start_values(self):
        """Start values matching get_param_names() order."""
        args = self.args
        NL_start  = [0.0] * 5
        abc_start = [args.a, args.b, args.c]
        return (
            [0.0]        # Nrea_tao
            + NL_start
            + abc_start
            + [0.0, 0.0]
            + [0.0]      # core_power_tao
            + [0.0]      # eff_nuis_tao
            + [0.0]
        )

    def apply_limits(self, m):
        """Apply Minuit limits for TAO-specific parameters."""
        for p in ["Nrea_tao", "core_power_tao", "eff_nuis_tao", "snf_noneq_norm_tao0", "snf_noneq_norm_tao1", "bckg_rate"]:
            m.limits[p] = [-1, None]
        # NL pull params: unconstrained below (pull penalty handles it)
        for i in range(5):
            m.limits[f"NL_tao{i}"] = [-1, None]

        # abc: unconstrained (individual pull penalties)
        # no explicit limits needed beyond [-inf, inf]

    def get_first_fix(self):
        """Parameters to fix in the first Minuit stage."""
        return (
            ["Nrea_tao"]
            + [f"NL_tao{i}" for i in range(5)]
            + ["abc_tao0", "abc_tao1", "abc_tao2"]
            + [f"snf_noneq_norm_tao{i}" for i in range(2)]
            + ["core_power_tao"]
            + ["eff_nuis_tao"]
            + ["bckg_rate"]
        )

    # ════════════════════════════════════════════════════════════════════════
    #  Internal cached helpers
    # ════════════════════════════════════════════════════════════════════════

    def _get_resp_matrix(self, a, b, c):
        if self._linearize_resp:
            M = (self._M0
                 + (a - self._a0) * self._dMda
                 + (b - self._b0) * self._dMdb
                 + (c - self._c0) * self._dMdc)
            return RespMatrix(data=M, ebins=self.fine_bins, pebins=self.fine_bins)
        # fallback
        key = (a, b, c)
        if key not in self._resp_matrix_cache:
            resp_matrix = CalcRespMatrix_abc(a, b, c, escale=1,
                                             ebins=self.fine_bins, pebins=self.fine_bins)
            self._resp_matrix_cache[key] = RespMatrix(
                data=resp_matrix.data,
                ebins=self.fine_bins, pebins=self.fine_bins)
        return self._resp_matrix_cache[key]

    def _get_nuisance_fine(self, nuisance_coarse):
        key = tuple(nuisance_coarse)
        if key not in self._nuisance_fine_cache:
            self._nuisance_fine_cache[key] = expand_nuisances_to_fine(
                self.E_segments, self.fine_bins, nuisance_coarse)
        return self._nuisance_fine_cache[key]
    # ════════════════════════════════════════════════════════════════════════
    #  Main spectrum builder
    # ════════════════════════════════════════════════════════════════════════

    def build_spectrum(self,
                       Nrea_tao, NL_tao, abc_tao, snf_noneq_norm_tao,
                       core_power_tao, eff_nuis_tao, bckg_rate, nuisance):
        """
        Builds the TAO predicted spectrum.

        Returns (s_tao_reactor, s_tot_tao, pulls, pull_unc).
        """
        args  = self.args
        ensp  = self.ensp_nom
        ndays = self.ndays

        pulls    = []
        pull_unc = []

        # ── core powers (single core nuisance) ───────────────────────────────
        core_powers_new = list(
            np.array(self.core_powers_std) * (1 + core_power_tao))
        pulls.append(core_power_tao)
        pull_unc.append(args.core_flux_unc)

        nuisance_coarse = np.array(nuisance)
        nuisance_fine      = self._get_nuisance_fine(nuisance_coarse)
        nuisance_multiplier = 1.0 + nuisance_fine
        # ── build raw IBD spectrum ───────────────────────────────────────────
        core_scales = np.zeros(self.n_cores)
        for i in range(self.n_cores):
            core_scales[i] = compute_extrafactors(
                args,
                efission=args.efission,
                alpha=self.fission_fracs[i],
                core_baselines=[self.core_baselines[i]],
                core_powers=[core_powers_new[i]]
            )
        raw_ibd_tot     = np.sum(self.fission_arrays * core_scales[:, np.newaxis], axis=0)
        raw_ibd_tot     = raw_ibd_tot * nuisance_multiplier
        rfis_param_post = Spectrum(bins=self.dummy_sp.bins, bin_cont=raw_ibd_tot)


        s_tao = rfis_param_post

        # ── positron energy shift ────────────────────────────────────────────
        s_tao = s_tao.GetWithPositronEnergy(s_ibd=self.s_ibd,inputfile=args.input_data_file,
                                             tf2name=args.pos_ene_TF2).ApplyDetResp(self.ene_leak, pecrop=args.ene_crop)

        # ── NL: always pull mode ─────────────────────────────────────────────
        # NL_tao = [pull0, pull1, pull2, pull3, LY_pull]  (5 params)
        new_nonl = Spectrum(bins=ensp['scintNL'].bins,
                            bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
        new_nonl.bin_cont = (ensp['scintNL'].bin_cont
            + NL_tao[0] * (ensp['NL_pull'][0].bin_cont - ensp['scintNL'].bin_cont)
            + NL_tao[1] * (ensp['NL_pull'][1].bin_cont - ensp['scintNL'].bin_cont)
            + NL_tao[2] * (ensp['NL_pull'][2].bin_cont - ensp['scintNL'].bin_cont)
            + NL_tao[3] * (ensp['NL_pull'][3].bin_cont - ensp['scintNL'].bin_cont))
        s_tao = s_tao.GetWithModifiedEnergy(mode='spectrum',
                                             spectrum=new_nonl, LY_shared=NL_tao[4])
        pulls.extend([NL_tao[0], NL_tao[1], NL_tao[2], NL_tao[3], NL_tao[4]])
        pull_unc.extend([args.NL_sd] * 4)
        pull_unc.append(args.ene_scale_unc)

        # ── detector response with individual abc pull penalties ──────────────
        if abc_tao:
            resp_matrix = self._get_resp_matrix(abc_tao[0], abc_tao[1], abc_tao[2])
            s_tao = s_tao.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            # Individual pull terms (no 2D covariance penalty)
            pulls.extend([abc_tao[0] - args.a,
                          abc_tao[1] - args.b,
                          abc_tao[2] - args.c])
            pull_unc.extend([args.a_err, args.b_err, args.c_err])
        else:
            s_tao = s_tao.ApplyDetResp(self.rm, pecrop=args.ene_crop)

        if snf_noneq_norm_tao:
            s_tao = s_tao + ensp['snf_final'].GetScaledFit(1+snf_noneq_norm_tao[0]) +  ensp['noneq_final'].GetScaledFit(1+snf_noneq_norm_tao[1])
            pulls.extend(snf_noneq_norm_tao)
            pull_unc.extend([args.snf_unc, args.noneq_unc])

        else:
            s_tao = s_tao + ensp['snf_final'] +  ensp['noneq_final']
        # ── efficiency ───────────────────────────────────────────────────────
        #if eff_nuis_tao:
        pulls.append(eff_nuis_tao)
        pull_unc.append(args.eff_unc)
        s_tao.GetScaled(1 + eff_nuis_tao)

        # ── reactor normalisation ────────────────────────────────────────────
        #if Nrea_tao:
        pulls.append(Nrea_tao)
        pull_unc.append(args.r2_unc)
        s_tao = s_tao.GetScaledFit(1 + Nrea_tao)

        if args.rebin_nonuniform:
            s_tao.Rebin_nonuniform(args.bins_nonuniform)

        bckg = ensp['bckg'].GetScaledFit(1+bckg_rate)
        if bckg_rate:
            pulls.append(bckg_rate)
            pull_unc.append(args.bckg_rate_unc)

        s_tot_tao = s_tao + bckg

        #print(s_tot_tao.GetIntegral())

        return s_tao, s_tot_tao, pulls, pull_unc, rfis_param_post, bckg

    # ════════════════════════════════════════════════════════════════════════
    #  chi2 contribution
    # ════════════════════════════════════════════════════════════════════════

    def chi2_contribution(self,
                          Nrea_tao, NL_tao, abc_tao, snf_noneq_norm_tao,
                          core_power_tao, eff_nuis_tao,bckg_rate,
                          unc, nuisance):
        """
        Returns (chi2_tao, chi2_nolog, s_tao, s_tot_tao).
        """
        args = self.args

        (s_tao, s_tot_tao, pulls, pull_unc,
         rfis_param_post, bckg) = self.build_spectrum(
            Nrea_tao, NL_tao, abc_tao, snf_noneq_norm_tao,
            core_power_tao, eff_nuis_tao,bckg_rate, nuisance)

        sp_obs = self._get_obs_spectrum()

        if unc != 'stat':
            cm_t  = CovMatrix(
            data=self.cm['all'].data * np.outer(s_tao.bin_cont, s_tao.bin_cont),
            bins=self.cm['all'].bins)
        else:
            ebins = self.ensp_nom['rdet'].bins
            nbins = len(ebins)
            cm_t  = CovMatrix(data=np.zeros((nbins - 1, nbins - 1)), bins=ebins)

        chi2_nolog, chi2_tao = Chi2(
            cm_t, sp_obs, s_tot_tao, sp_obs, s_tot_tao,
            'all', args.stat_method_opt,
            pulls=pulls, pull_unc=pull_unc, include_logV=False)

        return chi2_tao, chi2_nolog, sp_obs, s_tot_tao, bckg, cm_t

    def _get_obs_spectrum(self):
        ensp = self.ensp_nom
        args = self.args
        if args.is_data:
            return ensp['data']

        else:
            return ensp["rtot"]
