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
                 s_ibd, s_bump_lin, ene_leak_tao, E_segments):
        self.ndays           = ndays
        self.ensp_nom        = ensp_nom
        self.rm              = rm
        self.cm              = cm
        self.args            = args          # args_tao
        self.core_baselines  = core_baselines
        self.core_powers_std = core_powers_std
        self.fine_bins       = ensp_nom['ribd'].bins
        self.s_ibd           = s_ibd
        self.s_bump_lin      = s_bump_lin
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
            if self.args.apply_bump:
                dummy_rfis = dummy_rfis.GetWeightedWithFunction(s_bump_lin)
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

    @property
    def _nl_in_leak(self):
        """When True, the NL is baked into the energy-leakage matrix, so TAO
        applies no separate NL curve and carries no NL fit nuisances."""
        return getattr(self.args, "nl_in_leak", False)

    @property
    def _tao_snf(self):
        """When False (config fit_options.tao_snf_corr: False), the SNF flux
        correction is dropped from the TAO model spectrum entirely — only the
        non-equilibrium correction is kept. The snf nuisance stays in the
        parameter set but, multiplying nothing, simply relaxes to its prior."""
        return getattr(self.args, "tao_snf_corr", True)

    def get_param_names(self):
        """TAO-specific Minuit parameter names.

        The NL pull block (5 shape pulls + shared LY at index 5) is included
        only when NL is NOT baked into the leakage matrix (nl_in_leak False).
        """
        NL_names  = [] if self._nl_in_leak else [f"NL_tao{i}" for i in range(6)]
        abc_names = ["abc_tao0", "abc_tao1", "abc_tao2"]
        # Energy-scale nuisance on the visible energy (before reco). In the
        # old NL-pull mode this lived in NL_tao[5] (LY); with the baked-in
        # leakage matrix (nl_in_leak) it is a standalone floated parameter.
        escale_names = ["escale_tao"] if self._nl_in_leak else []
        return (
            ["Nrea_tao"]
            + NL_names
            + abc_names
            + escale_names
            + [f"snf_noneq_norm_tao{i}" for i in range(2)]
            + ["core_power_tao"]
            + ["eff_nuis_tao"]
            + ["bckg_rate"]

        )

    def get_start_values(self):
        """Start values matching get_param_names() order."""
        args = self.args
        NL_start  = [] if self._nl_in_leak else [0.0] * 6
        abc_start = [args.a, args.b, args.c]
        escale_start = [0.0] if self._nl_in_leak else []   # escale_tao
        return (
            [0.0]        # Nrea_tao
            + NL_start
            + abc_start
            + escale_start
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
        if not self._nl_in_leak:
            m.limits[f"NL_tao5"] = [-1, None]

        # abc: unconstrained (individual pull penalties)
        # no explicit limits needed beyond [-inf, inf]

    def get_first_fix(self):
        """Parameters to fix in the first Minuit stage."""
        NL_names = [] if self._nl_in_leak else [f"NL_tao{i}" for i in range(6)]
        escale_names = ["escale_tao"] if self._nl_in_leak else []
        return (
            ["Nrea_tao"]
            + NL_names
            + ["abc_tao0", "abc_tao1", "abc_tao2"]
            + escale_names
            + [f"snf_noneq_norm_tao{i}" for i in range(2)]
            + ["core_power_tao"]
            + ["eff_nuis_tao"]
            + ["bckg_rate"]
        )

    # ════════════════════════════════════════════════════════════════════════
    #  Internal cached helpers
    # ════════════════════════════════════════════════════════════════════════

    def _get_resp_matrix(self, a, b, c, escale=1.0):
        # Cache by (a,b,c,escale) for BOTH paths: migrad holds (a,b,c) fixed while it
        # perturbs the other nuisances, so even the linearized combination
        # (which still materialises a full fine×fine matrix) must be cached or
        # lin_resp=True ends up slower than the cached non-linear path. Bounded
        # to the working set to avoid multi-GB growth across the fit / FC toys.
        # escale multiplies the visible energy (loc = escale·Evis) before reco.
        key = (a, b, c, escale)
        rm = self._resp_matrix_cache.get(key)
        if rm is None:
            if self._linearize_resp and escale == 1.0:
                M = (self._M0
                     + (a - self._a0) * self._dMda
                     + (b - self._b0) * self._dMdb
                     + (c - self._c0) * self._dMdc)
                rm = RespMatrix(data=M, ebins=self.fine_bins, pebins=self.fine_bins)
            else:
                resp_matrix = CalcRespMatrix_abc(
                    a, b, c, escale=escale,
                    ebins=self.fine_bins, pebins=self.fine_bins,
                    escale_on_vis=(str(getattr(self.args, 'escale_mode', 'reco')) == 'vis'))
                rm = RespMatrix(data=resp_matrix.data,
                                ebins=self.fine_bins, pebins=self.fine_bins)
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
    # ════════════════════════════════════════════════════════════════════════
    #  Main spectrum builder
    # ════════════════════════════════════════════════════════════════════════

    def build_spectrum(self,
                       Nrea_tao, abc_tao, snf_noneq_norm_tao,
                       core_power_tao, eff_nuis_tao, bckg_rate, nuisance,
                       NL_tao=None, escale_tao=0.0):
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

        #print("ibd" , s_tao.GetIntegral())

        # ── positron energy shift ────────────────────────────────────────────
        s_tao = s_tao.GetWithPositronEnergy(s_ibd=self.s_ibd,inputfile=args.input_data_file,
                                             tf2name=args.pos_ene_TF2).ApplyDetResp(self.ene_leak, pecrop=args.ene_crop)

        # ── NL: pull mode, unless it is baked into the leakage matrix ────────
        # When nl_in_leak is set (new-style TAO inputs), the energy-leakage
        # matrix applied above already carries the nominal non-linearity, so
        # NL_tao is not a fit parameter and this block is skipped entirely.
        # Otherwise NL_tao = [pull0..pull4, LY_pull]: 5 shape pulls + shared LY.
        if NL_tao is not None:
            new_nonl = Spectrum(bins=ensp['scintNL'].bins,
                                bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
            new_nonl.bin_cont = (ensp['scintNL'].bin_cont
                + NL_tao[0] * (ensp['NL_pull'][0].bin_cont - ensp['scintNL'].bin_cont)
                + NL_tao[1] * (ensp['NL_pull'][1].bin_cont - ensp['scintNL'].bin_cont)
                + NL_tao[2] * (ensp['NL_pull'][2].bin_cont - ensp['scintNL'].bin_cont)
                + NL_tao[3] * (ensp['NL_pull'][3].bin_cont - ensp['scintNL'].bin_cont)
                + NL_tao[4] * (ensp['NL_pull'][4].bin_cont - ensp['scintNL'].bin_cont))
            s_tao = s_tao.GetWithModifiedEnergy(mode='spectrum',
                                                 spectrum=new_nonl, LY_shared=NL_tao[5])
            pulls.extend([NL_tao[0], NL_tao[1], NL_tao[2], NL_tao[3], NL_tao[4], NL_tao[5]])
            pull_unc.extend([args.NL_sd] * 5)
            pull_unc.append(args.ene_scale_unc)

        # ── detector response with individual abc pull penalties ──────────────
        # escale_tao multiplies the visible energy (loc = (1+escale)·Evis) before
        # reco — the TAO energy-scale nuisance (σ = ene_scale_unc), active with the
        # baked-in leakage matrix where the old LY/NL pull is switched off.
        if abc_tao:
            resp_matrix = self._get_resp_matrix(abc_tao[0], abc_tao[1], abc_tao[2],
                                                escale=1.0 + escale_tao)
            s_tao = s_tao.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            # Individual pull terms (no 2D covariance penalty)
            pulls.extend([abc_tao[0] - args.a,
                          abc_tao[1] - args.b,
                          abc_tao[2] - args.c])
            pull_unc.extend([args.a_err, args.b_err, args.c_err])
            if self._nl_in_leak:
                pulls.append(escale_tao)
                pull_unc.append(args.ene_scale_unc)
        else:
            s_tao = s_tao.ApplyDetResp(self.rm, pecrop=args.ene_crop)

        if snf_noneq_norm_tao:
            if self._tao_snf:
                s_tao = s_tao + ensp['snf_final'].GetScaledFit(1+snf_noneq_norm_tao[0]) +  ensp['noneq_final'].GetScaledFit(1+snf_noneq_norm_tao[1])
            else:
                # SNF correction removed completely: keep only non-equilibrium.
                s_tao = s_tao + ensp['noneq_final'].GetScaledFit(1+snf_noneq_norm_tao[1])
            pulls.extend(snf_noneq_norm_tao)
            pull_unc.extend([args.snf_unc, args.noneq_unc])

        else:
            s_tao = s_tao + (ensp['snf_final'] if self._tao_snf else 0) +  ensp['noneq_final']
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
    #  Step-by-step TAO spectra
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _stretch_energy(spec, k):
        """Map E -> k*E on the same binning (content-conserving)."""
        e = np.asarray(spec.bins, float)
        cum = np.concatenate([[0.0], np.cumsum(spec.bin_cont)])
        cum_new = np.interp(e, k * e, cum, left=0.0, right=cum[-1])
        return Spectrum(bins=spec.bins, bin_cont=np.diff(cum_new))

    def build_stages(self, Nrea_tao, abc_tao, snf_noneq_norm_tao,
                     core_power_tao, eff_nuis_tao, bckg_rate, nuisance,
                     NL_tao=None, escale_tao=0.0):
        """Spectra at every step of the TAO chain, same inputs as
        build_spectrum.  Eff/Nrea normalisations are applied to every stage
        so all stages are in events.

          unosc       IBD flux x xsec x exposure (true E), segments applied
          osc         = unosc (43 m: oscillation not applied in the TAO model)
          snf, noneq  SNF / non-eq corrections (true E), x (1+pull)
          osc_wcorr   osc + snf + noneq (true E)
          dep_noleak  osc_wcorr after positron conversion (deposited E)
          dep         after leakage+LSNL matrix
          vis         after energy scale (E -> (1+escale_tao)*E)
          rec         after resolution (a,b,c), fine binning
          reactor     the model reactor spectrum the fit uses (analysis bins)
          bkg, totalpred
        Returns (stages dict of Spectrum, params dict)."""
        args = self.args
        ensp = self.ensp_nom
        norm = (1 + eff_nuis_tao) * (1 + Nrea_tao)

        core_powers_new = list(np.array(self.core_powers_std) * (1 + core_power_tao))
        core_scales = np.array([compute_extrafactors(
            args, efission=args.efission, alpha=self.fission_fracs[i],
            core_baselines=[self.core_baselines[i]],
            core_powers=[core_powers_new[i]]) for i in range(self.n_cores)])
        raw = np.sum(self.fission_arrays * core_scales[:, np.newaxis], axis=0)
        raw = raw * (1.0 + self._get_nuisance_fine(np.array(nuisance)))
        st = {}
        st["unosc"] = Spectrum(bins=self.dummy_sp.bins, bin_cont=raw * norm)
        st["osc"]   = Spectrum(bins=self.dummy_sp.bins, bin_cont=raw * norm)
        f_snf   = (1 + snf_noneq_norm_tao[0]) if snf_noneq_norm_tao else 1.0
        f_noneq = (1 + snf_noneq_norm_tao[1]) if snf_noneq_norm_tao else 1.0
        snf = ensp["snf"].GetScaledFit(f_snf * norm if self._tao_snf else 0.0)
        noneq = ensp["noneq"].GetScaledFit(f_noneq * norm)
        st["snf"], st["noneq"] = snf, noneq
        st["osc_wcorr"] = st["osc"] + snf + noneq
        st["dep_noleak"] = st["osc_wcorr"].GetWithPositronEnergy(
            s_ibd=self.s_ibd, inputfile=args.input_data_file, tf2name=args.pos_ene_TF2)
        st["dep"] = st["dep_noleak"].ApplyDetResp(self.ene_leak, pecrop=args.ene_crop)
        st["vis"] = self._stretch_energy(st["dep"], 1.0 + escale_tao)
        rm = self._get_resp_matrix(abc_tao[0], abc_tao[1], abc_tao[2],
                                   escale=1.0 + escale_tao)
        st["rec"] = st["dep"].ApplyDetResp(rm, pecrop=args.ene_crop)

        s_rea, s_tot, _, _, _, bckg = self.build_spectrum(
            Nrea_tao=Nrea_tao, abc_tao=abc_tao, snf_noneq_norm_tao=snf_noneq_norm_tao,
            core_power_tao=core_power_tao, eff_nuis_tao=eff_nuis_tao,
            bckg_rate=bckg_rate, nuisance=nuisance, NL_tao=NL_tao,
            escale_tao=escale_tao)
        st["reactor"], st["bkg"], st["totalpred"] = s_rea, bckg, s_tot

        params = dict(
            Efficiency=args.detector_efficiency * (1 + eff_nuis_tao),
            LY=1.0 + escale_tao, a=abc_tao[0], b=abc_tao[1], c=abc_tao[2],
            power_GW=float(np.sum(self.core_powers_std)),
            baseline_km=float(self.core_baselines[0]) * 1e-3,
            R_flux=1.0 + Nrea_tao, core_norm=1.0 + core_power_tao)
        params["fission_fractions"] = [args.U235_scale, args.U238_scale,
                                       args.Pu239_scale, args.Pu241_scale]
        return st, params

    # ════════════════════════════════════════════════════════════════════════
    #  chi2 contribution
    # ════════════════════════════════════════════════════════════════════════

    def chi2_contribution(self,
                          Nrea_tao, abc_tao, snf_noneq_norm_tao,
                          core_power_tao, eff_nuis_tao,bckg_rate,
                          unc, nuisance, NL_tao=None, escale_tao=0.0):
        """
        Returns (chi2_tao, chi2_nolog, s_tao, s_tot_tao).
        """
        args = self.args

        (s_tao, s_tot_tao, pulls, pull_unc,
         rfis_param_post, bckg) = self.build_spectrum(
            Nrea_tao, abc_tao, snf_noneq_norm_tao,
            core_power_tao, eff_nuis_tao,bckg_rate, nuisance, NL_tao=NL_tao,
            escale_tao=escale_tao)

        sp_obs = self._get_obs_spectrum()

        if unc != 'stat':
            cm_t  = CovMatrix(
            data=self.cm['all'].data * np.outer(s_tao.bin_cont, s_tao.bin_cont),
            bins=self.cm['all'].bins)
            # Background bin-to-bin (b2b) covariance: per-bin relative
            # uncertainty sigma_i = sqrt(N_i)/N_i, giving a diagonal
            # var_i = (sigma_i * N_i)^2 = N_i.
            bckg_cont = np.asarray(bckg.bin_cont, dtype=float)
            b2b_rel = np.divide(np.sqrt(bckg_cont), bckg_cont,
                                out=np.zeros_like(bckg_cont), where=bckg_cont > 0)
            cm_t.data += np.diag((b2b_rel * bckg_cont) ** 2)
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
