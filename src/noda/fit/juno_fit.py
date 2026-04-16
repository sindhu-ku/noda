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
from ..energy_response_cov import covariance_er
from .. import nuosc as nuosc


class JUNODetector:
    """
    Builds the JUNO predicted spectrum and returns a chi2 contribution.

    Parameters passed at construction time are things that never change
    between Minuit calls (spectra templates, response matrices, config).
    Parameters passed at call time are the floating nuisances.
    """

    def __init__(self, ndays, ensp_nom, rm, cm, args, backgrounds,
                 core_baselines, core_powers_std, core_fractions,
                 E_segments, s_ibd, s_bump_lin):
        self.ndays          = ndays
        self.ensp_nom       = ensp_nom
        self.rm             = rm
        self.cm             = cm
        self.args           = args
        self.backgrounds    = backgrounds
        self.core_baselines = core_baselines
        self.core_powers_std = core_powers_std
        self.core_fractions = core_fractions
        self.fine_bins      = ensp_nom['ribd'].bins
        self.E_segments     = E_segments
        self.s_ibd          = s_ibd
        self.s_bump_lin     = s_bump_lin

        # ── pre-build fission arrays (expensive, do once) ──────────────────
        args_juno = args
        n_cores = len(core_baselines)
        if args_juno.CI_powers:
            dummy_sp = ensp_nom["rfis0"]
        else:
            dummy_sp = ensp_nom["rfis0_core0"]
        self.dummy_sp = dummy_sp

        fission_fracs = []
        for frac in core_fractions:
            fission_fracs.append([frac['U235'], frac['U238'],
                                   frac['Pu239'], frac['Pu241']])
        self.fission_fracs = fission_fracs

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
        self._osc_cache            = {}
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

    # ════════════════════════════════════════════════════════════════════════
    #  Nuisance parameter interface (consumed by combined_fit.py)
    # ════════════════════════════════════════════════════════════════════════

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
        core_names     = [f"core_power{i}" for i in range(10)]

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
        core_start     = list(np.zeros(10))

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

        for i in range(10):
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
            + [f"core_power{i}" for i in range(10)]
            + [f"efission{i}" for i in range(len(args.efission))]
        )

    # ════════════════════════════════════════════════════════════════════════
    #  Internal cached helpers  (identical logic to original run_minuit)
    # ════════════════════════════════════════════════════════════════════════

    def _get_resp_matrix(self, a, b, c):
        key = (a, b, c)
        if key not in self._resp_matrix_cache:
            self._resp_matrix_cache[key] = CalcRespMatrix_abc(
                a, b, c, escale=1, ebins=self.fine_bins, pebins=self.fine_bins)
        return self._resp_matrix_cache[key]

    def _get_nuisance_fine(self, nuisance_coarse):
        key = tuple(nuisance_coarse)
        if key not in self._nuisance_fine_cache:
            self._nuisance_fine_cache[key] = expand_nuisances_to_fine(
                self.E_segments, self.fine_bins, nuisance_coarse)
        return self._nuisance_fine_cache[key]

    def _get_chi2_penalty_er(self, a, b, c, kB, fC, kI, LY_shared):
        key = (a, b, c, kB, fC, kI, LY_shared)
        if key not in self._chi2_penalty_er_cache:
            theta = np.array([a, b, c, kB, fC, kI, LY_shared])
            diff  = theta - self.theta0_er
            inv_cov = np.linalg.inv(covariance_er)
            self._chi2_penalty_er_cache[key] = diff @ inv_cov @ diff
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
            U  = U.GetWithPositronEnergy(type=args.posE_shift,
                                         inputfile=args.input_data_file,
                                         tf2name=args.pos_ene_TF2)
            Th = Th.GetWithPositronEnergy(type=args.posE_shift,
                                          inputfile=args.input_data_file,
                                          tf2name=args.pos_ene_TF2)
            U  = U.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            Th = Th.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            U.ApplyDetResp(self.rm, pecrop=args.ene_crop)
            Th.ApplyDetResp(self.rm, pecrop=args.ene_crop)
            U.Rebin_nonuniform(args.bins_nonuniform)
            Th.Rebin_nonuniform(args.bins_nonuniform)
            self._geo_spectrum_cache[key] = Th + U
        return self._geo_spectrum_cache[key]

    def _get_wr_spectrum(self, resp_matrix, a, b, c, kB, fC, kI, LY_shared):
        key = (a, b, c, kB, fC, kI, LY_shared)
        if key not in self._wr_spectrum_cache:
            args  = self.args
            ensp  = self.ensp_nom
            ndays = self.ndays
            bckg  = ensp['rfis_wr_pos'].GetWithModifiedEnergy(
                mode='G4', kB=kB, fC=fC, kI=kI, LY_shared=LY_shared)
            bckg.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
            rate  = self.backgrounds['JUNO']['wr']['rate']
            scale = rate * ndays / (args.duty_cycle * bckg.GetIntegral())
            bckg  = bckg.GetScaledFit(scale)
            bckg.Rebin_nonuniform(args.bins_nonuniform)
            self._wr_spectrum_cache[key] = bckg
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
        return self._snf_noneq_cache[key]

    # ════════════════════════════════════════════════════════════════════════
    #  Main spectrum builder
    # ════════════════════════════════════════════════════════════════════════

    def build_spectrum(self, sin2_12, sin2_13, dm2_21, dm2_31,me_rho,
                       Nrea, NL, abc, bckgs_for_chi2, snf_noneq_norm,
                       eff_nuis, ThU_ratio,
                       core_powers_delta, efission,nuisance,
                       opp=False):
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
            pulls.append(dm2_31 - nuosc.op_nom['dm2_31'])
            pull_unc.append(nuosc.op_nom['dm2_31_err'])
        if args.pull_dm2_31_lbl:
            from .minuit import f_lbl  # keep f_lbl where it is
            ordering    = "NO" if dm2_31 > 0 else "IO"
            dm2_31_lbl  = f_lbl(args.dm2_31_lbl, sin2_12, sin2_13,
                                 dm2_21, args.theta23, args.dcp, ordering)
            pulls.append(dm2_31 - dm2_31_lbl)
            pull_unc.append(args.dm2_31_lbl_err)

        # ── core powers ─────────────────────────────────────────────────────
        if len(core_powers_delta):
            core_powers_new = list(
                np.array(self.core_powers_std) * (1 + np.array(core_powers_delta)))
            pulls.extend(core_powers_delta)
            pull_unc.extend([args.core_flux_unc] * 10)
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
        core_scales = np.zeros(self.n_cores)
        for i in range(self.n_cores):
            core_scales[i] = compute_extrafactors(
                args,
                efission=efission_scaled,
                alpha=self.fission_fracs[i],
                core_baselines=[self.core_baselines[i]],
                core_powers=[core_powers_new[i]]
            )
        raw_ibd_tot    = np.sum(self.fission_arrays * core_scales[:, np.newaxis], axis=0)
        raw_ibd_tot    = raw_ibd_tot * nuisance_multiplier
        rfis_param_post = Spectrum(bins=self.dummy_sp.bins, bin_cont=raw_ibd_tot)

        # ── matter density ───────────────────────────────────────────────────
        if me_rho != 0.0:
            pulls.append(me_rho - args.me_rho)
            pull_unc.append(args.me_rho_scale)
            rho = me_rho
        else:
            rho = args.me_rho

        # ── oscillation ──────────────────────────────────────────────────────
        s_juno = rfis_param_post.GetOscillated(
                L=list(self.core_baselines),
                sin2_th12=sin2_12, sin2_th13=sin2_13,
                dm2_21=dm2_21, dm2_31=dm2_31,
                core_powers=list(core_powers_new),
                me_rho=rho, ene_mode='true', opp=opp, args=self.args)

        # ── positron energy shift ────────────────────────────────────────────
        s_juno = s_juno.GetWithPositronEnergy(inputfile=args.input_data_file,
                                               tf2name=args.pos_ene_TF2)

        # ── non-linearity ────────────────────────────────────────────────────
        if NL:
            if args.NL == "pull":
                new_nonl = Spectrum(bins=ensp['scintNL'].bins,
                                    bin_cont=np.zeros(len(ensp['scintNL'].bin_cont)))
                new_nonl.bin_cont = (ensp['scintNL'].bin_cont
                    + NL[0] * (ensp['NL_pull'][0].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[1] * (ensp['NL_pull'][1].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[2] * (ensp['NL_pull'][2].bin_cont - ensp['scintNL'].bin_cont)
                    + NL[3] * (ensp['NL_pull'][3].bin_cont - ensp['scintNL'].bin_cont))
                s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum',
                                                      spectrum=new_nonl, LY_shared=NL[4])
                pulls.extend([NL[0], NL[1], NL[2], NL[3], NL[4]])
                pull_unc.extend([args.NL_sd] * 4)
                pull_unc.append(0.005)
            else:
                kB = NL[0]; fC = NL[1]; kI = NL[2]; LY_shared = NL[3]
                s_juno = s_juno.GetWithModifiedEnergy(mode='G4', kB=kB, fC=fC,
                                                      kI=kI, LY_shared=LY_shared)
                chi2_penalty_er = self._get_chi2_penalty_er(
                    abc[0], abc[1], abc[2], kB, fC, kI, LY_shared)
        else:
            if args.NL == "pull":
                s_juno = s_juno.GetWithModifiedEnergy(mode='spectrum',
                                                      spectrum=ensp['scintNL'])
            else:
                s_juno = s_juno.GetWithModifiedEnergy(mode='G4',
                    kB=args.kB, fC=args.fC, kI=args.kI, LY_shared=args.LY_shared)
            new_nonl = None  # not used below in the NL=="pull" geo branch

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

        # ── SNF + non-equilibrium ────────────────────────────────────────────
        if snf_noneq_norm:
            pulls.extend(snf_noneq_norm)
            pull_unc.extend([args.snf_unc] * 3)
            pull_unc.extend([args.noneq_unc] * 3)
            s_juno = s_juno + self._get_snf_noneq(snf_noneq_norm)
        else:
            s_juno = s_juno + ensp['snf_final'] + ensp['noneq_final']

        # ── efficiency ───────────────────────────────────────────────────────
        if eff_nuis:
            pulls.append(eff_nuis)
            pull_unc.append(args.eff_unc)
            s_juno.GetScaled(1 + eff_nuis)

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
                    rate  = float(args.geo_tnu_rate) * float(args.tnu_to_cpd)
                    scale = rate * ndays / args.duty_cycle
                    Th = ensp['rfis0_geoth'].GetScaledFit(
                        ThU_ratio * scale / ((1 + ThU_ratio) * ensp['rfis0_geoth'].GetIntegral()))
                    U  = ensp['rfis0_geou'].GetScaledFit(
                        scale / ((1 + ThU_ratio) * ensp['rfis0_geou'].GetIntegral()))
                    U_pos  = U.GetWithPositronEnergy(type=args.posE_shift,
                                                     inputfile=args.input_data_file,
                                                     tf2name=args.pos_ene_TF2)
                    Th_pos = Th.GetWithPositronEnergy(type=args.posE_shift,
                                                      inputfile=args.input_data_file,
                                                      tf2name=args.pos_ene_TF2)
                    U_nonl  = U_pos.GetWithModifiedEnergy(mode='spectrum',
                                                          spectrum=new_nonl, LY_shared=NL[4])
                    Th_nonl = Th_pos.GetWithModifiedEnergy(mode='spectrum',
                                                           spectrum=new_nonl, LY_shared=NL[4])
                    U_final  = U_nonl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
                    Th_final = Th_nonl.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
                    if args.rebin_nonuniform:
                        U_final.Rebin_nonuniform(args.bins_nonuniform)
                        Th_final.Rebin_nonuniform(args.bins_nonuniform)
                    bckg = U_final + Th_final
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
                    bckg.ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
                    rate  = self.backgrounds['JUNO']['wr']['rate']
                    scale = rate * ndays / (args.duty_cycle * bckg.GetIntegral())
                    bckg  = bckg.GetScaledFit(scale)
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

            if info["b2b"] > 0:
                b2bunc = info["b2b"]
                if cm_bckg is None:
                    cm_bckg = bckg.GetB2BCovMatrix(b2bunc, 0.1)
                else:
                    cm_bckg += bckg.GetB2BCovMatrix(b2bunc, 0.1)

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
                          unc, opp=False):
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
            core_powers_delta, efission, nuisance, opp=opp)

        sp_obs = self._get_obs_spectrum()

        if unc != 'stat':
            cm_j = self.cm[unc]
        else:
            ebins = self.ensp_nom['rdet'].bins
            nbins = len(ebins)
            cm_j  = CovMatrix(data=np.zeros((nbins - 1, nbins - 1)), bins=ebins)

        cm_j = cm_j + cm_bckg

        chi2_nolog, chi2_juno = Chi2(
            cm_j, sp_obs, s_tot_juno, sp_obs, s_tot_juno,
            'bckg',
            args.stat_method_opt,
            pulls=pulls, pull_unc=pull_unc, include_logV=False)

        return (chi2_juno, chi2_nolog, chi2_penalty_er,
                s_juno, s_tot_juno, fit_bckgs_spectra, rfis_param_post)

    def _get_obs_spectrum(self):
        args = self.args
        ensp = self.ensp_nom
        if bool(args.toymc):
            return ensp["rtot_toy"]
        elif args.is_data:
            return ensp['data']
        else:
            return ensp["rtot"]
