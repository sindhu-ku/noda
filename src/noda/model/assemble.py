"""CreateSpectra: builds the nominal (prefit) prediction of one detector -
signal + backgrounds -> total - plus prefit plots, event summary and
the prefit ROOT output.

Split out of model/spectra.py (pure move)."""
import os
import sys
import csv

import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import simpson, trapezoid, quad
from scipy import constants
from joblib import Parallel, delayed
from datetime import datetime

from ..noda import *
from .detection import _apply_nonlinearity, _apply_oec_correction, _apply_positron_energy, _build_energy_leak, _ensure_dir, _load_nonlinearity, _load_or_build_resp_matrix
from .reactor import _build_per_core_steps, _build_reactor_flux_per_core, _build_reactor_flux_simple, _build_snf_noneq_corrections_per_core, _build_snf_noneq_simple, _write_groupC_csvs
from .backgrounds import _build_geo_spectra, _load_juno_backgrounds
from . import backgrounds as bkgdef


# ---------------------------------------------------------------------------
# Detector-specific total spectrum assembly
# ---------------------------------------------------------------------------

def _assemble_total_spectrum_dyb(ensp, args, ndays):
    extrafactors = compute_extrafactors(
        args, alpha=[args.U235_scale, args.U238_scale,
                     args.Pu239_scale, args.Pu241_scale])
    ensp["rtot"] = ensp["rdet"]

    if args.is_data:
        if args.model == "CPC":
            names = ["Bin_Center", "Sigma", "Rel_Unc"]
        else:
            names = ["Bin_lowE", "Bin_highE", "Sigma", "bin_err"]
        dyb = pd.read_csv(
            args.dyb_spectrum, sep='\s+',
            names=names,
        )

        ensp["data"] = Spectrum(bins=ensp["rdet"].bins, bin_cont=dyb["Sigma"].to_numpy())
        if args.model == "PRL":
            ensp["data"].GetScaled(1e-43)

        ensp["data"].GetScaled(extrafactors)
        ensp["data"].GetScaled(ndays)
        print("DYB data scaling 1e-43: ", extrafactors*ndays)
        ensp["data"].ScaleByBinWidth()


        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

        return (
                [ensp["data"]],
                ["DYB expectation", "DYB data"],
                ["brown", "steelblue"],
            )

    if args.rebin_nonuniform:
        ensp["rtot"].Rebin_nonuniform(args.bins_nonuniform)


    return (
        [ensp["rtot"]],
        ["DYB expectation", "DYB data"],
        ["brown", "steelblue"],
    )


def _assemble_total_spectrum_tao(ensp, args, ndays):
    ensp["bckg"] = GetSpectrumFromROOT(args.data_file, args.bckg_hist_name)
    ensp["bckg"].ScaleByBinWidth()
    ensp["bckg"].GetScaled(ndays*args.bckg_rate/ensp['bckg'].GetIntegral())
    # reactor-on/off DAQ livetime ratio (Group A convention); 1.0 if absent
    ensp["bckg"].GetScaled(getattr(args, "bckg_scale", 1.0))
    print("TAO background: ", ensp['bckg'].GetIntegral())
    if args.rebin_nonuniform:
        ensp["bckg"].Rebin_nonuniform(args.bins_nonuniform)

    ensp["rtot"] = ensp["rdet"] + ensp["bckg"]

    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        ensp["data"].ScaleByBinWidth()
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)


        return  ([ensp['data'], ensp['bckg'], ensp['rtot']],
                ['TAO reactor expectation', 'TAO data', 'TAO background', 'TAO reactor+bckg expectation'],
                ['brown', 'steelblue', 'orange', 'black'])
    return  ([ensp['rtot'], ensp['bckg']],
                ['TAO expectation', 'TAO data', 'TAO background'],
                ['brown', 'steelblue', 'orange'])

def _assemble_total_spectrum_juno(ensp, args, backgrounds):
    """Event summary + (spectra, labels, colors) for the JUNO prefit plot.
    Loops over the backgrounds listed in the yaml (any keys)."""
    bkgs = backgrounds["JUNO"]
    events = {
        "rtot":   ensp["rtot"].GetIntegral(),
        "rdet":   ensp["rdet"].GetIntegral(),
        "rbckg":  ensp["bckg"].GetIntegral() if ensp.get("bckg") is not None else 0.0,
        "ribd":   ensp["ribd"].GetIntegral(),
        "rosc":   ensp["rosc"].GetIntegral(),
        "bkgs":   [(key, ensp[key].GetIntegral()) for key in bkgs],
    }
    _print_juno_event_summary(events)

    extra = [ensp[k] for k in bkgs]
    labels = ["Reactor antineutrinos"] + [bkgdef.display_name(k, i) for k, i in bkgs.items()]
    palette = ["darkred", "green", "navy", "orange", "magenta", "lightblue",
               "yellow", "lightgreen", "purple", "grey", "teal", "olive",
               "pink", "brown", "cyan"]
    colors = ["black"] + [palette[n % len(palette)] for n in range(len(bkgs))]

    if args.is_data:
        ensp["data"] = GetSpectrumFromROOT(args.data_spectrum, args.data_hist_name)
        if args.rebin_nonuniform:
            ensp["data"].Rebin_nonuniform(args.bins_nonuniform)

        extra.append(ensp['data'])
        labels.append('Data')
        colors.append('blue')
    return extra, labels, colors


def _print_juno_event_summary(ev):
    print("\n" + "=" * 46)
    print(f"  Reac × IBD:          {ev['ribd']:>10.2f} events")
    print(f"  Oscillated:          {ev['rosc']:>10.2f} events")
    print(f"  Detected:            {ev['rdet']:>10.2f} events")
    print(f"  IBD + BG:            {ev['rtot']:>10.2f} events")
    print(f"  Backgrounds (total): {ev['rbckg']:>10.2f} events")
    for key, n in ev["bkgs"]:
        print(f"    {key + ':':<18} {n:>10.2f}")
    print("=" * 46 + "\n")


# ---------------------------------------------------------------------------
# Optional write-out
# ---------------------------------------------------------------------------

def _build_nl_graph(ensp, args, kB=None, fC=None, kI=None, LY_shared=None,
                    NL_pulls=None):
    """Positron NL curve (Evis/Edep vs Edep), exported as
    <det>/response/<stage>/nl_positron: Geant4 model anchored to the
    2.224589 MeV gamma.  Parameters default to the pre-fit nominals.

    In pull mode, *NL_pulls* = [p0, …, p_{k-1}, LY_pull] gives the post-fit
    curve: nominal + Σ pᵢ·(pullᵢ − nominal), scaled by (1 + LY_pull) — the
    same construction as build_spectrum().  The number of shape pulls k is
    inferred from len(NL_pulls)−1, so this handles both JUNO (4 shape + LY)
    and TAO (5 shape + LY); the last element is always the LY pull."""
    if args.NL == "pull":
        e = 0.5 * (ensp["scintNL"].bins[:-1] + ensp["scintNL"].bins[1:])
        nom = ensp["scintNL"].bin_cont
        if NL_pulls is not None:
            n_shape = len(NL_pulls) - 1          # last element is the LY pull
            ratio = (nom + sum(NL_pulls[i] * (ensp["NL_pull"][i].bin_cont - nom)
                               for i in range(n_shape))) * (1 + NL_pulls[-1])
        else:
            ratio = nom
    else:
        kB = args.kB if kB is None else kB
        fC = args.fC if fC is None else fC
        kI = args.kI if kI is None else kI
        LY_shared = args.LY_shared if LY_shared is None else LY_shared
        # 1.022 MeV = minimum deposited energy of an IBD positron (physics, not a range choice)
        e = np.arange(1.022, analysis_range(args)[1], 0.001)
        ratio = (G4model.eval(e, kB, fC=fC, kI=kI) * LY_shared
                 / G4model_gamma.eval(2.224589, kB, fC=fC, kI=kI))
    graph = ROOT.TGraph(len(e),
                        np.ascontiguousarray(e, dtype=float),
                        np.ascontiguousarray(ratio, dtype=float))
    graph.SetName("nl_positron")
    return graph


def analysis_range(args):
    """(E_lo, E_hi) of the analysis in MeV, from the yaml: first/last edge of
    bins_nonuniform when rebin_nonuniform is set, else min_ene/max_ene."""
    bn = getattr(args, 'bins_nonuniform', None)
    if getattr(args, 'rebin_nonuniform', False) and bn:
        return float(bn[0][0]), float(bn[-1][1])
    return float(args.min_ene), float(args.max_ene)


def _build_resolution_graph(a, b, c, emin=0.8, emax=12.0):
    """Relative energy resolution σ_E/E vs Evis, same parametrisation as
    CalcRespMatrix_abc: sqrt(a²/E + b² + c²/E²)."""
    e = np.arange(max(emin, 0.1), emax, 0.001)   # sigma/E diverges at E -> 0
    res = np.sqrt(a**2 / e + b**2 + c**2 / e**2)
    graph = ROOT.TGraph(len(e),
                        np.ascontiguousarray(e, dtype=float),
                        np.ascontiguousarray(res, dtype=float))
    graph.SetName("resolution")
    return graph


# Map NODA shell labels to the top-level JUNO directory name in the output
# ROOT file. Output names and layout follow the ORSA export so the two
# fitters' files compare directly. Extend this when new shells are added.
_SHELL_DIR = {'S-15': 'juno_inner', 'S-15-172': 'juno_outer'}


def juno_output_dirname(args):
    """Top-level JUNO ROOT directory for this dataset.

    A single-shell run keeps the legacy bare ``juno``.  A multi-shell run maps
    each shell to its own self-contained top-level folder
    (``juno_inner`` / ``juno_outer``), falling back to
    ``juno_<shell>`` for shell labels not in the map.  The multi-shell signal
    is the per-dataset ``root_suffix`` carrying a shell component (``__s``)."""
    sfx = getattr(args, 'root_suffix', '')
    if '__s' not in sfx:
        return 'juno'
    shell = str(getattr(args, 'shell', ''))
    return _SHELL_DIR.get(shell, f"juno_{shell}")


def _write_juno_output(ensp, ndays, args, backgrounds, s_ibd=None):
    from ..fit.minuit import write_param_dbl
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    jdir = juno_output_dirname(args)            # 'juno' | 'juno_inner' | 'juno_outer'
    output_file = ROOT.TFile(tag, "update")

    # Each shell writes its OWN top-level juno_<shell> directory
    # with bare object names. get-or-create: in "update" mode the dir may
    # already exist from a previous write, and TDirectory.mkdir() returns null
    # for an existing directory (which would crash on .cd()).
    juno_dir = output_file.GetDirectory(jdir) or output_file.mkdir(jdir)
    juno_dir.cd()
    write_param_dbl("LivetimeDays", ndays)

    dir_pre = juno_dir.GetDirectory("prefit") or juno_dir.mkdir("prefit")
    dir_pre.cd()
    write_param_dbl("Efficiency", args.detector_efficiency)
    if s_ibd is not None:
        enu = np.arange(0.0, max(12.0, float(args.max_ene)) + 0.001, 0.001)
        xsec_vals = s_ibd(enu)
        xsec_graph = ROOT.TGraph(len(enu),
                                 np.ascontiguousarray(enu, dtype=float),
                                 np.ascontiguousarray(xsec_vals, dtype=float))
        xsec_graph.Write("xsec", ROOT.TObject.kOverwrite)

    dir_resp = juno_dir.GetDirectory("response") or juno_dir.mkdir("response")
    dir_resp.cd()
    dir_resp_prefit = dir_resp.GetDirectory("prefit") or dir_resp.mkdir('prefit')
    dir_resp_prefit.cd()
    g_nl = _build_nl_graph(ensp, args);  g_nl.SetName("nl_positron")
    g_nl.Write("", ROOT.TObject.kOverwrite)
    g_res = _build_resolution_graph(args.a, args.b, args.c, *analysis_range(args))
    g_res.SetName("resolution");   g_res.Write("", ROOT.TObject.kOverwrite)
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["ribd"].WritetoROOT("unosc", tag, f"{jdir}/prefit")
    ensp["rosc"].WritetoROOT("osc", tag, f"{jdir}/prefit")
    if "rosc_wcorr" in ensp:
        ensp["rosc_wcorr"].WritetoROOT("osc_wcorr", tag, f"{jdir}/prefit")
    ensp["rvis_nonl_temp"].WritetoROOT("unosc_dep", tag, f"{jdir}/prefit")
    ensp["rvis_nonl"].WritetoROOT("dep", tag, f"{jdir}/prefit")
    ensp["rvis"].WritetoROOT("vis", tag, f"{jdir}/prefit")
    ensp["rdet_fine"].WritetoROOT("rec", tag, f"{jdir}/prefit")
    ensp["rdet"].WritetoROOT("reactor", tag, f"{jdir}/prefit")
    for core, steps in ensp.get("per_core_steps", {}).items():
        for step, spec in steps.items():
            spec.WritetoROOT(step, tag, f"{jdir}/prefit/per_core/{core}")
    ensp["rtot"].WritetoROOT("totalpred", tag, f"{jdir}/prefit")
    for key in backgrounds["JUNO"]:
        ensp[key].WritetoROOT(f"{key}", tag, f"{jdir}/prefit")
    if args.is_data: ensp["data"].WritetoROOT("data", tag, jdir)



def _write_params(tag, directory, params):
    """Write TParameter<double>s into <directory> of the output ROOT file."""
    from ..fit.minuit import write_param_dbl
    f = ROOT.TFile(tag, "update")
    d = f.GetDirectory(directory) or f.mkdir(directory)
    d.cd()
    for k, v in params.items():
        write_param_dbl(k, float(v))
    force_vector_streamers(f)
    f.Close()


def _write_dyb_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    if not output_file.GetDirectory("dyb"):
        output_file.mkdir("dyb")
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["rdet"].WritetoROOT("totalpred", tag, "dyb/prefit")
    ensp["rdet"].WritetoROOT("reactor", tag, "dyb/prefit")   # reactor only (no bkg)
    _write_params(tag, "dyb/prefit", {"Efficiency": 1.0})
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "dyb")


def _write_tao_output(ensp, args, backgrounds):
    tag = f"Postfit_{args.file_prefix}_NO_{args.NMO_opt}_dm2_31_pull_{args.pull_dm2_31}_{args.fit_mode}.root"
    output_file = ROOT.TFile(tag, "update")

    tao_dir = output_file.GetDirectory("tao") or output_file.mkdir("tao")
    dir_resp = tao_dir.GetDirectory("response") or tao_dir.mkdir("response")
    dir_resp_prefit = dir_resp.GetDirectory("prefit") or dir_resp.mkdir("prefit")
    dir_resp_prefit.cd()
    # NL baked into the leakage matrix (nl_in_leak): no separate NL curve exists,
    # so skip the pre-fit NL graph and write only the resolution curve.
    if not getattr(args, "nl_in_leak", False):
        _build_nl_graph(ensp, args).Write("", ROOT.TObject.kOverwrite)
    _build_resolution_graph(args.a, args.b, args.c,
                            *analysis_range(args)).Write("", ROOT.TObject.kOverwrite)
    force_vector_streamers(output_file)
    output_file.Close()

    ensp["rtot"].WritetoROOT("totalpred", tag, "tao/prefit")
    ensp["bckg"].WritetoROOT("bkg", tag, "tao/prefit")       # (was "bckg")
    ensp["rdet"].WritetoROOT("reactor", tag, "tao/prefit")
    _write_params(tag, "tao/prefit", {"Efficiency": args.detector_efficiency})
    _write_params(tag, "tao", {"LivetimeDays": float(str(args.stat_opt).replace("days", ""))})
    if args.is_data: ensp["data"].WritetoROOT("data", tag, "tao")
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def CreateSpectra(ndays=10, ebins=None, detector="JUNO", s_ibd=None, s_bump_lin=None, backgrounds=None, args=None, dm2_31=None):
    """
    Build all reactor antineutrino spectra for *detector* over *ndays* days.

    Parameters
    ----------
    ndays      : float  – live days
    ebins      : array  – bin edges in MeV
    detector   : str    – "JUNO", "TAO", or "DYB"
    backgrounds: dict   – background config (used for JUNO)
    args       : namespace – all configuration parameters

    Returns
    -------
    (ensp, resp_matrix)                 for JUNO / DYB
    (ensp, resp_matrix, ene_leak_nd)    for TAO
    """
    _ensure_dir(args.plots_folder)

    ensp  = {}
    binw  = (ebins[1] - ebins[0]) * 1000   # keV

    print(f"\n{'='*50}")
    print(f"  Building spectra  |  detector={detector}  ndays={ndays}")
    print(f"{'='*50}\n")


    resp_matrix = _load_or_build_resp_matrix(args, detector, ebins)
    # Keep ene_leak_nd for TAO return value
    ene_leak_nd=None
    if detector == "TAO":
        ene_leak_nd = _build_energy_leak(args, detector, ebins)

    _load_nonlinearity(ensp, args, detector)

    # ------------------------------------------------------------------
    # 2. Reactor flux  (per-core JUNO path vs. simple single-flux path)
    # ------------------------------------------------------------------
    print("# Building reactor flux …")

    if detector == "JUNO": # and not args.CI_powers:
        print("Building spectrum with unblinded powers")
        core_names, core_powers, core_fractions, core_baselines = load_reactor_summary(args.reactor_summary_yaml)
        _build_reactor_flux_per_core(
            ensp, args, ebins, s_ibd, s_bump_lin, ndays,
            core_names, core_powers, core_fractions, core_baselines,
            dm2_31
        )
        _build_snf_noneq_corrections_per_core(
            ensp, args,ndays, ebins, core_fractions, core_baselines, core_powers, resp_matrix,s_ibd, dm2_31,
            core_names=core_names,
        )
        if args.write_output: _write_groupC_csvs(ensp, args, core_names, core_powers, core_fractions)
    else:
        print("Building spectrum with powers from config")
        core_baselines = args.core_baselines
        core_powers    = args.core_powers
        _build_reactor_flux_simple(ensp, args, ebins, s_ibd, s_bump_lin, ndays)
        _build_snf_noneq_simple(ensp, args, ebins, detector, resp_matrix, s_ibd, ene_leak=ene_leak_nd)

    print(f"  Expected IBDs (no osc):  {ensp['rfis'].GetIntegral():.2f}")

    # ------------------------------------------------------------------
    # 3. SNF + Non-equilibrium corrections
    # ------------------------------------------------------------------
    print("# Building SNF / non-eq corrections …")

    #if detector == "JUNO" and not args.CI_powers:

    # else:
    #
    # ------------------------------------------------------------------
    # 4. Oscillated spectrum
    # ------------------------------------------------------------------
    ensp["ribd"] = ensp["rfis"].Copy()
    if detector == "JUNO":
        if getattr(args, "osc_per_core", False):
            ensp["rosc"] = ensp["rosc_pcore"]
        else:
            ensp["rosc"] = ensp["ribd"].GetOscillated(
                L=list(core_baselines), core_powers=list(core_powers),
                me_rho=args.me_rho, ene_mode="true", dm2_31=dm2_31, args=args,
            )
    else:
        ensp["rosc"] = ensp["ribd"]

    # Oscillated spectrum with SNF + Non-Eq corrections added (true-energy level).
    # snf_osc / noneq_osc are only built in the per-core (Group C) path.
    if "snf_osc" in ensp and "noneq_osc" in ensp:
        ensp["rosc_wcorr"] = ensp["rosc"] + ensp["snf_osc"] + ensp["noneq_osc"]

    ensp["rosc"].Plot(
        f"{args.plots_folder}/osc_spectrum_{detector}.png",
        xlabel="Neutrino energy (MeV)",
        ylabel=f"Events per {binw:.1f} keV",
        extra_spectra=[ensp["ribd"]],
        leg_labels=["oscillated", "unoscillated"],
        colors=["black", "darkred"],
        xmin=0, xmax=10, ymin=0, ymax=None, log_scale=False,
    )

    # ------------------------------------------------------------------
    # 5. Energy-response chain for the main reactor spectrum
    # ------------------------------------------------------------------
    print("# Applying energy-response chain …")

    if detector == "DYB" and args.model == "PRL":
        ensp["rdet"] = ensp["rosc"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)

    else:
        ensp["rvis_nonl"]      = _apply_positron_energy(ensp["rosc"], args,s_ibd=s_ibd)
        ensp["rvis_nonl_temp"] = _apply_positron_energy(ensp["ribd"], args, s_ibd=s_ibd)
        #ensp["rvis_nonl_temp_wcorr"] = _apply_positron_energy((ensp["rosc"]+ensp['snf_osc']+ensp['noneq_osc']), args, s_ibd=s_ibd)
        if detector == "TAO":
            ensp["rvis_nonl_leak"] = ensp["rvis_nonl"].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)
            ensp["rvis_nonl_temp_leak"] = ensp["rvis_nonl_temp"].ApplyDetResp(ene_leak_nd, pecrop=args.ene_crop)

        else:
            ensp["rvis_nonl_leak"] = ensp["rvis_nonl"]
            ensp["rvis_nonl_temp_leak"] = ensp["rvis_nonl_temp"]

        tao_nl_in_leak = (detector == "TAO" and getattr(args, "nl_in_leak", False))
        if detector != "DYB" and not tao_nl_in_leak:
            ensp["rvis"]      = _apply_nonlinearity(ensp["rvis_nonl_leak"],      args, ensp)
            ensp["rvis_temp"] = _apply_nonlinearity(ensp["rvis_nonl_temp_leak"], args, ensp)
            #ensp["rvis_temp_wcorr"] = _apply_nonlinearity(ensp["rvis_nonl_temp_wcorr"], args, ensp)
        elif tao_nl_in_leak:
            # NL already carried by the energy-leakage matrix — no separate curve.
            ensp["rvis"]      = ensp["rvis_nonl_leak"]
            ensp["rvis_temp"] = ensp["rvis_nonl_temp_leak"]
        else:
            ensp["rvis"]      = ensp["rvis_nonl"]
            ensp["rvis_temp"] = ensp["rvis_nonl_temp"]
            #ensp["rvis_temp_wcorr"] = ensp["rvis_nonl_temp_wcorr"]

        ensp["rdet"] = ensp["rvis"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        ensp["rdet_temp"] = ensp["rvis_temp"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)
        #ensp["rdet_temp_wcorr"] = ensp["rvis_temp_wcorr"].ApplyDetResp(resp_matrix, pecrop=args.ene_crop)


    # Add corrections
    if detector == "DYB":
        if args.model == "CPC":
            ensp["rdet"] = ensp["rdet"] + ensp["noneq_final"]
        else:
            ensp["rdet"] = ensp["rdet"]
    else:
        ensp["rdet"] = ensp["rdet"] + ensp["snf_final"] + ensp["noneq_final"]
    # OMILREC 0.6 MeV threshold correction on reactor IBDs (Check 5-2).
    # Applied on the fine reco spectrum, before rebinning.
    if detector == "JUNO":
        _apply_oec_correction(ensp["rdet"], args, label="reactor")
    ensp['rdet_fine'] = ensp['rdet']
    if args.rebin_nonuniform:
        ensp["rdet"].Rebin_nonuniform(args.bins_nonuniform)

    # Per-core step-by-step reactor spectra (JUNO, per-core reactor path) for
    # bin-by-bin comparisons. Written by _write_juno_output under
    # <shell>/prefit/per_core/<core>/. Toggle with fit_options write_per_core.
    if (detector == "JUNO" and "snf_final_core0" in ensp
            and getattr(args, "write_output", False)
            and bool(getattr(args, "write_per_core", True))):
        ensp["per_core_steps"] = _build_per_core_steps(
            ensp, args, resp_matrix, s_ibd, core_names, core_baselines,
            core_powers, dm2_31)


    # ------------------------------------------------------------------
    # 7. Backgrounds + total spectrum
    # ------------------------------------------------------------------
    print("# Assembling total spectrum …")

    if detector == "JUNO":
        ensp["bckg"] = _load_juno_backgrounds(ensp,backgrounds, args, ebins, resp_matrix, ndays, s_ibd=s_ibd)
        if ensp["bckg"] is None:      # empty background list in the yaml
            ensp["bckg"] = Spectrum(bins=ensp["rdet"].bins,
                                    bin_cont=np.zeros(len(ensp["rdet"].bin_cont)))
        # ------------------------------------------------------------------
        # 6. Geo-neutrino spectra  (JUNO analytic path)
        # ------------------------------------------------------------------
        if args.geo_spectra == "ana":
            print("# Building analytic geo-neutrino spectra …")
            if args.geo_fit_type == "mantle":
                ensp["geomantle"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "mantle",
                    args.mantle_rate.get(args.mantle_model), args.mantle_Th_U_ratio.get(args.mantle_model),
                )
                ensp["geocrust"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "crust",
                    float(args.crust_rate_tnu) * args.tnu_to_cpd, args.crust_Th_U_ratio,
                )
                ensp["geo_ana"] = ensp["geocrust"] + ensp["geomantle"]
            elif "geo" in backgrounds["JUNO"]:
                ensp["geo_ana"] = _build_geo_spectra(
                    ensp, args, ebins, s_ibd, ndays, resp_matrix, "geo",
                    float(args.geo_tnu_rate) * float(args.tnu_to_cpd), args.Th_U_ratio,
                )
                ensp["bckg"] = ensp["bckg"] - ensp["geo"] + ensp["geo_ana"]
                ensp["geo"]  = ensp["geo_ana"]
                print(f"  geo: {ensp['geo'].GetIntegral():.2f}")
        ensp["rtot"] = ensp["rdet"] + ensp["bckg"]
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_juno(ensp, args, backgrounds)

        if args.write_output:
            _write_juno_output(ensp, ndays, args, backgrounds, s_ibd=s_ibd)

    elif detector == "DYB":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_dyb(ensp, args, ndays)
        if args.write_output: _write_dyb_output(ensp, args, backgrounds)
    elif detector == "TAO":
        extra_spectra, leg_labels, colors = _assemble_total_spectrum_tao(ensp, args, ndays)
        if args.write_output: _write_tao_output(ensp, args, backgrounds)
    ensp["rdet"].Plot(
            f"{args.plots_folder}/det_spectra_{detector}.png",
            xlabel="Reconstructed energy (MeV)",
            ylabel="Events / bin",
            extra_spectra=extra_spectra,
            leg_labels=leg_labels,
            colors=colors,
            xmin=analysis_range(args)[0], xmax=analysis_range(args)[1], ymin=1e-3, log_scale=False,
        )

    if detector != "DYB":
        ensp['rvis_nonl'].Plot(f"{args.plots_folder}/stages_{detector}.png", \
        extra_spectra = [ensp['rvis_nonl_leak'],ensp['rvis']], colors=['black', 'blue', 'orange'],\
        leg_labels = ['rvis_nonl', 'rvis_nonl_leak', 'rvis'], xmin=analysis_range(args)[0], xmax=2.0)  # low-E zoom
    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    _print_diagnostics(ensp, detector, args)

    if detector == "TAO":
        return ensp, resp_matrix, ene_leak_nd
    return ensp, resp_matrix


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_diagnostics(ensp, detector, args):
    print("\n--- Spectral integrals ---")
    for key in ("rfis0", "rfis", "ribd", "rosc", "rvis_nonl", "rvis", "rdet",
                "snf_final", "noneq_final", "rtot"):
        if key in ensp:
            print(f"  {key:<20s}: {ensp[key].GetIntegral():.2f}")
    if args.is_data:
        print(f"data: {ensp['data'].GetIntegral():.2f}")
    print(f"  2–6 MeV (ribd): {ensp['ribd'].GetIntegral(left_edge=2, right_edge=6):.2f}")
