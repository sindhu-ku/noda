# Flexible signal for NODA (DSNB etc.) — handover

Repo: `/home/sindhu/Documents/JUNO/Fitter/noda/noda`, branch `main`.
Always run with `./nodaenv` and `export NODA_BASE=$PWD`.

---

## 0. Package layout (Nov 2026)

```
src/noda/
  main.py        run_noda entry point (setup_one_phase, CreateSpectra calls)
  noda.py        compatibility facade: re-exports core/ + physics/ (`from .noda import *`)
  core/          spectrum (Spectrum), covmatrix, response (RespMatrix, CalcRespMatrix_abc),
                 binning, likelihood (Chi2), fileio (GetSpectrumFromROOT, pickles), root_utils
  physics/       nuosc, nl_geant4, constants, reactor (flux normalisation, GetPee), tao_bkg
  model/         assemble (CreateSpectra: signal + backgrounds -> total, prefit output)
                 reactor (reactor flux -> IBD x osc -> per-core chain, SNF/non-eq)
                 detection (positron -> NL -> response -> leakage -> OEC efficiency)
                 backgrounds (yaml definitions + background loading, world reactors, geo)
                 matrices (systematic covariances), DYB_model
  fit/           juno_fit (JUNODetector), tao_fit, dyb_fit, combined_fit, minuit, dispatcher
  fc/ bayesian/  Feldman-Cousins, PT-MCMC
  legacy/        superseded code, not used
```

`noda.nuosc` / `noda.constants` are aliases of `noda.physics.*` (same module objects).

## 1. Done

- **Energy range from the yaml.** Nothing assumes 0–12 MeV: `model.assemble.analysis_range(args)`
  gives (E_lo, E_hi) from `bins_nonuniform` or `min_ene/max_ene`; IBD cross-section table
  extends to `max(12, max_ene)`; exported NL/resolution graphs and plot axes follow the yaml.
- **Generic backgrounds.** Any set of keys in the backgrounds yaml (or none). Optional
  per-background `efficiency: [detector, eff_nuis, oec]` (default: geo/wr all three, atm
  `detector`, others none) — see `model/backgrounds.py`. Postfit histograms are named by the
  yaml key (name -> key registry filled in `main.py`).
- **Package restructure** (section 0). Pure moves.

All three are bit-identical on the three standard fits (section 4).

## 2. To do: flexible signal (DSNB)

Use case: DSNB search, prompt energy **12–30 MeV, 2 MeV bins** (JUNO 206.5-day preliminary
uses E_nu 12.8–30.8 MeV = prompt ~12–30). No reactor signal, no oscillation parameters,
no TAO/DYB. Reactor fits must stay bit-identical.

### 2a. yaml

```yaml
juno_inputs:
  min_ene: 12
  max_ene: 30
  rebin_nonuniform: true
  bins_nonuniform: ...          # 2 MeV bins, 12-30
  include_tao: false
  include_dyb: false
  stat_method_opt: poisson      # new
  backgrounds_yaml: ${NODA_BASE}/config/standard/dsnb/backgrounds.yaml
  signal_model: template        # reactor (default) | template | analytic (later)
  signal:
    rootfile: ${NODA_BASE}/data/dsnb/dsnb_mc.root
    histname: prompt_after_cuts # RECO prompt energy after all cuts/PSD, fine bins (<=0.1 MeV)
    stage: reco                 # reco: used like a background template; true: E_nu, through detection chain
    rate: ...                   # expected events/day after cuts
    poi: dsnb_mu                # signal strength (1 = model)
    poi_limits: [0, 20]
    efficiency: []              # same options as backgrounds
```
Backgrounds (atm NC, atm CC, fast n, ...) use the existing generic format; set `b2b_unc: 0`
for the Poisson likelihood.

### 2b. Code

| step | where | what |
|---|---|---|
| 1 | new `model/signal.py` | `SignalModel` interface: `param_names, start_values, limits, pull_defs, prefit(ensp,args,ndays), spectrum(p,ctx), uses_oscillation()`. `ReactorSignal` wraps `model/reactor.py` **unchanged** (same param/pull order). Regression must be bit-identical. |
| 1 | `model/assemble.py` `CreateSpectra` | JUNO reactor block -> `signal.prefit(...)` |
| 1 | `fit/juno_fit.py` `JUNODetector.build_spectrum` (~565–820) and `get_param_names/get_start_values/apply_limits/get_pull_defs` | reactor block -> `signal.spectrum(...)`; params from the signal model. Background loop after it stays. |
| 2 | `model/signal.py` `TemplateSignal` | reco histogram -> rebin -> x rate x ndays x mu |
| 2 | `fit/combined_fit.py` | if `not uses_oscillation()`: fix the 4 osc params + `me_rho` (keep `N_SHARED` indexing) |
| 2 | `main.py` `setup_one_phase` | skip building TAO/DYB when switched off (TAO setup partly runs today even with `include_tao: false`) |
| 2 | output writer | `signal/` dir: prefit/postfit signal, POI value/error |
| 3 | `core/likelihood.py` | `PoissonChi2` (Baker–Cousins `2*sum[exp-obs+obs*ln(obs/exp)]` + pulls). **Keep obs=0 bins** (they contribute `2*exp`; `Chi2` drops them). Refuse a non-zero covariance in Poisson mode. Select in `JUNODetector.chi2_contribution` (`juno_fit.py`, call to `Chi2`). |
| 4 | `fit/minuit.py` | generalise `profile_dm2_31` -> `profile_param(m, name, bound, size, conditional, csv)`; keep `profile_dm2_31` as wrapper (bit-identical output) |
| 4 | new `fit/limits.py` | upper limit from the POI profile (Δχ² 2.71 / 3.84 one-sided, mu>=0 boundary), q0 discovery; hook in `combined_fit.py` next to the dm2_31 profile, yaml `minuit: profile_params: [dsnb_mu]` |
| later | `fc/` | FC/CLs toys for the POI (currently dm2_31-specific) |

Test: toy config in `config/standard/dsnb/`; Asimov fit must return `dsnb_mu = 1`.

### 2c. Notes

- A **reco-level** MC template avoids the reactor-specific inputs that stop near 12 MeV
  (positron TF2 `Epositron_Enu_cos_StrumiaVissani`, NL curve, response grid). A **true-level**
  spectrum needs those extended to 30+ MeV and a coarser true grid (50–100 keV; the 20 keV +
  4-point rule is for reactor oscillations only).
- Energy-scale uncertainty on reco templates: later, a scale nuisance E -> (1+a)E on all templates.
- PSD categories / volumes: one manifest dataset per category, shared POI.

## 3. Known pre-existing issues (not fixed)

- `legacy/toymc.py` imports a non-existent `.minuit`; `model/DYB_model.py` needs `dayabay_model`
  (only for `use_dagflow: true`).
- Undefined names in rarely used branches: `valid` in `Combined_Chi2`, `axis_label` in
  `GetCovMatrixFromROOT`, `Enu` in one `Spectrum` branch.
- With `NL: pull`, the world-reactor OEC correction uses the reactor column (`juno_fit.py`).

## 4. Regression (run after every step)

```bash
export NODA_BASE=$PWD; R=./nodaenv/bin/run_noda
$R --config config/standard/p26b_2shell/manifest.yaml --file_prefix reg_2shell \
   --profile_dm2_31 False --save_prefit False > reg_2shell.log 2>&1
$R --config config/standard/p26b_165m/config.yaml --file_prefix reg_165m \
   --profile_dm2_31 False --use_prefit True --save_prefit False \
   --prefit_path $PWD/prefit_data/240_allP_165_TAO911_Nov1-2026.npz > reg_165m.log 2>&1
$R --config config/standard/p25c_prl/config.yaml --file_prefix reg_p25c \
   --save_prefit False > reg_p25c.log 2>&1
grep -E "Function minimum|^sin2_th12:|^dm2_31:|Traceback" reg_*.log
```

Expected (χ²_min, sin²θ₁₂, Δm²₃₁), must match to all digits:
- 2-shell: 231.8422130919622, 0.305315, 2.50286
- 16.5 m: 148.81455643734114, 0.303575, 2.49571
- P25C: 85.55307658036263, 0.308046, 2.54015

Never reuse an existing `--file_prefix` (outputs and prefit caches are overwritten; many
are untracked). Delete the `Postfit_reg_*` files and `Plots/*reg_*` afterwards.
