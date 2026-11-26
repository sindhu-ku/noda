# NODA — Neutrino Oscillation Data Analysis

NODA is a fitting framework for JUNO reactor-antineutrino analyses. It fits JUNO
together with TAO and Daya Bay (DYB) to measure the oscillation parameters
(sin²θ₁₂, Δm²₂₁, Δm²₃₁, sin²θ₁₃), the neutrino mass ordering (NMO) and the
geoneutrino rate. It supports both real data and Asimov sensitivity studies.

Capabilities:
- **Frequentist fit** (iminuit): JUNO (one or more fiducial shells) + TAO + DYB,
  with three strategies for the reactor-shape constraint.
- **Δm²₃₁ profiles**, solar (sin²θ₁₂–Δm²₂₁) contours, θ₁₃–Δm²₃₁ maps.
- **NMO significance**: Δχ²(IO−NO) with DYB and/or NuFIT long-baseline Δm²₃₁ constraints.
- **Feldman–Cousins** toy studies for Δm²₃₁ intervals and NMO (`run_fc_toy`, `run_fc_observed`, `fc_analyze`).
- **Bayesian** parallel-tempering MCMC on the same χ² (`run_noda_mcmc`, `noda_mcmc_results`).
- **Geoneutrinos**: free or constrained geo rate, Th/U ratio, crust/mantle split.
- ROOT output with prefit/postfit spectra, nuisance pulls, response curves and
  step-by-step spectra (layout compatible with the ORSA fitter, for direct comparison).

---

## Installation

Requires Python ≥ 3.9 with **PyROOT**. ROOT must be built for the Python you use.

```bash
git clone <repo-url> noda && cd noda
python3 -m venv nodaenv && source nodaenv/bin/activate
pip install -e .            # editable: changes in src/noda are live
pip install numba           # needed by the Geant4 non-linearity model
```

Check: `python -c "import ROOT, noda"` must work in that environment.

Every config path uses `${NODA_BASE}`, which should point at the repository root:

```bash
export NODA_BASE=$PWD       # or pass --base_path to any command
```

Input data (`data/`, ROOT files and text tables) and prefit caches (`prefit_data/`)
are not in git; copy them into the tree before running.

---

## Repository layout

```
src/noda/
  main.py            run_noda entry point: builds the datasets, runs the fit
  spectra.py         reactor, background and geo spectra, prefit ROOT output
  noda.py            Spectrum / response / covariance classes, oscillation sampling
  nuosc.py           oscillation probability (3 flavours, matter) and parameter sets (PDG_opt)
  fit/
    combined_fit.py  bundled multi-detector Minuit fit, prefit stage, postfit output
    juno_fit.py      JUNO model (per shell)
    tao_fit.py       TAO model (incl. step-by-step stages)
    dyb_fit.py       Daya Bay model
    dispatcher.py, shared_nuisances.py, param_policy.py   parameter sharing across shells/phases
    manifest.py      multi-phase / multi-shell manifests
    minuit.py        profiles, contours, ROOT parameter writers
  fc/                Feldman–Cousins tools
  bayesian/          PT-MCMC tools
config/
  standard/          the maintained configurations (see below)
  legacy/            older and test configurations, kept for reference
scripts/             plotting and analysis scripts (e.g. plot_nuis_new.py)
combine_profile.py   merges chunked Δm²₃₁ profile jobs into one ROOT graph
```

---

## Configurations

A configuration is a YAML file with sections `fit_options`, `common_det_inputs`,
`juno_inputs`, `dyb_inputs` and `tao_inputs`. Every key can be overridden on the
command line as `--key value`, but **only keys that already exist in the YAML**.

Multi-shell or multi-phase fits use a **manifest**: `common:` points at one full
config, and `phases:` lists only what differs per dataset (efficiency,
backgrounds file, data spectrum, response file, binning, …).

| Standard config | Use |
|---|---|
| `config/standard/p26b_2shell/manifest.yaml` | P26B data, 2 shells (FV < 15 m and 15–17.2 m), JUNO + DYB + TAO. Main analysis. |
| `config/standard/p26b_165m/config.yaml` | P26B data, single 16.5 m volume |
| `config/standard/p25c_prl/config.yaml` | P25C (PRL 2025) dataset |
| `config/standard/geo_sens/config.yaml` | Geoneutrino sensitivity (JUNO 1-year Asimov + DYB) |

Each folder holds its own backgrounds and `reactor_summary.yaml`.

Conventions used in the standard configs:
- True-energy grid: `bins: 601` (20 keV, 0–12 MeV) with `osc_points_per_bin: 4`
  (P_ee averaged over 4 points per bin; avoids aliasing of the far reactor cores).
- `stat_method_opt: CNP` (combined Neyman–Pearson χ²).
- `fit_mode: prefit_wosegs` (see below).

---

## Usage

### Basic fit

```bash
export NODA_BASE=$PWD
run_noda --config config/standard/p26b_2shell/manifest.yaml --file_prefix mytest
```

Output: `Postfit_mytest_NO_True_dm2_31_pull_True_prefit_wosegs.root` in the
working directory, plus plots and CSV tables in `Plots/`. The fit results
(oscillation parameters, background rates, nuisances, validity flags) are
printed to the terminal.

**Always give a new `--file_prefix`**; an existing file with the same name is updated in place.

### Fit strategies (`--fit_mode`)

| `fit_mode` | Reactor-shape constraint |
|---|---|
| `combined` | JUNO, TAO and DYB fitted together in one Minuit fit |
| `prefit_wsegs` | DYB + TAO prefit; reactor-shape segments float in the JUNO fit |
| `prefit_wosegs` | DYB + TAO prefit; segments fixed at the prefit best fit, their covariance propagated to JUNO (standard) |

### Prefit cache

In the prefit modes the DYB + TAO fit is cached in a `.npz` file (`prefit_path`).

```bash
# reuse an existing cache
run_noda --config ... --file_prefix run1 --use_prefit True --save_prefit False \
         --prefit_path "$PWD/prefit_data/<cache>.npz"

# rebuild it (needed after ANY change of TAO/DYB inputs, binning or TAO settings)
run_noda --config ... --file_prefix run2 --use_prefit False --save_prefit True \
         --prefit_path "$PWD/prefit_data/<new_cache>.npz"
```

Pass `--prefit_path` as an absolute path (`${NODA_BASE}` is not expanded on the
command line). Save new caches under a new name; don't overwrite existing ones.

### Common options

| Option | Effect |
|---|---|
| `--is_data False` | Asimov fit instead of data |
| `--pull_dm2_31 0/1` | Daya Bay Δm²₃₁ constraint |
| `--pull_dm2_31_lbl True` | NuFIT long-baseline Δm²₃₁ constraint |
| `--NMO_sig True` | also fit the inverted ordering and report Δχ²(IO−NO) |
| `--profile_dm2_31 False` | skip the Δm²₃₁ profile scan (faster) |
| `--include_tao 0`, `--include_dyb 0` | drop a detector |
| `--write_output 0` | no ROOT output |
| `--write_per_core True` | also write per-core step-by-step JUNO reactor spectra |
| `--th13_dm31_2d 1 --th13_dm31_tag <t>` | sin²θ₁₃–Δm²₃₁ Δχ² map |
| `--solar_mncontour true` | profiled sin²θ₁₂–Δm²₂₁ contour |

### Reading results

ROOT file (read with uproot or ROOT):
- `oscillation_parameters/<name>` and `<name>_err`: best-fit values (TParameter).
- `juno*/nuisances/<name>`: [pull, error] in units of the prior σ. Constrained
  background pulls are relative to `rate_unc`.
- `juno*/prefit|postfit/…`, `tao/…`, `dyb/…`: spectra. Multi-shell runs write
  `juno_inner/` and `juno_outer/`.
- `tao/prefit|postfit/{unosc, osc_wcorr, dep, vis, rec, reactor, bkg, totalpred}`:
  TAO spectrum at every step (`osc_wcorr` is the antineutrino spectrum in true energy).
- `profiles/chi2_profile`: Δm²₃₁ profile. Quote Δm²₃₁ from the profile: parabolic
  vertex plus the Δχ² = 1 crossings (`read_dm2_31_profile` in `scripts/plot_nuis_new.py`).

Comparison plots of several fits:

```bash
python scripts/plot_nuis_new.py <output_name>    # edit the FILES dict at the top first
```

### Δm²₃₁ profiles on a batch system

Split the profile into chunks, run each as a job, then merge:

```bash
run_noda --config ... --profile_nchunks N --profile_chunk i \
         --profile_csv chunk_i.csv --profile_make_graph 0
python combine_profile.py ...        # writes the profiles/dm2_31 TGraph
```

### Feldman–Cousins

```bash
# 1. fits to the data (free fit + fixed-Δm²₃₁ grid)
run_fc_observed --config <cfg> --prefit-cache <cache.npz> --output fc_obs.npz --mode both

# 2. toys (one task per job; seed = seed_base + task_id*n_toys + i)
run_fc_toy --config <cfg> --task-id 0 --n-toys 100 --mode nmo --true-no 1 \
           --prefit-cache <cache.npz> --initial-fits fc_obs.npz --output-dir toys/

# 3. intervals, NMO significance, coverage
fc_analyze --input-dir toys/ --output-dir fc_results/ --cls 0.6827,0.9545 \
           --t-obs-nmo <chi2_IO - chi2_NO> [--coverage]
```

`--PDG_opt` must be the same for `run_fc_toy` and `run_fc_observed`.

### Bayesian MCMC

```bash
# oscillation parameters only, nuisances at their best fit (fast)
run_noda_mcmc --config <cfg> --ordering both --params osc --output mcmc_osc.npz

# all free parameters (slow; use many cores)
run_noda_mcmc --config <cfg> --ordering both --params all --nsweeps 4000 \
              --ncores 16 --output mcmc_all.npz

noda_mcmc_results --input mcmc_all.npz --outdir mcmc_plots/ [--add-pull dyb,lbl]
```

The posterior is exp(−χ²/2) of the same χ² that the frequentist fit minimises.
With `--ordering both` the NO/IO posterior odds follow from the chain.

---

## Notes

- Fit time: about 5 min for the 2-shell standard with a cached prefit; add a few
  minutes for the Δm²₃₁ profile.
- `lin_resp: True` (linearised energy response) is faster and uses much less memory.
