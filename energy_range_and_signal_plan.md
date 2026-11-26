# Energy range from the yaml + plan for a flexible signal (DSNB etc.)

Status note for a new session (Nov 2026). Repo: `/home/sindhu/Documents/JUNO/Fitter/noda/noda`,
branch `main`. Run everything with `./nodaenv`, `export NODA_BASE=$PWD`.

---

## 1. DONE (uncommitted): energy range no longer hard-coded to 0–12 MeV

Goal: nothing in the code assumes 0–12 MeV; ranges come from the yaml, and every
existing config (all have `max_ene: 12`) behaves exactly as before.

### New helper — `src/noda/spectra.py`

```python
def analysis_range(args):
    """(E_lo, E_hi) of the analysis in MeV, from the yaml: first/last edge of
    bins_nonuniform when rebin_nonuniform is set, else min_ene/max_ene."""
```

### Changes

| file | what | before → after |
|---|---|---|
| `src/noda/main.py` `_build_ibd_interpolator(ebins, emax=12.0)` | IBD cross-section table | 0–12 MeV fixed (σ = 0 above 12) → 0–`emax`, called with `emax = max(12, JUNO/DYB/TAO max_ene)` |
| `src/noda/spectra.py` `_build_nl_graph` | exported NL curve | 1.022–12 → 1.022–`analysis_range[1]` (1.022 MeV kept: minimum deposited energy of an IBD positron, physics not a range) |
| `src/noda/spectra.py` `_build_resolution_graph(a,b,c, emin=0.8, emax=12.0)` | exported σ/E curve | 0.8–12 → `analysis_range` (guard: emin ≥ 0.1 MeV, σ/E diverges at 0) |
| `src/noda/spectra.py` `_write_juno_output` | exported IBD xsec graph | 0–12 → 0–`max(12, max_ene)` |
| `src/noda/spectra.py` `CreateSpectra` plots | `det_spectra_*.png` axis | xmin 0.7, xmax 12 → `analysis_range` |
| `src/noda/spectra.py` `stages_*.png` | low-E zoom | xmin 0.7 → `analysis_range[0]` (xmax 2.0 kept: zoom) |
| `src/noda/fit/combined_fit.py` (2× `rfis_posfit.png`) | plot axis | 0.7–12 → `analysis_range(args_juno)` |
| `src/noda/fit/combined_fit.py` postfit response graphs (JUNO per shell, TAO) | resolution curves | 0.8–12 → `analysis_range(b.args)` / TAO args |

`combined_fit.py` imports `analysis_range` locally (like its other `..spectra` imports).
Pre-change copies were in the scratchpad (`*.pre_erange`, may be gone); `git diff src/noda/main.py src/noda/spectra.py src/noda/fit/combined_fit.py` shows the change.

Only visible differences for existing configs: exported resolution graphs now start at the
analysis lower edge (JUNO 0.7 MeV instead of 0.8; TAO 0.8), plot axes follow the yaml.

### Checks done

- Cross-section: `_build_ibd_interpolator(emax=40)` gives σ(25 MeV) = 4.39e-41 cm² (was 0);
  identical to the 12 MeV table below 12 MeV.
- Step 1 (IBD xsec + 12 → max_ene parts) regression: **bit-identical** on the three standards
  (χ²_min, sin²θ₁₂, Δm²₃₁):
  - 2-shell: 231.8422130919622, 0.305315, 2.50286
  - 16.5 m: 148.81455643734114, 0.303575, 2.49571
  - P25C: 85.55307658036263, 0.308046, 2.54015

### TEST STILL TO RUN (interrupted)

The regression after the second part (`analysis_range` for graphs/plots) was interrupted.
Code compiles and imports. Re-run and compare with the numbers above; fits must be
bit-identical (only output graphs/plots changed):

```bash
export NODA_BASE=$PWD; R=./nodaenv/bin/run_noda
$R --config config/standard/p26b_2shell/manifest.yaml --file_prefix erange_test_2shell \
   --profile_dm2_31 False --save_prefit False > erange_2shell.log 2>&1
$R --config config/standard/p26b_165m/config.yaml --file_prefix erange_test_165m \
   --profile_dm2_31 False --use_prefit True --save_prefit False \
   --prefit_path $PWD/prefit_data/240_allP_165_TAO911_Nov1-2026.npz > erange_165m.log 2>&1
$R --config config/standard/p25c_prl/config.yaml --file_prefix erange_test_p25c \
   --save_prefit False > erange_p25c.log 2>&1
grep -E "Function minimum|^sin2_th12:|^dm2_31:|Traceback" erange_*.log
```

Also check the exported graphs of the 2-shell file:
`juno_inner/response/{prefit,postfit}/resolution` should span 0.7–12 MeV,
`tao/response/prefit/resolution` 0.8–12 MeV. Then delete the test `Postfit_erange*` files
(a partial `Postfit_erange2_2shell_…root` from the interrupted run is still in the repo root).

---

## 2. PROPOSED (not started): flexible signal, everything switched by yaml

Use case: DSNB (or any non-reactor signal) with a new energy range and backgrounds yaml.
Defaults must keep the reactor analysis bit-identical.

### 2a. Signal model chosen in the yaml

```yaml
juno_inputs:
  signal_model: reactor          # reactor (default) | analytic | template
```

- **`reactor`**: today's code, unchanged.
- **`template`** (MC signal spectrum, e.g. DSNB from simulation):
  ```yaml
  signal:
    rootfile: ${NODA_BASE}/data/dsnb_mc.root
    histname: dsnb_prompt
    stage: reco          # reco = already reconstructed (skip positron/NL/response)
                         # true = E_nu spectrum, goes through xsec+positron+NL+response
    rate: 1.0            # expected events/day (or scale)
    norm_name: dsnb_norm # parameter of interest, free, >= 0
  ```
  A `reco` template is used like a background histogram (rebinned, efficiency,
  `(1+norm)`); a `true` template goes through the same chain as the reactor flux.
- **`analytic`**: parametrised flux (e.g. pinched Fermi–Dirac with mean energy) through the
  full chain; extra shape parameters listed in the yaml.

### 2b. Where the reactor is hard-wired today (what to touch)

- `JUNODetector.build_spectrum` (`src/noda/fit/juno_fit.py` ~565–720): one contiguous
  block builds the signal in deposited energy — core powers, efission, shape segments,
  IBD flux, matter density, oscillation, positron conversion — returning `s_juno`,
  `osc_key` (reused as the NL-cache key) and `rfis_param_post`. SNF/non-eq is added after
  the response. These become the signal hooks:
  `deposited(…) -> (spectrum, cache_key, extra)`, `reco_corrections(…)`.
- `JUNODetector.get_param_names / get_start_values / apply_limits / pull defs`:
  reactor parameters `Nrea, snf_noneq_norm*, core_power*, efission*` (+ `ThU_ratio` from geo).
  Signal parameters must come from the signal model. **Keep parameter and pull order**
  for the reactor case (Minuit order / sum order → bit-identical).
- `spectra.CreateSpectra` JUNO block (`_build_reactor_flux_per_core`,
  `_build_snf_noneq_corrections_per_core`, `_build_reactor_flux_simple`, bump interpolator).
- `combined_fit`: oscillation parameters are always shared Minuit parameters →
  make optional, or add a generic yaml `fixed_params: [...]` (fit_options) to fix them.
- Output writer: `oscillation_parameters/` → also `signal_parameters/<name>`.

### 2c. Backgrounds (generic)

- `_assemble_total_spectrum_juno` (`spectra.py`) requires the keys
  `acc, fastn, lihe, alphan, geo, atm, wr, doublen, bipo` → loop over the yaml keys.
- Efficiency on backgrounds is hard-wired to `wr, geo, atm` (`_load_juno_backgrounds`) →
  per-background yaml key, e.g. `eff: [ibd, psd]` (default = today's behaviour).
- Reactor neutrinos as a background for DSNB: reactor model at fixed oscillation, or a template.

### 2d. Other items for DSNB

- TAO/DYB: `include_tao: false`, `include_dyb: false`, `fit_mode: combined`; the setup
  still builds the TAO namespace (needs a `tao_inputs` block) → make it optional.
- Inputs above 12 MeV: positron TF2 (`Epositron_Enu_cos_StrumiaVissani` in
  `JUNOInputs2022`), NL curve (extrapolation + uncertainty above ~10 MeV) must cover the range.
- Grid: 0–60 MeV at 20 keV = 3000 bins (~70 MB matrices); the 20 keV + 4-pt rule is for
  reactor oscillations only → 50–100 keV true bins, ~1–2 MeV analysis bins for DSNB.
- Statistics: low counts → add a Poisson (−2 ln L / Baker–Cousins) option to
  `stat_method_opt`; limits via CLs/FC on the signal norm, discovery q₀; generalise
  `minuit.profile_dm2_31` and the FC tools to any named parameter.
- PSD categories / volumes: reuse the manifest (one dataset per category, shared POI).

### Suggested order

1. (done) energy range from yaml — regression re-run Nov 25: bit-identical, graphs 0.7/0.8–12 MeV.
2. (done, uncommitted) generic background keys + per-background `efficiency: [detector, eff_nuis, oec]`
   (`src/noda/backgrounds.py`; defaults geo/wr all three, atm detector only) — regression bit-identical on all 3 standards.
3. Signal interface with `reactor` only → regression.
4. `template` signal (reco and true stage) → Asimov closure test on a toy DSNB config
   in `config/standard/dsnb/` (needs DSNB MC spectrum, backgrounds, PSD efficiencies).
5. Poisson likelihood, generic profile/limit tools.
