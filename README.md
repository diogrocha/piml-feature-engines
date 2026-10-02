# piml-feature-engines

Physics-informed feature engines for survival-based prognostics, built around a Landau–Ginzburg
stability order parameter, together with the experiments that compare them with other feature
paradigms under one fixed gradient-boosting survival learner (GBSA).

## Quick start

**Google Colab.** Open [`notebooks/reproduce_all.ipynb`](notebooks/reproduce_all.ipynb) and run all
cells. It clones the repository, installs the dependencies, downloads the public datasets and runs every
experiment, saving results to Google Drive. Set `SMOKE = True` for a ten-minute end-to-end check.

**Locally** (Python 3.10–3.12):

```bash
pip install -r requirements.txt
bash run_all.sh --smoke     # a few minutes
bash run_all.sh             # full run, about 7-8 h on 2 CPUs
```

Outputs are written to `results/`.

## Layout

| Folder | Contents |
|---|---|
| `src/` | feature engines (`engines.py`: Landau–Ginzburg order parameter, health index, Weibull, entropy, exponential degradation, rolling statistics), multi-mode LG, data loaders, cross-validation, the shared evaluation protocol (`pipeline.py`), LG calibration (`optimize.py`) |
| `theory_checks/` | checks of the LG feature map: collapse of the four descriptors onto one order parameter, identifiability, Kramers regime; plot of the energy landscape |
| `cmapss_paradigms/` | grouped five-fold cross-validation of every feature arm on C-MAPSS FD001–FD004; rate-based arms and non-physical controls; supervised against unsupervised multi-mode directions |
| `azure_batteries/` | the same comparison on Azure predictive maintenance and the NASA batteries |
| `mode_channels/` | the three unsupervised order-parameter channels on FD003 |
| `critical_slowing_down/` | critical-slowing-down test with a memoryless null model |
| `turning_simulation/` | regenerative turning with a known Hopf boundary, and model-based state estimators (no data needed) |
| `robustness/` | integrated Brier score, time-dependent AUC and remaining-life error |
| `case_study/` | operator warning zones on a test engine and warning lead times |
| `cost/` | per-sample inference cost, and a microcontroller implementation of the LG feature map in `edge/` |
| `cox_kramers/` | Cox model with a Kramers-type term against a free spline |
| `notebooks/` | Colab notebook that runs everything |

Run every script from the repository root (for example `python cmapss_paradigms/run_cv.py`), except
those in `turning_simulation/`, which run from their own folder (see its README).

## Protocol

- Grouped five-fold cross-validation by unit, identical folds for every arm; every feature engine is
  calibrated inside the training fold only.
- 6000 training rows per fold, seed 42; every arm, including the baseline, uses the same 150-tree GBSA
  configuration (`src/config.py`).
- Paired differences between arms with 95% intervals from Student's t.

## Feature arms

| Code | Meaning |
|---|---|
| `Base-AUG` | baseline features (rolling means and standard deviations), 150 trees |
| `+LG-eta` | baseline + the Landau–Ginzburg order parameter |
| `+LG` | baseline + the four LG descriptors (R, ΔF, Γ, κ) |
| `+LG-mm2` | baseline + two order parameters on the leading principal directions |
| `LG-comb-mm` | the above + their per-unit rates of change |
| `HI-fix` | baseline + health index with a per-unit rate |
| `PC-twin` | baseline + principal scores and per-unit rates, without the LG map |
| `+Stats`, `+HI`, `+Weibull`, `+Entropy`, `+ExpDeg` | other feature paradigms |

## Data

See [`DATA.md`](DATA.md). No data are distributed with this repository.

## Licence

MIT, see [`LICENSE`](LICENSE).
