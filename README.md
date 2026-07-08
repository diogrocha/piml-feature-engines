# piml-feature-engines

Reproduction code for a fixed-learner comparison of feature paradigms for
survival-based prognostics. All scripts share one gradient-boosting survival
learner and one random seed, so results depend only on the feature set.

## 1. Setup

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

The pinned, verified working set (Python 3.10/3.11) is:
`numpy==1.26.4  pandas==2.2.2  scipy==1.13.1  scikit-learn==1.5.2
scikit-survival==0.23.1  optuna==3.6.1  matplotlib==3.9.2`.

## 2. Data

The datasets are not included. Download them and place the files as described
in [`DATA_SETUP.md`](DATA_SETUP.md):

- C-MAPSS  -> `data/CMAPSSData/`
- Azure    -> `data/azure/`
- Battery  -> `data/battery/`

Keep the original file names (the loaders locate the folders by anchor files).

## 3. Running the pipelines

Most run scripts follow a three-step pattern: `prep` (build features and
labels), `fit` (train the shared learner), and `report` (print the scores).

### C-MAPSS (per subset FD001–FD004)

```bash
python cmapss_fd_run.py prep FD003
python cmapss_fd_run.py fit FD003
python cmapss_fd_run.py report FD003
```

### Azure predictive maintenance

```bash
python azure_run.py prep        # all components (or pass component names)
python azure_run.py fit <component>
python azure_run.py report
```

### NASA battery

```bash
python battery_run.py
```

Note: `battery_run.py` uses fixed input/output paths near the top of the file
(`DL.load(...)` and the output `json.dump(...)`). Edit those two paths to point
at your `data/battery/` folder and your desired output location before running.

## 4. Cross-paradigm comparison

Compares every feature paradigm under the shared learner on one dataset. Uses
the reported hyperparameters by default; pass `--optimize` to re-run the Optuna
search instead.

```bash
python cross_paradigm_cmapss.py --data data/CMAPSSData --out results_cmapss.json
python cross_paradigm_xjtu.py   --data <xjtu_dir>       --out results_xjtu.json
```

## 5. Cross-validation

```bash
python crossval.py --dataset FD003 --data data/CMAPSSData --folds 3
```

## 6. Critical-slowing-down analysis

```bash
python csd_cmapss.py
python csd_features.py --data data/CMAPSSData --window 30
```

## 7. Bootstrap confidence intervals

```bash
python bootstrap_ci.py fit FD003
python bootstrap_ci.py ci  FD003
```

## 8. Figures

```bash
python figures.py --cmapss data/CMAPSSData --xjtu <xjtu_dir>
```

Individual figure scripts (`fig1_figure.py`, `landscape_figure.py`,
`mode_loadings_figure.py`, `pipeline_figure.py`, `diagnostic_figure.py`,
`azure_figure.py`, `battery_figure.py`, `csd_figure.py`, `fig_fd003_bars.py`)
are imported/driven by `figures.py` and the run scripts; run `figures.py` rather
than calling them directly.

## 9. Verification

Checks that the produced rankings match the expected ordering:

```bash
python verify.py --cmapss data/CMAPSSData --xjtu <xjtu_dir>
```

## Edge deployment (`edge/`)

C implementation of the LG descriptor kernel for an Arm Cortex-M0+ core, run
under emulation. See the build/run commands inside the files in `edge/`
(`export_lg.py` regenerates `lg_data.h`; `lg.c` / `lgbench.c` are the kernel and
benchmark; `startup.s` and `link.ld` are the bare-metal boot and linker script).

## Google Colab

`run_colab.ipynb` (XJTU per-paradigm comparison) and `run_FD003_colab.ipynb`
(C-MAPSS FD003 full run) install the dependencies and run end to end in Colab;
open them there and execute the cells top to bottom.

## Notes

- Fixed random seed (`SEED = 42` in `config.py`); all hyperparameters live in
  `config.py`.
- Modules (`config.py`, `engines.py`, `*_data.py`, `*_figure.py`, `optimize.py`,
  `origin_style.py`, `master_table.py`, the `cv_*_check.py` and
  `*_multimode_check.py` helpers) are imported by the run scripts and are not
  meant to be executed on their own.
