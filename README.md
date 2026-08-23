# A Comparative Study of EEG Denoising Methods: Assessing Their Impact on Downstream Task Performance

This study evaluates how six EEG preprocessing conditions (Raw, bandpass filtering, ICA, ASR with k=5 and k=20, and IC-U-Net) affect downstream classification performance across BCI, seizure detection, and attention state tasks. Using eight traditional and deep learning classifiers under consistent experimental settings, the benchmark evaluates each preprocessing condition with Balanced Accuracy as the primary metric and AUC as the secondary metric. The goal is to assess preprocessing methods in the context of their intended downstream application.
![Overview](<figure/Figure 1.png>)

## Installation

**Step1:** Clone this repo
```bash
git clone https://github.com/ase12345636/DSBI-EEG-Denoising
```
**Step2:** Create a conda enviorment for this work
```bash
# Recommand using python 3.10 for this work.
conda create --name EEG-Denoising python==3.10

conda activate EEG-Denoising
```
**Step3:** Install package from requirements.txt
```bash
pip install -r requirements.txt
```
## Usage

### Data access

The datasets are downloaded automatically on the first run. The BCI and attention-state datasets require a [Kaggle](https://www.kaggle.com/) account. Place your API token at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`) and accept the [BCI competition rules](https://www.kaggle.com/c/inria-bci-challenge/rules) before running the pipeline. The CHB-MIT seizure dataset and the included IC-U-Net checkpoint do not require authentication.

To download all datasets without running any experiments:

```bash
python main.py --download-only
```

### Check the available components

```bash
python main.py --list
python main.py --dry-run
```

`--list` prints all registered tasks, denoising methods, and classifiers. `--dry-run` validates the selected components and displays the required data, cache, and output paths without downloading or training anything.

### Quick test

Run a small one-repeat smoke test before starting the full benchmark:

```bash
python main.py --quick
```

Quick-test results are written to `output/quick_smoke/` and are not intended to be used as report results.

### Run the benchmark

Running without arguments evaluates all combinations of 3 tasks, 6 preprocessing methods, 8 classifiers, and 10 repeats (1,440 model runs):

```bash
python main.py
```

Individual components can be selected with `--tasks`, `--methods`, and `--classifiers`. Multiple names may be supplied after each option:

```bash
# Run only the BCI task
python main.py --tasks bci_errp

# Compare Raw and bandpass filtering on two tasks
python main.py --tasks seizure_detection attention_state --methods raw bandpass

# Run selected classifiers only
python main.py --classifiers logistic_regression svm eegnet

# Combine all selection options
python main.py --tasks bci_errp --methods raw ica --classifiers svm random_forest
```

Available component names are:

- Tasks: `bci_errp`, `seizure_detection`, `attention_state`
- Methods: `raw`, `bandpass`, `asr`, `asr20`, `ic_unet`, `ica`
- Classifiers: `logistic_regression`, `svm`, `random_forest`, `lightgbm`, `mlp`, `eegnet`, `vit`, `mobilenet`

To use a different configuration file:

```bash
python main.py --config path/to/config.json
```

### Download options

```bash
# Use data that have already been downloaded
python main.py --skip-download

# Download the requested data again
python main.py --force-download --download-only
```

`--skip-download` requires the expected dataset files to already exist under each task's `dataset/<task>/data/raw/` directory. Interrupted HTTP downloads are retained as `.part` files and resumed automatically.

### Resume an interrupted run

Run the same command again after an interruption. Completed model runs are loaded from `output/all_runs.partial.csv`, and the saved `output/repeat_seed_plan.json` ensures that the same repeat seeds are reused. Prepared signals, features, and model artifacts are cached under `.cache/` to reduce repeated work.

If the code, configuration, source data, or IC-U-Net checkpoint changes, remove `.cache/` before starting a new benchmark to avoid reusing stale cached data.

### Outputs

Results are written to `output/` (or `output/quick_smoke/` in quick mode), including:

- `all_runs.csv`: metrics for every model run
- `summary.csv`: mean results grouped by task, method, and classifier
- `wilcoxon_two_sided_vs_raw.csv` and `wilcoxon_one_sided_vs_raw.csv`: paired statistical tests against Raw with Holm correction
- `run_manifest.json` and `repeat_seed_plan.json`: run configuration and repeat seeds
- Per-task directories containing classifier tables, method-average tables, and performance figures
