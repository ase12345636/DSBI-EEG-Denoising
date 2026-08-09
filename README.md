# EEG denoising downstream-task reproduction — revised build

This package runs the three report tasks with one entry point:

```bash
python main.py
```

It downloads the public data, prepares Raw / Band-pass / ASR / IC-U-Net / ICA variants, runs six classifiers for ten seeds, and writes Tables 5–10, Figures 6–14, run-level CSV files, and Mann–Whitney tests into `output/`.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m pip install -r requirements.txt
```

BCI and Attention use Kaggle. Put `kaggle.json` in `~/.kaggle/kaggle.json`, set file permission to 600, and accept the BCI competition rules in the browser. CHB-MIT and the included IC-U-Net best checkpoint do not require Kaggle authentication.

## Common runs

```bash
python main.py
python main.py --tasks bci_errp
python main.py --tasks seizure_detection attention_state
python main.py --dry-run
```

A stopped run resumes automatically from `.cache/run_state/all_runs.partial.csv`.
Each new run creates a random repeat-seed plan; a resumed run reuses the saved
plan instead of fixing a project-wide seed. Attention and Seizure evaluate two
complete five-repeat split cycles, so every subject is assigned to a test fold
once per cycle. BCI keeps the authors' fixed train/test subject split.
Signal, feature, seed, and partial-result caches are all disposable: after changing
the code, configuration, checkpoint, or source data, remove `.cache` before the
next run. Final files in `output/` are never used as resume state. Runs over a
selected subset of tasks, methods, or classifiers still write the available
tables and figures; a complete 900-run matrix is not required.

## Selected implementation choices

- BCI uses the author repository's fixed 16-subject training / 10-subject testing split, 30 channels, -100 to 600 ms epochs, xDAWN covariance (`nfilter=5`), tangent-space features, and training-only SMOTE.
- Seizure testing uses two complete subject-disjoint five-fold cycles, with every person tested exactly once per cycle. The two recordings from the same person (`chb01` and `chb21`) stay on the same side of each split. Bipolar EDF channels are reordered by channel name before preprocessing.
- Attention testing uses two complete leave-one-subject-out cycles across the five participants, so every participant is tested exactly twice and recordings from one participant never appear on both sides of a split.
- xDAWN is fitted on training data and only transformed on test data. The label-leaking legacy branch is disabled.
- BCI SVM uses a linear kernel with probability estimates, matching the author repository. The CPU implementation is scikit-learn so the package does not require RAPIDS/cuML.
- EEGNet uses a consistent channels-last layout, batch size 32, and the same `val_loss` monitor for checkpointing and early stopping. Validation remains subject-disjoint: Attention and Seizure use the next fold in each five-fold outer cycle, giving every subject exactly two test, two validation, and six gradient-training roles across ten repeats. BCI keeps its official test subjects fixed and rotates the 16 training subjects through four validation folds, giving each subject two or three validation roles.
- BCI ASR filters each complete session to 1--40 Hz, calibrates and applies ASR once to that session, corrects ASR's 0.25-second look-ahead offset, and only then slices event epochs and subtracts their baselines. The final 0.25 seconds use the zero-padding behavior of the official `asrpy` high-level transform.
- Attention and Seizure ASR apply a zero-phase 1 Hz high-pass to each complete recording before calibration and cleanup, as required by `asrpy`. ASR uses the library default cutoff of 20 rather than the older, more aggressive clean_rawdata value of 5.
- CHB-MIT is stored in volts; only the Seizure EEGNet input layer applies a fixed `1e6` V-to-microvolt conversion. This is identical for every denoising method and does not use fitted dataset statistics.
- The included IC-U-Net weight is `BEST_checkpoint.pth.tar` from the author BCI repository.
- Seizure and Attention classical models standardize STFT features using training data only. This resolves the severe scale sensitivity observed for LR, SVM, and MLP while leaving tree models unscaled.
- Attention ASR and filtering operate on each complete 30-minute recording before 5-second windowing. IC-U-Net uses overlap-add chunks over the complete recording instead of treating every 5-second window as an independent recording.

## Important scope note

The BCI implementation is directly grounded in the recovered author repository. The report does not provide the original Seizure and Attention source code, so those two tasks remain best-supported reconstructions of the written methods rather than byte-for-byte recovery of the authors' hidden scripts.
