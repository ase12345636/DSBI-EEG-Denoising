# EEG denoising downstream-task reproduction — revised build

This package runs the three report tasks with one entry point:

```bash
python main.py
```

It downloads the public data, prepares Raw / Band-pass / ASR (k=5 and k=20) / IC-U-Net / ICA variants, runs eight classifiers for ten seeds, and writes Tables 5–10, Figures 6–14, run-level CSV files, and paired Wilcoxon tests into `output/`. The primary test is two-sided with Holm correction; the directional one-sided test is supplementary.

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

A stopped run resumes automatically from `output/all_runs.partial.csv`.
Each new run creates a random repeat-seed plan; a resumed run reuses the saved
plan instead of fixing a project-wide seed. Attention and Seizure evaluate two
complete five-repeat split cycles, so every subject is assigned to a test fold
once per cycle. BCI keeps the authors' fixed train/test subject split.
Signal, feature, seed, and partial-result caches are all disposable: after changing
the code, configuration, checkpoint, or source data, remove `.cache` before the
next run. Final files in `output/` are never used as resume state. Runs over a
selected subset of tasks, methods, or classifiers still write the available
tables and figures; a complete 1440-run matrix is not required.

## Selected implementation choices

- BCI uses the author repository's fixed 16-subject training / 10-subject testing split, 30 channels, -100 to 600 ms epochs, xDAWN covariance (`nfilter=5`), tangent-space features, and training-only SMOTE.
- Seizure uses five-fold stratified patient-level splits. Bipolar EDF channels are reordered by channel name before preprocessing.
- Seizure selection uses all downloaded seizure-containing EDFs, maps every supported EDF layout to the same 18 bipolar derivations, retains every seizure second, and uses the experiment master seed to draw an equal-sized non-seizure sample list before any denoiser runs.
- Attention uses five-fold subject-level splits.
- xDAWN is fitted on training data and only transformed on test data. The label-leaking legacy branch is disabled.
- Every task records stable sample IDs, and Seizure creates the sample list before applying any denoiser.
- BCI SVM uses the configured cuML linear SVM with probability estimates.
- EEGNet uses a consistent channels-last layout and batch size 32. It checkpoints by `val_loss`, early-stops by `val_accuracy`, and uses a stratified validation split from the training set.
- ViT standardizes its time-domain input with training-set statistics, tokenizes 16-sample temporal patches spanning all EEG channels, and uses trainable positional embeddings with four transformer encoder blocks. MobileNet uses one-dimensional depthwise-separable convolutions along the time axis. Both use the same training protocol as EEGNet.
- ASR and ICA use the same deterministic, label-free calibration ranges: at most 300 seconds in ten uniformly spaced blocks for each continuous recording. The two ASR variants use cutoffs `k=5` and `k=20`; both use a 1 Hz high-pass processing copy, correct the library look-ahead through its high-level transform, and transfer only the ASR reconstruction to the original signal. BCI epochs are formed and baseline-corrected after this continuous-session processing.
- CHB-MIT is stored in volts. Classical STFT power features receive a fixed `1e12` V²-to-µV² scaling; EEGNet receives the same volt-scale signals for every method.
- The included IC-U-Net weight is `BEST_checkpoint.pth.tar` from the author BCI repository.
- Seizure and Attention classical models standardize STFT features using training data only. This resolves the severe scale sensitivity observed for LR, SVM, and MLP while leaving tree models unscaled.
- Attention denoising operates on each physical-unit 30-minute recording. The report Z-score is then applied identically after every method and before 5-second windowing. IC-U-Net uses overlap-add chunks over the complete recording.
- The report's unspecified STFT details are fixed explicitly as Hann windows, 256 samples, 50% overlap, zero boundary/padding, and mean squared magnitude over the time-frame axis (`mean_power_over_time`).

## Important scope note

The BCI implementation is directly grounded in the recovered author repository. The report does not provide the original Seizure and Attention source code, so those two tasks remain best-supported reconstructions of the written methods rather than byte-for-byte recovery of the authors' hidden scripts.
