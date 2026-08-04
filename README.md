# EEG denoising downstream-task reproduction — revised build

This package runs the three report tasks with one entry point:

```bash
python main.py
```

It downloads the public data, prepares Raw / Band-pass / ASR / IC-U-Net variants, runs six classifiers for ten seeds, and writes Tables 5–10, Figures 6–14, run-level CSV files, and Mann–Whitney tests into `output/`.

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
python main.py --quick
python main.py --dry-run
```

A stopped full run resumes automatically from `output/all_runs.partial.csv`. This revised build records a pipeline signature. Old outputs created by an incompatible code/configuration are moved to a timestamped backup instead of being silently reused. Cache filenames are versioned, so old signal and STFT caches are not reused.

## Selected implementation choices

- BCI uses the author repository's fixed 16-subject training / 10-subject testing split, 30 channels, -100 to 600 ms epochs, xDAWN covariance (`nfilter=5`), tangent-space features, and training-only SMOTE.
- xDAWN is fitted on training data and only transformed on test data. The label-leaking legacy branch is disabled.
- BCI SVM uses a linear kernel with probability estimates, matching the author repository. The CPU implementation is scikit-learn so the package does not require RAPIDS/cuML.
- EEGNet uses a consistent channels-last layout, batch size 32, and the same `val_loss` monitor for checkpointing and early stopping.
- The included IC-U-Net weight is `BEST_checkpoint.pth.tar` from the author BCI repository.
- Seizure and Attention classical models standardize STFT features using training data only. This resolves the severe scale sensitivity observed for LR, SVM, and MLP while leaving tree models unscaled.
- Attention ASR and filtering operate on each complete 30-minute recording before 5-second windowing. IC-U-Net uses overlap-add chunks over the complete recording instead of treating every 5-second window as an independent recording.

## Important scope note

The BCI implementation is directly grounded in the recovered author repository. The report does not provide the original Seizure and Attention source code, so those two tasks remain best-supported reconstructions of the written methods rather than byte-for-byte recovery of the authors' hidden scripts. See `SOURCE_NOTES.md` for the exact boundary.
