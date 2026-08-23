# Pending rerun

`all_runs.partial.csv` contains the results that remain valid after the requested protocol changes.
Running `python main.py` will resume from this file.

Pending configurations: **390 runs** out of 1,440 total.

They consist of:
- all BCI IC-U-Net runs, because the IC-U-Net preprocessing adapter changed;
- BCI EEGNet, ViT, and MobileNet runs under all preprocessing conditions, because the internal validation fraction changed from 25% to 20%;
- all Attention IC-U-Net runs, because IC-U-Net now uses the 30-channel template mapping and per-4-s-block inference preprocessing;
- all Seizure IC-U-Net runs, because bipolar EEG is now adapted to the IC-U-Net scalp template and processed continuously in 4-s blocks.

No other completed runs were removed.
