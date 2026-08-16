#!/usr/bin/env python3
"""Command-line entry point for the EEG benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from utils.config import load_config
from utils.pipeline import ReproductionPipeline


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the EEG datasets and run the benchmark. Running without "
            "arguments executes all 1440 model runs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.json",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Subset: bci_errp seizure_detection attention_state",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Subset: raw bandpass asr asr20 ic_unet ica",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        help="Subset of classifier registry names",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run written to output/quick_smoke; not a report result",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selections and print work without downloading",
    )
    parser.add_argument("--list", action="store_true", help="List registered components")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, ROOT)
    pipeline = ReproductionPipeline(config)
    if args.list:
        pipeline.print_registry()
        return 0
    return pipeline.run(
        tasks=args.tasks,
        methods=args.methods,
        classifiers=args.classifiers,
        download_only=args.download_only,
        skip_download=args.skip_download,
        force_download=args.force_download,
        quick=args.quick,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
