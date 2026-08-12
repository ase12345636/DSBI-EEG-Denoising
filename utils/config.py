"""Configuration loader with paths resolved from the project root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    pipeline_version: str
    tasks: tuple[str, ...]
    methods: tuple[str, ...]
    classifiers: tuple[str, ...]
    repeats: int
    output_dir: Path
    cache_dir: Path
    stft: dict[str, Any]
    eegnet: dict[str, Any]
    asr: dict[str, Any]
    ic_unet: dict[str, Any]
    ica: dict[str, Any]
    calibration: dict[str, Any]


def load_config(path: Path, root: Path) -> ProjectConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ProjectConfig(
        root=root,
        pipeline_version=str(raw.get("pipeline_version", "unknown")),
        tasks=tuple(raw["tasks"]),
        methods=tuple(raw["methods"]),
        classifiers=tuple(raw["classifiers"]),
        repeats=int(raw.get("repeats", 10)),
        output_dir=root / raw.get("output_dir", "output"),
        cache_dir=root / raw.get("cache_dir", ".cache"),
        stft=dict(raw.get("stft", {})),
        eegnet=dict(raw.get("eegnet", {})),
        asr=dict(raw.get("asr", {})),
        ic_unet=dict(raw.get("ic_unet", {})),
        ica=dict(raw.get("ica", {})),
        calibration=dict(raw.get("calibration", {})),
    )
