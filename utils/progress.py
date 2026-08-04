"""Consistent terminal progress bars for downloads and experiment stages."""

from __future__ import annotations

import os
from typing import Any

from tqdm.auto import tqdm


def progress(iterable=None, **kwargs: Any):
    """Create a progress bar with project-wide display defaults.

    Set ``EEG_PROGRESS=0`` to disable bars in non-interactive automation.
    """
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.25)
    kwargs.setdefault("smoothing", 0.1)
    kwargs.setdefault("disable", os.environ.get("EEG_PROGRESS", "1") == "0")
    return tqdm(iterable, **kwargs)


def progress_write(message: str) -> None:
    if os.environ.get("EEG_PROGRESS", "1") == "0":
        print(message)
    else:
        tqdm.write(message)
