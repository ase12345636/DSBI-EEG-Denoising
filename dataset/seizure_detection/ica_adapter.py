"""Convert CHB-MIT bipolar derivations to a scalp representation for ICA."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Sequence

import numpy as np


@lru_cache(maxsize=1)
def _standard_scalp_names() -> set[str]:
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    return {name.casefold() for name in montage.ch_names}


def _pair(denoiser, channel_name: str) -> tuple[str, str] | None:
    value = str(channel_name).strip()
    value = re.sub(r"^EEG\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-(REF|LE|RE)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-(\d+)$", "", value)  # duplicate EDF label suffix
    parts = [part for part in value.replace(" ", "").split("-") if part]
    if len(parts) != 2:
        return None
    first = denoiser.normalise_channel_name(parts[0])
    second = denoiser.normalise_channel_name(parts[1])
    return None if first.casefold() == second.casefold() else (first, second)


def clean_bipolar_recording(
    denoiser,
    signal: np.ndarray,
    sampling_rate: float,
    *,
    channel_names: Sequence[str],
    task_name: str | None = None,
    recording_id: str | None = None,
) -> np.ndarray:
    """ICA-clean valid scalp bipolar channels; leave auxiliary channels untouched."""
    values = np.asarray(signal, dtype=np.float64)
    scalp = _standard_scalp_names()
    parsed = []
    electrodes = []
    seen = set()

    for channel_index, name in enumerate(channel_names):
        pair = _pair(denoiser, name)
        if pair is None or any(e.casefold() not in scalp for e in pair):
            continue
        parsed.append((channel_index, *pair))
        for electrode in pair:
            if electrode.casefold() not in seen:
                seen.add(electrode.casefold())
                electrodes.append(electrode)

    if len(parsed) < 4 or len(electrodes) < 5:
        raise ValueError("Not enough scalp bipolar channels for ICA")

    index = {name.casefold(): i for i, name in enumerate(electrodes)}
    incidence = np.zeros((len(parsed), len(electrodes)), dtype=np.float64)
    bipolar_indices = []
    for row, (channel_index, first, second) in enumerate(parsed):
        incidence[row, index[first.casefold()]] = 1.0
        incidence[row, index[second.casefold()]] = -1.0
        bipolar_indices.append(channel_index)

    bipolar = values[bipolar_indices]
    monopolar = np.linalg.pinv(incidence) @ bipolar
    cleaned_monopolar = denoiser.transform_recording(
        monopolar,
        sampling_rate,
        channel_names=electrodes,
        unit_scale_to_volts=1.0,  # MNE EDF reader already returns volts
        task_name=task_name,
        recording_id=recording_id,
    )

    # The least-squares bipolar->monopolar conversion is only an ICA adapter.
    # Do not replace the original bipolar data by its cycle-consistent
    # projection, because that would modify CHB-MIT even when ICA excludes no
    # component. Map only the ICA-induced change back to the original channels.
    correction = incidence @ (cleaned_monopolar - monopolar)
    output = values.copy()
    output[bipolar_indices] = bipolar + correction
    return output.astype(np.float32)
