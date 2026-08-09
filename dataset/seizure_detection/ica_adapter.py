"""CHB-MIT bipolar-montage adapter for the generic ICA denoiser.

CHB-MIT stores bipolar derivations such as ``Fp1-F7`` rather than monopolar
scalp electrode potentials. ICLabel requires positioned scalp channels, so the
bipolar differences are represented as an incidence matrix ``B`` and converted
to a minimum-norm monopolar representative with ``pinv(B)``. ICA is run in that
monopolar space and the cleaned signals are projected back with the same
incidence matrix.

Unlike ICA v3, v4 does NOT discard smaller disconnected electrode graphs. All
parseable bipolar derivations are included in one block-disconnected incidence
matrix, so every valid downstream bipolar channel receives the same ICA
intervention. Disconnected graphs have independent additive-reference freedoms;
the Moore-Penrose solution fixes each to its minimum-norm representative, while
ICA fitting itself is high-pass filtered at 1 Hz by the generic denoiser.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np


def _standard_scalp_names() -> set[str]:
    """Return normalized scalp-electrode names available in MNE standard_1020.

    Auxiliary CHB-MIT channels such as LOC/ROC are not scalp EEG electrodes and
    must not be presented to ICLabel as EEG channels with invented positions.
    """
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    return {str(name).casefold() for name in montage.ch_names}


def _parse_bipolar_pair(denoiser, channel_name: str) -> tuple[str, str] | None:
    value = str(channel_name).strip()
    value = re.sub(r"^EEG\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"-(REF|LE|RE)$", "", value, flags=re.IGNORECASE)
    # MNE appends -0/-1/... to duplicate EDF channel labels.
    value = re.sub(r"-(\d+)$", "", value)
    parts = [part for part in value.replace(" ", "").split("-") if part]
    if len(parts) != 2:
        return None
    first = denoiser.normalise_channel_name(parts[0])
    second = denoiser.normalise_channel_name(parts[1])
    if first.casefold() == second.casefold():
        return None
    return first, second


def _connected_component_count(
    electrodes: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> int:
    adjacency = {name.casefold(): set() for name in electrodes}
    for first, second in pairs:
        first_key = first.casefold()
        second_key = second.casefold()
        adjacency[first_key].add(second_key)
        adjacency[second_key].add(first_key)

    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        start = next(iter(remaining))
        stack = [start]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)
        remaining -= visited
    return count


def clean_bipolar_recording(
    denoiser,
    signal: np.ndarray,
    sampling_rate: float,
    *,
    channel_names: Sequence[str],
    task_name: str | None = None,
    recording_id: str | None = None,
) -> np.ndarray:
    """Apply generic ICA/ICLabel to every parseable CHB-MIT bipolar channel."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(channel_names):
        raise ValueError(
            f"Bipolar signal/name mismatch: {values.shape} vs {len(channel_names)}"
        )

    scalp_names = _standard_scalp_names()
    parsed: list[tuple[int, str, str]] = []
    electrode_order: list[str] = []
    seen: set[str] = set()
    skipped_non_scalp: list[str] = []
    skipped_unparseable: list[str] = []
    for index, name in enumerate(channel_names):
        pair = _parse_bipolar_pair(denoiser, name)
        if pair is None:
            skipped_unparseable.append(str(name))
            continue
        first, second = pair

        # ICLabel requires positioned scalp EEG channels. CHB-MIT EDFs can also
        # contain auxiliary bipolar channels such as LOC-ROC. Do not pretend
        # these are scalp EEG electrodes: leave those original bipolar channels
        # untouched. In the seizure task they are auxiliary channels removed by
        # the existing downstream 23-channel normalization.
        if first.casefold() not in scalp_names or second.casefold() not in scalp_names:
            skipped_non_scalp.append(str(name))
            continue

        parsed.append((index, first, second))
        for electrode in pair:
            key = electrode.casefold()
            if key not in seen:
                seen.add(key)
                electrode_order.append(electrode)

    if len(parsed) < 4 or len(electrode_order) < 5:
        raise ValueError(
            "Could not recover enough valid bipolar EEG derivations for ICA/ICLabel: "
            f"{channel_names}"
        )

    electrode_index = {
        name.casefold(): index for index, name in enumerate(electrode_order)
    }
    incidence = np.zeros(
        (len(parsed), len(electrode_order)),
        dtype=np.float64,
    )
    bipolar_indices: list[int] = []
    graph_pairs: list[tuple[str, str]] = []
    for row, (channel_index, first, second) in enumerate(parsed):
        incidence[row, electrode_index[first.casefold()]] = 1.0
        incidence[row, electrode_index[second.casefold()]] = -1.0
        bipolar_indices.append(channel_index)
        graph_pairs.append((first, second))

    bipolar = values[np.asarray(bipolar_indices)]
    monopolar = np.linalg.pinv(incidence, rcond=1e-10) @ bipolar

    # Diagnostic only: for consistent bipolar differences, B @ pinv(B) @ x
    # should reproduce x up to numerical precision (or least-squares error for
    # duplicated/inconsistent derivations).
    reconstructed_bipolar = incidence @ monopolar
    denominator = max(float(np.linalg.norm(bipolar)), np.finfo(float).eps)
    relative_reconstruction_error = float(
        np.linalg.norm(reconstructed_bipolar - bipolar) / denominator
    )

    cleaned_monopolar = denoiser.transform_recording(
        monopolar,
        sampling_rate,
        channel_names=electrode_order,
        unit_scale_to_volts=1.0,
        task_name=task_name,
        recording_id=recording_id,
    )
    cleaned_bipolar = incidence @ cleaned_monopolar

    output = values.copy()
    output[np.asarray(bipolar_indices)] = cleaned_bipolar

    denoiser.update_last_report(
        {
            "bipolar_adapter": {
                "version": "all-graphs-v4",
                "input_channels": len(channel_names),
                "parseable_scalp_bipolar_channels": len(parsed),
                "cleaned_bipolar_channels": len(parsed),
                "unchanged_non_scalp_channels": len(skipped_non_scalp),
                "unchanged_unparseable_channels": len(skipped_unparseable),
                "skipped_non_scalp_channel_names": skipped_non_scalp,
                "skipped_unparseable_channel_names": skipped_unparseable,
                "connected_electrode_graphs": _connected_component_count(
                    electrode_order,
                    graph_pairs,
                ),
                "reconstructed_electrodes": electrode_order,
                "incidence_rank": int(np.linalg.matrix_rank(incidence)),
                "relative_reconstruction_error": relative_reconstruction_error,
            }
        }
    )
    return output.astype(np.float32)
