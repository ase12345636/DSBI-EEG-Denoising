"""IC-U-Net inference adapters for the downstream EEG tasks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import firwin, lfilter, resample_poly

from denoise.ic_unet.model import UNet1
from utils.progress import progress


class ICUNetDenoiser:
    name = "ic_unet"
    target_rate = 256.0
    lowcut = 1.0
    highcut = 50.0
    fir_numtaps = 1000
    template_channels = (
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT7", "FC3", "FCz",
        "FC4", "FT8", "T7", "C3", "Cz", "C4", "T8", "TP7", "CP3", "CPz",
        "CP4", "TP8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
    )

    def __init__(self) -> None:
        self._model = None
        self._model_path = None
        self._device = None

    def _load_model(self, checkpoint_path: Path):
        import torch

        checkpoint_path = Path(checkpoint_path).resolve()
        if self._model is not None and self._model_path == checkpoint_path:
            return self._model, self._device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet1(n_channels=30, n_classes=30)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        self._model, self._model_path, self._device = model, checkpoint_path, device
        return model, device

    @classmethod
    def _channel_coordinates(cls, names: Sequence[str]) -> np.ndarray:
        """Return standard-10/20 coordinates for channel names."""
        import mne

        positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
        lookup = {name.casefold(): np.asarray(value, dtype=np.float64) for name, value in positions.items()}
        coordinates = []
        missing = []
        for name in names:
            key = str(name).strip().casefold()
            if key not in lookup:
                missing.append(str(name))
            else:
                coordinates.append(lookup[key])
        if missing:
            raise ValueError(
                "IC-U-Net channel mapping requires standard scalp locations; missing: "
                + ", ".join(missing)
            )
        return np.stack(coordinates)

    @classmethod
    def _map_channels(
        cls,
        signal: np.ndarray,
        channel_names: Sequence[str] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map input scalp channels to the 30-channel IC-U-Net template.

        Exact template-name matches are used first. Remaining input channels are
        assigned one-to-one to the nearest unfilled template locations. Any still
        missing template channel is imputed by the mean of its four nearest input
        scalp channels. The returned index map restores the reconstructed template
        signal to the original input channel order after inference.
        """
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"IC-U-Net expects a 2-D channels x time array, got {values.shape}")

        if channel_names is None:
            if values.shape[0] != 30:
                raise ValueError(
                    "channel_names are required when IC-U-Net input does not already have 30 channels"
                )
            return values, np.arange(30, dtype=int)

        names = tuple(str(name).strip() for name in channel_names)
        if len(names) != values.shape[0]:
            raise ValueError(
                f"IC-U-Net received {values.shape[0]} channels but {len(names)} channel names"
            )

        template = cls.template_channels
        template_lookup = {name.casefold(): index for index, name in enumerate(template)}
        input_coordinates = cls._channel_coordinates(names)
        template_coordinates = cls._channel_coordinates(template)

        mapped = np.empty((30, values.shape[1]), dtype=np.float64)
        assigned_template: dict[int, int] = {}
        used_template = set()
        unmatched_inputs = []

        for input_index, name in enumerate(names):
            template_index = template_lookup.get(name.casefold())
            if template_index is not None and template_index not in used_template:
                assigned_template[input_index] = template_index
                used_template.add(template_index)
            else:
                unmatched_inputs.append(input_index)

        remaining_templates = [index for index in range(30) if index not in used_template]
        if unmatched_inputs:
            if len(unmatched_inputs) > len(remaining_templates):
                raise ValueError("More unmatched input channels than available IC-U-Net template channels")
            cost = np.linalg.norm(
                input_coordinates[np.asarray(unmatched_inputs)][:, None, :]
                - template_coordinates[np.asarray(remaining_templates)][None, :, :],
                axis=2,
            )
            row_indices, col_indices = linear_sum_assignment(cost)
            for row_index, col_index in zip(row_indices, col_indices):
                input_index = unmatched_inputs[int(row_index)]
                template_index = remaining_templates[int(col_index)]
                assigned_template[input_index] = template_index
                used_template.add(template_index)

        for input_index, template_index in assigned_template.items():
            mapped[template_index] = values[input_index]

        # The released CNElab channel-mapping interface supports mean-based
        # imputation and uses nearby input channels for missing template sites.
        # Use the four nearest available scalp channels for each missing site.
        for template_index in range(30):
            if template_index in used_template:
                continue
            distances = np.linalg.norm(
                input_coordinates - template_coordinates[template_index],
                axis=1,
            )
            nearest = np.argsort(distances)[: min(4, len(names))]
            mapped[template_index] = values[nearest].mean(axis=0)

        restore_indices = np.asarray(
            [assigned_template[index] for index in range(len(names))], dtype=int
        )
        return mapped, restore_indices

    @classmethod
    def _fir(cls, signal: np.ndarray) -> np.ndarray:
        coeff = firwin(
            cls.fir_numtaps,
            [cls.lowcut, cls.highcut],
            pass_zero=False,
            fs=cls.target_rate,
        )
        return lfilter(coeff, 1.0, signal, axis=-1)

    @staticmethod
    def _resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
        if np.isclose(source_rate, target_rate):
            return np.asarray(signal, dtype=np.float64)
        ratio = (Fraction(str(target_rate)) / Fraction(str(source_rate))).limit_denominator(10000)
        return resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)

    @staticmethod
    def _match_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
        if signal.shape[-1] < target_samples:
            signal = np.pad(
                signal,
                ((0, 0), (0, target_samples - signal.shape[-1])),
                mode="edge",
            )
        return signal[:, :target_samples]

    @staticmethod
    def _normalise(signal: np.ndarray) -> tuple[np.ndarray, float]:
        std = float(np.std(signal))
        mean = float(np.mean(signal))
        if std == 0:
            return signal - mean, 0.0
        return (signal - mean) / std, std

    @staticmethod
    def _infer(model, device, signal: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(signal[np.newaxis], dtype=torch.float32, device=device)
            return model(tensor).detach().cpu().numpy()[0].astype(np.float64)

    def _decode_block(self, model, device, signal: np.ndarray) -> np.ndarray:
        """Normalize one model block, infer it, and restore its amplitude scale."""
        normalized, std = self._normalise(signal)
        decoded = self._infer(model, device, normalized)
        if std > 0:
            decoded *= std
        return decoded

    def _generic_epoch(
        self,
        epoch: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        channel_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        model, device = self._load_model(checkpoint_path)
        original_samples = epoch.shape[-1]
        data, restore_indices = self._map_channels(epoch, channel_names)
        data = self._resample(data, sampling_rate, self.target_rate)
        data = self._fir(data)

        model_samples = data.shape[-1]
        padded_samples = int(np.ceil(model_samples / 8.0) * 8)
        if padded_samples != model_samples:
            data = np.pad(data, ((0, 0), (0, padded_samples - model_samples)), mode="edge")

        decoded = self._decode_block(model, device, data)[:, :model_samples]
        decoded = decoded[restore_indices]
        decoded = self._resample(decoded, self.target_rate, sampling_rate)
        decoded = self._match_length(decoded, original_samples)
        return decoded.astype(np.float32)

    def transform(
        self,
        signals: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        task_name: str | None = None,
        channel_names: Sequence[str] | None = None,
        **_,
    ) -> np.ndarray:
        output = []
        for epoch in progress(
            signals,
            total=len(signals),
            desc=f"IC-U-Net {task_name or 'EEG'}",
            unit="epoch",
            leave=False,
        ):
            output.append(
                self._generic_epoch(
                    epoch,
                    sampling_rate,
                    checkpoint_path,
                    channel_names=channel_names,
                )
            )
        return np.stack(output)

    def transform_recording(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        *,
        channel_names: Sequence[str] | None = None,
        task_name: str | None = None,
        chunk_seconds: int = 4,
        overlap_seconds: int = 0,
    ) -> np.ndarray:
        """Denoise a continuous scalp recording and return it at its native rate."""
        model, device = self._load_model(checkpoint_path)
        original_samples = signal.shape[-1]
        data, restore_indices = self._map_channels(signal, channel_names)

        data = self._resample(data, sampling_rate, self.target_rate)
        data = self._fir(data)

        chunk = int(chunk_seconds * self.target_rate)
        chunk -= chunk % 8
        if chunk <= 0:
            raise ValueError("IC-U-Net chunk duration is too short")
        overlap = min(int(overlap_seconds * self.target_rate), chunk // 2)
        step = chunk - overlap
        total = data.shape[-1]

        if overlap == 0:
            decoded = np.zeros_like(data)
            starts = list(range(0, total, chunk))
            for start in progress(
                starts,
                total=len(starts),
                desc=f"IC-U-Net {task_name or 'recording'}",
                unit="chunk",
                leave=False,
            ):
                stop = min(start + chunk, total)
                length = stop - start
                piece = data[:, start:stop]
                if length < chunk:
                    piece = np.pad(piece, ((0, 0), (0, chunk - length)), mode="edge")
                estimate = self._decode_block(model, device, piece)[:, :length]

                # Match the released reconstruction step at block boundaries.
                if start > 0 and length > 1:
                    smooth = (decoded[:, start - 1] + estimate[:, 1]) / 2.0
                    decoded[:, start - 1] = smooth
                    estimate[:, 1] = smooth
                decoded[:, start:stop] = estimate
        elif total <= chunk:
            padded_total = int(np.ceil(total / 8.0) * 8)
            piece = data
            if padded_total != total:
                piece = np.pad(piece, ((0, 0), (0, padded_total - total)), mode="edge")
            decoded = self._decode_block(model, device, piece)[:, :total]
        else:
            summed = np.zeros_like(data)
            weights = np.zeros(total, dtype=np.float64)
            starts = list(range(0, max(total - chunk + 1, 1), step))
            if not starts or starts[-1] + chunk < total:
                starts.append(max(0, total - chunk))
            for start in progress(
                starts,
                total=len(starts),
                desc=f"IC-U-Net {task_name or 'recording'}",
                unit="chunk",
                leave=False,
            ):
                piece = data[:, start : start + chunk]
                estimate = self._decode_block(model, device, piece)
                length = min(estimate.shape[-1], piece.shape[-1])
                estimate = estimate[:, :length]
                window = np.maximum(np.hanning(length), 0.05)
                if start == 0:
                    window[: min(overlap, length)] = 1.0
                if start + length >= total:
                    window[max(0, length - overlap) :] = 1.0
                summed[:, start : start + length] += estimate * window
                weights[start : start + length] += window
            decoded = summed / np.maximum(weights, 1e-8)[None, :]

        decoded = decoded[restore_indices]
        decoded = self._resample(decoded, self.target_rate, sampling_rate)
        decoded = self._match_length(decoded, original_samples)
        return decoded.astype(np.float32)

    @staticmethod
    def _bipolar_pair(channel_name: str) -> tuple[str, str]:
        parts = [part for part in str(channel_name).strip().replace(" ", "").split("-") if part]
        if len(parts) != 2:
            raise ValueError(f"Invalid bipolar channel name for IC-U-Net: {channel_name}")
        return parts[0], parts[1]

    def transform_bipolar_recording(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        *,
        channel_names: Sequence[str],
        task_name: str | None = None,
    ) -> np.ndarray:
        """Apply IC-U-Net through a scalp representation of bipolar EEG.

        A least-squares scalp representation is used only as an adapter. After
        denoising, only the IC-U-Net-induced correction is mapped back to the
        original bipolar derivations so the adapter itself does not replace the
        measured bipolar signal by its projection.
        """
        values = np.asarray(signal, dtype=np.float64)
        pairs = [self._bipolar_pair(name) for name in channel_names]
        electrodes: list[str] = []
        seen = set()
        for first, second in pairs:
            for electrode in (first, second):
                key = electrode.casefold()
                if key not in seen:
                    seen.add(key)
                    electrodes.append(electrode)

        index = {name.casefold(): i for i, name in enumerate(electrodes)}
        incidence = np.zeros((len(pairs), len(electrodes)), dtype=np.float64)
        for row, (first, second) in enumerate(pairs):
            incidence[row, index[first.casefold()]] = 1.0
            incidence[row, index[second.casefold()]] = -1.0

        scalp = np.linalg.pinv(incidence) @ values
        cleaned_scalp = self.transform_recording(
            scalp,
            sampling_rate,
            checkpoint_path,
            channel_names=electrodes,
            task_name=task_name,
            chunk_seconds=4,
            overlap_seconds=0,
        )
        correction = incidence @ (cleaned_scalp - scalp)
        return (values + correction).astype(np.float32)


DENOISER = ICUNetDenoiser()
