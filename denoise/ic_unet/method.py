"""IC-U-Net inference adapters for the three downstream EEG tasks."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from denoise.ic_unet.model import UNet1
from utils.progress import progress


class ICUNetDenoiser:
    name = "ic_unet"
    target_rate = 256.0
    lowcut = 1.0
    highcut = 50.0
    fir_numtaps = 1000

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

    @staticmethod
    def _pad_channels(signal: np.ndarray) -> tuple[np.ndarray, int]:
        signal = np.asarray(signal, dtype=np.float64)
        original = signal.shape[0]
        if original < 30:
            signal = np.vstack([signal] + [signal[-1:]] * (30 - original))
        elif original > 30:
            signal = signal[:30]
        return signal, min(original, 30)

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

    def _decode(self, model, device, signal: np.ndarray) -> np.ndarray:
        normalized, std = self._normalise(signal)
        decoded = self._infer(model, device, normalized)
        if std > 0:
            decoded *= std
        return decoded

    def _seizure_epoch(self, epoch: np.ndarray, checkpoint_path: Path) -> np.ndarray:
        """Process a 1-second CHB-MIT epoch with the fixed 256-Hz model."""
        model, device = self._load_model(checkpoint_path)
        original_samples = epoch.shape[-1]
        data, channels = self._pad_channels(epoch)
        data = np.concatenate([data] * 8, axis=-1)
        data = self._fir(data)

        pieces = []
        for start in range(0, (data.shape[-1] // 1024) * 1024, 1024):
            pieces.append(self._decode(model, device, data[:, start : start + 1024]))
        if not pieces:
            raise ValueError("IC-U-Net seizure input did not produce a 1024-sample block")

        decoded = pieces[0]
        for piece in pieces[1:]:
            piece = piece.copy()
            smooth = (decoded[:, -1] + piece[:, 1]) / 2.0
            decoded[:, -1] = smooth
            piece[:, 1] = smooth
            decoded = np.concatenate([decoded, piece], axis=-1)
        return decoded[:channels, -original_samples:].astype(np.float32)

    def _generic_epoch(
        self,
        epoch: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
    ) -> np.ndarray:
        model, device = self._load_model(checkpoint_path)
        original_samples = epoch.shape[-1]
        data, channels = self._pad_channels(epoch)
        data = self._resample(data, sampling_rate, self.target_rate)
        data = self._fir(data)

        # The network downsamples by 2 three times. Pad only the time axis so
        # inference preserves the complete resampled epoch, then crop the pad.
        model_samples = data.shape[-1]
        padded_samples = int(np.ceil(model_samples / 8.0) * 8)
        if padded_samples != model_samples:
            data = np.pad(data, ((0, 0), (0, padded_samples - model_samples)), mode="edge")

        decoded = self._decode(model, device, data)[:channels, :model_samples]
        decoded = self._resample(decoded, self.target_rate, sampling_rate)
        decoded = self._match_length(decoded, original_samples)
        return decoded.astype(np.float32)

    def transform(
        self,
        signals: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        task_name: str | None = None,
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
            if task_name == "seizure_detection":
                cleaned = self._seizure_epoch(epoch, checkpoint_path)
            else:
                cleaned = self._generic_epoch(epoch, sampling_rate, checkpoint_path)
            output.append(cleaned)
        return np.stack(output)

    def transform_recording(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        *,
        task_name: str | None = None,
        chunk_seconds: int = 30,
        overlap_seconds: int = 2,
    ) -> np.ndarray:
        """Denoise a continuous recording and return it at its native rate."""
        model, device = self._load_model(checkpoint_path)
        original_samples = signal.shape[-1]
        data, channels = self._pad_channels(signal)

        # The public IC-U-Net preprocessing operates in a 256-Hz, 1-50-Hz
        # signal domain. Exact rational resampling is used for non-256-Hz data.
        data = self._resample(data, sampling_rate, self.target_rate)
        data = self._fir(data)
        normalized, std = self._normalise(data)

        chunk = int(chunk_seconds * self.target_rate)
        chunk -= chunk % 8
        if chunk <= 0:
            raise ValueError("IC-U-Net chunk duration is too short")
        overlap = min(int(overlap_seconds * self.target_rate), chunk // 2)
        step = chunk - overlap
        total = normalized.shape[-1]

        if overlap == 0:
            # The public IC-U-Net pipeline uses non-overlapping 4-s / 1024-point
            # blocks. Pad only the final incomplete block and crop it afterward
            # so no native samples are discarded.
            decoded = np.zeros_like(normalized)
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
                piece = normalized[:, start:stop]
                if length < chunk:
                    piece = np.pad(piece, ((0, 0), (0, chunk - length)), mode="edge")
                estimate = self._infer(model, device, piece)[:, :length]
                decoded[:, start:stop] = estimate
        elif total <= chunk:
            padded_total = int(np.ceil(total / 8.0) * 8)
            piece = normalized
            if padded_total != total:
                piece = np.pad(piece, ((0, 0), (0, padded_total - total)), mode="edge")
            decoded = self._infer(model, device, piece)[:, :total]
        else:
            summed = np.zeros_like(normalized)
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
                piece = normalized[:, start : start + chunk]
                estimate = self._infer(model, device, piece)
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

        if std > 0:
            decoded *= std
        decoded = decoded[:channels]
        decoded = self._resample(decoded, self.target_rate, sampling_rate)
        decoded = self._match_length(decoded, original_samples)
        return decoded.astype(np.float32)


DENOISER = ICUNetDenoiser()
