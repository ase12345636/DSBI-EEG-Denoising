"""IC-U-Net inference; BCI path follows the author's preprocessing notebook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from denoise.ic_unet.model import UNet1
from utils.progress import progress


class ICUNetDenoiser:
    name = "ic_unet"
    target_rate = 256

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

    @staticmethod
    def _fir(signal: np.ndarray, highcut: float) -> np.ndarray:
        coeff = firwin(1000, [1.0, highcut], pass_zero=False, fs=256.0)
        return lfilter(coeff, 1.0, signal, axis=-1)

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

    def _bci_epoch(self, epoch: np.ndarray, checkpoint_path: Path) -> np.ndarray:
        """Match preprocessing_data.ipynb, including its 140 -> 136 sample output."""
        model, device = self._load_model(checkpoint_path)
        data, channels = self._pad_channels(epoch)

        # The author's resample() uses p=int(256/200)=1, so 200-Hz BCI epochs
        # are effectively not resampled before the 256-Hz FIR/model path.
        data = self._fir(data, 40.0)
        decoded = self._decode(model, device, data)
        return decoded[:channels].astype(np.float32)

    def _seizure_epoch(self, epoch: np.ndarray, checkpoint_path: Path) -> np.ndarray:
        """Keep the original integrated CHB-MIT reproduction path."""
        model, device = self._load_model(checkpoint_path)
        original_samples = epoch.shape[-1]
        data, channels = self._pad_channels(epoch)
        data = np.concatenate([data] * 8, axis=-1)
        data = self._fir(data, 50.0)

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

    def _generic_epoch(self, epoch: np.ndarray, sampling_rate: float,
                       checkpoint_path: Path) -> np.ndarray:
        model, device = self._load_model(checkpoint_path)
        original_samples = epoch.shape[-1]
        data, channels = self._pad_channels(epoch)
        if sampling_rate != 256:
            data = resample_poly(data, 256, int(round(sampling_rate)), axis=-1)
        data = self._fir(data, 50.0)
        decoded = self._decode(model, device, data)[:channels]
        if sampling_rate != 256:
            decoded = resample_poly(decoded, int(round(sampling_rate)), 256, axis=-1)
        if decoded.shape[-1] < original_samples:
            decoded = np.pad(decoded, ((0, 0), (0, original_samples - decoded.shape[-1])), mode="edge")
        return decoded[:, :original_samples].astype(np.float32)

    def transform(self, signals: np.ndarray, sampling_rate: float, checkpoint_path: Path,
                  task_name: str | None = None, **_) -> np.ndarray:
        output = []
        for epoch in progress(
            signals,
            total=len(signals),
            desc=f"IC-U-Net {task_name or 'EEG'}",
            unit="epoch",
            leave=False,
        ):
            if task_name == "bci_errp":
                cleaned = self._bci_epoch(epoch, checkpoint_path)
            elif task_name == "seizure_detection":
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
        """Long-recording path retained for the Attention reproduction."""
        model, device = self._load_model(checkpoint_path)
        original_samples = signal.shape[-1]
        data, channels = self._pad_channels(signal)
        if sampling_rate != 256:
            data = resample_poly(data, 256, int(round(sampling_rate)), axis=-1)
        data = self._fir(data, 50.0)
        normalized, std = self._normalise(data)

        chunk = int(chunk_seconds * 256)
        chunk -= chunk % 8
        overlap = min(int(overlap_seconds * 256), chunk // 2)
        step = chunk - overlap
        total = normalized.shape[-1]

        if total <= chunk:
            decoded = self._infer(model, device, normalized)
        else:
            summed = np.zeros_like(normalized)
            weights = np.zeros(total, dtype=np.float64)
            starts = list(range(0, max(total - chunk + 1, 1), step))
            if not starts or starts[-1] + chunk < total:
                starts.append(max(0, total - chunk))
            for start in progress(starts, total=len(starts), desc="IC-U-Net Attention", unit="chunk", leave=False):
                piece = normalized[:, start : start + chunk]
                estimate = self._infer(model, device, piece)
                length = estimate.shape[-1]
                window = np.maximum(np.hanning(length), 0.05)
                if start == 0:
                    window[:min(overlap, length)] = 1.0
                if start + length >= total:
                    window[max(0, length - overlap):] = 1.0
                summed[:, start : start + length] += estimate * window
                weights[start : start + length] += window
            decoded = summed / np.maximum(weights, 1e-8)[None, :]

        if std > 0:
            decoded *= std
        decoded = decoded[:channels]
        if sampling_rate != 256:
            decoded = resample_poly(decoded, int(round(sampling_rate)), 256, axis=-1)
        if decoded.shape[-1] < original_samples:
            decoded = np.pad(decoded, ((0, 0), (0, original_samples - decoded.shape[-1])), mode="edge")
        return decoded[:, :original_samples].astype(np.float32)


DENOISER = ICUNetDenoiser()
