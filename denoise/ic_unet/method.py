"""IC-U-Net denoising inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

from denoise.ic_unet.model import UNet1
from utils.progress import progress


class ICUNetDenoiser:
    name = "ic_unet"
    target_rate = 256
    target_channels = 30

    def __init__(self) -> None:
        self._model = None
        self._model_path: Path | None = None
        self._device = None

    def _load_model(self, checkpoint_path: Path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("IC-U-Net needs PyTorch; install requirements.txt") from exc

        checkpoint_path = Path(checkpoint_path).resolve()
        if self._model is not None and self._model_path == checkpoint_path:
            return self._model, self._device

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet1(n_channels=30, n_classes=30)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        if "state_dict" not in checkpoint:
            raise ValueError(f"IC-U-Net checkpoint has no state_dict: {checkpoint_path}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        model.eval()
        self._model = model
        self._model_path = checkpoint_path
        self._device = device
        return model, device

    @staticmethod
    def _fir_filter(signal: np.ndarray, highcut: float) -> np.ndarray:
        # IC-U-Net preprocessing uses fs=256, 1000 taps, and causal filtering.
        coefficients = firwin(
            1000,
            [1.0, highcut],
            pass_zero=False,
            fs=256.0,
        )
        return lfilter(coefficients, 1.0, signal, axis=-1)

    @staticmethod
    def _pad_channels(signal: np.ndarray) -> tuple[np.ndarray, int]:
        original = int(signal.shape[0])
        if original < 30:
            signal = np.vstack([signal] + [signal[-1:, :]] * (30 - original))
        elif original > 30:
            signal = signal[:30]
        return signal, min(original, 30)

    @staticmethod
    def _pad_samples_to_eight(signal: np.ndarray) -> tuple[np.ndarray, int]:
        original = int(signal.shape[-1])
        remainder = original % 8
        if remainder:
            pad = 8 - remainder
            mode = "reflect" if original > 1 else "edge"
            signal = np.pad(signal, ((0, 0), (0, pad)), mode=mode)
        return signal, original

    @staticmethod
    def _normalise(signal: np.ndarray) -> tuple[np.ndarray, float]:
        standard_deviation = float(np.std(signal))
        average = float(np.mean(signal))
        if standard_deviation > 0:
            return (signal - average) / standard_deviation, standard_deviation
        return signal - average, 0.0

    @staticmethod
    def _infer_array(model, device, signal: np.ndarray) -> np.ndarray:
        import torch

        padded, original_samples = ICUNetDenoiser._pad_samples_to_eight(signal)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                padded[np.newaxis],
                dtype=torch.float32,
                device=device,
            )
            decoded = model(tensor).detach().cpu().numpy()[0].astype(np.float64)
        return decoded[:, :original_samples]

    def _decode_source_chunk(self, model, device, signal: np.ndarray) -> np.ndarray:
        normalized, standard_deviation = self._normalise(signal)
        decoded = self._infer_array(model, device, normalized)
        if standard_deviation > 0:
            decoded *= standard_deviation
        return decoded

    def _process_epoch(
        self,
        epoch: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        task_name: str | None,
    ) -> np.ndarray:
        model, device = self._load_model(checkpoint_path)
        original_samples = int(epoch.shape[-1])
        data, original_channels = self._pad_channels(
            np.asarray(epoch, dtype=np.float64)
        )

        if task_name == "seizure_detection":
            if float(sampling_rate) != self.target_rate:
                raise ValueError(
                    "The CHB-MIT IC-U-Net pipeline expects 256 Hz input"
                )

            # Pad 23 channels to 30 and repeat each 1-second
            # segment eight times, applies a 1--50 Hz 1000-tap causal FIR,
            # then normalizes and decodes each 1024-sample block separately.
            data = np.concatenate([data] * 8, axis=-1)
            data = self._fir_filter(data, 50.0)
            usable = (data.shape[-1] // 1024) * 1024
            chunks = [
                self._decode_source_chunk(model, device, data[:, start : start + 1024])
                for start in range(0, usable, 1024)
            ]
            if not chunks:
                raise ValueError("CHB-MIT IC-U-Net input did not produce a 1024-sample block")

            decoded = chunks[0]
            for piece in chunks[1:]:
                piece = piece.copy()
                smooth = (decoded[:, -1] + piece[:, 1]) / 2.0
                decoded[:, -1] = smooth
                piece[:, 1] = smooth
                decoded = np.concatenate([decoded, piece], axis=-1)

            return decoded[:original_channels, -original_samples:].astype(np.float32)

        highcut = 40.0 if task_name == "bci_errp" else 50.0
        if float(sampling_rate) != self.target_rate:
            data = resample_poly(
                data,
                self.target_rate,
                int(round(sampling_rate)),
                axis=-1,
            )

        data = self._fir_filter(data, highcut)
        decoded = self._decode_source_chunk(model, device, data)
        decoded = decoded[:original_channels]

        if float(sampling_rate) != self.target_rate:
            decoded = resample_poly(
                decoded,
                int(round(sampling_rate)),
                self.target_rate,
                axis=-1,
            )
        if decoded.shape[-1] < original_samples:
            decoded = np.pad(
                decoded,
                ((0, 0), (0, original_samples - decoded.shape[-1])),
                mode="edge",
            )
        return decoded[:, :original_samples].astype(np.float32)

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
        """Denoise a long recording with overlap-add model chunks.

        Filtering and normalization are performed on the complete recording;
        only neural-network inference is chunked.  This avoids treating each
        short Attention window as an independent ASR/IC-U-Net calibration unit.
        """
        model, device = self._load_model(checkpoint_path)
        signal = np.asarray(signal, dtype=np.float64)
        original_samples = int(signal.shape[-1])
        data, original_channels = self._pad_channels(signal)

        if float(sampling_rate) != self.target_rate:
            data = resample_poly(
                data,
                self.target_rate,
                int(round(sampling_rate)),
                axis=-1,
            )
        highcut = 40.0 if task_name == "bci_errp" else 50.0
        data = self._fir_filter(data, highcut)
        normalized, standard_deviation = self._normalise(data)

        chunk = max(8, int(chunk_seconds * self.target_rate))
        chunk -= chunk % 8
        overlap = max(0, int(overlap_seconds * self.target_rate))
        overlap = min(overlap, chunk // 2)
        step = chunk - overlap
        total = normalized.shape[-1]

        if total <= chunk:
            decoded = self._infer_array(model, device, normalized)
        else:
            accumulator = np.zeros_like(normalized, dtype=np.float64)
            weights = np.zeros(total, dtype=np.float64)
            starts = list(range(0, total, step))
            if starts[-1] + chunk < total:
                starts.append(total - chunk)
            starts = sorted(set(min(start, max(0, total - chunk)) for start in starts))

            for start in progress(
                starts,
                total=len(starts),
                desc=f"IC-U-Net recording {task_name or 'EEG'}",
                unit="chunk",
                leave=False,
            ):
                end = min(start + chunk, total)
                piece = normalized[:, start:end]
                estimate = self._infer_array(model, device, piece)
                length = estimate.shape[-1]
                window = np.hanning(length) if length > 2 else np.ones(length)
                window = np.maximum(window, 0.05)
                if start == 0:
                    window[: min(overlap, length)] = 1.0
                if start + length >= total:
                    window[max(0, length - overlap) :] = 1.0
                accumulator[:, start : start + length] += estimate * window
                weights[start : start + length] += window

            decoded = accumulator / np.maximum(weights, 1e-8)[np.newaxis, :]

        if standard_deviation > 0:
            decoded *= standard_deviation
        decoded = decoded[:original_channels]
        if float(sampling_rate) != self.target_rate:
            decoded = resample_poly(
                decoded,
                int(round(sampling_rate)),
                self.target_rate,
                axis=-1,
            )
        if decoded.shape[-1] < original_samples:
            decoded = np.pad(
                decoded,
                ((0, 0), (0, original_samples - decoded.shape[-1])),
                mode="edge",
            )
        return decoded[:, :original_samples].astype(np.float32)

    def transform(
        self,
        signals: np.ndarray,
        sampling_rate: float,
        checkpoint_path: Path,
        task_name: str | None = None,
        **_: object,
    ) -> np.ndarray:
        output: list[np.ndarray] = []
        epochs = progress(
            signals,
            total=len(signals),
            desc=f"IC-U-Net {task_name or 'EEG'}",
            unit="epoch",
            leave=False,
        )
        for epoch in epochs:
            output.append(
                self._process_epoch(
                    epoch,
                    sampling_rate,
                    checkpoint_path,
                    task_name,
                )
            )
        return np.stack(output)


DENOISER = ICUNetDenoiser()
