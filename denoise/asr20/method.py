"""Artifact Subspace Reconstruction with cutoff k=20."""

from denoise.asr.method import ASRDenoiser


class ASR20Denoiser(ASRDenoiser):
    name = "asr20"

    def configure(self, settings: dict) -> None:
        super().configure(settings)
        self.cutoff = 20


DENOISER = ASR20Denoiser()
