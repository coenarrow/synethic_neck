"""Illumination stage: ambient gain, slow OU drift, mains flicker, specular blob."""
from __future__ import annotations

import numpy as np

from ..config import IlluminationParams


class IlluminationStage:
    def __init__(self, params: IlluminationParams, n: int, fps: float, n_frames: int, shading: np.ndarray):
        self.p = params
        self.fps = fps
        rng = np.random.default_rng(params.seed)

        # Ornstein-Uhlenbeck drift, stationary sd = drift_sd, correlation time drift_tau_s.
        dt = 1.0 / fps
        a = np.exp(-dt / params.drift_tau_s) if params.drift_tau_s > 0 else 0.0
        x = np.zeros(n_frames)
        if params.drift_sd > 0:
            x[0] = rng.standard_normal() * params.drift_sd
            kick = params.drift_sd * np.sqrt(1 - a * a)
            for i in range(1, n_frames):
                x[i] = a * x[i - 1] + kick * rng.standard_normal()
        self._drift = x

        # Specular highlight: Gaussian blob centred on the shading peak nearest the frame centre.
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float64)
        row = shading[n // 2]
        peak_x = int(np.argmax(row))
        sigma = params.specular_sigma_frac * n
        self.specular = params.specular_amp * np.exp(-0.5 * ((xs - peak_x) ** 2 + (ys - n / 2) ** 2) / sigma ** 2)

    def gain(self, i: int) -> float:
        t = i / self.fps
        flicker = 1.0 + self.p.flicker_amp * np.sin(2 * np.pi * self.p.flicker_hz * t)
        return float(self.p.ambient_gain * (1.0 + self._drift[i]) * flicker)

    def __call__(self, img: np.ndarray, i: int) -> np.ndarray:
        g = self.gain(i)
        spec = self.specular if img.ndim == 2 else self.specular[..., None]
        if g == 1.0 and self.p.specular_amp == 0.0:
            return img
        return img * g + spec
