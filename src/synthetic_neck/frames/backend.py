"""Where the frame stages run: numpy on the CPU, or torch on a GPU (CUDA, or Apple MPS). The batched render and
sensor are written against `xp`, the numpy or torch module, whose names agree for everything they use (cos, where,
clip, floor, round, concatenate, matmul); the few operations that differ live here. Float32 throughout: the delivered
types are 8 and 16 bit, and the sub-millimetre lifts sit on a ~700 mm depth with 6e-5 mm to spare.

The sensor noise comes from the sample's numpy generator on the CPU, and on a GPU from torch's generator there,
seeded by a draw from the sample's, so on either a seed reproduces its frames; the noise realisations of the two
differ, and the rest of the frame agrees to float32 rounding.
"""
from __future__ import annotations

import numpy as np

DEVICES = ("auto", "cpu", "cuda", "mps")


class Backend:
    name: str
    xp = np

    @staticmethod
    def create(device: str = "auto") -> "Backend":
        """`cpu` is numpy; `cuda` and `mps` need the `gpu` extra (torch); `auto` takes the first GPU torch finds,
        else numpy."""
        if device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
        if device == "cpu":
            return NumpyBackend()
        try:
            import torch
        except ImportError:
            if device == "auto":
                return NumpyBackend()
            raise RuntimeError(f"device {device!r} needs torch: install the gpu extra (uv sync --extra gpu)") from None
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            return NumpyBackend() if device == "cpu" else TorchBackend(device)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch finds no CUDA device")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but torch finds no MPS device")
        return TorchBackend(device)

    def array(self, x):
        """A numpy array (or list) onto the device, floats as float32, bool and integer types kept."""
        raise NotImplementedError

    def numpy(self, x) -> np.ndarray:
        raise NotImplementedError

    def int(self, x):
        """Float values (already whole) as int64, for indexing."""
        raise NotImplementedError

    def normal(self, shape: tuple[int, ...], rng: np.random.Generator):
        """Standard normal float32 noise of `shape` on the device, from `rng` or seeded by it."""
        raise NotImplementedError

    def uniform(self, shape: tuple[int, ...], rng: np.random.Generator):
        """Uniform [0, 1) float32 noise of `shape` on the device, from `rng` or seeded by it."""
        raise NotImplementedError

    def lookup(self, y, pos):
        """Linear interpolation of the uniformly sampled `y` (m,) at fractional sample positions `pos` (any shape),
        held at the end values outside [0, m - 1] as np.interp does."""
        xp = self.xp
        i = xp.clip(xp.floor(pos), 0, y.shape[0] - 2)
        frac = xp.clip(pos - i, 0.0, 1.0)
        idx = self.int(i)
        return y[idx] * (1.0 - frac) + y[idx + 1] * frac


class NumpyBackend(Backend):
    name = "cpu"
    xp = np

    def array(self, x):
        x = np.asarray(x)
        return x.astype(np.float32) if x.dtype.kind == "f" else x

    def numpy(self, x):
        return np.asarray(x)

    def int(self, x):
        return x.astype(np.int64)

    def normal(self, shape, rng):
        return rng.standard_normal(shape, dtype=np.float32)

    def uniform(self, shape, rng):
        return rng.random(shape, dtype=np.float32)


class TorchBackend(Backend):
    def __init__(self, device: str):
        import torch
        self.torch, self.xp, self.name = torch, torch, device
        self.device = torch.device(device)

    def array(self, x):
        x = np.asarray(x)
        if x.dtype.kind == "f":
            x = x.astype(np.float32)
        return self.torch.asarray(x, device=self.device)

    def numpy(self, x):
        return x.cpu().numpy()

    def int(self, x):
        return x.to(self.torch.int64)

    def _generator(self, rng):
        return self.torch.Generator(device=self.device).manual_seed(int(rng.integers(0, 2**63 - 1)))

    def normal(self, shape, rng):
        return self.torch.randn(shape, generator=self._generator(rng), device=self.device, dtype=self.torch.float32)

    def uniform(self, shape, rng):
        return self.torch.rand(shape, generator=self._generator(rng), device=self.device, dtype=self.torch.float32)
