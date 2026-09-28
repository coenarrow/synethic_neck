"""Exact area-average resampling between the native grid and the delivered size, for the sensor stage and for the
texture drawn at native resolution to match the delivered one."""
from __future__ import annotations

import numpy as np


def area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix averaging n_in samples into n_out by the overlap of the output pixel with each input
    pixel, exact for any ratio; rows sum to 1."""
    edges_out = np.linspace(0.0, n_in, n_out + 1)
    w = np.zeros((n_out, n_in))
    for i in range(n_out):
        lo, hi = edges_out[i], edges_out[i + 1]
        j = np.arange(int(np.floor(lo)), int(np.ceil(hi)))
        w[i, j] = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
    return w / (n_in / n_out)


def noise_gain(n_in: int, n_out: int) -> float:
    """Factor by which the area average from n_in to n_out scales the sd of white noise, over both axes: the mean
    over output pixels of the root of the sum of squared weights, squared for the two axes. 1 / k for an integer
    ratio k."""
    if n_in == n_out:
        return 1.0
    w = area_weights(n_in, n_out)
    return float(np.mean(np.sqrt((w * w).sum(1))) ** 2)


def area_average(x: np.ndarray, n_out: int) -> np.ndarray:
    """Area-average an (n, n) or (n, n, c) image to (n_out, n_out[, c])."""
    n = x.shape[0]
    if n_out == n:
        return np.asarray(x, dtype=np.float64).copy()
    w = area_weights(n, n_out)
    y = np.tensordot(w, np.asarray(x, dtype=np.float64), axes=(1, 0))    # rows averaged: (n_out, n[, c])
    y = np.tensordot(w, y, axes=(1, 1))                                   # columns averaged: (n_out cols, n_out rows[, c])
    return np.swapaxes(y, 0, 1)


def resample_labels(labels: np.ndarray, n_out: int) -> np.ndarray:
    """Labels at the delivered size by majority of the area each output pixel covers."""
    ids = np.unique(labels)
    if n_out == labels.shape[0]:
        return labels.copy()
    votes = np.stack([area_average((labels == i).astype(np.float64), n_out) for i in ids], axis=-1)
    return ids[np.argmax(votes, axis=-1)].astype(labels.dtype)
