"""QA tool: per-pixel FFT power and phase maps for every channel of a sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render.camera import DEPTH_UNITS_MM
from .video import read_mkv


def load_channels(sample_dir: Path) -> dict[str, np.ndarray]:
    meta = json.loads((sample_dir / "metadata.json").read_text())
    p = sample_dir / "rgbid_video.mkv"
    rgb = read_mkv(p, "rgb", 0).astype(np.float64)
    out = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
    if "ir" in meta["streams"]:
        out["IR"] = read_mkv(p, "gray", meta["streams"].index("ir")).astype(np.float64)
        out["Depth (mm)"] = read_mkv(p, "gray16", meta["streams"].index("depth")).astype(np.float64) * DEPTH_UNITS_MM
    return out


def fft_maps(sample_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Return ({channel: power}, {channel: phase}, hr_bin_hz). Power is the peak |FFT|^2
    within ±1 bin of the heart-rate bin; phase is the FFT angle at that bin."""
    meta = json.loads((sample_dir / "metadata.json").read_text())
    fps, hr_hz = meta["fps"], meta["params"]["trace"]["heart_rate_bpm"] / 60.0
    chans = load_channels(sample_dir)
    n = next(iter(chans.values())).shape[0]
    freqs = np.fft.rfftfreq(n, 1 / fps)
    k = int(np.argmin(np.abs(freqs - hr_hz)))
    win = np.hanning(n)[:, None, None]
    power, phase = {}, {}
    for name, v in chans.items():
        spec = np.fft.rfft((v - v.mean(0)) * win, axis=0)
        power[name] = np.abs(spec[k - 1:k + 2]).max(0) ** 2
        phase[name] = np.angle(spec[k])
    return power, phase, float(freqs[k])


def vessel_phase_summary(phase: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Circular-mean phase over artery / vein masks and their difference, per channel."""
    out = {}
    for name, ph in phase.items():
        a = np.angle(np.exp(1j * ph[ids == 1]).mean())
        v = np.angle(np.exp(1j * ph[ids == 2]).mean())
        out[name] = (float(a), float(v), float(np.angle(np.exp(1j * (v - a)))))
    return out


def vessel_power_ratio(power: dict[str, np.ndarray], ids: np.ndarray) -> dict[str, tuple[float, float]]:
    """(artery/skin, vein/skin) mean power ratios per channel: how far each vessel stands out above the
    pulsing skin around it."""
    out = {}
    for name, pw in power.items():
        bg = pw[ids == 0].mean()
        out[name] = (float(pw[ids == 1].mean() / bg), float(pw[ids == 2].mean() / bg))
    return out


def cardiac_snr(series: np.ndarray, fps: float, hr_hz: float) -> float:
    """Cardiac signal-to-noise of a 1-D series: the mean over HR, 2xHR and 3xHR of the peak Hann-windowed
    rFFT power within ±1 bin, divided by the median bin power in 5-9 Hz (the series' own noise floor).
    Harmonics matter because the venous pulse carries most of its cardiac energy at 2xHR."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fps)
    peaks = []
    for h in (1, 2, 3):
        k = int(np.argmin(np.abs(f - h * hr_hz)))
        peaks.append(p[max(k - 1, 0):k + 2].max())
    return float(np.mean(peaks) / np.median(p[(f >= 5.0) & (f <= 9.0)]))


def vessel_snr(sample_dir: Path) -> dict[str, tuple[float, float, float]]:
    """(artery, vein, skin) cardiac SNR per channel, from each region mask's per-frame mean signal; the skin is
    the background mask."""
    meta = json.loads((sample_dir / "metadata.json").read_text())
    fps, hr_hz = meta["fps"], meta["params"]["trace"]["heart_rate_bpm"] / 60.0
    ids = np.load(sample_dir / "vessel_ids.npy")
    out = {}
    for name, v in load_channels(sample_dir).items():
        out[name] = tuple(cardiac_snr(v[:, ids == k].mean(1), fps, hr_hz) for k in (1, 2, 0))
    return out


def plot_maps(sample_dir: Path, out_path: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    power, phase, f_hz = fft_maps(sample_dir)
    fig, ax = plt.subplots(2, len(power), figsize=(4 * len(power), 8), squeeze=False)
    for j, name in enumerate(power):
        im = ax[0, j].imshow(power[name], cmap="viridis")
        ax[0, j].set_title(f"{name}: peak power at HR ({f_hz:.1f}±0.1 Hz)")
        fig.colorbar(im, ax=ax[0, j], fraction=0.046)
        im = ax[1, j].imshow(phase[name], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[1, j].set_title(f"{name}: phase at {f_hz:.1f} Hz (rad)")
        fig.colorbar(im, ax=ax[1, j], fraction=0.046)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"{sample_dir.name} — per-pixel FFT (Hann window, DC removed)")
    fig.tight_layout()
    out_path = out_path or sample_dir / "fft_maps.png"
    fig.savefig(out_path, dpi=80)
    plt.close(fig)
    return out_path


def inspect_root(root: Path, samples: list[str] | None = None) -> None:
    names = samples or sorted((p.name for p in root.iterdir() if p.is_dir()), key=int)
    for name in names:
        d = root / name
        out = plot_maps(d)
        power, phase, _ = fft_maps(d)
        ids = np.load(d / "vessel_ids.npy")
        s = vessel_phase_summary(phase, ids)["G"]
        r = vessel_power_ratio(power, ids)["G"]
        skin = vessel_snr(d)["G"][2]
        print(f"{name}: {out.name}  G power artery x{r[0]:.1f} vein x{r[1]:.1f} over skin  skin SNR {skin:.0f}  "
              f"phase artery {s[0]:+.2f} vein {s[1]:+.2f} diff {s[2]:+.2f} rad")
