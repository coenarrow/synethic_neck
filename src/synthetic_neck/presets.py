"""Named presets: fully populated GeneratorConfigs for the three use cases."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import (AppearanceConfig, GeneratorConfig, GeometryConfig, IlluminationConfig, PulseConfig,
                     Range, SensorConfig)

DEFAULT_PRIORS = Path(__file__).resolve().parents[2] / "priors" / "neckflix.json"

# Approximate sRGB of the ten Monk Skin Tone swatches (index 0 == Monk 1). The
# neckflix preset replaces these with measured per-tone skin colour when the
# priors file has them.
MONK_SKIN_RGB: tuple[tuple[float, float, float], ...] = (
    (246, 237, 228), (243, 231, 219), (240, 226, 200), (234, 218, 186), (215, 189, 150),
    (160, 126, 86), (130, 92, 67), (96, 65, 52), (58, 49, 42), (41, 36, 32),
)


def lesson() -> GeneratorConfig:
    """Clean, obvious signal: large amplitude, flat lighting, whole vein pulsing."""
    return GeneratorConfig(
        pulse=PulseConfig(amplitude_levels=Range(7, 12)),
        geometry=GeometryConfig(vein_visible_fraction=Range(1.0, 1.0)),
        appearance=AppearanceConfig(shading_strength=Range(0.0, 0.0)),
        illumination=IlluminationConfig(),
        sensor=SensorConfig(read_noise_sd=Range(1.0, 2.0), depth_noise_mm_at_1m=Range(0.4, 0.4)),
    )


def benchmark() -> GeneratorConfig:
    """Moderate amplitude with controlled degradations turned on at modest levels."""
    return GeneratorConfig(
        pulse=PulseConfig(amplitude_levels=Range(2, 5)),
        geometry=GeometryConfig(vein_visible_fraction=Range(0.6, 1.0)),
        appearance=AppearanceConfig(shading_strength=Range(0.3, 0.8)),
        illumination=IlluminationConfig(
            ambient_gain=Range(0.85, 1.15), drift_sd=Range(0.0, 0.02), drift_tau_s=Range(10, 40),
            flicker_amp=Range(0.0, 0.01), specular_amp=Range(0.0, 15.0)),
        sensor=SensorConfig(read_noise_sd=Range(1.0, 3.0), shot_noise_gain=Range(0.0, 0.03),
                            blur_sigma_px=Range(0.0, 1.5)),
    )


def neckflix(priors_path: Path = DEFAULT_PRIORS) -> GeneratorConfig:
    """Ranges calibrated from the Neckflix dataset (see calibrate.py)."""
    priors_path = Path(priors_path)
    if not priors_path.exists():
        raise FileNotFoundError(
            f"{priors_path} not found; run `synthetic-neck calibrate --root <Neckflix dir> --out {priors_path}`")
    from .calibrate import config_from_priors   # implemented in Task 16
    return config_from_priors(priors_path)


PRESETS: dict[str, Callable[[], GeneratorConfig]] = {"lesson": lesson, "benchmark": benchmark, "neckflix": neckflix}


def get_preset(name: str) -> GeneratorConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]()
