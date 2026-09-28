"""Static appearance: the skin of the neck in a Monk Skin Tone, lit as a Lambertian cylinder, with multiplicative
texture; a flat colour with noise beside the neck; and the near-infrared level. Everything here is drawn once per
sample and does not move. The pulse changes it at the next stage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.priors import IntRange, Normal, Range
from .resample import noise_gain
from .scene import Scene, SceneMaps

# The ten swatches of the Monk Skin Tone scale, index 0 == Monk 1 (skintone.google).
MONK_HEX = ("#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
            "#a07e56", "#825c43", "#604134", "#3a312a", "#292420")


def monk_rgb(tone: int) -> np.ndarray:
    """sRGB levels (0-255) of Monk tone 1..10."""
    h = MONK_HEX[tone - 1].lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


@dataclass(frozen=True)
class AppearanceConfig:
    monk_tone: IntRange
    texture_frac: Normal            # sd of the multiplicative skin texture, as a fraction of the level
    ambient: Range                  # share of the light that is ambient; the rest is directional
    light_angle_deg: Range          # direction of the light in the cross-section plane, 0 = from the camera
    ir_level: Normal                # near-infrared level of lit skin, 16-bit units
    background_level: Range         # flat colour beside the neck, one draw per channel, 8-bit levels
    background_ir_level: Normal     # and its near-infrared level, 16-bit units


@dataclass(frozen=True)
class Appearance:
    monk_tone: int
    skin_rgb: tuple[float, float, float]
    texture_frac: float
    ambient: float
    light_angle_deg: float
    ir_level: float
    background_rgb: tuple[float, float, float]
    background_ir: float


def draw_appearance(cfg: AppearanceConfig, rng: np.random.Generator) -> Appearance:
    tone = cfg.monk_tone.draw(rng)
    return Appearance(tone, tuple(monk_rgb(tone)), cfg.texture_frac.draw(rng), cfg.ambient.draw(rng),
                      cfg.light_angle_deg.draw(rng), cfg.ir_level.draw(rng),
                      tuple(cfg.background_level.draw(rng) for _ in range(3)), cfg.background_ir_level.draw(rng))


def shading(maps: SceneMaps, app: Appearance, tilt: np.ndarray | float = 0.0,
            light_angle_deg: float | None = None) -> np.ndarray:
    """Lambertian factor on the cylinder: ambient + (1 - ambient) cos(theta), where theta is the angle between the
    outward normal at angle phi round the cylinder, turned by `tilt` (radians) where the skin is lifted, and a light
    in the cross-section plane at `light_angle_deg` from the camera axis (the drawn angle when None; 0 for the
    infrared, whose light is the depth camera's own illuminator). 1 off the neck."""
    lam = np.deg2rad(app.light_angle_deg if light_angle_deg is None else light_angle_deg)
    cos_theta = np.clip(np.cos(maps.phi + tilt - lam), 0.0, None)
    return np.where(maps.on_neck, app.ambient + (1.0 - app.ambient) * cos_theta, 1.0)


@dataclass(frozen=True)
class BaseFrame:
    rgb: np.ndarray                 # (n, n, 3) float, 8-bit levels
    ir: np.ndarray                  # (n, n) float, 16-bit levels
    shading: np.ndarray             # (n, n), the Lambertian factor under the drawn light
    texture: np.ndarray             # (n, n), the multiplicative field, mean 1


def base_frame(scene: Scene, maps: SceneMaps, app: Appearance, output_px: int, rng: np.random.Generator) -> BaseFrame:
    """The unpulsed frame on the grid of `maps`. The texture sd is given at the delivered resolution; on a finer
    grid it is scaled up by the inverse of the noise gain of the area average down to `output_px`, so that average
    returns it."""
    n = maps.depth_mm.shape[0]
    sh, sh_ir = shading(maps, app), shading(maps, app, light_angle_deg=0.0)
    texture = 1.0 + app.texture_frac / noise_gain(n, output_px) * rng.standard_normal((n, n))
    skin = np.asarray(app.skin_rgb)[None, None, :] * sh[..., None]
    bg = np.asarray(app.background_rgb)[None, None, :] * np.ones((n, n, 1))
    rgb = np.where(maps.on_neck[..., None], skin, bg) * texture[..., None]
    ir = np.where(maps.on_neck, app.ir_level * sh_ir, app.background_ir) * texture
    return BaseFrame(rgb, ir, sh, texture)
