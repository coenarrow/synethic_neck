"""Configuration (ranges) -> concrete per-sample parameters.

A `GeneratorConfig` is a tree of blocks whose random fields are `Range`,
`IntRange` or `Choice`. `sample()` draws each of them once, in a fixed order,
producing `SampleParams`: parallel dataclasses of plain numbers that the
trace synthesiser and renderer consume.
"""
from __future__ import annotations

import math
import types
import typing
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import numpy as np


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- distributions

_KNOT_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class Range:
    lo: float
    hi: float
    knots: tuple[float, ...] | None = None   # p05, p25, p50, p75, p95: draw by inverse-CDF interpolation

    def __post_init__(self):
        object.__setattr__(self, "lo", float(self.lo))
        object.__setattr__(self, "hi", float(self.hi))
        if self.lo > self.hi:
            raise ConfigError(f"Range lo {self.lo} > hi {self.hi}")
        if self.knots is not None:
            k = tuple(float(x) for x in self.knots)
            if (len(k) != len(_KNOT_LEVELS) or any(b < a for a, b in zip(k, k[1:]))
                    or k[0] != self.lo or k[-1] != self.hi):
                raise ConfigError("Range knots must be 5 non-decreasing quantiles running from lo to hi")
            object.__setattr__(self, "knots", k)

    def draw(self, rng: np.random.Generator) -> float:
        if self.lo == self.hi:
            return self.lo
        if self.knots is None:
            return float(rng.uniform(self.lo, self.hi))
        q = float(rng.uniform(_KNOT_LEVELS[0], _KNOT_LEVELS[-1]))
        return float(np.interp(q, _KNOT_LEVELS, self.knots))


@dataclass(frozen=True)
class IntRange:
    lo: int
    hi: int

    def __post_init__(self):
        if self.lo > self.hi:
            raise ConfigError(f"IntRange lo {self.lo} > hi {self.hi}")

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.lo, self.hi + 1))


@dataclass(frozen=True)
class Choice:
    values: tuple
    weights: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))
        if self.weights is not None:
            w = tuple(float(x) for x in self.weights)
            if len(w) != len(self.values) or sum(w) <= 0:
                raise ConfigError("Choice weights must match values and sum > 0")
            object.__setattr__(self, "weights", w)

    def draw(self, rng: np.random.Generator):
        p = None if self.weights is None else np.asarray(self.weights) / sum(self.weights)
        return self.values[int(rng.choice(len(self.values), p=p))]


# --------------------------------------------------------------------------- config blocks

@dataclass(frozen=True)
class VideoConfig:
    frame_size: int = 300
    fps: float = 30.0
    duration_s: float = 10.0
    trace_sample_rate_hz: float = 1000.0


@dataclass(frozen=True)
class StreamsConfig:
    depth_ir_probability: float = 1.0     # P(sample includes IR + depth streams)


@dataclass(frozen=True)
class TraceConfig:
    heart_rate_bpm: Range = Range(55, 95)
    hr_variability: Range = Range(0.01, 0.05)
    resp_rate_bpm: Range = Range(10, 18)
    diastolic_mmhg: Range = Range(62, 88)
    pulse_pressure_mmhg: Range = Range(35, 60)
    cvp_mean_mmhg: Range = Range(6, 15)
    a_wave_mmhg: Range = Range(2.5, 2.5)
    c_wave_mmhg: Range = Range(1.0, 1.0)
    x_descent_mmhg: Range = Range(1.8, 1.8)
    v_wave_mmhg: Range = Range(2.0, 2.0)
    y_descent_mmhg: Range = Range(1.2, 1.2)
    resp_abp_swing_mmhg: Range = Range(2.0, 2.0)
    resp_cvp_swing_mmhg: Range = Range(1.5, 1.5)
    abp_noise_mmhg: Range = Range(0.3, 0.3)
    cvp_noise_mmhg: Range = Range(0.15, 0.15)
    posture_deg: Choice = Choice((0.0, 45.0, 90.0))
    # Timing (s): R-wave -> aortic valve opening, then transit to each measurement site. Sources: spec section 4.
    # pep, site, brachial transit, radial transit, radial amplification, radial->finger.
    pep_s: Range = Range(0.08, 0.12)
    abp_site: Choice = Choice(("radial", "brachial"))
    brachial_transit_s: Range = Range(0.05, 0.09)
    radial_transit_s: Range = Range(0.02, 0.04)
    radial_amplification: Range = Range(1.05, 1.15)    # brachial -> radial pulse-pressure amplification
    radial_to_finger_s: Range = Range(0.03, 0.08)
    skin_transit_s: Range = Range(0.02, 0.04)  # carotid -> neck skin capillaries (frame-physiology spec, section 3)
    # Respiration: phase at t = 0 and the RR shortening at end-inspiration (respiratory sinus arrhythmia).
    resp_phase_rad: Range = Range(0.0, 2 * math.pi)
    rsa_fraction: Range = Range(0.02, 0.08)
    # ECG (mV): McSharry ECGSYN morphology scaled to the R-peak; respiratory modulation; noise.
    r_amplitude_mv: Range = Range(0.8, 1.5)
    ecg_r_modulation: Range = Range(0.03, 0.10)
    ecg_wander_mv: Range = Range(0.02, 0.08)
    ecg_noise_mv: Range = Range(0.005, 0.02)
    # Finger PPG (arb, one beat spans ~0..1): respiratory amplitude modulation, baseline wander, noise.
    ppg_am_frac: Range = Range(0.05, 0.15)
    ppg_wander_frac: Range = Range(0.05, 0.15)
    ppg_noise: Range = Range(0.005, 0.02)


@dataclass(frozen=True)
class CameraConfig:
    distance_mm: Range = Range(500, 1000)
    native_width_px: int = 3840
    hfov_deg: float = 90.0
    crop_ratio: float = 650 / 300          # rendered crop size / output size


@dataclass(frozen=True)
class GeometryConfig:
    length_mm: Range = Range(40, 80)
    separation_mm: Range = Range(10, 16)
    artery_width_mm: Range = Range(6, 9)
    vein_width_mm: Range = Range(9, 15)
    angle_deg: Range = Range(55, 125)
    centre_jitter_frac: float = 0.1
    vein_visible_fraction: Range = Range(1.0, 1.0)
    heart_to_neck_artery_m: float = 0.20
    heart_to_neck_vein_m: float = 0.15


@dataclass(frozen=True)
class AppearanceConfig:
    skin_base: Range = Range(90, 215)          # red-channel level
    skin_g_ratio: Range = Range(0.70, 0.80)
    skin_b_ratio: Range = Range(0.55, 0.68)
    monk_tone: Choice | None = None            # when set, skin colour comes from skin_rgb_by_monk
    skin_rgb_by_monk: tuple[tuple[float, float, float], ...] | None = None   # index 0 == Monk 1
    texture_sd: Range = Range(3.0, 3.0)
    vignette: Range = Range(25.0, 25.0)
    static_vessel_contrast: Range = Range(2.0, 2.0)
    ir_base: Range = Range(150, 200)
    ir_vein_contrast: Range = Range(8.0, 8.0)
    neck_radius_mm: Range = Range(50, 70)
    shading_strength: Range = Range(0.0, 0.0)  # 0 flat, 1 full Lambertian cylinder


@dataclass(frozen=True)
class PulseConfig:
    amplitude_levels: Range = Range(7, 12)     # artery, green channel, peak-to-peak
    vein_ratio: Range = Range(0.4, 0.6)        # vein amplitude / artery amplitude
    skin_ratio: Range = Range(0.1, 0.3)        # skin-wide PPG amplitude / artery amplitude
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: Range = Range(0.35, 0.35)
    artery_lift_mm: Range = Range(0.3, 0.3)
    vein_lift_mm: Range = Range(0.5, 0.5)
    resp_gain_frac: Range = Range(0.003, 0.010)   # whole-scene brightness swing with breathing, fractional
    resp_lift_mm: Range = Range(0.5, 1.0)         # whole-frame depth swing with breathing, mm


@dataclass(frozen=True)
class IlluminationConfig:
    ambient_gain: Range = Range(1.0, 1.0)
    drift_sd: Range = Range(0.0, 0.0)          # fractional gain sd of the slow OU drift
    drift_tau_s: Range = Range(20.0, 20.0)
    flicker_amp: Range = Range(0.0, 0.0)       # fractional
    flicker_hz: Range = Range(100.0, 100.0)
    specular_amp: Range = Range(0.0, 0.0)      # levels
    specular_sigma_frac: Range = Range(0.15, 0.15)   # of frame size


@dataclass(frozen=True)
class SensorConfig:
    read_noise_sd: Range = Range(1.0, 2.0)     # levels, RGB
    shot_noise_gain: Range = Range(0.0, 0.0)   # variance per level of signal
    ir_read_noise_sd: Range = Range(2.0, 2.0)
    blur_sigma_px: Range = Range(0.0, 0.0)     # at rendered (crop) resolution
    depth_noise_mm_at_1m: Range = Range(1.6, 1.6)


@dataclass(frozen=True)
class MotionConfig:
    """Reserved for subject motion (deferred)."""


@dataclass(frozen=True)
class RhythmConfig:
    """Reserved for arrhythmia (deferred)."""


@dataclass(frozen=True)
class GeneratorConfig:
    video: VideoConfig = VideoConfig()
    streams: StreamsConfig = StreamsConfig()
    trace: TraceConfig = TraceConfig()
    camera: CameraConfig = CameraConfig()
    geometry: GeometryConfig = GeometryConfig()
    appearance: AppearanceConfig = AppearanceConfig()
    pulse: PulseConfig = PulseConfig()
    illumination: IlluminationConfig = IlluminationConfig()
    sensor: SensorConfig = SensorConfig()
    motion: MotionConfig = MotionConfig()
    rhythm: RhythmConfig = RhythmConfig()


# --------------------------------------------------------------------------- concrete params

@dataclass(frozen=True)
class VideoParams:
    frame_size: int
    fps: float
    duration_s: float

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps))


@dataclass(frozen=True)
class StreamsParams:
    has_depth_ir: bool


@dataclass(frozen=True)
class TraceParams:
    duration_s: float = 10.0
    sample_rate_hz: float = 1000.0
    heart_rate_bpm: float = 72.0
    hr_variability: float = 0.03
    resp_rate_bpm: float = 14.0
    systolic_mmhg: float = 120.0
    diastolic_mmhg: float = 78.0
    cvp_mean_mmhg: float = 6.0
    a_wave_mmhg: float = 2.5
    c_wave_mmhg: float = 1.0
    x_descent_mmhg: float = 1.8
    v_wave_mmhg: float = 2.0
    y_descent_mmhg: float = 1.2
    resp_abp_swing_mmhg: float = 2.0
    resp_cvp_swing_mmhg: float = 1.5
    abp_noise_mmhg: float = 0.3
    cvp_noise_mmhg: float = 0.15
    posture_deg: float = 45.0
    pep_s: float = 0.10
    abp_site: str = "brachial"
    brachial_transit_s: float = 0.07
    radial_transit_s: float = 0.03
    radial_amplification: float = 1.10
    radial_to_finger_s: float = 0.05
    skin_transit_s: float = 0.03
    resp_phase_rad: float = 0.0
    rsa_fraction: float = 0.05
    r_amplitude_mv: float = 1.0
    ecg_r_modulation: float = 0.05
    ecg_wander_mv: float = 0.05
    ecg_noise_mv: float = 0.01
    ppg_am_frac: float = 0.10
    ppg_wander_frac: float = 0.10
    ppg_noise: float = 0.01
    seed: int = 0

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sample_rate_hz)) + 1

    @property
    def abp_site_delay_s(self) -> float:
        """Aortic root -> the stored ABP's catheter site."""
        if self.abp_site == "radial":
            return self.brachial_transit_s + self.radial_transit_s
        return self.brachial_transit_s

    @property
    def r_to_abp_foot_s(self) -> float:
        return self.pep_s + self.abp_site_delay_s

    @property
    def r_to_ppg_foot_s(self) -> float:
        """Pulse arrival time: R-wave -> finger PPG foot. The finger is distal to both catheter sites."""
        return self.pep_s + self.brachial_transit_s + self.radial_transit_s + self.radial_to_finger_s


@dataclass(frozen=True)
class CameraParams:
    distance_mm: float = 500.0
    native_width_px: int = 3840
    hfov_deg: float = 90.0
    crop_px: int = 650
    output_px: int = 300

    @property
    def native_scale_mm(self) -> float:
        """mm per native pixel at the centre of the frame."""
        return 2.0 * self.distance_mm * np.tan(np.deg2rad(self.hfov_deg / 2)) / self.native_width_px

    @property
    def render_scale(self) -> float:
        return self.crop_px / self.output_px

    @property
    def pixel_scale_mm(self) -> float:
        """mm per output pixel."""
        return self.native_scale_mm * self.render_scale


@dataclass(frozen=True)
class GeometryParams:
    length_mm: float = 60.0
    separation_mm: float = 13.0
    artery_width_mm: float = 7.5
    vein_width_mm: float = 12.0
    angle_deg: float = 80.0
    centre_frac_xy: tuple[float, float] = (0.5, 0.5)
    artery_side: int = 1
    vein_visible_fraction: float = 1.0
    heart_to_neck_artery_m: float = 0.20
    heart_to_neck_vein_m: float = 0.15


@dataclass(frozen=True)
class AppearanceParams:
    skin_rgb: tuple[float, float, float] = (196.0, 150.0, 124.0)
    monk_tone: int | None = None
    texture_sd: float = 3.0
    vignette: float = 25.0
    static_vessel_contrast: float = 2.0
    ir_base: float = 170.0
    ir_vein_contrast: float = 8.0
    neck_radius_mm: float = 60.0
    shading_strength: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class PulseParams:
    amplitude_levels: float = 10.0
    vein_ratio: float = 0.5
    skin_ratio: float = 0.2
    channel_gain: tuple[float, float, float] = (0.55, 1.0, 0.7)
    ir_gain: float = 0.35
    artery_lift_mm: float = 0.3
    vein_lift_mm: float = 0.5
    resp_gain_frac: float = 0.005
    resp_lift_mm: float = 0.7

    @property
    def vein_amplitude_levels(self) -> float:
        return self.amplitude_levels * self.vein_ratio

    @property
    def skin_amplitude_levels(self) -> float:
        return self.amplitude_levels * self.skin_ratio


@dataclass(frozen=True)
class IlluminationParams:
    ambient_gain: float = 1.0
    drift_sd: float = 0.0
    drift_tau_s: float = 20.0
    flicker_amp: float = 0.0
    flicker_hz: float = 100.0
    specular_amp: float = 0.0
    specular_sigma_frac: float = 0.15
    seed: int = 0


@dataclass(frozen=True)
class SensorParams:
    read_noise_sd: float = 1.5
    shot_noise_gain: float = 0.0
    ir_read_noise_sd: float = 2.0
    blur_sigma_px: float = 0.0
    depth_noise_mm_at_1m: float = 1.6
    seed: int = 0


@dataclass(frozen=True)
class SampleParams:
    video: VideoParams
    streams: StreamsParams
    trace: TraceParams
    camera: CameraParams
    geometry: GeometryParams
    appearance: AppearanceParams
    pulse: PulseParams
    illumination: IlluminationParams
    sensor: SensorParams


# --------------------------------------------------------------------------- sampling

def _draw_fields(block, rng: np.random.Generator) -> dict[str, Any]:
    """Draw every Range/IntRange/Choice field of a config block; copy the rest."""
    out = {}
    for f in fields(block):
        v = getattr(block, f.name)
        out[f.name] = v.draw(rng) if isinstance(v, (Range, IntRange, Choice)) else v
    return out


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def vein_visible_fraction(posture_deg: float, cvp_mean_mmhg: float, length_mm: float, bounds: Range) -> float:
    """Fraction of the vein (from the caudal end) that pulsates visibly.

    Heuristic hydrostatics: the venous column stands ~1.36 cm per mmHg above
    the right atrium, which sits ~5 cm below the sternal angle. Supine (0 deg)
    the neck is horizontal so the whole run is below the column top. Upright,
    only the part of the neck below the column top pulsates. Always clamped to
    `bounds` so a venous signal is present in every sample.
    """
    if posture_deg <= 0:
        return bounds.hi
    column_cm = max(0.0, 1.36 * cvp_mean_mmhg - 5.0)
    neck_rise_cm = (length_mm / 10.0) * np.sin(np.deg2rad(posture_deg))
    raw = column_cm / neck_rise_cm if neck_rise_cm > 0 else 1.0
    return float(np.clip(raw, bounds.lo, bounds.hi))


def sample(config: GeneratorConfig, rng: np.random.Generator) -> SampleParams:
    """Draw one concrete parameter set. Order is fixed so seeds are reproducible."""
    v = config.video
    video = VideoParams(frame_size=v.frame_size, fps=v.fps, duration_s=v.duration_s)
    streams = StreamsParams(has_depth_ir=bool(rng.random() < config.streams.depth_ir_probability))

    t = _draw_fields(config.trace, rng)
    dia, pp = t.pop("diastolic_mmhg"), t.pop("pulse_pressure_mmhg")
    trace = TraceParams(duration_s=v.duration_s, sample_rate_hz=v.trace_sample_rate_hz,
                        diastolic_mmhg=dia, systolic_mmhg=dia + pp, seed=_seed(rng), **t)

    c = config.camera
    crop_px = int(round(v.frame_size * c.crop_ratio))
    camera = CameraParams(distance_mm=c.distance_mm.draw(rng), native_width_px=c.native_width_px,
                          hfov_deg=c.hfov_deg, crop_px=crop_px, output_px=v.frame_size)

    g = _draw_fields(config.geometry, rng)
    jitter = g.pop("centre_jitter_frac")
    g.pop("vein_visible_fraction")
    centre = (0.5 + float(rng.uniform(-jitter, jitter)), 0.5 + float(rng.uniform(-jitter, jitter)))
    side = int(rng.choice([-1, 1]))
    vis = vein_visible_fraction(trace.posture_deg, trace.cvp_mean_mmhg, g["length_mm"],
                                config.geometry.vein_visible_fraction)
    geometry = GeometryParams(centre_frac_xy=centre, artery_side=side, vein_visible_fraction=vis, **g)

    a = _draw_fields(config.appearance, rng)
    base, gr, br = a.pop("skin_base"), a.pop("skin_g_ratio"), a.pop("skin_b_ratio")
    monk, table = a.pop("monk_tone"), a.pop("skin_rgb_by_monk")
    if monk is not None and table is not None:
        skin = tuple(float(x) for x in table[int(monk) - 1])
    else:
        skin = (base, base * gr, base * br)
    appearance = AppearanceParams(skin_rgb=skin, monk_tone=None if monk is None else int(monk),
                                  seed=_seed(rng), **a)

    pulse = PulseParams(**_draw_fields(config.pulse, rng))
    illumination = IlluminationParams(seed=_seed(rng), **_draw_fields(config.illumination, rng))
    sensor = SensorParams(seed=_seed(rng), **_draw_fields(config.sensor, rng))
    return SampleParams(video, streams, trace, camera, geometry, appearance, pulse, illumination, sensor)


# --------------------------------------------------------------------------- validation

def validate(config: GeneratorConfig) -> None:
    g, v, c = config.geometry, config.video, config.camera
    if g.vein_visible_fraction.lo < 0.1:
        raise ConfigError("geometry.vein_visible_fraction lower bound must be >= 0.1 so the vein always pulses")
    if not 0.0 <= config.streams.depth_ir_probability <= 1.0:
        raise ConfigError("streams.depth_ir_probability must be in [0, 1]")
    if v.fps < 20:
        raise ConfigError("video.fps must be >= 20: the cardiac SNR noise band is 5-9 Hz")
    if c.distance_mm.hi > 1250:
        raise ConfigError("camera.distance_mm upper bound exceeds the depth sensor's range: 16-bit depth at "
                          "0.02 mm resolution stops at 1310 mm")
    if config.appearance.monk_tone is not None and config.appearance.skin_rgb_by_monk is None:
        raise ConfigError("appearance.monk_tone needs appearance.skin_rgb_by_monk")
    # Both vessel axes (frame centre +/- half the separation, plus centre jitter) must lie inside the
    # frame at the nearest camera, so both vessels are always at least partly in view. Whole-length fit
    # is not required: small test frames (e.g. 48-96 px) are centre crops of the same scene.
    near = CameraParams(distance_mm=c.distance_mm.lo, native_width_px=c.native_width_px, hfov_deg=c.hfov_deg,
                        crop_px=int(round(v.frame_size * c.crop_ratio)), output_px=v.frame_size)
    offset_px = (g.separation_mm.hi / 2) / near.pixel_scale_mm + g.centre_jitter_frac * v.frame_size
    if offset_px > v.frame_size / 2:
        raise ConfigError(f"geometry cannot fit in a {v.frame_size}px frame at {c.distance_mm.lo} mm: vessel axes "
                          f"up to {offset_px:.0f}px from centre; reduce geometry.separation_mm or "
                          f"geometry.centre_jitter_frac, or increase video.frame_size")


# --------------------------------------------------------------------------- overrides

def _parse_value(hint, text: str):
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if text.lower() == "none":
            return None
        return _parse_value(non_none[0], text)
    if hint is Range:
        parts = [float(p) for p in text.split(",")]
        return Range(parts[0], parts[-1])
    if hint is IntRange:
        parts = [int(p) for p in text.split(",")]
        return IntRange(parts[0], parts[-1])
    if hint is Choice:
        return Choice(tuple(float(p) for p in text.split("|")))
    if hint is bool:
        return text.lower() in ("1", "true", "yes")
    if hint is int:
        return int(text)
    if hint is float:
        return float(text)
    if hint is str:
        return text
    if origin is tuple:
        return tuple(float(p) for p in text.split(","))
    raise ConfigError(f"cannot parse override for type {hint}")


def _parse_string_choice(current: Choice, text: str) -> Choice:
    """Override for a Choice whose values are strings (trace.abp_site): every token must be one the field admits."""
    values = tuple(text.split("|"))
    unknown = sorted(set(values) - set(current.values))
    if unknown:
        raise ConfigError(f"{unknown} not admitted; choose from {sorted(current.values)}")
    return Choice(values)


def apply_override(config: GeneratorConfig, key: str, text: str) -> GeneratorConfig:
    """Return a copy of `config` with `block.field` set from `text`.

    Range: "lo,hi" or "v" (fixed). Choice: "a|b|c". Tuples: "a,b,c".
    """
    try:
        block_name, field_name = key.split(".", 1)
    except ValueError:
        raise ConfigError(f"override key must be block.field, got {key!r}")
    if block_name not in {f.name for f in fields(config)}:
        raise ConfigError(f"unknown config block {block_name!r}")
    block = getattr(config, block_name)
    hints = typing.get_type_hints(type(block))
    if field_name not in hints:
        raise ConfigError(f"unknown field {key!r}")
    current = getattr(block, field_name)
    try:
        if isinstance(current, Choice) and all(isinstance(v, str) for v in current.values):
            value = _parse_string_choice(current, text)
        else:
            value = _parse_value(hints[field_name], text)
    except (ValueError, IndexError) as e:
        raise ConfigError(f"cannot parse {text!r} for {key}: {e}") from e
    return replace(config, **{block_name: replace(block, **{field_name: value})})


# --------------------------------------------------------------------------- serialisation

def _to_jsonable(obj):
    if isinstance(obj, Range):
        d = {"lo": obj.lo, "hi": obj.hi}
        if obj.knots is not None:
            d["knots"] = list(obj.knots)
        return d
    if isinstance(obj, IntRange):
        return {"lo": obj.lo, "hi": obj.hi}
    if isinstance(obj, Choice):
        return {"values": list(obj.values), "weights": None if obj.weights is None else list(obj.weights)}
    if is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def config_to_dict(config: GeneratorConfig) -> dict:
    return _to_jsonable(config)


def params_to_dict(params: SampleParams) -> dict:
    return _to_jsonable(params)


def _from_jsonable(hint, value):
    if value is None:
        return None
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        return _from_jsonable([a for a in args if a is not type(None)][0], value)
    if hint is Range:
        return Range(value["lo"], value["hi"], None if value.get("knots") is None else tuple(value["knots"]))
    if hint is IntRange:
        return IntRange(value["lo"], value["hi"])
    if hint is Choice:
        return Choice(tuple(value["values"]), None if value["weights"] is None else tuple(value["weights"]))
    if is_dataclass(hint):
        hints = typing.get_type_hints(hint)
        return hint(**{k: _from_jsonable(hints[k], v) for k, v in value.items()})
    if origin is tuple:
        inner = args[0] if args else float
        return tuple(_from_jsonable(inner, v) for v in value)
    return value


def config_from_dict(d: dict) -> GeneratorConfig:
    return _from_jsonable(GeneratorConfig, d)
