"""The generator config, loaded from a YAML priors file (default priors/base.yaml).

YAML keys are UPPER_CASE and mirror the dataclass fields in lower case. A scalar is a fixed value; a
[lo, hi] pair becomes a Range (uniform), or an IntRange (uniform integers, inclusive) where the field asks for one; a
{MEAN, SD, MIN?, MAX?} mapping becomes a Normal (truncated by clipping). Sections (RESPIRATORY, ECG, ABP, CVP, PPG, CAMERA, GEOMETRY, PROPAGATION,
APPEARANCE, DISTENSION, OPTICS, SENSOR) become the nested dataclasses that each
module owns.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .frames.appearance import AppearanceConfig
from .frames.camera import CameraConfig
from .frames.distension import DistensionConfig
from .frames.optics import OpticsConfig
from .frames.propagation import PropagationConfig
from .frames.scene import GeometryConfig
from .frames.sensor import SensorConfig
from .traces.abp import AbpConfig
from .traces.cvp import CvpConfig
from .traces.ecg import EcgConfig
from .traces.ppg import PpgConfig
from .traces.priors import IntRange, Normal, Range
from .traces.respiratory import RespConfig

DEFAULT_PRIORS = Path(__file__).resolve().parents[2] / "priors" / "base.yaml"


@dataclass(frozen=True)
class Config:
    duration_s: float
    sample_rate_hz: float
    respiratory: RespConfig
    ecg: EcgConfig
    abp: AbpConfig
    cvp: CvpConfig
    ppg: PpgConfig
    camera: CameraConfig
    geometry: GeometryConfig
    propagation: PropagationConfig
    appearance: AppearanceConfig
    distension: DistensionConfig
    optics: OpticsConfig
    sensor: SensorConfig


def _build(cls, d: dict[str, Any], where: str):
    """Build dataclass `cls` from the YAML mapping `d`, matching UPPER_CASE keys to lower_case fields."""
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {d!r}")
    by_field = {k.lower(): k for k in d}
    names = {f.name for f in fields(cls)}
    unknown, missing = set(by_field) - names, names - set(by_field)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(by_field[k] for k in unknown)}, "
                         f"missing keys {sorted(k.upper() for k in missing)}")
    out, hints = {}, get_type_hints(cls)
    for f in fields(cls):
        value, key = d[by_field[f.name]], f"{where}.{by_field[f.name]}"
        hint = hints[f.name]
        if is_dataclass(hint) and hint not in (Range, IntRange, Normal):
            out[f.name] = _build(hint, value, key)
        elif hint is Range:
            if not (isinstance(value, list) and len(value) == 2):
                raise ValueError(f"{key}: expected [lo, hi], got {value!r}")
            out[f.name] = Range(float(value[0]), float(value[1]))
        elif hint is IntRange:
            if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value)):
                raise ValueError(f"{key}: expected [lo, hi] integers, got {value!r}")
            out[f.name] = IntRange(int(value[0]), int(value[1]))
        elif hint is Normal:
            if not (isinstance(value, dict) and {"MEAN", "SD"} <= set(value) <= {"MEAN", "SD", "MIN", "MAX"}):
                raise ValueError(f"{key}: expected {{MEAN, SD, MIN?, MAX?}}, got {value!r}")
            out[f.name] = Normal(float(value["MEAN"]), float(value["SD"]),
                                 None if "MIN" not in value else float(value["MIN"]),
                                 None if "MAX" not in value else float(value["MAX"]))
        else:
            if isinstance(value, (list, dict)):
                raise ValueError(f"{key}: expected a single number, got {value!r}")
            out[f.name] = float(value)
    return cls(**out)


def load(path: Path = DEFAULT_PRIORS) -> Config:
    """Load a Config from `path`. Every field must be present; unknown keys are an error."""
    path = Path(path)
    return _build(Config, yaml.safe_load(path.read_text()), path.name)
