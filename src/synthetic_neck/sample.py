"""One sample: every prior drawn once from one seed, the traces on the padded grid, the scene and its fields.

`draw_sample` is the whole of trace_generation.md and the drawing half of frame_rendering.md; `frame_at` renders and
records one frame; `stored_traces` gives the five ground-truth traces on the frame clock. The drawing order is fixed,
so a seed reproduces a sample exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .frames.appearance import Appearance, BaseFrame, base_frame, draw_appearance
from .frames.camera import Camera, draw_camera
from .frames.distension import Distension, VolumeField, draw_distension, skin_pulse
from .frames.optics import Optics, draw_optics
from .frames.propagation import Propagation, PulseField, carotid_pressure, draw_propagation
from .frames.render import render_frame
from .frames.resample import resample_labels
from .frames.scene import Scene, SceneMaps, draw_scene, scene_maps
from .frames.sensor import SensedFrame, Sensor, draw_sensor, sense
from .traces.abp import abp
from .traces.cvp import cvp
from .traces.ecg import ecg, r_wave_times
from .traces.grid import TRACE_HZ, crop, resample, time_grid
from .traces.ppg import ppg
from .traces.respiratory import respiration, respiratory_rate

ABP_SITES = ("brachial", "radial")
TRACE_UNITS = {"abp": "mmHg", "cvp": "mmHg", "ecg": "mV", "ppg": "arb", "rr": "arb"}


@dataclass(frozen=True)
class Traces:
    """The traces on the padded 1 kHz grid `t`."""
    t: np.ndarray
    resp: np.ndarray                # respiratory waveform x(t) in [-1, 1]
    r_times: np.ndarray
    ecg: np.ndarray
    abp: np.ndarray                 # at the drawn catheter site, with its noise
    cvp: np.ndarray                 # with its noise
    ppg: np.ndarray                 # finger
    carotid: np.ndarray             # carotid pressure at the neck, no noise: what the artery reads
    cvp_clean: np.ndarray           # CVP without noise: what the vein reads
    skin: np.ndarray                # dimensionless skin pulse


@dataclass(frozen=True)
class Sample:
    seed: int
    output_px: int
    draws: dict[str, Any]           # every scalar drawn for the traces, by name
    abp_site: str
    traces: Traces
    camera: Camera
    scene: Scene
    propagation: Propagation
    distension: Distension
    appearance: Appearance
    optics: Optics
    sensor: Sensor
    maps: SceneMaps = field(repr=False)
    base: BaseFrame = field(repr=False)
    volume: VolumeField = field(repr=False)
    rng: np.random.Generator = field(repr=False)     # continues the seed's stream for the sensor noise

    @property
    def n_frames(self) -> int:
        return len(self.frame_times_s)

    @property
    def frame_times_s(self) -> np.ndarray:
        """The frame clock: j / f_s for 0 <= j < T f_s."""
        return np.arange(int(round(self._duration_s * self.sample_rate_hz))) / self.sample_rate_hz

    @property
    def _duration_s(self) -> float:
        return float(self.draws["duration_s"])

    @property
    def sample_rate_hz(self) -> float:
        return float(self.draws["sample_rate_hz"])

    @property
    def labels(self) -> np.ndarray:
        """Vessel labels at the delivered size: 0 background, 1 artery, 2 vein."""
        return resample_labels(self.maps.labels, self.output_px)

    @property
    def neck_mask(self) -> np.ndarray:
        return resample_labels(self.maps.on_neck.astype(np.uint8), self.output_px).astype(bool)


def draw_traces(cfg: Config, rng: np.random.Generator) -> tuple[dict[str, Any], str, Traces, Propagation, Distension]:
    """All trace draws and the traces themselves, in trace_generation.md order, then the propagation and
    distension draws the neck-side traces need."""
    d = {"duration_s": cfg.duration_s, "sample_rate_hz": cfg.sample_rate_hz}
    t = time_grid(cfg.duration_s)
    r, e, a, c, p = cfg.respiratory, cfg.ecg, cfg.abp, cfg.cvp, cfg.ppg
    d["resp_rate_bpm"], d["resp_rate_drift_frac"], d["resp_rate_drift_tau_s"] = (
        r.rate_bpm.draw(rng), r.rate_drift_frac.draw(rng), r.rate_drift_tau_s.draw(rng))
    d["resp_phase_rad"] = float(rng.uniform(0, 2 * np.pi))
    rate = respiratory_rate(t, d["resp_rate_bpm"], d["resp_rate_drift_frac"], d["resp_rate_drift_tau_s"], rng)
    resp = respiration(t, rate, d["resp_phase_rad"])

    d["heart_rate_bpm"], d["hr_variability"], d["rsa_fraction"], d["mayer_fraction"] = (
        e.heart_rate_bpm.draw(rng), e.hr_variability.draw(rng), e.rsa_fraction.draw(rng), e.mayer_fraction.draw(rng))
    r_times = r_wave_times(t, resp, d["heart_rate_bpm"], d["hr_variability"], d["rsa_fraction"], d["mayer_fraction"], rng)
    d["ecg_r_amplitude"], d["ecg_am_fraction"], d["ecg_wander"], d["ecg_noise"] = (
        e.r_amplitude.draw(rng), e.am_fraction.draw(rng), e.wander.draw(rng), e.noise.draw(rng))
    ecg_tr = ecg(t, r_times, resp, d["heart_rate_bpm"], d["ecg_r_amplitude"], d["ecg_am_fraction"], d["ecg_wander"],
                 d["ecg_noise"], rng)

    d["map_mmhg"], d["pulse_pressure_mmhg"], d["pre_ejection_s"] = a.map_mmhg.draw(rng), a.pulse_pressure_mmhg.draw(rng), a.pre_ejection_s.draw(rng)
    d["aortic_to_brachial_s"], d["brachial_to_radial_s"], d["radial_amplification"] = (
        a.aortic_to_brachial_s.draw(rng), a.brachial_to_radial_s.draw(rng), a.radial_amplification.draw(rng))
    d["abp_resp_swing_mmhg"], d["abp_noise_mmhg"] = a.resp_swing_mmhg.draw(rng), a.noise_mmhg.draw(rng)
    site = ABP_SITES[int(rng.integers(0, 2))]
    transit = d["aortic_to_brachial_s"] + (d["brachial_to_radial_s"] if site == "radial" else 0.0)
    amplification = d["radial_amplification"] if site == "radial" else 1.0
    abp_tr = abp(t, r_times, resp, d["heart_rate_bpm"], cfg.duration_s, d["map_mmhg"], d["pulse_pressure_mmhg"],
                 d["pre_ejection_s"], transit, site, amplification, d["abp_resp_swing_mmhg"], d["abp_noise_mmhg"], rng)

    d["cvp_mean_mmhg"], d["cvp_pulse_mmhg"] = c.mean_mmhg.draw(rng), c.pulse_mmhg.draw(rng)
    d["cvp_c_to_a"], d["cvp_v_to_a"], d["cvp_x_to_a"], d["cvp_y_to_a"] = (
        c.c_to_a.draw(rng), c.v_to_a.draw(rng), c.x_to_a.draw(rng), c.y_to_a.draw(rng))
    d["cvp_p_to_a_s"], d["cvp_resp_swing_mmhg"], d["cvp_noise_mmhg"] = c.p_to_a_s.draw(rng), c.resp_swing_mmhg.draw(rng), c.noise_mmhg.draw(rng)
    cvp_args = (t, r_times, resp, d["heart_rate_bpm"], cfg.duration_s, d["cvp_mean_mmhg"], d["cvp_pulse_mmhg"],
                d["cvp_c_to_a"], d["cvp_v_to_a"], d["cvp_x_to_a"], d["cvp_y_to_a"], d["pre_ejection_s"],
                d["cvp_p_to_a_s"], d["cvp_resp_swing_mmhg"])
    cvp_clean = cvp(*cvp_args, 0.0, np.random.default_rng(0))
    cvp_tr = cvp(*cvp_args, d["cvp_noise_mmhg"], rng)

    d["ppg_amplitude"], d["radial_to_finger_s"] = p.amplitude.draw(rng), p.radial_to_finger_s.draw(rng)
    d["ppg_am_fraction"], d["ppg_wander"], d["ppg_noise"] = p.am_fraction.draw(rng), p.wander.draw(rng), p.noise.draw(rng)
    ppg_tr = ppg(t, r_times, resp, d["heart_rate_bpm"], cfg.duration_s, d["ppg_amplitude"], d["pre_ejection_s"],
                 d["aortic_to_brachial_s"] + d["brachial_to_radial_s"] + d["radial_to_finger_s"],
                 d["ppg_am_fraction"], d["ppg_wander"], d["ppg_noise"], rng)

    prop, dist = draw_propagation(cfg.propagation, rng), draw_distension(cfg.distension, rng)
    carotid = carotid_pressure(t, r_times, resp, d["heart_rate_bpm"], cfg.duration_s, d["map_mmhg"],
                               d["pulse_pressure_mmhg"], d["pre_ejection_s"], prop, d["abp_resp_swing_mmhg"])
    skin = skin_pulse(t, r_times, resp, d["heart_rate_bpm"], cfg.duration_s, d["pre_ejection_s"], prop, dist,
                      d["ppg_am_fraction"], d["ppg_wander"])
    traces = Traces(t, resp, r_times, ecg_tr, abp_tr, cvp_tr, ppg_tr, carotid, cvp_clean, skin)
    return d, site, traces, prop, dist


def draw_sample(cfg: Config, seed: int, output_px: int | None = None) -> Sample:
    """Draw everything for one sample from `seed` and build its fields. Frames are rendered on the native crop and
    delivered at `output_px` (the crop size when None)."""
    rng = np.random.default_rng(seed)
    draws, site, traces, prop, dist = draw_traces(cfg, rng)
    camera = draw_camera(cfg.camera, rng, output_px)
    scene = draw_scene(cfg.geometry, rng)
    app = draw_appearance(cfg.appearance, rng)
    opt = draw_optics(cfg.optics, rng)
    sensor = draw_sensor(cfg.sensor, rng)
    maps = scene_maps(scene, camera, camera.crop_px)
    base = base_frame(scene, maps, app, camera.output_px, rng)
    pulse = PulseField(traces.t, traces.carotid, traces.cvp_clean, scene, maps, prop)
    vol = VolumeField(pulse, scene, maps, dist, cfg.duration_s, traces.skin)
    return Sample(seed, camera.output_px, draws, site, traces, camera, scene, prop, dist, app, opt, sensor,
                  maps, base, vol, rng)


def frame_at(sample: Sample, time_s: float) -> SensedFrame:
    """Render and record the frame at `time_s`."""
    f = render_frame(time_s, sample.maps, sample.appearance, sample.base, sample.volume, sample.optics)
    return sense(f, sample.sensor, sample.output_px, sample.rng)


def stored_traces(sample: Sample) -> dict[str, np.ndarray]:
    """The five ground-truth traces on the frame clock: cropped to [0, T) and resampled to the sample rate."""
    tr = sample.traces
    fs = sample.sample_rate_hz
    out = {}
    for name, x in (("abp", tr.abp), ("cvp", tr.cvp), ("ecg", tr.ecg), ("ppg", tr.ppg), ("rr", tr.resp)):
        out[name] = resample(crop(tr.t, x, sample._duration_s), TRACE_HZ, fs)[:sample.n_frames]
    return out


def describe(sample: Sample) -> dict[str, Any]:
    """Everything drawn, as plain values for the metadata."""
    def plain(obj):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(obj).items()}
    vol = sample.volume
    return {
        "seed": sample.seed,
        "output_px": sample.output_px,
        "n_frames": sample.n_frames,
        "abp_site": sample.abp_site,
        "posture": sample.scene.posture,
        "traces": sample.draws,
        "camera": plain(sample.camera),
        "scene": plain(sample.scene),
        "propagation": plain(sample.propagation),
        "distension": plain(sample.distension),
        "appearance": plain(sample.appearance),
        "optics": plain(sample.optics),
        "sensor": plain(sample.sensor),
        "derived": {
            "pixel_scale_mm": sample.camera.pixel_scale_mm,
            "artery_axis_depth_mm": vol.artery_axis_depth_mm,
            "vein_axis_depth_mm": vol.vein_axis_depth_mm,
            "artery_compliance_mm2_per_mmhg": vol.artery_compliance_mm2_per_mmhg,
            "vein_compliance_mm2_per_mmhg": vol.vein_compliance_mm2_per_mmhg,
            "carotid_mean_mmhg": vol.artery_mean_mmhg,
            "cvp_mean_mmhg": vol.vein_mean_mmhg,
            "artery_lift_ptp_mm": float(np.ptp(sample.traces.carotid) * vol.artery_compliance_mm2_per_mmhg
                                        / (np.pi * vol.artery_axis_depth_mm)),
            "vein_lift_ptp_mm": float(np.ptp(sample.traces.cvp_clean) * vol.vein_compliance_mm2_per_mmhg
                                      / (np.pi * vol.vein_axis_depth_mm)),
        },
    }
