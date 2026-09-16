import csv
import json

import numpy as np
import pytest
from conftest import needs_ffmpeg

from synthetic_neck.calibrate import (
    appearance_priors, config_from_priors, detect_r_peaks, histogram, population_priors, quantiles,
    run_calibration, skin_mask, waveform_priors_for_recording,
)
from synthetic_neck.config import TraceParams, validate
from synthetic_neck.traces import generate_trace
from synthetic_neck.video import MkvWriter

INFO_COLUMNS = ["Recording_Directory", "Participant_ID", "K1_RGB.mkv", "K1_IR.mkv", "ECG (mV)", "CVP (mmHg)", "Sex",
                "Age (years)", "Neck_Circumference_Mid (cm)", "Skin_Tone_Clinician", "Skin_Tone_Recorder",
                "BP_Sys (mmHG)", "BP_Dia (mmHg)", "JVP Height Estimate (cm)", "Average_BPM"]


def _row(pid, tone, posture, jvp, bpm, depth="D"):
    name = f"P{pid:03d}_S01_R1_{posture}_{depth}"
    return dict(zip(INFO_COLUMNS, [name, str(pid), "TRUE", "TRUE" if depth == "D" else "FALSE", "TRUE", "TRUE", "M",
                                   "60", "40", str(tone), str(tone), "120", "70", jvp, str(bpm)]))


def _write_stand_in_root(root):
    rows = [_row(1, 4, 0, "3", 70), _row(2, 6, 45, "Not Visible", 85, "N"), _row(3, 2, 90, "5", 60)]
    (root / "data").mkdir(parents=True)
    with (root / "dataset_info.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, INFO_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    rng = np.random.default_rng(0)
    for r in rows:
        d = root / "data" / r["Recording_Directory"]
        d.mkdir()
        fs = 2000.0
        p = TraceParams(duration_s=10.0, sample_rate_hz=fs, heart_rate_bpm=float(r["Average_BPM"]), hr_variability=0.0,
                        rsa_fraction=0.0, cvp_mean_mmhg=8.0, seed=1)
        tr = generate_trace(p)
        t = tr[:, 0]
        rr = 60.0 / p.heart_rate_bpm
        onsets = np.arange(-2.0, 12.0, rr)          # matches beat_onsets with zero variability and zero RSA
        ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in onsets) + 20 * rng.standard_normal(t.shape)
        with (d / "trace_data.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Time (s)", "CVP (mmHg)", "ECG (mV)"])
            for ti, ci, ei in zip(t, tr[:, 2], ecg):
                w.writerow([f"{ti:.5f}", f"{ci:.4f}", f"{ei:.3f}"])
        skin = np.array([180, 130, 100]) if r["Skin_Tone_Clinician"] != "6" else np.array([110, 75, 55])
        with MkvWriter(d / "K1_RGB.mkv", (64, 64), "rgb", 30.0) as w:
            for _ in range(15):
                frame = skin[None, None, :] + rng.standard_normal((64, 64, 3)) * 2.0
                frame[:8] = (40, 80, 160)       # a blue gown strip
                w.write(np.clip(frame, 0, 255).astype(np.uint8))
        if r["Recording_Directory"].endswith("_D"):
            with MkvWriter(d / "K1_IR.mkv", (64, 64), "gray16", 30.0) as w:
                for _ in range(15):
                    w.write(np.clip(3000 + rng.standard_normal((64, 64)) * 40, 0, 65535).astype(np.uint16))
    return rows


def test_quantiles_and_histogram():
    q = quantiles([1, 2, 3, 4, 5])
    assert q["p50"] == 3 and q["p05"] < q["p25"] < q["p75"] < q["p95"]
    h = histogram(["a", "b", "a"])
    assert h == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_population_priors_use_monk_and_posture_and_skip_not_visible(tmp_path):
    rows = _write_stand_in_root(tmp_path)
    pop = population_priors(rows)
    assert set(pop["monk_tone"]) == {"2", "4", "6"}
    assert set(pop["posture_deg"]) == {"0", "45", "90"}
    assert pop["depth_ir_share"] == pytest.approx(2 / 3)
    assert pop["jvp_height_cm"]["p50"] == 4.0                   # 3 and 5; "Not Visible" excluded
    assert pop["heart_rate_bpm"]["p50"] == 70


def test_pulse_pressure_pairs_within_rows(tmp_path):
    rows = _write_stand_in_root(tmp_path)
    rows[0]["BP_Sys (mmHG)"], rows[0]["BP_Dia (mmHg)"] = "120", "70"
    rows[1]["BP_Sys (mmHG)"], rows[1]["BP_Dia (mmHg)"] = "N/A", "60"
    rows[2]["BP_Sys (mmHG)"], rows[2]["BP_Dia (mmHg)"] = "130", "80"
    pop = population_priors(rows)
    assert pop["pulse_pressure_mmhg"]["p50"] == 50.0      # zipping filtered columns would give 60
    assert pop["bp_dia_mmhg"]["p50"] == 75.0


def test_detect_r_peaks_on_stand_in_ecg():
    fs = 2000.0
    t = np.arange(0, 10, 1 / fs)
    ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in np.arange(0.5, 10, 0.8))
    ecg += 20 * np.random.default_rng(0).standard_normal(t.shape)
    peaks = detect_r_peaks(ecg, fs)
    assert len(peaks) == 12 and abs(np.diff(peaks).mean() / fs - 0.8) < 0.01


@needs_ffmpeg
def test_waveform_priors_recover_cvp_shape(tmp_path):
    _write_stand_in_root(tmp_path)
    w = waveform_priors_for_recording(tmp_path / "data" / "P001_S01_R1_0_D" / "trace_data.csv")
    assert abs(w["heart_rate_bpm"] - 70) < 1 and w["hr_variability"] < 0.02
    assert 7 < w["cvp_mean_mmhg"] < 9
    assert w["a_wave_mmhg"] > 1.0 and w["v_wave_mmhg"] > 1.0 and w["x_descent_mmhg"] > 0.5
    assert 0.5 < w["resp_cvp_swing_mmhg"] < 3.0


def test_skin_mask_excludes_gown_and_black():
    rgb = np.zeros((20, 20, 3))
    rgb[:, :10] = (180, 130, 100)
    rgb[:, 10:] = (40, 80, 160)
    m = skin_mask(rgb, central=False)
    assert m[:, :10].all() and not m[:, 10:].any()


@needs_ffmpeg
def test_run_calibration_writes_schema_and_neckflix_config_validates(tmp_path):
    _write_stand_in_root(tmp_path)
    out = tmp_path / "priors.json"
    priors = run_calibration(tmp_path, out, n_recordings=3, seed=0)
    d = json.loads(out.read_text())
    assert d == priors
    assert set(d) == {"schema_version", "provenance", "population", "waveform", "appearance"}
    assert d["provenance"]["n_recordings"] == 3 and "Recording_Directory" not in json.dumps(d)
    assert "P001" not in json.dumps(d)
    assert d["appearance"]["skin_rgb_by_monk"] == {}
    assert 0.5 < d["appearance"]["read_noise_sd"]["p50"] < 4.0
    cfg = config_from_priors(out)
    validate(cfg)
    assert cfg.pulse.amplitude_levels.lo == 0.5 and cfg.pulse.amplitude_levels.hi == 2.0
    assert cfg.geometry.vein_visible_fraction.lo == 0.4
    assert cfg.appearance.monk_tone is not None and len(cfg.appearance.skin_rgb_by_monk) == 10
    assert cfg.streams.depth_ir_probability == pytest.approx(2 / 3)
    assert cfg.sensor.read_noise_sd.knots is not None
    assert cfg.sensor.read_noise_sd.hi <= 6.0


@needs_ffmpeg
def test_config_from_priors_scales_noise_to_output_pixels(tmp_path):
    _write_stand_in_root(tmp_path)
    out = tmp_path / "priors.json"
    run_calibration(tmp_path, out, n_recordings=3, seed=0)
    d = json.loads(out.read_text())
    d["appearance"]["read_noise_sd"] = {"p05": 2.17, "p25": 4.34, "p50": 6.51, "p75": 8.68, "p95": 10.85}
    out.write_text(json.dumps(d))
    cfg = config_from_priors(out)
    knots = cfg.sensor.read_noise_sd.knots
    assert knots is not None
    for actual, expected in zip(knots, (1.0, 2.0, 3.0, 4.0, 5.0)):
        assert abs(actual - expected) < 0.02


def _write_trace(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in zip(*(columns[n] for n in names)):
            w.writerow([f"{x:.5f}" for x in row])


def _synthetic_trace(hr, **waves):
    fs = 2000.0
    p = TraceParams(duration_s=20.0, sample_rate_hz=fs, heart_rate_bpm=hr, hr_variability=0.0, rsa_fraction=0.0,
                    cvp_mean_mmhg=8.0, cvp_noise_mmhg=0.15, resp_cvp_swing_mmhg=1.5, seed=3, **waves)
    tr = generate_trace(p)
    t = tr[:, 0]
    ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in np.arange(-2.0, 22.0, 60.0 / hr))
    return t, tr[:, 2], tr[:, 1], ecg


def test_waveform_priors_select_columns_by_name(tmp_path):
    t, cvp, abp, ecg = _synthetic_trace(70)
    path = tmp_path / "trace_data.csv"
    _write_trace(path, {"Time (s)": t, "CVP (mmHg)": cvp, "ABP (mmHg)": abp, "ECG (mV)": ecg})
    w = waveform_priors_for_recording(path)
    assert abs(w["heart_rate_bpm"] - 70) < 1 and 7 < w["cvp_mean_mmhg"] < 9
    bad = tmp_path / "no_ecg.csv"
    _write_trace(bad, {"Time (s)": t, "CVP (mmHg)": cvp})
    with pytest.raises(ValueError, match="lacks columns"):
        waveform_priors_for_recording(bad)


def test_waveform_priors_recover_wave_amplitudes_at_fast_heart_rate(tmp_path):
    truth = dict(a_wave_mmhg=2.5, c_wave_mmhg=1.0, x_descent_mmhg=1.8, v_wave_mmhg=1.0, y_descent_mmhg=1.2)
    t, cvp, _, ecg = _synthetic_trace(110, **truth)
    path = tmp_path / "trace_data.csv"
    _write_trace(path, {"Time (s)": t, "CVP (mmHg)": cvp, "ECG (mV)": ecg})
    w = waveform_priors_for_recording(path)
    for k, v in truth.items():
        assert abs(w[k] - v) < 0.3, (k, w[k], v)


@needs_ffmpeg
def test_run_calibration_skips_unusable_recordings(tmp_path):
    _write_stand_in_root(tmp_path)
    bad = tmp_path / "data" / "P003_S01_R1_90_D" / "trace_data.csv"
    lines = bad.read_text().splitlines()
    bad.write_text("\n".join(lines[:1001]) + "\n")        # 0.5 s of signal: too few beats
    priors = run_calibration(tmp_path, tmp_path / "priors.json", n_recordings=3, seed=0)
    assert priors["provenance"]["n_recordings"] == 2 and priors["provenance"]["n_skipped"] == 1


def test_waveform_priors_ignore_missed_beats(tmp_path):
    hr = 70
    fs = 2000.0
    p = TraceParams(duration_s=20.0, sample_rate_hz=fs, heart_rate_bpm=hr, hr_variability=0.0, rsa_fraction=0.0,
                    cvp_mean_mmhg=8.0, cvp_noise_mmhg=0.15, resp_cvp_swing_mmhg=1.5, seed=3)
    tr = generate_trace(p)
    t, cvp = tr[:, 0], tr[:, 2]
    onsets = list(np.arange(-2.0, 22.0, 60.0 / hr))
    del onsets[10]        # a missed 11th beat: the ECG detector sees one ~2x-length RR interval
    ecg = sum(1000.0 * np.exp(-0.5 * ((t - r0) / 0.01) ** 2) for r0 in onsets)
    path = tmp_path / "trace_data.csv"
    _write_trace(path, {"Time (s)": t, "CVP (mmHg)": cvp, "ECG (mV)": ecg})
    w = waveform_priors_for_recording(path)
    assert w["hr_variability"] < 0.02
    assert abs(w["heart_rate_bpm"] - 70) < 1


def test_appearance_priors_require_three_recordings_per_monk_cell():
    rec = lambda rgb: {"skin_rgb": list(rgb), "skin_fraction": 0.2, "texture_sd": 3.0,
                       "read_noise_sd": 2.0, "shot_noise_gain": 0.0}
    recs = [rec((180, 130, 100)), rec((170, 120, 90)), rec((190, 140, 110)), rec((100, 70, 50))]
    out = appearance_priors(recs, [4, 4, 4, 6])
    assert set(out["skin_rgb_by_monk"]) == {"4"}
    assert out["skin_rgb_by_monk"]["4"] == [180.0, 130.0, 100.0]
