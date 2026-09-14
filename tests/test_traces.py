import numpy as np

from synthetic_neck.config import GeneratorConfig, TraceParams, sample
from synthetic_neck.traces import generate_trace, read_trace_csv, write_trace_csv


def test_trace_shape_and_ranges():
    p = TraceParams(systolic_mmhg=130, diastolic_mmhg=80, cvp_mean_mmhg=6)
    tr = generate_trace(p)
    assert tr.shape == (10001, 3)
    t, abp, cvp = tr.T
    assert t[0] == 0 and abs(t[-1] - 10.0) < 1e-9
    assert 60 < abp.min() and abp.max() < 145
    assert 2 <= cvp.min() and cvp.max() <= 20
    assert abs(abp.max() - 130) < 4 and abs(abp.min() - 80) < 4


def test_trace_heart_rate_is_respected():
    p = TraceParams(heart_rate_bpm=90, hr_variability=0.0)
    abp = generate_trace(p)[:, 1]
    above = abp > (abp.min() + 0.6 * (abp.max() - abp.min()))
    rises = np.flatnonzero(np.diff(above.astype(int)) == 1)
    rises = rises[np.insert(np.diff(rises) > 300, 0, True)]  # 300 ms refractory vs noise
    assert abs(len(rises) - 15) <= 1


def test_cvp_a_wave_precedes_arterial_upstroke():
    p = TraceParams(heart_rate_bpm=60, hr_variability=0.0, abp_upstroke_delay_s=0.10)
    t, abp, cvp = generate_trace(p).T
    win = (t > 2.5) & (t < 3.5)      # R-wave lands at t=3.0 by construction
    t_abp_peak = t[win][np.argmax(abp[win])]
    pre = (t > t_abp_peak - 0.35) & (t < t_abp_peak - 0.05)
    t_a = t[pre][np.argmax(cvp[pre])]
    assert t_a < t_abp_peak


def test_wave_amplitudes_scale_cvp_pulse_pressure():
    small = TraceParams(hr_variability=0.0, a_wave_mmhg=1.0, v_wave_mmhg=1.0, cvp_noise_mmhg=0.0,
                        resp_cvp_swing_mmhg=0.0)
    big = TraceParams(hr_variability=0.0, a_wave_mmhg=4.0, v_wave_mmhg=4.0, cvp_noise_mmhg=0.0,
                      resp_cvp_swing_mmhg=0.0)
    pp = lambda p: np.ptp(generate_trace(p)[:, 2])
    assert pp(big) > 1.5 * pp(small)


def test_sampled_cvp_stays_within_2_to_20_mmhg():
    rng = np.random.default_rng(3)
    for _ in range(20):
        cvp = generate_trace(sample(GeneratorConfig(), rng).trace)[:, 2]
        assert 2 <= cvp.min() and cvp.max() <= 20
        assert 4 < cvp.mean() < 17


def test_trace_csv_round_trip(tmp_path):
    tr = generate_trace(TraceParams())
    write_trace_csv(tr, tmp_path / "trace.csv")
    back = read_trace_csv(tmp_path / "trace.csv")
    assert (tmp_path / "trace.csv").read_text().splitlines()[0] == "Time,ABP,CVP"
    np.testing.assert_allclose(back, tr, atol=1e-3)
