import numpy as np

from synthetic_neck.config import GeometryParams, PulseParams, TraceParams
from synthetic_neck.geometry import VesselGeometry, tube_fields
from synthetic_neck.render.pulse import PulseStage, normalise
from synthetic_neck.traces import generate_trace


def _stage(**geom):
    tp = TraceParams(heart_rate_bpm=60, hr_variability=0.0)
    tr = generate_trace(tp)
    g = VesselGeometry(frame_size=120, angle_deg=90, length_px=80, centre_xy=(60, 60), **geom)
    return PulseStage(PulseParams(amplitude_levels=10.0, vein_ratio=0.5, skin_ratio=0.0, resp_gain_frac=0.0,
                                   resp_lift_mm=0.0), GeometryParams(), g, 0.5, tr, tp), g, tr


def test_normalise_maps_to_half_range():
    x = np.linspace(0, 1, 1000)
    n = normalise(x)
    assert n.min() >= -0.5 and n.max() <= 0.5 and abs(n.mean()) < 0.02


def test_wave_propagates_from_caudal_to_cranial_end():
    st, g, _ = _stage()
    w, s = tube_fields(g, "vein")
    caudal = np.unravel_index(np.argmax((s < 1) * w), s.shape)
    cranial = np.unravel_index(np.argmax((s > g.length_px - 1) * w), s.shape)
    assert st.delay_vein[cranial] > st.delay_vein[caudal]
    assert st.delay_art[cranial] > st.delay_art[caudal]
    times = np.arange(0, 10, 1 / 1000)
    pv_caudal = np.interp(times - st.delay_vein[caudal], st.t, st.cvp_n)
    pv_cranial = np.interp(times - st.delay_vein[cranial], st.t, st.cvp_n)
    lag = np.argmax(np.correlate(pv_cranial - pv_cranial.mean(), pv_caudal - pv_caudal.mean(), "full")) - (len(times) - 1)
    assert lag > 0


def test_modulation_is_negative_with_pressure_and_scaled_by_amplitude():
    st, g, tr = _stage()
    w, _ = tube_fields(g, "artery")
    px = np.unravel_index(np.argmax(w), w.shape)
    times = np.arange(300) / 30.0
    green = np.array([st.rgb_mod(t)[px][1] for t in times])
    abp = np.array([np.interp(t - st.delay_art[px] + st.site_delay_s, st.t, tr[:, 1]) for t in times])
    assert np.corrcoef(green, abp)[0, 1] < -0.9
    assert 9 <= np.ptp(green) <= 11
    lift = np.array([st.depth_lift_mm(t)[px] for t in times])
    assert np.corrcoef(lift, abp)[0, 1] > 0.9 and 0.2 < np.ptp(lift) < 0.4


def test_vein_taper_silences_cranial_end():
    st, g, _ = _stage(vein_visible_fraction=0.5)
    w, s = tube_fields(g, "vein")
    caudal = np.unravel_index(np.argmax((s < 1) * w), s.shape)
    cranial = np.unravel_index(np.argmax((s > g.length_px - 1) * w), s.shape)
    assert st.w_vein[caudal] > 0.9 and st.w_vein[cranial] < 0.01
