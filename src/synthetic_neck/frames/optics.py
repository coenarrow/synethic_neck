"""Blood to colour: the skin pulse changes the blood volume fraction of the dermis, and the diffuse reflectance of the
skin in each camera band changes with it through the absorption of haemoglobin (Jacques 2013). The change is applied
as a factor on the resting colour, which is the Monk swatch, so only the ratio of reflectances matters and the
epidermal melanin, a filter common to both, cancels.

Bands are nominal centre wavelengths: blue 460, green 540, red 610 and the depth camera's near-infrared 850 nm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traces.priors import Normal, Range

BANDS = ("r", "g", "b", "ir")
WAVELENGTH_NM = {"b": 460.0, "g": 540.0, "r": 610.0, "ir": 850.0}

# Molar extinction (cm^-1 / M) of oxy- and deoxy-haemoglobin (Prahl's compilation), and the absorption of whole
# blood at 150 g/L, mu_a = 2.303 * eps * 2.326e-3 M.
_EPS_OXY = {"b": 44480.0, "g": 53236.0, "r": 1506.0, "ir": 1058.0}
_EPS_DEOXY = {"b": 23388.8, "g": 46592.0, "r": 9443.6, "ir": 691.32}
_M_HB = 150.0 / 64500.0
MU_A_OXY = {k: 2.303 * v * _M_HB for k, v in _EPS_OXY.items()}        # cm^-1, whole blood
MU_A_DEOXY = {k: 2.303 * v * _M_HB for k, v in _EPS_DEOXY.items()}
MU_A_WATER = {"b": 1.1e-4, "g": 4.5e-4, "r": 2.7e-3, "ir": 4.3e-2}     # cm^-1, Hale and Querry
WATER_FRACTION = 0.65
SCATTER_A, SCATTER_B = 46.0, 1.421                                     # skin: mu_s' = a (lambda / 500 nm)^-b, Jacques 2013
REFRACTIVE_INDEX = 1.4


def baseline_mu_a(band: str) -> float:
    """Bloodless, melanin-free skin absorption (cm^-1), Jacques' skin baseline 0.244 + 85.3 exp(-(lambda - 154) / 66.2)."""
    return 0.244 + 85.3 * np.exp(-(WAVELENGTH_NM[band] - 154.0) / 66.2)


def mu_s_prime(band: str) -> float:
    return SCATTER_A * (WAVELENGTH_NM[band] / 500.0) ** (-SCATTER_B)


def skin_mu_a(band: str, blood_frac, saturation) -> np.ndarray:
    """Absorption (cm^-1) of dermis with blood volume fraction `blood_frac` at oxygen saturation `saturation`."""
    blood = np.asarray(blood_frac, dtype=np.float64)
    return (blood * (saturation * MU_A_OXY[band] + (1.0 - saturation) * MU_A_DEOXY[band])
            + WATER_FRACTION * MU_A_WATER[band] + baseline_mu_a(band))


def diffuse_reflectance(mu_a, mu_s_p: float, n: float = REFRACTIVE_INDEX) -> np.ndarray:
    """Total diffuse reflectance of a semi-infinite medium (Jacques 1998): with transport albedo a' = mu_s' / (mu_a + mu_s')
    and the internal reflection parameter A for index n, R = (a'/2) (1 + exp(-(4/3) A sqrt(3 (1 - a')))) exp(-sqrt(3 (1 - a')))."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    r_i = 0.6681 + 0.0636 * n + 0.7099 / n - 1.4399 / n**2
    A = (1.0 + r_i) / (1.0 - r_i)
    albedo = mu_s_p / (mu_a + mu_s_p)
    root = np.sqrt(3.0 * (1.0 - albedo))
    return 0.5 * albedo * (1.0 + np.exp(-4.0 / 3.0 * A * root)) * np.exp(-root)


@dataclass(frozen=True)
class OpticsConfig:
    blood_fraction: Normal          # resting dermal blood volume fraction
    pulse_green_fraction: Normal    # peak-to-trough pulse of the green level over a beat, as a fraction of the level
    rest_saturation: Range          # oxygen saturation of the resting dermal blood, arterial and venous mixed


@dataclass(frozen=True)
class Optics:
    blood_fraction: float
    pulse_blood_fraction: float     # peak-to-trough change in the blood fraction over a beat
    rest_saturation: float
    pulse_saturation: float = 0.98  # the added blood is arterial


def draw_optics(cfg: OpticsConfig, rng: np.random.Generator) -> Optics:
    """Draw the resting blood and saturation, then the pulsatile blood fraction that gives the drawn green pulse
    through the sensitivity of the model at that resting state."""
    blood, sat, green = cfg.blood_fraction.draw(rng), cfg.rest_saturation.draw(rng), cfg.pulse_green_fraction.draw(rng)
    sens = pulse_sensitivity(Optics(blood, 0.0, sat))
    return Optics(blood, green / abs(sens["g"]), sat)


def rest_reflectance(opt: Optics) -> dict[str, float]:
    return {k: float(diffuse_reflectance(skin_mu_a(k, opt.blood_fraction, opt.rest_saturation), mu_s_prime(k)))
            for k in BANDS}


def pulse_factor(opt: Optics, skin_pulse) -> dict[str, np.ndarray]:
    """Factor on the resting colour in each band for a dimensionless skin pulse value (peak-to-trough 1, zero mean):
    the reflectance with `pulse_blood_fraction` times the pulse added at arterial saturation, over the resting one."""
    b = np.asarray(skin_pulse, dtype=np.float64)
    out = {}
    for k in BANDS:
        mu_rest = skin_mu_a(k, opt.blood_fraction, opt.rest_saturation)
        d_blood = opt.pulse_blood_fraction * b
        mu = mu_rest + d_blood * (opt.pulse_saturation * MU_A_OXY[k] + (1.0 - opt.pulse_saturation) * MU_A_DEOXY[k])
        out[k] = diffuse_reflectance(mu, mu_s_prime(k)) / diffuse_reflectance(mu_rest, mu_s_prime(k))
    return out


def pulse_sensitivity(opt: Optics) -> dict[str, float]:
    """d ln R / d(blood fraction) in each band at rest, for added arterial blood: the fractional colour change per unit
    of pulsatile blood. Negative, largest in green."""
    eps = 1e-5
    up = pulse_factor(Optics(opt.blood_fraction, eps, opt.rest_saturation, opt.pulse_saturation), 1.0)
    return {k: float(np.log(up[k]) / eps) for k in BANDS}
