"""Small set of physics-guided vibration features for milling."""

from __future__ import annotations

import math

import numpy as np


def spindle_frequency_hz(speed_rpm: float | None) -> float:
    if speed_rpm is None or not math.isfinite(float(speed_rpm)) or speed_rpm <= 0:
        return 0.0
    return float(speed_rpm) / 60.0


def tooth_passing_frequency_hz(speed_rpm: float | None, tooth_count: int = 4) -> float:
    return spindle_frequency_hz(speed_rpm) * max(1, int(tooth_count))


def normalized_band_energy(
    signal: np.ndarray,
    sample_rate_hz: float,
    center_hz: float,
    *,
    half_width_hz: float = 5.0,
) -> float:
    """Return spectrum energy near a center frequency, normalized by total energy."""

    if center_hz <= 0 or sample_rate_hz <= 0:
        return 0.0

    x = np.asarray(signal, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0

    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(x)) ** 2
    total = float(np.sum(power[1:]))
    if total <= 0:
        return 0.0

    band = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    return float(np.sum(power[band]) / total)


def dominant_frequency_hz(signal: np.ndarray, sample_rate_hz: float) -> float:
    x = np.asarray(signal, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 8 or sample_rate_hz <= 0:
        return 0.0
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    amplitude = np.abs(np.fft.rfft(x))
    if amplitude.size <= 1:
        return 0.0
    amplitude[0] = 0.0
    return float(freqs[int(np.argmax(amplitude))])


def extract_physics_features(
    signal: np.ndarray,
    sample_rate_hz: float,
    *,
    speed_rpm: float | None,
    tooth_count: int = 4,
) -> dict[str, float]:
    """Extract physics-guided features around spindle and tooth-passing bands."""

    spindle_hz = spindle_frequency_hz(speed_rpm)
    tpf_hz = tooth_passing_frequency_hz(speed_rpm, tooth_count)
    dominant_hz = dominant_frequency_hz(signal, sample_rate_hz)
    half_width = max(3.0, 0.08 * tpf_hz) if tpf_hz else 5.0

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator > 0 else 0.0

    return {
        "physics_spindle_freq_hz": spindle_hz,
        "physics_tooth_passing_freq_hz": tpf_hz,
        "physics_spindle_band_energy": normalized_band_energy(
            signal, sample_rate_hz, spindle_hz, half_width_hz=max(2.0, half_width / 2.0)
        ),
        "physics_tpf_band_energy": normalized_band_energy(signal, sample_rate_hz, tpf_hz, half_width_hz=half_width),
        "physics_2x_tpf_band_energy": normalized_band_energy(
            signal, sample_rate_hz, 2.0 * tpf_hz, half_width_hz=half_width
        ),
        "physics_dominant_to_spindle_ratio": ratio(dominant_hz, spindle_hz),
        "physics_dominant_to_tpf_ratio": ratio(dominant_hz, tpf_hz),
    }

