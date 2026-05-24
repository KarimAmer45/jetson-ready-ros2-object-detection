"""Windowing and feature extraction for milling sensor signals."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from data_loader import MillingRun
from physics_features import extract_physics_features


WEAR_CLASS_ORDER = ("low", "medium", "high")
METADATA_COLUMNS = {
    "case",
    "run",
    "window_id",
    "window_start",
    "window_end",
    "sample_rate_hz",
    "speed_rpm",
    "feed",
    "depth_of_cut",
    "tooth_count",
    "material",
    "source",
    "VB",
    "wear_class",
}


def wear_class_from_vb(vb: float) -> str | None:
    """Map flank wear VB in mm to a small class label."""

    if vb is None or not math.isfinite(float(vb)):
        return None
    if vb < 0.20:
        return "low"
    if vb < 0.40:
        return "medium"
    return "high"


def iter_windows(
    signals: pd.DataFrame,
    sample_rate_hz: float,
    *,
    window_seconds: float = 1.0,
    overlap: float = 0.5,
) -> Iterable[tuple[int, int, pd.DataFrame]]:
    """Yield start, end, and signal frame for each time window."""

    if signals.empty:
        return

    window_size = max(8, int(round(sample_rate_hz * window_seconds)))
    step = max(1, int(round(window_size * (1.0 - overlap))))
    n = len(signals)

    if n <= window_size:
        yield 0, n, signals.copy()
        return

    start = 0
    while start + window_size <= n:
        end = start + window_size
        yield start, end, signals.iloc[start:end].copy()
        start += step


def extract_window_features(
    window: pd.DataFrame,
    sample_rate_hz: float,
    *,
    channels: Iterable[str] | None = None,
) -> dict[str, float]:
    """Extract time and spectrum features from each numeric signal channel."""

    selected = list(channels) if channels is not None else list(window.columns)
    features: dict[str, float] = {}
    for channel in selected:
        if channel not in window:
            continue
        values = pd.to_numeric(window[channel], errors="coerce").to_numpy(dtype=float)
        features.update(_single_channel_features(channel, values, sample_rate_hz))
    return features


def extract_dataset_features(
    runs: list[MillingRun],
    *,
    include_physics: bool,
    window_seconds: float = 1.0,
    overlap: float = 0.5,
    primary_signal: str = "vib_spindle",
) -> pd.DataFrame:
    """Build one feature row per signal window."""

    rows: list[dict[str, object]] = []
    for milling_run in runs:
        for window_id, (start, end, window) in enumerate(
            iter_windows(
                milling_run.signals,
                milling_run.sample_rate_hz,
                window_seconds=window_seconds,
                overlap=overlap,
            )
        ):
            row: dict[str, object] = extract_window_features(window, milling_run.sample_rate_hz)
            if include_physics:
                signal_name = primary_signal if primary_signal in window else window.columns[0]
                row.update(
                    extract_physics_features(
                        window[signal_name].to_numpy(dtype=float),
                        milling_run.sample_rate_hz,
                        speed_rpm=milling_run.speed_rpm,
                        tooth_count=milling_run.tooth_count,
                    )
                )

            row.update(
                {
                    "case": milling_run.case,
                    "run": milling_run.run,
                    "window_id": window_id,
                    "window_start": start,
                    "window_end": end,
                    "sample_rate_hz": milling_run.sample_rate_hz,
                    "speed_rpm": milling_run.speed_rpm,
                    "feed": milling_run.feed,
                    "depth_of_cut": milling_run.depth_of_cut,
                    "tooth_count": milling_run.tooth_count,
                    "material": milling_run.material,
                    "source": milling_run.source,
                    "VB": milling_run.vb,
                    "wear_class": wear_class_from_vb(milling_run.vb),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def select_feature_columns(df: pd.DataFrame, *, include_physics: bool) -> list[str]:
    """Select numeric feature columns and optionally exclude physics columns."""

    columns = []
    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue
        if not include_physics and column.startswith("physics_"):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def _single_channel_features(channel: str, values: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {
            f"{channel}_rms": 0.0,
            f"{channel}_peak_to_peak": 0.0,
            f"{channel}_kurtosis": 0.0,
            f"{channel}_skewness": 0.0,
            f"{channel}_dominant_freq_hz": 0.0,
            f"{channel}_spectral_centroid_hz": 0.0,
        }

    centered = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    amplitude = np.abs(np.fft.rfft(centered))
    if amplitude.size:
        amplitude[0] = 0.0
    amplitude_sum = float(np.sum(amplitude))
    dominant_freq = float(freqs[int(np.argmax(amplitude))]) if amplitude.size else 0.0
    centroid = float(np.sum(freqs * amplitude) / amplitude_sum) if amplitude_sum > 0 else 0.0

    return {
        f"{channel}_rms": float(np.sqrt(np.mean(np.square(x)))),
        f"{channel}_peak_to_peak": float(np.ptp(x)),
        f"{channel}_kurtosis": _finite_stat(kurtosis(x, fisher=True, bias=False)),
        f"{channel}_skewness": _finite_stat(skew(x, bias=False)),
        f"{channel}_dominant_freq_hz": dominant_freq,
        f"{channel}_spectral_centroid_hz": centroid,
    }


def _finite_stat(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0

