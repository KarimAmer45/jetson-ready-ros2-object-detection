"""Data loading utilities for milling vibration and tool wear experiments."""

from __future__ import annotations

import ast
import math
import zipfile
from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


DEFAULT_SAMPLE_RATE_HZ = 250.0
DEFAULT_TOOTH_COUNT = 4

SENSOR_COLUMNS = (
    "vib_table",
    "vib_spindle",
    "AE_table",
    "AE_spindle",
    "smcAC",
    "smcDC",
)

FIELD_ALIASES = {
    "case": ("case", "case_id", "experiment", "test_case"),
    "run": ("run", "run_id", "trial", "pass"),
    "VB": ("VB", "vb", "wear", "tool_wear", "flank_wear", "wear_vb"),
    "time": ("time", "timestamp", "t"),
    "DOC": ("DOC", "doc", "depth_of_cut", "ap"),
    "feed": ("feed", "feed_rate", "fz"),
    "material": ("material", "workpiece_material"),
    "speed_rpm": ("speed_rpm", "rpm", "spindle_speed", "spindle_speed_rpm", "speed"),
    "tooth_count": ("tooth_count", "teeth", "num_teeth", "n_teeth"),
}

SENSOR_ALIASES = {
    "vib_table": ("vib_table", "table_vibration", "vibration_table"),
    "vib_spindle": ("vib_spindle", "spindle_vibration", "vibration_spindle", "vibration"),
    "AE_table": ("AE_table", "ae_table", "acoustic_table"),
    "AE_spindle": ("AE_spindle", "ae_spindle", "acoustic_spindle", "acoustic"),
    "smcAC": ("smcAC", "smc_ac", "ac_current", "spindle_current_ac"),
    "smcDC": ("smcDC", "smc_dc", "dc_current", "spindle_current_dc"),
}


@dataclass(frozen=True)
class MillingRun:
    """One machining run with aligned sensor channels and wear metadata."""

    case: int
    run: int
    vb: float
    sample_rate_hz: float
    signals: pd.DataFrame
    speed_rpm: float | None = None
    feed: float | None = None
    depth_of_cut: float | None = None
    tooth_count: int = DEFAULT_TOOTH_COUNT
    material: str | None = None
    source: str = "unknown"


def load_milling_runs(
    data_path: str | Path | None = None,
    *,
    demo: bool = False,
    sample_rate_hz: float | None = None,
) -> list[MillingRun]:
    """Load milling runs from NASA-style MAT/ZIP/CSV data or generate a demo set."""

    if demo or data_path is None:
        return generate_synthetic_milling_runs(sample_rate_hz=sample_rate_hz or 2000.0)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {path}")

    if path.is_dir():
        mat_files = sorted(path.rglob("*.mat"))
        csv_files = sorted(path.rglob("*.csv"))
        if mat_files:
            return _load_mat_file(mat_files[0], sample_rate_hz=sample_rate_hz)
        if csv_files:
            return _load_csv_file(csv_files[0], sample_rate_hz=sample_rate_hz)
        raise ValueError(f"No .mat or .csv files found under {path}")

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _load_zip_file(path, sample_rate_hz=sample_rate_hz)
    if suffix == ".mat":
        return _load_mat_file(path, sample_rate_hz=sample_rate_hz)
    if suffix == ".csv":
        return _load_csv_file(path, sample_rate_hz=sample_rate_hz)

    raise ValueError(f"Unsupported data file type: {path.suffix}")


def generate_synthetic_milling_runs(
    *,
    n_cases: int = 12,
    runs_per_case: int = 6,
    sample_rate_hz: float = 2000.0,
    duration_s: float = 2.5,
    random_state: int = 7,
) -> list[MillingRun]:
    """Generate a small milling-like dataset for demos and CI smoke tests."""

    rng = np.random.default_rng(random_state)
    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    tooth_count = DEFAULT_TOOTH_COUNT
    speeds = np.array([900.0, 1200.0, 1500.0, 1800.0])
    feeds = np.array([0.04, 0.08, 0.12])
    depths = np.array([0.5, 1.0])
    runs: list[MillingRun] = []

    for case_idx in range(n_cases):
        speed_rpm = float(speeds[case_idx % len(speeds)])
        feed = float(feeds[(case_idx // len(speeds)) % len(feeds)])
        depth = float(depths[(case_idx // (len(speeds) * len(feeds))) % len(depths)])
        spindle_hz = speed_rpm / 60.0
        tooth_passing_hz = spindle_hz * tooth_count
        chatter_hz = 310.0 + 15.0 * (case_idx % 5)

        for run_idx in range(runs_per_case):
            progress = run_idx / max(1, runs_per_case - 1)
            vb = float(0.045 + 0.49 * progress + 0.015 * rng.normal())
            vb = float(np.clip(vb, 0.03, 0.65))
            wear_gain = 1.0 + 2.5 * vb
            noise_scale = 0.05 + 0.16 * vb
            chatter_gain = max(0.0, vb - 0.28) * 2.2
            phase = rng.uniform(0, 2 * np.pi)

            spindle = 0.11 * np.sin(2 * np.pi * spindle_hz * t + phase)
            tooth = (0.18 * wear_gain) * np.sin(2 * np.pi * tooth_passing_hz * t)
            harmonic = (0.06 + 0.14 * vb) * np.sin(2 * np.pi * 2 * tooth_passing_hz * t)
            chatter = chatter_gain * np.sin(2 * np.pi * chatter_hz * t + phase / 2.0)
            vibration = spindle + tooth + harmonic + chatter + rng.normal(0.0, noise_scale, len(t))

            table_vibration = 0.65 * vibration + rng.normal(0.0, noise_scale * 0.7, len(t))
            ae_spindle = (
                0.18 * np.sin(2 * np.pi * 420.0 * t)
                + 0.55 * chatter
                + rng.normal(0.0, 0.08 + 0.12 * vb, len(t))
            )
            ae_table = 0.75 * ae_spindle + rng.normal(0.0, 0.05, len(t))
            smc_ac = 0.45 + 0.04 * np.sin(2 * np.pi * spindle_hz * t) + 0.08 * vb + rng.normal(0, 0.02, len(t))
            smc_dc = 0.30 + 0.03 * np.sin(2 * np.pi * 0.5 * spindle_hz * t) + 0.05 * vb + rng.normal(0, 0.015, len(t))

            signals = pd.DataFrame(
                {
                    "vib_table": table_vibration,
                    "vib_spindle": vibration,
                    "AE_table": ae_table,
                    "AE_spindle": ae_spindle,
                    "smcAC": smc_ac,
                    "smcDC": smc_dc,
                }
            )
            runs.append(
                MillingRun(
                    case=case_idx + 1,
                    run=run_idx + 1,
                    vb=vb,
                    sample_rate_hz=float(sample_rate_hz),
                    signals=signals,
                    speed_rpm=speed_rpm,
                    feed=feed,
                    depth_of_cut=depth,
                    tooth_count=tooth_count,
                    material="synthetic_steel",
                    source="synthetic",
                )
            )

    return runs


def _load_zip_file(path: Path, *, sample_rate_hz: float | None) -> list[MillingRun]:
    with zipfile.ZipFile(path) as archive:
        mat_members = [name for name in archive.namelist() if name.lower().endswith(".mat")]
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if mat_members:
            with archive.open(mat_members[0]) as handle:
                data = BytesIO(handle.read())
            return _load_mat_file(data, sample_rate_hz=sample_rate_hz, source=f"{path}!{mat_members[0]}")
        if csv_members:
            with archive.open(csv_members[0]) as handle:
                text = TextIOWrapper(handle, encoding="utf-8")
                df = pd.read_csv(text)
            return _runs_from_csv_frame(df, sample_rate_hz=sample_rate_hz, source=f"{path}!{csv_members[0]}")
    raise ValueError(f"No .mat or .csv files found inside {path}")


def _load_mat_file(
    path_or_buffer: str | Path | BytesIO,
    *,
    sample_rate_hz: float | None,
    source: str | None = None,
) -> list[MillingRun]:
    raw = loadmat(path_or_buffer, squeeze_me=True, struct_as_record=False)
    records = list(_iter_mat_records(raw))
    runs = [
        run
        for record in records
        if (run := _run_from_record(record, sample_rate_hz=sample_rate_hz, source=source or str(path_or_buffer)))
        is not None
    ]
    if not runs:
        raise ValueError("No milling runs with sensor arrays were found in the MAT file.")
    return runs


def _load_csv_file(path: Path, *, sample_rate_hz: float | None) -> list[MillingRun]:
    df = pd.read_csv(path)
    return _runs_from_csv_frame(df, sample_rate_hz=sample_rate_hz, source=str(path))


def _runs_from_csv_frame(df: pd.DataFrame, *, sample_rate_hz: float | None, source: str) -> list[MillingRun]:
    normalized = {_normalize_name(col): col for col in df.columns}
    case_col = _find_column(normalized, FIELD_ALIASES["case"])
    run_col = _find_column(normalized, FIELD_ALIASES["run"])

    if case_col and run_col and df.groupby([case_col, run_col]).size().max() > 1:
        runs = []
        for (case, run), group in df.groupby([case_col, run_col], sort=True):
            record = {col: group[col].to_numpy() for col in group.columns}
            record["case"] = case
            record["run"] = run
            runs.append(_run_from_record(record, sample_rate_hz=sample_rate_hz, source=source))
        return [run for run in runs if run is not None]

    runs = []
    for _, row in df.iterrows():
        record = {col: _parse_csv_cell(row[col]) for col in df.columns}
        runs.append(_run_from_record(record, sample_rate_hz=sample_rate_hz, source=source))
    return [run for run in runs if run is not None]


def _iter_mat_records(raw: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for name, value in raw.items():
        if name.startswith("__"):
            continue
        yield from _iter_record_like(value)


def _iter_record_like(value: Any) -> Iterable[dict[str, Any]]:
    if hasattr(value, "_fieldnames"):
        yield _mat_struct_to_dict(value)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for item in value.ravel():
                yield {field: _mat_to_python(item[field]) for field in value.dtype.names}
            return
        for item in value.ravel():
            if hasattr(item, "_fieldnames") or isinstance(item, np.void):
                yield from _iter_record_like(item)


def _mat_struct_to_dict(obj: Any) -> dict[str, Any]:
    return {field: _mat_to_python(getattr(obj, field)) for field in obj._fieldnames}


def _mat_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return _mat_struct_to_dict(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [_mat_to_python(item) for item in value.ravel()]
        return np.asarray(value).squeeze()
    return value


def _run_from_record(record: dict[str, Any], *, sample_rate_hz: float | None, source: str) -> MillingRun | None:
    normalized = {_normalize_name(key): key for key in record.keys()}

    signals: dict[str, np.ndarray] = {}
    for sensor, aliases in SENSOR_ALIASES.items():
        key = _find_column(normalized, aliases)
        if key is None:
            continue
        values = _as_numeric_array(record[key])
        if values.size > 1:
            signals[sensor] = values

    if not signals:
        return None

    min_length = min(len(values) for values in signals.values())
    if min_length < 8:
        return None

    signal_frame = pd.DataFrame({name: values[:min_length] for name, values in signals.items()})
    case = int(_scalar_field(record, normalized, FIELD_ALIASES["case"], default=0) or 0)
    run = int(_scalar_field(record, normalized, FIELD_ALIASES["run"], default=0) or 0)
    vb = _as_float(_scalar_field(record, normalized, FIELD_ALIASES["VB"], default=np.nan))
    speed_rpm = _as_float_or_none(_scalar_field(record, normalized, FIELD_ALIASES["speed_rpm"], default=None))
    feed = _as_float_or_none(_scalar_field(record, normalized, FIELD_ALIASES["feed"], default=None))
    depth = _as_float_or_none(_scalar_field(record, normalized, FIELD_ALIASES["DOC"], default=None))
    material_value = _scalar_field(record, normalized, FIELD_ALIASES["material"], default=None)
    tooth_count_value = _scalar_field(record, normalized, FIELD_ALIASES["tooth_count"], default=DEFAULT_TOOTH_COUNT)
    tooth_count = int(_as_float(tooth_count_value, default=DEFAULT_TOOTH_COUNT))
    inferred_sample_rate = sample_rate_hz or _infer_sample_rate(record, normalized, min_length)

    return MillingRun(
        case=case,
        run=run,
        vb=vb,
        sample_rate_hz=float(inferred_sample_rate),
        signals=signal_frame,
        speed_rpm=speed_rpm,
        feed=feed,
        depth_of_cut=depth,
        tooth_count=tooth_count,
        material=None if material_value is None else str(material_value),
        source=source,
    )


def _infer_sample_rate(record: dict[str, Any], normalized: dict[str, str], length: int) -> float:
    time_key = _find_column(normalized, FIELD_ALIASES["time"])
    if time_key is not None:
        values = _as_numeric_array(record[time_key])
        if values.size == length and values.size > 2:
            diffs = np.diff(values.astype(float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                return float(1.0 / np.median(diffs))
        duration = _as_float(record[time_key], default=np.nan)
        if math.isfinite(duration) and duration > 0:
            return float(length / duration)
    return DEFAULT_SAMPLE_RATE_HZ


def _find_column(normalized_columns: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def _scalar_field(
    record: dict[str, Any],
    normalized: dict[str, str],
    aliases: Iterable[str],
    *,
    default: Any,
) -> Any:
    key = _find_column(normalized, aliases)
    if key is None:
        return default
    value = record[key]
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        return flat[-1] if flat.size else default
    if isinstance(value, list):
        return value[-1] if value else default
    return value


def _as_numeric_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        parsed = _parse_csv_cell(value)
        if parsed is value:
            return np.array([], dtype=float)
        value = parsed
    arr = np.asarray(value)
    if arr.dtype == object:
        flattened = []
        for item in arr.ravel():
            try:
                flattened.append(float(item))
            except (TypeError, ValueError):
                continue
        return np.asarray(flattened, dtype=float)
    return np.asarray(arr, dtype=float).ravel()


def _parse_csv_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return np.nan
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return np.asarray(parsed, dtype=float)
        except (SyntaxError, ValueError, TypeError):
            return np.fromstring(text.strip("[]"), sep=",")
    return value


def _as_float(value: Any, *, default: float = np.nan) -> float:
    try:
        if isinstance(value, np.ndarray):
            value = value.ravel()[-1] if value.size else default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float_or_none(value: Any) -> float | None:
    result = _as_float(value, default=np.nan)
    return result if math.isfinite(result) else None


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

