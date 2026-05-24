"""Train and compare pure ML and physics-guided milling wear models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from data_loader import MillingRun, load_milling_runs
from feature_extraction import WEAR_CLASS_ORDER, extract_dataset_features, select_feature_columns, wear_class_from_vb
from physics_features import spindle_frequency_hz, tooth_passing_frequency_hz
from train_baseline import build_model, make_group_split


def run_evaluation(
    *,
    data_path: str | None,
    demo: bool,
    output_dir: str | Path = "results",
    window_seconds: float = 1.0,
    overlap: float = 0.5,
    random_state: int = 42,
) -> dict[str, object]:
    runs = load_milling_runs(data_path, demo=demo or data_path is None)
    features = extract_dataset_features(
        runs,
        include_physics=True,
        window_seconds=window_seconds,
        overlap=overlap,
    ).dropna(subset=["wear_class"])
    features = features.reset_index(drop=True)

    train_idx, test_idx = make_group_split(features, random_state=random_state)
    baseline = _fit_named_model(features, include_physics=False, train_idx=train_idx, test_idx=test_idx, random_state=random_state)
    physics = _fit_named_model(features, include_physics=True, train_idx=train_idx, test_idx=test_idx, random_state=random_state)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_models(out_dir, baseline, physics)
    _plot_confusion_matrices(out_dir / "confusion_matrix.png", baseline, physics)
    _plot_feature_importance(out_dir / "feature_importance.png", physics)
    _plot_fft_examples(out_dir / "fft_examples.png", runs)
    _plot_prediction_vs_true(out_dir / "prediction_vs_true.png", features.loc[test_idx], baseline, physics)

    predictions = pd.DataFrame(
        {
            "case": features.loc[test_idx, "case"].to_numpy(),
            "run": features.loc[test_idx, "run"].to_numpy(),
            "window_id": features.loc[test_idx, "window_id"].to_numpy(),
            "true": baseline["y_test"].to_numpy(),
            "baseline_pred": baseline["predictions"].to_numpy(),
            "physics_guided_pred": physics["predictions"].to_numpy(),
        }
    )
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "baseline": _compact_metrics(baseline),
        "physics_guided": _compact_metrics(physics),
        "n_windows": int(len(features)),
        "n_train_windows": int(len(train_idx)),
        "n_test_windows": int(len(test_idx)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _fit_named_model(
    features: pd.DataFrame,
    *,
    include_physics: bool,
    train_idx: pd.Index,
    test_idx: pd.Index,
    random_state: int,
) -> dict[str, object]:
    feature_columns = select_feature_columns(features, include_physics=include_physics)
    model = build_model(random_state=random_state)
    X_train = features.loc[train_idx, feature_columns]
    y_train = features.loc[train_idx, "wear_class"]
    X_test = features.loc[test_idx, feature_columns]
    y_test = features.loc[test_idx, "wear_class"]
    model.fit(X_train, y_train)
    predictions = pd.Series(model.predict(X_test), name="prediction")
    return {
        "model": model,
        "feature_columns": feature_columns,
        "y_test": y_test.reset_index(drop=True),
        "predictions": predictions.reset_index(drop=True),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }


def _save_models(out_dir: Path, baseline: dict[str, object], physics: dict[str, object]) -> None:
    joblib.dump(
        {
            "model": baseline["model"],
            "feature_columns": baseline["feature_columns"],
            "include_physics": False,
        },
        out_dir / "baseline_model.joblib",
    )
    joblib.dump(
        {
            "model": physics["model"],
            "feature_columns": physics["feature_columns"],
            "include_physics": True,
        },
        out_dir / "physics_guided_model.joblib",
    )


def _compact_metrics(result: dict[str, object]) -> dict[str, object]:
    return {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "n_features": len(result["feature_columns"]),
        "classification_report": result["classification_report"],
    }


def _ordered_labels(*series: pd.Series) -> list[str]:
    present = {label for values in series for label in values.dropna().unique()}
    ordered = [label for label in WEAR_CLASS_ORDER if label in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _plot_confusion_matrices(path: Path, baseline: dict[str, object], physics: dict[str, object]) -> None:
    labels = _ordered_labels(baseline["y_test"], baseline["predictions"], physics["predictions"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, title, result in [
        (axes[0], "Pure ML features", baseline),
        (axes[1], "Physics-guided features", physics),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            result["y_test"],
            result["predictions"],
            labels=labels,
            display_labels=labels,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_feature_importance(path: Path, result: dict[str, object], *, top_n: int = 15) -> None:
    rf = result["model"].named_steps["model"]
    importances = pd.Series(rf.feature_importances_, index=result["feature_columns"]).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    importances.sort_values().plot(kind="barh", ax=ax, color="#2e7d6f")
    ax.set_title("Physics-guided model feature importance")
    ax.set_xlabel("Random Forest importance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_fft_examples(path: Path, runs: list[MillingRun]) -> None:
    examples = _choose_example_runs(runs)
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.8 * len(examples)), constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]

    for ax, milling_run in zip(axes, examples):
        signal_name = "vib_spindle" if "vib_spindle" in milling_run.signals else milling_run.signals.columns[0]
        signal = milling_run.signals[signal_name].to_numpy(dtype=float)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / milling_run.sample_rate_hz)
        amplitude = np.abs(np.fft.rfft(signal - np.mean(signal)))
        max_freq = min(500.0, milling_run.sample_rate_hz / 2.0)
        mask = freqs <= max_freq
        ax.plot(freqs[mask], amplitude[mask], color="#263238", linewidth=1.0)
        spindle_hz = spindle_frequency_hz(milling_run.speed_rpm)
        tpf_hz = tooth_passing_frequency_hz(milling_run.speed_rpm, milling_run.tooth_count)
        if spindle_hz:
            ax.axvline(spindle_hz, color="#d14f3f", linestyle="--", linewidth=1.0, label="spindle")
        if tpf_hz:
            ax.axvline(tpf_hz, color="#3266a8", linestyle="--", linewidth=1.0, label="tooth passing")
        ax.set_title(
            f"case {milling_run.case}, run {milling_run.run}, VB={milling_run.vb:.3f}, class={wear_class_from_vb(milling_run.vb)}"
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _choose_example_runs(runs: list[MillingRun]) -> list[MillingRun]:
    examples: list[MillingRun] = []
    for label in WEAR_CLASS_ORDER:
        matching = [run for run in runs if wear_class_from_vb(run.vb) == label]
        if matching:
            examples.append(matching[len(matching) // 2])
    return examples or runs[:1]


def _plot_prediction_vs_true(
    path: Path,
    test_rows: pd.DataFrame,
    baseline: dict[str, object],
    physics: dict[str, object],
) -> None:
    labels = _ordered_labels(baseline["y_test"], baseline["predictions"], physics["predictions"])
    label_to_code = {label: idx for idx, label in enumerate(labels)}
    order = (
        test_rows.reset_index(drop=True)
        .assign(_row=np.arange(len(test_rows)))
        .sort_values(["case", "run", "window_id"])["_row"]
        .to_numpy()
    )
    x = np.arange(len(order))
    true_codes = [label_to_code[label] for label in baseline["y_test"].iloc[order]]
    baseline_codes = [label_to_code[label] for label in baseline["predictions"].iloc[order]]
    physics_codes = [label_to_code[label] for label in physics["predictions"].iloc[order]]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(x, true_codes, "o-", label="true", linewidth=1.2, markersize=3)
    ax.plot(x, baseline_codes, "s--", label="baseline", linewidth=1.0, markersize=3)
    ax.plot(x, physics_codes, "^--", label="physics-guided", linewidth=1.0, markersize=3)
    ax.set_yticks(list(label_to_code.values()), labels=list(label_to_code.keys()))
    ax.set_xlabel("Test windows sorted by case/run")
    ax.set_ylabel("Wear class")
    ax.set_title("Prediction vs true class")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None, help="Path to NASA mill.zip, .mat, .csv, or extracted folder.")
    parser.add_argument("--demo", action="store_true", help="Use built-in synthetic milling data.")
    parser.add_argument("--output-dir", default="results", help="Where to write plots, models, and metrics.")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_evaluation(
        data_path=args.data_path,
        demo=args.demo,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
    )
    print(json.dumps({name: values for name, values in metrics.items() if name in {"baseline", "physics_guided"}}, indent=2))


if __name__ == "__main__":
    main()

