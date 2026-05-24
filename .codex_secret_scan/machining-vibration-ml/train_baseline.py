"""Train a Random Forest baseline on pure signal features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_milling_runs
from feature_extraction import extract_dataset_features, select_feature_columns


def build_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def make_group_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Split by machining run to avoid leaking windows from the same run."""

    df = df.reset_index(drop=True)
    groups = df["case"].astype(str) + "_" + df["run"].astype(str)
    if groups.nunique() >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, df["wear_class"], groups))
        return pd.Index(train_idx), pd.Index(test_idx)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["wear_class"] if df["wear_class"].nunique() > 1 else None,
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def train_from_feature_table(
    features: pd.DataFrame,
    *,
    include_physics: bool,
    random_state: int = 42,
    train_idx: pd.Index | None = None,
    test_idx: pd.Index | None = None,
) -> dict[str, object]:
    data = features.dropna(subset=["wear_class"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No labeled windows available. Check that VB/tool-wear values are present.")

    feature_columns = select_feature_columns(data, include_physics=include_physics)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found.")

    if train_idx is None or test_idx is None:
        train_idx, test_idx = make_group_split(data, random_state=random_state)

    model = build_model(random_state=random_state)
    X_train = data.loc[train_idx, feature_columns]
    y_train = data.loc[train_idx, "wear_class"]
    X_test = data.loc[test_idx, feature_columns]
    y_test = data.loc[test_idx, "wear_class"]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "feature_columns": feature_columns,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_test": y_test.reset_index(drop=True),
        "predictions": pd.Series(predictions, name="prediction"),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "feature_table": data,
    }


def run_training(
    *,
    data_path: str | None,
    demo: bool,
    include_physics: bool,
    output_dir: str | Path = "results",
    model_name: str = "baseline",
    window_seconds: float = 1.0,
    overlap: float = 0.5,
    random_state: int = 42,
) -> dict[str, object]:
    runs = load_milling_runs(data_path, demo=demo or data_path is None)
    features = extract_dataset_features(
        runs,
        include_physics=include_physics,
        window_seconds=window_seconds,
        overlap=overlap,
    )
    result = train_from_feature_table(features, include_physics=include_physics, random_state=random_state)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result["model"],
            "feature_columns": result["feature_columns"],
            "include_physics": include_physics,
        },
        out_dir / f"{model_name}_model.joblib",
    )
    metrics = {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "classification_report": result["classification_report"],
        "n_windows": int(len(result["feature_table"])),
        "n_features": int(len(result["feature_columns"])),
    }
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=None, help="Path to NASA mill.zip, .mat, .csv, or extracted folder.")
    parser.add_argument("--demo", action="store_true", help="Use built-in synthetic milling data.")
    parser.add_argument("--output-dir", default="results", help="Where to write model and metrics artifacts.")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        data_path=args.data_path,
        demo=args.demo,
        include_physics=False,
        output_dir=args.output_dir,
        model_name="baseline",
        window_seconds=args.window_seconds,
        overlap=args.overlap,
    )
    print(f"Baseline accuracy: {result['accuracy']:.3f}")
    print(f"Baseline macro F1: {result['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
