"""Train a Random Forest model with signal and physics-guided features."""

from __future__ import annotations

import argparse

from train_baseline import run_training


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
        include_physics=True,
        output_dir=args.output_dir,
        model_name="physics_guided",
        window_seconds=args.window_seconds,
        overlap=args.overlap,
    )
    print(f"Physics-guided accuracy: {result['accuracy']:.3f}")
    print(f"Physics-guided macro F1: {result['macro_f1']:.3f}")


if __name__ == "__main__":
    main()

