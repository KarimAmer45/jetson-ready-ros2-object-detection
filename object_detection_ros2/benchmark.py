"""Standalone FPS/latency benchmark for detector backends."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

import numpy as np

from object_detection_ros2.detectors import create_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="torchvision", choices=["torchvision", "yolo"])
    parser.add_argument("--model", default="", help="Torchvision weights name or YOLO .pt path")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--image", default="", help="Optional image path for repeat inference")
    parser.add_argument("--width", type=int, default=640, help="Synthetic frame width")
    parser.add_argument("--height", type=int, default=480, help="Synthetic frame height")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--json", dest="json_path", default="", help="Optional output JSON path")
    return parser.parse_args()


def load_image(path: str, width: int, height: int) -> np.ndarray:
    if path:
        import cv2

        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    # A synthetic frame keeps the benchmark runnable before a camera is connected.
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = 80
    return frame


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return sorted(values)[index]


def main() -> None:
    args = parse_args()
    frame = load_image(args.image, args.width, args.height)
    detector = create_detector(
        backend=args.backend,
        model=args.model,
        device=args.device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    for _ in range(args.warmup):
        detector.predict(frame)

    latencies_ms = []
    detections_seen = 0
    started = time.perf_counter()
    for _ in range(args.iterations):
        iteration_start = time.perf_counter()
        detections = detector.predict(frame)
        latencies_ms.append((time.perf_counter() - iteration_start) * 1000.0)
        detections_seen = len(detections)
    elapsed = time.perf_counter() - started

    summary = {
        "backend": args.backend,
        "model": args.model or "default",
        "device": args.device,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "frame_shape": list(frame.shape),
        "fps": args.iterations / elapsed if elapsed else 0.0,
        "latency_ms_mean": statistics.fmean(latencies_ms),
        "latency_ms_median": statistics.median(latencies_ms),
        "latency_ms_p95": percentile(latencies_ms, 95),
        "last_detection_count": detections_seen,
    }

    print(
        "backend={backend} model={model} device={device} "
        "fps={fps:.2f} latency_ms_mean={latency_ms_mean:.2f} "
        "latency_ms_p95={latency_ms_p95:.2f} detections={last_detection_count}".format(
            **summary
        )
    )

    if args.json_path:
        output_path = Path(args.json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
