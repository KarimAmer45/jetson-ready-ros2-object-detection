from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from app.schemas import PredictionRequest, PredictionResponse


MODEL_PATH = Path(__file__).resolve().parent.parent / "model_artifacts" / "tool_wear_baseline.json"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class VibrationSummary:
    mean_abs: float
    rms: float
    peak: float

    @classmethod
    def from_values(cls, values: list[float]) -> "VibrationSummary":
        count = len(values)
        mean_abs = sum(abs(value) for value in values) / count
        rms = math.sqrt(sum(value * value for value in values) / count)
        peak = max(abs(value) for value in values)
        return cls(mean_abs=mean_abs, rms=rms, peak=peak)


class ToolWearModel:
    """Small deterministic surrogate for a trained machining model.

    The JSON artifact stores the coefficients and operating limits. In a real
    system this wrapper would load a serialized sklearn, ONNX, or torch model.
    """

    def __init__(self, artifact_path: Path = MODEL_PATH) -> None:
        with artifact_path.open("r", encoding="utf-8") as artifact_file:
            self.artifact = json.load(artifact_file)
        self.version: str = self.artifact["version"]

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        vibration = VibrationSummary.from_values(request.vibration_features)
        features = self._normalized_features(request, vibration)
        wear = self._predict_wear(features)
        risk = self._surface_risk(wear, features)
        confidence = self._confidence(wear, features)

        return PredictionResponse(
            predicted_tool_wear=round(wear, 3),
            surface_quality_risk=risk,
            confidence=round(confidence, 2),
        )

    def _normalized_features(self, request: PredictionRequest, vibration: VibrationSummary) -> dict[str, float]:
        scales = self.artifact["normalization"]
        return {
            "spindle_speed": request.spindle_speed / scales["spindle_speed"],
            "feed_rate": request.feed_rate / scales["feed_rate"],
            "depth_of_cut": request.depth_of_cut / scales["depth_of_cut"],
            "vibration_mean_abs": vibration.mean_abs / scales["vibration_mean_abs"],
            "vibration_rms": vibration.rms / scales["vibration_rms"],
            "vibration_peak": vibration.peak / scales["vibration_peak"],
        }

    def _predict_wear(self, features: dict[str, float]) -> float:
        coefficients = self.artifact["tool_wear"]
        wear = coefficients["intercept"]
        for name, value in features.items():
            wear += coefficients["weights"].get(name, 0.0) * value
        return _clamp(wear)

    def _surface_risk(self, wear: float, features: dict[str, float]) -> str:
        score = (
            wear
            + 0.06 * features["feed_rate"]
            + 0.08 * features["depth_of_cut"]
            + 0.1 * features["vibration_rms"]
            + 0.06 * features["vibration_peak"]
        )
        thresholds = self.artifact["surface_quality_thresholds"]
        if score < thresholds["medium"]:
            return "low"
        if score < thresholds["high"]:
            return "medium"
        return "high"

    def _confidence(self, wear: float, features: dict[str, float]) -> float:
        params = self.artifact["confidence"]
        operating_span = max(
            abs(features["spindle_speed"] - 0.8),
            abs(features["feed_rate"] - 0.7),
            abs(features["depth_of_cut"] - 0.7),
        )
        confidence = (
            params["base"]
            - params["wear_penalty"] * wear
            - params["vibration_penalty"] * features["vibration_peak"]
            - params["operating_span_penalty"] * operating_span
        )
        return _clamp(confidence, lower=0.35, upper=0.96)
