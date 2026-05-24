import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    spindle_speed: float = Field(..., gt=0, description="Spindle speed in RPM")
    feed_rate: float = Field(..., gt=0, description="Feed rate in mm/min")
    depth_of_cut: float = Field(..., gt=0, description="Axial depth of cut in mm")
    vibration_features: list[float] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Windowed vibration signal or derived vibration features",
        examples=[[0.12, 0.18, 0.2, 0.16, 0.22]],
    )

    @field_validator("vibration_features")
    @classmethod
    def reject_non_finite_values(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("vibration_features must contain finite numbers")
        return values


class PredictionResponse(BaseModel):
    predicted_tool_wear: float = Field(..., ge=0, le=1, description="Estimated normalized tool wear")
    surface_quality_risk: Literal["low", "medium", "high"]
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
