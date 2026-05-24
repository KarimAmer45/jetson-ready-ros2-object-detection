from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import ToolWearModel
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="Machining Digital Twin Microservice",
    description="Predicts tool wear and surface quality risk from machining parameters and vibration features.",
    version="0.1.0",
)
model = ToolWearModel()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_alias() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=model.version)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    return model.predict(payload)
