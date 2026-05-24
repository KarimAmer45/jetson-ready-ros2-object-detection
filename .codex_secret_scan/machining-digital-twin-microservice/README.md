# Machining Digital Twin Microservice

A compact FastAPI service for estimating tool wear and surface quality risk from machining parameters and vibration features. It is intentionally small: the service wraps a deterministic JSON-backed baseline model so the API, dashboard, tests, and Docker path are easy to inspect.

## Run locally

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open:

- Dashboard: <http://127.0.0.1:8000/dashboard>
- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## API

`POST /predict`

```json
{
  "spindle_speed": 8000,
  "feed_rate": 500,
  "depth_of_cut": 1.5,
  "vibration_features": [0.16, 0.19, 0.24, 0.21, 0.18, 0.27]
}
```

Example response:

```json
{
  "predicted_tool_wear": 0.18,
  "surface_quality_risk": "medium",
  "confidence": 0.82
}
```

## Docker

```bash
docker build -t machining-digital-twin .
docker run --rm -p 8000:8000 machining-digital-twin
```

Or:

```bash
docker compose up --build
```

## Screenshots

Dashboard:

![Dashboard screenshot](docs/screenshots/dashboard.png)

Swagger:

![Swagger screenshot](docs/screenshots/swagger.png)

## Proof It Runs

Verified locally on April 30, 2026 with Uvicorn serving the FastAPI app at `http://127.0.0.1:8000`.

`GET /health`

```json
{
  "status": "ok",
  "model_version": "baseline-linear-v1"
}
```

`POST /predict`

```json
{
  "predicted_tool_wear": 0.18,
  "surface_quality_risk": "medium",
  "confidence": 0.82
}
```

## API workflow

- Typed FastAPI request and response contracts with validation.
- A replaceable model wrapper around a versioned artifact.
- Basic service health reporting for container orchestration.
- A small dashboard that exercises the same `/predict` route as API clients.
- Docker packaging for repeatable deployment.

## Hardening plan

- The current model is a deterministic surrogate, not a production-trained model.
- Confidence is a heuristic based on operating range and vibration intensity.
- Add real training data, model evaluation, and model registry metadata.
- Add drift monitoring, auth, richer telemetry, and batch prediction support.
