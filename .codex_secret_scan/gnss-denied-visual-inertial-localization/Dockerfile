FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY tests ./tests
COPY results/example/metrics.json ./results/example/metrics.json
COPY results/example/*.png ./results/example/

RUN python -m pip install --no-cache-dir -e .

CMD ["python", "-m", "gnss_denied_vio.simulate", "--seed", "7", "--output", "results/example"]

