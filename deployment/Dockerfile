FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ src/
COPY api/ api/
COPY config/ config/
COPY main.py .
COPY scripts/ scripts/

# Data & models (mounted as volumes in production)
RUN mkdir -p data/raw data/processed models reports/figures logs

EXPOSE 8000 8787

# Default: run the API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
