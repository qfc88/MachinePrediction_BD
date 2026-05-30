# Predictive Maintenance and Failure Detection using Dask

A scalable equipment failure prediction system built with Dask, FastAPI, and Streamlit.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI    │────▶│  Dask Cluster│
│  Dashboard   │     │   /predict   │     │  (4 workers) │
│  :8501       │     │   /monitor   │     │  :8787       │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │   src/       │
                     │  data/       │  Core business logic
                     │  models/     │  (preprocessing,
                     │  monitoring/ │   training, alerting)
                     │  optimization│
                     └──────────────┘
```

## Quick Start

### 1. Environment Setup
```bash
micromamba create -n bigdata -f environment.yml -y
micromamba activate bigdata
pip install fastapi uvicorn[standard]
```

### 2. Download Dataset
```bash
kaggle datasets download -d shashanknecrothapa/machine-failure-predictions -p data/raw/ --unzip
```

### 3. Train Models
```bash
python main.py
```
This runs the full pipeline: load → preprocess → feature engineering → train → evaluate → save.

### 4. Start API Server
```bash
make serve
# or: uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 5. Start Dashboard
```bash
make dashboard
# or: streamlit run dashboard/app.py --server.port 8501
```

### 6. Run Streaming Simulation
```bash
make stream
# or: python scripts/simulate_stream.py
```

## Docker Deployment
```bash
cd deployment
cp .env.example .env
docker compose up --build -d
```

Services:
- API: http://localhost:8000 (docs at /docs)
- Dashboard: http://localhost:8501
- Dask Dashboard: http://localhost:8787

## Project Structure

```
bigdata/
├── src/                          # Core business logic (BACKBONE)
│   ├── config.py                 # Configuration loader
│   ├── data/
│   │   ├── loader.py             # Dask data loading
│   │   ├── preprocessing.py      # Cleaning, encoding, scaling
│   │   └── feature_engineering.py# Feature extraction
│   ├── models/
│   │   ├── trainer.py            # Model training pipeline
│   │   ├── evaluator.py          # Metrics & visualization
│   │   └── registry.py           # Model save/load
│   ├── monitoring/
│   │   ├── stream.py             # Streaming simulation
│   │   └── alerting.py           # Alert management
│   └── optimization/             # Dask performance tuning
├── api/                          # FastAPI server
│   ├── server.py                 # App entrypoint
│   ├── schemas.py                # Request/response models
│   └── routes/                   # Endpoint handlers
├── dashboard/
│   └── app.py                    # Streamlit UI
├── scripts/                      # CLI tools
├── config/config.yaml            # All settings
├── deployment/                   # Docker configs
├── main.py                       # E2E pipeline orchestrator
├── Makefile                      # Convenience commands
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/monitor/status` | Stream status |
| POST | `/monitor/start` | Start streaming |
| POST | `/monitor/stop` | Stop streaming |
| WS | `/ws/stream` | Live WebSocket feed |

## Configuration

All settings in `config/config.yaml`. Override with environment variables:
- `DASK_N_WORKERS` — number of Dask workers
- `DASK_MEMORY_LIMIT` — per-worker memory limit
- `API_PORT` — FastAPI port
- `API_URL` — API URL for dashboard
