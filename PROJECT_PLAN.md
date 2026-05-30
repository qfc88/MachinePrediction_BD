# Predictive Maintenance and Failure Detection using Dask

## 📋 Tổng quan dự án
- **Môn học**: Big Data Analysis
- **Chủ đề**: Predictive Maintenance & Failure Detection
- **Công nghệ chính**: Python, Dask, Dask-ML, Scikit-learn, FastAPI, Streamlit, Nginx, Tailscale
- **Dataset**: [Machine Failure Predictions — Kaggle](https://kaggle.com/datasets/shashanknecrothapa/machine-failure-predictions/data)
  - 10.000 mẫu × 14 cột, không có missing values
  - Target: `Machine failure` (binary, imbalanced ~3.39%)
  - Failure subtypes: TWF, HDF, PWF, OSF, RNF

---

## 🖥️ Thông số hạ tầng

| Resource | Giá trị |
|----------|---------|
| CPU | 16 cores |
| RAM | 16 GB |
| Disk | ~33 GB |
| OS | Ubuntu Linux |
| Python Env | Micromamba (bigdata env) |
| Public Domain | ism.repme.tech (HTTPS) |
| VPN | Tailscale (100.121.93.93) |

---

## 🏗️ Kiến trúc hệ thống

```
Internet
    │
    ▼
[VPS — Nginx Reverse Proxy]  ←── SSL/TLS (Let's Encrypt)
    │  ism.repme.tech
    │
    │  Tailscale tunnel (100.121.93.93)
    ├──/api/*  ──────────────────▶ FastAPI :8000
    │                               ├─ /predict      (single/batch inference)
    │                               ├─ /monitor/*    (stream status, start/stop)
    │                               └─ /health       (healthcheck)
    │
    ├──/dask/* ──────────────────▶ Dask Dashboard :8787
    │                               (task graph, cluster metrics)
    │
    └──/*     ───────────────────▶ Streamlit :8501
                                    (monitoring UI, live predictions)


src/ (Business Logic)
├── data/        loader → preprocessing → feature_engineering
├── models/      trainer → evaluator → registry
├── monitoring/  stream → alerting
└── optimization/ partition tuning, benchmarking
```

---

## 👥 Phân công thành viên

### Thành viên 1 — [Tên] — Project Scaffolding, Hạ tầng & Real-time Monitoring
**Mức đóng góp: ~25%**

#### 1A. Dựng khung toàn bộ project
| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Thiết kế toàn bộ cấu trúc thư mục dự án (`src/`, `api/`, `dashboard/`, `scripts/`, `deployment/`, ...) | Project layout |
| Viết `main.py` — E2E pipeline orchestrator (load → preprocess → feature eng → train → evaluate → save) | `main.py` |
| Cấu hình tổng thể project (paths, Dask, data, training, monitoring, dashboard) | `config/config.yaml` |
| Khai báo môi trường Python (micromamba, packages) | `environment.yml`, `requirements.txt` |
| Makefile shortcuts (`make serve`, `make dashboard`, `make stream`, `make train`) | `Makefile` |
| Viết README hướng dẫn setup, chạy, deploy | `README.md` |
| Khởi tạo toàn bộ `__init__.py`, module structure cho `src/`, `api/` | `src/__init__.py`, `api/__init__.py`, ... |

#### 1B. DevOps / Hạ tầng
| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Cấu hình Nginx reverse proxy trên VPS | `deployment/ism.repme.tech.conf` |
| SSL/TLS Let's Encrypt, HTTP→HTTPS redirect | `ism.repme.tech.conf` (ssl block) |
| Rate limiting, proxy headers, WebSocket support | `ism.repme.tech.conf` (limit_req_zone) |
| Kết nối Tailscale VPN (100.121.93.93), expose nội bộ | Tailscale CLI |
| Viết systemd service chạy tự động khi boot | `/etc/systemd/system/bigdata-api.service` |
| | `/etc/systemd/system/bigdata-dashboard.service` |

#### 1C. Real-time Prediction Simulation (Streamlit)
| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Dashboard Live Monitor tab (start/stop stream, biểu đồ real-time) | `dashboard/app.py` (page "Live Monitor") |
| CLI script chạy streaming standalone | `scripts/simulate_stream.py` |
| StreamSimulator: batch ingestion, predict, log | `src/monitoring/stream.py` |
| Threshold-based alerting, severity phân loại (low/medium/high/critical) | `src/monitoring/alerting.py` |
| Cấu hình monitoring (threshold, cooldown, batch_size, interval) | `config/config.yaml` (monitoring section) |

#### 1D. Tham gia ML (Inference Pipeline)
| Nhiệm vụ | Chi tiết |
|----------|---------|
| Inference pipeline trong StreamSimulator | `predict_batch()` trong `stream.py`: align features, scale, predict_proba |
| Risk level classification từ failure probability | `_get_risk_level()`, `_severity()` — xây dựng ngưỡng phân loại |
| Tích hợp model đã train vào real-time pipeline | Load artifact từ `registry`, apply scaler cho streaming data |

#### 1E. FastAPI Backend (Monitoring & Health)
| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Khởi tạo StreamSimulator trong app_state khi startup | `api/server.py` — `_init_stream_simulator()` |
| Endpoint `GET /monitor/status` — trả về stream stats + recent alerts | `api/routes/monitor.py` |
| Endpoint `POST /monitor/start` / `POST /monitor/stop` — điều khiển streaming | `api/routes/monitor.py` |
| WebSocket `/monitor/ws` — live feed real-time tới dashboard | `api/routes/monitor.py` |
| Endpoint `GET /health` — healthcheck (model loaded, Dask status) | `api/routes/health.py` |

---

### Thành viên 2 — [Tên] — Data Engineering & Preprocessing
**Mức đóng góp: ~20%**

| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Khảo sát dataset: dtypes, shape, null counts, class imbalance | EDA, `data/raw/machine failure.csv` |
| Load CSV bằng `dask.dataframe.read_csv()`, repartition 4 partitions | `src/data/loader.py` — `load_raw_data()` |
| Khởi tạo Dask LocalCluster (4 workers, 2 threads, 3GB/worker) | `src/data/loader.py` — `get_dask_client()` |
| Xử lý missing values (median/mode imputation trên Dask) | `src/data/preprocessing.py` — `handle_missing_values()` |
| Phát hiện và loại outliers (IQR × 3.0 factor) | `src/data/preprocessing.py` — `remove_outliers()` |
| One-hot encoding cột `Type` (L/M/H) với Dask get_dummies | `src/data/preprocessing.py` — `encode_categoricals()` |
| StandardScaler cho sensor columns (fit khi train, transform khi infer) | `src/data/preprocessing.py` — `scale_features()` |
| Lưu processed data sang Parquet partitioned (4 parts) | `src/data/loader.py` — `save_processed_data()` |
| Cấu hình data pipeline | `config/config.yaml` (data section) |

---

### Thành viên 3 — [Tên] — Feature Engineering
**Mức đóng góp: ~20%**

| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Rolling statistics: mean + std, windows = [3, 5, 10] cho 5 sensor columns (→ 30 features) | `src/data/feature_engineering.py` — `add_rolling_features()` |
| Rate-of-change (first derivative / diff) cho sensor columns (→ 5 features) | `src/data/feature_engineering.py` — `add_rate_of_change()` |
| Interaction features dựa trên domain knowledge thiết bị: | `src/data/feature_engineering.py` — `add_interaction_features()` |
| — `temp_diff` = Process temp − Air temp (nhiệt độ chênh lệch) | |
| — `power` = RPM × Torque (công suất cơ học) | |
| — `torque_speed_ratio` = Torque / RPM (chỉ số tải) | |
| — `wear_torque` = Tool wear × Torque | |
| — `overstrain` = Tool wear × power | |
| Z-score anomaly indicators (z > 2 → flag = 1) cho sensor columns (→ 5 features) | `src/data/feature_engineering.py` — `add_anomaly_indicators()` |
| Usage cycle: `wear_stage` (bin tool wear: new/normal/worn/critical), `cumulative_usage` | `src/data/feature_engineering.py` — `add_usage_cycle_features()` |
| Tạo `stream_test_data.parquet` (1983 rows, 61 features, unseen during training) | `main.py` pipeline, `run_feature_engineering()` |
| Cấu hình feature engineering | `config/config.yaml` (feature_engineering section) |

---

### Thành viên 4 — [Tên] — Model Training & Evaluation
**Mức đóng góp: ~20%**

| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| Stratified train/test split (test_size=0.2, random_state=42) | `src/models/trainer.py` — `prepare_data()` |
| Xử lý class imbalance bằng SMOTE (failure rate ~3.39%) | `src/models/trainer.py` — SMOTE trong `prepare_data()` |
| Train 6 mô hình song song: | `src/models/trainer.py` — `train_all_models()` |
| — Logistic Regression (baseline) | |
| — Decision Tree | |
| — Random Forest (200 estimators, class_weight=balanced) | |
| — Gradient Boosting (200 estimators) ← **best F1=0.775** | |
| — XGBoost (scale_pos_weight=28) | |
| — LightGBM (is_unbalance=True) | |
| Voting Ensemble (soft voting) | `src/models/trainer.py` |
| Đánh giá tất cả models: Accuracy, Precision, Recall, F1, ROC-AUC | `src/models/evaluator.py` — `evaluate_all_models()` |
| Stratified k-fold cross-validation (k=5, scoring=f1) | `src/models/evaluator.py` — `cross_validate_model()` |
| Vẽ Confusion Matrix, ROC Curves, Precision-Recall Curves, Feature Importance | `src/models/evaluator.py` — `plot_*` functions |
| Lưu bảng so sánh models | `reports/model_comparison.csv` |
| Model registry: save/load với versioning (timestamp), pickle artifacts | `src/models/registry.py` |
| Lưu scaler + feature_names cùng model artifact | `src/models/registry.py` — `save_scaler()`, `save_feature_names()` |

**Kết quả model (test set)**:

| Model | F1 | ROC-AUC | Train time |
|-------|----|---------|-----------|
| Gradient Boosting | **0.7748** | **0.9688** | 27.1s |
| Random Forest | 0.6667 | 0.9595 | 0.96s |
| Voting Ensemble | 0.5529 | 0.9298 | 29.3s |
| Decision Tree | 0.5000 | 0.8439 | 0.60s |
| Logistic Regression | 0.1826 | 0.8368 | 23.8s |

---

### Thành viên 5 — [Tên] — FastAPI Backend & Deployment
**Mức đóng góp: ~20%**

| Nhiệm vụ | File/Chi tiết |
|----------|--------------|
| FastAPI app entrypoint, lifespan (startup/shutdown), CORS middleware | `api/server.py` |
| Load model artifact, scaler, feature_names khi startup | `api/server.py` — `_load_model_artifacts()` |
| Pydantic schemas: `SensorReading`, `PredictionResponse`, `BatchPredictionRequest`, `StreamStatus` | `api/schemas.py` |
| Endpoint `POST /predict` — single inference | `api/routes/predict.py` |
| Endpoint `POST /predict/batch` — batch inference | `api/routes/predict.py` |
| Feature preparation cho inference (align → interaction → OHE → scale) | `api/routes/predict.py` — `_prepare_features()` |
| Dockerfile cho API service | `deployment/Dockerfile` |
| Dockerfile cho Dashboard service | `deployment/Dockerfile.dashboard` |
| docker-compose.yml (API + Dashboard + volume mounts) | `deployment/docker-compose.yml` |
| Dask performance benchmarking (partition sizes, throughput) | `src/optimization/__init__.py` |

---

## 📁 Cấu trúc thư mục & file ownership

```
bigdata/
├── config/
│   └── config.yaml                  # TV2 (data) + TV1 (monitoring)
├── data/
│   ├── raw/machine failure.csv      # dataset gốc
│   └── processed/                   # TV2 (preprocessing output)
├── src/
│   ├── config.py                    # chung
│   ├── data/
│   │   ├── loader.py                # TV2
│   │   ├── preprocessing.py         # TV2
│   │   └── feature_engineering.py   # TV3
│   ├── models/
│   │   ├── trainer.py               # TV4
│   │   ├── evaluator.py             # TV4
│   │   └── registry.py              # TV4
│   ├── monitoring/
│   │   ├── stream.py                # TV1
│   │   └── alerting.py              # TV1
│   └── optimization/
│       └── __init__.py              # TV5
├── api/
│   ├── server.py                    # TV5
│   ├── schemas.py                   # TV5
│   └── routes/
│       ├── predict.py               # TV5
│       ├── monitor.py               # TV1
│       └── health.py                # TV1
├── dashboard/
│   └── app.py                       # TV1
├── scripts/
│   ├── train.py                     # TV4
│   └── simulate_stream.py           # TV1
├── deployment/
│   ├── Dockerfile                   # TV5
│   ├── Dockerfile.dashboard         # TV5
│   ├── docker-compose.yml           # TV5
│   └── ism.repme.tech.conf          # TV1
├── models/                          # TV4 (artifacts)
├── reports/
│   ├── model_comparison.csv         # TV4
│   └── figures/                     # TV4
└── main.py                          # TV3 (pipeline orchestrator)
```

---

## 🔧 Technology Stack

| Category | Tools |
|----------|-------|
| **Big Data Processing** | Dask DataFrame, Dask Distributed (4 workers × 2 threads) |
| **ML/Modeling** | Scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE) |
| **API Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend/Dashboard** | Streamlit, Plotly |
| **Infrastructure** | Nginx (reverse proxy), Tailscale (VPN), systemd |
| **Containerization** | Docker, Docker Compose |
| **Storage Format** | CSV (raw) → Parquet partitioned (processed) |
| **Environment** | Micromamba / Conda, Python 3.10 |

---

## 🗓️ Kế hoạch thực hiện (7 Phases)

### Phase 1 — Data Acquisition & EDA *(TV2)*
- Download dataset, load bằng Dask, khám phá cấu trúc
- Visualize distributions, class imbalance, correlations
- Convert sang Parquet

### Phase 2 — Preprocessing *(TV2)*
- Missing values, outlier removal (IQR), one-hot encoding
- StandardScaler (fit on train, transform on test/stream)

### Phase 3 — Feature Engineering *(TV3)*
- Rolling stats (windows 3/5/10), rate-of-change, interaction features
- Anomaly indicators (z-score), wear_stage, cumulative_usage
- Tạo `stream_test_data.parquet` (61 features)

### Phase 4 — Model Training *(TV4)*
- SMOTE, stratified split
- Train 6 algorithms + Voting Ensemble
- Hyperparameter tuning

### Phase 5 — Model Evaluation *(TV4)*
- Metrics, cross-validation, ROC/PR curves, SHAP / feature importance
- Chọn best model: Gradient Boosting (F1=0.7748, AUC=0.9688)

### Phase 6 — Real-time Monitoring & Alerting *(TV1 + TV5)*
- FastAPI backend: /predict, /predict/batch, schemas *(TV5)*
- FastAPI backend: /monitor/*, /health, WebSocket, `_init_stream_simulator` *(TV1)*
- StreamSimulator: batch streaming, inference, alerting *(TV1)*
- Streamlit Live Monitor dashboard *(TV1)*

### Phase 7 — Deployment & Infrastructure *(TV1 + TV5)*
- Docker + docker-compose *(TV5)*
- Nginx reverse proxy + SSL + Tailscale *(TV1)*
- systemd services (auto-start on boot) *(TV1)*
- Dask performance benchmarking *(TV5)*
