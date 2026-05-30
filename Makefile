.PHONY: train serve dashboard stream all clean

# ── Pipeline ────────────────────────────────────────────────────
train:                          ## Run full training pipeline
	python main.py

# ── Services ────────────────────────────────────────────────────
serve:                          ## Start FastAPI prediction server
	python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

dashboard:                      ## Start Streamlit monitoring dashboard
	python -m streamlit run dashboard/app.py --server.port 8501

stream:                         ## Run streaming simulation
	python scripts/simulate_stream.py

# ── Docker ──────────────────────────────────────────────────────
docker-up:                      ## Start all services with Docker
	cd deployment && docker compose up --build -d

docker-down:                    ## Stop all Docker services
	cd deployment && docker compose down

# ── All-in-one ──────────────────────────────────────────────────
all: train serve                ## Train then serve

# ── Cleanup ─────────────────────────────────────────────────────
clean:                          ## Remove generated artifacts
	rm -rf models/*.pkl data/processed/* reports/figures/* logs/*.log

help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
