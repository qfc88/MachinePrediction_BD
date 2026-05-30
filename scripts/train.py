"""CLI script — Run full training pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import setup_logging, run_pipeline

if __name__ == "__main__":
    setup_logging()
    results = run_pipeline()
    print("\n✅ Training complete. Model comparison:")
    print(results.to_string(index=False))
