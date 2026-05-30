"""Dask performance optimization — partition tuning, benchmarking."""

import logging
import time
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from src.config import get_config
from src.data.loader import get_dask_client

logger = logging.getLogger(__name__)


def benchmark_partitions(filepath: str, partition_sizes: list[int] | None = None) -> pd.DataFrame:
    """Benchmark different partition sizes for Dask operations.

    Returns DataFrame with partition_count, operation, and elapsed time.
    """
    if partition_sizes is None:
        partition_sizes = [1, 2, 4, 8, 16]

    client = get_dask_client()
    results = []

    for n in partition_sizes:
        ddf = dd.read_csv(filepath).repartition(npartitions=n)

        # Benchmark: full scan (describe)
        t0 = time.time()
        ddf.describe().compute()
        t_describe = time.time() - t0

        # Benchmark: groupby aggregation
        cat_col = get_config()["data"]["categorical_columns"][0]
        if cat_col in ddf.columns:
            t0 = time.time()
            ddf.groupby(cat_col).mean(numeric_only=True).compute()
            t_groupby = time.time() - t0
        else:
            t_groupby = None

        # Benchmark: filter
        sensor = get_config()["data"]["sensor_columns"][0]
        t0 = time.time()
        median_val = ddf[sensor].mean().compute()
        ddf[ddf[sensor] > median_val].compute()
        t_filter = time.time() - t0

        results.append({"partitions": n, "describe_sec": t_describe,
                        "groupby_sec": t_groupby, "filter_sec": t_filter})
        logger.info("Partitions=%d: describe=%.3fs, groupby=%.3fs, filter=%.3fs",
                     n, t_describe, t_groupby or 0, t_filter)

    return pd.DataFrame(results)


def optimize_partition_size(filepath: str) -> int:
    """Find optimal partition count by benchmarking."""
    df = benchmark_partitions(filepath)
    # Use the partition count with lowest total time
    df["total"] = df["describe_sec"] + df["groupby_sec"].fillna(0) + df["filter_sec"]
    best = df.loc[df["total"].idxmin()]
    optimal = int(best["partitions"])
    logger.info("Optimal partition count: %d (total=%.3fs)", optimal, best["total"])
    return optimal


def benchmark_pipeline(pipeline_fn, *args, n_runs: int = 3, **kwargs) -> dict:
    """Benchmark any pipeline function over n_runs."""
    times = []
    for i in range(n_runs):
        t0 = time.time()
        pipeline_fn(*args, **kwargs)
        elapsed = time.time() - t0
        times.append(elapsed)
        logger.info("Run %d/%d: %.3fs", i + 1, n_runs, elapsed)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "runs": times,
    }


def get_cluster_info() -> dict:
    """Get current Dask cluster diagnostics."""
    client = get_dask_client()
    info = client.scheduler_info()
    workers = info.get("workers", {})
    return {
        "n_workers": len(workers),
        "total_threads": sum(w.get("nthreads", 0) for w in workers.values()),
        "total_memory": sum(w.get("memory_limit", 0) for w in workers.values()),
        "dashboard_link": client.dashboard_link,
        "scheduler": info.get("address", ""),
    }
