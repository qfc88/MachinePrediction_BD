"""Data loading module — Dask-based data ingestion from local/distributed storage."""

import logging
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from src.config import get_config

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_dask_client() -> Client:
    """Get or create a Dask distributed client (singleton)."""
    global _client
    if _client is None or _client.status != "running":
        cfg = get_config()["dask"]
        cluster = LocalCluster(
            n_workers=cfg["n_workers"],
            threads_per_worker=cfg["threads_per_worker"],
            memory_limit=cfg["memory_limit"],
            dashboard_address=f":{cfg['dashboard_port']}",
        )
        _client = Client(cluster)
        logger.info("Dask client started: %s", _client.dashboard_link)
    return _client


def shutdown_dask():
    """Gracefully shutdown Dask client and cluster."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("Dask client shut down.")


def load_raw_data(path: str | None = None, npartitions: int = 4) -> dd.DataFrame:
    """Load raw CSV data into a Dask DataFrame.

    Supports local paths and S3 URIs (s3://bucket/key).
    """
    if path is None:
        path = get_config()["paths"]["raw_data"]

    logger.info("Loading data from: %s", path)
    ddf = dd.read_csv(path)

    # Repartition for balanced workload
    ddf = ddf.repartition(npartitions=npartitions)
    logger.info("Loaded %d partitions, columns: %s", ddf.npartitions, list(ddf.columns))
    return ddf


def load_processed_data(path: str | None = None) -> dd.DataFrame:
    """Load processed Parquet data."""
    if path is None:
        path = get_config()["paths"]["processed_data"]

    logger.info("Loading processed data from: %s", path)
    return dd.read_parquet(path)


def save_processed_data(ddf: dd.DataFrame, path: str | None = None):
    """Save processed data as partitioned Parquet."""
    if path is None:
        path = get_config()["paths"]["processed_data"]

    Path(path).mkdir(parents=True, exist_ok=True)
    ddf.to_parquet(path, overwrite=True, engine="pyarrow")
    logger.info("Saved processed data to: %s", path)
