"""Health check endpoint."""

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health, model status, and Dask cluster."""
    from api.server import app_state

    dask_status = "unknown"
    try:
        from src.data.loader import get_dask_client
        client = get_dask_client()
        dask_status = client.status
    except Exception:
        dask_status = "disconnected"

    return HealthResponse(
        status="healthy",
        model_loaded=app_state.get("model") is not None,
        dask_status=dask_status,
        version="1.0.0",
    )
