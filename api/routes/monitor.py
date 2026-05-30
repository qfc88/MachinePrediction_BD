"""Monitoring endpoints — stream status and WebSocket live feed."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.schemas import StreamStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitoring"])


@router.get("/monitor/status", response_model=StreamStatus)
async def stream_status():
    """Get current streaming pipeline status."""
    from api.server import app_state

    simulator = app_state.get("stream_simulator")
    if simulator is None:
        return StreamStatus(
            is_running=False, total_predictions=0,
            failure_rate=0, avg_probability=0, total_alerts=0,
        )

    stats = simulator.get_stats()
    return StreamStatus(
        is_running=simulator.is_running,
        total_predictions=stats.get("total_predictions", 0),
        failure_rate=round(stats.get("failure_rate", 0), 4),
        avg_probability=round(stats.get("avg_probability", 0), 4),
        total_alerts=stats.get("total_alerts", 0),
        recent_alerts=simulator.alert_manager.get_recent_alerts(10),
    )


@router.post("/monitor/start")
async def start_streaming():
    """Start the streaming simulation in background."""
    from api.server import app_state

    simulator = app_state.get("stream_simulator")
    if simulator is None:
        return {"error": "Stream simulator not initialized. Run training first."}

    if simulator.is_running:
        return {"status": "already_running"}

    # Run in background thread
    import threading
    thread = threading.Thread(target=simulator.run, kwargs={"max_batches": 200}, daemon=True)
    thread.start()
    return {"status": "started"}


@router.post("/monitor/stop")
async def stop_streaming():
    """Stop the streaming simulation."""
    from api.server import app_state

    simulator = app_state.get("stream_simulator")
    if simulator:
        simulator.stop()
    return {"status": "stopped"}


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for live prediction stream."""
    from api.server import app_state

    await websocket.accept()
    simulator = app_state.get("stream_simulator")

    if simulator is None:
        await websocket.send_json({"error": "Simulator not initialized"})
        await websocket.close()
        return

    try:
        last_count = 0
        while True:
            stats = simulator.get_stats()
            current_count = stats.get("total_predictions", 0)

            if current_count > last_count:
                # Send latest predictions
                recent = list(simulator.predictions_log)[-10:]
                payload = {
                    "stats": stats,
                    "recent_predictions": recent,
                    "alerts": simulator.alert_manager.get_recent_alerts(5),
                }
                # Convert numpy types for JSON serialization
                await websocket.send_text(json.dumps(payload, default=str))
                last_count = current_count

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
