# File role: WebSocket route exposing the driver live-alert stream.
# Auth uses a JWT passed as a query parameter (`?token=...`) because RN's
# WebSocket cannot reliably send Authorization headers on all platforms.
# Key symbols/vars: router, ws_alerts.
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from jose import JWTError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.jwt import decode_token
from app.db.session import get_db
from app.realtime.hub import alert_hub
from app.realtime.live_detector import live_alert_detector
from app.repositories.trip_repository import SqlTripRepository
from app.repositories.user_repository import SqlUserRepository
from app.services.live_monitor_service import LiveMonitorService

logger = logging.getLogger(__name__)

router = APIRouter()

WS_AUTH_FAILURE_CODE = 4401
PING_INTERVAL_S = 20


@router.websocket("/ws/alerts")
async def ws_alerts(
    websocket: WebSocket,
    token: str = Query(default=""),
    db: Session = Depends(get_db),
):
    """Stream live driving-event alerts for the authenticated user.

    Frame shape (JSON):
      {"type": "connected", "trip_id": null}
      {"type": "event_alert", "trip_id": "...", "event": {...}, "sent_at": "..."}
      {"type": "ping"}  # sent every 20s while idle to keep proxies alive
    """
    user_id = None
    try:
        if not token:
            await websocket.close(code=WS_AUTH_FAILURE_CODE)
            return
        try:
            payload = decode_token(token)
        except JWTError:
            await websocket.close(code=WS_AUTH_FAILURE_CODE)
            return
        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=WS_AUTH_FAILURE_CODE)
            return

        user = SqlUserRepository(db).get_by_id(user_id)
        if not user:
            await websocket.close(code=WS_AUTH_FAILURE_CODE)
            return
    except Exception:
        logger.exception("WebSocket auth failed")
        await websocket.close(code=WS_AUTH_FAILURE_CODE)
        return

    await websocket.accept()
    queue = alert_hub.subscribe(user_id)
    await websocket.send_json({"type": "connected", "trip_id": None})
    logger.info("Alert stream connected for user %s", user_id)

    loop = asyncio.get_running_loop()
    last_ping_at = loop.time()

    try:
        while True:
            # Race the next queued alert against client disconnect / ping. The
            # timeout guarantees the keepalive branch below is reachable even
            # when the connection is idle (both tasks otherwise block forever).
            recv_task = asyncio.ensure_future(websocket.receive_text())
            alert_task = asyncio.ensure_future(queue.get())
            done, pending = await asyncio.wait(
                {recv_task, alert_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=PING_INTERVAL_S,
            )
            for task in pending:
                task.cancel()

            if recv_task in done and not recv_task.cancelled():
                try:
                    await recv_task
                except WebSocketDisconnect:
                    break
                # Client sent a message (e.g. ping/pong) - ignore and continue.

            if alert_task in done and not alert_task.cancelled():
                message = alert_task.result()
                await websocket.send_json(message)

            # Keep the connection alive through idle proxies.
            now = loop.time()
            if now - last_ping_at >= PING_INTERVAL_S:
                last_ping_at = now
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Alert stream error for user %s", user_id)
    finally:
        alert_hub.unsubscribe(user_id, queue)
        logger.info("Alert stream closed for user %s", user_id)


@router.get("/trips/{trip_id}/telemetry")
def trip_live_telemetry(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Live telemetry for the caller's own trip (Phase 6 glance/details modes).

    Returns current speed, IMU/longitudinal acceleration, latest GPS fix, trip
    state, and live event counters/alert replay from the in-memory detector.
    Event counters reset on server restart (transient live alerts are not
    persisted); finalize remains the source of truth for stored events.
    """
    service = LiveMonitorService(db)
    return service.get_trip_telemetry(user_id=user.id, trip_id=trip_id)


@router.get("/trips/{trip_id}/alerts/recent")
def recent_trip_alerts(
    trip_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Recent live alerts for the caller's own trip (reconnect replay buffer).

    In-memory only (per process); returns [] after a restart. Used by the
    driver app to backfill alerts it may have missed while disconnected.
    """
    repo = SqlTripRepository(db)
    if not repo.get_by_id(trip_id=trip_id, user_id=user.id):
        from app.core.errors import NotFoundError

        raise NotFoundError(message_key="trip.not_found")
    alerts = live_alert_detector.recent_alerts(trip_id)
    return {"trip_id": trip_id, "alerts": alerts}
