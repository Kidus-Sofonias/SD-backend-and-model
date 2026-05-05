from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.core.config import settings


@dataclass
class RoutePoint:
    ts: datetime
    lat: float
    lon: float
    speed_mps: float | None = None
    accuracy_m: float | None = None


@dataclass
class RouteSnapResult:
    snapped_points: list[dict]
    source: str | None
    status: str


class RouteSnapService:
    def __init__(self) -> None:
        self.base_url = settings.route_snap_base_url.rstrip("/")
        self.enabled = settings.route_snap_enabled

    def snap(self, points: list[dict]) -> RouteSnapResult:
        if not self.enabled:
            return RouteSnapResult(snapped_points=[], source=None, status="disabled")
        if len(points) < 4:
            return RouteSnapResult(snapped_points=[], source=None, status="not_enough_points")

        sampled = self._sample_points(points, max_points=80)
        if len(sampled) < 4:
            return RouteSnapResult(snapped_points=[], source=None, status="not_enough_points")

        matched = self._try_match(sampled)
        if matched:
            return RouteSnapResult(snapped_points=matched, source="osrm_match", status="snapped")

        routed = self._try_route(sampled)
        if routed:
            return RouteSnapResult(snapped_points=routed, source="osrm_route", status="snapped")

        return RouteSnapResult(snapped_points=[], source=None, status="unavailable")

    def _try_match(self, points: list[dict]) -> list[dict]:
        coordinates = ";".join(f"{point['lon']:.6f},{point['lat']:.6f}" for point in points)
        url = f"{self.base_url}/match/v1/driving/{coordinates}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "tidy": "true",
            "steps": "false",
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        if payload.get("code") != "Ok":
            return []
        matchings = payload.get("matchings") or []
        if not matchings:
            return []
        coordinates_geo = ((matchings[0].get("geometry") or {}).get("coordinates")) or []
        return self._coordinates_to_points(coordinates_geo, points)

    def _try_route(self, points: list[dict]) -> list[dict]:
        coordinates = ";".join(f"{point['lon']:.6f},{point['lat']:.6f}" for point in points)
        url = f"{self.base_url}/route/v1/driving/{coordinates}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
            "continue_straight": "true",
        }
        try:
            with httpx.Client(timeout=8.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            return []

        if payload.get("code") != "Ok":
            return []
        routes = payload.get("routes") or []
        if not routes:
            return []
        coordinates_geo = ((routes[0].get("geometry") or {}).get("coordinates")) or []
        return self._coordinates_to_points(coordinates_geo, points)

    def _coordinates_to_points(self, coordinates_geo: list, original_points: list[dict]) -> list[dict]:
        if not coordinates_geo:
            return []
        started_at = original_points[0]["ts"]
        ended_at = original_points[-1]["ts"]
        start_epoch = started_at.timestamp()
        end_epoch = ended_at.timestamp()
        span_seconds = max(1.0, end_epoch - start_epoch)
        total = max(1, len(coordinates_geo) - 1)
        points: list[dict] = []
        for index, coordinate in enumerate(coordinates_geo):
            if not isinstance(coordinate, list) or len(coordinate) < 2:
                continue
            lon = float(coordinate[0])
            lat = float(coordinate[1])
            ratio = index / total
            ts = datetime.fromtimestamp(start_epoch + span_seconds * ratio, tz=timezone.utc)
            points.append(
                {
                    "ts": ts,
                    "lat": lat,
                    "lon": lon,
                    "speed_mps": None,
                    "accuracy_m": None,
                }
            )
        return points

    def _sample_points(self, points: list[dict], max_points: int) -> list[dict]:
        if len(points) <= max_points:
            return points
        stride = max(1, round((len(points) - 1) / (max_points - 1)))
        sampled = [points[index] for index in range(0, len(points), stride)]
        if sampled[-1]["ts"] != points[-1]["ts"]:
            sampled.append(points[-1])
        return sampled[:max_points]

