# File role: Weather lookup for Phase 6 driver monitoring (details mode).
# Proxies Open-Meteo (free, no API key, model-based grid coverage that works in
# remote/rural areas where station-based providers have gaps - relevant for
# Ethiopian roads) and caches results with a TTL so polling the details mode
# never hammers the upstream and survives brief network flakiness.
# Connects to: httpx, app.core.errors.
# Key symbols/vars: WeatherService, weather_service, WMO_WEATHER_LABELS.
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import httpx

from app.core.errors import AppError

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_S = 6.0
CACHE_TTL_S = 15 * 60  # 15 minutes: current conditions change slowly
# Rounding granularity for the cache key: 2 decimals ~= 1.1 km grid, plenty for
# weather and keeps the cache small across many GPS fixes.
CACHE_ROUND = 2
# Cache key: (lat, lon) -> (fetched_at, payload)
WeatherCache = Dict[Tuple[float, float], Tuple[float, dict]]

# WMO weather interpretation codes (https://open-meteo.com/en/docs).
WMO_WEATHER_LABELS: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Dense freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Light showers",
    81: "Showers",
    82: "Violent showers",
    85: "Snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def wmo_weather_label(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return WMO_WEATHER_LABELS.get(int(code), "Unknown")


@dataclass
class WeatherService:
    cache: WeatherCache = field(default_factory=dict)
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=REQUEST_TIMEOUT_S))

    @staticmethod
    def _cache_key(lat: float, lon: float) -> Tuple[float, float]:
        return (round(float(lat), CACHE_ROUND), round(float(lon), CACHE_ROUND))

    def _build_payload(self, raw: dict) -> dict:
        """Map the Open-Meteo response into the compact shape the app renders."""
        current = raw.get("current") or {}
        daily = raw.get("daily") or {}

        forecast: list[dict] = []
        dates = daily.get("time") or []
        codes = daily.get("weather_code") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        precip = daily.get("precipitation_probability_max") or []
        for index, date in enumerate(dates):
            forecast.append(
                {
                    "date": date,
                    "code": codes[index] if index < len(codes) else None,
                    "t_max_c": tmax[index] if index < len(tmax) else None,
                    "t_min_c": tmin[index] if index < len(tmin) else None,
                    "precip_prob_pct": precip[index] if index < len(precip) else None,
                }
            )

        return {
            "timezone": raw.get("timezone_abbreviation"),
            "current": {
                "ts": current.get("time"),
                "temp_c": current.get("temperature_2m"),
                "apparent_temp_c": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precip_mm": current.get("precipitation"),
                "wind_kph": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "weather_label": wmo_weather_label(current.get("weather_code")),
                "is_day": current.get("is_day"),
            },
            "forecast": forecast,
        }

    def lookup(self, *, lat: float, lon: float) -> dict:
        """Return current conditions + short forecast for a lat/lon.

        Serves from cache when fresh; falls back to stale cache (marked
        ``stale``) if the upstream is unreachable so the app never blocks the
        live-monitoring view on a weather outage.
        """
        key = self._cache_key(lat, lon)
        now = time.monotonic()

        cached = self.cache.get(key)
        if cached and now - cached[0] < CACHE_TTL_S:
            return {**cached[1], "cached": True, "stale": False}

        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m,is_day"
                ),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            }
            response = self.client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            payload = self._build_payload(response.json())
            self.cache[key] = (now, payload)
            return {**payload, "cached": True, "stale": False}
        except Exception:
            logger.exception("Open-Meteo lookup failed (lat=%s, lon=%s)", lat, lon)
            if cached:
                # Serve slightly outdated data rather than failing the screen.
                return {**cached[1], "cached": True, "stale": True}
            raise AppError(
                message_key="weather.unavailable",
                status_code=503,
                details={"reason": "weather provider unreachable"},
            )


weather_service = WeatherService()
