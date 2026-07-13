"""Daily weather at each home ballpark, via Open-Meteo (free, no API key).

Two endpoints: the historical archive (for training data, one request per
team covering its whole date range) and the forecast API (for live picks,
which need today's/tomorrow's conditions rather than history).

Caveat: there's no free source for game-time roof status, so these are
outdoor-condition proxies at the park's location -- noisier for parks with a
retractable or fixed roof (TOR, MIA, ARI, HOU, MIL, TEX, SEA, ATH), since the
weather feature won't reflect whether the roof was actually closed that day.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

from sba.config import WEATHER_CACHE_PATH
from sba.data.stadiums import STADIUMS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_FIELDS = "temperature_2m_max,windspeed_10m_max,winddirection_10m_dominant,precipitation_sum"

WEATHER_COLUMNS = ["temp_max_c", "wind_speed_kmh", "wind_dir_deg", "precip_mm"]


def _daily_frame(team: str, payload: dict) -> pd.DataFrame:
    daily = payload["daily"]
    return pd.DataFrame(
        {
            "team": team,
            "date": pd.to_datetime(daily["time"]),
            "temp_max_c": daily["temperature_2m_max"],
            "wind_speed_kmh": daily["windspeed_10m_max"],
            "wind_dir_deg": daily["winddirection_10m_dominant"],
            "precip_mm": daily["precipitation_sum"],
        }
    )


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """Open-Meteo's free tier has a modest per-minute rate limit -- back off on
    429s and on 5xx server errors (its archive API occasionally 502s). Also
    retries transient network errors (timeouts, connection resets): a single
    bad response among ~30 team requests shouldn't crash the whole
    `sba train` run.

    Both the request itself AND the status check must be inside the retry
    loop's try/except -- an earlier version only wrapped requests.get(), so a
    502 (raised by raise_for_status(), not by requests.get() itself) skipped
    the retry logic entirely and crashed on the first bad response.
    """
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code in (429, 502, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(10 * (attempt + 1))
    if last_error is not None:
        raise last_error
    resp.raise_for_status()
    return resp


def fetch_team_history(team: str, start_date: str, end_date: str) -> pd.DataFrame:
    stadium = STADIUMS[team]
    resp = _get_with_retry(
        ARCHIVE_URL,
        params={
            "latitude": stadium["lat"], "longitude": stadium["lon"],
            "start_date": start_date, "end_date": end_date,
            "daily": DAILY_FIELDS, "timezone": stadium["tz"],
        },
    )
    time.sleep(2)
    return _daily_frame(team, resp.json())


def fetch_team_forecast(team: str) -> pd.DataFrame:
    """Today plus the next few days' forecast (and a few recent past days) for one park."""
    stadium = STADIUMS[team]
    resp = _get_with_retry(
        FORECAST_URL,
        params={
            "latitude": stadium["lat"], "longitude": stadium["lon"],
            "daily": DAILY_FIELDS, "timezone": stadium["tz"], "past_days": 1, "forecast_days": 10,
        },
    )
    return _daily_frame(team, resp.json())


def build_weather_history(start_date: str, end_date: str) -> pd.DataFrame:
    """One archive request per team (not per game) covering the whole range."""
    frames = [fetch_team_history(team, start_date, end_date) for team in STADIUMS]
    return pd.concat(frames, ignore_index=True)


def fetch_weather_history(start_date: str, end_date: str, *, force_refresh: bool = False) -> pd.DataFrame:
    if WEATHER_CACHE_PATH.exists() and not force_refresh:
        cached = pd.read_parquet(WEATHER_CACHE_PATH)
        if cached["date"].min() <= pd.Timestamp(start_date) and cached["date"].max() >= pd.Timestamp(end_date):
            return cached[(cached["date"] >= start_date) & (cached["date"] <= end_date)].reset_index(drop=True)

    fresh = build_weather_history(start_date, end_date)
    fresh.to_parquet(WEATHER_CACHE_PATH, index=False)
    return fresh


def game_day_weather(team: str, date: pd.Timestamp) -> dict:
    """A single home team's forecasted conditions for an upcoming game date."""
    forecast = fetch_team_forecast(team)
    match = forecast[forecast["date"] == pd.Timestamp(date).normalize()]
    if match.empty:
        return {col: None for col in WEATHER_COLUMNS}
    return match.iloc[0][WEATHER_COLUMNS].to_dict()
