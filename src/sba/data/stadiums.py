"""Static per-team home-ballpark location, used for travel-distance and
weather features. Coordinates are approximate (city-level, not the exact
gate), which is precise enough for travel-distance and matching a weather grid
cell -- this is not used for anything requiring rooftop-level accuracy.

Run-scoring park factors are deliberately NOT hardcoded here; see
park_factors.py, which computes them empirically from this project's own
games.parquet instead of trusting a possibly-stale published number.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0

# lat/lon: home ballpark location. tz: home ballpark's local timezone (IANA name).
STADIUMS: dict[str, dict] = {
    "ARI": {"lat": 33.4455, "lon": -112.0667, "tz": "America/Phoenix"},
    "ATL": {"lat": 33.8908, "lon": -84.4678, "tz": "America/New_York"},
    "BAL": {"lat": 39.2839, "lon": -76.6218, "tz": "America/New_York"},
    "BOS": {"lat": 42.3467, "lon": -71.0972, "tz": "America/New_York"},
    "CHC": {"lat": 41.9484, "lon": -87.6553, "tz": "America/Chicago"},
    "CHW": {"lat": 41.8299, "lon": -87.6338, "tz": "America/Chicago"},
    "CIN": {"lat": 39.0975, "lon": -84.5071, "tz": "America/New_York"},
    "CLE": {"lat": 41.4962, "lon": -81.6852, "tz": "America/New_York"},
    "COL": {"lat": 39.7559, "lon": -104.9942, "tz": "America/Denver"},
    "DET": {"lat": 42.3390, "lon": -83.0485, "tz": "America/New_York"},
    "HOU": {"lat": 29.7573, "lon": -95.3555, "tz": "America/Chicago"},
    "KCR": {"lat": 39.0517, "lon": -94.4803, "tz": "America/Chicago"},
    "LAA": {"lat": 33.8003, "lon": -117.8827, "tz": "America/Los_Angeles"},
    "LAD": {"lat": 34.0739, "lon": -118.2400, "tz": "America/Los_Angeles"},
    "MIA": {"lat": 25.7781, "lon": -80.2196, "tz": "America/New_York"},
    "MIL": {"lat": 43.0280, "lon": -87.9712, "tz": "America/Chicago"},
    "MIN": {"lat": 44.9817, "lon": -93.2777, "tz": "America/Chicago"},
    "NYM": {"lat": 40.7571, "lon": -73.8458, "tz": "America/New_York"},
    "NYY": {"lat": 40.8296, "lon": -73.9262, "tz": "America/New_York"},
    "OAK": {"lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles"},
    "ATH": {"lat": 38.5802, "lon": -121.4931, "tz": "America/Los_Angeles"},  # Sacramento, temporary home 2025+
    "PHI": {"lat": 39.9061, "lon": -75.1665, "tz": "America/New_York"},
    "PIT": {"lat": 40.4469, "lon": -80.0057, "tz": "America/New_York"},
    "SDP": {"lat": 32.7073, "lon": -117.1566, "tz": "America/Los_Angeles"},
    "SEA": {"lat": 47.5914, "lon": -122.3325, "tz": "America/Los_Angeles"},
    "SFG": {"lat": 37.7786, "lon": -122.3893, "tz": "America/Los_Angeles"},
    "STL": {"lat": 38.6226, "lon": -90.1928, "tz": "America/Chicago"},
    "TBR": {"lat": 27.7683, "lon": -82.6534, "tz": "America/New_York"},
    "TEX": {"lat": 32.7473, "lon": -97.0842, "tz": "America/Chicago"},
    "TOR": {"lat": 43.6414, "lon": -79.3894, "tz": "America/New_York"},
    "WSN": {"lat": 38.8730, "lon": -77.0074, "tz": "America/New_York"},
}

# Fixed UTC offsets (hours) for each timezone, ignoring DST transitions -- close
# enough for a "how many time zones did the traveling team cross" feature.
TZ_UTC_OFFSET: dict[str, int] = {
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Phoenix": -7,
    "America/Los_Angeles": -8,
}


def haversine_km(team_a: str, team_b: str) -> float:
    """Great-circle distance between two teams' home ballparks."""
    a, b = STADIUMS[team_a], STADIUMS[team_b]
    lat1, lon1, lat2, lon2 = radians(a["lat"]), radians(a["lon"]), radians(b["lat"]), radians(b["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def timezone_shift(from_team: str, to_team: str) -> int:
    """Hours the `from_team` must adjust traveling to play at `to_team`'s park."""
    return TZ_UTC_OFFSET[STADIUMS[to_team]["tz"]] - TZ_UTC_OFFSET[STADIUMS[from_team]["tz"]]
