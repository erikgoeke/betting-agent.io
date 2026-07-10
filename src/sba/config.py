from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CURRENT_YEAR = datetime.now().year
DEFAULT_SEASONS = list(range(CURRENT_YEAR - 5, CURRENT_YEAR + 1))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLAYERS_DIR = DATA_DIR / "players"
PLAYER_BATTING_DIR = PLAYERS_DIR / "batting"
PLAYER_PITCHING_DIR = PLAYERS_DIR / "pitching"

for _dir in (DATA_DIR, MODELS_DIR, LOGS_DIR, PLAYERS_DIR, PLAYER_BATTING_DIR, PLAYER_PITCHING_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
GAMES_CACHE_PATH = DATA_DIR / "games.parquet"
MODEL_PATH = MODELS_DIR / "mlb_moneyline.joblib"
PICKS_LOG_PATH = LOGS_DIR / "picks.csv"
PLAYER_ID_CACHE_PATH = PLAYERS_DIR / "id_lookup.json"
STARTS_CACHE_PATH = DATA_DIR / "starts.parquet"
WEATHER_CACHE_PATH = DATA_DIR / "weather.parquet"
TEAM_GAME_STATS_CACHE_PATH = DATA_DIR / "team_game_stats.parquet"
HARD_HIT_CACHE_PATH = DATA_DIR / "hard_hit.parquet"
