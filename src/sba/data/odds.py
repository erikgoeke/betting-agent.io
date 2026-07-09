"""Client for The Odds API (the-odds-api.com) + devig math.

Free tier: 25 requests/day, MLB included, h2h (moneyline) market only.
No historical odds on the free tier -- see README for what that means for backtesting.
"""

from __future__ import annotations

import requests

from sba.config import ODDS_API_KEY

BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"

# The Odds API returns full team names; our model works off Baseball-Reference abbreviations.
TEAM_NAME_TO_ABBREV = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFG",
    "St. Louis Cardinals": "STL",
    "St Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


class OddsAPIError(RuntimeError):
    pass


def american_to_prob(odds: float) -> float:
    """Convert American odds to raw (vig-included) implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Proportional devig: normalize two implied probabilities to sum to 1."""
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds (total payout per unit staked)."""
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / -odds


def fetch_mlb_odds(api_key: str | None = None) -> list[dict]:
    """Fetch today's MLB moneyline odds. Raises OddsAPIError on failure or missing key."""
    key = api_key or ODDS_API_KEY
    if not key:
        raise OddsAPIError("No ODDS_API_KEY set. Get a free key at https://the-odds-api.com/ and add it to .env")

    resp = requests.get(
        BASE_URL,
        params={"apiKey": key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
        timeout=15,
    )
    if resp.status_code != 200:
        raise OddsAPIError(f"Odds API request failed ({resp.status_code}): {resp.text}")
    return resp.json()


def games_with_devigged_odds(raw_games: list[dict]) -> list[dict]:
    """Parse raw API games into a simplified structure with a fair (devigged),
    bookmaker-averaged home win probability."""
    parsed = []
    for game in raw_games:
        home_name, away_name = game["home_team"], game["away_team"]
        home_abbrev = TEAM_NAME_TO_ABBREV.get(home_name)
        away_abbrev = TEAM_NAME_TO_ABBREV.get(away_name)
        if home_abbrev is None or away_abbrev is None:
            continue  # unmapped team name; skip rather than guess

        fair_home_probs = []
        home_prices, away_prices = [], []
        for bookmaker in game.get("bookmakers", []):
            h2h = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
            if h2h is None:
                continue
            outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
            if home_name not in outcomes or away_name not in outcomes:
                continue
            raw_home_prob = american_to_prob(outcomes[home_name])
            raw_away_prob = american_to_prob(outcomes[away_name])
            fair_home_prob, _ = devig_two_way(raw_home_prob, raw_away_prob)
            fair_home_probs.append(fair_home_prob)
            home_prices.append(outcomes[home_name])
            away_prices.append(outcomes[away_name])

        if not fair_home_probs:
            continue

        parsed.append(
            {
                "game_id": game["id"],
                "commence_time": game["commence_time"],
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "n_bookmakers": len(fair_home_probs),
                "market_home_win_prob": sum(fair_home_probs) / len(fair_home_probs),
                # Best (highest payout) price available for each side, for stake sizing.
                "best_home_price": max(home_prices, key=american_to_decimal),
                "best_away_price": max(away_prices, key=american_to_decimal),
            }
        )
    return parsed
