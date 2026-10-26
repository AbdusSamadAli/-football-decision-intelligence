import os
import re
import time
from pathlib import Path
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_URL = "https://api.football-data.org/v4"
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing FOOTBALL_DATA_API_KEY in .env")

VALID_STATUSES = {
    "SCHEDULED",
    "TIMED",
    "IN_PLAY",
    "PAUSED",
    "EXTRA_TIME",
    "PENALTY_SHOOTOUT",
    "FINISHED",
    "POSTPONED",
    "SUSPENDED",
    "CANCELLED",
    "AWARDED",
}

MAX_MATCH_LIMIT = 20
MAX_SCORER_LIMIT = 50


SUPPORTED_COMPETITIONS = {
    "PL": "Premier League (England)",
    "ELC": "Championship (England)",
    "PD": "La Liga (Spain)",
    "SA": "Serie A (Italy)",
    "BL1": "Bundesliga (Germany)",
    "FL1": "Ligue 1 (France)",
    "DED": "Eredivisie (Netherlands)",
    "PPL": "Primeira Liga (Portugal)",
    "BSA": "Campeonato Brasileiro Serie A (Brazil)",
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "WC": "FIFA World Cup",
}


def validate_competition_code(code):
    """Validate the basic format of a competition code."""
    if not isinstance(code, str):
        raise ValueError("competition_code must be a string")

    code = code.strip().upper()

    if not re.fullmatch(r"[A-Z0-9]{2,5}", code):
        raise ValueError(
            "Invalid competition code format; use a code such as PL"
        )

    return code


def validate_status(status):
    """Validate a football-data.org match status."""
    if not isinstance(status, str):
        raise ValueError("status must be a string")

    status = status.strip().upper()

    if status not in VALID_STATUSES:
        raise ValueError(
            f"Unsupported status '{status}'. "
            f"Allowed statuses: {', '.join(sorted(VALID_STATUSES))}"
        )

    return status


def validate_match_limit(limit):
    """Allow a reasonable number of matches per request."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= MAX_MATCH_LIMIT:
        raise ValueError(
            f"limit must be between 1 and {MAX_MATCH_LIMIT}"
        )

    return limit


def validate_scorer_limit(limit):
    """Allow a reasonable number of top scorers per request."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")

    if not 1 <= limit <= MAX_SCORER_LIMIT:
        raise ValueError(
            f"limit must be between 1 and {MAX_SCORER_LIMIT}"
        )

    return limit


def validate_match_id(match_id):
    """Require a positive integer match ID."""
    if isinstance(match_id, bool) or not isinstance(match_id, int):
        raise ValueError("match_id must be an integer")

    if match_id <= 0:
        raise ValueError("match_id must be positive")

    return match_id


def api_get(endpoint, params=None):
    """Make an authenticated request with clear HTTP/network errors."""
    url = f"{BASE_URL}{endpoint}"


    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers={"X-Auth-Token": API_KEY},
                params=params,
                timeout=20,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                "Football data API request timed out"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not connect to football data API: {exc}"
            ) from exc

        if response.status_code == 429:
            if attempt == 2:
                raise RuntimeError(
                    "Football data API rate limit reached (HTTP 429). "
                    "Wait before making more requests."
                )


            retry_after = response.headers.get("Retry-After", "")
            wait_seconds = 2

            try:
                wait_seconds = min(max(int(retry_after), 1), 10)
            except (ValueError, TypeError):
                try:
                    retry_date = parsedate_to_datetime(retry_after)
                    if retry_date.tzinfo is None:
                        retry_date = retry_date.replace(tzinfo=timezone.utc)
                    wait_seconds = min(
                        max(
                            int((retry_date - datetime.now(timezone.utc)).total_seconds()),
                            1,
                        ),
                        10,
                    )
                except (ValueError, TypeError, OverflowError):
                    pass

            time.sleep(wait_seconds)
            continue

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Football data API rejected the request "
                f"(HTTP {response.status_code}). Check your API key "
                "and subscription permissions."
            )

        if response.status_code == 404:
            raise RuntimeError(
                f"Football data API resource not found (HTTP 404): "
                f"{endpoint}"
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Football data API returned HTTP "
                f"{response.status_code}: {response.text[:300]}"
            ) from exc

        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError(
                "Football data API returned an invalid JSON response"
            ) from exc

    raise RuntimeError("Football data API request failed after retries")


def get_competition(code="PL"):
    """Retrieve competition metadata."""
    code = validate_competition_code(code)
    data = api_get(f"/competitions/{code}")

    try:
        return {
            "id": data["id"],
            "name": data["name"],
            "country": data["area"]["name"],
            "code": data["code"],
        }
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Competition API response is missing required fields"
        ) from exc


def get_matches(competition_code="PL", status="FINISHED", limit=10):
    """Retrieve and normalize multiple matches."""
    competition_code = validate_competition_code(competition_code)
    status = validate_status(status)
    limit = validate_match_limit(limit)

    data = api_get(
        f"/competitions/{competition_code}/matches",
        params={"status": status},
    )

    if not isinstance(data.get("matches"), list):
        raise RuntimeError(
            "Matches API response is missing its matches list"
        )

    matches = []

    for match in data["matches"][:limit]:
        try:
            score = match.get("score") or {}
            full_time = score.get("fullTime") or {}
            competition = match.get("competition") or {}
            home_team = match.get("homeTeam") or {}
            away_team = match.get("awayTeam") or {}

            matches.append({
                "match_id": match["id"],
                "date": match["utcDate"],
                "status": match["status"],
                "competition": competition.get("name"),
                "home_team": home_team.get("name"),
                "away_team": away_team.get("name"),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
            })
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "A match in the API response is missing required fields"
            ) from exc

    return matches


def get_match_details(match_id: int):
    """Retrieve and normalize one match by its ID."""
    match_id = validate_match_id(match_id)
    match = api_get(f"/matches/{match_id}")

    try:
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}
        competition = match.get("competition") or {}
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}

        return {
            "match_id": match["id"],
            "date": match["utcDate"],
            "status": match["status"],
            "competition": competition.get("name"),
            "home_team": home_team.get("name"),
            "away_team": away_team.get("name"),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
        }
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Match API response is missing required fields"
        ) from exc


def get_standings(competition_code="PL"):
    """Retrieve the full league table (TOTAL standings) for a competition."""
    competition_code = validate_competition_code(competition_code)
    data = api_get(f"/competitions/{competition_code}/standings")

    standings = data.get("standings")
    if not isinstance(standings, list) or not standings:
        raise RuntimeError("Standings API response is missing standings")


    total = next(
        (s for s in standings if s.get("type") == "TOTAL"),
        standings[0],
    )

    table = total.get("table")
    if not isinstance(table, list):
        raise RuntimeError("Standings API response is missing a table")

    rows = []
    for row in table:
        try:
            team = row.get("team") or {}
            rows.append({
                "position": row["position"],
                "team": team.get("name"),
                "played": row.get("playedGames"),
                "won": row.get("won"),
                "draw": row.get("draw"),
                "lost": row.get("lost"),
                "points": row.get("points"),
                "goals_for": row.get("goalsFor"),
                "goals_against": row.get("goalsAgainst"),
                "goal_difference": row.get("goalDifference"),
            })
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "A standings row is missing required fields"
            ) from exc

    return {"competition_code": competition_code, "table": rows}


def get_scorers(competition_code="PL", limit=10):
    """Retrieve the top scorers for a competition."""
    competition_code = validate_competition_code(competition_code)
    limit = validate_scorer_limit(limit)

    data = api_get(
        f"/competitions/{competition_code}/scorers",
        params={"limit": limit},
    )

    scorers = data.get("scorers")
    if not isinstance(scorers, list):
        raise RuntimeError("Scorers API response is missing its scorers list")

    result = []
    for entry in scorers:
        player = entry.get("player") or {}
        team = entry.get("team") or {}
        result.append({
            "player": player.get("name"),
            "team": team.get("name"),
            "goals": entry.get("goals"),
            "assists": entry.get("assists"),
            "penalties": entry.get("penalties"),
        })

    return result


if __name__ == "__main__":
    print("COMPETITION")
    print(get_competition())

    print("\nRECENT FINISHED MATCHES")
    for match in get_matches(limit=5):
        print(match)

    print("\nSTANDINGS (top 5)")
    for row in get_standings()["table"][:5]:
        print(row)

    print("\nTOP SCORERS (top 5)")
    for scorer in get_scorers(limit=5):
        print(scorer)
