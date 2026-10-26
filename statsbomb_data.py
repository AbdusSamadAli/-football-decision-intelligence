import warnings
import pickle
from pathlib import Path

import pandas as pd


warnings.filterwarnings("ignore", module="statsbombpy")

from statsbombpy import sb

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / ".cache_statsbomb"
CACHE_DIR.mkdir(exist_ok=True)


CURATED_MATCHES = {
    "wc_2022_final": {
        "match_id": 3869685,
        "label": "World Cup 2022 Final - Argentina 3-3 France",
        "home_team": "Argentina", "away_team": "France",
    },
    "ucl_2019_final": {
        "match_id": 22912,
        "label": "Champions League 2019 Final - Tottenham 0-2 Liverpool",
        "home_team": "Tottenham Hotspur", "away_team": "Liverpool",
    },
    "wc_2018_final": {
        "match_id": 8658,
        "label": "World Cup 2018 Final - France 4-2 Croatia",
        "home_team": "France", "away_team": "Croatia",
    },
    "wc_2022_arg_cro_semi": {
        "match_id": 3869519,
        "label": "World Cup 2022 Semi-final - Argentina 3-0 Croatia",
        "home_team": "Argentina", "away_team": "Croatia",
    },
}


def list_competitions() -> list[dict]:
    """Return the free competitions/seasons available in StatsBomb open data."""
    comps = sb.competitions()
    return comps[
        [
            "competition_id",
            "season_id",
            "country_name",
            "competition_name",
            "season_name",
        ]
    ].to_dict("records")


def list_matches(competition_id: int, season_id: int) -> list[dict]:
    """Return normalized matches for a competition+season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    cols = {
        "match_id": "match_id",
        "match_date": "date",
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
    }
    available = [c for c in cols if c in matches.columns]
    out = matches[available].rename(columns=cols)
    return out.to_dict("records")


def get_events(match_id: int) -> pd.DataFrame:
    """Return the full event DataFrame for a match, cached to disk."""
    if not isinstance(match_id, int) or isinstance(match_id, bool) or match_id <= 0:
        raise ValueError("match_id must be a positive integer")

    cache_file = CACHE_DIR / f"events_{match_id}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    events = sb.events(match_id=match_id)
    if events is None or events.empty:
        raise RuntimeError(f"No events returned for match_id {match_id}")

    with open(cache_file, "wb") as fh:
        pickle.dump(events, fh)
    return events


if __name__ == "__main__":
    print("Free competitions (first 10):")
    for comp in list_competitions()[:10]:
        print(
            f"  [{comp['competition_id']}/{comp['season_id']}] "
            f"{comp['competition_name']} {comp['season_name']}"
        )

    demo = CURATED_MATCHES["ucl_2019_final"]
    print(f"\nLoading events for: {demo['label']}")
    ev = get_events(demo["match_id"])
    print(f"Events: {len(ev)} rows, {len(ev.columns)} columns")
    print(f"Teams: {list(ev['team'].dropna().unique())}")
