import json

import database
import statsbomb_data
import tactical_metrics
from coaching_assistant import scan_chain, recommend_chain


def list_curated() -> dict:
    return statsbomb_data.CURATED_MATCHES


def list_competitions() -> list[dict]:
    comps = statsbomb_data.list_competitions()
    for c in comps:
        c["label"] = f"{c['competition_name']} {c['season_name']} ({c['country_name']})"
    return comps


def list_matches(competition_id: int, season_id: int) -> list[dict]:
    ms = statsbomb_data.list_matches(competition_id, season_id)
    for m in ms:
        m["label"] = f"{m['home_team']} {m['home_score']}-{m['away_score']} {m['away_team']}"
    return ms


def analyze(match_id: int, team: str, coach_context: str) -> dict:
    """Deterministic metrics, then the LLM problem scan."""
    metrics = tactical_metrics.compute_match_metrics(int(match_id))
    scan = scan_chain.invoke({
        "team": team,
        "coach_context": coach_context,
        "metrics": json.dumps(metrics, indent=2),
    })
    return {
        "match_id": int(match_id),
        "team": team,
        "coach_context": coach_context,
        "metrics": metrics,
        "overview": scan.overview,
        "problems": [p.model_dump() for p in scan.problems],
    }


def recommend(analysis: dict, problem: dict) -> dict:
    """Options for the coach-chosen problem, persisted to the database."""
    rec = recommend_chain.invoke({
        "team": analysis["team"],
        "coach_context": analysis["coach_context"],
        "metrics": json.dumps(analysis["metrics"], indent=2),
        "problem": json.dumps(problem, indent=2),
    }).model_dump()
    session_id = database.save_analysis(
        match_id=analysis["match_id"], team=analysis["team"],
        recommendation=rec, overview=analysis["overview"],
    )
    return {
        "session_id": session_id,
        "recommendation": rec,
        "suggestions": database.get_suggestions(session_id),
    }


def record_outcome(suggestion_id: int, applied: str, result: str) -> int:
    return database.record_outcome(suggestion_id, applied, result)


def list_sessions(limit: int = 10) -> list[dict]:
    return database.list_sessions(limit)
