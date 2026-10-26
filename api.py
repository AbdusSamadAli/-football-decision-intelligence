from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

import database
import statsbomb_data
from coaching_assistant import coaching_graph

app = FastAPI(title="Football Decision Intelligence API", version="1.0")


@app.on_event("startup")
def _startup():
    database.init_db()


class AnalyzeRequest(BaseModel):
    match_id: int = 22912
    team: str = "Tottenham Hotspur"
    coach_context: str = (
        "Possession-based philosophy; developing young squad; want to turn "
        "territory and chances into goals without over-committing."
    )


class RecommendRequest(BaseModel):
    thread_id: str
    choice: int


class OutcomeRequest(BaseModel):
    suggestion_id: int
    applied: str = "unknown"
    result: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/curated")
def curated():
    """Featured famous matches (friendly names) so users never type an id."""
    return statsbomb_data.CURATED_MATCHES


@app.get("/competitions")
def competitions():
    """All free competitions/seasons, with a display label for a dropdown."""
    comps = statsbomb_data.list_competitions()
    for c in comps:
        c["label"] = (f"{c['competition_name']} {c['season_name']} "
                      f"({c['country_name']})")
    return {"competitions": comps}


@app.get("/matches")
def matches(competition_id: int, season_id: int):
    """Matches for a competition+season, with a 'Home x-y Away' label."""
    ms = statsbomb_data.list_matches(competition_id, season_id)
    for m in ms:
        m["label"] = (f"{m['home_team']} {m['home_score']}-{m['away_score']} "
                      f"{m['away_team']}")
    return {"matches": ms}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Run metrics + problem scan, then pause for the coach's decision."""
    thread_id = uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    initial = {
        "match_id": req.match_id, "team": req.team,
        "coach_context": req.coach_context,
        "metrics": {}, "problem_scan": {}, "chosen_problem": {}, "recommendation": {},
    }
    try:
        result = await coaching_graph.ainvoke(initial, cfg)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}")

    if "__interrupt__" not in result:
        raise HTTPException(status_code=500, detail="Expected a coach decision point")

    scan = result["problem_scan"]
    return {
        "thread_id": thread_id,
        "match_id": req.match_id,
        "team": req.team,
        "overview": scan["overview"],
        "problems": scan["problems"],
        "metrics": result["metrics"],
    }


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """Resume the paused analysis with the coach's choice; persist the result."""
    cfg = {"configurable": {"thread_id": req.thread_id}}
    try:
        result = await coaching_graph.ainvoke(Command(resume=req.choice), cfg)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"Could not resume thread (expired?): {exc}")

    rec = result.get("recommendation")
    if not rec:
        raise HTTPException(status_code=500, detail="No recommendation produced")

    session_id = database.save_analysis(
        match_id=result["match_id"], team=result["team"],
        recommendation=rec, overview=result["problem_scan"]["overview"],
    )
    return {
        "session_id": session_id,
        "recommendation": rec,
        "suggestions": database.get_suggestions(session_id),
    }


@app.post("/outcome")
def outcome(req: OutcomeRequest):
    """Record the real-world outcome of a suggestion (DI feedback loop)."""
    oid = database.record_outcome(req.suggestion_id, req.applied, req.result)
    return {"outcome_id": oid, "status": "recorded"}


@app.get("/sessions")
def sessions(limit: int = 10):
    return {"sessions": database.list_sessions(limit)}
