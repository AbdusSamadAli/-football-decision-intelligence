import asyncio
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from mcp import Client, StdioServerParameters

import config

PROJECT_DIR = Path(__file__).resolve().parent
tactical_server = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_DIR / "tactical_mcp_server.py")],
)


class ProblemArea(BaseModel):
    """One issue the data suggests - phrased as an observation, not a verdict."""
    title: str = Field(description="Short name for the problem area.")
    description: str = Field(description="What the data suggests went wrong.")
    evidence: str = Field(description="The exact metric(s) behind this, copied verbatim.")
    severity: Literal["low", "medium", "high"]


class ProblemScan(BaseModel):
    """The assistant's read of where the team struggled."""
    overview: str = Field(description="A 2-sentence neutral summary of the match.")
    problems: list[ProblemArea] = Field(description="2-4 problem areas, most severe first.")


class TacticalOption(BaseModel):
    """ONE candidate idea - an option for the coach to consider, not an order."""
    idea: str = Field(description="The suggested adjustment, concisely.")
    rationale: str = Field(description="Why the data suggests this could help.")
    trade_offs: str = Field(description="What this costs or risks - the downside.")
    evidence: str = Field(description="The metric(s) that motivate this option.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How strongly the DATA supports this (not certainty)."
    )
    how_to_implement: str = Field(description="A concrete drill or in-game adjustment.")


class CoachingRecommendation(BaseModel):
    """Options for one chosen problem, framed for a coach who decides."""
    focus_problem: str
    options: list[TacticalOption] = Field(description="2-3 options with trade-offs.")
    assistant_note: str = Field(description=(
        "A short note reminding the coach THEY decide, what to verify on video, "
        "and what the data cannot see (morale, fitness, dressing-room)."
    ))


config.require_groq_key()
LLM = ChatGroq(model=config.GROQ_MODEL, temperature=config.LLM_TEMPERATURE)

ASSISTANT_FRAMING = (
    "You are an ASSISTANT to a football coach - a problem-solver, not a "
    "replacement for their judgement. You surface what the data shows and offer "
    "OPTIONS; the coach decides. Use ONLY the provided metrics; never invent a "
    "statistic. Cite the exact figure behind every claim. Be humble about what "
    "numbers cannot capture (morale, fitness, opponent specifics, dressing-room)."
)

scan_chain = (
    ChatPromptTemplate.from_messages([
        ("system", ASSISTANT_FRAMING),
        ("human",
         "Analyse the team '{team}' in this match.\n"
         "Coach context (honour these constraints): {coach_context}\n\n"
         "Deterministic tactical metrics (JSON):\n{metrics}\n\n"
         "Identify where {team} struggled. Ground every problem in the metrics."),
    ])
    | LLM.with_structured_output(ProblemScan)
)

recommend_chain = (
    ChatPromptTemplate.from_messages([
        ("system", ASSISTANT_FRAMING),
        ("human",
         "Team: {team}. Coach context: {coach_context}\n\n"
         "Metrics (JSON):\n{metrics}\n\n"
         "The coach chose to focus on this problem:\n{problem}\n\n"
         "Offer 2-3 tactical OPTIONS to improve it. For each: rationale, "
         "trade-offs, the evidence metric, a confidence level, and a concrete "
         "drill or in-game adjustment. End with your assistant_note."),
    ])
    | LLM.with_structured_output(CoachingRecommendation)
)


class CoachingState(TypedDict):
    match_id: int
    team: str
    coach_context: str
    metrics: dict
    problem_scan: dict
    chosen_problem: dict
    recommendation: dict


async def gather_metrics(state: CoachingState) -> dict:
    """Pull deterministic tactical metrics through the tactical MCP server."""
    print("[1/3] Gathering tactical metrics via MCP...")
    async with Client(tactical_server) as client:
        res = await client.call_tool("get_match_metrics", {"match_id": state["match_id"]})
    metrics = (
        res.structured_content
        if res.structured_content is not None
        else json.loads(res.content[0].text)
    )
    return {"metrics": metrics}


async def scan_problems(state: CoachingState) -> dict:
    """LLM identifies problem areas, grounded in the metrics."""
    print("[2/3] Scanning for problem areas (LLM)...")
    scan = await scan_chain.ainvoke({
        "team": state["team"],
        "coach_context": state["coach_context"],
        "metrics": json.dumps(state["metrics"], indent=2),
    })
    return {"problem_scan": scan.model_dump()}


def choose_focus(state: CoachingState) -> dict:
    """HUMAN-IN-THE-LOOP: pause and let the coach pick what to focus on."""
    problems = state["problem_scan"]["problems"]
    selection = interrupt({
        "question": "Which problem should I analyse in depth? Reply with its index.",
        "problems": [
            {"index": i, "title": p["title"], "severity": p["severity"]}
            for i, p in enumerate(problems)
        ],
    })
    idx = max(0, min(int(selection), len(problems) - 1))
    return {"chosen_problem": problems[idx]}


async def recommend(state: CoachingState) -> dict:
    """LLM proposes options for the coach-chosen problem."""
    print("[3/3] Generating tactical options (LLM)...")
    rec = await recommend_chain.ainvoke({
        "team": state["team"],
        "coach_context": state["coach_context"],
        "metrics": json.dumps(state["metrics"], indent=2),
        "problem": json.dumps(state["chosen_problem"], indent=2),
    })
    return {"recommendation": rec.model_dump()}


builder = StateGraph(CoachingState)
builder.add_node("gather_metrics", gather_metrics)
builder.add_node("scan_problems", scan_problems)
builder.add_node("choose_focus", choose_focus)
builder.add_node("recommend", recommend)

builder.add_edge(START, "gather_metrics")
builder.add_edge("gather_metrics", "scan_problems")
builder.add_edge("scan_problems", "choose_focus")
builder.add_edge("choose_focus", "recommend")
builder.add_edge("recommend", END)


coaching_graph = builder.compile(checkpointer=MemorySaver())


SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def render_recommendation(scan: dict, rec: dict) -> str:
    lines = ["\n" + "=" * 68, "COACHING ASSISTANT", "=" * 68,
             f"\nMatch read: {scan['overview']}\n", "FOCUS: " + rec["focus_problem"],
             "-" * 68]
    for i, opt in enumerate(rec["options"], 1):
        lines += [
            f"\nOption {i}: {opt['idea']}  [confidence: {opt['confidence']}]",
            f"  Why       : {opt['rationale']}",
            f"  Trade-offs: {opt['trade_offs']}",
            f"  Evidence  : {opt['evidence']}",
            f"  Implement : {opt['how_to_implement']}",
        ]
    lines += ["\n" + "-" * 68, f"Assistant note: {rec['assistant_note']}", "=" * 68]
    return "\n".join(lines)


async def main():
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else 22912
    team = sys.argv[2] if len(sys.argv) > 2 else "Tottenham Hotspur"

    initial: CoachingState = {
        "match_id": match_id,
        "team": team,
        "coach_context": (
            "Possession-based philosophy; developing young squad; want to turn "
            "territory and chances into goals without over-committing."
        ),
        "metrics": {}, "problem_scan": {}, "chosen_problem": {}, "recommendation": {},
    }
    cfg = {"configurable": {"thread_id": f"session-{match_id}-{team}"}}

    print(f"Analysing {team} in match {match_id}...\n")
    result = await coaching_graph.ainvoke(initial, cfg)


    payload = result["__interrupt__"][0].value
    print("\n--- COACH DECISION POINT (human-in-the-loop) ---")
    print(payload["question"])
    for p in payload["problems"]:
        print(f"  [{p['index']}] ({p['severity']}) {p['title']}")


    scan = result["problem_scan"]
    choice = min(
        range(len(scan["problems"])),
        key=lambda i: SEVERITY_RANK.get(scan["problems"][i]["severity"], 9),
    )
    print(f"(Simulating coach selection -> [{choice}]. In Streamlit this is a click.)")


    result = await coaching_graph.ainvoke(Command(resume=choice), cfg)
    print(render_recommendation(result["problem_scan"], result["recommendation"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
