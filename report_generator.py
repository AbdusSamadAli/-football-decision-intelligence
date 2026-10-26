import json

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import config


class Finding(BaseModel):
    """One analytical claim that must be backed by a number from the data."""

    claim: str = Field(
        description="A specific, factual insight about the teams or matches."
    )
    evidence: str = Field(
        description=(
            "The exact figure(s) from the provided data that support the "
            "claim - e.g. 'Arsenal: 21 pts, +11 GD (table position 2)'. "
            "Copy numbers verbatim; never invent them."
        )
    )


class MatchAnalysisReport(BaseModel):
    """A structured, evidence-grounded football analysis report."""

    headline: str = Field(description="A short, punchy headline for the report.")
    summary: str = Field(
        description="A 2-3 sentence overview grounded ONLY in the provided data."
    )
    key_findings: list[Finding] = Field(
        description="3 to 5 findings, each with supporting evidence from the data."
    )
    standout_team: str = Field(
        description=(
            "The best-performing team in the provided data, ranked by points "
            "then goal difference."
        )
    )
    verdict: str = Field(description="A single-sentence closing verdict.")


SYSTEM_PROMPT = (
    "You are a precise football analyst. You will be given PRE-COMPUTED, "
    "deterministic statistics. Write your report using ONLY those numbers. "
    "Never invent scores, teams, standings, or statistics that are not "
    "present in the data. Every finding must cite the exact figure it is "
    "based on. If something is not in the data, do not mention it."
)

HUMAN_PROMPT = (
    "Competition: {competition}\n\n"
    "Deterministic statistics (JSON):\n{analysis}\n\n"
    "Matches considered (JSON):\n{matches}\n\n"
    "Produce the structured analysis report."
)


def build_report_chain():
    """Build the LCEL chain: ChatPromptTemplate | ChatGroq(structured)."""
    config.require_groq_key()

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
    )


    structured_llm = llm.with_structured_output(MatchAnalysisReport)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )


    return prompt | structured_llm


def generate_report_llm(competition, analysis, matches) -> MatchAnalysisReport:
    """Turn deterministic stats into a validated, evidence-grounded report."""
    chain = build_report_chain()
    return chain.invoke(
        {
            "competition": competition.get("name", "Unknown"),
            "analysis": json.dumps(analysis, indent=2),
            "matches": json.dumps(matches, indent=2),
        }
    )


def format_report(report: MatchAnalysisReport) -> str:
    """Render the structured report as readable text."""
    lines = [
        "=" * 60,
        report.headline.upper(),
        "=" * 60,
        "",
        report.summary,
        "",
        "KEY FINDINGS",
        "-" * 60,
    ]
    for i, finding in enumerate(report.key_findings, 1):
        lines.append(f"{i}. {finding.claim}")
        lines.append(f"   Evidence: {finding.evidence}")
    lines.extend([
        "",
        f"Standout team: {report.standout_team}",
        f"Verdict: {report.verdict}",
        "=" * 60,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


    demo_competition = {"name": "Premier League"}
    demo_analysis = {
        "matches_analysed": 3,
        "total_goals": 8,
        "average_goals_per_match": 2.67,
        "ranked_teams": [
            {"team": "Arsenal FC", "played": 1, "wins": 1, "draws": 0,
             "losses": 0, "goals_for": 3, "goals_against": 0,
             "points": 3, "goal_difference": 3},
            {"team": "Everton FC", "played": 1, "wins": 1, "draws": 0,
             "losses": 0, "goals_for": 2, "goals_against": 0,
             "points": 3, "goal_difference": 2},
            {"team": "Coventry City FC", "played": 1, "wins": 0, "draws": 0,
             "losses": 1, "goals_for": 0, "goals_against": 3,
             "points": 0, "goal_difference": -3},
        ],
    }
    demo_matches = [
        {"home_team": "Arsenal FC", "away_team": "Coventry City FC",
         "home_score": 3, "away_score": 0, "match_id": 1},
        {"home_team": "Everton FC", "away_team": "Crystal Palace FC",
         "home_score": 2, "away_score": 0, "match_id": 2},
    ]

    print(f"Generating report with Groq model: {config.GROQ_MODEL}\n")
    report = generate_report_llm(demo_competition, demo_analysis, demo_matches)
    print(format_report(report))
