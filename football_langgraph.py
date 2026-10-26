import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict, Any

from mcp import Client, StdioServerParameters
from langgraph.graph import StateGraph, START, END

from report_generator import generate_report_llm


PROJECT_DIR = Path(__file__).resolve().parent

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_DIR / "football_mcp_server.py")],
)


class FootballState(TypedDict):
    competition: dict[str, Any]
    matches: list[dict[str, Any]]
    analysis: dict[str, Any]
    report: str
    llm_report: dict[str, Any]
    error: str


def extract_json_results(result) -> list[dict[str, Any]]:
    """
    Extract JSON objects from MCP text content.

    The current MCP server returns JSON-formatted text.
    This helper also handles a JSON array or one JSON object.
    """

    if result.is_error:
        raise RuntimeError("MCP tool returned an error.")

    extracted = []


    structured = getattr(result, "structured_content", None)

    if structured is not None:
        if isinstance(structured, list):
            return structured

        if isinstance(structured, dict):
            return [structured]

    for item in result.content or []:
        if not hasattr(item, "text"):
            continue

        text = item.text.strip()

        if not text:
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            extracted.extend(
                entry for entry in data
                if isinstance(entry, dict)
            )
        elif isinstance(data, dict):
            extracted.append(data)

    return extracted


async def fetch_matches(state: FootballState) -> dict:
    """
    Connect to the MCP server and retrieve competition
    metadata and completed matches.
    """

    print("\n[NODE 1] Fetching football data through MCP...")

    try:
        async with Client(server_params) as client:


            tools_result = await client.list_tools()

            available_tools = {
                tool.name for tool in tools_result.tools
            }

            if "get_matches" not in available_tools:
                raise RuntimeError(
                    "The MCP server does not expose get_matches."
                )

            if "get_competition" not in available_tools:
                raise RuntimeError(
                    "The MCP server does not expose get_competition."
                )


            competition_result = await client.call_tool(
                "get_competition",
                {"code": "PL"},
            )

            competition_data = extract_json_results(
                competition_result
            )

            if not competition_data:
                raise RuntimeError(
                    "No competition data was returned."
                )


            matches_result = await client.call_tool(
                "get_matches",
                {
                    "competition_code": "PL",
                    "status": "FINISHED",
                    "limit": 5,
                },
            )

            matches = extract_json_results(matches_result)

            if not matches:
                raise RuntimeError(
                    "No match data was returned."
                )

            print(f"Competition: {competition_data[0].get('name')}")
            print(f"Matches retrieved: {len(matches)}")

            return {
                "competition": competition_data[0],
                "matches": matches,
                "error": "",
            }

    except Exception as exc:
        return {
            "error": f"Failed to fetch football data: {exc}"
        }


def analyse_results(state: FootballState) -> dict:
    """
    Calculate team results and goals from the retrieved matches.

    This is deterministic Python logic, not an LLM.
    """

    print("\n[NODE 2] Analysing match results...")

    if state.get("error"):
        return {}

    matches = state["matches"]

    team_stats = {}
    total_goals = 0
    processed_matches = 0

    for match in matches:

        home_team = match.get("home_team")
        away_team = match.get("away_team")
        home_score = match.get("home_score")
        away_score = match.get("away_score")


        if (
            not home_team
            or not away_team
            or not isinstance(home_score, int)
            or isinstance(home_score, bool)
            or not isinstance(away_score, int)
            or isinstance(away_score, bool)
            or home_score < 0
            or away_score < 0
        ):
            continue

        processed_matches += 1
        total_goals += home_score + away_score

        for team in (home_team, away_team):
            if team not in team_stats:
                team_stats[team] = {
                    "played": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "points": 0,
                }


        team_stats[home_team]["played"] += 1
        team_stats[away_team]["played"] += 1

        team_stats[home_team]["goals_for"] += home_score
        team_stats[home_team]["goals_against"] += away_score

        team_stats[away_team]["goals_for"] += away_score
        team_stats[away_team]["goals_against"] += home_score


        if home_score > away_score:
            team_stats[home_team]["wins"] += 1
            team_stats[home_team]["points"] += 3
            team_stats[away_team]["losses"] += 1

        elif away_score > home_score:
            team_stats[away_team]["wins"] += 1
            team_stats[away_team]["points"] += 3
            team_stats[home_team]["losses"] += 1

        else:
            team_stats[home_team]["draws"] += 1
            team_stats[away_team]["draws"] += 1

            team_stats[home_team]["points"] += 1
            team_stats[away_team]["points"] += 1

    if processed_matches == 0:
        return {
            "error": (
                "No matches contained valid completed-match scores."
            )
        }


    ranked_teams = sorted(
        team_stats.items(),
        key=lambda item: (
            item[1]["points"],
            item[1]["goals_for"] - item[1]["goals_against"],
            item[1]["goals_for"],
        ),
        reverse=True,
    )

    analysis = {
        "matches_analysed": processed_matches,
        "total_goals": total_goals,
        "average_goals_per_match": round(
            total_goals / processed_matches, 2
        ),
        "teams": team_stats,
        "ranked_teams": [
            {
                "team": team,
                **stats,
                "goal_difference": (
                    stats["goals_for"] - stats["goals_against"]
                ),
            }
            for team, stats in ranked_teams
        ],
    }

    print(f"Matches analysed: {processed_matches}")
    print(f"Total goals: {total_goals}")

    return {"analysis": analysis}


def generate_report(state: FootballState) -> dict:
    """
    Convert the deterministic statistics into a readable report.
    """

    print("\n[NODE 3] Generating report...")

    if state.get("error"):
        return {"report": state["error"]}

    competition = state["competition"]
    analysis = state["analysis"]

    lines = [
        "=" * 60,
        f"FOOTBALL MATCH ANALYSIS: {competition.get('name', 'Unknown')}",
        "=" * 60,
        "",
        f"Matches analysed: {analysis['matches_analysed']}",
        f"Total goals: {analysis['total_goals']}",
        (
            "Average goals per match: "
            f"{analysis['average_goals_per_match']}"
        ),
        "",
        "MATCH RESULTS",
        "-" * 60,
    ]

    for match in state["matches"]:
        home = match.get("home_team", "Unknown")
        away = match.get("away_team", "Unknown")
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if home_score is None or away_score is None:
            score = "Score unavailable"
        else:
            score = f"{home_score} - {away_score}"

        lines.append(f"{home} {score} {away}")

    lines.extend([
        "",
        "TEAM PERFORMANCE",
        "-" * 60,
        (
            f"{'Team':30} {'P':>2} {'W':>2} {'D':>2} "
            f"{'L':>2} {'GF':>3} {'GA':>3} {'Pts':>3}"
        ),
    ])

    for team in analysis["ranked_teams"]:
        lines.append(
            f"{team['team'][:30]:30} "
            f"{team['played']:>2} "
            f"{team['wins']:>2} "
            f"{team['draws']:>2} "
            f"{team['losses']:>2} "
            f"{team['goals_for']:>3} "
            f"{team['goals_against']:>3} "
            f"{team['points']:>3}"
        )

    lines.extend([
        "",
        "NOTES",
        "-" * 60,
        (
            "This report analyses only the matches returned by the "
            "MCP server. It is not a full Premier League table."
        ),
        (
            "The ranking is based on the selected matches only, "
            "using points, goal difference, and goals scored."
        ),
        "",
        "=" * 60,
    ])

    report = "\n".join(lines)

    print("Report generated successfully.")

    return {"report": report}


def generate_llm_report(state: FootballState) -> dict:
    """
    Turn the deterministic stats into a structured, evidence-grounded
    report using the LLM. The LLM writes prose only - all numbers come
    from NODE 2. A failure here does NOT kill the run: the deterministic
    report from NODE 3 still stands.
    """

    print("\n[NODE 4] Generating structured LLM report (Groq)...")

    if state.get("error"):
        return {}

    try:
        report = generate_report_llm(
            state["competition"],
            state["analysis"],
            state["matches"],
        )
        print(f"LLM report ready: {report.headline}")
        return {"llm_report": report.model_dump()}
    except Exception as exc:
        print(f"LLM report step failed (continuing): {exc}")
        return {"llm_report": {"error": str(exc)}}


builder = StateGraph(FootballState)

builder.add_node("fetch_matches", fetch_matches)
builder.add_node("analyse_results", analyse_results)
builder.add_node("generate_report", generate_report)
builder.add_node("generate_llm_report", generate_llm_report)

builder.add_edge(START, "fetch_matches")
builder.add_edge("fetch_matches", "analyse_results")
builder.add_edge("analyse_results", "generate_report")
builder.add_edge("generate_report", "generate_llm_report")
builder.add_edge("generate_llm_report", END)

football_graph = builder.compile()


async def main():

    initial_state: FootballState = {
        "competition": {},
        "matches": [],
        "analysis": {},
        "report": "",
        "llm_report": {},
        "error": "",
    }

    print("\nStarting Football Decision Intelligence Agent...")

    final_state = await football_graph.ainvoke(initial_state)

    print("\n")

    if final_state.get("error"):
        print("WORKFLOW ERROR")
        print(final_state["error"])
        return


    print(final_state["report"])


    llm_report = final_state.get("llm_report") or {}
    if llm_report and "error" not in llm_report:
        from report_generator import MatchAnalysisReport, format_report
        print("\n")
        print(format_report(MatchAnalysisReport.model_validate(llm_report)))
    elif llm_report.get("error"):
        print(f"\n(LLM report unavailable: {llm_report['error']})")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
