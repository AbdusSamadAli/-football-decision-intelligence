import json

from mcp.server import MCPServer

from football_data import (
    SUPPORTED_COMPETITIONS,
    get_competition as fetch_competition,
    get_matches as fetch_matches,
    get_match_details as fetch_match_details,
    get_standings as fetch_standings,
    get_scorers as fetch_scorers,
)

server = MCPServer(
    "Football Data Server",
    instructions=(
        "Provides read-only access to football-data.org: competitions, "
        "matches, standings and top scorers. All numbers come straight from "
        "the API - never invent statistics."
    ),
)


@server.tool()
def get_competition(code: str = "PL") -> dict:
    """Get competition metadata by code (e.g. PL, PD, SA, BL1)."""
    return fetch_competition(code)


@server.tool()
def get_matches(
    competition_code: str = "PL",
    status: str = "FINISHED",
    limit: int = 10,
) -> list:
    """Get recent matches for a competition. status is one of SCHEDULED,
    FINISHED, IN_PLAY, etc. limit is 1-20."""
    return fetch_matches(
        competition_code=competition_code,
        status=status,
        limit=limit,
    )


@server.tool()
def get_match_details(match_id: int) -> dict:
    """Get details and the final score for a single match by its ID."""
    if match_id <= 0:
        raise ValueError("match_id must be positive")
    return fetch_match_details(match_id)


@server.tool()
def get_standings(competition_code: str = "PL") -> dict:
    """Get the full league table (position, points, W/D/L, goals) for a
    competition. Use this to compare where teams sit in the table."""
    return fetch_standings(competition_code)


@server.tool()
def get_scorers(competition_code: str = "PL", limit: int = 10) -> list:
    """Get the top goal scorers for a competition. limit is 1-50."""
    return fetch_scorers(competition_code=competition_code, limit=limit)


@server.resource(
    "football://competitions",
    name="Supported competitions",
    description="The competitions available on the football-data.org free tier.",
    mime_type="application/json",
)
def list_competitions() -> str:
    """A static resource: the catalogue of competition codes a client can use.

    A client can READ this to learn valid `competition_code` values before
    calling any tool - no API call, no arguments.
    """
    return json.dumps(
        [
            {"code": code, "name": name}
            for code, name in SUPPORTED_COMPETITIONS.items()
        ],
        indent=2,
    )


@server.resource(
    "football://competition/{code}",
    name="Competition metadata",
    description="Live metadata for one competition, by code.",
    mime_type="application/json",
)
def competition_resource(code: str) -> str:
    """A resource *template*: the `{code}` in the URI becomes a parameter.

    Reading `football://competition/PL` returns metadata for the Premier
    League. This is the RESOURCE (read-by-URI) counterpart to the
    get_competition TOOL - same data, different MCP primitive, so you can
    see the distinction.
    """
    return json.dumps(fetch_competition(code), indent=2)


@server.prompt(
    name="compare_teams",
    description="Template for a head-to-head comparison of two teams.",
)
def compare_teams_prompt(
    team_a: str,
    team_b: str,
    competition: str = "PL",
) -> str:
    """Returns a filled prompt the client/LLM can use as its instruction.

    Prompts let the SERVER own the wording of a task, so every client asks
    the question the same, well-engineered way.
    """
    return (
        f"You are a football analyst. Compare {team_a} and {team_b} in the "
        f"{competition} competition.\n\n"
        "Use ONLY data retrieved from the football tools (standings, recent "
        "matches, top scorers). For every claim, cite the exact numbers and "
        "the match or table position they came from. If a stat is "
        "unavailable, say so rather than guessing. End with a clear, "
        "evidence-based verdict on which team is in better form."
    )


@server.prompt(
    name="match_report",
    description="Template for a concise, evidence-grounded match report.",
)
def match_report_prompt(home_team: str, away_team: str) -> str:
    """Prompt template for writing a single-match report."""
    return (
        f"Write a concise, factual match report for {home_team} vs "
        f"{away_team}. State the final score first, then the key facts. "
        "Use only the retrieved match data. Do not invent events, scorers, "
        "or statistics that are not in the data."
    )


if __name__ == "__main__":
    server.run()
