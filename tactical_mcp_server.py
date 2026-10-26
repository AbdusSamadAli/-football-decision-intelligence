import json

from mcp.server import MCPServer

import statsbomb_data
import tactical_metrics

server = MCPServer(
    "Football Tactical Server",
    instructions=(
        "Provides deterministic, event-level tactical metrics from StatsBomb "
        "open data (xG, PPDA pressing intensity, possession, field tilt, ball "
        "losses by pitch third, final-third entries). All numbers are computed "
        "in Python - never invent tactical statistics. Data (c) StatsBomb, free "
        "for public non-commercial use with attribution."
    ),
)


@server.tool()
def list_competitions() -> list:
    """List the free competitions/seasons available in StatsBomb open data."""
    return statsbomb_data.list_competitions()


@server.tool()
def list_matches(competition_id: int, season_id: int) -> list:
    """List matches for a StatsBomb competition_id + season_id (get these from
    list_competitions)."""
    return statsbomb_data.list_matches(competition_id, season_id)


@server.tool()
def get_match_metrics(match_id: int) -> dict:
    """Get the full deterministic tactical metrics for a match: per-team goals,
    xG, shots, passes, pass completion, possession, field tilt, PPDA (pressing),
    ball losses by pitch third, and final-third entries. This is the core
    tactical-analysis tool."""
    return tactical_metrics.compute_match_metrics(match_id)


@server.resource(
    "statsbomb://curated-matches",
    name="Curated example matches",
    description="Hand-picked well-known matches in the free data.",
    mime_type="application/json",
)
def curated_matches() -> str:
    return json.dumps(statsbomb_data.CURATED_MATCHES, indent=2)


if __name__ == "__main__":
    server.run()
