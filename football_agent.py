import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from mcp import Client, StdioServerParameters
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import config

PROJECT_DIR = Path(__file__).resolve().parent
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_DIR / "football_mcp_server.py")],
)


async def _mcp_call(name: str, arguments: dict) -> str:
    """Open a short-lived MCP session, call one tool, return its text result."""
    async with Client(server_params) as client:
        result = await client.call_tool(name, arguments)

    if result.is_error:
        return f"ERROR: MCP tool '{name}' failed."

    if getattr(result, "structured_content", None) is not None:
        return json.dumps(result.structured_content)

    texts = [
        item.text
        for item in (result.content or [])
        if hasattr(item, "text")
    ]
    return "\n".join(texts) if texts else "No data returned."


@tool
async def get_standings(competition_code: str = "PL") -> str:
    """Get the league table (position, points, W/D/L, goals for/against, goal
    difference) for a competition. Use for form/ranking comparisons.
    Codes: PL, PD, SA, BL1, FL1, DED, PPL, ELC, CL."""
    return await _mcp_call("get_standings", {"competition_code": competition_code})


@tool
async def get_matches(
    competition_code: str = "PL",
    status: str = "FINISHED",
    limit: int = 10,
) -> str:
    """Get recent matches for a competition. status is FINISHED or SCHEDULED;
    limit is 1-20. Use when the question is about specific results/fixtures."""
    return await _mcp_call(
        "get_matches",
        {"competition_code": competition_code, "status": status, "limit": limit},
    )


@tool
async def get_scorers(competition_code: str = "PL", limit: int = 10) -> str:
    """Get the top goal scorers (player, team, goals) for a competition.
    limit is 1-50. Use for questions about leading scorers."""
    return await _mcp_call(
        "get_scorers", {"competition_code": competition_code, "limit": limit}
    )


@tool
async def get_competition(code: str = "PL") -> str:
    """Get metadata (name, country) for a competition code."""
    return await _mcp_call("get_competition", {"code": code})


TOOLS = [get_standings, get_matches, get_scorers, get_competition]


config.require_groq_key()
LLM = ChatGroq(model=config.GROQ_MODEL, temperature=config.LLM_TEMPERATURE)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)

SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a football analyst agent. Answer the user's question by calling "
    "tools to fetch REAL data, then reasoning over it.\n\n"
    "Competition codes: PL=Premier League (England), PD=La Liga (Spain), "
    "SA=Serie A (Italy), BL1=Bundesliga (Germany), FL1=Ligue 1 (France), "
    "ELC=Championship (England), DED=Eredivisie, PPL=Primeira Liga, "
    "CL=UEFA Champions League.\n\n"
    "Rules:\n"
    "- Decide which tool(s) to call; you may call several across turns.\n"
    "- Use ONLY numbers returned by tools. NEVER invent scores, points, "
    "positions, or statistics.\n"
    "- Cite the exact figures (points, goal difference, table position) that "
    "support each statement.\n"
    "- If the data needed is unavailable, say so plainly.\n"
    "- Keep the final answer concise and well structured."
))


class AgentState(TypedDict):

    messages: Annotated[list, add_messages]


async def agent_node(state: AgentState) -> dict:
    """Call the LLM. It either answers or asks to call tools."""
    response = await LLM_WITH_TOOLS.ainvoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")


builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

agent_graph = builder.compile()


async def ask(question: str, show_trace: bool = True) -> str:
    """Run the agent on a question and return its final answer."""
    final_state = await agent_graph.ainvoke(
        {"messages": [HumanMessage(content=question)]}
    )

    if show_trace:
        print("\n--- AGENT TRACE ---")
        for msg in final_state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"  [LLM decided] call {call['name']}({call['args']})")
            elif msg.__class__.__name__ == "ToolMessage":
                preview = msg.content.replace("\n", " ")[:90]
                print(f"  [tool result] {preview}...")
        print("--- END TRACE ---\n")

    return final_state["messages"][-1].content


async def main():
    questions = [
        "How are Arsenal doing in the Premier League, and how do they "
        "compare to Manchester City? Who is higher in the table?",
    ]
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]

    for question in questions:
        print("=" * 70)
        print(f"Q: {question}")
        print("=" * 70)
        answer = await ask(question)
        print("ANSWER:\n")
        print(answer)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
