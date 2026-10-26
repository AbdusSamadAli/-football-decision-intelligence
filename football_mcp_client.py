import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(PROJECT_DIR / "football_mcp_server.py")],
)


def show_content(result):
    """Print the text payload of a tool/resource result."""
    if getattr(result, "structured_content", None) is not None:
        print("structured_content:", result.structured_content)
    for item in getattr(result, "content", None) or []:
        if hasattr(item, "text"):
            print(item.text)


async def main():
    print("Connecting to Football Data MCP server...\n")

    async with Client(server_params) as client:


        print("=" * 60, "\nTOOLS (actions the LLM can call)\n", "=" * 60)
        tools = await client.list_tools()
        for tool in tools.tools:
            print(f"  - {tool.name}: {tool.description.splitlines()[0]}")

        print("\n-> call_tool get_standings(PL):")
        standings = await client.call_tool("get_standings", {"competition_code": "PL"})
        show_content(standings)


        print("\n", "=" * 60, "\nRESOURCES (data read by URI)\n", "=" * 60)
        resources = await client.list_resources()
        for res in resources.resources:
            print(f"  - {res.uri}  ({res.name})")

        templates = await client.list_resource_templates()
        for tmpl in templates.resource_templates:
            print(f"  - {tmpl.uri_template}  (template: {tmpl.name})")

        print("\n-> read_resource football://competitions:")
        catalog = await client.read_resource("football://competitions")
        for item in catalog.contents:
            if hasattr(item, "text"):
                print(item.text[:300], "...")


        print("\n", "=" * 60, "\nPROMPTS (server-authored templates)\n", "=" * 60)
        prompts = await client.list_prompts()
        for prompt in prompts.prompts:
            args = ", ".join(a.name for a in (prompt.arguments or []))
            print(f"  - {prompt.name}({args}): {prompt.description}")

        print("\n-> get_prompt compare_teams(Arsenal, Chelsea):")
        filled = await client.get_prompt(
            "compare_teams",
            {"team_a": "Arsenal", "team_b": "Chelsea", "competition": "PL"},
        )
        for msg in filled.messages:
            text = getattr(msg.content, "text", msg.content)
            print(f"  [{msg.role}] {text}")

        print("\nAll client operations completed.")


if __name__ == "__main__":
    asyncio.run(main())
