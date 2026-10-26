# Football Decision Intelligence Agent

An agentic **coaching decision-support assistant**. A coach asks a natural-language
question about a match; the system retrieves **real event-level data through MCP
tools**, computes **deterministic tactical metrics** in Python, and uses an **LLM
(LangChain + Groq)** to surface problems and propose **tactical options** — each
with trade-offs, evidence, and a confidence level. It is an **assistant, not a
replacement**: the coach stays in control via a human-in-the-loop step.

> **Guiding rule:** the LLM reasons, explains, and proposes options — it never
> invents a number, and it never replaces the coach's judgement.

**Stack:** Python · MCP (SDK v2) · LangGraph · LangChain · Groq · FastAPI ·
Streamlit · SQLAlchemy/MySQL · StatsBomb Open Data + football-data.org

---

## Architecture

```
Streamlit UI ─HTTP─▶ FastAPI /ask ─▶ LangGraph agent ─▶ MCP servers ─▶ data APIs
     │                                     │                 │
 coach chooses focus (human-in-the-loop)   │        tactical + results servers
                                           ▼
            Deterministic metrics (Python)  +  LLM reasoning (Groq)
                                           ▼
              MySQL/SQLite (sessions, suggestions, outcomes = feedback loop)
```

**Two-layer principle:** deterministic Python computes all numbers (correct,
repeatable, testable); the LLM only interprets and proposes, citing the exact
figures. This is what makes the assistant trustworthy rather than a chatbot.

---

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file in this folder with your keys:

```
FOOTBALL_DATA_API_KEY=your_football_data_key
GROQ_API_KEY=your_groq_key
```

## Run

```bash
# Explore the MCP layer
python football_mcp_client.py       # tools / resources / prompts
python tactical_metrics.py          # deterministic tactical metrics for a match

# The agents
python football_agent.py "Compare the top two teams in La Liga"   # ReAct agent
python coaching_assistant.py        # coaching assistant with human-in-the-loop

# Quality
python evaluation.py --with-llm     # repeatable eval suite + quality metrics

# The app (single process - no backend needed)
streamlit run streamlit_app.py

# Optional: the same assistant as an HTTP API (full MCP + LangGraph interrupt path)
uvicorn api:app --reload
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. On share.streamlit.io: **New app** → select the repo → main file `streamlit_app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_key"
   ```
4. Deploy. Locally, the same keys come from `.env` (or `.streamlit/secrets.toml`, which is git-ignored).

The hosted app runs the deterministic metrics engine and the LLM chains in-process
(`engine.py`) so it works as a single Streamlit process. The full MCP-server +
LangGraph human-in-the-loop orchestration lives in `coaching_assistant.py` and `api.py`.
Note: on Community Cloud the SQLite database and event cache are reset on restart.

---

## Files

| File | Role |
|------|------|
| `football_data.py` | football-data.org wrapper (results/standings) — validation, retries |
| `statsbomb_data.py` | StatsBomb Open Data (event-level) + disk cache |
| `tactical_metrics.py` | **Deterministic** engine: xG, PPDA, possession, field tilt, losses-by-third |
| `football_mcp_server.py` | MCP server — results tools + resources + prompts |
| `tactical_mcp_server.py` | MCP server — tactical metrics tools |
| `engine.py` | In-process engine used by the Streamlit app (metrics + LLM chains + persistence) |
| `report_generator.py` | LangChain + Groq structured report (Pydantic) |
| `football_agent.py` | LangGraph ReAct agent over MCP tools |
| `coaching_assistant.py` | LangGraph coaching workflow with human-in-the-loop |
| `api.py` | FastAPI backend |
| `streamlit_app.py` | Streamlit UI |
| `database.py` | SQLAlchemy persistence (SQLite default, MySQL-ready) |
| `evaluation.py` | Repeatable evaluation suite with quality metrics |
| `config.py` | Central settings |

---

## Data attribution

Event data © **StatsBomb** — [Open Data](https://github.com/statsbomb/open-data),
free for public, non-commercial use **with attribution**. Results/standings via
[football-data.org](https://www.football-data.org/).

## Build phases

0. Hygiene · 1. Deepen MCP (tools + resources + prompts) · 2. Structured LLM
reports · 3. Agentic ReAct loop · 4. Tactical data + deterministic metrics ·
5. Coaching assistant + human-in-the-loop · 6. FastAPI + Streamlit + persistence
+ evaluation. **All phases complete.**
