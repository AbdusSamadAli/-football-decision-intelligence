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
