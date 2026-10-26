import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Football Decision Intelligence", page_icon="⚽", layout="wide")

load_dotenv(Path(__file__).resolve().parent / ".env")


def _load_streamlit_secrets() -> str | None:
    """Copy Streamlit secrets into the environment; return an error type if parsing failed."""
    try:
        items = dict(st.secrets)
    except Exception as exc:
        return type(exc).__name__
    for key, value in items.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                os.environ.setdefault(sub_key, str(sub_value))
        else:
            os.environ.setdefault(key, str(value))
    return None


secrets_error = _load_streamlit_secrets()

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "On Streamlit Cloud: **Manage app → Settings → Secrets**, paste exactly this "
        "(TOML format — the quotes are required, no section header):\n\n"
        '```toml\nGROQ_API_KEY = "gsk_your_key_here"\n```\n\n'
        "Save, then **Reboot app**. Locally, put `GROQ_API_KEY=...` in `.env`."
        + (f"\n\n(Secrets could not be read: `{secrets_error}` — usually invalid TOML.)"
           if secrets_error else "")
    )
    st.stop()

import engine  # noqa: E402  (must load after secrets are in the environment)

st.title("⚽ Football Decision Intelligence — Coaching Assistant")
st.caption(
    "An assistant, not a replacement. It surfaces what the data shows and "
    "offers options — you decide. Data © StatsBomb (open data)."
)


@st.cache_data(ttl=3600, show_spinner=False)
def curated():
    return engine.list_curated()


@st.cache_data(ttl=3600, show_spinner=False)
def competitions():
    return engine.list_competitions()


@st.cache_data(ttl=3600, show_spinner=False)
def matches(competition_id, season_id):
    return engine.list_matches(competition_id, season_id)


with st.sidebar:
    st.header("Choose a match")
    match_id, teams = None, []
    try:
        mode = st.radio("Find a match by:", ["Featured matches", "Browse competitions"])

        if mode == "Featured matches":
            featured = curated()
            keys = list(featured.keys())
            sel = st.selectbox("Match", keys, format_func=lambda k: featured[k]["label"])
            m = featured[sel]
            match_id = m["match_id"]
            teams = [m["home_team"], m["away_team"]]
        else:
            comps = competitions()
            ci = st.selectbox("Competition / season", range(len(comps)),
                              format_func=lambda i: comps[i]["label"])
            comp = comps[ci]
            ms = matches(comp["competition_id"], comp["season_id"])
            if ms:
                mi = st.selectbox("Match", range(len(ms)),
                                  format_func=lambda i: ms[i]["label"])
                match_id = ms[mi]["match_id"]
                teams = [ms[mi]["home_team"], ms[mi]["away_team"]]
            else:
                st.caption("No matches in this competition's open data.")
    except Exception as exc:
        st.error(f"Could not load match list: {exc}")

    team = st.selectbox("Team to analyse", teams) if teams else None
    coach_context = st.text_area(
        "Your context (philosophy, squad, injuries)",
        value=("Possession-based philosophy; developing young squad; want to "
               "turn territory and chances into goals without over-committing."),
        height=120,
    )
    analyse = st.button("Analyse match", type="primary",
                        disabled=not (match_id and team))


if analyse:
    try:
        with st.spinner("Computing tactical metrics and scanning for problems..."):
            st.session_state["analysis"] = engine.analyze(match_id, team, coach_context)
        st.session_state.pop("recommendation", None)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

analysis = st.session_state.get("analysis")

if analysis:
    st.subheader(f"Match read — {analysis['team']}")
    st.info(analysis["overview"])

    metrics = analysis["metrics"]["metrics"]
    cols = st.columns(len(metrics))
    for col, (tname, m) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"**{tname}**")
            st.metric("Goals / xG", f"{m['goals']} / {m['xg']}")
            st.metric("Possession %", m["possession_pct"])
            st.metric("Shots (on target)", f"{m['shots']} ({m['shots_on_target']})")
            st.metric("PPDA (lower = press)", m["ppda"])

    st.subheader("Where should I dig in? (you choose)")
    problems = analysis["problems"]
    labels = [f"[{p['severity']}] {p['title']}" for p in problems]
    choice = st.radio("Problems the data suggests:", options=list(range(len(problems))),
                      format_func=lambda i: labels[i])
    with st.expander("See the evidence behind each problem"):
        for p in problems:
            st.markdown(f"**{p['title']}** ({p['severity']}) — {p['description']}")
            st.caption(f"Evidence: {p['evidence']}")

    if st.button("Get tactical options", type="primary"):
        try:
            with st.spinner("Generating options..."):
                st.session_state["recommendation"] = engine.recommend(
                    analysis, problems[int(choice)]
                )
        except Exception as exc:
            st.error(f"Could not get recommendations: {exc}")


rec_payload = st.session_state.get("recommendation")
if rec_payload:
    rec = rec_payload["recommendation"]
    st.subheader(f"Options for: {rec['focus_problem']}")

    db_suggestions = rec_payload["suggestions"]
    for i, opt in enumerate(rec["options"]):
        with st.expander(f"Option {i+1}: {opt['idea']}  —  confidence: {opt['confidence']}"):
            st.markdown(f"**Why:** {opt['rationale']}")
            st.markdown(f"**Trade-offs:** {opt['trade_offs']}")
            st.markdown(f"**Evidence:** `{opt['evidence']}`")
            st.markdown(f"**How to implement:** {opt['how_to_implement']}")

            if i < len(db_suggestions):
                sid = db_suggestions[i]["id"]
                with st.form(f"outcome_{sid}"):
                    st.caption("Did you try this? Record the outcome (feedback loop).")
                    applied = st.selectbox("Applied?", ["unknown", "yes", "no"], key=f"a_{sid}")
                    result = st.text_input("What happened?", key=f"r_{sid}")
                    if st.form_submit_button("Save outcome"):
                        try:
                            engine.record_outcome(sid, applied, result)
                            st.success("Outcome saved.")
                        except Exception as exc:
                            st.error(f"Save failed: {exc}")

    st.success(f"💡 Assistant note: {rec['assistant_note']}")


with st.expander("Recent sessions (history)"):
    try:
        sessions = engine.list_sessions()
        if sessions:
            st.table(sessions)
        else:
            st.caption("No sessions saved yet.")
    except Exception as exc:
        st.caption(f"History unavailable: {exc}")
