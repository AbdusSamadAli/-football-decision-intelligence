import pandas as pd

import statsbomb_data

DEF_THIRD = 40.0
ATT_THIRD = 80.0


PPDA_DEF_ACTIONS = {"Interception", "Duel", "Foul Committed", "Block"}


def _x(loc):
    """Safely pull the x coordinate from a StatsBomb [x, y] location."""
    if isinstance(loc, (list, tuple)) and loc and loc[0] is not None:
        return float(loc[0])
    return None


def _third(x) -> str | None:
    if x is None:
        return None
    if x < DEF_THIRD:
        return "defensive"
    if x < ATT_THIRD:
        return "middle"
    return "attacking"


def compute_match_metrics(match_id: int) -> dict:
    """Compute per-team tactical metrics for one match."""
    ev = statsbomb_data.get_events(match_id)


    ev = ev.copy()
    ev["x"] = ev["location"].apply(_x)

    teams = [t for t in ev["team"].dropna().unique().tolist()]
    if len(teams) != 2:
        raise RuntimeError(f"Expected 2 teams, found {teams}")

    metrics = {team: _team_metrics(ev, team, teams) for team in teams}
    score = {team: metrics[team]["goals"] for team in teams}

    return {
        "match_id": match_id,
        "teams": teams,
        "score": score,
        "metrics": metrics,
        "notes": (
            "All metrics computed deterministically from StatsBomb event data. "
            "PPDA and possession are standard approximations (see code). "
            "Data (c) StatsBomb."
        ),
    }


def _team_metrics(ev: pd.DataFrame, team: str, teams: list) -> dict:
    opponent = teams[1] if teams[0] == team else teams[0]

    team_ev = ev[ev["team"] == team]
    shots = team_ev[team_ev["type"] == "Shot"]
    passes = team_ev[team_ev["type"] == "Pass"]


    goals = int((shots["shot_outcome"] == "Goal").sum())
    if "Own Goal For" in set(ev["type"].unique()):
        goals += int(((ev["type"] == "Own Goal For") & (ev["team"] == team)).sum())


    xg = float(round(shots["shot_statsbomb_xg"].fillna(0).sum(), 3))
    shots_on_target = int(shots["shot_outcome"].isin(["Goal", "Saved"]).sum())


    n_passes = int(len(passes))
    completed = int(passes["pass_outcome"].isna().sum())
    pass_completion_pct = round(100 * completed / n_passes, 1) if n_passes else 0.0


    total_passes = int((ev["type"] == "Pass").sum())
    possession_pct = round(100 * n_passes / total_passes, 1) if total_passes else 0.0


    losses = team_ev[
        (team_ev["type"].isin(["Dispossessed", "Miscontrol"]))
        | ((team_ev["type"] == "Pass") & (team_ev["pass_outcome"] == "Incomplete"))
    ]
    losses_by_third = {"defensive": 0, "middle": 0, "attacking": 0}
    for x in losses["x"]:
        third = _third(x)
        if third:
            losses_by_third[third] += 1
    ball_losses = int(len(losses))


    final_third_entries = _final_third_entries(team_ev)


    field_tilt_pct = _field_tilt(ev, team, opponent)


    ppda = _ppda(ev, team, opponent)

    return {
        "goals": goals,
        "xg": xg,
        "shots": int(len(shots)),
        "shots_on_target": shots_on_target,
        "passes": n_passes,
        "pass_completion_pct": pass_completion_pct,
        "possession_pct": possession_pct,
        "ball_losses": ball_losses,
        "losses_by_third": losses_by_third,
        "final_third_entries": final_third_entries,
        "field_tilt_pct": field_tilt_pct,
        "ppda": ppda,
    }


def _final_third_entries(team_ev: pd.DataFrame) -> int:
    count = 0
    passes = team_ev[team_ev["type"] == "Pass"]
    for start, end in zip(passes["x"], passes["pass_end_location"]):
        end_x = _x(end)
        if start is not None and end_x is not None and start < ATT_THIRD <= end_x:
            count += 1
    if "carry_end_location" in team_ev.columns:
        carries = team_ev[team_ev["type"] == "Carry"]
        for start, end in zip(carries["x"], carries["carry_end_location"]):
            end_x = _x(end)
            if start is not None and end_x is not None and start < ATT_THIRD <= end_x:
                count += 1
    return count


def _field_tilt(ev: pd.DataFrame, team: str, opponent: str) -> float:
    action_types = ["Pass", "Carry", "Shot"]
    in_final = ev[(ev["type"].isin(action_types)) & (ev["x"] >= ATT_THIRD)]
    team_ft = int((in_final["team"] == team).sum())
    opp_ft = int((in_final["team"] == opponent).sum())
    total = team_ft + opp_ft
    return round(100 * team_ft / total, 1) if total else 0.0


def _ppda(ev: pd.DataFrame, team: str, opponent: str) -> float | None:

    opp_passes = ev[
        (ev["team"] == opponent) & (ev["type"] == "Pass") & (ev["x"] <= ATT_THIRD)
    ]

    def_actions = ev[
        (ev["team"] == team)
        & (ev["type"].isin(PPDA_DEF_ACTIONS))
        & (ev["x"] >= DEF_THIRD)
    ]
    n_actions = len(def_actions)
    if n_actions == 0:
        return None
    return round(len(opp_passes) / n_actions, 2)


def format_metrics(result: dict) -> str:
    """Render the metrics as a readable comparison table."""
    teams = result["teams"]
    a, b = teams
    ma, mb = result["metrics"][a], result["metrics"][b]

    def row(label, ka, kb):
        return f"{label:<24}{str(ka):>22}{str(kb):>22}"

    lines = [
        "=" * 68,
        f"TACTICAL METRICS  (match {result['match_id']})",
        "=" * 68,
        row("", a[:21], b[:21]),
        "-" * 68,
        row("Goals", ma["goals"], mb["goals"]),
        row("xG", ma["xg"], mb["xg"]),
        row("Shots (on target)", f"{ma['shots']} ({ma['shots_on_target']})",
            f"{mb['shots']} ({mb['shots_on_target']})"),
        row("Passes", ma["passes"], mb["passes"]),
        row("Pass completion %", ma["pass_completion_pct"], mb["pass_completion_pct"]),
        row("Possession %", ma["possession_pct"], mb["possession_pct"]),
        row("Field tilt %", ma["field_tilt_pct"], mb["field_tilt_pct"]),
        row("PPDA (lower=press)", ma["ppda"], mb["ppda"]),
        row("Ball losses", ma["ball_losses"], mb["ball_losses"]),
        row("  in defensive third", ma["losses_by_third"]["defensive"],
            mb["losses_by_third"]["defensive"]),
        row("  in middle third", ma["losses_by_third"]["middle"],
            mb["losses_by_third"]["middle"]),
        row("  in attacking third", ma["losses_by_third"]["attacking"],
            mb["losses_by_third"]["attacking"]),
        row("Final-third entries", ma["final_third_entries"], mb["final_third_entries"]),
        "=" * 68,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    demo = statsbomb_data.CURATED_MATCHES["ucl_2019_final"]
    print(f"Computing metrics for: {demo['label']}\n")
    result = compute_match_metrics(demo["match_id"])
    print(format_metrics(result))
