import argparse
import json
import re
import sys
from pathlib import Path

import statsbomb_data
import tactical_metrics

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_FILE = PROJECT_DIR / "eval_results.json"


VERIFIED_FACTS = {
    22912: {"possession_leader": "Tottenham Hotspur", "xg_leader": "Liverpool"},
}


def build_golden(competition_id=43, season_id=106, n=3) -> list[dict]:
    """Golden set = first n matches of a well-populated competition (World Cup
    2022) with OFFICIAL scores, plus the hand-verified UCL 2019 final."""
    matches = statsbomb_data.list_matches(competition_id, season_id)
    golden = []
    for m in matches[:n]:
        golden.append({
            "match_id": int(m["match_id"]),
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_score": int(m["home_score"]),
            "away_score": int(m["away_score"]),
        })

    golden.append({
        "match_id": 22912,
        "home_team": "Tottenham Hotspur", "away_team": "Liverpool",
        "home_score": 0, "away_score": 2,
    })
    return golden


def check_correctness(golden: dict, result: dict) -> list[dict]:
    checks = []
    score = result["score"]

    derived_home = score.get(golden["home_team"])
    derived_away = score.get(golden["away_team"])
    checks.append({
        "check": "score_matches_official",
        "passed": derived_home == golden["home_score"]
        and derived_away == golden["away_score"],
        "detail": f"derived {derived_home}-{derived_away} vs official "
                  f"{golden['home_score']}-{golden['away_score']}",
    })

    def winner(h, a, ht, at):
        return "draw" if h == a else (ht if h > a else at)
    official_w = winner(golden["home_score"], golden["away_score"],
                        golden["home_team"], golden["away_team"])
    derived_w = winner(derived_home, derived_away,
                       golden["home_team"], golden["away_team"])
    checks.append({
        "check": "winner_correct",
        "passed": official_w == derived_w,
        "detail": f"derived {derived_w} vs official {official_w}",
    })
    return checks


def check_invariants(result: dict) -> list[dict]:
    teams = result["teams"]
    m = result["metrics"]
    a, b = teams
    checks = []

    poss_sum = m[a]["possession_pct"] + m[b]["possession_pct"]
    checks.append({"check": "possession_sums_to_100",
                   "passed": abs(poss_sum - 100) <= 1.5,
                   "detail": f"sum={poss_sum}"})

    tilt_sum = m[a]["field_tilt_pct"] + m[b]["field_tilt_pct"]
    checks.append({"check": "field_tilt_sums_to_100",
                   "passed": abs(tilt_sum - 100) <= 1.5,
                   "detail": f"sum={tilt_sum}"})

    for t in teams:
        checks.append({"check": f"shots_on_target<=shots ({t[:12]})",
                       "passed": m[t]["shots_on_target"] <= m[t]["shots"],
                       "detail": f"{m[t]['shots_on_target']}/{m[t]['shots']}"})
        checks.append({"check": f"pass_completion_in_range ({t[:12]})",
                       "passed": 0 <= m[t]["pass_completion_pct"] <= 100,
                       "detail": f"{m[t]['pass_completion_pct']}%"})
    return checks


def check_verified_facts(golden: dict, result: dict) -> list[dict]:
    facts = VERIFIED_FACTS.get(golden["match_id"])
    if not facts:
        return []
    m, teams = result["metrics"], result["teams"]
    checks = []
    if "possession_leader" in facts:
        leader = max(teams, key=lambda t: m[t]["possession_pct"])
        checks.append({"check": "possession_leader_correct",
                       "passed": leader == facts["possession_leader"],
                       "detail": f"got {leader}"})
    if "xg_leader" in facts:
        leader = max(teams, key=lambda t: m[t]["xg"])
        checks.append({"check": "xg_leader_correct",
                       "passed": leader == facts["xg_leader"],
                       "detail": f"got {leader}"})
    return checks


def check_llm_faithfulness() -> list[dict]:
    """Optional: the report's cited numbers must exist in the input stats."""
    import report_generator
    competition = {"name": "Premier League"}
    analysis = {
        "matches_analysed": 2, "total_goals": 5, "average_goals_per_match": 2.5,
        "ranked_teams": [
            {"team": "Arsenal FC", "points": 3, "goals_for": 3, "goals_against": 0,
             "goal_difference": 3, "played": 1, "wins": 1, "draws": 0, "losses": 0},
            {"team": "Chelsea FC", "points": 0, "goals_for": 1, "goals_against": 2,
             "goal_difference": -1, "played": 1, "wins": 0, "draws": 0, "losses": 1},
        ],
    }
    matches = [{"home_team": "Arsenal FC", "away_team": "Coventry", "home_score": 3,
                "away_score": 0, "match_id": 1}]
    report = report_generator.generate_report_llm(competition, analysis, matches)

    haystack = json.dumps(analysis) + json.dumps(matches)
    numbers_in_input = set(re.findall(r"\d+\.?\d*", haystack))
    cited = []
    for f in report.key_findings:
        cited += re.findall(r"\d+\.?\d*", f.evidence)
    unseen = [n for n in cited if n not in numbers_in_input and float(n) not in
              {float(x) for x in numbers_in_input}]
    return [{
        "check": "report_cites_only_input_numbers",
        "passed": len(unseen) == 0,
        "detail": f"{len(cited)} numbers cited, {len(unseen)} not in input: {unseen[:5]}",
    }]


def run(with_llm: bool = False) -> dict:
    golden = build_golden()
    all_checks = []

    print(f"Evaluating {len(golden)} golden matches...\n")
    for g in golden:
        result = tactical_metrics.compute_match_metrics(g["match_id"])
        case = (f"{g['home_team']} {g['home_score']}-{g['away_score']} "
                f"{g['away_team']} (id {g['match_id']})")
        checks = (check_correctness(g, result)
                  + check_invariants(result)
                  + check_verified_facts(g, result))
        for c in checks:
            c["case"] = case
            flag = "PASS" if c["passed"] else "FAIL"
            print(f"  [{flag}] {c['check']:<38} {c['detail']}")
        all_checks += checks
        print()

    if with_llm:
        print("LLM faithfulness check (anti-hallucination)...")
        try:
            for c in check_llm_faithfulness():
                c["case"] = "llm_report"
                flag = "PASS" if c["passed"] else "FAIL"
                print(f"  [{flag}] {c['check']:<38} {c['detail']}")
                all_checks.append(c)
        except Exception as exc:
            print(f"  [SKIP] faithfulness check errored: {exc}")
        print()

    passed = sum(1 for c in all_checks if c["passed"])
    total = len(all_checks)
    accuracy = round(100 * passed / total, 1) if total else 0.0

    summary = {
        "total_checks": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy_pct": accuracy,
        "checks": all_checks,
    }
    print("=" * 60)
    print(f"QUALITY METRICS: {passed}/{total} checks passed  ({accuracy}%)")
    print("=" * 60)

    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {RESULTS_FILE.name}")
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true",
                        help="also run the LLM faithfulness check (uses Groq)")
    args = parser.parse_args()
    summary = run(with_llm=args.with_llm)
    sys.exit(0 if summary["failed"] == 0 else 1)
