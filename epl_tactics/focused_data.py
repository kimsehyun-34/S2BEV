"""Long histories for a predeclared six-club cohort, anchored to independent fixtures."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from .data import normalize_formation
from .recent_data import ESPN, FD_NAMES, _request

CLUBS = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham Hotspur"]
HISTORICAL_NAMES = {**FD_NAMES, "Bolton":"Bolton Wanderers", "Wigan":"Wigan Athletic",
                    "Charlton":"Charlton Athletic", "Cardiff":"Cardiff City", "QPR":"Queens Park Rangers",
                    "Huddersfield":"Huddersfield Town", "Derby":"Derby County"}
TEAM_STATS = {
    "shots": "totalShots", "sot": "shotsOnTarget", "possession": "possessionPct",
    "corners": "wonCorners", "fouls": "foulsCommitted", "yellow": "yellowCards", "red": "redCards",
    "passes": "totalPasses", "accurate_passes": "accuratePasses", "crosses": "totalCrosses",
    "long_balls": "totalLongBalls", "tackles": "totalTackles", "interceptions": "interceptions",
    "saves": "saves", "blocked_shots": "blockedShots",
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def number(value):
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def parse_facts(summary: dict) -> dict:
    header = summary["header"]
    game = header["competitions"][0]
    if not game["status"]["type"].get("completed"):
        raise ValueError("Incomplete event")
    row = {"event_id": str(header["id"]), "season": f'{header["season"]["year"]}/{int(header["season"]["year"])+1}',
           "date": str(pd.Timestamp(game["date"]).tz_convert("Europe/London").date())}
    competitors = {c["homeAway"]: c for c in game["competitors"]}
    rosters = {r["homeAway"]: r for r in summary.get("rosters", [])}
    stats = {t["homeAway"]: {s["name"]: s["displayValue"] for s in t.get("statistics", [])}
             for t in summary.get("boxscore", {}).get("teams", [])}
    for side in ["home", "away"]:
        row[f"{side}_team"] = competitors[side]["team"]["displayName"]
        row[f"{side}_goals"] = int(competitors[side]["score"])
        roster = rosters.get(side, {})
        raw = roster.get("formation")
        row[f"{side}_formation_raw"] = raw
        try:
            row[f"{side}_formation"] = normalize_formation(raw.removesuffix("-d")) if raw else None
        except ValueError:
            row[f"{side}_formation"] = None
        players = []
        for p in roster.get("roster", []):
            athlete = p.get("athlete", {})
            if not athlete.get("id"):
                continue
            per_player = {s["name"]: number(s.get("value", s.get("displayValue"))) for s in p.get("stats", [])}
            players.append({"id": str(athlete["id"]), "starter": bool(p.get("starter")),
                            "position": p.get("position", {}).get("abbreviation"),
                            "goals": per_player.get("totalGoals"), "assists": per_player.get("goalAssists"),
                            "shots": per_player.get("totalShots"), "sot": per_player.get("shotsOnTarget"),
                            "appearance": per_player.get("appearances")})
        row[f"{side}_players"] = players
        for name, provider in TEAM_STATS.items():
            row[f"{side}_{name}"] = number(stats.get(side, {}).get(provider))
    return row


def _facts(event_id: str, root: Path) -> dict:
    path = root / "facts" / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = parse_facts(_request(f"{ESPN}/summary?event={event_id}").json())
    write_json(path, data)
    time.sleep(.12)
    return data


def official_fixtures(start: int, cutoff: str, root: Path) -> pd.DataFrame:
    frames = []
    for year in range(start, int(cutoff[:4]) + 1):
        code = f"{year % 100:02d}{(year+1)%100:02d}"
        path = root / "football_data" / f"E0_{code}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_request(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv").content)
        data = pd.read_csv(path)
        data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        data["date"] = pd.to_datetime(data.Date, format="mixed", dayfirst=True).dt.normalize()
        data = data[data.date <= pd.Timestamp(cutoff)]
        output = pd.DataFrame({"date": data.date, "season": f"{year}/{year+1}",
                               "home_team": data.HomeTeam.map(lambda x: HISTORICAL_NAMES.get(x, x)),
                               "away_team": data.AwayTeam.map(lambda x: HISTORICAL_NAMES.get(x, x)),
                               "home_goals": data.FTHG.astype(int), "away_goals": data.FTAG.astype(int)})
        for name, home, away in [("shots", "HS", "AS"), ("sot", "HST", "AST"), ("corners", "HC", "AC"),
                                 ("fouls", "HF", "AF"), ("yellow", "HY", "AY"), ("red", "HR", "AR")]:
            for side, column in [("home", home), ("away", away)]:
                output[f"{side}_{name}"] = pd.to_numeric(data[column], errors="coerce") if column in data else float("nan")
        for outcome, suffix in [("loss", "A"), ("draw", "D"), ("win", "H")]:
            output[f"market_{outcome}"] = float("nan")
        output["market_columns"] = ""
        for prefix in ["AvgC", "Avg", "B365C", "B365"]:
            cols = [prefix+s for s in ["A", "D", "H"]]
            if not all(c in data for c in cols):
                continue
            odds = data[cols].apply(pd.to_numeric, errors="coerce")
            mask = odds.gt(1).all(axis=1) & output.market_columns.eq("")
            inverse = 1/odds.loc[mask]
            probability = inverse.div(inverse.sum(axis=1), axis=0)
            for i, label in enumerate(["loss", "draw", "win"]):
                output.loc[mask, f"market_{label}"] = probability.iloc[:, i]
            output.loc[mask, "market_columns"] = ",".join(cols)
        frames.append(output)
    frame = pd.concat(frames, ignore_index=True).sort_values(["date", "home_team"]).reset_index(drop=True)
    frame["match_id"] = [f"fd_{s.replace('/', '_')}_{h}_{a}" for s,h,a in zip(frame.season, frame.home_team, frame.away_team)]
    if frame.match_id.duplicated().any():
        raise ValueError("Duplicate official league fixture")
    return frame


def collect_focused(start: int = 2006, cutoff: str = "2026-10-29", workers: int = 4,
                     output_dir: str | Path = "data/focused") -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    league = official_fixtures(start, cutoff, root)
    focused = league[league.home_team.isin(CLUBS) | league.away_team.isin(CLUBS)].copy()
    official_keys = {(str(r.date.date()), r.home_team, r.away_team): r.match_id for r in focused.itertuples()}
    events: dict[str, list[str]] = {}
    for year in range(start, int(cutoff[:4]) + 1):
        cache = root / "indexes" / f"{year}_{cutoff}.json"
        if cache.exists():
            entries = json.loads(cache.read_text(encoding="utf-8"))
        else:
            entries = []
            response = _request(f"{ESPN}/scoreboard?dates={year}&limit=1000").json()
            for event in response.get("events", []):
                game = event["competitions"][0]
                sides = {c["homeAway"]: c["team"]["displayName"] for c in game["competitors"]}
                if not game["status"]["type"].get("completed") or not set(sides.values()).intersection(CLUBS):
                    continue
                entries.append({"id": str(event["id"]), "date": str(pd.Timestamp(event["date"]).tz_convert("Europe/London").date()),
                                "home_team": sides["home"], "away_team": sides["away"]})
            write_json(cache, entries)
        for entry in entries:
            key = (entry["date"], entry["home_team"], entry["away_team"])
            if key in official_keys:
                events.setdefault(official_keys[key], [])
                if entry["id"] not in events[official_keys[key]]:
                    events[official_keys[key]].append(entry["id"])
        print(f"Index {year}: {len(events)}/{len(focused)} focused fixtures matched", flush=True)
    event_ids = sorted({event_id for ids in events.values() for event_id in ids})
    facts, failures = {}, []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_facts, event_id, root): event_id for event_id in event_ids}
        for completed, future in enumerate(as_completed(jobs), 1):
            event_id = jobs[future]
            try:
                facts[event_id] = future.result()
            except Exception as exc:
                failures.append({"event_id": event_id, "error": str(exc)})
            if completed % 100 == 0 or completed == len(jobs):
                print(f"Detailed collection {completed}/{len(jobs)}; failures={len(failures)}", flush=True)
    write_json(root / "fetch_failures.json", failures)
    if failures:
        raise RuntimeError("Fetch failures recorded; rerun to resume cached collection")
    rows, mismatches = [], []
    for fixture in focused.to_dict(orient="records"):
        candidates = []
        for event_id in events.get(fixture["match_id"], []):
            detail = facts[event_id]
            if any(detail.get(key) != fixture[key] for key in ["home_team", "away_team", "home_goals", "away_goals", "season"]):
                mismatches.append({"match_id": fixture["match_id"], "event_id": event_id, "reason": "identity or result differs"})
                continue
            if detail["date"] != str(fixture["date"].date()):
                continue
            richness = sum(detail.get(f"{s}_formation") is not None for s in ["home", "away"])*100
            richness += sum(len(detail.get(f"{s}_players", [])) for s in ["home", "away"])
            richness += sum(detail.get(f"{s}_{name}") is not None for s in ["home", "away"] for name in TEAM_STATS)
            candidates.append((richness, event_id, detail))
        if candidates:
            _, event_id, detail = max(candidates, key=lambda value: (value[0], value[1]))
            fixture["event_id"] = event_id
            fixture["espn_verified"] = True
            for side in ["home", "away"]:
                for key in ["formation", "formation_raw", "players", *TEAM_STATS]:
                    name = f"{side}_{key}"
                    value = detail.get(name)
                    if value is not None:
                        fixture[name] = json.dumps(value, ensure_ascii=False) if key == "players" else value
        else:
            fixture["espn_verified"] = False
        for side in ["home", "away"]:
            fixture.setdefault(f"{side}_players", "[]")
            fixture.setdefault(f"{side}_formation", None)
        rows.append(fixture)
    frame = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    frame.to_csv(root / "matches.csv", index=False, date_format="%Y-%m-%d")
    # Opponent histories use complete league result/stat context; supervised examples stay in CLUBS.
    league.to_csv(root / "league_context.csv", index=False, date_format="%Y-%m-%d")
    coverage = []
    for team in CLUBS:
        appearances = pd.concat([
            frame[frame.home_team.eq(team)].assign(side="home"), frame[frame.away_team.eq(team)].assign(side="away")])
        for season, period in appearances.groupby("season"):
            coverage.append({"team": team, "season": season, "matches": len(period),
                             "formation_recorded": sum(pd.notna(r[f'{r["side"]}_formation']) for r in period.to_dict("records")),
                             "starting_eleven_recorded": sum(sum(p["starter"] for p in json.loads(r[f'{r["side"]}_players'])) == 11
                                                             for r in period.to_dict("records"))})
    pd.DataFrame(coverage).to_csv(root / "coverage.csv", index=False)
    write_json(root / "source_mismatches.json", mismatches)
    manifest = {"clubs": CLUBS, "start_season": start, "cutoff": cutoff, "unique_matches": len(frame),
                "club_appearances": pd.DataFrame(coverage).groupby("team").matches.sum().to_dict(),
                "coverage": coverage, "espn_verified_unique_matches": int(frame.espn_verified.sum()),
                "unmatched_fixture_count": int((~frame.espn_verified).sum()), "rejected_espn_alternatives": len(mismatches),
                "league_context_matches": len(league),
                "sha256": hashlib.sha256((root / "matches.csv").read_bytes()).hexdigest(),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": ["https://football-data.co.uk/englandm.php", ESPN],
                "scope": "Supervised rows restricted to six clubs; all their EPL opponents included. League results used only as opponent context.",
                "license_note": "Public website access does not assert an open redistribution license; ESPN endpoint undocumented.",
                "missing_data": "No formation inferred. Missing lineup/stats retained, coverage exported. Provider per-player match stats used only after date."}
    write_json(root / "manifest.json", manifest)
    return manifest
