"""Cached collection of factual match/lineup fields from ESPN's public website API.

This is an undocumented website endpoint, not an open-licensed provider contract.
No articles, commentary, photographs, or live team records are retained as features.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import pandas as pd
import requests

from .data import normalize_formation, validate_matches

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
_local = threading.local()
FD_NAMES = {
    "Bournemouth": "AFC Bournemouth", "Brighton": "Brighton & Hove Albion",
    "Leeds": "Leeds United", "Leicester": "Leicester City", "Luton": "Luton Town",
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Newcastle": "Newcastle United", "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest", "Sheffield United": "Sheffield United",
    "Tottenham": "Tottenham Hotspur", "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United", "Wolves": "Wolverhampton Wanderers",
    "Ipswich": "Ipswich Town", "Coventry": "Coventry City", "Swansea": "Swansea City",
    "Stoke": "Stoke City", "Birmingham": "Birmingham City", "Blackburn": "Blackburn Rovers",
    "Hull": "Hull City",
}


def _request(url: str) -> requests.Response:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "EPLTacticsResearch/0.2 (local, cached factual analysis)"
    for attempt in range(3):
        try:
            response = _local.session.get(url, timeout=(15, 45))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def parse_summary(summary: dict) -> dict:
    header = summary["header"]
    competition = header["competitions"][0]
    if not competition["status"]["type"].get("completed"):
        raise ValueError("Only completed games are permitted")
    competitors = {r["homeAway"]: r for r in competition["competitors"]}
    rosters = {r["homeAway"]: r for r in summary.get("rosters", [])}
    stats = {r["homeAway"]: {s["name"]: s["displayValue"] for s in r.get("statistics", [])}
             for r in summary.get("boxscore", {}).get("teams", [])}
    year = int(header["season"]["year"])
    row = {"match_id": "espn_" + str(header["id"]),
           "date": str(pd.Timestamp(competition["date"]).tz_convert("Europe/London").date()),
           "season": f"{year}/{year+1}",
           "source_url": f'https://www.espn.com/soccer/lineups/_/gameId/{header["id"]}'}
    for side in ["home", "away"]:
        roster = rosters.get(side, {})
        row[f"{side}_team"] = competitors[side]["team"]["displayName"]
        row[f"{side}_goals"] = int(competitors[side]["score"])
        formation = roster.get("formation")
        row[f"{side}_formation_raw"] = formation
        # ESPN's '-d' is a diamond subtype, not an additional line of players.
        if formation and formation.endswith("-d"):
            formation = formation[:-2]
        row[f"{side}_formation"] = normalize_formation(formation) if formation else None
        row[f"{side}_starters"] = ";".join(sorted(str(p["athlete"]["id"])
                                                     for p in roster.get("roster", []) if p.get("starter")))
        for name, provider in [("shots", "totalShots"), ("shots_on_target", "shotsOnTarget"),
                               ("possession", "possessionPct")]:
            raw = stats.get(side, {}).get(provider)
            row[f"{side}_{name}"] = float(str(raw).strip("%")) if raw is not None else None
    return row


def _summary(event_id: str, cache_dir: Path) -> dict:
    path = cache_dir / f"{event_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    row = parse_summary(_request(f"{ESPN}/summary?event={event_id}").json())
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.15)
    return row


def fetch_recent(output: str | Path = "data/epl_recent.csv", start_year: int = 2019,
                 cutoff: str = "2026-10-28", workers: int = 4,
                 cache_dir: str | Path = "data/raw/espn") -> dict:
    cutoff_date = pd.Timestamp(cutoff).normalize()
    if start_year < 2010 or start_year > cutoff_date.year or not 1 <= workers <= 6:
        raise ValueError("Invalid year, cutoff or worker count")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    events = {}
    for year in range(start_year, cutoff_date.year + 1):
        path = cache_dir / f"index_{year}_{cutoff}.json"
        if path.exists():
            index = json.loads(path.read_text(encoding="utf-8"))
        else:
            index = _request(f"{ESPN}/scoreboard?dates={year}&limit=1000").json()
            # Retain only event discovery fields; never use current records/form.
            index = [{"id": e["id"], "date": e["date"], "year": e["season"]["year"],
                      "completed": e["competitions"][0]["status"]["type"].get("completed", False)}
                     for e in index.get("events", [])]
            path.write_text(json.dumps(index), encoding="utf-8")
        for event in index:
            if (int(event["year"]) >= start_year and event["completed"]
                    and pd.Timestamp(event["date"]).tz_localize(None).normalize() <= cutoff_date):
                events[str(event["id"])] = event
    print(f"Discovered {len(events)} completed matches through {cutoff}", flush=True)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_summary, event_id, cache_dir): event_id for event_id in events}
        for completed, future in enumerate(as_completed(jobs), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"event_id": jobs[future], "error": str(exc)})
            if completed % 100 == 0 or completed == len(jobs):
                print(f"Collected match facts: {completed}/{len(jobs)}; failed={len(failures)}", flush=True)
    if failures:
        (cache_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} fetch failures; rerun to resume, see {cache_dir}/failures.json")
    (cache_dir / "failures.json").write_text("[]", encoding="utf-8")
    all_rows = pd.DataFrame(rows).sort_values(["date", "match_id"])
    complete = all_rows.dropna(subset=["home_formation", "away_formation"])
    frame = validate_matches(complete)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output.with_name(output.stem + "_all_results.csv"), index=False)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d")
    manifest = {"source": "ESPN public website data endpoints", "endpoint": ESPN,
                "access": "No authentication required at collection time; undocumented endpoint; no open-data license asserted.",
                "cutoff": cutoff, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "discovered_completed_matches": len(events), "formation_complete_matches": len(frame),
                "missing_formation_matches": len(all_rows) - len(frame),
                "non_11_starter_records": {side: int(frame[f"{side}_starters"].map(
                    lambda value: len([p for p in str(value).split(";") if p]) != 11).sum()) for side in ["home", "away"]},
                "matches_by_season": frame.groupby("season").size().to_dict(),
                "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "post_match_fields": "Shots/possession retained for lagged histories only. Never features of the same match."}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def crosscheck_football_data(matches_path: str | Path, output_dir: str | Path = "data/football_data") -> dict:
    """Independent result verification and market baseline, not model features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    espn = pd.read_csv(matches_path)
    frames = []
    for season in sorted(espn.season.unique()):
        start = int(season[:4])
        code = f"{start % 100:02d}{(start+1) % 100:02d}"
        path = output_dir / f"E0_{code}.csv"
        if not path.exists():
            path.write_bytes(_request(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv").content)
        data = pd.read_csv(path)
        data = data.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        data["season"] = season
        data["home_team"] = data.HomeTeam.map(lambda name: FD_NAMES.get(name, name))
        data["away_team"] = data.AwayTeam.map(lambda name: FD_NAMES.get(name, name))
        data["fd_date"] = pd.to_datetime(data.Date, dayfirst=True).dt.strftime("%Y-%m-%d")
        frames.append(data)
    fd = pd.concat(frames, ignore_index=True)
    joined = espn.merge(fd, on=["season", "home_team", "away_team"], how="left", validate="one_to_one", indicator=True)
    found = joined._merge.eq("both")
    scores_ok = joined.home_goals.eq(joined.FTHG) & joined.away_goals.eq(joined.FTAG)
    dates_ok = joined.date.eq(joined.fd_date)
    mismatch = joined.loc[~found | ~scores_ok | ~dates_ok,
                          ["match_id", "date", "season", "home_team", "away_team", "home_goals", "away_goals", "FTHG", "FTAG", "fd_date"]]
    mismatch.to_csv(output_dir / "crosscheck_mismatches.csv", index=False)
    market = joined[["match_id", "season"]].copy()
    # Prefer closing average odds; retain exact column names per record.
    labels = [("A", "loss"), ("D", "draw"), ("H", "win")]
    for _, label in labels:
        market[f"odds_{label}"] = float("nan")
    market["odds_columns"] = ""
    for prefix in ["AvgC", "Avg", "B365C", "B365"]:
        columns = [prefix + side for side, _ in labels]
        if not all(column in joined for column in columns):
            continue
        values = joined[columns].apply(pd.to_numeric, errors="coerce")
        usable = values.gt(1).all(axis=1) & market.odds_columns.eq("")
        for i, (_, label) in enumerate(labels):
            market.loc[usable, f"odds_{label}"] = values.loc[usable].iloc[:, i]
        market.loc[usable, "odds_columns"] = ",".join(columns)
    valid = market[["odds_loss", "odds_draw", "odds_win"]].gt(1).all(axis=1) & found & scores_ok & dates_ok
    inverse = 1 / market.loc[valid, ["odds_loss", "odds_draw", "odds_win"]]
    normalized = inverse.div(inverse.sum(axis=1), axis=0)
    for i, label in enumerate(["loss", "draw", "win"]):
        market.loc[valid, f"p_{label}"] = normalized.iloc[:, i]
    market["verified"] = found & scores_ok & dates_ok
    market.to_csv(output_dir / "market_probabilities.csv", index=False)
    report = {"source": "https://football-data.co.uk/englandm.php", "espn_matches": len(espn),
              "verified_matches": int((found & scores_ok & dates_ok).sum()),
              "mismatches_or_unmatched": len(mismatch), "market_coverage": int(valid.sum()),
              "football_data_matches_by_season": fd.groupby("season").size().to_dict(),
              "note": "Market odds are a pre-kickoff reference with additional information, never input features. "
                      "Closing odds preferred; otherwise average then Bet365 available columns. This is not an equal-input comparator."}
    (output_dir / "crosscheck.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
