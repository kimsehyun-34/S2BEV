"""Fetch only the opening Starting XI events, not entire event datasets."""

from __future__ import annotations

import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterator

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data"
REQUIRED = ["match_id", "date", "season", "home_team", "away_team", "home_formation",
            "away_formation", "home_goals", "away_goals"]


def normalize_formation(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if re.fullmatch(r"[0-6]{3,5}", text):
        parts = list(text)
    elif re.fullmatch(r"[0-6](?:-[0-6]){2,4}", text):
        parts = text.split("-")
    else:
        raise ValueError(f"Invalid formation: {value!r}; use e.g. 4-2-3-1 (no goalkeeper)")
    if any(int(p) == 0 for p in parts[:-1]) or sum(map(int, parts)) != 10:
        raise ValueError(f"Formation must contain ten outfield players: {value!r}")
    return "-".join(parts)


def validate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame.copy()
    if frame.empty or frame[REQUIRED].isna().any().any():
        raise ValueError("Match data is empty or contains missing required values")
    frame["match_id"] = frame["match_id"].astype(str).str.strip()
    if frame["match_id"].eq("").any() or frame["match_id"].duplicated().any():
        raise ValueError("match_id must be nonempty and unique")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Match dates must not be NaT or missing")
    if frame["date"].dt.tz is not None:
        raise ValueError("Use local match dates without timezone offsets")
    for col in ["season", "home_team", "away_team"]:
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].eq("").any():
            raise ValueError(f"Empty {col}")
    if frame["home_team"].eq(frame["away_team"]).any():
        raise ValueError("A team cannot play itself")
    for col in ["home_formation", "away_formation"]:
        frame[col] = frame[col].map(normalize_formation)
    for col in ["home_goals", "away_goals"]:
        numeric = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"{col} must be nonnegative integers")
        frame[col] = numeric.astype(int)
    # Day-based features deliberately do not depend on uncertain kickoff timezones.
    appearances = pd.concat([
        frame[["season", "date", "home_team"]].rename(columns={"home_team": "team"}),
        frame[["season", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    if appearances.duplicated(["season", "date", "team"]).any():
        raise ValueError("A team appears twice on the same date in one season")
    return frame.sort_values(["date", "match_id"]).reset_index(drop=True)


def read_matches(path: str | Path) -> pd.DataFrame:
    return validate_matches(pd.read_csv(path, dtype={"match_id": str, "season": str}))


def iter_json_array(chunks: Iterator[bytes]) -> Iterator[dict]:
    """Incrementally decode JSON objects in an array, including split UTF-8 chars."""
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8-sig")()
    buffer = ""
    started = False
    for chunk in chunks:
        buffer += utf8.decode(chunk)
        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    break
                if not buffer.startswith("["):
                    raise ValueError("Expected a JSON event array")
                buffer = buffer[1:]
                started = True
                continue
            buffer = buffer.lstrip(" \r\n\t,")
            if buffer.startswith("]"):
                return
            if not buffer:
                break
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            if not isinstance(item, dict):
                raise ValueError("Expected an event object")
            buffer = buffer[end:]
            yield item
    raise ValueError("Truncated event array before both Starting XI records were found")


def _get_json(url: str) -> object:
    with requests.get(url, timeout=(15, 60)) as response:
        response.raise_for_status()
        return response.json()


def _starting_xi(match_id: int, root: Path, ref: str) -> list[dict]:
    cache = root / f"starting_xi_{match_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{BASE_URL}/{ref}/data/events/{match_id}.json"
    for attempt in range(3):
        try:
            found = []
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                for index, event in enumerate(iter_json_array(response.iter_content(8192))):
                    if event.get("type", {}).get("name") == "Starting XI":
                        found.append(event)
                    if len(found) == 2:
                        break
                    if index > 100:
                        raise ValueError(f"Missing opening Starting XI: {match_id}")
            if len(found) != 2:
                raise ValueError(f"Missing Starting XI: {match_id}")
            cache.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
            return found
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def fetch_statsbomb(output: str | Path, season_id: int = 27, workers: int = 4,
                    ref: str = "master", cache_dir: str | Path = "data/raw") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", ref):
        raise ValueError("ref must be a simple Git revision or commit SHA")
    if season_id < 1 or not 1 <= workers <= 8:
        raise ValueError("season_id must be positive and workers must be in 1..8")
    root = Path(cache_dir) / ref / f"2_{season_id}"
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "matches.json"
    if index_path.exists():
        matches = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        matches = _get_json(f"{BASE_URL}/{ref}/data/matches/2/{season_id}.json")
        index_path.write_text(json.dumps(matches, ensure_ascii=False), encoding="utf-8")
    if not isinstance(matches, list) or not matches:
        raise ValueError("No EPL matches returned")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(_starting_xi, m["match_id"], root, ref): m for m in matches}
        for completed, future in enumerate(as_completed(jobs), 1):
            match = jobs[future]
            try:
                events = future.result()
                formations = {e["team"]["id"]: normalize_formation(e["tactics"]["formation"])
                              for e in events}
                rows.append({
                    "match_id": str(match["match_id"]), "date": match["match_date"],
                    "season": match["season"]["season_name"],
                    "home_team": match["home_team"]["home_team_name"],
                    "away_team": match["away_team"]["away_team_name"],
                    "home_formation": formations[match["home_team"]["home_team_id"]],
                    "away_formation": formations[match["away_team"]["away_team_id"]],
                    "home_goals": match["home_score"], "away_goals": match["away_score"],
                })
            except Exception as exc:
                failures.append({"match_id": match["match_id"], "error": str(exc)})
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Downloaded Starting XI: {completed}/{len(jobs)}", flush=True)
    # Never silently learn on a partial sample after fetch failures.
    if failures:
        (root / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"Failed to fetch {len(failures)} matches; rerun to resume cached downloads")
    frame = validate_matches(pd.DataFrame(rows))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    manifest = {
        "source": "StatsBomb Open Data", "repository": "https://github.com/statsbomb/open-data",
        "license": "https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf",
        "attribution": "Data provided by StatsBomb (Hudl).",
        "competition_id": 2, "season_id": season_id, "git_ref": ref,
        "seasons": sorted(frame.season.unique().tolist()), "matches": len(frame),
        "teams": sorted(set(frame.home_team) | set(frame.away_team)),
        "first_date": str(frame.date.min().date()), "last_date": str(frame.date.max().date()),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "note": "Only starting formations and match results. No full events or player attributes.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
