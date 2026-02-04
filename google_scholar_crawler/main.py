# from scholarly import scholarly
# import jsonpickle
# import json
# from datetime import datetime
# import os

# author: dict = scholarly.search_author_id(os.environ['GOOGLE_SCHOLAR_ID'])
# scholarly.fill(author, sections=['basics', 'indices', 'counts', 'publications'])
# name = author['name']
# author['updated'] = str(datetime.now())
# author['publications'] = {v['author_pub_id']:v for v in author['publications']}
# print(json.dumps(author, indent=2))
# os.makedirs('results', exist_ok=True)
# with open(f'results/gs_data.json', 'w') as outfile:
#     json.dump(author, outfile, ensure_ascii=False)

# shieldio_data = {
#   "schemaVersion": 1,
#   "label": "citations",
#   "message": f"{author['citedby']}",
# }
# with open(f'results/gs_data_shieldsio.json', 'w') as outfile:
#     json.dump(shieldio_data, outfile, ensure_ascii=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust Google Scholar stats fetcher for GitHub Actions.

Key features:
- Avoids hanging indefinitely (hard timeout)
- Retries with exponential backoff
- Keeps previous results on failure (so shields badge won't break)
- Atomic writes to prevent corrupted JSON
- Optional publications fetching via env FETCH_PUBLICATIONS=1
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict

from scholarly import scholarly


# -----------------------------
# Configuration (via env vars)
# -----------------------------
AUTHOR_ID_ENV = "GOOGLE_SCHOLAR_ID"
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
OUT_AUTHOR_JSON = os.path.join(RESULTS_DIR, "gs_data.json")
OUT_SHIELDS_JSON = os.path.join(RESULTS_DIR, "gs_data_shieldsio.json")

# Hard timeout for the *whole* fetching process (seconds)
HARD_TIMEOUT_S = int(os.getenv("HARD_TIMEOUT_S", "240"))  # 4 minutes by default

# Retry settings
MAX_TRIES = int(os.getenv("MAX_TRIES", "4"))
BASE_BACKOFF_S = float(os.getenv("BASE_BACKOFF_S", "2.0"))
MAX_BACKOFF_S = float(os.getenv("MAX_BACKOFF_S", "45.0"))

# Whether to fetch publications (expensive / higher ban risk)
FETCH_PUBLICATIONS = os.getenv("FETCH_PUBLICATIONS", "0") == "1"

# Sections to fill (keep minimal by default)
SECTIONS_MINIMAL = ["basics", "indices", "counts"]
SECTIONS_WITH_PUBS = ["basics", "indices", "counts", "publications"]


# -----------------------------
# Utilities
# -----------------------------
class TimeoutError(Exception):
    """Raised when hard timeout is reached."""


@contextmanager
def hard_timeout(seconds: int):
    """
    A hard timeout using SIGALRM (works on Linux; GitHub Actions runners are Linux by default).
    Ensures we never hang for hours if scholarly gets stuck.
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Hard timeout reached: {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def atomic_write_json(path: str, obj: Any):
    """Write JSON atomically to avoid partial/corrupt files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_json_if_exists(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        # If file exists but corrupted, ignore and treat as missing
        return None


def backoff_sleep(attempt_idx: int):
    """Exponential backoff with jitter."""
    exp = BASE_BACKOFF_S * (2 ** attempt_idx)
    sleep_s = min(MAX_BACKOFF_S, exp)
    jitter = random.uniform(0, 0.35 * sleep_s)
    time.sleep(sleep_s + jitter)


def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------
# Fetch logic
# -----------------------------
def fetch_author_data(author_id: str) -> Dict[str, Any]:
    """
    Fetch scholar author data with scholarly.
    May raise exceptions; caller handles retry/timeout.
    """
    author: Dict[str, Any] = scholarly.search_author_id(author_id)

    sections = SECTIONS_WITH_PUBS if FETCH_PUBLICATIONS else SECTIONS_MINIMAL
    scholarly.fill(author, sections=sections)

    # Normalize output fields
    author["updated"] = now_str()

    # publications could be absent if not fetched or if fill failed partially
    pubs = author.get("publications", [])
    if isinstance(pubs, list):
        # Key by author_pub_id when available (more stable for diffs)
        keyed = {}
        for p in pubs:
            if isinstance(p, dict) and "author_pub_id" in p:
                keyed[p["author_pub_id"]] = p
        if keyed:
            author["publications"] = keyed

    return author


def build_shields(author: Dict[str, Any]) -> Dict[str, Any]:
    citedby = author.get("citedby")
    msg = "?"
    if citedby is not None:
        msg = str(citedby)

    return {
        "schemaVersion": 1,
        "label": "citations",
        "message": msg,
    }


def main() -> int:
    author_id = os.getenv(AUTHOR_ID_ENV)
    if not author_id:
        print(f"[ERROR] Missing env var {AUTHOR_ID_ENV}.", file=sys.stderr)
        # Keep previous output if exists; exit 0 so badge doesn't break CI
        prev = load_json_if_exists(OUT_AUTHOR_JSON)
        if prev is None:
            # No previous data; still create placeholder shields so badge won't 404
            atomic_write_json(OUT_SHIELDS_JSON, {"schemaVersion": 1, "label": "citations", "message": "?"})
        return 0

    # Attempt fetch with retries + hard timeout guarding the whole operation
    last_err: Exception | None = None
    author: Dict[str, Any] | None = None

    try:
        with hard_timeout(HARD_TIMEOUT_S):
            for i in range(MAX_TRIES):
                try:
                    print(f"[INFO] Fetch attempt {i+1}/{MAX_TRIES} (publications={FETCH_PUBLICATIONS})")
                    author = fetch_author_data(author_id)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"[WARN] Attempt {i+1} failed: {repr(e)}", file=sys.stderr)
                    if i < MAX_TRIES - 1:
                        backoff_sleep(i)
    except TimeoutError as e:
        last_err = e
        print(f"[WARN] {repr(e)}", file=sys.stderr)

    if author is None:
        # Fetch failed: keep previous files if possible
        print("[WARN] Fetch failed; keeping previous results (if any).", file=sys.stderr)

        prev_author = load_json_if_exists(OUT_AUTHOR_JSON)
        prev_shields = load_json_if_exists(OUT_SHIELDS_JSON)

        # If nothing exists yet, create placeholder shields so endpoint exists
        if prev_author is None and prev_shields is None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            atomic_write_json(OUT_SHIELDS_JSON, {"schemaVersion": 1, "label": "citations", "message": "?"})

        # Exit 0 so workflow doesn't go red
        return 0

    # Success: write outputs atomically
    os.makedirs(RESULTS_DIR, exist_ok=True)
    atomic_write_json(OUT_AUTHOR_JSON, author)
    atomic_write_json(OUT_SHIELDS_JSON, build_shields(author))

    print("[INFO] Success. citedby =", author.get("citedby"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
