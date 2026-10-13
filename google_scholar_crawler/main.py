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
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict

import requests
from bs4 import BeautifulSoup


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
MAX_TRIES = int(os.getenv("MAX_TRIES", "2"))
BASE_BACKOFF_S = float(os.getenv("BASE_BACKOFF_S", "2.0"))
MAX_BACKOFF_S = float(os.getenv("MAX_BACKOFF_S", "45.0"))

# Whether to fetch publications (expensive / higher ban risk)
FETCH_PUBLICATIONS = os.getenv("FETCH_PUBLICATIONS", "0") == "1"
REQUEST_TIMEOUT_S = int(os.getenv("REQUEST_TIMEOUT_S", "20"))
PROFILE_URLS = [
    "https://scholar.google.com/citations",
    "https://scholar.googleusercontent.com/citations",
    "https://scholar.google.co.uk/citations",
]

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
def normalize_author_id(value: str) -> str:
    """Accept either a bare Scholar ID or a copied profile URL/query string."""
    value = value.strip()
    if "user=" in value:
        value = value.split("user=", 1)[1]
    return value.split("&", 1)[0].split("?", 1)[0].strip()


def parse_metric_number(text: str) -> int:
    digits = re.sub(r"[^0-9]", "", text)
    if not digits:
        raise ValueError(f"No numeric value found in {text!r}")
    return int(digits)


def author_from_profile_html(html: str, scholar_id: str) -> Dict[str, Any]:
    """Parse the public Google Scholar profile page."""
    soup = BeautifulSoup(html, "html.parser")
    name_node = soup.select_one("#gsc_prf_in")
    rows = soup.select("#gsc_rsb_st tbody tr")
    metrics: Dict[str, list[int]] = {}

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(" ", strip=True).lower()
        values = []
        for cell in cells[1:]:
            text = cell.get_text(" ", strip=True)
            if text:
                values.append(parse_metric_number(text))
        if values:
            metrics[label] = values

    citation_key = next((k for k in metrics if k.startswith("citations")), None)
    if name_node is None or citation_key is None:
        title = soup.title.get_text(" ", strip=True) if soup.title else "no title"
        raise RuntimeError(
            f"Scholar profile markers missing (page title: {title!r}); "
            "the endpoint may have returned a CAPTCHA."
        )

    previous = load_json_if_exists(OUT_AUTHOR_JSON)
    author: Dict[str, Any] = dict(previous) if isinstance(previous, dict) else {}
    author.update(
        {
            "container_type": "Author",
            "filled": ["basics", "indices", "counts"],
            "scholar_id": scholar_id,
            "source": "AUTHOR_PROFILE_PAGE",
            "name": name_node.get_text(" ", strip=True),
            "updated": now_str(),
        }
    )

    citations = metrics[citation_key]
    author["citedby"] = citations[0]
    if len(citations) > 1:
        author["citedby5y"] = citations[1]

    h_key = next((k for k in metrics if k.startswith("h-index")), None)
    i10_key = next((k for k in metrics if k.startswith("i10-index")), None)
    if h_key:
        author["hindex"] = metrics[h_key][0]
        if len(metrics[h_key]) > 1:
            author["hindex5y"] = metrics[h_key][1]
    if i10_key:
        author["i10index"] = metrics[i10_key][0]
        if len(metrics[i10_key]) > 1:
            author["i10index5y"] = metrics[i10_key][1]

    affiliation = soup.select_one(".gsc_prf_il")
    if affiliation:
        author["affiliation"] = affiliation.get_text(" ", strip=True)
    interests = [
        item.get_text(" ", strip=True)
        for item in soup.select(".gsc_prf_inta")
        if item.get_text(" ", strip=True)
    ]
    if interests:
        author["interests"] = interests

    years = [x.get_text(strip=True) for x in soup.select(".gsc_g_t")]
    counts = [x.get_text(strip=True) for x in soup.select(".gsc_g_al")]
    if years and len(years) == len(counts):
        author["cites_per_year"] = {
            year: parse_metric_number(count)
            for year, count in zip(years, counts)
            if year and count
        }

    return author


def fetch_author_data(author_id: str) -> Dict[str, Any]:
    """
    Fetch with bounded HTTP requests, then fall back to a real headless browser.

    Google often rejects request-library traffic from cloud runners even when
    the public profile remains accessible in a browser. Every path has a hard
    timeout so a scheduled workflow cannot hang for hours.
    """
    scholar_id = normalize_author_id(author_id)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    errors = []

    for base_url in PROFILE_URLS:
        try:
            response = requests.get(
                base_url,
                params={"user": scholar_id, "hl": "en"},
                headers=headers,
                timeout=(10, REQUEST_TIMEOUT_S),
                allow_redirects=True,
            )
            print(
                f"[INFO] Profile endpoint {base_url}: "
                f"HTTP {response.status_code}, {len(response.content)} bytes"
            )
            response.raise_for_status()
            return author_from_profile_html(response.text, scholar_id)
        except Exception as exc:
            errors.append(f"{base_url}: {exc!r}")
            print(f"[WARN] Direct endpoint failed: {base_url}: {exc!r}", file=sys.stderr)

    driver = None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1440,1200")
        options.add_argument(f"--user-agent={headers['User-Agent']}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        url = f"https://scholar.google.com/citations?user={scholar_id}&hl=en"
        print("[INFO] Trying headless Chrome fallback")
        driver.get(url)
        WebDriverWait(driver, 20).until(
            lambda d: d.find_elements("css selector", "#gsc_prf_in")
            or "not a robot" in d.page_source.lower()
        )
        return author_from_profile_html(driver.page_source, scholar_id)
    except Exception as exc:
        errors.append(f"headless Chrome: {exc!r}")
        print(f"[WARN] Headless Chrome failed: {exc!r}", file=sys.stderr)
    finally:
        if driver is not None:
            driver.quit()

    raise RuntimeError("All Scholar profile methods failed: " + " | ".join(errors))


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
