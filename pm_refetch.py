#!/usr/bin/env python3
"""Cache Wikipedia opinion-polling page HTML locally via the MediaWiki API.

The plain article URLs get aggressively 429'd; api.php with a descriptive
User-Agent (per Wikimedia's robot policy) works fine.
"""
import sys
import time
from pathlib import Path

import requests

CACHE = Path(__file__).parent / "raw_data" / "wikipedia_polling"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "nz-polls-research/1.0 (danylmc@gmail.com) python-requests"}
PAGE = "Opinion polling for the {year} New Zealand general election"


def fetch(year: int, session: requests.Session, force: bool = False) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{year}.html"
    if path.exists() and not force:
        print(f"  {year}: cached")
        return path

    params = {
        "action": "parse",
        "page": PAGE.format(year=year),
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    delay = 5
    for _ in range(6):
        r = session.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            print(f"  {year}: 429, sleeping {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print(f"  {year}: {data['error'].get('info')}")
            return None
        html = data["parse"]["text"]
        path.write_text(html, encoding="utf-8")
        print(f"  {year}: fetched ({len(html)} chars)")
        return path
    raise RuntimeError(f"{year}: gave up after repeated 429s")


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]]
    session = requests.Session()
    for i, y in enumerate(years):
        if i:
            time.sleep(2)
        fetch(y, session, force=True)
