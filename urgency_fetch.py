#!/usr/bin/env python3
"""
Fetch the legislative history of every public Act, 1999-2026, from legislation.govt.nz.

Each Act's as-enacted PDF ends with a "Legislative history" table (introduction,
first reading and referral to select committee, report-back, second reading,
committee of the whole House, third reading, Royal assent). Tables start in 2000;
1999 Acts are fetched but mostly have none.

Steps per Act (Act list from the site's sitemap.xml):
  1. latest page   /act/public/YYYY/N/en/latest/   -> a dated version link
  2. versions page /act/public/YYYY/N/en/<date>/versions/  -> earliest (as-enacted) date
  3. as-enacted PDF /act/public/YYYY/N/en/<earliest>.pdf -> legislative history text

robots.txt disallows query strings, so none are used. Results are cached in
raw_data/legislation_history/ (one JSON per Act); reruns skip cached Acts.

    python urgency_fetch.py            # all years
    python urgency_fetch.py 2024 2025  # selected years
"""

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

BASE = Path(__file__).parent
CACHE = BASE / "raw_data" / "legislation_history"
SITE = "https://www.legislation.govt.nz"
UA = "nz-polls research (danylmc@gmail.com)"
FIRST_YEAR, LAST_YEAR = 1999, 2026
DELAY = 0.25

session = requests.Session()
session.headers["User-Agent"] = UA


def get(url, binary=False):
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(DELAY)
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.content if binary else r.content.decode("utf-8", "replace")
    raise RuntimeError(f"failed: {url}")


def act_list(years):
    sm = get(f"{SITE}/sitemap.xml")
    acts = set()
    for y, n in re.findall(r"/act/public/(\d{4})/(\d+)/en/latest/", sm):
        if int(y) in years:
            acts.add((int(y), int(n)))
    return sorted(acts)


def history_from_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_bytes)
        f.flush()
        text = subprocess.run(["pdftotext", "-layout", f.name, "-"],
                              capture_output=True).stdout.decode("utf-8", "replace")
    i = text.rfind("Legislative history")
    if i < 0:
        return None
    block = text[i:]
    # stop at the administering-department line or the publication footer
    m = re.search(r"\n\s*(This Act is administered|Wellington, New Zealand)", block)
    return block[:m.start()] if m else block[:3000]


def fetch_act(year, num):
    prefix = f"/act/public/{year}/{num}/en/"
    rec = {"year": year, "num": num}
    latest = get(f"{SITE}{prefix}latest/")
    if latest is None:
        rec["error"] = "no latest page"
        return rec
    t = re.search(r"<title>\s*(.*?)\s*\|", latest, re.S)
    rec["title"] = re.sub(r"\s+", " ", t.group(1)).strip() if t else ""
    dates = re.findall(re.escape(prefix) + r"(\d{4}-\d{2}-\d{2}[A-Z]?)", latest)
    if not dates:
        rec["error"] = "no dated version"
        return rec
    versions = get(f"{SITE}{prefix}{dates[0]}/versions/") or ""
    all_dates = set(dates) | set(re.findall(re.escape(prefix) + r"(\d{4}-\d{2}-\d{2}[A-Z]?)", versions))
    first = min(all_dates)
    rec["as_enacted"] = first
    pdf = get(f"{SITE}{prefix}{first}.pdf", binary=True)
    if pdf is None:
        rec["error"] = "no pdf"
        return rec
    rec["history"] = history_from_pdf(pdf)
    return rec


def main():
    years = {int(a) for a in sys.argv[1:]} or set(range(FIRST_YEAR, LAST_YEAR + 1))
    CACHE.mkdir(parents=True, exist_ok=True)
    acts = act_list(years)
    todo = [(y, n) for y, n in acts if not (CACHE / f"{y}_{n:04d}.json").exists()]
    print(f"{len(acts)} Acts in sitemap, {len(todo)} to fetch", flush=True)
    for i, (y, n) in enumerate(todo, 1):
        try:
            rec = fetch_act(y, n)
        except Exception as e:  # keep going; failures are retried on rerun
            print(f"  {y}/{n}: {e}", flush=True)
            continue
        (CACHE / f"{y}_{n:04d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} ({y}/{n})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
