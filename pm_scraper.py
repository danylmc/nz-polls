#!/usr/bin/env python3
"""
NZ Preferred Prime Minister Polling Data Scraper

Scrapes the "Preferred prime minister" section of the Wikipedia opinion-polling
article for each NZ general election, 1999 onwards (the earliest cycle for which
Wikipedia carries a PM table; 2005 has no PM section).

Pages are fetched via the MediaWiki API and cached under
raw_data/wikipedia_polling/ — the plain article URLs get rate-limited hard.

Writes:
  data/{year}_pm_polling.json   one file per election cycle
  data/pm_polling_all.csv       deduplicated long-format table of every poll
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from pm_refetch import fetch, CACHE

# Election cycles with a Preferred PM section on Wikipedia.
# 1993, 1996 and 2005 have no PM table.
ELECTION_YEARS = [1999, 2002, 2008, 2011, 2014, 2017, 2020, 2023, 2026]

ROOT = Path(__file__).parent

# Header cells that are metadata, not candidates
NON_CANDIDATE_HEADERS = {
    'poll', 'polls', 'date', 'polling organisation', 'polling organization',
    'sample size', 'sample', 'lead', 'others', 'other', 'undecided',
    'don\'t know', 'dont know', 'none', 'none of the above', 'someone else',
    'no one', 'refused', 'n/a',
}

MONTHS = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def clean_text(text: str) -> str:
    """Strip footnote markers and collapse whitespace"""
    if not text:
        return ""
    text = re.sub(r'\[\s*[\w\s]+\s*\]', '', text)
    text = re.sub(r'[†‡§¶*]+', '', text)
    return ' '.join(text.replace('\xa0', ' ').split()).strip()


def parse_percentage(value: str) -> Optional[float]:
    """Parse a percentage cell. Dashes and blanks mean 'not polled'."""
    cleaned = clean_text(value)
    if not cleaned or cleaned in {'–', '-', '—', 'n/a', 'N/A', '?'}:
        return None
    cleaned = re.sub(r'[^\d.]', '', cleaned)
    if not cleaned or cleaned.count('.') > 1:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return val if 0 <= val <= 100 else None


def parse_date(date_str: str, election_year: int) -> Optional[str]:
    """Parse a Wikipedia date cell to ISO format, using the END of any range"""
    cleaned = clean_text(date_str)
    if not cleaned:
        return None

    patterns = [
        # 8–15 Oct 2020  /  30 Sep – 5 Oct 2020
        (r'(\d{1,2})\s*(?:\w+)?\s*[-–]\s*(\d{1,2})\s+(\w{3,})\s+(\d{4})',
         lambda m: (int(m.group(4)), m.group(3), int(m.group(2)))),
        # 26 Nov 1999
        (r'(\d{1,2})\s+(\w{3,})\s+(\d{4})',
         lambda m: (int(m.group(3)), m.group(2), int(m.group(1)))),
        # Nov 1999
        (r'(\w{3,})\s+(\d{4})',
         lambda m: (int(m.group(2)), m.group(1), 1)),
        # 8–15 Oct  (year implied by cycle)
        (r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s+(\w{3,})$',
         lambda m: (election_year, m.group(3), int(m.group(2)))),
        (r'(\d{1,2})\s+(\w{3,})$',
         lambda m: (election_year, m.group(2), int(m.group(1)))),
    ]

    for pattern, extract in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            continue
        year, month_str, day = extract(match)
        month = MONTHS.get(month_str.lower()[:3])
        if not month or not 1 <= day <= 31:
            continue
        return f"{year}-{month:02d}-{day:02d}"
    return None


def surname(full_name: str) -> str:
    """'Jacinda Ardern' -> 'Ardern'; already-short names pass through"""
    parts = clean_text(full_name).split()
    return parts[-1] if parts else full_name


def find_pm_table(soup: BeautifulSoup):
    """Return the wikitable sitting under the 'Preferred prime minister' h2"""
    for table in soup.find_all('table', class_='wikitable'):
        for heading in table.find_all_previous(['h2', 'h3', 'h4']):
            if heading.name == 'h2':
                if 'minister' in heading.get_text().lower():
                    return table
                break
    return None


def row_values(row) -> list:
    """Flatten a row's cells, expanding colspans"""
    values = []
    for cell in row.find_all(['td', 'th']):
        colspan = int(cell.get('colspan') or 1)
        values.extend([clean_text(cell.get_text(' '))] * colspan)
    return values


def parse_header(headers: list) -> tuple:
    """Map header cells to (date_col, pollster_col, sample_col, {col: surname})"""
    date_col = pollster_col = sample_col = None
    candidate_cols = {}

    for idx, header in enumerate(headers):
        key = header.lower()
        if 'date' in key:
            date_col = idx
        elif 'sample' in key:
            sample_col = idx
        elif key.startswith('poll'):
            pollster_col = idx
        elif key and key not in NON_CANDIDATE_HEADERS:
            candidate_cols[idx] = surname(header)

    return date_col, pollster_col, sample_col, candidate_cols


def is_header_row(row) -> bool:
    cells = row.find_all(['td', 'th'])
    return len(cells) > 2 and all(c.name == 'th' for c in cells)


def parse_pm_table(table, year: int) -> list:
    """Extract poll rows from a Preferred PM wikitable.

    Header blocks repeat mid-table and the candidate columns can change when a
    party swaps leader (e.g. Shearer → Cunliffe in the 2014 cycle), so the
    column mapping is re-read every time a header row appears.
    """
    rows = table.find_all('tr')
    headers = row_values(rows[0])
    date_col, pollster_col, sample_col, candidate_cols = parse_header(headers)

    if not candidate_cols:
        return []

    polls = []
    for row in rows[1:]:
        # Full-width spanner rows carry event annotations ("X is sworn in as the
        # 41st Prime Minister"); colspan expansion would smear them across every
        # candidate column, so drop them before parsing.
        if len(row.find_all(['td', 'th'])) < 3:
            continue
        values = row_values(row)
        if not any(values):
            continue
        if is_header_row(row):
            new_cols = parse_header(values)
            if new_cols[3]:
                date_col, pollster_col, sample_col, candidate_cols = new_cols
            continue
        if len(values) < len(headers) - 1:
            continue

        date = parse_date(values[date_col], year) if date_col is not None else None
        if not date:
            continue

        candidates = {}
        for idx, name in candidate_cols.items():
            if idx < len(values):
                pct = parse_percentage(values[idx])
                if pct is not None:
                    candidates[name] = pct
        if not candidates:
            continue

        sample = None
        if sample_col is not None and sample_col < len(values):
            digits = re.sub(r'[^\d]', '', values[sample_col])
            sample = int(digits) if digits else None

        polls.append({
            "date": date,
            "pollster": values[pollster_col] if pollster_col is not None and pollster_col < len(values) else None,
            "sample_size": sample,
            "candidates": candidates,
        })

    polls.sort(key=lambda p: p["date"])
    return polls


def scrape_year(year: int, session: requests.Session) -> dict:
    path = fetch(year, session)
    soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'lxml')

    table = find_pm_table(soup)
    if table is None:
        print(f"  {year}: no Preferred PM section")
        return {"election_year": year, "polls": [], "error": "No PM section"}

    polls = parse_pm_table(table, year)
    names = sorted({n for p in polls for n in p["candidates"]})
    print(f"  {year}: {len(polls)} polls, candidates: {', '.join(names)}")
    return {"election_year": year, "polls": polls}


def write_combined_csv(all_data: list, out_path: Path) -> int:
    """One row per (poll, candidate), deduplicated across overlapping cycles"""
    seen = set()
    rows = []
    for data in all_data:
        year = data["election_year"]
        for poll in data["polls"]:
            for name, pct in sorted(poll["candidates"].items()):
                key = (poll["date"], poll["pollster"], name)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "date": poll["date"],
                    "election_cycle": year,
                    "pollster": poll["pollster"],
                    "sample_size": poll["sample_size"],
                    "candidate": name,
                    "percent": pct,
                })

    rows.sort(key=lambda r: (r["date"], r["pollster"] or "", r["candidate"]))
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    print("NZ Preferred PM Polling Data Scraper")
    print("=" * 40)

    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_data = []

    for year in ELECTION_YEARS:
        data = scrape_year(year, session)
        all_data.append(data)
        with open(output_dir / f"{year}_pm_polling.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "pm_polling_all.csv"
    n_rows = write_combined_csv(all_data, csv_path)

    total = sum(len(d["polls"]) for d in all_data)
    dates = [p["date"] for d in all_data for p in d["polls"]]
    print("=" * 40)
    print(f"Total PM polls: {total} ({min(dates)} to {max(dates)})")
    print(f"Wrote {n_rows} poll-candidate rows to {csv_path}")


if __name__ == "__main__":
    main()
