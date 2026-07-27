#!/usr/bin/env python3
"""
NZ Government "Right Direction / Wrong Direction" Approval Scraper

Scrapes the "Government approval rating" tables from the 2020 and 2023 NZ
general election Wikipedia pages. These are the only election pages with a
structured right-direction/wrong-direction table (checked 1993-2026; the
2026 page has a differently-shaped "Leadership approval rating" table
instead, and 1993-2017 pages have no such table at all).
"""

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

URL_TEMPLATE = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_{year}_New_Zealand_general_election"

YEARS_WITH_APPROVAL_TABLE = [2020, 2023]

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def parse_date(date_str: str, election_year: int) -> Optional[str]:
    """Parse dates like 'Sep 2020', '4 Sep - 8 Oct 2023', '28-30 Aug 2023'."""
    date_str = re.sub(r"\[.*?\]", "", date_str).strip()
    # Use the END of a range (most recent date in the field window)
    tail = date_str.split("–")[-1].split("-")[-1].strip()

    # 'Sep 2020' style
    m = re.match(r"^([A-Za-z]+)\s+(\d{4})$", tail)
    if m:
        mon, year = m.groups()
        month_num = MONTHS.get(mon[:3].lower())
        if month_num:
            return f"{year}-{month_num:02d}-01"

    # '8 Oct 2023' style
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", tail)
    if m:
        day, mon, year = m.groups()
        month_num = MONTHS.get(mon[:3].lower())
        if month_num:
            return f"{year}-{month_num:02d}-{int(day):02d}"

    # '30 Aug 2023' where tail lost month (e.g. '28-30 Aug 2023' -> tail '30 Aug 2023')
    return None


def clean_pollster(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"Archived.*?Wayback Machine", "", name)
    name = re.sub(r"permanent dead link", "", name)
    return name.strip() or None


def parse_number(value: str) -> Optional[float]:
    value = re.sub(r"\[.*?\]", "", value).strip().replace("%", "").replace(",", "")
    if value in ("", "-", "–", "—"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def scrape_approval_year(year: int, session: requests.Session) -> list:
    url = URL_TEMPLATE.format(year=year)
    print(f"  Fetching {url}")
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    heading = soup.find(["h2", "h3"], id="Government_approval_rating")
    if heading is None:
        print(f"  No 'Government approval rating' table found for {year}")
        return []

    table = heading.find_next("table")
    rows = table.find_all("tr")

    header_cells = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    col_idx = {name: i for i, name in enumerate(header_cells)}

    required = ["Date", "Right direction", "Wrong direction"]
    # header cells may have footnote markers e.g. 'Date[a]'
    def find_col(label):
        for name, i in col_idx.items():
            if name.startswith(label):
                return i
        return None

    date_i = find_col("Date")
    pollster_i = find_col("Polling organisation")
    sample_i = find_col("Sample")
    right_i = find_col("Right direction")
    wrong_i = find_col("Wrong direction")
    dk_i = find_col("Do not know")

    records = []
    for row in rows[1:]:
        cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        if len(cells) < len(header_cells):
            continue
        raw_date = cells[date_i] if date_i is not None else None
        date = parse_date(raw_date, year) if raw_date else None
        right = parse_number(cells[right_i]) if right_i is not None else None
        wrong = parse_number(cells[wrong_i]) if wrong_i is not None else None
        if date is None or right is None or wrong is None:
            continue
        records.append({
            "date": date,
            "election_year": year,
            "pollster": clean_pollster(cells[pollster_i]) if pollster_i is not None else None,
            "sample_size": parse_number(cells[sample_i]) if sample_i is not None else None,
            "right_direction": right,
            "wrong_direction": wrong,
            "dont_know": parse_number(cells[dk_i]) if dk_i is not None else None,
            "net": round(right - wrong, 2),
        })

    print(f"  Found {len(records)} approval readings for {year}")
    return records


def main():
    session = requests.Session()
    all_records = []
    for year in YEARS_WITH_APPROVAL_TABLE:
        print(f"Scraping government approval rating for {year}...")
        all_records.extend(scrape_approval_year(year, session))
        time.sleep(1.5)

    all_records.sort(key=lambda r: r["date"])

    output_dir = Path(__file__).parent / "data"
    output_file = output_dir / "govt_approval_right_wrong_track.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_records)} records to {output_file}")
    print("NOTE: Wikipedia only has this table for the 2020 and 2023 election pages.")
    print("No structured right/wrong-direction data exists for 1993-2017 or 2026.")


if __name__ == "__main__":
    main()
