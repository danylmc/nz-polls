#!/usr/bin/env python3
"""
NZ Preferred Prime Minister Polling Data Scraper

Scrapes "Preferred Prime Minister" polling data from Wikipedia for NZ elections.
Stores results separately from party vote polling data.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import requests
from bs4 import BeautifulSoup


# Election years to scrape
ELECTION_YEARS = [2008, 2011, 2014, 2017, 2020, 2023, 2026]

# Wikipedia URL pattern
URL_TEMPLATE = "https://en.wikipedia.org/wiki/Opinion_polling_for_the_{year}_New_Zealand_general_election"

# Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.5",
}

# Known PM candidates by era
PM_CANDIDATES = {
    # 2008 onwards - names that might appear in polling
    "Key": "John Key",
    "Clark": "Helen Clark",
    "English": "Bill English",
    "Goff": "Phil Goff",
    "Shearer": "David Shearer",
    "Cunliffe": "David Cunliffe",
    "Little": "Andrew Little",
    "Ardern": "Jacinda Ardern",
    "Hipkins": "Chris Hipkins",
    "Luxon": "Christopher Luxon",
    "Peters": "Winston Peters",
    "Seymour": "David Seymour",
    "Collins": "Judith Collins",
    "Bridges": "Simon Bridges",
    "Muller": "Todd Muller",
    "Davidson": "Marama Davidson",
    "Shaw": "James Shaw",
    "Norman": "Russel Norman",
    "Turei": "Metiria Turei",
}


def clean_text(text: str) -> str:
    """Remove footnotes and clean whitespace"""
    if not text:
        return ""
    text = re.sub(r'\[[\w\s]+\]', '', text)
    text = re.sub(r'[†‡§¶*]+', '', text)
    return ' '.join(text.split()).strip()


def parse_percentage(value: str) -> Optional[float]:
    """Parse percentage value"""
    if not value:
        return None
    cleaned = clean_text(value)
    cleaned = re.sub(r'[^\d.]', '', cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def month_to_num(month_str: str) -> int:
    """Convert month name to number"""
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }
    return months.get(month_str.lower()[:3], 1)


def parse_date(date_str: str, election_year: int) -> Optional[str]:
    """Parse date string to ISO format"""
    if not date_str:
        return None

    cleaned = clean_text(date_str)
    if not cleaned:
        return None

    patterns = [
        (r'(\d{1,2})[-–](\d{1,2})\s+(\w+)\s+(\d{4})', lambda m: f"{m.group(4)}-{month_to_num(m.group(3)):02d}-{int(m.group(2)):02d}"),
        (r'(\d{1,2})\s+(\w+)\s+(\d{4})', lambda m: f"{m.group(3)}-{month_to_num(m.group(2)):02d}-{int(m.group(1)):02d}"),
        (r'(\w+)\s+(\d{4})', lambda m: f"{m.group(2)}-{month_to_num(m.group(1)):02d}-01"),
        (r'(\d{1,2})[-–](\d{1,2})\s+(\w+)$', lambda m: f"{election_year}-{month_to_num(m.group(3)):02d}-{int(m.group(2)):02d}"),
        (r'(\d{1,2})\s+(\w+)$', lambda m: f"{election_year}-{month_to_num(m.group(2)):02d}-{int(m.group(1)):02d}"),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except (ValueError, KeyError):
                continue
    return None


def is_pm_name(text: str) -> bool:
    """Check if text appears to be a PM candidate name"""
    cleaned = clean_text(text).lower()
    # Check against known names
    for short_name in PM_CANDIDATES.keys():
        if short_name.lower() in cleaned:
            return True
    # Also check for generic patterns that look like names
    if cleaned and cleaned[0].isupper() and len(cleaned) > 2 and len(cleaned) < 20:
        # Exclude party names and common column headers
        excludes = ['national', 'labour', 'green', 'act', 'date', 'poll', 'sample', 'lead', 'others', 'undecided']
        if not any(exc in cleaned.lower() for exc in excludes):
            return True
    return False


def normalize_pm_name(text: str) -> str:
    """Normalize PM candidate name"""
    cleaned = clean_text(text)
    for short_name, full_name in PM_CANDIDATES.items():
        if short_name.lower() in cleaned.lower():
            return short_name
    return cleaned


def find_pm_polling_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Find the Preferred PM polling table"""
    # Look for all wikitables
    tables = soup.find_all('table', class_='wikitable')

    for table in tables:
        rows = table.find_all('tr')
        if not rows:
            continue

        # Get first row headers
        first_row = rows[0]
        cells = first_row.find_all(['th', 'td'])
        headers = [clean_text(c.get_text()) for c in cells]

        # Check if this looks like a PM polling table
        # It should have politician names as columns, not party names
        pm_name_count = sum(1 for h in headers if is_pm_name(h) and h.lower() not in ['national', 'labour', 'green', 'act', 'nz first'])
        party_name_count = sum(1 for h in headers if h.lower() in ['national', 'labour', 'green', 'act', 'nz first'])

        # PM table should have more PM names than party names
        if pm_name_count >= 2 and pm_name_count > party_name_count:
            return table

    return None


def scrape_pm_polling(year: int, session: requests.Session) -> dict:
    """Scrape PM polling data for a single election year"""
    url = URL_TEMPLATE.format(year=year)
    print(f"  Fetching {url}")

    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching {year}: {e}")
        return {"election_year": year, "polls": [], "error": str(e)}

    soup = BeautifulSoup(response.content, 'lxml')

    # Find PM polling table
    table = find_pm_polling_table(soup)
    if not table:
        print(f"  No PM polling table found for {year}")
        return {"election_year": year, "polls": [], "error": "No PM table found"}

    # Parse headers
    rows = table.find_all('tr')
    if not rows:
        return {"election_year": year, "polls": [], "error": "No rows found"}

    # Get headers
    header_row = rows[0]
    header_cells = header_row.find_all(['th', 'td'])
    headers = []
    for cell in header_cells:
        colspan = int(cell.get('colspan', 1))
        text = clean_text(cell.get_text())
        headers.extend([text] * colspan)

    # Identify PM columns
    pm_cols = {}
    date_col = None
    pollster_col = None

    for idx, header in enumerate(headers):
        header_lower = header.lower()
        if 'date' in header_lower:
            date_col = idx
        elif 'poll' in header_lower and 'sample' not in header_lower:
            pollster_col = idx
        elif is_pm_name(header) and header_lower not in ['national', 'labour', 'green', 'act']:
            pm_cols[idx] = normalize_pm_name(header)

    if not pm_cols:
        print(f"  No PM candidates found in headers for {year}")
        return {"election_year": year, "polls": [], "error": "No PM candidates in headers"}

    print(f"  Found PM candidates: {list(pm_cols.values())}")

    # Parse data rows
    polls = []
    for row in rows[1:]:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 3:
            continue

        cell_values = []
        for cell in cells:
            colspan = int(cell.get('colspan', 1))
            text = clean_text(cell.get_text())
            cell_values.extend([text] * colspan)

        if len(cell_values) < max(pm_cols.keys(), default=0) + 1:
            continue

        poll_data = {
            "date": None,
            "pollster": None,
            "candidates": {}
        }

        # Get date
        if date_col is not None and date_col < len(cell_values):
            poll_data["date"] = parse_date(cell_values[date_col], year)
        elif len(cell_values) > 0:
            poll_data["date"] = parse_date(cell_values[0], year)

        # Get pollster
        if pollster_col is not None and pollster_col < len(cell_values):
            poll_data["pollster"] = cell_values[pollster_col] if cell_values[pollster_col] else None

        # Get PM percentages
        for col_idx, pm_name in pm_cols.items():
            if col_idx < len(cell_values):
                pct = parse_percentage(cell_values[col_idx])
                if pct is not None and 0 <= pct <= 100:
                    poll_data["candidates"][pm_name] = pct

        # Only add if we have candidate data
        if poll_data["candidates"]:
            polls.append(poll_data)

    print(f"  Found {len(polls)} PM polls for {year}")
    return {"election_year": year, "polls": polls}


def main():
    """Main function"""
    print("NZ Preferred PM Polling Data Scraper")
    print("=" * 40)

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    session = requests.Session()

    for year in ELECTION_YEARS:
        print(f"\nScraping {year} PM polling data...")

        data = scrape_pm_polling(year, session)

        # Save to JSON
        output_file = output_dir / f"{year}_pm_polling.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {output_file}")
        time.sleep(1.5)

    print("\n" + "=" * 40)
    print("PM polling scraping complete!")

    # Print summary
    total_polls = 0
    for year in ELECTION_YEARS:
        output_file = output_dir / f"{year}_pm_polling.json"
        if output_file.exists():
            with open(output_file, 'r') as f:
                data = json.load(f)
                total_polls += len(data.get("polls", []))

    print(f"Total PM polls collected: {total_polls}")


if __name__ == "__main__":
    main()
