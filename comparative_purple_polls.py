#!/usr/bin/env python3
"""Collect 2026 traditional-party poll shares from Wikipedia polling tables.

NZ polls are read from the project's existing Wikipedia-derived local JSON.
Peer tables are fetched through Wikimedia's parse API. Run again to refresh.
"""

import json
import re
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
AS_OF = date(2026, 10, 6)
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "nz-polls-comparative-research/1.0 (https://github.com/)"}
PAGES = {
    "United Kingdom": ("Opinion polling for the next United Kingdom general election", 0),
    "Australia": ("Opinion polling for the next Australian federal election", 0),
    "Canada": ("Opinion polling for the 46th Canadian federal election", 0),
    "Ireland": ("Next Irish general election", 2),
    "Germany": ("Opinion polling for the next German federal election", 0),
    "Austria": ("Next Austrian legislative election", 0),
    "Spain": ("Opinion polling for the 2026 Spanish general election", 0),
    "Netherlands": ("Next Dutch general election", 2),
}
COUNTRY_ORDER = ["New Zealand", "Australia", "United Kingdom", "Canada", "Ireland", "Germany", "Austria", "Spain", "Netherlands"]
COLORS = {"New Zealand": "#5b3c88", "Australia": "#db8b34", "United Kingdom": "#bc5550", "Canada": "#5486ae", "Ireland": "#4e9b75", "Germany": "#735870", "Austria": "#a57c44", "Spain": "#cf6269", "Netherlands": "#688985"}


def clean(value):
    return re.sub(r"\s*\[[^]]+\]", "", str(value)).strip()


def pct(value):
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", clean(value))
    return float(match.group(1)) if match else None


def leading_pct(value):
    """Spain prints a vote estimate followed by a seat estimate in each cell."""
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\b", clean(value))
    return float(match.group(1)) if match else None


def date_end(value, year=2026):
    value = clean(value).replace("Sept", "Sep")
    us_style = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(20\d{2})\b", value, re.I)
    if us_style:
        try:
            return pd.Timestamp(f"{us_style.group(2)} {us_style.group(1)} {us_style.group(3)}").date()
        except ValueError:
            return None
    # Table dates can be ranges with one or two month names. Use the final day.
    matches = re.findall(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b(?:\s*,?\s*(20\d{2}))?", value, re.I)
    if not matches:
        return None
    day, month, explicit_year = matches[-1]
    try:
        return pd.Timestamp(f"{day} {month} {explicit_year or year}").date()
    except ValueError:
        return None


def wiki_table(page, index, session):
    response = session.get(API, params={"action": "parse", "page": page, "prop": "text", "format": "json", "formatversion": "2", "redirects": "1"}, headers=HEADERS, timeout=40)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"{page}: {payload['error']}")
    tables = BeautifulSoup(payload["parse"]["text"], "html.parser").select("table.wikitable")
    return tables[index], f"https://en.wikipedia.org/wiki/{page.replace(' ', '_')}"


def cells(row):
    return [clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"], recursive=False)]


def record(country, day, pollster, red, blue, source, sample=None):
    if day is None or not date(2026, 1, 1) <= day <= AS_OF:
        return None
    if red is None or blue is None or red + blue > 100:
        return None
    return {"country": country, "date": day.isoformat(), "pollster": pollster, "red_pct": red, "blue_pct": blue, "purple_pct": red + blue, "sample_size": sample, "source_url": source}


def parse_table(country, table, source):
    rows = []
    if country == "Australia":
        # The 2026 table has five fixed columns, then ALP, two L/NP columns
        # (sometimes merged), followed by Green, One Nation and others.
        for tr in table.select("tr"):
            c = cells(tr)
            if len(c) < 9:
                continue
            day = date_end(c[0])
            red = pct(c[5])
            if red is None:
                continue
            party_cells = tr.find_all(["th", "td"], recursive=False)
            if len(party_cells) < 7:
                continue
            first_blue = pct(c[6])
            if first_blue is None:
                continue
            # A colspan=2 cell holds the combined Coalition primary vote.
            blue = first_blue if party_cells[6].get("colspan") == "2" else first_blue + (pct(c[7]) or 0)
            row = record(country, day, c[1], red, blue, source, c[4])
            if row:
                rows.append(row)
    else:
        for tr in table.select("tr"):
            c = cells(tr)
            if country == "United Kingdom" and len(c) >= 7:
                row = record(country, date_end(c[0]), c[1], pct(c[5]), pct(c[6]), source, c[4])
            elif country == "Canada" and len(c) >= 5:
                row = record(country, date_end(c[1]), c[0], pct(c[3]), pct(c[4]), source, c[11] if len(c) > 11 else None)
            elif country == "Ireland" and len(c) >= 8:
                row = record(country, date_end(c[0]), c[2], pct(c[6]), pct(c[7]), source, c[3])
            elif country == "Germany" and len(c) >= 7:
                row = record(country, date_end(c[1]), c[0], pct(c[6]), pct(c[4]), source, c[2])
            elif country == "Austria" and len(c) >= 7:
                row = record(country, date_end(c[1]), c[0], pct(c[6]), pct(c[5]), source, c[2])
            elif country == "Spain" and len(c) >= 6:
                row = record(country, date_end(c[1]), c[0], leading_pct(c[5]), leading_pct(c[4]), source, c[2])
            elif country == "Netherlands" and len(c) >= 7:
                row = record(country, date_end(c[1]), c[0], pct(c[6]), pct(c[5]), source, c[2])
            else:
                continue
            if row:
                rows.append(row)
    return rows


def nz_polls():
    source = ROOT / "data/2026_polling.json"
    payload = json.loads(source.read_text())
    rows = []
    for poll in payload["polls"]:
        parties = poll.get("parties", {})
        row = record("New Zealand", date.fromisoformat(poll["date"]), poll.get("pollster"), parties.get("Labour"), parties.get("National"), payload["source"], poll.get("sample_size"))
        if row:
            rows.append(row)
    return rows


def make_chart(polls):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for country in COUNTRY_ORDER:
        subset = polls[polls.country == country].sort_values("date")
        if subset.empty:
            continue
        ax.scatter(subset.date, subset.purple_pct, s=9, alpha=0.15, color=COLORS[country])
        series = subset.set_index("date").purple_pct.resample("D").mean()
        # Each plotted day uses polls whose fieldwork ended in the last 45 days.
        trend = series.rolling("45D", min_periods=2).mean()
        ax.plot(trend.index, trend, lw=2.2, label=f"{country} (n={len(subset)})", color=COLORS[country])
    ax.set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp(AS_OF))
    ax.set_ylim(20, 95)
    ax.set_ylabel("Combined primary vote / voting intention (%)")
    ax.set_title("Traditional governing party pairs in 2026 polls")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Dots: individual polls. Lines: trailing 45-day mean. Ireland: FF+FG; Netherlands: PRO+VVD. Source: Wikipedia polling tables; NZ local data.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "reports/purple_vote_2026_trends.png"
    fig.savefig(out, dpi=180)
    print(f"Saved {out}")


def main():
    rows = nz_polls()
    with requests.Session() as session:
        for country, (page, index) in PAGES.items():
            table, source = wiki_table(page, index, session)
            found = parse_table(country, table, source)
            if not found:
                raise ValueError(f"No 2026 polls parsed for {country}; table layout may have changed")
            rows.extend(found)
            print(f"{country}: {len(found)} rows")
    polls = pd.DataFrame(rows).drop_duplicates(subset=["country", "date", "pollster", "red_pct", "blue_pct"])
    polls = polls.sort_values(["country", "date", "pollster"])
    polls["date"] = pd.to_datetime(polls["date"])
    out = ROOT / "data/purple_vote_polls_2026.csv"
    polls.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"Saved {out}: {len(polls)} polls")
    make_chart(polls)
    cutoff = pd.Timestamp(AS_OF) - pd.Timedelta(days=45)
    for country in COUNTRY_ORDER:
        recent = polls[(polls.country == country) & (polls.date >= cutoff)]
        print(f"{country}: recent n={len(recent)}, mean={recent.purple_pct.mean():.1f}, median={recent.purple_pct.median():.1f}, latest={recent.date.max().date()}")


if __name__ == "__main__":
    main()
