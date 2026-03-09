#!/usr/bin/env python3
"""Download comparative datasets for political theory analysis."""

import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_oecd_price_to_income():
    """Download OECD Price-to-Income Ratios."""
    print("\n[1/5] Downloading OECD Price-to-Income Ratios...")
    # Dimension order: REF_AREA.FREQ.MEASURE.UNIT_MEASURE
    # HPI_YDH = Price to income ratio (confirmed via NZL test query)
    url = "https://sdmx.oecd.org/public/rest/data/OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,/.A..?format=csvfilewithlabels"
    out = DATA_DIR / "oecd_price_to_income.csv"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        df = pd.read_csv(out)
        nz = df[df["REF_AREA"] == "NZL"] if "REF_AREA" in df.columns else pd.DataFrame()
        print(f"  OK: {len(df)} rows, {len(nz)} NZL rows")
        print(f"  Columns: {list(df.columns[:8])}")
        if len(nz):
            print(f"  NZ years: {sorted(nz['TIME_PERIOD'].unique())[:5]}...{sorted(nz['TIME_PERIOD'].unique())[-3:]}")
    except Exception as e:
        print(f"  FAILED: {e}")


def download_wgi_governance():
    """Download World Governance Indicators from World Bank API."""
    print("\n[2/5] Downloading World Governance Indicators...")
    indicators = {
        "GE.EST": "Government Effectiveness",
        "RL.EST": "Rule of Law",
        "VA.EST": "Voice and Accountability",
        "RQ.EST": "Regulatory Quality",
        "CC.EST": "Control of Corruption",
        "PV.EST": "Political Stability",
    }
    out = DATA_DIR / "wgi_governance.json"
    results = {}
    for code, name in indicators.items():
        print(f"  Fetching {code} ({name})...")
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            results[code] = data
            # Count NZ records
            if isinstance(data, list) and len(data) > 1:
                nz_records = [rec for rec in data[1] if rec.get("countryiso3code") == "NZL"]
                print(f"    OK: {len(data[1])} total records, {len(nz_records)} NZL")
            else:
                print(f"    OK: response received (unexpected format)")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[code] = None
        time.sleep(1)
    out.write_text(json.dumps(results, indent=2))
    print(f"  Saved to {out}")


def download_oecd_labour_productivity():
    """Download OECD Labour Productivity data."""
    print("\n[3/5] Downloading OECD Labour Productivity...")
    # 9 dimensions: REF_AREA.FREQ.MEASURE.ACTIVITY.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION.ADJUSTMENT.CONVERSION_TYPE
    url = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PDB@DF_PDB_LV,/.A.GDPHRS......?format=csvfilewithlabels"
    out = DATA_DIR / "oecd_labour_productivity.csv"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        df = pd.read_csv(out)
        nz = df[df["REF_AREA"] == "NZL"] if "REF_AREA" in df.columns else pd.DataFrame()
        print(f"  OK: {len(df)} rows, {len(nz)} NZL rows")
        print(f"  Columns: {list(df.columns[:8])}")
        if len(nz):
            print(f"  NZ years: {sorted(nz['TIME_PERIOD'].unique())[:5]}...{sorted(nz['TIME_PERIOD'].unique())[-3:]}")
    except Exception as e:
        print(f"  FAILED: {e}")


def download_oecd_trust():
    """Download OECD Trust in Government data."""
    print("\n[4/5] Downloading OECD Trust in Government...")
    url = "https://sdmx.oecd.org/public/rest/data/OECD.GOV.GIP,DSD_GOV@DF_GOV_2023,/all?format=csvfilewithlabels"
    out = DATA_DIR / "oecd_trust_govt.csv"
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        df = pd.read_csv(out)
        nz = df[df["REF_AREA"] == "NZL"] if "REF_AREA" in df.columns else pd.DataFrame()
        print(f"  OK: {len(df)} rows, {len(nz)} NZL rows")
        print(f"  Columns: {list(df.columns[:8])}")
    except Exception as e:
        print(f"  FAILED: {e}")


def download_freedom_house():
    """Download Freedom House country scores."""
    print("\n[5/5] Downloading Freedom House Scores...")
    url = "https://freedomhouse.org/sites/default/files/2025-02/Country_and_Territory_Ratings_and_Statuses_FIW_1973-2024.xlsx"
    out = DATA_DIR / "freedom_house_scores.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        xl = pd.ExcelFile(out, engine="openpyxl")
        print(f"  Sheets: {xl.sheet_names}")
        # Read the 'Country Ratings, Statuses' sheet (main data)
        sheet_name = [s for s in xl.sheet_names if "Country" in s][0]
        df = pd.read_excel(out, engine="openpyxl", sheet_name=sheet_name, header=None)
        print(f"  OK: {len(df)} rows, {len(df.columns)} columns (sheet: {sheet_name})")
        # Look for NZ in any column
        for col in df.columns:
            nz = df[df[col].astype(str).str.contains("New Zealand", case=False, na=False)]
            if len(nz):
                print(f"  Found {len(nz)} NZ rows in column index {col}")
                print(f"  Sample NZ data: {list(nz.iloc[0, :10].values)}")
                break
    except Exception as e:
        print(f"  FAILED: {e}")


def main():
    print("=" * 60)
    print("Downloading comparative datasets for NZ political analysis")
    print("=" * 60)

    download_oecd_price_to_income()
    time.sleep(1)
    download_wgi_governance()
    time.sleep(1)
    download_oecd_labour_productivity()
    time.sleep(1)
    download_oecd_trust()
    time.sleep(1)
    download_freedom_house()

    # Final verification
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    files = [
        ("oecd_price_to_income.csv", "OECD Price-to-Income"),
        ("wgi_governance.json", "World Governance Indicators"),
        ("oecd_labour_productivity.csv", "OECD Labour Productivity"),
        ("oecd_trust_govt.csv", "OECD Trust in Government"),
        ("freedom_house_scores.xlsx", "Freedom House Scores"),
    ]
    for fname, label in files:
        path = DATA_DIR / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  OK  {label}: {fname} ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING  {label}: {fname}")


if __name__ == "__main__":
    main()
