#!/usr/bin/env python3
"""
NZ Economic Data Scraper

Fetches economic indicators for New Zealand from the World Bank API:
- GDP growth rate (annual %)
- Unemployment rate (%)
- CPI inflation (annual %)

Data is stored in data/economic/ as CSV files for use by analysis.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import requests


# World Bank API endpoints for NZ indicators
INDICATORS = {
    "gdp": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "name": "GDP Growth Rate",
        "description": "Annual GDP growth (%)",
    },
    "unemployment": {
        "code": "SL.UEM.TOTL.ZS",
        "name": "Unemployment Rate",
        "description": "Unemployment, total (% of labor force)",
    },
    "cpi": {
        "code": "FP.CPI.TOTL.ZG",
        "name": "CPI Inflation",
        "description": "Inflation, consumer prices (annual %)",
    },
}

API_BASE = "https://api.worldbank.org/v2/country/NZL/indicator/{indicator}?format=json&per_page=100"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NZPollingAnalysis/1.0)",
    "Accept": "application/json",
}


def fetch_indicator(indicator_code: str) -> List[Dict]:
    """Fetch data for a single indicator from World Bank API"""
    url = API_BASE.format(indicator=indicator_code)

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        # World Bank API returns [metadata, data_array]
        if len(data) < 2 or not data[1]:
            return []

        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "year": int(item["date"]),
                    "value": round(float(item["value"]), 2),
                })

        # Sort by year ascending
        records.sort(key=lambda x: x["year"])
        return records

    except requests.RequestException as e:
        print(f"  Error fetching {indicator_code}: {e}")
        return []
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Error parsing {indicator_code}: {e}")
        return []


def save_to_csv(data: List[Dict], filepath: Path, indicator_name: str) -> None:
    """Save indicator data to CSV file"""
    if not data:
        print(f"  No data to save for {indicator_name}")
        return

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "value"])
        writer.writeheader()
        writer.writerows(data)

    print(f"  Saved {len(data)} records to {filepath}")


def calculate_yoy_change(data: List[Dict]) -> List[Dict]:
    """Add year-over-year change to the data"""
    if len(data) < 2:
        return data

    result = []
    for i, record in enumerate(data):
        new_record = record.copy()
        if i > 0:
            prev_value = data[i - 1]["value"]
            new_record["yoy_change"] = round(record["value"] - prev_value, 2)
        else:
            new_record["yoy_change"] = None
        result.append(new_record)

    return result


def create_combined_dataset(output_dir: Path) -> None:
    """Create a combined CSV with all indicators aligned by year"""
    gdp_path = output_dir / "gdp.csv"
    unemployment_path = output_dir / "unemployment.csv"
    cpi_path = output_dir / "cpi.csv"

    # Read all data files
    data_by_year = {}

    for filepath, col_name in [
        (gdp_path, "gdp_growth"),
        (unemployment_path, "unemployment_rate"),
        (cpi_path, "cpi_inflation"),
    ]:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    year = int(row["year"])
                    if year not in data_by_year:
                        data_by_year[year] = {"year": year}
                    data_by_year[year][col_name] = float(row["value"])

    # Sort by year and write combined file
    combined = sorted(data_by_year.values(), key=lambda x: x["year"])

    if combined:
        combined_path = output_dir / "economic_combined.csv"
        fieldnames = ["year", "gdp_growth", "unemployment_rate", "cpi_inflation"]
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in combined:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        print(f"\n  Created combined dataset: {combined_path}")


def get_economic_data_for_year(year: int) -> Optional[Dict]:
    """Get economic data for a specific year (for use by analysis.py)"""
    data_dir = Path(__file__).parent / "data" / "economic"
    combined_path = data_dir / "economic_combined.csv"

    if not combined_path.exists():
        return None

    with open(combined_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["year"]) == year:
                return {
                    "year": year,
                    "gdp_growth": float(row["gdp_growth"]) if row.get("gdp_growth") else None,
                    "unemployment_rate": float(row["unemployment_rate"]) if row.get("unemployment_rate") else None,
                    "cpi_inflation": float(row["cpi_inflation"]) if row.get("cpi_inflation") else None,
                }
    return None


def load_economic_data() -> List[Dict]:
    """Load all economic data from combined CSV (for use by analysis.py)"""
    data_dir = Path(__file__).parent / "data" / "economic"
    combined_path = data_dir / "economic_combined.csv"

    if not combined_path.exists():
        return []

    with open(combined_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        result = []
        for row in reader:
            result.append({
                "year": int(row["year"]),
                "gdp_growth": float(row["gdp_growth"]) if row.get("gdp_growth") else None,
                "unemployment_rate": float(row["unemployment_rate"]) if row.get("unemployment_rate") else None,
                "cpi_inflation": float(row["cpi_inflation"]) if row.get("cpi_inflation") else None,
            })
        return result


def main():
    """Main function to scrape all economic indicators"""
    print("NZ Economic Data Scraper")
    print("=" * 40)
    print("Data source: World Bank Open Data API\n")

    # Create output directory
    output_dir = Path(__file__).parent / "data" / "economic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch each indicator
    for key, info in INDICATORS.items():
        print(f"Fetching {info['name']} ({info['code']})...")
        data = fetch_indicator(info["code"])

        if data:
            # Add YoY change
            data_with_change = calculate_yoy_change(data)

            # Save to CSV
            filepath = output_dir / f"{key}.csv"
            save_to_csv(data, filepath, info["name"])

    # Create combined dataset
    create_combined_dataset(output_dir)

    print("\n" + "=" * 40)
    print("Economic data scraping complete!")

    # Print summary statistics
    print("\nSummary (1993-2024, NZ election period):")
    data = load_economic_data()
    election_data = [d for d in data if 1993 <= d["year"] <= 2024]

    if election_data:
        gdp_values = [d["gdp_growth"] for d in election_data if d["gdp_growth"] is not None]
        unemp_values = [d["unemployment_rate"] for d in election_data if d["unemployment_rate"] is not None]
        cpi_values = [d["cpi_inflation"] for d in election_data if d["cpi_inflation"] is not None]

        if gdp_values:
            print(f"  GDP Growth: {min(gdp_values):.1f}% to {max(gdp_values):.1f}% (avg: {sum(gdp_values)/len(gdp_values):.1f}%)")
        if unemp_values:
            print(f"  Unemployment: {min(unemp_values):.1f}% to {max(unemp_values):.1f}% (avg: {sum(unemp_values)/len(unemp_values):.1f}%)")
        if cpi_values:
            print(f"  CPI Inflation: {min(cpi_values):.1f}% to {max(cpi_values):.1f}% (avg: {sum(cpi_values)/len(cpi_values):.1f}%)")


if __name__ == "__main__":
    main()
