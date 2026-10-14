#!/usr/bin/env python3
"""
Fetch Stats NZ household disposable income, CPI and population series.

Downloads the release CSVs with Camoufox (stats.govt.nz blocks plain requests)
into raw_data/stats_nz/, then extracts the series used by
household_income_analysis.py into data/household_disposable_income.csv.

Run with the Camoufox venv:
    /mnt/d/data/.venv/bin/python household_income_fetch.py

Series (all quarterly, seasonally adjusted):
- Household net disposable income, $m nominal (SNEQ.SG04SB06NR00S500)
  National accounts (income, saving, assets, and liabilities), experimental, 2016Q2-
- CPI all groups (CPIQ.SE9A)
- GDP(E) nominal (SNEQ.SG02NSC00B15) / GDP(E) nominal per capita
  (SNEQ.SG09NSC00B15NZ) -> implied population
- Real gross national disposable income per capita, chain volume
  (SNEQ.SG09RSC00B06NZ), 1991Q1- : a long national (not household) proxy
"""

import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW = BASE / "raw_data" / "stats_nz"
OUT = BASE / "data" / "household_disposable_income.csv"

RELEASES = {
    "hioa": "https://www.stats.govt.nz/experimental/national-accounts-income-saving-assets-and-liabilities-june-2026-quarter/",
    "cpi": "https://www.stats.govt.nz/information-releases/consumers-price-index-june-2026-quarter/",
    "gdp": "https://www.stats.govt.nz/information-releases/gross-domestic-product-june-2026-quarter/",
}
FILES = {
    "hioa": "na-isal-june-2026-quarter-institutional-sector-accounts.csv",
    "cpi": "consumers-price-index-june-2026-quarter-index-numbers.csv",
    "gdp": "gross-domestic-product-june-2026-quarter.csv",
}


def download():
    from camoufox.sync_api import Camoufox

    RAW.mkdir(parents=True, exist_ok=True)
    with Camoufox(headless=True) as b:
        pg = b.new_page()
        for key, url in RELEASES.items():
            if (RAW / FILES[key]).exists():
                continue
            pg.goto(url, timeout=60000)
            pg.wait_for_timeout(6000)
            hrefs = [a.get_attribute("href") or "" for a in pg.query_selector_all("a")]
            match = [h for h in hrefs if h.split("?")[0].endswith(FILES[key])]
            if not match:
                sys.exit(f"{FILES[key]} not linked from {url}")
            href = match[0] if match[0].startswith("http") else "https://www.stats.govt.nz" + match[0]
            r = pg.request.get(href, timeout=180000)
            (RAW / FILES[key]).write_bytes(r.body())
            print(f"saved {FILES[key]} ({r.status}, {len(r.body()):,} bytes)")


def period_to_quarter(p):
    year, month = str(p).split(".")
    return pd.Period(f"{year}-{int(month):02d}", freq="M").asfreq("Q")


def series(path, ref, ref_col, period_col, value_col):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    s = df[df[ref_col] == ref][[period_col, value_col]].copy()
    s[value_col] = pd.to_numeric(s[value_col], errors="coerce")
    s.index = s[period_col].map(period_to_quarter)
    return s[value_col].dropna()


def extract():
    hh = series(RAW / FILES["hioa"], "SNEQ.SG04SB06NR00S500", "Series_reference", "period", "value")
    cpi = series(RAW / FILES["cpi"], "CPIQ.SE9A", "Series_reference", "Period", "Data_value")
    gdp = RAW / FILES["gdp"]
    gdp_nom = series(gdp, "SNEQ.SG02NSC00B15", "Series_reference", "Period", "Data_value")
    gdp_pc = series(gdp, "SNEQ.SG09NSC00B15NZ", "Series_reference", "Period", "Data_value")
    rgndi_pc = series(gdp, "SNEQ.SG09RSC00B06NZ", "Series_reference", "Period", "Data_value")

    df = pd.DataFrame({
        "hh_ndi_nominal_m": hh,
        "cpi": cpi,
        "population": gdp_nom * 1e6 / gdp_pc,
        "rgndi_pc": rgndi_pc,
    })
    df = df[df.index >= pd.Period("1991Q1")]
    # Real household net disposable income per person, June 2026 dollars, quarterly
    cpi_base = df["cpi"].loc[pd.Period("2026Q2")]
    df["real_hh_ndi_pc"] = df["hh_ndi_nominal_m"] * 1e6 / df["population"] * cpi_base / df["cpi"]
    df.index = df.index.to_timestamp()
    df.index.name = "quarter"
    df.round({"population": 0, "real_hh_ndi_pc": 0}).to_csv(OUT)
    print(f"wrote {OUT} ({len(df)} quarters; household series {hh.index.min()}–{hh.index.max()})")


if __name__ == "__main__":
    download()
    extract()
