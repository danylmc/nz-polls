#!/usr/bin/env python3
"""
Comparative political/economic statistics for NZ briefing document.
Reads OECD, World Bank, and Freedom House data; prints formatted stats.
"""

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

OECD = {
    "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TUR", "GBR", "USA",
}

WGI_LABELS = {
    "GE.EST": "Government Effectiveness",
    "RL.EST": "Rule of Law",
    "VA.EST": "Voice and Accountability",
    "RQ.EST": "Regulatory Quality",
    "CC.EST": "Control of Corruption",
    "PV.EST": "Political Stability",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> str:
    return f"\n{'=' * 72}\n  {title}\n{'=' * 72}\n"


def country_name_map(df: pd.DataFrame) -> dict:
    """Build REF_AREA -> 'Reference area' name map from an OECD CSV."""
    return dict(df.groupby("REF_AREA")["Reference area"].first())


# ---------------------------------------------------------------------------
# 1. Price-to-Income
# ---------------------------------------------------------------------------

def price_to_income() -> str:
    out = [section("PRICE-TO-INCOME RATIO (OECD, index 2015=100)")]

    df = pd.read_csv(DATA_DIR / "oecd_price_to_income.csv")
    pti = df[df["MEASURE"] == "HPI_YDH"].copy()
    pti["TIME_PERIOD"] = pti["TIME_PERIOD"].astype(int)
    pti["OBS_VALUE"] = pd.to_numeric(pti["OBS_VALUE"], errors="coerce")
    pti = pti.dropna(subset=["OBS_VALUE"])
    names = country_name_map(pti)

    # Latest year available for NZ
    nz = pti[pti["REF_AREA"] == "NZL"]
    latest_yr = int(nz["TIME_PERIOD"].max())
    nz_latest = float(nz.loc[nz["TIME_PERIOD"] == latest_yr, "OBS_VALUE"].iloc[0])

    out.append(f"NZ value ({latest_yr}): {nz_latest:.1f}")
    base = nz.loc[nz["TIME_PERIOD"] == 2015, "OBS_VALUE"]
    if len(base):
        out.append(f"NZ value (2015, base): {float(base.iloc[0]):.1f}")

    # OECD ranking (latest year)
    oecd_latest = (
        pti[(pti["REF_AREA"].isin(OECD)) & (pti["TIME_PERIOD"] == latest_yr)]
        .groupby("REF_AREA")["OBS_VALUE"].first()
        .sort_values(ascending=False)
    )
    rank = list(oecd_latest.index).index("NZL") + 1
    out.append(f"NZ OECD rank ({latest_yr}): {rank} of {len(oecd_latest)} "
               f"(1 = most expensive)")
    out.append(f"OECD median: {oecd_latest.median():.1f}")
    out.append(f"OECD mean:   {oecd_latest.mean():.1f}")

    # Trajectory
    traj_years = [y for y in [2000, 2005, 2010, 2015, 2020, latest_yr]
                  if y in nz["TIME_PERIOD"].values]
    out.append("\nNZ trajectory:")
    for y in traj_years:
        v = float(nz.loc[nz["TIME_PERIOD"] == y, "OBS_VALUE"].iloc[0])
        out.append(f"  {y}: {v:.1f}")

    # Top 5 / Bottom 5
    out.append(f"\nTop 5 OECD ({latest_yr}, most expensive):")
    for code, val in oecd_latest.head(5).items():
        name = names.get(code, code)
        out.append(f"  {name:25s} {val:.1f}")
    out.append(f"\nBottom 5 OECD ({latest_yr}, most affordable):")
    for code, val in oecd_latest.tail(5).items():
        name = names.get(code, code)
        out.append(f"  {name:25s} {val:.1f}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. WGI Governance
# ---------------------------------------------------------------------------

def wgi_governance() -> str:
    out = [section("WORLDWIDE GOVERNANCE INDICATORS (World Bank)")]

    with open(DATA_DIR / "wgi_governance.json") as f:
        raw = json.load(f)

    # Build a tidy dataframe from all indicators
    rows = []
    for code, payload in raw.items():
        # payload is [metadata_dict, records_list]
        records = payload[1]
        for r in records:
            val = r.get("value")
            if val is None:
                continue
            rows.append({
                "indicator": code,
                "country": r["countryiso3code"],
                "year": int(r["date"]),
                "value": float(val),
            })
    df = pd.DataFrame(rows)

    # Latest year with NZ data
    nz = df[df["country"] == "NZL"]
    latest_yr = int(nz["year"].max())

    out.append(f"Latest year: {latest_yr}\n")
    out.append(f"{'Dimension':<30s} {'NZ':>6s} {'OECD mean':>10s} {'OECD rank':>10s}")
    out.append("-" * 60)

    nz_scores = {}
    for ind_code, ind_label in WGI_LABELS.items():
        nz_val = nz.loc[(nz["indicator"] == ind_code) & (nz["year"] == latest_yr), "value"]
        if nz_val.empty:
            continue
        nz_v = float(nz_val.iloc[0])
        nz_scores[ind_code] = nz_v

        oecd_vals = (
            df[(df["indicator"] == ind_code) &
               (df["country"].isin(OECD)) &
               (df["year"] == latest_yr)]
            .groupby("country")["value"].first()
            .sort_values(ascending=False)
        )
        oecd_mean = oecd_vals.mean()
        rank = list(oecd_vals.index).index("NZL") + 1
        out.append(f"{ind_label:<30s} {nz_v:>6.2f} {oecd_mean:>10.2f} {rank:>5d}/{len(oecd_vals)}")

    # Strongest / weakest
    best = max(nz_scores, key=nz_scores.get)
    worst = min(nz_scores, key=nz_scores.get)
    out.append(f"\nStrongest: {WGI_LABELS[best]} ({nz_scores[best]:.2f})")
    out.append(f"Weakest:   {WGI_LABELS[worst]} ({nz_scores[worst]:.2f})")

    # NZ trajectory
    traj_years = [1996, 2002, 2010, 2015, 2020, latest_yr]
    out.append(f"\nNZ trajectory (all 6 dimensions, mean score):")
    for y in traj_years:
        vals = nz.loc[nz["year"] == y, "value"]
        if vals.empty:
            continue
        out.append(f"  {y}: {vals.mean():.2f}  (n={len(vals)} indicators)")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. Labour Productivity
# ---------------------------------------------------------------------------

def labour_productivity() -> str:
    out = [section("LABOUR PRODUCTIVITY (OECD, GDP per hour worked, USD PPP)")]

    df = pd.read_csv(DATA_DIR / "oecd_labour_productivity.csv")
    # USD PPP only (not local currency), current prices (not constant)
    df = df[(df["UNIT_MEASURE"] == "USD_PPP_H") & (df["PRICE_BASE"] == "V")]
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(int)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE"])
    names = country_name_map(df)

    nz = df[df["REF_AREA"] == "NZL"]
    latest_yr = int(nz["TIME_PERIOD"].max())
    nz_latest = float(nz.loc[nz["TIME_PERIOD"] == latest_yr, "OBS_VALUE"].iloc[0])

    out.append(f"NZ GDP/hour ({latest_yr}): ${nz_latest:.1f}")

    # OECD ranking
    oecd_latest = (
        df[(df["REF_AREA"].isin(OECD)) & (df["TIME_PERIOD"] == latest_yr)]
        .groupby("REF_AREA")["OBS_VALUE"].first()
        .sort_values(ascending=False)
    )
    rank = list(oecd_latest.index).index("NZL") + 1
    oecd_med = oecd_latest.median()
    oecd_mean = oecd_latest.mean()

    out.append(f"NZ OECD rank ({latest_yr}): {rank} of {len(oecd_latest)} "
               f"(1 = highest)")
    out.append(f"OECD median: ${oecd_med:.1f}")
    out.append(f"OECD mean:   ${oecd_mean:.1f}")
    out.append(f"NZ as % of OECD average: {100 * nz_latest / oecd_mean:.1f}%")

    # Trajectory
    traj_years = [y for y in [2000, 2005, 2010, 2015, 2020, latest_yr]
                  if y in nz["TIME_PERIOD"].values]
    out.append("\nNZ trajectory:")
    for y in traj_years:
        v = float(nz.loc[nz["TIME_PERIOD"] == y, "OBS_VALUE"].iloc[0])
        out.append(f"  {y}: ${v:.1f}")

    # Top 5
    out.append(f"\nTop 5 OECD ({latest_yr}):")
    for code, val in oecd_latest.head(5).items():
        name = names.get(code, code)
        out.append(f"  {name:25s} ${val:.1f}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. Trust in Government
# ---------------------------------------------------------------------------

def trust_in_government() -> str:
    out = [section("TRUST IN GOVERNMENT")]
    out.append("NZ 46% trust in national government (OECD 39% average)")
    out.append("  Source: OECD Government at a Glance 2023")
    out.append("")
    out.append("Note: Ipsos NZ Issues Monitor shows govt approval dropped to")
    out.append("3.9/10 (Oct 2025, record low). Trust != approval — trust is a")
    out.append("broader institutional measure that changes slowly; approval is")
    out.append("a performance rating that responds to current conditions.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. Freedom House
# ---------------------------------------------------------------------------

def freedom_house() -> str:
    out = [section("FREEDOM HOUSE RATINGS")]

    wb = openpyxl.load_workbook(
        DATA_DIR / "freedom_house_scores.xlsx", read_only=True
    )
    ws = wb["Country Ratings, Statuses "]

    # Row 1 (idx 0): year headers (survey edition labels)
    # Row 2 (idx 1): year under review
    # Row 3 (idx 2): PR / CL / Status column labels
    all_rows = list(ws.iter_rows(values_only=True))
    wb.close()

    year_row = all_rows[1]  # "Year(s) Under Review"
    # Extract years from positions 1, 4, 7, ... (every 3rd starting at 1)
    years = []
    for i in range(1, len(year_row), 3):
        y = year_row[i]
        if y is not None:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                # Some entries are date ranges like 'Jan.1981-Aug. 1982'
                import re
                m = re.search(r"(\d{4})", str(y))
                years.append(int(m.group(1)) if m else None)

    # Find NZ row
    nz_row = None
    for row in all_rows[3:]:
        if row[0] and "New Zealand" in str(row[0]):
            nz_row = row
            break

    if nz_row is None:
        out.append("ERROR: New Zealand not found in Freedom House data")
        return "\n".join(out)

    # Parse PR, CL, Status for each year
    pr_vals, cl_vals, statuses = [], [], []
    for i in range(len(years)):
        base = 1 + i * 3
        pr = nz_row[base] if base < len(nz_row) else None
        cl = nz_row[base + 1] if base + 1 < len(nz_row) else None
        st = nz_row[base + 2] if base + 2 < len(nz_row) else None
        pr_vals.append(pr)
        cl_vals.append(cl)
        statuses.append(st)

    # Check consistency
    all_free = all(s == "F" for s in statuses if s)
    all_pr1 = all(p == 1 for p in pr_vals if p is not None)
    all_cl1 = all(c == 1 for c in cl_vals if c is not None)

    out.append(f"Years covered: {min(years)}–{max(years)} ({len(years)} years)")

    if all_free and all_pr1 and all_cl1:
        out.append(f"NZ: consistently Free since {min(years)}, PR=1, CL=1 in all years")
    else:
        out.append("NZ ratings:")
        for y, pr, cl, st in zip(years, pr_vals, cl_vals, statuses):
            if pr != 1 or cl != 1 or st != "F":
                out.append(f"  {y}: PR={pr}, CL={cl}, Status={st}")

    out.append("")
    out.append("PR = Political Rights (1=most free, 7=least free)")
    out.append("CL = Civil Liberties (1=most free, 7=least free)")
    out.append("F = Free, PF = Partly Free, NF = Not Free")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sections = [
        price_to_income(),
        wgi_governance(),
        labour_productivity(),
        trust_in_government(),
        freedom_house(),
    ]
    output = "\n".join(sections) + "\n"

    print(output)

    REPORTS_DIR.mkdir(exist_ok=True)
    outfile = REPORTS_DIR / "comparative_stats.txt"
    outfile.write_text(output)
    print(f"\n[Saved to {outfile}]")


if __name__ == "__main__":
    main()
