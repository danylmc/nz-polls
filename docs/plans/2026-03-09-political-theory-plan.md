# Political Theory Briefing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Download comparative datasets (OECD housing, productivity, governance, trust, Freedom House) and write `reports/briefing_theory.md` — a reference briefing connecting NZ polling findings to political theory (Arendt, Olson, Klein/Thompson, Fukuyama, Acemoglu).

**Architecture:** Three-phase pipeline: (1) download 5 comparative datasets via API/web, (2) run a single analysis script that extracts NZ rankings and comparative stats, (3) write the briefing document synthesizing existing findings with new comparative data.

**Tech Stack:** Python 3.12, pandas, numpy, requests, openpyxl (for XLSX). No venv — install with `--break-system-packages`. Output is markdown.

**Project conventions:** Scripts in project root. Data in `data/`. Reports in `reports/`. Use `matplotlib.use('Agg')`. Use `Path(__file__).parent` for paths. No tests — this project is analytical scripts, not a software product.

---

### Task 1: Download OECD Price-to-Income Ratios

**Files:**
- Create: `data/oecd_price_to_income.csv`
- Create: `download_comparative_data.py`

**Step 1: Write the download script (first dataset)**

Create `download_comparative_data.py` with a function to fetch OECD price-to-income ratios via SDMX API. This is the first of 5 datasets; subsequent tasks will add to this script.

```python
#!/usr/bin/env python3
"""Download comparative datasets for political theory analysis."""

import requests
import pandas as pd
from pathlib import Path
import time
import json
import zipfile
import io

DATA_DIR = Path(__file__).parent / "data"


def download_oecd_price_to_income():
    """OECD Analytical House Prices: price-to-income ratio, all countries, annual."""
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,/"
        ".A.PRTP2INC..?"
        "format=csvfilewithlabels"
    )
    print("Downloading OECD price-to-income ratios...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    outpath = DATA_DIR / "oecd_price_to_income.csv"
    outpath.write_text(resp.text)
    # Verify
    df = pd.read_csv(outpath)
    nz = df[df["REF_AREA"].str.contains("NZL|New Zealand", na=False)]
    print(f"  Saved {len(df)} rows, {len(nz)} NZ rows to {outpath}")
    return df


if __name__ == "__main__":
    download_oecd_price_to_income()
```

**Step 2: Run the download**

Run: `cd /mnt/d/data/nz-polls && python download_comparative_data.py`
Expected: CSV saved with NZ + OECD annual price-to-income ratios. If the SDMX API fails (it sometimes does), try the alternative URL format or adjust column filter.

**Step 3: Verify the data**

Run: `python -c "import pandas as pd; df = pd.read_csv('data/oecd_price_to_income.csv'); print(df.columns.tolist()); print(df.head(2))"`
Expected: Columns include REF_AREA (country), TIME_PERIOD (year), OBS_VALUE (ratio). NZ rows present.

---

### Task 2: Download World Governance Indicators

**Files:**
- Create: `data/wgi_governance.json`
- Modify: `download_comparative_data.py`

**Step 1: Add WGI download function**

Add to `download_comparative_data.py`:

```python
def download_wgi():
    """World Bank Worldwide Governance Indicators — 6 dimensions, all countries."""
    indicators = {
        "GE.EST": "Government Effectiveness",
        "RL.EST": "Rule of Law",
        "VA.EST": "Voice and Accountability",
        "RQ.EST": "Regulatory Quality",
        "CC.EST": "Control of Corruption",
        "PV.EST": "Political Stability",
    }
    all_data = {}
    for code, name in indicators.items():
        url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000"
        print(f"Downloading WGI: {name}...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if len(data) > 1:
            all_data[code] = {"name": name, "records": data[1]}
        time.sleep(1)  # Be polite
    outpath = DATA_DIR / "wgi_governance.json"
    outpath.write_text(json.dumps(all_data, indent=2))
    # Count NZ records
    nz_count = sum(
        1 for ind in all_data.values()
        for r in ind["records"]
        if r.get("country", {}).get("id") == "NZL"
    )
    print(f"  Saved {len(all_data)} indicators, {nz_count} NZ records to {outpath}")
```

Update `if __name__` block to call both functions.

**Step 2: Run the download**

Run: `cd /mnt/d/data/nz-polls && python download_comparative_data.py`
Expected: JSON saved with 6 WGI dimensions for all countries.

---

### Task 3: Download OECD Labour Productivity

**Files:**
- Create: `data/oecd_labour_productivity.csv`
- Modify: `download_comparative_data.py`

**Step 1: Add productivity download function**

```python
def download_oecd_productivity():
    """OECD GDP per hour worked (USD PPP), all countries, annual."""
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.TPS,DSD_PDB@DF_PDB_LV,/"
        ".A.GDPHRS..USD_PPP_H.V?"
        "format=csvfilewithlabels"
    )
    print("Downloading OECD labour productivity...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    outpath = DATA_DIR / "oecd_labour_productivity.csv"
    outpath.write_text(resp.text)
    df = pd.read_csv(outpath)
    nz = df[df["REF_AREA"].str.contains("NZL|New Zealand", na=False)]
    print(f"  Saved {len(df)} rows, {len(nz)} NZ rows to {outpath}")
```

**Step 2: Run and verify**

Run: `cd /mnt/d/data/nz-polls && python download_comparative_data.py`
Expected: CSV with GDP per hour worked for OECD countries.

---

### Task 4: Download OECD Trust in Government

**Files:**
- Create: `data/oecd_trust_govt.csv`
- Modify: `download_comparative_data.py`

**Step 1: Add trust download function**

```python
def download_oecd_trust():
    """OECD Government at a Glance — trust in government indicator."""
    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.GOV.GIP,DSD_GOV@DF_GOV_2023,/"
        "all?"
        "format=csvfilewithlabels"
    )
    print("Downloading OECD trust in government...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    outpath = DATA_DIR / "oecd_trust_govt.csv"
    outpath.write_text(resp.text)
    df = pd.read_csv(outpath)
    nz = df[df.apply(lambda r: "NZL" in str(r.values) or "New Zealand" in str(r.values), axis=1)]
    print(f"  Saved {len(df)} rows, {len(nz)} NZ rows to {outpath}")
```

Note: The Government at a Glance SDMX endpoint may return a large dataset with many indicators beyond trust. If so, filter to trust-related indicators in the analysis script. If the endpoint fails or returns too much, fall back to downloading the trust-specific data from the OECD Data Explorer CSV export.

**Step 2: Run and verify**

Run: `cd /mnt/d/data/nz-polls && python download_comparative_data.py`

---

### Task 5: Download Freedom House Scores

**Files:**
- Create: `data/freedom_house_scores.xlsx`
- Modify: `download_comparative_data.py`

**Step 1: Add Freedom House download function**

```python
def download_freedom_house():
    """Freedom House — Freedom in the World ratings 1973-2024."""
    url = (
        "https://freedomhouse.org/sites/default/files/2025-02/"
        "Country_and_Territory_Ratings_and_Statuses_FIW_1973-2024.xlsx"
    )
    print("Downloading Freedom House scores...")
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    outpath = DATA_DIR / "freedom_house_scores.xlsx"
    outpath.write_bytes(resp.content)
    df = pd.read_excel(outpath, sheet_name=0)
    print(f"  Saved {len(df)} rows to {outpath}")
    print(f"  Columns: {df.columns.tolist()[:10]}")
```

Note: Freedom House may block automated downloads or may have changed the URL. If the direct download fails, the analysis script should handle a manually-downloaded file, or we can try alternative URLs. Install openpyxl if needed: `pip install openpyxl --break-system-packages`.

**Step 2: Run and verify**

Run: `cd /mnt/d/data/nz-polls && python download_comparative_data.py`

---

### Task 6: Download NZ Building Consents

**Files:**
- Create: `data/nz_building_consents.csv`
- Modify: `download_comparative_data.py`

**Step 1: Add NZ building consents download**

Stats NZ Infoshare doesn't have a direct API, but Stats NZ publishes building consent CSVs. Try the large-datasets CSV download page or use a direct URL.

```python
def download_nz_building_consents():
    """Stats NZ building consents — try direct CSV or fallback to manual."""
    # Stats NZ publishes building consent data via their large-datasets page
    # Try the most recent release XLSX first
    url = (
        "https://www.stats.govt.nz/assets/Uploads/Building-consents-issued/"
        "Building-consents-issued-January-2025/Download-data/"
        "building-consents-issued-january-2025.xlsx"
    )
    print("Downloading NZ building consents...")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        outpath = DATA_DIR / "nz_building_consents.xlsx"
        outpath.write_bytes(resp.content)
        # Try to read first sheet
        df = pd.read_excel(outpath, sheet_name=0)
        print(f"  Saved {len(df)} rows to {outpath}")
    except Exception as e:
        print(f"  Direct download failed: {e}")
        print("  Will need manual download from Stats NZ Infoshare")
        print("  Go to: https://infoshare.stats.govt.nz/ -> Industry Sectors -> Building Consents (BLD)")
```

**Step 2: Run and verify**

This may require manual intervention if Stats NZ blocks automated downloads. If so, note the failure and proceed — the analysis script should handle missing data gracefully.

---

### Task 7: Write Comparative Analysis Script

**Files:**
- Create: `political_theory_analysis.py`

**Step 1: Write the analysis script**

This script reads all downloaded comparative data and extracts the specific statistics needed for the briefing document. It prints a summary report to stdout.

```python
#!/usr/bin/env python3
"""
Extract comparative statistics for political theory briefing.

Reads OECD, World Bank, and Freedom House data to compute:
- NZ price-to-income ratio vs OECD average and rank
- NZ labour productivity vs OECD average and rank
- NZ governance scores (WGI) vs OECD average and rank
- NZ trust in government vs OECD average
- NZ Freedom House trajectory (1973-2024)
- NZ building consents trend (if available)
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


def analyse_price_to_income():
    """NZ housing affordability in OECD context."""
    path = DATA_DIR / "oecd_price_to_income.csv"
    if not path.exists():
        print("SKIP: oecd_price_to_income.csv not found")
        return {}
    df = pd.read_csv(path)
    # Identify country and value columns (SDMX format varies)
    # Look for REF_AREA, TIME_PERIOD, OBS_VALUE
    print("\n=== OECD Price-to-Income Ratio ===")
    print(f"Columns: {df.columns.tolist()}")
    # Filter to most recent year available
    # Adapt column names based on actual download
    results = {"raw_columns": df.columns.tolist(), "n_rows": len(df)}
    return results


def analyse_wgi():
    """NZ governance quality vs OECD peers."""
    path = DATA_DIR / "wgi_governance.json"
    if not path.exists():
        print("SKIP: wgi_governance.json not found")
        return {}
    with open(path) as f:
        data = json.load(f)

    # OECD member country codes
    oecd = {
        "AUS", "AUT", "BEL", "CAN", "CHL", "COL", "CRI", "CZE", "DNK",
        "EST", "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR",
        "ITA", "JPN", "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL",
        "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE", "CHE", "TUR",
        "GBR", "USA",
    }

    print("\n=== World Governance Indicators ===")
    results = {}
    for code, info in data.items():
        name = info["name"]
        records = info["records"]
        # Most recent year
        nz_records = [
            r for r in records
            if r.get("country", {}).get("id") == "NZL" and r.get("value") is not None
        ]
        if not nz_records:
            continue
        # Sort by year desc
        nz_records.sort(key=lambda r: r.get("date", ""), reverse=True)
        latest = nz_records[0]
        latest_year = latest["date"]

        # Get all OECD scores for same year
        oecd_scores = [
            r["value"] for r in records
            if r.get("country", {}).get("id") in oecd
            and r.get("date") == latest_year
            and r.get("value") is not None
        ]
        nz_val = latest["value"]
        oecd_mean = np.mean(oecd_scores) if oecd_scores else None
        oecd_rank = sum(1 for s in oecd_scores if s > nz_val) + 1 if oecd_scores else None
        n_oecd = len(oecd_scores)

        print(f"  {name} ({latest_year}): NZ = {nz_val:.2f}, OECD mean = {oecd_mean:.2f}, "
              f"rank {oecd_rank}/{n_oecd}")
        results[code] = {
            "name": name, "year": latest_year, "nz": nz_val,
            "oecd_mean": oecd_mean, "rank": oecd_rank, "n": n_oecd,
        }

        # Also get NZ trajectory (all years)
        nz_trajectory = [(r["date"], r["value"]) for r in nz_records]
        nz_trajectory.sort()
        if len(nz_trajectory) > 2:
            first_val = nz_trajectory[0][1]
            last_val = nz_trajectory[-1][1]
            change = last_val - first_val
            print(f"    Trajectory: {nz_trajectory[0][0]}={first_val:.2f} → "
                  f"{nz_trajectory[-1][0]}={last_val:.2f} (change={change:+.2f})")

    return results


def analyse_productivity():
    """NZ labour productivity vs OECD."""
    path = DATA_DIR / "oecd_labour_productivity.csv"
    if not path.exists():
        print("SKIP: oecd_labour_productivity.csv not found")
        return {}
    df = pd.read_csv(path)
    print("\n=== OECD Labour Productivity ===")
    print(f"Columns: {df.columns.tolist()}")
    results = {"raw_columns": df.columns.tolist(), "n_rows": len(df)}
    return results


def analyse_trust():
    """OECD trust in government."""
    path = DATA_DIR / "oecd_trust_govt.csv"
    if not path.exists():
        print("SKIP: oecd_trust_govt.csv not found")
        return {}
    df = pd.read_csv(path)
    print("\n=== OECD Trust in Government ===")
    print(f"Columns: {df.columns.tolist()[:15]}")
    print(f"Total rows: {len(df)}")
    results = {"raw_columns": df.columns.tolist(), "n_rows": len(df)}
    return results


def analyse_freedom_house():
    """NZ Freedom House trajectory."""
    xlsx_path = DATA_DIR / "freedom_house_scores.xlsx"
    if not xlsx_path.exists():
        print("SKIP: freedom_house_scores.xlsx not found")
        return {}
    df = pd.read_excel(xlsx_path, sheet_name=0)
    print("\n=== Freedom House ===")
    print(f"Columns: {df.columns.tolist()[:10]}")
    # Find NZ rows
    nz = df[df.apply(lambda r: "New Zealand" in str(r.values), axis=1)]
    print(f"NZ rows: {len(nz)}")
    if len(nz) > 0:
        print(nz.head(3).to_string())
    results = {"n_nz_rows": len(nz)}
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARATIVE DATA ANALYSIS FOR POLITICAL THEORY BRIEFING")
    print("=" * 60)

    r1 = analyse_price_to_income()
    r2 = analyse_wgi()
    r3 = analyse_productivity()
    r4 = analyse_trust()
    r5 = analyse_freedom_house()

    print("\n" + "=" * 60)
    print("DONE — review output above, then refine column parsing")
    print("=" * 60)
```

**Step 2: Run the analysis**

Run: `cd /mnt/d/data/nz-polls && python political_theory_analysis.py`

The first run will reveal actual column names and data structures. The script will need refinement based on what the APIs actually return (OECD SDMX column names vary). This is expected — refine the parsing functions based on actual column names.

**Step 3: Refine analysis functions**

After seeing actual column names, update each `analyse_*` function to:
- Filter to most recent year
- Compute NZ rank among OECD countries
- Compute OECD mean/median
- Extract NZ time series for trajectory analysis
- Print formatted summary statistics

**Step 4: Run refined analysis and capture output**

Run: `cd /mnt/d/data/nz-polls && python political_theory_analysis.py > reports/comparative_stats.txt`
This output will be the source for embedding statistics in the briefing.

---

### Task 8: Write the Political Theory Briefing

**Files:**
- Create: `reports/briefing_theory.md`

**Step 1: Read source materials**

Read the following files to gather all evidence:
- `reports/briefing_political.md` (existing political briefing — all NZ polling findings)
- `reports/briefing_economic.md` (existing economic briefing — causal chain, pocketbook)
- `reports/comparative_stats.txt` (new comparative data from Task 7)
- `docs/plans/2026-03-09-political-theory-design.md` (section specs and theoretical framing)

**Step 2: Write the briefing document**

Write `reports/briefing_theory.md` following the design document's section structure:

1. **Header & Framework** (~200 words) — scope, evidence tiers, three-tensions thesis, theorist key
2. **The Scarcity Trap** (~800 words) — Olson/Klein/Thompson + pocketbook findings + new comparative data
3. **The Accountability Paradox** (~800 words) — Arendt/Fiorina + retrospective voting + perceptual screen + trust data
4. **Identity, Realignment, and the New Cleavage** (~600 words) — Fukuyama + dealignment findings
5. **The Narrow Corridor** (~600 words) — Acemoglu + MMP mechanics + WGI/Freedom House scores
6. **Distributional Coalitions and the Abundance Failure** (~600 words) — Olson + null house price finding + building consents + productivity
7. **Propositions and Open Questions** (~400 words) — testable hypotheses, gaps, cross-references

Style: Dense reference format matching existing briefings. Statistics front-and-centre. Evidence tiers on each section. Cross-references to `briefing_political.md` and `briefing_economic.md` where findings are covered in more detail.

---

### Task 9: Review for Consistency

**Step 1: Cross-check statistics**

Verify that every statistic cited in `briefing_theory.md` matches the source briefings. Key values to check:
- Causal chain mediation percentages (68%, 49%, 50%)
- Retrospective voting mean r (0.376)
- Perceptual screen gap (+0.68)
- House prices null (r=-0.069)
- Cost-of-living → approval (R²=0.755)
- Dealignment pseudo-R² (0.032→0.044)
- Maori identity OR (0.14-0.39)
- L-R positional voting (r=0.533→0.760)

**Step 2: Check evidence tiers**

Ensure evidence tiers in the theory briefing are consistent with the political and economic briefings. No finding should be upgraded or downgraded without justification.

**Step 3: Check cross-references**

Every cross-reference to `briefing_political.md` or `briefing_economic.md` should point to a real section.

---

### Task 10: Commit

**Step 1: Stage and commit all new files**

```bash
git add download_comparative_data.py political_theory_analysis.py \
    data/oecd_price_to_income.csv data/wgi_governance.json \
    data/oecd_labour_productivity.csv data/oecd_trust_govt.csv \
    data/freedom_house_scores.xlsx \
    reports/briefing_theory.md reports/comparative_stats.txt \
    docs/plans/2026-03-09-political-theory-design.md \
    docs/plans/2026-03-09-political-theory-plan.md
git commit -m "Add political theory briefing with comparative data

Downloads OECD housing affordability, productivity, trust, World
Governance Indicators, and Freedom House data. Synthesizes 30 NZ
polling analyses through Arendt, Olson, Klein/Thompson, Fukuyama,
and Acemoglu frameworks.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

Note: Only stage files that actually exist. Some downloads may have failed — exclude missing files.
