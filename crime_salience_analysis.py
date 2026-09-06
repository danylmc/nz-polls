#!/usr/bin/env python3
"""
Crime Statistics vs Crime Salience Analysis.

Tests: Does actual recorded crime drive Ipsos crime salience,
or is crime salience driven by media/perception independent of reality?

Extends the "perception vs reality" finding from economics to crime.
Crime salience correlates r=-0.744 with govt approval — but does actual crime
predict this, or is it media-driven?

Data sources:
- NZ Police monthly victimisations (Dec 2017 – Dec 2023) from Figure.NZ
- Ipsos Issues Monitor crime_law_order salience (28 waves, 2018–2025)
- Ipsos govt_performance ratings (30 waves, 2017–2025)
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# ── NZ Police monthly victimisations (Figure.NZ, source: NZ Police) ──
# Total victimisations for all offences, monthly
CRIME_DATA = {
    "2017-12": 21846, "2018-01": 23597, "2018-02": 21249, "2018-03": 22090,
    "2018-04": 20121, "2018-05": 21614, "2018-06": 21046, "2018-07": 20406,
    "2018-08": 21189, "2018-09": 20721, "2018-10": 22102, "2018-11": 22275,
    "2018-12": 22240, "2019-01": 24457, "2019-02": 21177, "2019-03": 22057,
    "2019-04": 21080, "2019-05": 22648, "2019-06": 21315, "2019-07": 22710,
    "2019-08": 23846, "2019-09": 23845, "2019-10": 25260, "2019-11": 26344,
    "2019-12": 27391, "2020-01": 28342, "2020-02": 27239, "2020-03": 24952,
    "2020-04": 12335, "2020-05": 15764, "2020-06": 19596, "2020-07": 21428,
    "2020-08": 21750, "2020-09": 21784, "2020-10": 22993, "2020-11": 23892,
    "2020-12": 24218, "2021-01": 25341, "2021-02": 22781, "2021-03": 25573,
    "2021-04": 24179, "2021-05": 26332, "2021-06": 25559, "2021-07": 25998,
    "2021-08": 22712, "2021-09": 19871, "2021-10": 23824, "2021-11": 24638,
    "2021-12": 26726, "2022-01": 28961, "2022-02": 25832, "2022-03": 28432,
    "2022-04": 26278, "2022-05": 29606, "2022-06": 26564, "2022-07": 29096,
    "2022-08": 32006, "2022-09": 30303, "2022-10": 31710, "2022-11": 32545,
    "2022-12": 32597, "2023-01": 33421, "2023-02": 28799, "2023-03": 33988,
    "2023-04": 31413, "2023-05": 33278, "2023-06": 30778, "2023-07": 31990,
    "2023-08": 31207, "2023-09": 31527, "2023-10": 31633, "2023-11": 29314,
    "2023-12": 32773,
}


def load_crime_monthly():
    """Load monthly crime data."""
    rows = [{"date": pd.to_datetime(k + "-01"), "victimisations": v}
            for k, v in CRIME_DATA.items()]
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    # 12-month rolling average to smooth seasonality
    df["crime_12m"] = df["victimisations"].rolling(12).mean()
    # Year-on-year change
    df["crime_yoy"] = df["victimisations"].pct_change(12) * 100
    return df


def load_ipsos_salience():
    """Load Ipsos crime salience data."""
    df = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "crime_law_order", "inflation_cost_of_living"]].copy()


def load_ipsos_govt():
    """Load Ipsos government performance data."""
    df = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "mean_score"]].copy()


def load_polls_monthly():
    """Load polling data aggregated to monthly."""
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            for p in data.get("polls", []):
                row = {"date": p["date"]}
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ["National", "Labour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[["National", "Labour"]].mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def main():
    crime = load_crime_monthly()
    salience = load_ipsos_salience()
    govt = load_ipsos_govt()
    polls = load_polls_monthly()

    print("=" * 70)
    print("CRIME STATISTICS vs CRIME SALIENCE ANALYSIS")
    print("Does actual crime drive public concern, or is it perception?")
    print("=" * 70)

    # ── 1. Match crime data to Ipsos survey dates ────────────────────
    print(f"\nCrime data: {crime.index.min().strftime('%Y-%m')} to {crime.index.max().strftime('%Y-%m')}")
    print(f"Ipsos crime salience: {len(salience)} waves")

    # For each Ipsos wave, get the crime rate in the preceding 3 months
    merged_rows = []
    for _, row in salience.iterrows():
        survey_date = row["date"]
        # Average crime in the 3 months before survey
        mask = (crime.index >= survey_date - pd.DateOffset(months=3)) & (crime.index < survey_date)
        sub = crime.loc[mask]
        if len(sub) > 0:
            merged_rows.append({
                "date": survey_date,
                "crime_salience": row["crime_law_order"],
                "col_salience": row["inflation_cost_of_living"],
                "crime_avg_3m": sub["victimisations"].mean(),
                "crime_12m": sub["crime_12m"].iloc[-1] if pd.notna(sub["crime_12m"].iloc[-1]) else np.nan,
                "crime_yoy": sub["crime_yoy"].iloc[-1] if pd.notna(sub["crime_yoy"].iloc[-1]) else np.nan,
            })

    merged = pd.DataFrame(merged_rows)

    # Add govt performance
    for _, row in govt.iterrows():
        closest_idx = (merged["date"] - row["date"]).abs().idxmin()
        if abs((merged.loc[closest_idx, "date"] - row["date"]).days) < 45:
            merged.loc[closest_idx, "govt_perf"] = row["mean_score"]

    # Add polling
    for idx, row in merged.iterrows():
        closest_month = row["date"].to_period("M").to_timestamp()
        if closest_month in polls.index:
            merged.loc[idx, "national"] = polls.loc[closest_month, "National"]
            merged.loc[idx, "labour"] = polls.loc[closest_month, "Labour"]

    print(f"Matched observations: {len(merged)}")

    # ── 2. Core test: does actual crime predict crime salience? ──────
    print(f"\n{'=' * 70}")
    print("1. DOES ACTUAL CRIME DRIVE CRIME SALIENCE?")
    print(f"{'=' * 70}")

    for crime_var, label in [("crime_avg_3m", "3-month avg victimisations"),
                              ("crime_12m", "12-month rolling avg"),
                              ("crime_yoy", "Year-on-year % change")]:
        sub = merged.dropna(subset=[crime_var, "crime_salience"])
        if len(sub) > 5:
            r, p = pearsonr(sub[crime_var], sub["crime_salience"])
            print(f"\n  {label}:")
            print(f"    r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 3. Does crime salience predict govt approval? ────────────────
    print(f"\n{'=' * 70}")
    print("2. CRIME SALIENCE → GOVERNMENT APPROVAL")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["crime_salience", "govt_perf"])
    if len(sub) > 5:
        r, p = pearsonr(sub["crime_salience"], sub["govt_perf"])
        print(f"\n  Crime salience → govt approval: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 4. Does crime salience predict party polling? ────────────────
    print(f"\n{'=' * 70}")
    print("3. CRIME SALIENCE → PARTY POLLING")
    print(f"{'=' * 70}")

    for party in ["national", "labour"]:
        sub = merged.dropna(subset=["crime_salience", party])
        if len(sub) > 5:
            r, p = pearsonr(sub["crime_salience"], sub[party])
            print(f"\n  Crime salience → {party.title()}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 5. Compare: actual crime vs crime salience as predictors ─────
    print(f"\n{'=' * 70}")
    print("4. ACTUAL CRIME vs CRIME SALIENCE: WHICH PREDICTS APPROVAL?")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["crime_avg_3m", "crime_salience", "govt_perf"])
    if len(sub) > 5:
        r_actual, p_actual = pearsonr(sub["crime_avg_3m"], sub["govt_perf"])
        r_salience, p_salience = pearsonr(sub["crime_salience"], sub["govt_perf"])
        print(f"\n  Actual crime → govt approval:     r = {r_actual:.3f}, p = {p_actual:.4f}")
        print(f"  Crime salience → govt approval:   r = {r_salience:.3f}, p = {p_salience:.4f}")
        print(f"\n  → {'Perception' if abs(r_salience) > abs(r_actual) else 'Reality'} is the stronger predictor")
        print(f"     (salience explains {r_salience**2:.1%} vs actual {r_actual**2:.1%} of variance)")

    # ── 6. Timeline comparison ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("5. TIMELINE: Crime trend vs salience trend")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["crime_avg_3m", "crime_salience"])
    print(f"\n  {'Date':>12s}  {'Actual crime':>13s}  {'Crime salience':>15s}")
    for _, row in sub.iterrows():
        print(f"  {row['date'].strftime('%Y-%m'):>12s}  {row['crime_avg_3m']:>13,.0f}  {row['crime_salience']:>14.0f}%")

    # ── 7. Does crime salience covary with cost-of-living salience? ──
    print(f"\n{'=' * 70}")
    print("6. IS CRIME SALIENCE INDEPENDENT OF COST-OF-LIVING SALIENCE?")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["crime_salience", "col_salience"])
    if len(sub) > 5:
        r, p = pearsonr(sub["crime_salience"], sub["col_salience"])
        print(f"\n  Crime salience ↔ Cost-of-living salience: r = {r:.3f}, p = {p:.4f}")
        print(f"  → {'Correlated (issues move together)' if abs(r) > 0.3 else 'Independent issues'}")

    # ── 8. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Crime victimisations time series with salience overlay
    ax = axes[0, 0]
    ax.plot(crime.index, crime["crime_12m"] / 1000, color="#E74C3C", linewidth=2, label="12m avg victimisations (000s)")
    ax.set_ylabel("Monthly victimisations (000s, 12m avg)", color="#E74C3C")
    ax.set_title("Recorded Crime vs Crime Salience")
    axr = ax.twinx()
    sub = merged.dropna(subset=["crime_salience"])
    axr.plot(sub["date"], sub["crime_salience"], "o-", color="#2C3E50", linewidth=2, label="Crime salience %")
    axr.set_ylabel("Crime salience (%)", color="#2C3E50")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # Panel 2: Scatter — actual crime vs crime salience
    ax = axes[0, 1]
    sub = merged.dropna(subset=["crime_avg_3m", "crime_salience"])
    ax.scatter(sub["crime_avg_3m"] / 1000, sub["crime_salience"], color="#E74C3C", alpha=0.6)
    if len(sub) > 2:
        m, b = np.polyfit(sub["crime_avg_3m"] / 1000, sub["crime_salience"], 1)
        xr = np.linspace(sub["crime_avg_3m"].min() / 1000, sub["crime_avg_3m"].max() / 1000, 50)
        ax.plot(xr, m * xr + b, "k--", alpha=0.5)
        r, _ = pearsonr(sub["crime_avg_3m"], sub["crime_salience"])
        ax.set_title(f"Actual Crime vs Crime Salience (r={r:.3f})")
    ax.set_xlabel("Monthly victimisations (000s, 3m avg)")
    ax.set_ylabel("Crime salience (%)")

    # Panel 3: Scatter — crime salience vs govt approval
    ax = axes[1, 0]
    sub = merged.dropna(subset=["crime_salience", "govt_perf"])
    ax.scatter(sub["crime_salience"], sub["govt_perf"], color="#2C3E50", alpha=0.6)
    if len(sub) > 2:
        m, b = np.polyfit(sub["crime_salience"], sub["govt_perf"], 1)
        xr = np.linspace(sub["crime_salience"].min(), sub["crime_salience"].max(), 50)
        ax.plot(xr, m * xr + b, "k--", alpha=0.5)
        r, _ = pearsonr(sub["crime_salience"], sub["govt_perf"])
        ax.set_title(f"Crime Salience vs Govt Approval (r={r:.3f})")
    ax.set_xlabel("Crime salience (%)")
    ax.set_ylabel("Govt performance rating (0-10)")

    # Panel 4: Comparison — actual crime and salience normalised
    ax = axes[1, 1]
    sub = merged.dropna(subset=["crime_avg_3m", "crime_salience"])
    # Normalise both to z-scores
    crime_z = (sub["crime_avg_3m"] - sub["crime_avg_3m"].mean()) / sub["crime_avg_3m"].std()
    sal_z = (sub["crime_salience"] - sub["crime_salience"].mean()) / sub["crime_salience"].std()
    ax.plot(sub["date"], crime_z, "o-", color="#E74C3C", linewidth=2, label="Actual crime (z)")
    ax.plot(sub["date"], sal_z, "s-", color="#2C3E50", linewidth=2, label="Crime salience (z)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_ylabel("Standard deviations from mean")
    ax.set_title("Crime Reality vs Perception (normalised)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = REPORTS_DIR / "crime_salience.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
