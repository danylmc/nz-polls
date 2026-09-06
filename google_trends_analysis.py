#!/usr/bin/env python3
"""
Google Trends NZ as Issue Salience Proxy.

Tests whether Google search intensity for key political terms predicts:
1. Ipsos issue salience (does search → survey concern?)
2. Government approval ratings
3. Party polling shifts

This gives us a high-frequency (monthly) proxy for public concern that sits
between objective conditions and survey-measured salience.

Data: Google Trends NZ monthly index (2017-2025), Ipsos Issues Monitor,
      polling data, and govt performance ratings.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Map Google Trends keywords to Ipsos issue columns
GT_TO_IPSOS = {
    "cost of living": "inflation_cost_of_living",
    "crime": "crime_law_order",
    "housing crisis": "housing_price",
    "immigration": "race_relations",  # closest proxy
    "healthcare": "healthcare_hospitals",
}


def load_trends():
    """Load Google Trends monthly data."""
    df = pd.read_csv(DATA_DIR / "google_trends_nz.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def load_ipsos_salience():
    """Load Ipsos issue salience data."""
    df = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_ipsos_govt():
    """Load government performance ratings."""
    df = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_polls_monthly():
    """Load polling data aggregated monthly."""
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
    for col in ["National", "Labour", "Green", "NZ First", "ACT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[["National", "Labour", "Green", "NZ First", "ACT"]].mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def match_trends_to_ipsos(trends, ipsos):
    """Match monthly Google Trends to nearest Ipsos survey date."""
    rows = []
    for _, irow in ipsos.iterrows():
        survey_date = irow["date"]
        # Average trends in 2 months before survey
        mask = (trends.index >= survey_date - pd.DateOffset(months=2)) & (trends.index <= survey_date)
        sub = trends.loc[mask]
        if len(sub) > 0:
            row = {"date": survey_date}
            for gt_key in GT_TO_IPSOS:
                if gt_key in sub.columns:
                    row[f"gt_{gt_key}"] = sub[gt_key].mean()
                ipsos_col = GT_TO_IPSOS[gt_key]
                if ipsos_col in irow.index and pd.notna(irow[ipsos_col]):
                    row[f"ipsos_{ipsos_col}"] = irow[ipsos_col]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    trends = load_trends()
    ipsos = load_ipsos_salience()
    govt = load_ipsos_govt()
    polls = load_polls_monthly()

    print("=" * 70)
    print("GOOGLE TRENDS NZ AS ISSUE SALIENCE PROXY")
    print("=" * 70)

    print(f"\nGoogle Trends data: {trends.index.min().strftime('%Y-%m')} to {trends.index.max().strftime('%Y-%m')}")
    print(f"Keywords: {list(trends.columns)}")

    # ── 1. Google Trends vs Ipsos salience ───────────────────────────
    print(f"\n{'=' * 70}")
    print("1. GOOGLE TRENDS vs IPSOS ISSUE SALIENCE")
    print("   Does search intensity predict survey-measured public concern?")
    print(f"{'=' * 70}")

    matched = match_trends_to_ipsos(trends, ipsos)
    print(f"\n  Matched observations: {len(matched)}")

    print(f"\n  {'Google Trends keyword':>25s}  {'Ipsos issue':>25s}  {'r':>7s}  {'p':>8s}  {'n':>4s}")
    for gt_key, ipsos_col in GT_TO_IPSOS.items():
        gt_col = f"gt_{gt_key}"
        ip_col = f"ipsos_{ipsos_col}"
        if gt_col in matched.columns and ip_col in matched.columns:
            sub = matched.dropna(subset=[gt_col, ip_col])
            if len(sub) > 5:
                r, p = pearsonr(sub[gt_col], sub[ip_col])
                print(f"  {gt_key:>25s}  {ipsos_col:>25s}  {r:>+7.3f}  {p:>8.4f}  {len(sub):>4d}")

    # ── 2. Google Trends vs Government Approval ──────────────────────
    print(f"\n{'=' * 70}")
    print("2. GOOGLE TRENDS → GOVERNMENT APPROVAL")
    print(f"{'=' * 70}")

    # Match trends to govt performance
    govt_matched = []
    for _, row in govt.iterrows():
        survey_date = row["date"]
        mask = (trends.index >= survey_date - pd.DateOffset(months=2)) & (trends.index <= survey_date)
        sub = trends.loc[mask]
        if len(sub) > 0:
            r = {"date": survey_date, "govt_perf": row["mean_score"]}
            for col in trends.columns:
                r[f"gt_{col}"] = sub[col].mean()
            govt_matched.append(r)
    govt_df = pd.DataFrame(govt_matched)

    if len(govt_df) > 5:
        print(f"\n  Matched observations: {len(govt_df)}")
        print(f"\n  {'Keyword':>20s}  {'r':>7s}  {'p':>8s}")
        for col in trends.columns:
            gt_col = f"gt_{col}"
            if gt_col in govt_df.columns:
                sub = govt_df.dropna(subset=[gt_col, "govt_perf"])
                if len(sub) > 5:
                    r, p = pearsonr(sub[gt_col], sub["govt_perf"])
                    print(f"  {col:>20s}  {r:>+7.3f}  {p:>8.4f}")

    # ── 3. Google Trends vs Party Polling ────────────────────────────
    print(f"\n{'=' * 70}")
    print("3. GOOGLE TRENDS → PARTY POLLING (monthly)")
    print(f"{'=' * 70}")

    # Merge trends with polls
    trends_polls = trends.join(polls, how="inner")
    print(f"\n  Matched months: {len(trends_polls)}")

    parties = ["National", "Labour"]
    keywords = list(trends.columns)

    print(f"\n  {'Keyword':>20s}  {'→ National':>11s}  {'p':>8s}  {'→ Labour':>9s}  {'p':>8s}")
    for kw in keywords:
        sub = trends_polls.dropna(subset=[kw, "National", "Labour"])
        if len(sub) > 10:
            r_nat, p_nat = pearsonr(sub[kw], sub["National"])
            r_lab, p_lab = pearsonr(sub[kw], sub["Labour"])
            print(f"  {kw:>20s}  {r_nat:>+11.3f}  {p_nat:>8.4f}  {r_lab:>+9.3f}  {p_lab:>8.4f}")

    # ── 4. Lead/lag: Do trends lead polls? ───────────────────────────
    print(f"\n{'=' * 70}")
    print("4. LEAD/LAG: Do Google Trends lead or lag polling changes?")
    print(f"{'=' * 70}")

    # Focus on cost-of-living (strongest predictor)
    print(f"\n  Cost-of-living search intensity vs Labour vote:")
    for lag in range(-3, 4):
        kw = "cost of living"
        if kw in trends.columns and "Labour" in polls.columns:
            if lag >= 0:
                gt_shifted = trends[kw].iloc[:len(trends)-lag if lag > 0 else len(trends)]
                poll_shifted = polls["Labour"].iloc[lag:] if lag > 0 else polls["Labour"]
            else:
                gt_shifted = trends[kw].iloc[-lag:]
                poll_shifted = polls["Labour"].iloc[:len(polls)+lag]

            # Align indices
            common = gt_shifted.index.intersection(poll_shifted.index)
            if len(common) > 10:
                r, p = pearsonr(gt_shifted.loc[common], poll_shifted.loc[common])
                marker = " ◄ peak" if abs(r) == max(abs(r) for _ in [0]) else ""
                print(f"    Lag {lag:+d} months: r = {r:+.3f}, p = {p:.4f}{marker}")

    # ── 5. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: All trends time series
    ax = axes[0, 0]
    for kw in keywords:
        # Normalise to z-score for comparison
        z = (trends[kw] - trends[kw].mean()) / trends[kw].std()
        ax.plot(trends.index, z, linewidth=1.5, alpha=0.7, label=kw)
    ax.set_ylabel("Google Trends (z-score)")
    ax.set_title("Google Trends NZ: Issue Search Intensity")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    # Panel 2: Cost-of-living search vs Ipsos salience
    ax = axes[0, 1]
    if "gt_cost of living" in matched.columns and "ipsos_inflation_cost_of_living" in matched.columns:
        sub = matched.dropna(subset=["gt_cost of living", "ipsos_inflation_cost_of_living"])
        ax.scatter(sub["gt_cost of living"], sub["ipsos_inflation_cost_of_living"],
                   color="#E74C3C", alpha=0.6)
        if len(sub) > 2:
            r, _ = pearsonr(sub["gt_cost of living"], sub["ipsos_inflation_cost_of_living"])
            ax.set_title(f"Google 'cost of living' vs Ipsos Salience (r={r:.3f})")
        ax.set_xlabel("Google Trends 'cost of living' index")
        ax.set_ylabel("Ipsos inflation/cost of living salience (%)")

    # Panel 3: Crime search vs Ipsos crime salience
    ax = axes[1, 0]
    if "gt_crime" in matched.columns and "ipsos_crime_law_order" in matched.columns:
        sub = matched.dropna(subset=["gt_crime", "ipsos_crime_law_order"])
        ax.scatter(sub["gt_crime"], sub["ipsos_crime_law_order"],
                   color="#2C3E50", alpha=0.6)
        if len(sub) > 2:
            r, _ = pearsonr(sub["gt_crime"], sub["ipsos_crime_law_order"])
            ax.set_title(f"Google 'crime' vs Ipsos Crime Salience (r={r:.3f})")
        ax.set_xlabel("Google Trends 'crime' index")
        ax.set_ylabel("Ipsos crime/law&order salience (%)")

    # Panel 4: Cost-of-living trends vs Labour polling
    ax = axes[1, 1]
    kw = "cost of living"
    if kw in trends.columns:
        common = trends.index.intersection(polls.index)
        ax2 = ax.twinx()
        ax.plot(trends.loc[common].index, trends.loc[common][kw], color="#E74C3C",
                linewidth=2, label=f"Google '{kw}'")
        ax2.plot(polls.loc[common].index, polls.loc[common]["Labour"], color="#d32f2f",
                 linewidth=2, alpha=0.5, label="Labour vote")
        ax.set_ylabel(f"Google '{kw}' index", color="#E74C3C")
        ax2.set_ylabel("Labour vote %", color="#d32f2f")
        ax.set_title("Cost of Living Search vs Labour Support")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    outpath = REPORTS_DIR / "google_trends_salience.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
