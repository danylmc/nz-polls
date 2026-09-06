#!/usr/bin/env python3
"""
House prices vs incumbent party support.

BIS real house price index (quarterly, 1962-present) vs polling.
Tests wealth effect vs affordability grievance.
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
from statsmodels.tsa.stattools import grangercausalitytests

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


def get_incumbent_at_date(date):
    if date < datetime(1999, 11, 27): return 'National'
    if date < datetime(2008, 11, 8): return 'Labour'
    if date < datetime(2017, 10, 26): return 'National'
    if date < datetime(2023, 11, 27): return 'Labour'
    return 'National'


def load_house_prices():
    """Load BIS real house price index."""
    df = pd.read_csv(DATA_DIR / "house_prices_real.csv")
    df["date"] = pd.to_datetime(df["observation_date"])
    df["hpi"] = pd.to_numeric(df["QNZR628BIS"], errors="coerce")
    df = df.dropna(subset=["hpi"]).sort_values("date").set_index("date")
    return df[["hpi"]]


def load_all_polls():
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
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for p in parties:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    return df


def main():
    hpi = load_house_prices()
    polls_raw = load_all_polls()

    polls_raw["incumbent"] = polls_raw["date"].apply(get_incumbent_at_date)
    polls_raw["inc_vote"] = polls_raw.apply(
        lambda r: pd.to_numeric(r.get(get_incumbent_at_date(r["date"])), errors="coerce"), axis=1
    )

    # Resample polls to quarterly to match HPI
    polls_raw["quarter"] = polls_raw["date"].dt.to_period("Q")
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    quarterly_polls = polls_raw.groupby("quarter").agg(
        inc_vote=("inc_vote", "mean"),
        **{p: (p, "mean") for p in parties}
    ).reset_index()
    quarterly_polls["date"] = quarterly_polls["quarter"].dt.to_timestamp()
    quarterly_polls = quarterly_polls.set_index("date").drop(columns=["quarter"])

    # Resample HPI to quarter-start
    hpi_q = hpi.resample("QS").mean()

    merged = hpi_q.join(quarterly_polls, how="inner").dropna(subset=["hpi", "inc_vote"])
    merged["incumbent"] = merged.index.to_series().apply(get_incumbent_at_date)
    print(f"House prices: {hpi.index.min():%Y-%m} to {hpi.index.max():%Y-%m}")
    print(f"Matched quarters: {len(merged)} ({merged.index.min():%Y-%m} to {merged.index.max():%Y-%m})")

    # ── 1. Level correlation ──────────────────────────────────────────
    r_level, p_level = pearsonr(merged["hpi"], merged["inc_vote"])
    print(f"\n=== LEVEL CORRELATION ===")
    print(f"  r = {r_level:.3f}, p = {p_level:.4f}, n = {len(merged)}")

    # ── 2. Change correlation (QoQ) ───────────────────────────────────
    merged["d_hpi"] = merged["hpi"].diff()
    merged["d_vote"] = merged["inc_vote"].diff()
    changes = merged.dropna(subset=["d_hpi", "d_vote"])
    r_change, p_change = pearsonr(changes["d_hpi"], changes["d_vote"])
    print(f"\n=== CHANGE CORRELATION (quarter-on-quarter) ===")
    print(f"  r = {r_change:.3f}, p = {p_change:.4f}, n = {len(changes)}")

    # ── 3. YoY change ────────────────────────────────────────────────
    merged["d_hpi_4q"] = merged["hpi"].diff(4)
    merged["d_vote_4q"] = merged["inc_vote"].diff(4)
    yoy = merged.dropna(subset=["d_hpi_4q", "d_vote_4q"])
    if len(yoy) > 10:
        r_yoy, p_yoy = pearsonr(yoy["d_hpi_4q"], yoy["d_vote_4q"])
        print(f"\n=== YEAR-ON-YEAR CHANGE ===")
        print(f"  r = {r_yoy:.3f}, p = {p_yoy:.4f}, n = {len(yoy)}")

    # ── 4. By government period ───────────────────────────────────────
    print(f"\n=== BY GOVERNMENT PERIOD (QoQ Δ) ===")
    for inc in ["Labour", "National"]:
        sub = changes[changes["incumbent"] == inc]
        if len(sub) > 5:
            r, p = pearsonr(sub["d_hpi"], sub["d_vote"])
            print(f"  {inc}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 5. HPI change vs party-specific effects ──────────────────────
    print(f"\n=== PARTY-LEVEL EFFECTS (YoY Δ) ===")
    for party in parties:
        merged[f"d_{party}_4q"] = merged[party].diff(4)
    yoy = merged.dropna(subset=["d_hpi_4q", "d_vote_4q"])
    for party in parties:
        col = f"d_{party}_4q"
        sub = yoy.dropna(subset=[col, "d_hpi_4q"])
        if len(sub) > 10:
            r, p = pearsonr(sub["d_hpi_4q"], sub[col])
            print(f"  {party:>10s}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 6. Granger causality ──────────────────────────────────────────
    print(f"\n=== GRANGER CAUSALITY (HPI → vote, max 4 lags) ===")
    gc_data = merged[["d_vote", "d_hpi"]].dropna()
    try:
        gc = grangercausalitytests(gc_data, maxlag=4, verbose=False)
        for lag, result in gc.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 7. Mediation through approval ─────────────────────────────────
    try:
        approval = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
        approval["date"] = pd.to_datetime(approval["date"])
        approval = approval.set_index("date").resample("QS").mean()
        merged_med = merged.join(approval[["mean_score"]], how="inner").dropna(
            subset=["hpi", "inc_vote", "mean_score"]
        )
        if len(merged_med) > 10:
            r_ha, p_ha = pearsonr(merged_med["hpi"], merged_med["mean_score"])
            print(f"\n=== HPI vs GOVT APPROVAL ===")
            print(f"  r = {r_ha:.3f}, p = {p_ha:.4f}, n = {len(merged_med)}")
    except Exception as e:
        print(f"\n  Approval analysis skipped: {e}")

    # ── 8. HPI vs Ipsos housing salience ──────────────────────────────
    try:
        salience = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
        salience["date"] = pd.to_datetime(salience["date"])
        salience = salience.set_index("date").resample("QS").mean()
        merged_sal = merged.join(salience[["housing_price"]], how="inner").dropna(
            subset=["hpi", "housing_price"]
        )
        if len(merged_sal) > 5:
            r_hs, p_hs = pearsonr(merged_sal["hpi"], merged_sal["housing_price"])
            print(f"\n=== HPI vs HOUSING SALIENCE (Ipsos) ===")
            print(f"  r = {r_hs:.3f}, p = {p_hs:.4f}, n = {len(merged_sal)}")
            # Change correlation
            d_hpi = merged_sal["hpi"].diff()
            d_sal = merged_sal["housing_price"].diff()
            sub = pd.DataFrame({"d_hpi": d_hpi, "d_sal": d_sal}).dropna()
            if len(sub) > 5:
                r, p = pearsonr(sub["d_hpi"], sub["d_sal"])
                print(f"  ΔHPI vs Δhousing salience: r = {r:.3f}, p = {p:.4f}")
    except Exception as e:
        print(f"\n  Salience analysis skipped: {e}")

    # ── 9. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax1 = axes[0]
    ax1r = ax1.twinx()
    ax1.plot(merged.index, merged["inc_vote"], color="steelblue", alpha=0.8, label="Incumbent vote %")
    ax1r.plot(merged.index, merged["hpi"], color="#8E44AD", alpha=0.8, label="Real house price index")
    ax1.set_ylabel("Incumbent vote %", color="steelblue")
    ax1r.set_ylabel("BIS Real HPI", color="#8E44AD")
    ax1.set_title("Real House Prices vs Incumbent Party Polling")
    govt_periods = [
        ("1993-01", "1999-11", "National"), ("1999-11", "2008-11", "Labour"),
        ("2008-11", "2017-10", "National"), ("2017-10", "2023-11", "Labour"),
        ("2023-11", "2026-03", "National")
    ]
    for s, e, party in govt_periods:
        color = "#d32f2f" if party == "Labour" else "#1565c0"
        try:
            ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.05, color=color)
        except Exception:
            pass
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax2 = axes[1]
    if len(yoy) > 10:
        colors_map = yoy["incumbent"].map({"Labour": "#d32f2f", "National": "#1565c0"})
        ax2.scatter(yoy["d_hpi_4q"], yoy["d_vote_4q"], c=colors_map, alpha=0.4, s=20)
        m, b = np.polyfit(yoy["d_hpi_4q"], yoy["d_vote_4q"], 1)
        x_range = np.linspace(yoy["d_hpi_4q"].min(), yoy["d_hpi_4q"].max(), 50)
        ax2.plot(x_range, m * x_range + b, "k--", alpha=0.7)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axvline(0, color="gray", lw=0.5)
        ax2.set_xlabel("Year-on-year change in real house price index")
        ax2.set_ylabel("Year-on-year change in incumbent vote (pp)")
        ax2.set_title(f"YoY HPI Change vs Incumbent Vote Change (r={r_yoy:.3f}, p={p_yoy:.4f})")

    plt.tight_layout()
    outpath = REPORTS_DIR / "house_prices_vs_incumbent.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
