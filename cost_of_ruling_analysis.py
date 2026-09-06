#!/usr/bin/env python3
"""
Cost of Ruling: Coalition vs Majority Decay.

Tests Paldam (1991): Does incumbent support erode faster for coalition governments?
Measures monthly polling decay rate for each government period.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress, ttest_ind

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Government periods with composition
# (start, end, PM party, type, label)
GOVERNMENTS = [
    ("1996-12-16", "1999-11-27", "National", "coalition", "Bolger/Shipley-NZF (1996-99)"),
    ("1999-11-27", "2002-07-27", "Labour", "coalition", "Clark-Alliance (1999-02)"),
    ("2002-07-27", "2005-09-17", "Labour", "coalition", "Clark minority (2002-05)"),
    ("2005-09-17", "2008-11-08", "Labour", "coalition", "Clark-NZF-UF (2005-08)"),
    ("2008-11-08", "2011-11-26", "National", "coalition", "Key-ACT-UF-MP (2008-11)"),
    ("2011-11-26", "2014-09-20", "National", "near_majority", "Key-ACT-UF (2011-14)"),
    ("2014-09-20", "2017-09-23", "National", "near_majority", "Key/English-ACT-UF (2014-17)"),
    ("2017-10-26", "2020-10-17", "Labour", "coalition", "Ardern-NZF-Grn (2017-20)"),
    ("2020-10-17", "2023-10-14", "Labour", "majority", "Ardern/Hipkins majority (2020-23)"),
    ("2023-11-27", "2026-03-01", "National", "coalition", "Luxon-ACT-NZF (2023-)"),
]


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
    for col in ["National", "Labour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    polls = load_all_polls()
    print("=" * 70)
    print("COST OF RULING: COALITION vs MAJORITY DECAY")
    print("=" * 70)

    results = []
    for start_str, end_str, party, gov_type, label in GOVERNMENTS:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        sub = polls[(polls["date"] >= start) & (polls["date"] <= end)].copy()
        sub = sub.dropna(subset=[party])

        if len(sub) < 10:
            continue

        # Calculate months from start
        sub["months"] = (sub["date"] - start).dt.days / 30.44
        sub["vote"] = sub[party]

        # Linear regression: vote ~ months
        slope, intercept, r_value, p_value, std_err = linregress(sub["months"], sub["vote"])
        initial = intercept
        final = intercept + slope * sub["months"].max()
        total_change = final - initial
        duration_months = sub["months"].max()

        results.append({
            "label": label,
            "party": party,
            "type": gov_type,
            "slope": slope,  # pp per month
            "r": r_value,
            "p": p_value,
            "initial": initial,
            "final": final,
            "change": total_change,
            "duration": duration_months,
            "n": len(sub),
        })

        print(f"\n  {label}")
        print(f"    Type: {gov_type}, Duration: {duration_months:.0f} months, n={len(sub)} polls")
        print(f"    Decay: {slope:+.3f} pp/month (r={r_value:.3f}, p={p_value:.4f})")
        print(f"    Estimated: {initial:.1f}% → {final:.1f}% ({total_change:+.1f}pp total)")

    res_df = pd.DataFrame(results)

    # ── Compare coalition vs majority ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMPARISON: COALITION vs MAJORITY/NEAR-MAJORITY")
    print(f"{'=' * 70}")

    coalition = res_df[res_df["type"] == "coalition"]["slope"]
    majority = res_df[res_df["type"].isin(["majority", "near_majority"])]["slope"]

    print(f"\n  Coalition govts ({len(coalition)}): mean decay = {coalition.mean():+.3f} pp/month")
    print(f"  Majority/near-majority ({len(majority)}): mean decay = {majority.mean():+.3f} pp/month")
    if len(coalition) > 2 and len(majority) > 1:
        t, p = ttest_ind(coalition, majority)
        print(f"  t-test: t = {t:.2f}, p = {p:.4f}")
        print(f"  → {'Coalitions decay faster' if coalition.mean() < majority.mean() and p < 0.05 else 'No significant difference'}")

    # All governments: average decay rate
    print(f"\n  Overall mean decay: {res_df['slope'].mean():+.3f} pp/month")
    print(f"  Median decay: {res_df['slope'].median():+.3f} pp/month")

    # ── Term length and decay ─────────────────────────────────────────
    print(f"\n--- TERM LENGTH vs TOTAL DECAY ---")
    if len(res_df) > 3:
        r, p = pearsonr(res_df["duration"], res_df["change"])
        print(f"  r(duration, total change) = {r:.3f}, p = {p:.4f}")

    # ── Visualization ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Decay rates by government
    ax = axes[0]
    colors = {"coalition": "#E67E22", "near_majority": "#3498DB", "majority": "#27AE60"}
    bar_colors = [colors[r["type"]] for _, r in res_df.iterrows()]
    y_pos = range(len(res_df))
    ax.barh(y_pos, res_df["slope"], color=bar_colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(res_df["label"], fontsize=7)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Decay rate (pp/month)")
    ax.set_title("Incumbent Polling Decay by Government")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#E67E22", alpha=0.7, label="Coalition"),
        Patch(facecolor="#3498DB", alpha=0.7, label="Near-majority"),
        Patch(facecolor="#27AE60", alpha=0.7, label="Majority"),
    ], fontsize=8)

    # Panel 2: Trend lines for each government
    ax = axes[1]
    for _, row in res_df.iterrows():
        start = pd.to_datetime([g[0] for g in GOVERNMENTS if g[4] == row["label"]][0])
        end = pd.to_datetime([g[1] for g in GOVERNMENTS if g[4] == row["label"]][0])
        sub = polls[(polls["date"] >= start) & (polls["date"] <= end)].copy()
        party = row["party"]
        sub = sub.dropna(subset=[party])
        if len(sub) > 0:
            sub["months"] = (sub["date"] - start).dt.days / 30.44
            color = "#d32f2f" if party == "Labour" else "#1565c0"
            ax.plot(sub["months"], sub[party], alpha=0.3, color=color, linewidth=0.5)
            # Trend line
            x = np.array([0, sub["months"].max()])
            y = row["initial"] + row["slope"] * x
            ax.plot(x, y, color=color, linewidth=2, alpha=0.7)
    ax.set_xlabel("Months into term")
    ax.set_ylabel("Incumbent party vote (%)")
    ax.set_title("Incumbent Polling Trajectories")

    plt.tight_layout()
    outpath = REPORTS_DIR / "cost_of_ruling.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
