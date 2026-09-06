#!/usr/bin/env python3
"""
Ipsos Issues Monitor Visualizations.

Generates three key visualizations from the Ipsos data:
1. Government performance rating + incumbent vote overlay
2. Issue salience heatmap over time
3. Party capability trends for key issues
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_polls_monthly():
    """Load polling data aggregated to monthly means."""
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


def get_incumbent(date):
    if date < pd.to_datetime("2017-10-26"): return "National"
    if date < pd.to_datetime("2023-11-27"): return "Labour"
    return "National"


def main():
    # Load data
    govt = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    govt["date"] = pd.to_datetime(govt["date"])

    salience = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
    salience["date"] = pd.to_datetime(salience["date"])

    capability = pd.read_csv(DATA_DIR / "ipsos_party_capability.csv")
    capability["date"] = pd.to_datetime(capability["date"])

    polls = load_polls_monthly()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: Government Performance + Incumbent Vote
    # ════════════════════════════════════════════════════════════════════
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Govt performance
    ax1.plot(govt["date"], govt["mean_score"], "o-", color="#2c3e50", linewidth=2,
             markersize=5, label="Govt performance (0-10)", zorder=3)
    ax1.set_ylabel("Government performance rating (0-10)", color="#2c3e50", fontsize=11)
    ax1.set_ylim(3, 8.5)
    ax1.tick_params(axis='y', labelcolor='#2c3e50')

    # Incumbent vote on secondary axis
    ax2 = ax1.twinx()
    # Get incumbent vote for each Ipsos date
    inc_dates = []
    inc_votes = []
    inc_colors = []
    for _, row in govt.iterrows():
        month = row["date"].to_period("M").to_timestamp()
        incumbent = get_incumbent(row["date"])
        if month in polls.index and incumbent in polls.columns:
            inc_dates.append(row["date"])
            inc_votes.append(polls.loc[month, incumbent])
            inc_colors.append("#d32f2f" if incumbent == "Labour" else "#1565c0")

    ax2.scatter(inc_dates, inc_votes, c=inc_colors, s=40, alpha=0.7, zorder=2)
    # Also plot continuous polling lines
    lab_range = polls.loc["2017-09":"2023-11", "Labour"]
    nat_range1 = polls.loc["2017-09":"2017-10", "National"]
    nat_range2 = polls.loc["2023-11":"2025-12", "National"]
    ax2.plot(lab_range.index, lab_range, color="#d32f2f", alpha=0.3, linewidth=1.5)
    ax2.plot(nat_range1.index, nat_range1, color="#1565c0", alpha=0.3, linewidth=1.5)
    ax2.plot(nat_range2.index, nat_range2, color="#1565c0", alpha=0.3, linewidth=1.5)
    ax2.set_ylabel("Incumbent party vote (%)", fontsize=11)
    ax2.set_ylim(20, 55)

    # Government change line
    ax1.axvline(pd.to_datetime("2023-11-27"), color="gray", ls="--", lw=1, alpha=0.7)
    ax1.text(pd.to_datetime("2023-11-27"), 8.2, " Govt\n change", fontsize=8, color="gray")

    # COVID line
    ax1.axvline(pd.to_datetime("2020-03-25"), color="#27ae60", ls=":", lw=1, alpha=0.7)
    ax1.text(pd.to_datetime("2020-03-25"), 8.2, " COVID\n lockdown", fontsize=8, color="#27ae60")

    ax1.set_title("Government Performance Rating vs Incumbent Party Vote (Ipsos, 2017-2025)",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax2.legend(handles=[
        Patch(facecolor="#d32f2f", alpha=0.7, label="Labour (incumbent)"),
        Patch(facecolor="#1565c0", alpha=0.7, label="National (incumbent)"),
    ], loc="upper right", fontsize=9)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ipsos_govt_vs_vote.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {REPORTS_DIR / 'ipsos_govt_vs_vote.png'}")
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: Issue Salience Heatmap
    # ════════════════════════════════════════════════════════════════════
    issues = ["inflation_cost_of_living", "healthcare_hospitals", "the_economy",
              "housing_price", "crime_law_order", "poverty_inequality",
              "unemployment", "climate_change", "education", "race_relations"]
    issue_labels = ["Cost of living", "Healthcare", "Economy", "Housing",
                    "Crime", "Poverty", "Unemployment", "Climate", "Education", "Race relations"]

    # Build matrix
    sal_pivot = salience.set_index("date")[issues].copy()
    sal_pivot.columns = issue_labels

    fig, ax = plt.subplots(figsize=(16, 6))
    # Transpose so issues are rows, dates are columns
    data = sal_pivot.T.values.astype(float)
    dates = sal_pivot.index

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(issue_labels)))
    ax.set_yticklabels(issue_labels, fontsize=9)

    # X-axis: dates
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.strftime("%b\n%Y") if d.month in [1, 7] or i == 0 or i == len(dates)-1
                        else d.strftime("%b") for i, d in enumerate(dates)],
                       fontsize=7, rotation=0)

    # Add value labels
    for i in range(len(issue_labels)):
        for j in range(len(dates)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold" if val > 30 else "normal")

    ax.set_title("Issue Salience Over Time (% naming as top concern, Ipsos Issues Monitor)",
                 fontsize=13, fontweight="bold")

    # Government change line
    govt_change_idx = None
    for i, d in enumerate(dates):
        if d >= pd.to_datetime("2023-09-01") and govt_change_idx is None:
            govt_change_idx = i
    if govt_change_idx is not None:
        ax.axvline(govt_change_idx - 0.5, color="white", ls="--", lw=2)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% naming as top concern", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ipsos_salience_heatmap.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {REPORTS_DIR / 'ipsos_salience_heatmap.png'}")
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Party Capability Trends (key issues)
    # ════════════════════════════════════════════════════════════════════
    key_issues = ["inflation_cost_of_living", "healthcare_hospitals", "crime_law_order",
                  "the_economy", "housing", "poverty_inequality"]
    issue_titles = {"inflation_cost_of_living": "Cost of Living",
                    "healthcare_hospitals": "Healthcare",
                    "crime_law_order": "Crime & Law & Order",
                    "the_economy": "The Economy",
                    "housing": "Housing",
                    "poverty_inequality": "Poverty & Inequality"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    party_colors = {"national": "#1565c0", "labour": "#d32f2f", "green": "#4caf50",
                    "act": "#ffeb3b", "nz_first": "#333333"}

    for ax, issue in zip(axes.flat, key_issues):
        sub = capability[capability["issue"] == issue].sort_values("date")
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        for party, color in party_colors.items():
            if party in sub.columns:
                vals = pd.to_numeric(sub[party], errors="coerce")
                lw = 2.5 if party in ["national", "labour"] else 1.5
                alpha = 1.0 if party in ["national", "labour"] else 0.6
                label = party.replace("_", " ").title()
                if party == "nz_first":
                    label = "NZ First"
                ax.plot(sub["date"], vals, "o-", color=color, linewidth=lw,
                        alpha=alpha, markersize=3, label=label)

        ax.set_title(issue_titles.get(issue, issue), fontsize=11, fontweight="bold")
        ax.set_ylabel("% 'best party'", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis='x', labelsize=8)

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("'Best Party to Manage...' (Ipsos Issues Monitor, 2023-2025)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ipsos_party_capability.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {REPORTS_DIR / 'ipsos_party_capability.png'}")
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 4: Issue salience line chart (top issues only)
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 7))

    top_issues = {
        "inflation_cost_of_living": ("#E74C3C", "Cost of living"),
        "healthcare_hospitals": ("#3498DB", "Healthcare"),
        "crime_law_order": ("#2C3E50", "Crime"),
        "housing_price": ("#E67E22", "Housing"),
        "the_economy": ("#9B59B6", "Economy"),
    }

    for col, (color, label) in top_issues.items():
        vals = pd.to_numeric(salience[col], errors="coerce")
        ax.plot(salience["date"], vals, "o-", color=color, linewidth=2,
                markersize=4, label=label, alpha=0.8)

    # Government change
    ax.axvline(pd.to_datetime("2023-11-27"), color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(pd.to_datetime("2023-12-15"), ax.get_ylim()[1] * 0.95,
            "Govt change", fontsize=9, color="gray")

    # COVID
    ax.axvline(pd.to_datetime("2020-03-25"), color="#27ae60", ls=":", lw=1, alpha=0.7)
    ax.text(pd.to_datetime("2020-04-10"), ax.get_ylim()[1] * 0.90,
            "COVID", fontsize=9, color="#27ae60")

    ax.set_ylabel("% naming as top concern", fontsize=11)
    ax.set_title("Top 5 Issue Salience Trends (Ipsos Issues Monitor, 2018-2025)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ipsos_salience_trends.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {REPORTS_DIR / 'ipsos_salience_trends.png'}")
    plt.close()

    print("\nAll Ipsos visualizations generated.")


if __name__ == "__main__":
    main()
