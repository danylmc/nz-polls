#!/usr/bin/env python3
"""
Phase 4: Ideological Dynamics

Has the NZ electorate shifted or polarized?
Uses NZES left-right data (available 1996-2020, with gaps).

Analyses:
4a. Electorate-level trends (mean, median, SD of left-right placement)
4b. Polarization by group (party voters, age, education)
4c. Party perception shifts (where voters place parties on L-R scale)
4d. Affective polarization (party like/dislike thermometers)
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")


def load_nzes():
    """Load harmonized NZES dataset."""
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    return df


# ─── 4a. Electorate-Level Trends ────────────────────────────────────────────────

def analyze_electorate_trends(df):
    """Track mean, median, SD of left-right self-placement over time."""
    print("\n4a. Electorate-Level Left-Right Trends")
    print("-" * 50)

    lr = df[df["lr_self"].notna()].copy()
    years = sorted(lr["election_year"].unique())
    print(f"  Elections with L-R data: {years}")

    results = []
    for year in years:
        yr = lr[lr["election_year"] == year]["lr_self"]
        results.append({
            "year": year,
            "mean": yr.mean(),
            "median": yr.median(),
            "std": yr.std(),
            "iqr": yr.quantile(0.75) - yr.quantile(0.25),
            "n": len(yr),
            "skew": yr.skew(),
        })
        print(f"  {year}: mean={yr.mean():.2f}, sd={yr.std():.2f}, n={len(yr)}")

    results_df = pd.DataFrame(results)

    # KDE overlay plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    for year, color in zip(years, colors):
        yr_data = lr[lr["election_year"] == year]["lr_self"]
        yr_data.plot.kde(ax=ax, label=str(year), color=color, linewidth=1.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Left-Right Self-Placement (0=Left, 10=Right)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Ideological Self-Placement")
    ax.legend(fontsize=8, title="Election")
    ax.grid(True, alpha=0.3)

    # Mean and SD over time
    ax = axes[1]
    ax.errorbar(results_df["year"], results_df["mean"],
                yerr=results_df["std"] / np.sqrt(results_df["n"]) * 1.96,
                fmt="o-", color="#00529F", capsize=3, label="Mean ± 95% CI")
    ax2 = ax.twinx()
    ax2.plot(results_df["year"], results_df["std"], "s--", color="#d62728",
             markersize=5, label="Std Dev")
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Mean L-R Placement", color="#00529F")
    ax2.set_ylabel("Std Deviation (dispersion)", color="#d62728")
    ax.set_title("Ideological Position and Dispersion Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Ideological Dynamics of the NZ Electorate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ideology_trends.png", dpi=150)
    plt.close()

    return results_df


# ─── 4b. Polarization by Group ──────────────────────────────────────────────────

def analyze_polarization_by_group(df):
    """L-R distribution by party voters, age, education."""
    print("\n4b. Polarization by Group")
    print("-" * 50)

    lr = df[df["lr_self"].notna()].copy()
    years = sorted(lr["election_year"].unique())

    # Party voter polarization
    party_results = []
    for year in years:
        yr = lr[lr["election_year"] == year]
        for party in ["National", "Labour", "Green", "ACT", "NZ First"]:
            party_lr = yr[yr["party_vote"] == party]["lr_self"]
            if len(party_lr) >= 10:
                party_results.append({
                    "year": year, "party": party,
                    "mean_lr": party_lr.mean(), "sd_lr": party_lr.std(),
                    "n": len(party_lr),
                })

    party_df = pd.DataFrame(party_results)

    # Party distance: how far apart are National and Labour voters?
    distance_results = []
    for year in years:
        nat = lr[(lr["election_year"] == year) & (lr["party_vote"] == "National")]["lr_self"]
        lab = lr[(lr["election_year"] == year) & (lr["party_vote"] == "Labour")]["lr_self"]
        if len(nat) >= 10 and len(lab) >= 10:
            distance = nat.mean() - lab.mean()
            distance_results.append({"year": year, "nat_lab_distance": distance,
                                     "nat_mean": nat.mean(), "lab_mean": lab.mean()})
            print(f"  {year}: Nat voters={nat.mean():.2f}, Lab voters={lab.mean():.2f}, gap={distance:.2f}")

    distance_df = pd.DataFrame(distance_results)

    # Visualization: party voter positions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    party_colors = {"National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
                    "ACT": "#FDE401", "NZ First": "#000000"}
    for party, color in party_colors.items():
        pdata = party_df[party_df["party"] == party]
        if not pdata.empty:
            ax.plot(pdata["year"], pdata["mean_lr"], "o-", color=color,
                    label=party, markersize=5)
            ax.fill_between(pdata["year"],
                           pdata["mean_lr"] - pdata["sd_lr"],
                           pdata["mean_lr"] + pdata["sd_lr"],
                           alpha=0.1, color=color)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Mean L-R Self-Placement")
    ax.set_title("Ideological Position by Party Voters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)

    # National-Labour distance over time
    ax = axes[1]
    if not distance_df.empty:
        ax.bar(distance_df["year"], distance_df["nat_lab_distance"],
               color="#00529F", alpha=0.7, width=2)
        ax.set_xlabel("Election Year")
        ax.set_ylabel("Ideological Distance (scale points)")
        ax.set_title("National-Labour Voter Ideological Distance")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Ideological Polarization Among Party Voters", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ideology_polarization.png", dpi=150)
    plt.close()

    return party_df, distance_df


# ─── 4c. Party Perception Shifts ────────────────────────────────────────────────

def analyze_party_perceptions(df):
    """Where voters place each party on the L-R scale over time."""
    print("\n4c. Party Perception Shifts")
    print("-" * 50)

    results = []
    for party, col in [("National", "lr_national"), ("Labour", "lr_labour"),
                        ("Green", "lr_green")]:
        valid = df[df[col].notna()]
        years = sorted(valid["election_year"].unique())
        for year in years:
            yr = valid[valid["election_year"] == year][col]
            if len(yr) >= 10:
                results.append({
                    "year": year, "party": party,
                    "mean_placement": yr.mean(), "sd_placement": yr.std(),
                    "n": len(yr),
                })
                print(f"  {year} {party}: placed at {yr.mean():.2f} (sd={yr.std():.2f})")

    results_df = pd.DataFrame(results)

    # Perceived party distance
    fig, ax = plt.subplots(figsize=(10, 6))
    party_colors = {"National": "#00529F", "Labour": "#D82A20", "Green": "#098137"}
    for party, color in party_colors.items():
        pdata = results_df[results_df["party"] == party]
        if not pdata.empty:
            ax.plot(pdata["year"], pdata["mean_placement"], "o-", color=color,
                    label=party, markersize=6, linewidth=2)
            ax.fill_between(pdata["year"],
                           pdata["mean_placement"] - pdata["sd_placement"],
                           pdata["mean_placement"] + pdata["sd_placement"],
                           alpha=0.1, color=color)

    ax.set_xlabel("Election Year")
    ax.set_ylabel("Perceived Position (0=Left, 10=Right)")
    ax.set_title("Where Voters Place Each Party on the Left-Right Scale", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ideology_party_placement.png", dpi=150)
    plt.close()

    return results_df


# ─── 4d. Affective Polarization ─────────────────────────────────────────────────

def analyze_affective_polarization(df):
    """Using party like/dislike thermometers — are voters becoming more negative?"""
    print("\n4d. Affective Polarization")
    print("-" * 50)

    # Affective polarization: in-party rating minus out-party rating
    like_cols = ["party_like_lab", "party_like_nat", "party_like_grn",
                 "party_like_nzf", "party_like_act"]
    party_like_map = {
        "Labour": "party_like_lab",
        "National": "party_like_nat",
        "Green": "party_like_grn",
        "NZ First": "party_like_nzf",
        "ACT": "party_like_act",
    }

    results = []
    years_with_like = sorted(df[df["party_like_lab"].notna()]["election_year"].unique())

    for year in years_with_like:
        yr = df[df["election_year"] == year]

        for own_party, own_col in party_like_map.items():
            voters = yr[(yr["party_vote"] == own_party) & yr[own_col].notna()]
            if len(voters) < 10:
                continue

            in_party_rating = voters[own_col].mean()
            other_cols = [c for p, c in party_like_map.items() if p != own_party]
            out_party_ratings = voters[other_cols].mean().mean()

            results.append({
                "year": year,
                "party": own_party,
                "in_party": in_party_rating,
                "out_party": out_party_ratings,
                "affective_gap": in_party_rating - out_party_ratings,
                "n": len(voters),
            })

    results_df = pd.DataFrame(results)

    # Average affective polarization over time
    yearly_avg = results_df.groupby("year").agg(
        mean_in_party=("in_party", "mean"),
        mean_out_party=("out_party", "mean"),
        mean_gap=("affective_gap", "mean"),
    ).reset_index()

    for _, row in yearly_avg.iterrows():
        print(f"  {int(row['year'])}: in-party={row['mean_in_party']:.2f}, "
              f"out-party={row['mean_out_party']:.2f}, gap={row['mean_gap']:.2f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: In-party vs out-party ratings over time
    ax = axes[0]
    if not yearly_avg.empty:
        ax.plot(yearly_avg["year"], yearly_avg["mean_in_party"], "o-", color="#2ca02c",
                label="Own party rating", markersize=6)
        ax.plot(yearly_avg["year"], yearly_avg["mean_out_party"], "o-", color="#d62728",
                label="Other parties (avg)", markersize=6)
        ax.fill_between(yearly_avg["year"], yearly_avg["mean_in_party"],
                        yearly_avg["mean_out_party"], alpha=0.1, color="gray")
        ax.set_xlabel("Election Year")
        ax.set_ylabel("Mean Rating (0-10)")
        ax.set_title("In-Party vs Out-Party Ratings")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)

    # Panel 2: Affective gap by party
    ax = axes[1]
    party_colors = {"National": "#00529F", "Labour": "#D82A20", "Green": "#098137"}
    for party, color in party_colors.items():
        pdata = results_df[results_df["party"] == party]
        if not pdata.empty:
            ax.plot(pdata["year"], pdata["affective_gap"], "o-", color=color,
                    label=party, markersize=5)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Affective Gap (in-party − out-party)")
    ax.set_title("Affective Polarization by Party")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Affective Polarization in NZ, 2011-2017", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ideology_affective.png", dpi=150)
    plt.close()

    return results_df, yearly_avg


# ─── Report ─────────────────────────────────────────────────────────────────────

def generate_report(trends_df, party_df, distance_df, perception_df, affective_df, affective_avg):
    """Generate markdown report."""
    report = []
    report.append("# Ideological Dynamics in New Zealand\n")
    report.append("*Has the NZ electorate shifted or polarized? (NZES left-right data, 1996-2020)*\n")

    # Electorate trends
    report.append("## 4a. Electorate-Level Trends\n")
    if trends_df is not None and not trends_df.empty:
        report.append("| Year | Mean L-R | Median | Std Dev | IQR | n |")
        report.append("|------|----------|--------|---------|-----|---|")
        for _, row in trends_df.iterrows():
            report.append(f"| {int(row['year'])} | {row['mean']:.2f} | {row['median']:.1f} | "
                         f"{row['std']:.2f} | {row['iqr']:.1f} | {int(row['n'])} |")
    report.append(f"\n![Ideology Trends](../graphs/ideology_trends.png)\n")

    # Polarization by group
    report.append("## 4b. Polarization by Party Voters\n")
    if distance_df is not None and not distance_df.empty:
        report.append("National-Labour voter ideological distance:\n")
        report.append("| Year | National voters | Labour voters | Distance |")
        report.append("|------|-----------------|---------------|----------|")
        for _, row in distance_df.iterrows():
            report.append(f"| {int(row['year'])} | {row['nat_mean']:.2f} | "
                         f"{row['lab_mean']:.2f} | {row['nat_lab_distance']:.2f} |")
    report.append(f"\n![Polarization](../graphs/ideology_polarization.png)\n")

    # Party perceptions
    report.append("## 4c. Party Perception Shifts\n")
    if perception_df is not None and not perception_df.empty:
        report.append("Where voters place each party on the L-R scale:\n")
        # Pivot for cleaner display
        pivot = perception_df.pivot(index="year", columns="party", values="mean_placement")
        report.append("| Year | " + " | ".join(pivot.columns) + " |")
        report.append("|------|" + "|".join(["------"] * len(pivot.columns)) + "|")
        for year, row in pivot.iterrows():
            vals = " | ".join([f"{v:.2f}" if pd.notna(v) else "—" for v in row])
            report.append(f"| {int(year)} | {vals} |")
    report.append(f"\n![Party Placement](../graphs/ideology_party_placement.png)\n")

    # Affective polarization
    report.append("## 4d. Affective Polarization\n")
    if affective_avg is not None and not affective_avg.empty:
        report.append("Average in-party vs out-party ratings:\n")
        report.append("| Year | In-Party | Out-Party | Affective Gap |")
        report.append("|------|----------|-----------|---------------|")
        for _, row in affective_avg.iterrows():
            report.append(f"| {int(row['year'])} | {row['mean_in_party']:.2f} | "
                         f"{row['mean_out_party']:.2f} | {row['mean_gap']:.2f} |")
    report.append(f"\n![Affective Polarization](../graphs/ideology_affective.png)\n")

    report_text = "\n".join(report)
    with open(REPORT_DIR / "ideology.md", "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {REPORT_DIR / 'ideology.md'}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 4: Ideological Dynamics")
    print("=" * 70)

    df = load_nzes()

    trends_df = analyze_electorate_trends(df)
    party_df, distance_df = analyze_polarization_by_group(df)
    perception_df = analyze_party_perceptions(df)
    affective_df, affective_avg = analyze_affective_polarization(df)

    generate_report(trends_df, party_df, distance_df, perception_df, affective_df, affective_avg)

    print("\n" + "=" * 70)
    print("Phase 4 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
