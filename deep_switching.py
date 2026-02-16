#!/usr/bin/env python3
"""
Phase 5: Vote Switching and Flows

Where do voters go when they switch? What predicts switching?
Uses NZES data on current and previous party vote.

Analyses:
5a. Transition matrices (party-to-party flows each election)
5b. Who switches? (demographics, political engagement, ideology)
5c. Direction of switching (what predicts left vs right switching)
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

try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
    HAS_SM = True
except ImportError:
    HAS_SM = False

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")

MAJOR_PARTIES = ["Labour", "National", "Green", "NZ First", "ACT"]
LEFT_PARTIES = {"Labour", "Green", "Alliance"}
RIGHT_PARTIES = {"National", "ACT", "NZ First"}


def load_nzes():
    """Load harmonized NZES dataset."""
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    return df


# ─── 5a. Transition Matrices ────────────────────────────────────────────────────

def analyze_transitions(df):
    """Build party-to-party transition matrices for each election."""
    print("\n5a. Transition Matrices")
    print("-" * 50)

    # Filter to respondents with both current and previous vote
    switchers = df[df["prev_vote"].notna() & df["party_vote"].notna()].copy()
    years = sorted(switchers["election_year"].unique())
    print(f"  Elections with previous vote data: {years}")
    print(f"  Total respondents with transition data: {len(switchers)}")

    matrices = {}
    retention_data = []

    for year in years:
        yr = switchers[switchers["election_year"] == year]
        if len(yr) < 50:
            continue

        # Build transition matrix
        ct = pd.crosstab(yr["prev_vote"], yr["party_vote"], margins=True, margins_name="Total")
        # Normalize rows to percentages
        ct_pct = pd.crosstab(yr["prev_vote"], yr["party_vote"], normalize="index") * 100

        matrices[year] = ct_pct
        print(f"\n  {year} Transition Matrix (n={len(yr)}):")

        # Retention rates
        for party in MAJOR_PARTIES:
            if party in ct_pct.index and party in ct_pct.columns:
                retention = ct_pct.loc[party, party]
                n_prev = yr[yr["prev_vote"] == party].shape[0]
                retention_data.append({
                    "year": year, "party": party,
                    "retention_pct": retention, "n": n_prev,
                })
                print(f"    {party}: {retention:.0f}% retention (n={n_prev})")

        # Identify largest flows (exclude self-retention)
        for prev_party in ct_pct.index:
            if prev_party in ["Other", "Total"]:
                continue
            for curr_party in ct_pct.columns:
                if curr_party in ["Other", "Total"] or curr_party == prev_party:
                    continue
                flow = ct_pct.loc[prev_party, curr_party] if prev_party in ct_pct.index and curr_party in ct_pct.columns else 0
                n_flow = yr[(yr["prev_vote"] == prev_party) & (yr["party_vote"] == curr_party)].shape[0]
                if flow > 10 and n_flow > 5:
                    print(f"    ** {prev_party} → {curr_party}: {flow:.0f}% ({n_flow} voters)")

    retention_df = pd.DataFrame(retention_data)

    # Visualization: retention rates over time
    fig, ax = plt.subplots(figsize=(10, 6))
    party_colors = {"National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
                    "ACT": "#FDE401", "NZ First": "#000000"}
    for party, color in party_colors.items():
        pdata = retention_df[retention_df["party"] == party]
        if not pdata.empty:
            ax.plot(pdata["year"], pdata["retention_pct"], "o-", color=color,
                    label=party, markersize=6, linewidth=2)

    ax.set_xlabel("Election Year")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("Party Voter Retention Rates, 1999-2023", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "switching_retention.png", dpi=150)
    plt.close()

    return matrices, retention_df


# ─── 5b. Who Switches? ──────────────────────────────────────────────────────────

def analyze_switcher_profile(df):
    """Profile switchers vs loyalists."""
    print("\n5b. Who Switches?")
    print("-" * 50)

    # Define switchers: different current vs previous vote
    valid = df[df["prev_vote"].notna() & df["party_vote"].notna()].copy()
    valid["switched"] = (valid["party_vote"] != valid["prev_vote"]).astype(int)

    overall_switch_rate = valid["switched"].mean() * 100
    print(f"  Overall switching rate: {overall_switch_rate:.1f}% (n={len(valid)})")

    # Switching rate by year
    yearly_switch = valid.groupby("election_year").agg(
        switch_rate=("switched", "mean"),
        n=("switched", "count"),
    ).reset_index()
    yearly_switch["switch_rate"] *= 100
    print(f"\n  Switching rate by year:")
    for _, row in yearly_switch.iterrows():
        print(f"    {int(row['election_year'])}: {row['switch_rate']:.1f}% (n={int(row['n'])})")

    # Compare switchers vs loyalists on demographics
    comparison = {}
    for var in ["age", "female", "education", "lr_self"]:
        if var in valid.columns:
            loyalists = valid[valid["switched"] == 0][var].dropna()
            switchers_data = valid[valid["switched"] == 1][var].dropna()
            if len(loyalists) > 10 and len(switchers_data) > 10:
                t_stat, p_val = stats.ttest_ind(loyalists, switchers_data)
                comparison[var] = {
                    "loyalist_mean": loyalists.mean(),
                    "switcher_mean": switchers_data.mean(),
                    "t": t_stat, "p": p_val,
                }
                sig = "*" if p_val < 0.05 else ""
                print(f"  {var}: loyalists={loyalists.mean():.2f}, "
                      f"switchers={switchers_data.mean():.2f}, p={p_val:.4f} {sig}")

    # Are switchers more centrist?
    lr_valid = valid[valid["lr_self"].notna()]
    if len(lr_valid) > 50:
        lr_valid["lr_extreme"] = abs(lr_valid["lr_self"] - 5)  # Distance from center
        loyalist_extreme = lr_valid[lr_valid["switched"] == 0]["lr_extreme"].mean()
        switcher_extreme = lr_valid[lr_valid["switched"] == 1]["lr_extreme"].mean()
        t_stat, p_val = stats.ttest_ind(
            lr_valid[lr_valid["switched"] == 0]["lr_extreme"],
            lr_valid[lr_valid["switched"] == 1]["lr_extreme"]
        )
        print(f"\n  Ideological extremity:")
        print(f"    Loyalists: {loyalist_extreme:.2f} from center")
        print(f"    Switchers: {switcher_extreme:.2f} from center, p={p_val:.4f}")
        comparison["lr_extreme"] = {
            "loyalist_mean": loyalist_extreme,
            "switcher_mean": switcher_extreme,
            "t": t_stat, "p": p_val,
        }

    # Logistic regression: P(switch)
    if HAS_SM:
        predictors = [v for v in ["age", "female", "education", "lr_self"] if v in valid.columns]
        reg_data = valid[["switched"] + predictors].dropna()
        if len(reg_data) > 50:
            try:
                X = sm.add_constant(reg_data[predictors])
                model = Logit(reg_data["switched"], X).fit(disp=0)
                print(f"\n  Logistic regression: P(switch)")
                print(f"  Pseudo R² = {model.prsquared:.4f}")
                for var in predictors:
                    coef = model.params.get(var, np.nan)
                    p = model.pvalues.get(var, np.nan)
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"    {var}: coef={coef:.3f}, p={p:.4f} {sig}")
            except Exception as e:
                print(f"  Logistic regression error: {e}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Switching rate by year
    ax = axes[0]
    ax.bar(yearly_switch["election_year"], yearly_switch["switch_rate"],
           color="#00529F", alpha=0.7, width=2)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Switching Rate (%)")
    ax.set_title("Vote Switching Rate Over Time")
    ax.grid(True, alpha=0.3, axis="y")

    # Switcher vs loyalist L-R distribution
    ax = axes[1]
    if "lr_self" in valid.columns:
        loyalists_lr = valid[(valid["switched"] == 0) & valid["lr_self"].notna()]["lr_self"]
        switchers_lr = valid[(valid["switched"] == 1) & valid["lr_self"].notna()]["lr_self"]
        if len(loyalists_lr) > 10 and len(switchers_lr) > 10:
            loyalists_lr.plot.kde(ax=ax, color="#2ca02c", label="Loyalists", linewidth=2)
            switchers_lr.plot.kde(ax=ax, color="#d62728", label="Switchers", linewidth=2)
            ax.set_xlabel("Left-Right Self-Placement")
            ax.set_ylabel("Density")
            ax.set_title("Ideological Profile: Switchers vs Loyalists")
            ax.legend(fontsize=9)
            ax.set_xlim(0, 10)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Vote Switching Patterns in NZ Elections", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "switching_profile.png", dpi=150)
    plt.close()

    return comparison, yearly_switch


# ─── 5c. Direction of Switching ──────────────────────────────────────────────────

def analyze_switching_direction(df):
    """Of those who switch, what predicts switching left vs right?"""
    print("\n5c. Direction of Switching")
    print("-" * 50)

    valid = df[df["prev_vote"].notna() & df["party_vote"].notna()].copy()
    valid["switched"] = valid["party_vote"] != valid["prev_vote"]
    switchers = valid[valid["switched"]].copy()

    # Classify switch direction
    def switch_direction(row):
        prev = row["prev_vote"]
        curr = row["party_vote"]
        prev_left = prev in LEFT_PARTIES
        prev_right = prev in RIGHT_PARTIES
        curr_left = curr in LEFT_PARTIES
        curr_right = curr in RIGHT_PARTIES

        if prev_right and curr_left:
            return "right_to_left"
        elif prev_left and curr_right:
            return "left_to_right"
        elif prev_left and curr_left:
            return "within_left"
        elif prev_right and curr_right:
            return "within_right"
        else:
            return "other"

    switchers["direction"] = switchers.apply(switch_direction, axis=1)

    direction_counts = switchers["direction"].value_counts()
    print(f"\n  Switch directions:")
    for d, c in direction_counts.items():
        print(f"    {d}: {c} ({c/len(switchers)*100:.1f}%)")

    # By year
    yearly_direction = switchers.groupby(["election_year", "direction"]).size().unstack(fill_value=0)
    yearly_direction_pct = yearly_direction.div(yearly_direction.sum(axis=1), axis=0) * 100

    print(f"\n  Cross-bloc switching by year:")
    for year in sorted(switchers["election_year"].unique()):
        yr = switchers[switchers["election_year"] == year]
        rtl = len(yr[yr["direction"] == "right_to_left"])
        ltr = len(yr[yr["direction"] == "left_to_right"])
        total = len(yr)
        print(f"    {year}: R→L={rtl} ({rtl/total*100:.0f}%), L→R={ltr} ({ltr/total*100:.0f}%), total={total}")

    # Economic perceptions and switching direction
    cross_bloc = switchers[switchers["direction"].isin(["right_to_left", "left_to_right"])].copy()
    cross_bloc["switched_left"] = (cross_bloc["direction"] == "right_to_left").astype(int)

    if "econ_change" in cross_bloc.columns:
        econ_valid = cross_bloc[cross_bloc["econ_change"].notna()]
        if len(econ_valid) > 20:
            r, p = stats.pointbiserialr(econ_valid["switched_left"], econ_valid["econ_change"])
            print(f"\n  Economic perception and switch direction:")
            print(f"    Correlation (switched left vs econ perception): r={r:.3f}, p={p:.4f}")

    # Visualization: flow diagram (simplified bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    if not yearly_direction_pct.empty:
        key_dirs = ["right_to_left", "left_to_right", "within_left", "within_right"]
        colors = ["#D82A20", "#00529F", "#ff9999", "#9999ff"]
        available_dirs = [d for d in key_dirs if d in yearly_direction_pct.columns]
        years = yearly_direction_pct.index
        bottom = np.zeros(len(years))
        for direction, color in zip(available_dirs, colors):
            vals = yearly_direction_pct[direction].values
            ax.bar(years, vals, bottom=bottom, color=color, alpha=0.7,
                   label=direction.replace("_", " ").title(), width=2)
            bottom += vals

        ax.set_xlabel("Election Year")
        ax.set_ylabel("Share of Switchers (%)")
        ax.set_title("Direction of Vote Switching", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "switching_direction.png", dpi=150)
    plt.close()

    return direction_counts, yearly_direction_pct


# ─── Report ─────────────────────────────────────────────────────────────────────

def generate_report(matrices, retention_df, comparison, yearly_switch,
                    direction_counts, yearly_direction):
    """Generate markdown report."""
    report = []
    report.append("# Vote Switching and Flows in NZ Elections\n")
    report.append("*Where do voters go when they switch? What predicts switching?*\n")

    # Transition matrices
    report.append("## 5a. Transition Matrices\n")
    for year, matrix in sorted(matrices.items()):
        report.append(f"\n### {year}\n")
        # Show only major parties
        parties = [p for p in MAJOR_PARTIES if p in matrix.index and p in matrix.columns]
        if parties:
            sub = matrix.loc[parties, parties].round(0)
            report.append(sub.to_markdown())
            report.append("")

    # Retention rates
    report.append("\n### Retention Rates\n")
    if retention_df is not None and not retention_df.empty:
        pivot = retention_df.pivot(index="year", columns="party", values="retention_pct")
        report.append("| Year | " + " | ".join(pivot.columns) + " |")
        report.append("|------|" + "|".join(["------"] * len(pivot.columns)) + "|")
        for year, row in pivot.iterrows():
            vals = " | ".join([f"{v:.0f}%" if pd.notna(v) else "—" for v in row])
            report.append(f"| {int(year)} | {vals} |")

    report.append(f"\n![Retention Rates](../graphs/switching_retention.png)\n")

    # Switcher profile
    report.append("## 5b. Who Switches?\n")
    if yearly_switch is not None:
        report.append("| Year | Switching Rate | n |")
        report.append("|------|---------------|---|")
        for _, row in yearly_switch.iterrows():
            report.append(f"| {int(row['election_year'])} | {row['switch_rate']:.1f}% | {int(row['n'])} |")

    if comparison:
        report.append("\n### Switcher vs Loyalist Demographics\n")
        report.append("| Variable | Loyalists | Switchers | p-value |")
        report.append("|----------|-----------|-----------|---------|")
        for var, vals in comparison.items():
            sig = "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {var} | {vals['loyalist_mean']:.2f} | "
                         f"{vals['switcher_mean']:.2f} | {vals['p']:.4f}{sig} |")

    report.append(f"\n![Switcher Profile](../graphs/switching_profile.png)\n")

    # Direction
    report.append("## 5c. Direction of Switching\n")
    if direction_counts is not None:
        report.append("| Direction | Count | Share |")
        report.append("|-----------|-------|-------|")
        total = direction_counts.sum()
        for d, c in direction_counts.items():
            report.append(f"| {d.replace('_', ' ').title()} | {c} | {c/total*100:.1f}% |")

    report.append(f"\n![Direction](../graphs/switching_direction.png)\n")

    report_text = "\n".join(report)
    with open(REPORT_DIR / "vote_switching.md", "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {REPORT_DIR / 'vote_switching.md'}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 5: Vote Switching and Flows")
    print("=" * 70)

    df = load_nzes()

    matrices, retention_df = analyze_transitions(df)
    comparison, yearly_switch = analyze_switcher_profile(df)
    direction_counts, yearly_direction = analyze_switching_direction(df)

    generate_report(matrices, retention_df, comparison, yearly_switch,
                    direction_counts, yearly_direction)

    print("\n" + "=" * 70)
    print("Phase 5 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
