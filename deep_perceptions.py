#!/usr/bin/env python3
"""
Phase 6: Economic Perceptions vs Reality

Do voters accurately perceive the economy? Does reality or perception drive voting?
Uses NZES economic assessments matched to actual quarterly GDP/CPI data.

Analyses:
6a. Perception-reality gap (subjective assessment vs objective indicators)
6b. Partisan filter (do government supporters rate economy more positively?)
6c. What drives voting: perception or reality?
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

# Incumbent party by election year
INCUMBENTS = {
    1996: "National", 1999: "National", 2002: "Labour", 2005: "Labour",
    2008: "Labour", 2011: "National", 2014: "National", 2017: "National",
    2020: "Labour", 2023: "Labour",
}

# Approximate GDP growth (y/y) around each election survey period
# Matched from quarterly_economics.csv: average of 2 quarters around election
ACTUAL_ECONOMY = {
    1996: {"gdp_yoy": 3.7, "inflation": 2.4},
    1999: {"gdp_yoy": 4.3, "inflation": -0.1},
    2002: {"gdp_yoy": 4.5, "inflation": 2.5},
    2005: {"gdp_yoy": 2.4, "inflation": 2.9},
    2008: {"gdp_yoy": 1.1, "inflation": 3.9},
    2011: {"gdp_yoy": 0.3, "inflation": 4.6},
    2014: {"gdp_yoy": 2.5, "inflation": 1.4},
    2017: {"gdp_yoy": 3.6, "inflation": 1.8},
    2020: {"gdp_yoy": -0.8, "inflation": 1.8},
    2023: {"gdp_yoy": 2.9, "inflation": 6.4},
}


def load_nzes():
    """Load harmonized NZES dataset."""
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    return df


# ─── 6a. Perception-Reality Gap ─────────────────────────────────────────────────

def analyze_perception_reality(df):
    """Match respondent economic assessments to actual GDP/CPI."""
    print("\n6a. Perception-Reality Gap")
    print("-" * 50)

    # Use econ_change (1-5 scale: 1=much better, 5=much worse) or econ_state
    econ_var = "econ_change"  # More comparable across years
    valid = df[df[econ_var].notna()].copy()
    years = sorted(valid["election_year"].unique())
    print(f"  Elections with economic perception data: {years}")

    results = []
    for year in years:
        yr = valid[valid["election_year"] == year][econ_var]
        actual = ACTUAL_ECONOMY.get(year, {})

        mean_perception = yr.mean()
        # Invert scale so higher = better perceived economy (to match GDP direction)
        inverted_mean = 6 - mean_perception  # Now 5=much better, 1=much worse

        results.append({
            "year": year,
            "mean_perception_raw": mean_perception,
            "mean_perception_inverted": inverted_mean,
            "n": len(yr),
            "actual_gdp_yoy": actual.get("gdp_yoy"),
            "actual_inflation": actual.get("inflation"),
        })
        print(f"  {year}: perceived={mean_perception:.2f} (raw), "
              f"GDP={actual.get('gdp_yoy', 'N/A')}%, inflation={actual.get('inflation', 'N/A')}%")

    results_df = pd.DataFrame(results)

    # Correlation between perception and reality
    valid_both = results_df.dropna(subset=["actual_gdp_yoy"])
    if len(valid_both) >= 5:
        r, p = stats.pearsonr(valid_both["mean_perception_inverted"],
                               valid_both["actual_gdp_yoy"])
        print(f"\n  Perception vs GDP growth: r={r:.3f}, p={p:.4f}")
        r_inf, p_inf = stats.pearsonr(valid_both["mean_perception_raw"],
                                        valid_both["actual_inflation"])
        print(f"  Perception (raw) vs Inflation: r={r_inf:.3f}, p={p_inf:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid_plot = results_df.dropna(subset=["actual_gdp_yoy"])
    if not valid_plot.empty:
        ax.scatter(valid_plot["actual_gdp_yoy"], valid_plot["mean_perception_inverted"],
                   s=80, color="#00529F", zorder=5)
        for _, row in valid_plot.iterrows():
            ax.annotate(str(int(row["year"])),
                       (row["actual_gdp_yoy"], row["mean_perception_inverted"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        if len(valid_plot) >= 3:
            r, _ = stats.pearsonr(valid_plot["actual_gdp_yoy"],
                                   valid_plot["mean_perception_inverted"])
            ax.set_title(f"Perception vs GDP (r={r:.3f})")
        ax.set_xlabel("Actual GDP Growth (y/y %)")
        ax.set_ylabel("Mean Perceived Economy\n(inverted: 5=much better, 1=much worse)")
        ax.grid(True, alpha=0.3)

    ax = axes[1]
    if not valid_plot.empty:
        ax2_data = results_df.dropna(subset=["actual_inflation"])
        ax.scatter(ax2_data["actual_inflation"], ax2_data["mean_perception_raw"],
                   s=80, color="#d62728", zorder=5)
        for _, row in ax2_data.iterrows():
            ax.annotate(str(int(row["year"])),
                       (row["actual_inflation"], row["mean_perception_raw"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
        if len(ax2_data) >= 3:
            r, _ = stats.pearsonr(ax2_data["actual_inflation"],
                                   ax2_data["mean_perception_raw"])
            ax.set_title(f"Perception vs Inflation (r={r:.3f})")
        ax.set_xlabel("Actual CPI Inflation (y/y %)")
        ax.set_ylabel("Mean Perceived Economy\n(1=much better, 5=much worse)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Do Voters Perceive the Economy Accurately?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "perceptions_vs_reality.png", dpi=150)
    plt.close()

    return results_df


# ─── 6b. Partisan Filter ────────────────────────────────────────────────────────

def analyze_partisan_filter(df):
    """Do government supporters rate the economy more positively?"""
    print("\n6b. Partisan Perceptual Screen")
    print("-" * 50)

    econ_var = "econ_change"
    valid = df[df[econ_var].notna() & df["party_vote"].notna()].copy()

    # Classify as govt supporter or opposition supporter
    valid["govt_supporter"] = valid.apply(
        lambda row: row["party_vote"] == INCUMBENTS.get(row["election_year"]),
        axis=1
    )

    results = []
    years = sorted(valid["election_year"].unique())

    for year in years:
        yr = valid[valid["election_year"] == year]
        govt = yr[yr["govt_supporter"]][econ_var]
        oppo = yr[~yr["govt_supporter"]][econ_var]

        if len(govt) < 10 or len(oppo) < 10:
            continue

        # Lower score = better economy (1=much better, 5=much worse)
        t_stat, p_val = stats.ttest_ind(govt, oppo)
        gap = oppo.mean() - govt.mean()  # Positive = govt supporters more optimistic

        results.append({
            "year": year,
            "govt_mean": govt.mean(),
            "oppo_mean": oppo.mean(),
            "gap": gap,
            "t": t_stat,
            "p": p_val,
            "n_govt": len(govt),
            "n_oppo": len(oppo),
            "incumbent": INCUMBENTS.get(year),
        })
        print(f"  {year} ({INCUMBENTS.get(year, '?')} govt): "
              f"govt={govt.mean():.2f}, oppo={oppo.mean():.2f}, "
              f"gap={gap:.2f}, p={p_val:.4f}")

    results_df = pd.DataFrame(results)

    # Has the partisan screen strengthened?
    if len(results_df) >= 3:
        r, p = stats.pearsonr(results_df["year"], results_df["gap"])
        print(f"\n  Trend in partisan gap: r={r:.3f}, p={p:.4f}")
        if r > 0 and p < 0.05:
            print("  ** Partisan perceptual screen is STRENGTHENING over time")
        elif r < 0 and p < 0.05:
            print("  ** Partisan perceptual screen is WEAKENING over time")
        else:
            print("  No significant trend")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if not results_df.empty:
        ax.plot(results_df["year"], results_df["govt_mean"], "o-", color="#2ca02c",
                label="Govt supporters", markersize=6)
        ax.plot(results_df["year"], results_df["oppo_mean"], "o-", color="#d62728",
                label="Opposition supporters", markersize=6)
        ax.fill_between(results_df["year"], results_df["govt_mean"],
                        results_df["oppo_mean"], alpha=0.1, color="gray")
        ax.set_xlabel("Election Year")
        ax.set_ylabel("Mean Econ Assessment\n(1=much better, 5=much worse)")
        ax.set_title("Economic Perception by Partisan Alignment")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # So "better" is up

    ax = axes[1]
    if not results_df.empty:
        colors = ["#00529F" if inc == "National" else "#D82A20"
                  for inc in results_df["incumbent"]]
        ax.bar(results_df["year"], results_df["gap"], color=colors, alpha=0.7, width=2)
        ax.set_xlabel("Election Year")
        ax.set_ylabel("Partisan Gap (scale points)")
        ax.set_title("Partisan Perceptual Screen\n(+ve = govt supporters more optimistic)")
        ax.grid(True, alpha=0.3, axis="y")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#00529F", label="National govt"),
                           Patch(facecolor="#D82A20", label="Labour govt")],
                  fontsize=8)

    plt.suptitle("The Partisan Filter on Economic Perceptions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "perceptions_partisan.png", dpi=150)
    plt.close()

    return results_df


# ─── 6c. What Drives Voting: Perception or Reality? ─────────────────────────────

def analyze_perception_vs_reality_models(df):
    """Compare models: does perceived or actual economy better predict vote choice?"""
    print("\n6c. What Drives Voting: Perception or Reality?")
    print("-" * 50)

    if not HAS_SM:
        print("  statsmodels not available")
        return None

    # Focus on National vs Labour voters
    major = df[df["party_vote"].isin(["National", "Labour"])].copy()
    major["vote_nat"] = (major["party_vote"] == "National").astype(float)

    # Add actual economic data to each respondent
    major["actual_gdp"] = major["election_year"].map(
        lambda y: ACTUAL_ECONOMY.get(y, {}).get("gdp_yoy"))
    major["actual_inflation"] = major["election_year"].map(
        lambda y: ACTUAL_ECONOMY.get(y, {}).get("inflation"))

    # Is the respondent an incumbent supporter?
    major["is_incumbent"] = major.apply(
        lambda row: 1 if row["party_vote"] == INCUMBENTS.get(row["election_year"]) else 0,
        axis=1
    )

    models = {}

    # Model 1: Actual economy only
    m1_data = major[["vote_nat", "actual_gdp", "actual_inflation", "age", "female"]].dropna()
    if len(m1_data) > 100:
        X1 = sm.add_constant(m1_data[["actual_gdp", "actual_inflation", "age", "female"]])
        try:
            m1 = Logit(m1_data["vote_nat"], X1).fit(disp=0)
            models["actual_only"] = m1
            print(f"  Model 1 (actual economy): Pseudo R² = {m1.prsquared:.4f}, AIC = {m1.aic:.0f}")
        except Exception as e:
            print(f"  Model 1 error: {e}")

    # Model 2: Perceived economy only
    econ_var = "econ_change"
    m2_data = major[["vote_nat", econ_var, "age", "female"]].dropna()
    if len(m2_data) > 100:
        X2 = sm.add_constant(m2_data[[econ_var, "age", "female"]])
        try:
            m2 = Logit(m2_data["vote_nat"], X2).fit(disp=0)
            models["perceived_only"] = m2
            print(f"  Model 2 (perceived economy): Pseudo R² = {m2.prsquared:.4f}, AIC = {m2.aic:.0f}")
        except Exception as e:
            print(f"  Model 2 error: {e}")

    # Model 3: Both
    m3_data = major[["vote_nat", econ_var, "actual_gdp", "actual_inflation",
                      "age", "female"]].dropna()
    if len(m3_data) > 100:
        X3 = sm.add_constant(m3_data[[econ_var, "actual_gdp", "actual_inflation",
                                       "age", "female"]])
        try:
            m3 = Logit(m3_data["vote_nat"], X3).fit(disp=0)
            models["both"] = m3
            print(f"  Model 3 (both): Pseudo R² = {m3.prsquared:.4f}, AIC = {m3.aic:.0f}")
            print(f"\n  Model 3 coefficients:")
            for var in m3.params.index:
                coef = m3.params[var]
                p = m3.pvalues[var]
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {var}: coef={coef:.3f}, p={p:.4f} {sig}")
        except Exception as e:
            print(f"  Model 3 error: {e}")

    # Summary interpretation
    if "perceived_only" in models and "actual_only" in models:
        if models["perceived_only"].prsquared > models["actual_only"].prsquared:
            print(f"\n  ** PERCEPTION dominates: R²={models['perceived_only'].prsquared:.4f} vs {models['actual_only'].prsquared:.4f}")
            print("  This explains why annual economic voting appeared insignificant:")
            print("  voters respond to *felt* economy, not statistics.")
        else:
            print(f"\n  ** REALITY dominates: R²={models['actual_only'].prsquared:.4f} vs {models['perceived_only'].prsquared:.4f}")

    # Visualization: model comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    model_names = []
    r2_values = []
    aic_values = []
    for name, model in models.items():
        model_names.append(name.replace("_", " ").title())
        r2_values.append(model.prsquared)
        aic_values.append(model.aic)

    if model_names:
        x = range(len(model_names))
        bars = ax.bar(x, r2_values, color=["#d62728", "#2ca02c", "#00529F"][:len(model_names)],
                      alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Pseudo R²")
        ax.set_title("Model Comparison: What Predicts Vote Choice?\n(Higher = Better Fit)",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, r2 in zip(bars, r2_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "perceptions_model_comparison.png", dpi=150)
    plt.close()

    return models


# ─── Report ─────────────────────────────────────────────────────────────────────

def generate_report(perception_df, partisan_df, models):
    """Generate markdown report."""
    report = []
    report.append("# Economic Perceptions vs Reality\n")
    report.append("*Do voters accurately perceive the economy? Does reality or perception drive voting?*\n")

    # Perception-reality gap
    report.append("## 6a. Perception-Reality Gap\n")
    if perception_df is not None and not perception_df.empty:
        report.append("| Year | Mean Perception (raw) | Actual GDP (y/y) | Actual Inflation |")
        report.append("|------|----------------------|-----------------|-----------------|")
        for _, row in perception_df.iterrows():
            gdp = f"{row['actual_gdp_yoy']:.1f}%" if pd.notna(row.get("actual_gdp_yoy")) else "—"
            inf = f"{row['actual_inflation']:.1f}%" if pd.notna(row.get("actual_inflation")) else "—"
            report.append(f"| {int(row['year'])} | {row['mean_perception_raw']:.2f} | {gdp} | {inf} |")
    report.append(f"\n![Perception vs Reality](../graphs/perceptions_vs_reality.png)\n")

    # Partisan filter
    report.append("## 6b. Partisan Perceptual Screen\n")
    if partisan_df is not None and not partisan_df.empty:
        report.append("| Year | Incumbent | Govt Supporters | Opposition | Gap | p-value |")
        report.append("|------|-----------|-----------------|------------|-----|---------|")
        for _, row in partisan_df.iterrows():
            sig = "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            report.append(f"| {int(row['year'])} | {row['incumbent']} | "
                         f"{row['govt_mean']:.2f} | {row['oppo_mean']:.2f} | "
                         f"{row['gap']:.2f} | {row['p']:.4f}{sig} |")
    report.append(f"\n![Partisan Filter](../graphs/perceptions_partisan.png)\n")

    # Models
    report.append("## 6c. Perception vs Reality: Which Drives Voting?\n")
    if models:
        report.append("| Model | Pseudo R² | AIC | Key Finding |")
        report.append("|-------|-----------|-----|-------------|")
        for name, model in models.items():
            r2 = model.prsquared
            aic = model.aic
            report.append(f"| {name.replace('_', ' ').title()} | {r2:.4f} | {aic:.0f} | |")

        if "both" in models:
            report.append("\n### Full Model Coefficients\n")
            m = models["both"]
            report.append("| Variable | Coefficient | Std Error | p-value |")
            report.append("|----------|-------------|-----------|---------|")
            for var in m.params.index:
                coef = m.params[var]
                se = m.bse[var]
                p = m.pvalues[var]
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                report.append(f"| {var} | {coef:.3f} | {se:.3f} | {p:.4f}{sig} |")

    report.append(f"\n![Model Comparison](../graphs/perceptions_model_comparison.png)\n")

    report_text = "\n".join(report)
    with open(REPORT_DIR / "perceptions_vs_reality.md", "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {REPORT_DIR / 'perceptions_vs_reality.md'}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 6: Economic Perceptions vs Reality")
    print("=" * 70)

    df = load_nzes()

    perception_df = analyze_perception_reality(df)
    partisan_df = analyze_partisan_filter(df)
    models = analyze_perception_vs_reality_models(df)

    generate_report(perception_df, partisan_df, models)

    print("\n" + "=" * 70)
    print("Phase 6 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
