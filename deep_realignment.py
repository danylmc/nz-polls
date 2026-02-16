#!/usr/bin/env python3
"""
Phase 3: Electoral Realignment Analysis

Tracks how the social bases of party support have shifted over 30 years
using NZES data (1996-2023, 10 elections, 35,000+ respondents).

Analyses:
3a. Demographic predictors by election year (logistic regression)
3b. Specific realignment hypotheses (class, education, gender, age, ethnicity)
3c. Visualizations (coefficient evolution, vote share heatmaps, probability plots)
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

PARTY_COLORS = {
    "National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
    "ACT": "#FDE401", "NZ First": "#000000", "Maori": "#B2001A",
}


def load_nzes():
    """Load harmonized NZES dataset."""
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    print(f"Loaded {len(df)} NZES respondents across {df['election_year'].nunique()} elections")
    return df


# ─── 3a. Demographic Predictors by Election Year ────────────────────────────────

def analyze_demographic_predictors(df):
    """Run logistic regression of vote National (vs Labour) on demographics per year."""
    print("\n3a. Demographic Predictors by Election Year")
    print("-" * 50)

    if not HAS_SM:
        print("  statsmodels not available")
        return None

    # Filter to Nat/Lab voters with demographics
    major = df[df["party_vote"].isin(["National", "Labour"])].copy()
    major["vote_nat"] = (major["party_vote"] == "National").astype(float)

    results = []
    years = sorted(major["election_year"].unique())

    for year in years:
        yr_df = major[major["election_year"] == year].copy()

        # Standardize age within year for comparability
        if "age" in yr_df.columns:
            age_mean = yr_df["age"].mean()
            age_std = yr_df["age"].std()
            if age_std > 0:
                yr_df["age_z"] = (yr_df["age"] - age_mean) / age_std
            else:
                yr_df["age_z"] = 0

        # Build predictor set based on availability
        predictors = []
        if "age_z" in yr_df.columns:
            predictors.append("age_z")
        if "female" in yr_df.columns:
            predictors.append("female")
        if "education" in yr_df.columns:
            predictors.append("education")

        if not predictors:
            continue

        yr_valid = yr_df[["vote_nat"] + predictors].dropna()
        if len(yr_valid) < 50:
            continue

        try:
            X = sm.add_constant(yr_valid[predictors])
            model = Logit(yr_valid["vote_nat"], X).fit(disp=0, maxiter=100)

            row = {"year": year, "n": len(yr_valid)}
            for pred in predictors:
                row[f"{pred}_coef"] = model.params.get(pred, np.nan)
                row[f"{pred}_se"] = model.bse.get(pred, np.nan)
                row[f"{pred}_p"] = model.pvalues.get(pred, np.nan)
            row["pseudo_r2"] = model.prsquared
            results.append(row)

            sig_vars = [p for p in predictors if model.pvalues.get(p, 1) < 0.05]
            print(f"  {year}: n={len(yr_valid)}, R²={model.prsquared:.3f}, sig: {sig_vars}")
        except Exception as e:
            print(f"  {year}: Error - {e}")

    return pd.DataFrame(results)


# ─── 3b. Specific Realignment Hypotheses ────────────────────────────────────────

def analyze_class_dealignment(df):
    """Has income become a weaker predictor of Nat vs Lab vote?"""
    print("\n3b-i. Class Dealignment (Income)")
    print("-" * 50)

    major = df[df["party_vote"].isin(["National", "Labour"])].copy()
    major["vote_nat"] = (major["party_vote"] == "National").astype(float)

    years = sorted(major["election_year"].unique())
    results = []

    for year in years:
        yr = major[(major["election_year"] == year) & major["income_bracket"].notna()]
        if len(yr) < 30:
            continue
        r, p = stats.pointbiserialr(yr["vote_nat"], yr["income_bracket"])
        results.append({"year": year, "r": r, "p": p, "n": len(yr)})
        print(f"  {year}: r={r:.3f}, p={p:.4f}, n={len(yr)}")

    return pd.DataFrame(results) if results else None


def analyze_education_polarization(df):
    """Has education become a stronger predictor, with degree-holders shifting left?"""
    print("\n3b-ii. Education Polarization")
    print("-" * 50)

    major = df[df["party_vote"].isin(["National", "Labour"])].copy()

    years_with_edu = sorted(major[major["education"].notna()]["election_year"].unique())
    results = []

    for year in years_with_edu:
        yr = major[(major["election_year"] == year) & major["education"].notna()]
        if len(yr) < 30:
            continue

        # National vote share by education level
        edu_groups = yr.groupby("education").agg(
            nat_pct=("party_vote", lambda x: (x == "National").mean() * 100),
            n=("party_vote", "count")
        )

        row = {"year": year}
        for edu_level in [0, 1, 2, 3]:
            if edu_level in edu_groups.index:
                row[f"nat_pct_edu{edu_level}"] = edu_groups.loc[edu_level, "nat_pct"]
                row[f"n_edu{edu_level}"] = edu_groups.loc[edu_level, "n"]

        # Education gap: University vs No qualification
        if 3 in edu_groups.index and 0 in edu_groups.index:
            row["edu_gap"] = edu_groups.loc[0, "nat_pct"] - edu_groups.loc[3, "nat_pct"]

        results.append(row)
        gap = row.get("edu_gap", "N/A")
        print(f"  {year}: Nat% no-qual={row.get('nat_pct_edu0', 'N/A'):.0f}, uni={row.get('nat_pct_edu3', 'N/A'):.0f}, gap={gap}")

    return pd.DataFrame(results) if results else None


def analyze_gender_gap(df):
    """Track the gender gap in NZ voting over time."""
    print("\n3b-iii. Gender Gap")
    print("-" * 50)

    major = df[df["party_vote"].isin(["National", "Labour"])].copy()

    years = sorted(major["election_year"].unique())
    results = []

    for year in years:
        yr = major[(major["election_year"] == year) & major["female"].notna()]
        if len(yr) < 30:
            continue

        male = yr[yr["female"] == 0]
        female = yr[yr["female"] == 1]

        male_nat = (male["party_vote"] == "National").mean() * 100
        female_nat = (female["party_vote"] == "National").mean() * 100
        gap = male_nat - female_nat  # Positive = men more National

        results.append({
            "year": year,
            "male_nat_pct": male_nat,
            "female_nat_pct": female_nat,
            "gender_gap": gap,
            "n_male": len(male),
            "n_female": len(female),
        })
        print(f"  {year}: Male Nat%={male_nat:.1f}, Female Nat%={female_nat:.1f}, Gap={gap:+.1f}pp")

    return pd.DataFrame(results) if results else None


def analyze_age_polarization(df):
    """Track the age gradient in voting over time."""
    print("\n3b-iv. Age Polarization")
    print("-" * 50)

    major = df[df["party_vote"].isin(["National", "Labour"])].copy()

    years = sorted(major["election_year"].unique())
    results = []

    for year in years:
        yr = major[(major["election_year"] == year) & major["age_group"].notna()]
        if len(yr) < 30:
            continue

        age_nat = yr.groupby("age_group", observed=True).agg(
            nat_pct=("party_vote", lambda x: (x == "National").mean() * 100),
            n=("party_vote", "count"),
        )

        row = {"year": year}
        for ag in ["18-29", "30-44", "45-59", "60+"]:
            if ag in age_nat.index:
                row[f"nat_pct_{ag}"] = age_nat.loc[ag, "nat_pct"]
                row[f"n_{ag}"] = age_nat.loc[ag, "n"]

        # Age gap: 60+ minus 18-29
        if "60+" in age_nat.index and "18-29" in age_nat.index:
            row["age_gap"] = age_nat.loc["60+", "nat_pct"] - age_nat.loc["18-29", "nat_pct"]

        results.append(row)
        gap = row.get("age_gap", "N/A")
        y_pct = row.get("nat_pct_18-29", "N/A")
        o_pct = row.get("nat_pct_60+", "N/A")
        print(f"  {year}: Nat% 18-29={y_pct:.0f}, 60+={o_pct:.0f}, gap={gap:+.1f}pp")

    return pd.DataFrame(results) if results else None


def analyze_ethnic_voting(df):
    """Track Māori and European voting patterns."""
    print("\n3b-v. Ethnic Voting Patterns")
    print("-" * 50)

    # Filter to years with ethnicity data
    years = sorted(df[df["ethnicity_euro"].notna()]["election_year"].unique())
    results = []

    for year in years:
        yr = df[df["election_year"] == year]

        euro = yr[yr["ethnicity_euro"] == True]
        maori = yr[yr["ethnicity_maori"] == True]

        row = {"year": year}

        # European voting pattern
        if len(euro) > 20:
            euro_votes = euro["party_vote"].value_counts(normalize=True) * 100
            row["euro_national"] = euro_votes.get("National", 0)
            row["euro_labour"] = euro_votes.get("Labour", 0)
            row["n_euro"] = len(euro)

        # Māori voting pattern
        if len(maori) > 20:
            maori_votes = maori["party_vote"].value_counts(normalize=True) * 100
            row["maori_national"] = maori_votes.get("National", 0)
            row["maori_labour"] = maori_votes.get("Labour", 0)
            row["maori_maori_party"] = maori_votes.get("Maori", 0)
            row["n_maori"] = len(maori)

        results.append(row)
        en = row.get("euro_national")
        mn = row.get("maori_national")
        ml = row.get("maori_labour")
        en_s = f"{en:.0f}" if isinstance(en, (int, float)) else "N/A"
        mn_s = f"{mn:.0f}" if isinstance(mn, (int, float)) else "N/A"
        ml_s = f"{ml:.0f}" if isinstance(ml, (int, float)) else "N/A"
        print(f"  {year}: Euro→Nat={en_s}%, Māori→Lab={ml_s}%, Māori→Nat={mn_s}%")

    return pd.DataFrame(results) if results else None


# ─── 3c. Visualizations ────────────────────────────────────────────────────────

def plot_coefficient_evolution(pred_df):
    """Plot how demographic coefficients change across elections."""
    if pred_df is None or pred_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    coef_map = {
        "age_z": ("Age (standardized)", axes[0]),
        "female": ("Female", axes[1]),
        "education": ("Education level", axes[2]),
    }

    for var, (label, ax) in coef_map.items():
        coef_col = f"{var}_coef"
        se_col = f"{var}_se"
        if coef_col not in pred_df.columns:
            ax.set_visible(False)
            continue

        valid = pred_df.dropna(subset=[coef_col])
        if valid.empty:
            ax.set_visible(False)
            continue

        ax.errorbar(valid["year"], valid[coef_col],
                    yerr=1.96 * valid[se_col],
                    fmt="o-", color="#00529F", capsize=3, markersize=5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Election Year")
        ax.set_ylabel("Logit Coefficient")
        ax.set_title(f"{label}\n(+ve = more National)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Demographic Predictors of National vs Labour Vote, 1996-2023",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "realignment_coefficients.png", dpi=150)
    plt.close()


def plot_realignment_dashboard(gender_df, age_df, edu_df, ethnic_df):
    """Multi-panel realignment dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Gender gap
    ax = axes[0, 0]
    if gender_df is not None and not gender_df.empty:
        ax.plot(gender_df["year"], gender_df["male_nat_pct"], "s-", color="#00529F",
                label="Male → National", markersize=5)
        ax.plot(gender_df["year"], gender_df["female_nat_pct"], "o-", color="#D82A20",
                label="Female → National", markersize=5)
        ax.fill_between(gender_df["year"], gender_df["male_nat_pct"],
                        gender_df["female_nat_pct"], alpha=0.1, color="gray")
        ax.set_ylabel("National Vote Share (%)")
        ax.set_title("Gender Gap")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 2. Age polarization
    ax = axes[0, 1]
    if age_df is not None and not age_df.empty:
        for ag, color in [("18-29", "#d62728"), ("30-44", "#ff7f0e"),
                           ("45-59", "#2ca02c"), ("60+", "#00529F")]:
            col = f"nat_pct_{ag}"
            if col in age_df.columns:
                valid = age_df.dropna(subset=[col])
                ax.plot(valid["year"], valid[col], "o-", color=color,
                        label=ag, markersize=5)
        ax.set_ylabel("National Vote Share (%)")
        ax.set_title("Age Groups")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 3. Age gap over time
    ax = axes[0, 2]
    if age_df is not None and "age_gap" in age_df.columns:
        valid = age_df.dropna(subset=["age_gap"])
        ax.bar(valid["year"], valid["age_gap"], color="#00529F", alpha=0.7, width=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Age Gap (60+ minus 18-29, pp)")
        ax.set_title("Age Polarization Gap")
        ax.grid(True, alpha=0.3, axis="y")

    # 4. Education polarization
    ax = axes[1, 0]
    if edu_df is not None and not edu_df.empty:
        edu_labels = {0: "No qual", 1: "School", 2: "Post-school", 3: "University"}
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#00529F"]
        for edu_level, (label, color) in enumerate(zip(edu_labels.values(), colors)):
            col = f"nat_pct_edu{edu_level}"
            if col in edu_df.columns:
                valid = edu_df.dropna(subset=[col])
                ax.plot(valid["year"], valid[col], "o-", color=color,
                        label=label, markersize=5)
        ax.set_ylabel("National Vote Share (%)")
        ax.set_title("Education Groups")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 5. Education gap over time
    ax = axes[1, 1]
    if edu_df is not None and "edu_gap" in edu_df.columns:
        valid = edu_df.dropna(subset=["edu_gap"])
        ax.bar(valid["year"], valid["edu_gap"], color="#00529F", alpha=0.7, width=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Education Gap (No qual minus Uni, pp)")
        ax.set_title("Education Polarization Gap")
        ax.grid(True, alpha=0.3, axis="y")

    # 6. Ethnic voting
    ax = axes[1, 2]
    if ethnic_df is not None and not ethnic_df.empty:
        if "euro_national" in ethnic_df.columns:
            valid = ethnic_df.dropna(subset=["euro_national"])
            ax.plot(valid["year"], valid["euro_national"], "s-", color="#00529F",
                    label="European → National", markersize=5)
        if "maori_labour" in ethnic_df.columns:
            valid = ethnic_df.dropna(subset=["maori_labour"])
            ax.plot(valid["year"], valid["maori_labour"], "o-", color="#D82A20",
                    label="Māori → Labour", markersize=5)
        if "maori_maori_party" in ethnic_df.columns:
            valid = ethnic_df.dropna(subset=["maori_maori_party"])
            ax.plot(valid["year"], valid["maori_maori_party"], "^-", color="#B2001A",
                    label="Māori → Te Pāti Māori", markersize=5)
        ax.set_ylabel("Vote Share (%)")
        ax.set_title("Ethnic Voting Patterns")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Electoral Realignment in New Zealand, 1996-2023",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "realignment_dashboard.png", dpi=150)
    plt.close()
    print(f"  Saved dashboard to {GRAPH_DIR / 'realignment_dashboard.png'}")


# ─── Report ─────────────────────────────────────────────────────────────────────

def generate_report(pred_df, class_df, edu_df, gender_df, age_df, ethnic_df):
    """Generate markdown report."""
    report = []
    report.append("# Electoral Realignment in New Zealand\n")
    report.append("*How the social bases of party support have shifted, 1996-2023*\n")

    # Demographic predictors
    report.append("## 3a. Demographic Predictors of National vs Labour Vote\n")
    report.append("Logistic regression coefficients (positive = more likely to vote National):\n")
    if pred_df is not None and not pred_df.empty:
        report.append("| Year | N | Age (z) | Female | Education | Pseudo R² |")
        report.append("|------|---|---------|--------|-----------|-----------|")
        for _, row in pred_df.iterrows():
            def fmt_coef(base):
                c = row.get(f"{base}_coef")
                p = row.get(f"{base}_p")
                if pd.isna(c):
                    return "—"
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                return f"{c:.3f}{sig}"

            report.append(f"| {int(row['year'])} | {int(row['n'])} | "
                         f"{fmt_coef('age_z')} | {fmt_coef('female')} | "
                         f"{fmt_coef('education')} | {row.get('pseudo_r2', 0):.3f} |")

    report.append(f"\n![Coefficient Evolution](../graphs/realignment_coefficients.png)\n")

    # Class dealignment
    report.append("## 3b-i. Class Dealignment (Income)\n")
    if class_df is not None and not class_df.empty:
        report.append("Correlation between income bracket and National vote:\n")
        report.append("| Year | r | p-value | n |")
        report.append("|------|---|---------|---|")
        for _, row in class_df.iterrows():
            sig = "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            report.append(f"| {int(row['year'])} | {row['r']:.3f}{sig} | {row['p']:.4f} | {int(row['n'])} |")

    # Education
    report.append("\n## 3b-ii. Education Polarization\n")
    if edu_df is not None and not edu_df.empty:
        report.append("National vote share by education level:\n")
        report.append("| Year | No qual | School | Post-school | University | Gap |")
        report.append("|------|---------|--------|-------------|------------|-----|")
        for _, row in edu_df.iterrows():
            vals = [f"{row.get(f'nat_pct_edu{i}', 0):.0f}%" if pd.notna(row.get(f'nat_pct_edu{i}')) else "—"
                    for i in range(4)]
            gap = f"{row.get('edu_gap', 0):+.0f}pp" if pd.notna(row.get("edu_gap")) else "—"
            report.append(f"| {int(row['year'])} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {gap} |")

    # Gender
    report.append("\n## 3b-iii. Gender Gap\n")
    if gender_df is not None and not gender_df.empty:
        report.append("| Year | Male→Nat | Female→Nat | Gap (M-F) |")
        report.append("|------|----------|------------|-----------|")
        for _, row in gender_df.iterrows():
            report.append(f"| {int(row['year'])} | {row['male_nat_pct']:.1f}% | "
                         f"{row['female_nat_pct']:.1f}% | {row['gender_gap']:+.1f}pp |")

    # Age
    report.append("\n## 3b-iv. Age Polarization\n")
    if age_df is not None and not age_df.empty:
        report.append("National vote share by age group:\n")
        report.append("| Year | 18-29 | 30-44 | 45-59 | 60+ | Gap (60+ − 18-29) |")
        report.append("|------|-------|-------|-------|-----|-------------------|")
        for _, row in age_df.iterrows():
            vals = [f"{row.get(f'nat_pct_{ag}', 0):.0f}%" if pd.notna(row.get(f'nat_pct_{ag}'))
                    else "—" for ag in ["18-29", "30-44", "45-59", "60+"]]
            gap = f"{row.get('age_gap', 0):+.0f}pp" if pd.notna(row.get("age_gap")) else "—"
            report.append(f"| {int(row['year'])} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {gap} |")

    # Ethnic
    report.append("\n## 3b-v. Ethnic Voting Patterns\n")
    if ethnic_df is not None and not ethnic_df.empty:
        report.append("| Year | Euro→Nat | Māori→Lab | Māori→TPM |")
        report.append("|------|----------|-----------|-----------|")
        for _, row in ethnic_df.iterrows():
            en = f"{row.get('euro_national', 0):.0f}%" if pd.notna(row.get("euro_national")) else "—"
            ml = f"{row.get('maori_labour', 0):.0f}%" if pd.notna(row.get("maori_labour")) else "—"
            mm = f"{row.get('maori_maori_party', 0):.0f}%" if pd.notna(row.get("maori_maori_party")) else "—"
            report.append(f"| {int(row['year'])} | {en} | {ml} | {mm} |")

    report.append(f"\n![Realignment Dashboard](../graphs/realignment_dashboard.png)\n")

    report_text = "\n".join(report)
    with open(REPORT_DIR / "realignment.md", "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {REPORT_DIR / 'realignment.md'}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 3: Electoral Realignment Analysis")
    print("=" * 70)

    df = load_nzes()

    pred_df = analyze_demographic_predictors(df)
    class_df = analyze_class_dealignment(df)
    edu_df = analyze_education_polarization(df)
    gender_df = analyze_gender_gap(df)
    age_df = analyze_age_polarization(df)
    ethnic_df = analyze_ethnic_voting(df)

    plot_coefficient_evolution(pred_df)
    plot_realignment_dashboard(gender_df, age_df, edu_df, ethnic_df)
    generate_report(pred_df, class_df, edu_df, gender_df, age_df, ethnic_df)

    print("\n" + "=" * 70)
    print("Phase 3 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
