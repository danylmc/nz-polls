#!/usr/bin/env python3
"""
Phase 2: Economic Voting at Quarterly Resolution

Re-tests the "economic voting" hypothesis with 1,016 polls matched to quarterly
GDP and CPI data — 100x more data points than the original annual analysis.

Analyses:
2a. Bivariate relationships (GDP/CPI vs incumbent support)
2b. Asymmetric effects (do recessions hurt more than growth helps?)
2c. Distributed lag model (how fast do voters respond?)
2d. Salient prices (food, housing, petrol vs headline CPI)
2e. Economic conditions and election outcomes
"""

import os
import warnings
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")

PARTY_COLORS = {
    "National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
    "ACT": "#FDE401", "NZ First": "#000000",
}

# NZ election results (party vote %) for the economic forecasting model
ELECTION_RESULTS = {
    1993: {"National": 35.05, "Labour": 34.68, "incumbent": "National", "inc_won": True},
    1996: {"National": 33.84, "Labour": 28.19, "incumbent": "National", "inc_won": True},
    1999: {"National": 30.50, "Labour": 38.74, "incumbent": "National", "inc_won": False},
    2002: {"National": 20.93, "Labour": 41.26, "incumbent": "Labour", "inc_won": True},
    2005: {"National": 39.10, "Labour": 41.10, "incumbent": "Labour", "inc_won": True},
    2008: {"National": 44.93, "Labour": 33.99, "incumbent": "Labour", "inc_won": False},
    2011: {"National": 47.31, "Labour": 27.48, "incumbent": "National", "inc_won": True},
    2014: {"National": 47.04, "Labour": 25.13, "incumbent": "National", "inc_won": True},
    2017: {"National": 44.45, "Labour": 36.89, "incumbent": "National", "inc_won": False},
    2020: {"National": 25.58, "Labour": 50.01, "incumbent": "Labour", "inc_won": True},
    2023: {"National": 38.93, "Labour": 26.91, "incumbent": "Labour", "inc_won": False},
}


def load_data():
    """Load polls-with-economics dataset."""
    df = pd.read_csv(DATA_DIR / "polls_with_economics.csv", parse_dates=["date"])
    # Drop polls without economic data or incumbent support
    df = df.dropna(subset=["incumbent_support", "gdp_growth_qoq"])
    print(f"Loaded {len(df)} polls with economic data")
    return df


def load_quarterly_economics():
    """Load quarterly economics for election outcome analysis."""
    econ = pd.read_csv(DATA_DIR / "quarterly_economics.csv", parse_dates=["quarter"],
                       index_col="quarter")
    return econ


# ─── 2a. Bivariate Relationships ────────────────────────────────────────────────

def analyze_bivariate(df):
    """Correlate incumbent support with economic indicators at various lags."""
    print("\n2a. Bivariate Relationships")
    print("-" * 50)

    econ_vars = {
        "GDP growth (q/q)": "gdp_growth_qoq",
        "GDP growth (y/y)": "gdp_growth_yoy",
        "CPI inflation": "inflation_yoy",
        "Food inflation": "food_inflation",
        "Housing inflation": "housing_inflation",
        "Petrol inflation": "petrol_inflation",
    }

    lags = {"concurrent": "", "1-quarter lag": "_lag1",
            "2-quarter lag": "_lag2", "4-quarter lag": "_lag4"}

    results = []
    for econ_name, econ_base in econ_vars.items():
        for lag_name, lag_suffix in lags.items():
            col = econ_base + lag_suffix
            if col not in df.columns:
                continue
            valid = df[["incumbent_support", col]].dropna()
            if len(valid) < 30:
                continue
            pearson_r, pearson_p = stats.pearsonr(valid["incumbent_support"], valid[col])
            spearman_r, spearman_p = stats.spearmanr(valid["incumbent_support"], valid[col])
            results.append({
                "indicator": econ_name, "lag": lag_name,
                "n": len(valid),
                "pearson_r": pearson_r, "pearson_p": pearson_p,
                "spearman_r": spearman_r, "spearman_p": spearman_p,
            })
            if pearson_p < 0.05:
                print(f"  ** {econ_name} ({lag_name}): r={pearson_r:.3f}, p={pearson_p:.4f}, n={len(valid)}")

    results_df = pd.DataFrame(results)

    # Key scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    scatter_vars = [
        ("gdp_growth_qoq", "GDP Growth (q/q %)"),
        ("gdp_growth_yoy", "GDP Growth (y/y %)"),
        ("inflation_yoy", "CPI Inflation (y/y %)"),
        ("food_inflation", "Food Inflation (y/y %)"),
        ("housing_inflation", "Housing Inflation (y/y %)"),
        ("petrol_inflation", "Petrol Inflation (y/y %)"),
    ]

    for ax, (col, title) in zip(axes.flat, scatter_vars):
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 10:
            ax.set_visible(False)
            continue
        ax.scatter(valid[col], valid["incumbent_support"], alpha=0.15, s=8, color="#333")
        # LOWESS smoother
        try:
            lowess = sm.nonparametric.lowess(valid["incumbent_support"], valid[col], frac=0.4)
            ax.plot(lowess[:, 0], lowess[:, 1], color="red", linewidth=2, label="LOWESS")
        except Exception:
            pass
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        ax.set_xlabel(title, fontsize=9)
        ax.set_ylabel("Incumbent Support (%)")
        ax.set_title(f"{title}\nr={r:.3f}, p={p:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Economic Indicators vs Incumbent Poll Support (1993-2025)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "econ_bivariate_scatter.png", dpi=150)
    plt.close()
    print(f"  Saved scatter plots to {GRAPH_DIR / 'econ_bivariate_scatter.png'}")

    return results_df


# ─── 2b. Asymmetric Effects ────────────────────────────────────────────────────

def analyze_asymmetric(df):
    """Test whether negative shocks hurt incumbents more than positive growth helps."""
    print("\n2b. Asymmetric Effects")
    print("-" * 50)

    results = {}

    # GDP asymmetry
    valid = df[["incumbent_support", "gdp_growth_qoq"]].dropna()
    positive = valid[valid["gdp_growth_qoq"] >= 0]
    negative = valid[valid["gdp_growth_qoq"] < 0]

    if len(positive) > 10 and len(negative) > 10:
        r_pos, p_pos = stats.pearsonr(positive["incumbent_support"], positive["gdp_growth_qoq"])
        r_neg, p_neg = stats.pearsonr(negative["incumbent_support"], negative["gdp_growth_qoq"])
        results["gdp_positive"] = {"r": r_pos, "p": p_pos, "n": len(positive)}
        results["gdp_negative"] = {"r": r_neg, "p": p_neg, "n": len(negative)}
        print(f"  GDP growth periods: r={r_pos:.3f}, p={p_pos:.4f} (n={len(positive)})")
        print(f"  GDP contraction:    r={r_neg:.3f}, p={p_neg:.4f} (n={len(negative)})")

    # Inflation asymmetry — split at median
    valid_inf = df[["incumbent_support", "inflation_yoy"]].dropna()
    median_inf = valid_inf["inflation_yoy"].median()
    low_inf = valid_inf[valid_inf["inflation_yoy"] <= median_inf]
    high_inf = valid_inf[valid_inf["inflation_yoy"] > median_inf]

    if len(low_inf) > 10 and len(high_inf) > 10:
        r_low, p_low = stats.pearsonr(low_inf["incumbent_support"], low_inf["inflation_yoy"])
        r_high, p_high = stats.pearsonr(high_inf["incumbent_support"], high_inf["inflation_yoy"])
        results["inflation_low"] = {"r": r_low, "p": p_low, "n": len(low_inf)}
        results["inflation_high"] = {"r": r_high, "p": p_high, "n": len(high_inf)}
        print(f"  Low inflation (≤{median_inf:.1f}%):  r={r_low:.3f}, p={p_low:.4f}")
        print(f"  High inflation (>{median_inf:.1f}%): r={r_high:.3f}, p={p_high:.4f}")

    # Threshold effects: binned GDP growth
    valid["gdp_bin"] = pd.cut(valid["gdp_growth_qoq"],
                               bins=[-10, -1, 0, 0.5, 1, 2, 10],
                               labels=["< -1%", "-1 to 0%", "0 to 0.5%",
                                        "0.5 to 1%", "1 to 2%", "> 2%"])
    bin_means = valid.groupby("gdp_bin", observed=True)["incumbent_support"].agg(["mean", "count", "std"])
    results["gdp_bins"] = bin_means

    fig, ax = plt.subplots(figsize=(8, 5))
    bin_means_valid = bin_means[bin_means["count"] >= 5]
    if len(bin_means_valid) > 0:
        x = range(len(bin_means_valid))
        ax.bar(x, bin_means_valid["mean"], color="#00529F", alpha=0.7,
               yerr=bin_means_valid["std"] / np.sqrt(bin_means_valid["count"]))
        ax.set_xticks(x)
        ax.set_xticklabels(bin_means_valid.index, rotation=15)
        ax.set_ylabel("Mean Incumbent Support (%)")
        ax.set_xlabel("GDP Growth (q/q)")
        ax.set_title("Incumbent Support by GDP Growth Category")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "econ_asymmetric_gdp.png", dpi=150)
    plt.close()

    return results


# ─── 2c. Distributed Lag Model ──────────────────────────────────────────────────

def analyze_distributed_lag(df):
    """Regress incumbent support on multiple GDP/CPI lags simultaneously."""
    print("\n2c. Distributed Lag Model")
    print("-" * 50)

    if not HAS_STATSMODELS:
        print("  statsmodels not available, skipping OLS")
        return None

    # Government tenure proxy: months since government formed
    from events import ELECTION_DATES
    election_dates = {y: pd.Timestamp(d) for y, d in ELECTION_DATES.items()}

    def months_in_office(poll_date, incumbent):
        # Find the most recent election that brought this incumbent to power
        for year in sorted(election_dates.keys(), reverse=True):
            if election_dates[year] <= poll_date:
                # Check if incumbent was the winner
                result = ELECTION_RESULTS.get(year)
                if result and result.get("incumbent") != incumbent and result.get("inc_won") is False:
                    # New government started
                    return (poll_date - election_dates[year]).days / 30
                elif result and result.get("incumbent") == incumbent and result.get("inc_won"):
                    return (poll_date - election_dates[year]).days / 30
        return None

    # Build regression variables
    lag_cols = [
        "gdp_growth_qoq", "gdp_growth_qoq_lag1", "gdp_growth_qoq_lag2", "gdp_growth_qoq_lag4",
        "inflation_yoy", "inflation_yoy_lag1", "inflation_yoy_lag2", "inflation_yoy_lag4",
    ]
    available_lags = [c for c in lag_cols if c in df.columns]

    # Time trend
    df = df.copy()
    df["year_decimal"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

    reg_cols = available_lags + ["year_decimal"]
    valid = df[["incumbent_support"] + reg_cols].dropna()
    print(f"  N = {len(valid)} polls with all lag data")

    if len(valid) < 50:
        print("  Insufficient data for distributed lag model")
        return None

    X = sm.add_constant(valid[reg_cols])
    y = valid["incumbent_support"]
    model = sm.OLS(y, X).fit(cov_type="HC1")

    print(f"\n  R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}")
    print(f"  F-statistic: {model.fvalue:.2f}, p={model.f_pvalue:.6f}")
    print(f"\n  {'Variable':<30} {'Coef':>8} {'SE':>8} {'t':>8} {'p':>8}")
    print(f"  {'-'*62}")
    for var in model.params.index:
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:<30} {coef:>8.3f} {se:>8.3f} {t:>8.2f} {p:>8.4f} {sig}")

    # Coefficient comparison plot
    gdp_lags = [c for c in available_lags if "gdp" in c]
    inf_lags = [c for c in available_lags if "inflation" in c and "food" not in c
                and "housing" not in c and "petrol" not in c]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, lag_set, title in [
        (axes[0], gdp_lags, "GDP Growth Lag Effects"),
        (axes[1], inf_lags, "Inflation Lag Effects"),
    ]:
        if not lag_set:
            continue
        coefs = [model.params.get(c, 0) for c in lag_set]
        cis = [1.96 * model.bse.get(c, 0) for c in lag_set]
        labels = [c.replace("gdp_growth_qoq", "GDP").replace("inflation_yoy", "Infl")
                  .replace("_lag", " t-") for c in lag_set]
        labels = [l if "t-" in l else l + " t" for l in labels]

        x = range(len(lag_set))
        ax.bar(x, coefs, yerr=cis, color=["#00529F", "#3B7DD8", "#7BAAF0", "#B8D4F8"][:len(lag_set)],
               alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Effect on Incumbent Support (pp)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Distributed Lag Model: How Fast Do Voters Respond?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "econ_distributed_lag.png", dpi=150)
    plt.close()

    return model


# ─── 2d. Salient Prices ────────────────────────────────────────────────────────

def analyze_salient_prices(df):
    """Compare effect sizes of headline CPI vs food vs housing vs petrol."""
    print("\n2d. Salient Prices Analysis")
    print("-" * 50)

    price_vars = {
        "Headline CPI": "inflation_yoy",
        "Food inflation": "food_inflation",
        "Housing inflation": "housing_inflation",
        "Petrol inflation": "petrol_inflation",
    }

    results = {}
    for name, col in price_vars.items():
        if col not in df.columns:
            continue
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[name] = {"r": r, "p": p, "n": len(valid)}
        print(f"  {name:<20}: r={r:.3f}, p={p:.4f}, n={len(valid)}")

    # OLS comparison — regress on each price type separately
    if HAS_STATSMODELS:
        print("\n  OLS coefficients (univariate):")
        fig, ax = plt.subplots(figsize=(8, 5))
        coefs, cis, labels = [], [], []

        for name, col in price_vars.items():
            if col not in df.columns:
                continue
            valid = df[["incumbent_support", col]].dropna()
            if len(valid) < 30:
                continue
            X = sm.add_constant(valid[[col]])
            y = valid["incumbent_support"]
            model = sm.OLS(y, X).fit(cov_type="HC1")
            coef = model.params[col]
            ci = 1.96 * model.bse[col]
            p = model.pvalues[col]
            coefs.append(coef)
            cis.append(ci)
            labels.append(name)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {name:<20}: β={coef:.3f} ± {ci:.3f} {sig}")

        if coefs:
            x = range(len(coefs))
            colors = ["#333", "#d62728", "#ff7f0e", "#2ca02c"]
            ax.barh(x, coefs, xerr=cis, color=colors[:len(coefs)], alpha=0.7, capsize=4)
            ax.set_yticks(x)
            ax.set_yticklabels(labels)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_xlabel("Effect on Incumbent Support (pp per 1% inflation)")
            ax.set_title("Which Prices Affect Voting?\nCoefficient Comparison", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(GRAPH_DIR / "econ_salient_prices.png", dpi=150)
            plt.close()

    # "Felt" inflation composite
    felt_cols = ["food_inflation", "housing_inflation"]
    available_felt = [c for c in felt_cols if c in df.columns]
    if len(available_felt) == 2:
        df_felt = df.copy()
        df_felt["felt_inflation"] = df_felt[available_felt].mean(axis=1)
        valid = df_felt[["incumbent_support", "felt_inflation", "inflation_yoy"]].dropna()
        if len(valid) > 30:
            r_felt, p_felt = stats.pearsonr(valid["incumbent_support"], valid["felt_inflation"])
            r_head, p_head = stats.pearsonr(valid["incumbent_support"], valid["inflation_yoy"])
            print(f"\n  Felt inflation (food+housing avg): r={r_felt:.3f}, p={p_felt:.4f}")
            print(f"  Headline CPI:                      r={r_head:.3f}, p={p_head:.4f}")
            results["felt_inflation"] = {"r": r_felt, "p": p_felt}

    return results


# ─── 2e. Economic Conditions and Election Outcomes ──────────────────────────────

def analyze_election_outcomes(econ_df):
    """Build simple economic forecasting model for NZ elections."""
    print("\n2e. Economic Conditions and Election Outcomes")
    print("-" * 50)

    rows = []
    for year, result in ELECTION_RESULTS.items():
        # Get average economic conditions in the 4 quarters before the election
        election_approx = pd.Timestamp(year=year, month=10, day=1)
        prior_start = election_approx - pd.DateOffset(months=12)

        mask = (econ_df.index >= prior_start) & (econ_df.index <= election_approx)
        period = econ_df[mask]

        if len(period) == 0:
            continue

        inc_party = result["incumbent"]
        inc_vote = result.get(inc_party, None)

        rows.append({
            "year": year,
            "incumbent": inc_party,
            "incumbent_vote": inc_vote,
            "incumbent_won": result["inc_won"],
            "avg_gdp_growth": period["gdp_growth_qoq"].mean() if "gdp_growth_qoq" in period else None,
            "avg_gdp_yoy": period["gdp_growth_yoy"].mean() if "gdp_growth_yoy" in period else None,
            "avg_inflation": period["inflation_yoy"].mean() if "inflation_yoy" in period else None,
        })

    elec_df = pd.DataFrame(rows)
    print(f"\n  Elections with economic data: {len(elec_df)}")
    print(elec_df[["year", "incumbent", "incumbent_vote", "incumbent_won",
                    "avg_gdp_yoy", "avg_inflation"]].to_string(index=False))

    # Correlation between pre-election economics and incumbent vote share
    valid = elec_df.dropna(subset=["incumbent_vote", "avg_gdp_yoy"])
    if len(valid) >= 5:
        r, p = stats.pearsonr(valid["incumbent_vote"], valid["avg_gdp_yoy"])
        print(f"\n  GDP growth vs incumbent vote: r={r:.3f}, p={p:.4f}")

    valid_inf = elec_df.dropna(subset=["incumbent_vote", "avg_inflation"])
    if len(valid_inf) >= 5:
        r, p = stats.pearsonr(valid_inf["incumbent_vote"], valid_inf["avg_inflation"])
        print(f"  Inflation vs incumbent vote:  r={r:.3f}, p={p:.4f}")

    # Visualization: GDP growth vs incumbent vote
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x_col, x_label in [
        (axes[0], "avg_gdp_yoy", "Avg GDP Growth (y/y %, 4 prior quarters)"),
        (axes[1], "avg_inflation", "Avg Inflation (y/y %, 4 prior quarters)"),
    ]:
        valid = elec_df.dropna(subset=["incumbent_vote", x_col])
        if len(valid) < 3:
            continue

        for _, row in valid.iterrows():
            color = "#00529F" if row["incumbent"] == "National" else "#D82A20"
            marker = "o" if row["incumbent_won"] else "x"
            ax.scatter(row[x_col], row["incumbent_vote"], color=color, marker=marker,
                       s=100, zorder=5)
            ax.annotate(str(int(row["year"])), (row[x_col], row["incumbent_vote"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

        if len(valid) >= 3:
            r, p = stats.pearsonr(valid["incumbent_vote"], valid[x_col])
            z = np.polyfit(valid[x_col], valid["incumbent_vote"], 1)
            poly = np.poly1d(z)
            x_range = np.linspace(valid[x_col].min(), valid[x_col].max(), 50)
            ax.plot(x_range, poly(x_range), "--", color="gray", alpha=0.5)
            ax.set_title(f"r={r:.3f}, p={p:.3f}")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Incumbent Party Vote (%)")
        ax.grid(True, alpha=0.3)
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#00529F", label="National inc.", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#D82A20", label="Labour inc.", markersize=8),
            Line2D([0], [0], marker="x", color="gray", label="Lost election", markersize=8, linestyle="None"),
        ]
        ax.legend(handles=legend_elements, fontsize=8)

    plt.suptitle("Pre-Election Economic Conditions vs Incumbent Vote Share", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "econ_election_outcomes.png", dpi=150)
    plt.close()

    return elec_df


# ─── Timeline Visualization ────────────────────────────────────────────────────

def plot_econ_timeline(df, econ_df):
    """GDP growth + inflation + incumbent poll share on a single timeline."""
    print("\nPlotting economic timeline...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: GDP growth
    valid_gdp = econ_df.dropna(subset=["gdp_growth_yoy"])
    if len(valid_gdp) > 0:
        ax = axes[0]
        ax.fill_between(valid_gdp.index, 0, valid_gdp["gdp_growth_yoy"],
                        where=valid_gdp["gdp_growth_yoy"] >= 0, color="#2ca02c", alpha=0.3)
        ax.fill_between(valid_gdp.index, 0, valid_gdp["gdp_growth_yoy"],
                        where=valid_gdp["gdp_growth_yoy"] < 0, color="#d62728", alpha=0.3)
        ax.plot(valid_gdp.index, valid_gdp["gdp_growth_yoy"], color="#333", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("GDP Growth (y/y %)")
        ax.set_title("GDP Growth", fontsize=11)
        ax.grid(True, alpha=0.3)

    # Panel 2: Inflation
    valid_inf = econ_df.dropna(subset=["inflation_yoy"])
    if len(valid_inf) > 0:
        ax = axes[1]
        ax.plot(valid_inf.index, valid_inf["inflation_yoy"], color="#d62728", linewidth=1)
        ax.axhline(2, color="gray", linewidth=0.5, linestyle="--", label="RBNZ target")
        ax.set_ylabel("CPI Inflation (y/y %)")
        ax.set_title("Inflation", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: Incumbent poll support
    ax = axes[2]
    df_sorted = df.sort_values("date")
    ax.scatter(df_sorted["date"], df_sorted["incumbent_support"], alpha=0.1, s=5, color="#333")
    # Rolling average
    df_roll = df_sorted.set_index("date")[["incumbent_support"]].rolling("90D").mean()
    ax.plot(df_roll.index, df_roll["incumbent_support"], color="#00529F", linewidth=1.5)
    ax.set_ylabel("Incumbent Support (%)")
    ax.set_title("Incumbent Poll Support (90-day rolling avg)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Election markers
    from events import ELECTION_DATES
    for year, date_str in ELECTION_DATES.items():
        dt = pd.Timestamp(date_str)
        for ax in axes:
            ax.axvline(dt, color="gray", alpha=0.3, linewidth=0.5)

    plt.suptitle("NZ Economy and Incumbent Support, 1993-2025", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "econ_timeline.png", dpi=150)
    plt.close()
    print(f"  Saved timeline to {GRAPH_DIR / 'econ_timeline.png'}")


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(bivariate_df, asymmetric, lag_model, salient, election_df):
    """Generate markdown report."""
    report = []
    report.append("# Economic Voting at Quarterly Resolution\n")
    report.append("*Re-testing the economic voting hypothesis with 1,000+ polls matched to quarterly GDP/CPI*\n")

    # Bivariate
    report.append("## 2a. Bivariate Relationships\n")
    report.append("Correlations between incumbent poll support and economic indicators:\n")
    report.append("| Indicator | Lag | Pearson r | p-value | Spearman r | n |")
    report.append("|-----------|-----|-----------|---------|------------|---|")
    if bivariate_df is not None:
        for _, row in bivariate_df.sort_values("pearson_p").iterrows():
            sig = "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
            report.append(f"| {row['indicator']} | {row['lag']} | {row['pearson_r']:.3f}{sig} | {row['pearson_p']:.4f} | {row['spearman_r']:.3f} | {row['n']} |")

    report.append(f"\n![Bivariate Scatter](../graphs/econ_bivariate_scatter.png)\n")

    # Asymmetric
    report.append("## 2b. Asymmetric Effects\n")
    if asymmetric:
        if "gdp_positive" in asymmetric and "gdp_negative" in asymmetric:
            report.append(f"- **GDP growth periods** (n={asymmetric['gdp_positive']['n']}): r={asymmetric['gdp_positive']['r']:.3f}, p={asymmetric['gdp_positive']['p']:.4f}")
            report.append(f"- **GDP contraction** (n={asymmetric['gdp_negative']['n']}): r={asymmetric['gdp_negative']['r']:.3f}, p={asymmetric['gdp_negative']['p']:.4f}")
            if abs(asymmetric['gdp_negative']['r']) > abs(asymmetric['gdp_positive']['r']):
                report.append("\n**Finding: Negative GDP shocks have a stronger relationship with incumbent support than positive growth.**\n")
            else:
                report.append("\n**Finding: No clear asymmetry detected — growth and contraction have similar-magnitude relationships.**\n")
    report.append(f"\n![GDP Asymmetry](../graphs/econ_asymmetric_gdp.png)\n")

    # Distributed lag
    report.append("## 2c. Distributed Lag Model\n")
    if lag_model:
        report.append(f"R² = {lag_model.rsquared:.4f}, Adj R² = {lag_model.rsquared_adj:.4f}\n")
        report.append("| Variable | Coefficient | Std Error | t-stat | p-value |")
        report.append("|----------|-------------|-----------|--------|---------|")
        for var in lag_model.params.index:
            coef = lag_model.params[var]
            se = lag_model.bse[var]
            t = lag_model.tvalues[var]
            p = lag_model.pvalues[var]
            sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.append(f"| {var} | {coef:.3f} | {se:.3f} | {t:.2f} | {p:.4f}{sig} |")
    report.append(f"\n![Distributed Lag](../graphs/econ_distributed_lag.png)\n")

    # Salient prices
    report.append("## 2d. Salient Prices\n")
    if salient:
        report.append("| Price Category | Pearson r | p-value | n |")
        report.append("|----------------|-----------|---------|---|")
        for name, vals in salient.items():
            if isinstance(vals, dict) and "r" in vals:
                sig = "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
                n = vals.get("n", "")
                report.append(f"| {name} | {vals['r']:.3f}{sig} | {vals['p']:.4f} | {n} |")
    report.append(f"\n![Salient Prices](../graphs/econ_salient_prices.png)\n")

    # Election outcomes
    report.append("## 2e. Economic Conditions and Election Outcomes\n")
    if election_df is not None:
        report.append("| Year | Incumbent | Inc. Vote % | Won? | Avg GDP (y/y) | Avg Inflation |")
        report.append("|------|-----------|-------------|------|---------------|---------------|")
        for _, row in election_df.iterrows():
            won = "Yes" if row.get("incumbent_won") else "No"
            gdp = f"{row['avg_gdp_yoy']:.1f}%" if pd.notna(row.get("avg_gdp_yoy")) else "N/A"
            inf = f"{row['avg_inflation']:.1f}%" if pd.notna(row.get("avg_inflation")) else "N/A"
            vote = f"{row['incumbent_vote']:.1f}%" if pd.notna(row.get("incumbent_vote")) else "N/A"
            report.append(f"| {int(row['year'])} | {row['incumbent']} | {vote} | {won} | {gdp} | {inf} |")

    report.append(f"\n![Election Outcomes](../graphs/econ_election_outcomes.png)\n")
    report.append(f"![Economic Timeline](../graphs/econ_timeline.png)\n")

    report_text = "\n".join(report)
    report_path = REPORT_DIR / "economic_voting_quarterly.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 2: Economic Voting at Quarterly Resolution")
    print("=" * 70)

    df = load_data()
    econ_df = load_quarterly_economics()

    bivariate_df = analyze_bivariate(df)
    asymmetric = analyze_asymmetric(df)
    lag_model = analyze_distributed_lag(df)
    salient = analyze_salient_prices(df)
    election_df = analyze_election_outcomes(econ_df)
    plot_econ_timeline(df, econ_df)

    generate_report(bivariate_df, asymmetric, lag_model, salient, election_df)

    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
