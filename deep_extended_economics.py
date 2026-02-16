#!/usr/bin/env python3
"""
Extended Economic Analysis: House Prices, Unemployment, and Consumer Confidence

New datasets:
1. Real house prices (BIS via FRED) — does housing wealth help incumbents?
2. Unemployment rate (OECD via FRED) — does joblessness hurt incumbents?
3. Consumer confidence (OECD CCI) — does sentiment predict polls better than reality?

Tests:
A. House prices vs housing costs: opposing effects on incumbent support?
B. Unemployment as economic voting predictor
C. Consumer confidence as a leading indicator of polling shifts
D. Extended distributed lag model with all indicators
E. Which economic indicator best predicts incumbent support?
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

import statsmodels.api as sm

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")

PARTY_COLORS = {
    "National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
    "ACT": "#FDE401", "NZ First": "#000000",
}


# ─── Data Loading ────────────────────────────────────────────────────────────────

def load_house_prices():
    """Load BIS real residential property price index from FRED CSV."""
    df = pd.read_csv(DATA_DIR / "house_prices_real.csv")
    df.columns = ["date", "hpi_real"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["hpi_real"])
    df = df.set_index("date").sort_index()
    # Compute year-on-year growth rate
    df["hpi_growth_yoy"] = df["hpi_real"].pct_change(4) * 100
    print(f"House prices: {len(df)} quarters, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_unemployment():
    """Load OECD unemployment rate from FRED CSV."""
    df = pd.read_csv(DATA_DIR / "unemployment_rate.csv")
    df.columns = ["date", "unemployment_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["unemployment_rate"])
    df = df.set_index("date").sort_index()
    # Compute year-on-year change
    df["unemployment_change_yoy"] = df["unemployment_rate"].diff(4)
    print(f"Unemployment: {len(df)} quarters, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_consumer_confidence():
    """Load OECD composite consumer confidence index."""
    df = pd.read_csv(DATA_DIR / "consumer_confidence.csv")
    # Extract just the date and value columns
    df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["date", "cci"]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["cci"] = pd.to_numeric(df["cci"], errors="coerce")
    df = df.dropna(subset=["cci"])
    df = df.set_index("date").sort_index()
    # Year-on-year change
    df["cci_change_yoy"] = df["cci"].diff(12)
    print(f"Consumer confidence: {len(df)} months, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_polls_with_economics():
    """Load the existing polls-with-economics dataset."""
    df = pd.read_csv(DATA_DIR / "polls_with_economics.csv", parse_dates=["date"])
    df = df.dropna(subset=["incumbent_support"])
    print(f"Polls: {len(df)} with incumbent support")
    return df


def merge_new_indicators(polls, hpi, unemp, cci):
    """Merge new economic indicators into the polls dataset using nearest prior observation."""
    df = polls.copy()

    # House prices — quarterly, match to most recent quarter
    hpi_quarters = hpi.index.sort_values()
    for col in ["hpi_real", "hpi_growth_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = hpi_quarters[hpi_quarters <= poll_date]
            if len(prior) > 0:
                vals.append(hpi.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Also add 4-quarter lag of house price growth
    vals_lag4 = []
    for poll_date in df["date"]:
        prior = hpi_quarters[hpi_quarters <= poll_date]
        if len(prior) > 4:
            vals_lag4.append(hpi.loc[prior[-5], "hpi_growth_yoy"])
        else:
            vals_lag4.append(np.nan)
    df["hpi_growth_yoy_lag4"] = vals_lag4

    # Unemployment — quarterly
    unemp_quarters = unemp.index.sort_values()
    for col in ["unemployment_rate", "unemployment_change_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = unemp_quarters[unemp_quarters <= poll_date]
            if len(prior) > 0:
                vals.append(unemp.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Consumer confidence — monthly, so we can match more precisely
    cci_months = cci.index.sort_values()
    for col in ["cci", "cci_change_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = cci_months[cci_months <= poll_date]
            if len(prior) > 0:
                vals.append(cci.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Also add 3-month lag of CCI
    vals_lag3 = []
    for poll_date in df["date"]:
        prior = cci_months[cci_months <= poll_date]
        if len(prior) > 3:
            vals_lag3.append(cci.loc[prior[-4], "cci"])
        else:
            vals_lag3.append(np.nan)
    df["cci_lag3"] = vals_lag3

    n_valid = df[["hpi_real", "unemployment_rate", "cci"]].notna().sum()
    print(f"\nMerged indicators — polls with valid data:")
    print(f"  House prices:        {n_valid['hpi_real']}")
    print(f"  Unemployment:        {n_valid['unemployment_rate']}")
    print(f"  Consumer confidence: {n_valid['cci']}")

    return df


# ─── Analysis A: House Prices vs Housing Costs ──────────────────────────────────

def analyze_house_prices(df):
    """Test whether house price appreciation helps incumbents (wealth effect)
    while housing costs (CPI Housing) hurt them."""
    print("\n" + "=" * 70)
    print("A. House Prices vs Housing Costs")
    print("=" * 70)

    results = {}

    # Bivariate correlations
    for col, label in [
        ("hpi_growth_yoy", "House price growth (y/y)"),
        ("hpi_growth_yoy_lag4", "House price growth (4Q lag)"),
        ("housing_inflation", "CPI Housing inflation"),
    ]:
        if col not in df.columns:
            continue
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            print(f"  {label}: insufficient data (n={len(valid)})")
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Both in the same model
    both_cols = ["hpi_growth_yoy", "housing_inflation"]
    available = [c for c in both_cols if c in df.columns]
    valid = df[["incumbent_support"] + available].dropna()
    if len(valid) > 50 and len(available) == 2:
        print(f"\n  Joint model (n={len(valid)}):")
        X = sm.add_constant(valid[available])
        y = valid["incumbent_support"]
        model = sm.OLS(y, X).fit(cov_type="HC1")
        results["joint_model"] = model
        for var in available:
            coef = model.params[var]
            p = model.pvalues[var]
            se = model.bse[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:<25}: β={coef:+.3f} (SE={se:.3f}), p={p:.4f} {sig}")
        print(f"    R² = {model.rsquared:.4f}")

    # With controls for incumbent party and time trend
    ctrl_cols = available + ["incumbent_is_national", "year_decimal"]
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    valid_ctrl = df_ctrl[["incumbent_support"] + ctrl_cols].dropna()
    if len(valid_ctrl) > 50:
        print(f"\n  Controlled model (n={len(valid_ctrl)}):")
        X = sm.add_constant(valid_ctrl[ctrl_cols])
        y = valid_ctrl["incumbent_support"]
        model_ctrl = sm.OLS(y, X).fit(cov_type="HC1")
        results["controlled_model"] = model_ctrl
        for var in ctrl_cols:
            coef = model_ctrl.params[var]
            p = model_ctrl.pvalues[var]
            se = model_ctrl.bse[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:<25}: β={coef:+.3f} (SE={se:.3f}), p={p:.4f} {sig}")
        print(f"    R² = {model_ctrl.rsquared:.4f}")

    # Within-government analysis
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        for col, label in [("hpi_growth_yoy", "House price growth"), ("housing_inflation", "Housing costs")]:
            if col not in subset.columns:
                continue
            valid = subset[["incumbent_support", col]].dropna()
            if len(valid) > 20:
                r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  Within {govt:>8} — {label:<20}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, title in [
        (axes[0], "hpi_growth_yoy", "House Price Growth (y/y %)"),
        (axes[1], "housing_inflation", "CPI Housing Inflation (y/y %)"),
        (axes[2], None, "Both in One Model"),
    ]:
        if col is None:
            # Coefficient comparison
            if "joint_model" in results:
                m = results["joint_model"]
                vars_ = ["hpi_growth_yoy", "housing_inflation"]
                coefs = [m.params[v] for v in vars_]
                cis = [1.96 * m.bse[v] for v in vars_]
                colors = ["#2ca02c", "#d62728"]
                ax.barh([0, 1], coefs, xerr=cis, color=colors, alpha=0.7, capsize=5, height=0.5)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["House prices\n(wealth)", "Housing costs\n(CPI)"])
                ax.axvline(0, color="black", linewidth=0.5)
                ax.set_xlabel("Effect on incumbent support (pp per 1%)")
                ax.set_title("Joint Model Coefficients", fontweight="bold")
                ax.grid(True, alpha=0.3, axis="x")
            continue

        if col not in df.columns:
            ax.set_visible(False)
            continue
        valid = df[["incumbent_support", col, "incumbent"]].dropna()
        if len(valid) < 10:
            ax.set_visible(False)
            continue

        colors = valid["incumbent"].map({"National": "#00529F", "Labour": "#D82A20"})
        ax.scatter(valid[col], valid["incumbent_support"], c=colors, alpha=0.15, s=8)
        try:
            lowess = sm.nonparametric.lowess(valid["incumbent_support"], valid[col], frac=0.4)
            ax.plot(lowess[:, 0], lowess[:, 1], color="black", linewidth=2)
        except Exception:
            pass
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        ax.set_xlabel(title)
        ax.set_ylabel("Incumbent Support (%)")
        ax.set_title(f"{title}\nr={r:+.3f}, p={p:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("House Prices vs Housing Costs: Opposing Effects?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext_house_prices.png", dpi=150)
    plt.close()
    print(f"  Saved: {GRAPH_DIR / 'ext_house_prices.png'}")

    return results


# ─── Analysis B: Unemployment ───────────────────────────────────────────────────

def analyze_unemployment(df):
    """Test unemployment as a predictor of incumbent support."""
    print("\n" + "=" * 70)
    print("B. Unemployment and Incumbent Support")
    print("=" * 70)

    results = {}

    # Bivariate
    for col, label in [
        ("unemployment_rate", "Unemployment rate (level)"),
        ("unemployment_change_yoy", "Unemployment change (y/y)"),
    ]:
        if col not in df.columns:
            continue
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Asymmetric: rising vs falling unemployment
    valid = df[["incumbent_support", "unemployment_change_yoy"]].dropna()
    rising = valid[valid["unemployment_change_yoy"] > 0]
    falling = valid[valid["unemployment_change_yoy"] <= 0]
    if len(rising) > 20 and len(falling) > 20:
        r_rise, p_rise = stats.pearsonr(rising["incumbent_support"], rising["unemployment_change_yoy"])
        r_fall, p_fall = stats.pearsonr(falling["incumbent_support"], falling["unemployment_change_yoy"])
        results["rising_unemp"] = {"r": r_rise, "p": p_rise, "n": len(rising)}
        results["falling_unemp"] = {"r": r_fall, "p": p_fall, "n": len(falling)}
        print(f"\n  Asymmetry test:")
        print(f"    Rising unemployment:  r={r_rise:+.3f}, p={p_rise:.4f} (n={len(rising)})")
        print(f"    Falling unemployment: r={r_fall:+.3f}, p={p_fall:.4f} (n={len(falling)})")

    # Controlled model
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    ctrl_cols = ["unemployment_rate", "incumbent_is_national", "year_decimal"]
    valid_ctrl = df_ctrl[["incumbent_support"] + ctrl_cols].dropna()
    if len(valid_ctrl) > 50:
        X = sm.add_constant(valid_ctrl[ctrl_cols])
        y = valid_ctrl["incumbent_support"]
        model = sm.OLS(y, X).fit(cov_type="HC1")
        results["controlled_model"] = model
        print(f"\n  Controlled model (n={len(valid_ctrl)}):")
        for var in ctrl_cols:
            coef = model.params[var]
            p = model.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:<25}: β={coef:+.3f}, p={p:.4f} {sig}")
        print(f"    R² = {model.rsquared:.4f}")

    # Within-government
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        valid = subset[["incumbent_support", "unemployment_rate"]].dropna()
        if len(valid) > 20:
            r, p = stats.pearsonr(valid["incumbent_support"], valid["unemployment_rate"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  Within {govt:>8}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "unemployment_rate", "Unemployment Rate (%)"),
        (axes[1], "unemployment_change_yoy", "Unemployment Change (y/y pp)"),
    ]:
        valid = df[["incumbent_support", col, "incumbent"]].dropna()
        if len(valid) < 10:
            ax.set_visible(False)
            continue
        colors = valid["incumbent"].map({"National": "#00529F", "Labour": "#D82A20"})
        ax.scatter(valid[col], valid["incumbent_support"], c=colors, alpha=0.15, s=8)
        try:
            lowess = sm.nonparametric.lowess(valid["incumbent_support"], valid[col], frac=0.4)
            ax.plot(lowess[:, 0], lowess[:, 1], color="black", linewidth=2)
        except Exception:
            pass
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        ax.set_xlabel(title)
        ax.set_ylabel("Incumbent Support (%)")
        ax.set_title(f"{title}\nr={r:+.3f}, p={p:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Unemployment and Incumbent Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext_unemployment.png", dpi=150)
    plt.close()
    print(f"  Saved: {GRAPH_DIR / 'ext_unemployment.png'}")

    return results


# ─── Analysis C: Consumer Confidence ────────────────────────────────────────────

def analyze_consumer_confidence(df):
    """Test consumer confidence as predictor and leading indicator."""
    print("\n" + "=" * 70)
    print("C. Consumer Confidence and Incumbent Support")
    print("=" * 70)

    results = {}

    # Bivariate
    for col, label in [
        ("cci", "Consumer confidence (level)"),
        ("cci_change_yoy", "CCI change (y/y)"),
        ("cci_lag3", "CCI 3-month lag"),
    ]:
        if col not in df.columns:
            continue
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Partisan filter: does CCI track incumbent support differently by government?
    print(f"\n  Within-government correlations:")
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        valid = subset[["incumbent_support", "cci"]].dropna()
        if len(valid) > 20:
            r, p = stats.pearsonr(valid["incumbent_support"], valid["cci"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {govt:>8}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")
            results[f"within_{govt.lower()}"] = {"r": r, "p": p, "n": len(valid)}

    # CCI vs objective indicators
    print(f"\n  CCI vs objective indicators:")
    for col, label in [
        ("gdp_growth_yoy", "GDP growth (y/y)"),
        ("inflation_yoy", "CPI inflation"),
        ("housing_inflation", "Housing inflation"),
        ("unemployment_rate", "Unemployment rate"),
    ]:
        if col not in df.columns:
            continue
        valid = df[["cci", col]].dropna()
        if len(valid) > 30:
            r, p = stats.pearsonr(valid["cci"], valid[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    CCI vs {label:<22}: r={r:+.3f}, p={p:.4f} {sig}")

    # Controlled model
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    ctrl_cols = ["cci", "incumbent_is_national", "year_decimal"]
    valid_ctrl = df_ctrl[["incumbent_support"] + ctrl_cols].dropna()
    if len(valid_ctrl) > 50:
        X = sm.add_constant(valid_ctrl[ctrl_cols])
        y = valid_ctrl["incumbent_support"]
        model = sm.OLS(y, X).fit(cov_type="HC1")
        results["controlled_model"] = model
        print(f"\n  Controlled model (n={len(valid_ctrl)}):")
        for var in ctrl_cols:
            coef = model.params[var]
            p = model.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:<25}: β={coef:+.3f}, p={p:.4f} {sig}")
        print(f"    R² = {model.rsquared:.4f}")

    # Granger-like test: does lagged CCI predict *changes* in incumbent support?
    print(f"\n  Leading indicator test:")
    df_sorted = df.sort_values("date").copy()
    df_sorted["inc_support_change"] = df_sorted["incumbent_support"].diff()
    valid = df_sorted[["inc_support_change", "cci_lag3", "cci"]].dropna()
    if len(valid) > 50:
        for col, label in [("cci", "Current CCI"), ("cci_lag3", "CCI 3-month lag")]:
            r, p = stats.pearsonr(valid["inc_support_change"], valid[col])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {label} vs poll change: r={r:+.3f}, p={p:.4f} {sig}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in [
        (axes[0], "cci", "Consumer Confidence Index"),
        (axes[1], "cci_change_yoy", "CCI Change (y/y)"),
    ]:
        valid = df[["incumbent_support", col, "incumbent"]].dropna()
        if len(valid) < 10:
            ax.set_visible(False)
            continue
        colors = valid["incumbent"].map({"National": "#00529F", "Labour": "#D82A20"})
        ax.scatter(valid[col], valid["incumbent_support"], c=colors, alpha=0.15, s=8)
        try:
            lowess = sm.nonparametric.lowess(valid["incumbent_support"], valid[col], frac=0.4)
            ax.plot(lowess[:, 0], lowess[:, 1], color="black", linewidth=2)
        except Exception:
            pass
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        ax.set_xlabel(title)
        ax.set_ylabel("Incumbent Support (%)")
        ax.set_title(f"{title}\nr={r:+.3f}, p={p:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Consumer Confidence and Incumbent Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext_consumer_confidence.png", dpi=150)
    plt.close()
    print(f"  Saved: {GRAPH_DIR / 'ext_consumer_confidence.png'}")

    return results


# ─── Analysis D: Horse Race — Which Indicator Wins? ─────────────────────────────

def analyze_horse_race(df):
    """Compare all economic indicators in a single model to find the best predictor."""
    print("\n" + "=" * 70)
    print("D. Horse Race: Which Economic Indicator Best Predicts Incumbent Support?")
    print("=" * 70)

    results = {}

    df_model = df.copy()
    df_model["incumbent_is_national"] = (df_model["incumbent"] == "National").astype(int)
    df_model["year_decimal"] = df_model["date"].dt.year + df_model["date"].dt.dayofyear / 365.25

    # Univariate R² comparison (each indicator alone + controls)
    controls = ["incumbent_is_national", "year_decimal"]
    indicators = [
        ("gdp_growth_yoy", "GDP growth (y/y)"),
        ("inflation_yoy", "CPI inflation"),
        ("housing_inflation", "Housing costs (CPI)"),
        ("hpi_growth_yoy", "House price growth"),
        ("unemployment_rate", "Unemployment rate"),
        ("cci", "Consumer confidence"),
    ]

    print(f"\n  Univariate models (each indicator + controls):")
    print(f"  {'Indicator':<25} {'β':>8} {'p':>8} {'R²':>8} {'Adj R²':>8}  {'n':>5}")
    print(f"  {'-'*68}")

    univariate_results = []
    for col, label in indicators:
        if col not in df_model.columns:
            continue
        model_cols = controls + [col]
        valid = df_model[["incumbent_support"] + model_cols].dropna()
        if len(valid) < 50:
            continue
        X = sm.add_constant(valid[model_cols])
        y = valid["incumbent_support"]
        model = sm.OLS(y, X).fit(cov_type="HC1")
        coef = model.params[col]
        p = model.pvalues[col]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<25} {coef:>+8.3f} {p:>8.4f} {model.rsquared:>8.4f} {model.rsquared_adj:>8.4f}  {len(valid):>5} {sig}")
        univariate_results.append({
            "indicator": label, "col": col, "beta": coef, "p": p,
            "r2": model.rsquared, "adj_r2": model.rsquared_adj, "n": len(valid),
        })

    results["univariate"] = pd.DataFrame(univariate_results)

    # Controls-only baseline
    valid_base = df_model[["incumbent_support"] + controls].dropna()
    if len(valid_base) > 50:
        X = sm.add_constant(valid_base[controls])
        y = valid_base["incumbent_support"]
        base_model = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"\n  Baseline (controls only): R²={base_model.rsquared:.4f}, n={len(valid_base)}")
        results["baseline_r2"] = base_model.rsquared

    # Kitchen sink model: all indicators together
    all_indicator_cols = [col for col, _ in indicators if col in df_model.columns]
    all_cols = controls + all_indicator_cols
    valid_all = df_model[["incumbent_support"] + all_cols].dropna()
    if len(valid_all) > 50:
        X = sm.add_constant(valid_all[all_cols])
        y = valid_all["incumbent_support"]
        full_model = sm.OLS(y, X).fit(cov_type="HC1")
        results["full_model"] = full_model
        print(f"\n  Full model (all indicators + controls, n={len(valid_all)}):")
        print(f"  R² = {full_model.rsquared:.4f}, Adj R² = {full_model.rsquared_adj:.4f}")
        print(f"  {'Variable':<25} {'β':>8} {'SE':>8} {'p':>8}")
        print(f"  {'-'*53}")
        for var in all_cols:
            coef = full_model.params[var]
            se = full_model.bse[var]
            p = full_model.pvalues[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<25} {coef:>+8.3f} {se:>8.3f} {p:>8.4f} {sig}")

    # Visualization: R² comparison
    if len(univariate_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: R² by indicator
        ur = results["univariate"].sort_values("adj_r2", ascending=True)
        ax = axes[0]
        colors = ["#d62728" if p < 0.05 else "#999999" for p in ur["p"]]
        ax.barh(range(len(ur)), ur["adj_r2"], color=colors, alpha=0.7)
        if "baseline_r2" in results:
            ax.axvline(results["baseline_r2"], color="black", linestyle="--", linewidth=1,
                       label=f"Controls only (R²={results['baseline_r2']:.3f})")
        ax.set_yticks(range(len(ur)))
        ax.set_yticklabels(ur["indicator"])
        ax.set_xlabel("Adjusted R²")
        ax.set_title("Which Indicator Adds Most Explanatory Power?", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 2: Standardized coefficients from full model
        if "full_model" in results:
            ax = axes[1]
            fm = results["full_model"]
            indicator_vars = [col for col, _ in indicators if col in fm.params.index]
            # Standardize: multiply by SD of X / SD of Y
            y_std = valid_all["incumbent_support"].std()
            std_coefs = []
            labels = []
            pvals = []
            for var in indicator_vars:
                x_std = valid_all[var].std()
                std_beta = fm.params[var] * x_std / y_std
                std_coefs.append(std_beta)
                label = [l for c, l in indicators if c == var][0]
                labels.append(label)
                pvals.append(fm.pvalues[var])

            colors = ["#d62728" if p < 0.05 else "#999999" for p in pvals]
            ax.barh(range(len(std_coefs)), std_coefs, color=colors, alpha=0.7)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Standardized β (full model)")
            ax.set_title("Standardized Effect Sizes\n(red = p < 0.05)", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.suptitle("Economic Indicator Horse Race", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "ext_horse_race.png", dpi=150)
        plt.close()
        print(f"  Saved: {GRAPH_DIR / 'ext_horse_race.png'}")

    return results


# ─── Analysis E: Timeline Visualization ─────────────────────────────────────────

def plot_extended_timeline(df, hpi, unemp, cci):
    """Multi-panel timeline with all indicators and incumbent support."""
    print("\n" + "=" * 70)
    print("E. Extended Timeline")
    print("=" * 70)

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1: House prices (y/y growth)
    ax = axes[0]
    valid = hpi.dropna(subset=["hpi_growth_yoy"])
    ax.fill_between(valid.index, 0, valid["hpi_growth_yoy"],
                    where=valid["hpi_growth_yoy"] >= 0, color="#2ca02c", alpha=0.3)
    ax.fill_between(valid.index, 0, valid["hpi_growth_yoy"],
                    where=valid["hpi_growth_yoy"] < 0, color="#d62728", alpha=0.3)
    ax.plot(valid.index, valid["hpi_growth_yoy"], color="#333", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("y/y %")
    ax.set_title("Real House Price Growth", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Unemployment
    ax = axes[1]
    ax.plot(unemp.index, unemp["unemployment_rate"], color="#d62728", linewidth=1.5)
    ax.set_ylabel("%")
    ax.set_title("Unemployment Rate", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: Consumer confidence
    ax = axes[2]
    ax.plot(cci.index, cci["cci"], color="#ff7f0e", linewidth=1)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--", label="Neutral (100)")
    ax.set_ylabel("Index")
    ax.set_title("Consumer Confidence (OECD CCI, 100=neutral)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Housing costs (CPI Housing inflation)
    econ = pd.read_csv(DATA_DIR / "quarterly_economics.csv", parse_dates=["quarter"], index_col="quarter")
    valid_h = econ.dropna(subset=["housing_inflation"])
    ax = axes[3]
    ax.plot(valid_h.index, valid_h["housing_inflation"], color="#9467bd", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("y/y %")
    ax.set_title("CPI Housing Inflation (rents, rates, energy)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 5: Incumbent support
    ax = axes[4]
    df_sorted = df.sort_values("date")
    ax.scatter(df_sorted["date"], df_sorted["incumbent_support"], alpha=0.1, s=5, color="#333")
    df_roll = df_sorted.set_index("date")[["incumbent_support"]].rolling("90D").mean()
    ax.plot(df_roll.index, df_roll["incumbent_support"], color="#00529F", linewidth=1.5)
    ax.set_ylabel("%")
    ax.set_title("Incumbent Poll Support (90-day rolling avg)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Election markers on all panels
    from events import ELECTION_DATES
    for year, date_str in ELECTION_DATES.items():
        dt = pd.Timestamp(date_str)
        for ax in axes:
            ax.axvline(dt, color="gray", alpha=0.3, linewidth=0.5)

    # Government shading
    from deep_data import GOVERNMENT_TIMELINE
    for start, end, party in GOVERNMENT_TIMELINE:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        color = "#00529F" if party == "National" else "#D82A20"
        for ax in axes:
            ax.axvspan(start_dt, end_dt, alpha=0.03, color=color)

    plt.suptitle("NZ Economic Indicators and Incumbent Support, 1993-2025",
                 fontsize=14, fontweight="bold")
    plt.xlim(pd.Timestamp("1993-01-01"), pd.Timestamp("2025-12-31"))
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext_timeline.png", dpi=150)
    plt.close()
    print(f"  Saved: {GRAPH_DIR / 'ext_timeline.png'}")


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(hp_results, unemp_results, cci_results, horse_results):
    """Generate markdown report for extended economics analysis."""
    report = []
    report.append("# Extended Economic Analysis: House Prices, Unemployment, and Consumer Confidence\n")
    report.append("*Adding three new datasets to the economic voting analysis*\n")
    report.append("---\n")

    # Section A: House Prices
    report.append("## A. House Prices vs Housing Costs\n")
    report.append("**Key question**: CPI Housing (rents, rates, energy) hurts incumbents. Do rising *house prices* (wealth effect) help them?\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in hp_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")

    if "joint_model" in hp_results:
        m = hp_results["joint_model"]
        report.append(f"\n### Joint Model (R² = {m.rsquared:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in ["hpi_growth_yoy", "housing_inflation"]:
            if var in m.params.index:
                sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")

    if "controlled_model" in hp_results:
        m = hp_results["controlled_model"]
        report.append(f"\n### Controlled Model (+ incumbent party + time trend, R² = {m.rsquared:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in m.params.index:
            if var == "const":
                continue
            sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
            report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")

    report.append(f"\n![House Prices](../graphs/ext_house_prices.png)\n")
    report.append("---\n")

    # Section B: Unemployment
    report.append("## B. Unemployment\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in unemp_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")

    if "controlled_model" in unemp_results:
        m = unemp_results["controlled_model"]
        report.append(f"\n### Controlled Model (R² = {m.rsquared:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in ["unemployment_rate", "incumbent_is_national", "year_decimal"]:
            if var in m.params.index:
                sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")

    report.append(f"\n![Unemployment](../graphs/ext_unemployment.png)\n")
    report.append("---\n")

    # Section C: Consumer Confidence
    report.append("## C. Consumer Confidence\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in cci_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")

    if "controlled_model" in cci_results:
        m = cci_results["controlled_model"]
        report.append(f"\n### Controlled Model (R² = {m.rsquared:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in ["cci", "incumbent_is_national", "year_decimal"]:
            if var in m.params.index:
                sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")

    report.append(f"\n![Consumer Confidence](../graphs/ext_consumer_confidence.png)\n")
    report.append("---\n")

    # Section D: Horse Race
    report.append("## D. Horse Race: Which Indicator Wins?\n")
    if "univariate" in horse_results:
        report.append("### Univariate Models (each indicator + controls)\n")
        report.append("| Indicator | β | p-value | Adj R² | n |")
        report.append("|-----------|---|---------|--------|---|")
        for _, row in horse_results["univariate"].sort_values("adj_r2", ascending=False).iterrows():
            sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            report.append(f"| {row['indicator']} | {row['beta']:+.3f}{sig} | {row['p']:.4f} | {row['adj_r2']:.4f} | {row['n']} |")

    if "full_model" in horse_results:
        m = horse_results["full_model"]
        report.append(f"\n### Full Model — All Indicators Together (R² = {m.rsquared:.4f}, Adj R² = {m.rsquared_adj:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in m.params.index:
            if var == "const":
                continue
            sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
            report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")

    report.append(f"\n![Horse Race](../graphs/ext_horse_race.png)\n")
    report.append(f"\n![Extended Timeline](../graphs/ext_timeline.png)\n")

    report_text = "\n".join(report)
    report_path = REPORT_DIR / "extended_economics.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Extended Economic Analysis")
    print("=" * 70)

    # Load new datasets
    hpi = load_house_prices()
    unemp = load_unemployment()
    cci = load_consumer_confidence()

    # Load existing polls with economics
    polls = load_polls_with_economics()

    # Merge new indicators
    df = merge_new_indicators(polls, hpi, unemp, cci)

    # Run analyses
    hp_results = analyze_house_prices(df)
    unemp_results = analyze_unemployment(df)
    cci_results = analyze_consumer_confidence(df)
    horse_results = analyze_horse_race(df)
    plot_extended_timeline(df, hpi, unemp, cci)

    # Generate report
    generate_report(hp_results, unemp_results, cci_results, horse_results)

    print("\n" + "=" * 70)
    print("Extended Economic Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
