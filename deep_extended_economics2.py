#!/usr/bin/env python3
"""
Extended Economic Analysis Part 2: Interest Rates, Dairy Prices, Migration, Exchange Rate

New datasets:
1. Interest rates (3-month interbank, OCR proxy) — does monetary tightening hurt incumbents?
2. FAO Dairy Price Index — does NZ's key commodity export affect incumbent support?
3. Net migration (annual) — does immigration predict NZ First support?
4. NZD/USD exchange rate — does currency weakness hurt incumbents through import prices?

Tests:
A. Interest rates and incumbent support (+ interaction with housing)
B. Dairy prices and incumbent support (commodity channel)
C. Net migration and NZ First support (immigration salience)
D. Exchange rate and incumbent support (import price channel)
E. Updated horse race with all 10 indicators
"""

import json
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


# ─── Data Loading ────────────────────────────────────────────────────────────────

def load_interest_rates():
    """Load NZ 3-month interbank rate (OCR proxy) from FRED."""
    df = pd.read_csv(DATA_DIR / "interest_rates.csv")
    df.columns = ["date", "interest_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["interest_rate"] = pd.to_numeric(df["interest_rate"], errors="coerce")
    df = df.dropna(subset=["interest_rate"])
    df = df.set_index("date").sort_index()
    # Year-on-year change
    df["interest_rate_change_yoy"] = df["interest_rate"].diff(12)
    print(f"Interest rates: {len(df)} months, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_dairy_prices():
    """Load FAO Dairy Price Index."""
    df = pd.read_csv(DATA_DIR / "fao_dairy_prices.csv", skiprows=3)
    # First row after headers might be empty
    df = df.iloc[:, :7]  # Keep only the 7 meaningful columns
    df.columns = ["date", "food_price_idx", "meat_idx", "dairy_idx",
                  "cereals_idx", "oils_idx", "sugar_idx"]
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["date"])
    df["dairy_idx"] = pd.to_numeric(df["dairy_idx"], errors="coerce")
    df = df.dropna(subset=["dairy_idx"])
    df = df.set_index("date").sort_index()
    # Year-on-year change (%)
    df["dairy_change_yoy"] = df["dairy_idx"].pct_change(12) * 100
    print(f"Dairy prices: {len(df)} months, {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_net_migration():
    """Load NZ net migration from World Bank JSON."""
    with open(DATA_DIR / "net_migration.json") as f:
        data = json.load(f)
    records = data[1]
    rows = []
    for r in records:
        if r["value"] is not None:
            rows.append({"year": int(r["date"]), "net_migration": int(r["value"])})
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    print(f"Net migration: {len(df)} years, {df['year'].min()} to {df['year'].max()}")
    return df


def load_exchange_rate():
    """Load NZD/USD daily exchange rate from FRED, resample to monthly."""
    df = pd.read_csv(DATA_DIR / "nzd_usd.csv")
    df.columns = ["date", "nzd_usd"]
    df["date"] = pd.to_datetime(df["date"])
    df["nzd_usd"] = pd.to_numeric(df["nzd_usd"], errors="coerce")
    df = df.dropna(subset=["nzd_usd"])
    df = df.set_index("date").sort_index()
    # Resample to monthly average
    monthly = df.resample("MS").mean()
    monthly = monthly.dropna()
    # Year-on-year change (%)
    monthly["nzd_change_yoy"] = monthly["nzd_usd"].pct_change(12) * 100
    print(f"Exchange rate: {len(monthly)} months, {monthly.index.min().date()} to {monthly.index.max().date()}")
    return monthly


def load_polls():
    """Load polls with economics dataset."""
    df = pd.read_csv(DATA_DIR / "polls_with_economics.csv", parse_dates=["date"])
    df = df.dropna(subset=["incumbent_support"])
    print(f"Polls: {len(df)} with incumbent support")
    return df


def load_all_polls_raw():
    """Load raw poll data for NZ First analysis."""
    polls = []
    for json_file in sorted(DATA_DIR.glob("*_polling.json")):
        if "_pm_" in json_file.name:
            continue
        with open(json_file) as f:
            data = json.load(f)
        for poll in data["polls"]:
            if poll.get("date"):
                row = {"date": poll["date"]}
                for party, pct in poll.get("parties", {}).items():
                    if pct is not None:
                        row[party] = pct
                polls.append(row)
    df = pd.DataFrame(polls)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def merge_new_indicators(polls, interest, dairy, exchange, migration):
    """Merge new indicators into polls dataset."""
    df = polls.copy()

    # Interest rates — monthly
    ir_months = interest.index.sort_values()
    for col in ["interest_rate", "interest_rate_change_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = ir_months[ir_months <= poll_date]
            if len(prior) > 0:
                vals.append(interest.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Dairy prices — monthly
    dairy_months = dairy.index.sort_values()
    for col in ["dairy_idx", "dairy_change_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = dairy_months[dairy_months <= poll_date]
            if len(prior) > 0:
                vals.append(dairy.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Exchange rate — monthly
    fx_months = exchange.index.sort_values()
    for col in ["nzd_usd", "nzd_change_yoy"]:
        vals = []
        for poll_date in df["date"]:
            prior = fx_months[fx_months <= poll_date]
            if len(prior) > 0:
                vals.append(exchange.loc[prior[-1], col])
            else:
                vals.append(np.nan)
        df[col] = vals

    # Net migration — annual, match to poll year
    mig_dict = dict(zip(migration["year"], migration["net_migration"]))
    df["net_migration"] = df["date"].dt.year.map(mig_dict)

    n_valid = df[["interest_rate", "dairy_idx", "nzd_usd", "net_migration"]].notna().sum()
    print(f"\nMerged indicators — polls with valid data:")
    for col in ["interest_rate", "dairy_idx", "nzd_usd", "net_migration"]:
        print(f"  {col:<25}: {n_valid[col]}")

    return df


# ─── Analysis A: Interest Rates ─────────────────────────────────────────────────

def analyze_interest_rates(df):
    """Test interest rates as predictor of incumbent support."""
    print("\n" + "=" * 70)
    print("A. Interest Rates and Incumbent Support")
    print("=" * 70)

    results = {}

    # Bivariate
    for col, label in [
        ("interest_rate", "Interest rate (level)"),
        ("interest_rate_change_yoy", "Interest rate change (y/y)"),
    ]:
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Within-government
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        for col, label in [("interest_rate", "rate level"), ("interest_rate_change_yoy", "rate change")]:
            valid = subset[["incumbent_support", col]].dropna()
            if len(valid) > 20:
                r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  Within {govt:>8} — {label:<15}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Controlled model
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    ctrl_cols = ["interest_rate", "incumbent_is_national", "year_decimal"]
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

    # Interest rate + housing costs + house prices (the housing trinity)
    housing_cols = ["interest_rate", "housing_inflation", "hpi_growth_yoy"]
    avail = [c for c in housing_cols if c in df.columns]
    valid_h = df[["incumbent_support"] + avail].dropna()
    if len(valid_h) > 50 and len(avail) == 3:
        X = sm.add_constant(valid_h[avail])
        y = valid_h["incumbent_support"]
        model_h = sm.OLS(y, X).fit(cov_type="HC1")
        results["housing_trinity"] = model_h
        print(f"\n  Housing trinity model (n={len(valid_h)}):")
        for var in avail:
            coef = model_h.params[var]
            p = model_h.pvalues[var]
            se = model_h.bse[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var:<25}: β={coef:+.3f} (SE={se:.3f}), p={p:.4f} {sig}")
        print(f"    R² = {model_h.rsquared:.4f}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "interest_rate", "3-Month Interbank Rate (%)"),
        (axes[1], "interest_rate_change_yoy", "Rate Change (y/y pp)"),
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

    plt.suptitle("Interest Rates and Incumbent Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext2_interest_rates.png", dpi=150)
    plt.close()

    return results


# ─── Analysis B: Dairy Prices ───────────────────────────────────────────────────

def analyze_dairy(df):
    """Test dairy prices as predictor of incumbent support."""
    print("\n" + "=" * 70)
    print("B. Dairy Prices and Incumbent Support")
    print("=" * 70)

    results = {}

    for col, label in [
        ("dairy_idx", "Dairy price index (level)"),
        ("dairy_change_yoy", "Dairy price change (y/y %)"),
    ]:
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Within-government
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        valid = subset[["incumbent_support", "dairy_change_yoy"]].dropna()
        if len(valid) > 20:
            r, p = stats.pearsonr(valid["incumbent_support"], valid["dairy_change_yoy"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  Within {govt:>8}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Controlled model
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    ctrl_cols = ["dairy_change_yoy", "incumbent_is_national", "year_decimal"]
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

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "dairy_idx", "FAO Dairy Price Index"),
        (axes[1], "dairy_change_yoy", "Dairy Price Change (y/y %)"),
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

    plt.suptitle("Dairy Prices and Incumbent Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext2_dairy.png", dpi=150)
    plt.close()

    return results


# ─── Analysis C: Net Migration and NZ First ─────────────────────────────────────

def analyze_migration(df, migration):
    """Test net migration as predictor of NZ First support."""
    print("\n" + "=" * 70)
    print("C. Net Migration and NZ First Support")
    print("=" * 70)

    results = {}

    # Load raw polls for NZ First data
    raw_polls = load_all_polls_raw()
    raw_polls = raw_polls[raw_polls["date"] >= "1996-01-01"]

    # Attach annual migration to each poll
    mig_dict = dict(zip(migration["year"], migration["net_migration"]))
    raw_polls["net_migration"] = raw_polls["date"].dt.year.map(mig_dict)
    # Also scale to thousands for readability
    raw_polls["net_migration_k"] = raw_polls["net_migration"] / 1000

    # NZ First support vs migration
    nzf_col = "NZ First" if "NZ First" in raw_polls.columns else None
    if nzf_col is None:
        # Try case variations
        for col in raw_polls.columns:
            if "first" in col.lower() and "nz" in col.lower():
                nzf_col = col
                break

    if nzf_col and nzf_col in raw_polls.columns:
        valid = raw_polls[[nzf_col, "net_migration_k"]].dropna()
        if len(valid) > 30:
            r, p = stats.pearsonr(valid[nzf_col], valid["net_migration_k"])
            results["nzf_vs_migration"] = {"r": r, "p": p, "n": len(valid)}
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  NZ First support vs net migration: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Migration vs incumbent support
    valid_inc = df[["incumbent_support", "net_migration"]].dropna()
    if len(valid_inc) > 30:
        r, p = stats.pearsonr(valid_inc["incumbent_support"], valid_inc["net_migration"])
        results["incumbent_vs_migration"] = {"r": r, "p": p, "n": len(valid_inc)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Incumbent support vs net migration: r={r:+.3f}, p={p:.4f}, n={len(valid_inc)} {sig}")

    # Within-government
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        valid = subset[["incumbent_support", "net_migration"]].dropna()
        if len(valid) > 20:
            r, p = stats.pearsonr(valid["incumbent_support"], valid["net_migration"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  Within {govt:>8} — incumbent vs migration: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Migration by era
    print(f"\n  Net migration by era:")
    for start, end, label in [
        (1996, 2002, "1996-2002 (National then Labour)"),
        (2003, 2008, "2003-2008 (Labour)"),
        (2009, 2017, "2009-2017 (National)"),
        (2018, 2024, "2018-2024 (Labour then National)"),
    ]:
        era = migration[(migration["year"] >= start) & (migration["year"] <= end)]
        if len(era) > 0:
            print(f"    {label}: mean={era['net_migration'].mean():,.0f}/year")

    # NZ First vs migration scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if nzf_col and nzf_col in raw_polls.columns:
        valid = raw_polls[[nzf_col, "net_migration_k"]].dropna()
        if len(valid) > 10:
            ax = axes[0]
            ax.scatter(valid["net_migration_k"], valid[nzf_col], alpha=0.15, s=8, color="#333")
            try:
                lowess = sm.nonparametric.lowess(valid[nzf_col], valid["net_migration_k"], frac=0.5)
                ax.plot(lowess[:, 0], lowess[:, 1], color="black", linewidth=2)
            except Exception:
                pass
            r, p = stats.pearsonr(valid[nzf_col], valid["net_migration_k"])
            ax.set_xlabel("Net Migration (thousands/year)")
            ax.set_ylabel("NZ First Support (%)")
            ax.set_title(f"NZ First vs Net Migration\nr={r:+.3f}, p={p:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)

    # Incumbent vs migration
    valid_inc = df[["incumbent_support", "net_migration", "incumbent"]].dropna()
    valid_inc["net_migration_k"] = valid_inc["net_migration"] / 1000
    if len(valid_inc) > 10:
        ax = axes[1]
        colors = valid_inc["incumbent"].map({"National": "#00529F", "Labour": "#D82A20"})
        ax.scatter(valid_inc["net_migration_k"], valid_inc["incumbent_support"],
                   c=colors, alpha=0.15, s=8)
        try:
            lowess = sm.nonparametric.lowess(valid_inc["incumbent_support"],
                                             valid_inc["net_migration_k"], frac=0.5)
            ax.plot(lowess[:, 0], lowess[:, 1], color="black", linewidth=2)
        except Exception:
            pass
        r, p = stats.pearsonr(valid_inc["incumbent_support"], valid_inc["net_migration_k"])
        ax.set_xlabel("Net Migration (thousands/year)")
        ax.set_ylabel("Incumbent Support (%)")
        ax.set_title(f"Incumbent Support vs Net Migration\nr={r:+.3f}, p={p:.4f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Net Migration and Political Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext2_migration.png", dpi=150)
    plt.close()

    return results


# ─── Analysis D: Exchange Rate ──────────────────────────────────────────────────

def analyze_exchange_rate(df):
    """Test NZD/USD as predictor of incumbent support."""
    print("\n" + "=" * 70)
    print("D. Exchange Rate and Incumbent Support")
    print("=" * 70)

    results = {}

    for col, label in [
        ("nzd_usd", "NZD/USD level"),
        ("nzd_change_yoy", "NZD/USD change (y/y %)"),
    ]:
        valid = df[["incumbent_support", col]].dropna()
        if len(valid) < 30:
            continue
        r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
        results[label] = {"r": r, "p": p, "n": len(valid)}
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35}: r={r:+.3f}, p={p:.4f}, n={len(valid)} {sig}")

    # Within-government
    for govt in ["National", "Labour"]:
        subset = df[df["incumbent"] == govt]
        valid = subset[["incumbent_support", "nzd_usd"]].dropna()
        if len(valid) > 20:
            r, p = stats.pearsonr(valid["incumbent_support"], valid["nzd_usd"])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  Within {govt:>8}: r={r:+.3f}, p={p:.4f} (n={len(valid)}) {sig}")

    # Controlled model
    df_ctrl = df.copy()
    df_ctrl["incumbent_is_national"] = (df_ctrl["incumbent"] == "National").astype(int)
    df_ctrl["year_decimal"] = df_ctrl["date"].dt.year + df_ctrl["date"].dt.dayofyear / 365.25
    ctrl_cols = ["nzd_usd", "incumbent_is_national", "year_decimal"]
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

    # FX vs inflation (import price channel)
    if "inflation_yoy" in df.columns:
        valid = df[["nzd_usd", "inflation_yoy"]].dropna()
        if len(valid) > 30:
            r, p = stats.pearsonr(valid["nzd_usd"], valid["inflation_yoy"])
            print(f"\n  NZD/USD vs CPI inflation: r={r:+.3f}, p={p:.4f} (weaker NZD → higher imports?)")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "nzd_usd", "NZD/USD Exchange Rate"),
        (axes[1], "nzd_change_yoy", "NZD/USD Change (y/y %)"),
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

    plt.suptitle("Exchange Rate and Incumbent Support", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext2_exchange_rate.png", dpi=150)
    plt.close()

    return results


# ─── Analysis E: Updated Horse Race ─────────────────────────────────────────────

def analyze_full_horse_race(df):
    """Extended horse race with all 10 indicators."""
    print("\n" + "=" * 70)
    print("E. Full Horse Race: 10 Economic Indicators")
    print("=" * 70)

    results = {}

    df_model = df.copy()
    df_model["incumbent_is_national"] = (df_model["incumbent"] == "National").astype(int)
    df_model["year_decimal"] = df_model["date"].dt.year + df_model["date"].dt.dayofyear / 365.25

    controls = ["incumbent_is_national", "year_decimal"]
    indicators = [
        ("gdp_growth_yoy", "GDP growth (y/y)"),
        ("inflation_yoy", "CPI inflation"),
        ("housing_inflation", "Housing costs (CPI)"),
        ("hpi_growth_yoy", "House price growth"),
        ("unemployment_rate", "Unemployment rate"),
        ("cci", "Consumer confidence"),
        ("interest_rate", "Interest rate"),
        ("dairy_change_yoy", "Dairy price change"),
        ("nzd_usd", "NZD/USD exchange rate"),
        ("net_migration", "Net migration"),
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
            print(f"  {label:<25} {'--':>8} {'--':>8} {'--':>8} {'--':>8}  {len(valid):>5} (insufficient)")
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

    # Visualization
    if len(univariate_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ur = results["univariate"].sort_values("adj_r2", ascending=True)
        colors = ["#d62728" if p < 0.05 else "#999999" for p in ur["p"]]
        ax.barh(range(len(ur)), ur["adj_r2"], color=colors, alpha=0.7)
        ax.set_yticks(range(len(ur)))
        ax.set_yticklabels(ur["indicator"])
        ax.set_xlabel("Adjusted R² (each indicator + government identity + time trend)")
        ax.set_title("Full Horse Race: 10 Economic Indicators\n(red = significant at p < 0.05)",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "ext2_full_horse_race.png", dpi=150)
        plt.close()
        print(f"\n  Saved: {GRAPH_DIR / 'ext2_full_horse_race.png'}")

    return results


# ─── Analysis F: Extended Timeline ──────────────────────────────────────────────

def plot_extended_timeline(df, interest, dairy, exchange, migration):
    """Multi-panel timeline with new indicators."""
    print("\n" + "=" * 70)
    print("F. Extended Timeline (New Indicators)")
    print("=" * 70)

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1: Interest rate
    ax = axes[0]
    ax.plot(interest.index, interest["interest_rate"], color="#d62728", linewidth=1.5)
    ax.set_ylabel("%")
    ax.set_title("3-Month Interbank Rate (OCR proxy)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Dairy price index
    ax = axes[1]
    ax.plot(dairy.index, dairy["dairy_idx"], color="#8c564b", linewidth=1.5)
    ax.set_ylabel("Index\n(2014-16=100)")
    ax.set_title("FAO Dairy Price Index", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 3: NZD/USD
    ax = axes[2]
    ax.plot(exchange.index, exchange["nzd_usd"], color="#2ca02c", linewidth=1.5)
    ax.set_ylabel("USD per NZD")
    ax.set_title("NZD/USD Exchange Rate", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: Net migration
    ax = axes[3]
    ax.bar(migration["year"], migration["net_migration"] / 1000, color="#ff7f0e", alpha=0.7)
    ax.set_ylabel("Thousands")
    ax.set_title("Annual Net Migration", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 5: Incumbent support
    ax = axes[4]
    df_sorted = df.sort_values("date")
    ax.scatter(df_sorted["date"], df_sorted["incumbent_support"], alpha=0.1, s=5, color="#333")
    df_roll = df_sorted.set_index("date")[["incumbent_support"]].rolling("90D").mean()
    ax.plot(df_roll.index, df_roll["incumbent_support"], color="#00529F", linewidth=1.5)
    ax.set_ylabel("%")
    ax.set_title("Incumbent Poll Support (90-day rolling avg)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Government shading and election markers
    from deep_data import GOVERNMENT_TIMELINE
    from events import ELECTION_DATES
    for start, end, party in GOVERNMENT_TIMELINE:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        color = "#00529F" if party == "National" else "#D82A20"
        for ax in axes:
            ax.axvspan(start_dt, end_dt, alpha=0.03, color=color)
    for year, date_str in ELECTION_DATES.items():
        dt = pd.Timestamp(date_str)
        for ax in axes:
            ax.axvline(dt, color="gray", alpha=0.3, linewidth=0.5)

    plt.xlim(pd.Timestamp("1993-01-01"), pd.Timestamp("2025-12-31"))
    plt.suptitle("NZ: Interest Rates, Dairy, FX, Migration & Incumbent Support",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "ext2_timeline.png", dpi=150)
    plt.close()
    print(f"  Saved: {GRAPH_DIR / 'ext2_timeline.png'}")


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(ir_results, dairy_results, mig_results, fx_results, horse_results):
    """Generate markdown report."""
    report = []
    report.append("# Extended Economic Analysis Part 2: Interest Rates, Dairy, Migration, Exchange Rate\n")
    report.append("*Four additional datasets tested against incumbent polling support*\n")
    report.append("---\n")

    # A: Interest Rates
    report.append("## A. Interest Rates\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in ir_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")
    if "housing_trinity" in ir_results:
        m = ir_results["housing_trinity"]
        report.append(f"\n### Housing Trinity Model (interest rate + housing costs + house prices, R² = {m.rsquared:.4f})\n")
        report.append("| Variable | β | SE | p-value |")
        report.append("|----------|---|-------|---------|")
        for var in ["interest_rate", "housing_inflation", "hpi_growth_yoy"]:
            if var in m.params.index:
                sig = "***" if m.pvalues[var] < 0.001 else "**" if m.pvalues[var] < 0.01 else "*" if m.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {m.params[var]:+.3f}{sig} | {m.bse[var]:.3f} | {m.pvalues[var]:.4f} |")
    report.append(f"\n![Interest Rates](../graphs/ext2_interest_rates.png)\n")
    report.append("---\n")

    # B: Dairy
    report.append("## B. Dairy Prices\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in dairy_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")
    report.append(f"\n![Dairy Prices](../graphs/ext2_dairy.png)\n")
    report.append("---\n")

    # C: Migration
    report.append("## C. Net Migration\n")
    for label, vals in mig_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"- **{label}**: r={vals['r']:+.3f}, p={vals['p']:.4f}, n={vals['n']} {sig}")
    report.append(f"\n![Migration](../graphs/ext2_migration.png)\n")
    report.append("---\n")

    # D: Exchange Rate
    report.append("## D. Exchange Rate\n")
    report.append("### Bivariate Correlations\n")
    report.append("| Indicator | r | p-value | n |")
    report.append("|-----------|---|---------|---|")
    for label, vals in fx_results.items():
        if isinstance(vals, dict) and "r" in vals:
            sig = "***" if vals["p"] < 0.001 else "**" if vals["p"] < 0.01 else "*" if vals["p"] < 0.05 else ""
            report.append(f"| {label} | {vals['r']:+.3f}{sig} | {vals['p']:.4f} | {vals['n']} |")
    report.append(f"\n![Exchange Rate](../graphs/ext2_exchange_rate.png)\n")
    report.append("---\n")

    # E: Horse Race
    report.append("## E. Full Horse Race: 10 Indicators\n")
    if "univariate" in horse_results:
        report.append("### Univariate Models (each indicator + controls)\n")
        report.append("| Rank | Indicator | β | p-value | Adj R² | n |")
        report.append("|------|-----------|---|---------|--------|---|")
        for rank, (_, row) in enumerate(horse_results["univariate"].sort_values("adj_r2", ascending=False).iterrows(), 1):
            sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
            report.append(f"| {rank} | {row['indicator']} | {row['beta']:+.3f}{sig} | {row['p']:.4f} | {row['adj_r2']:.4f} | {row['n']} |")
    report.append(f"\n![Horse Race](../graphs/ext2_full_horse_race.png)\n")
    report.append(f"\n![Timeline](../graphs/ext2_timeline.png)\n")

    report_text = "\n".join(report)
    report_path = REPORT_DIR / "extended_economics_2.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Extended Economic Analysis Part 2")
    print("=" * 70)

    # Load new datasets
    interest = load_interest_rates()
    dairy = load_dairy_prices()
    migration = load_net_migration()
    exchange = load_exchange_rate()

    # Load existing polls (already has house prices, unemployment, CCI from part 1)
    # We need to re-merge from the base polls_with_economics + part 1 indicators
    from deep_extended_economics import load_house_prices, load_unemployment, load_consumer_confidence
    hpi = load_house_prices()
    unemp = load_unemployment()
    cci = load_consumer_confidence()

    polls = pd.read_csv(DATA_DIR / "polls_with_economics.csv", parse_dates=["date"])
    polls = polls.dropna(subset=["incumbent_support"])
    print(f"Base polls: {len(polls)}")

    # Merge part 1 indicators
    from deep_extended_economics import merge_new_indicators as merge_part1
    df = merge_part1(polls, hpi, unemp, cci)

    # Merge part 2 indicators
    df = merge_new_indicators(df, interest, dairy, exchange, migration)

    # Run analyses
    ir_results = analyze_interest_rates(df)
    dairy_results = analyze_dairy(df)
    mig_results = analyze_migration(df, migration)
    fx_results = analyze_exchange_rate(df)
    horse_results = analyze_full_horse_race(df)
    plot_extended_timeline(df, interest, dairy, exchange, migration)

    # Generate report
    generate_report(ir_results, dairy_results, mig_results, fx_results, horse_results)

    print("\n" + "=" * 70)
    print("Extended Economic Analysis Part 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
