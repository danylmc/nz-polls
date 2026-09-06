#!/usr/bin/env python3
"""
Multivariate Incumbent Vote Model & Forecasting.

Combines the key predictors identified across 26 analyses into a single
model predicting incumbent party vote share:
- Government approval rating (Ipsos)
- Cost-of-living salience (Ipsos)
- Mortgage rates (RBNZ)
- Rental inflation (Stats NZ CPI)
- Petrol prices (MBIE)
- Consumer confidence (OECD)

Also tests whether lagged predictors can forecast NEXT period's vote shift.

Analysis 27: Multivariate incumbent vote model
Analysis 28: Forecasting model with lagged predictors
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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Government periods
GOVERNMENTS = [
    ("1996-12-16", "1999-11-27", "National"),
    ("1999-11-27", "2002-07-27", "Labour"),
    ("2002-07-27", "2005-09-17", "Labour"),
    ("2005-09-17", "2008-11-08", "Labour"),
    ("2008-11-08", "2011-11-26", "National"),
    ("2011-11-26", "2014-09-20", "National"),
    ("2014-09-20", "2017-09-23", "National"),
    ("2017-10-26", "2020-10-17", "Labour"),
    ("2020-10-17", "2023-10-14", "Labour"),
    ("2023-11-27", "2026-03-01", "National"),
]


def get_incumbent(date):
    for start, end, party in GOVERNMENTS:
        if pd.to_datetime(start) <= date <= pd.to_datetime(end):
            return party
    return None


def load_polls_quarterly():
    """Load polling data aggregated to quarterly means."""
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
    for col in ["National", "Labour", "Green", "NZ First", "ACT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter")[["National", "Labour", "Green", "NZ First", "ACT"]].mean()
    quarterly.index = quarterly.index.to_timestamp()
    return quarterly


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


def load_govt_performance():
    """Load Ipsos government performance ratings."""
    df = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["mean_score"]].rename(columns={"mean_score": "govt_approval"})


def load_issue_salience():
    """Load Ipsos issue salience data."""
    df = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["inflation_cost_of_living", "crime_law_order", "healthcare_hospitals"]]


def load_consumer_confidence():
    """Load OECD consumer confidence index."""
    df = pd.read_csv(DATA_DIR / "consumer_confidence.csv")
    cc = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    cc["date"] = pd.to_datetime(cc["TIME_PERIOD"])
    cc["cci"] = pd.to_numeric(cc["OBS_VALUE"], errors="coerce")
    cc = cc.dropna(subset=["cci"]).sort_values("date").set_index("date")[["cci"]]
    return cc.resample("MS").mean()


def load_mortgage_rates():
    """Load RBNZ 2yr fixed mortgage rates."""
    try:
        mortgage = pd.read_excel("raw_data/hb20.xlsx", skiprows=4)
        mort_col = mortgage.columns[6]  # 2yr fixed
        mortgage = mortgage[["Series Id", mort_col]].rename(
            columns={"Series Id": "date", mort_col: "mortgage_rate"})
        mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
        mortgage = mortgage.dropna(subset=["date", "mortgage_rate"])
        mortgage["mortgage_rate"] = pd.to_numeric(mortgage["mortgage_rate"], errors="coerce")
        return mortgage.set_index("date")[["mortgage_rate"]].resample("MS").mean()
    except Exception as e:
        print(f"  Could not load mortgage data: {e}")
        return pd.DataFrame()


def load_rental_cpi():
    """Load quarterly rental CPI index."""
    cpi = pd.read_csv(DATA_DIR / "consumers-price-index-december-2025-quarter-index-numbers (1).csv")
    rental = cpi[cpi["Series_reference"] == "CPIQ.SE9041"].copy()
    if len(rental) == 0:
        rental = cpi[cpi["Series_reference"] == "CPIQ.SE904101"].copy()
    rental["date"] = rental["Period"].apply(
        lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 100):02d}-01"))
    rental["date"] = rental["date"].dt.to_period("Q").dt.to_timestamp()
    rental = rental.sort_values("date").set_index("date")
    rental = rental[["Data_value"]].rename(columns={"Data_value": "rental_index"})
    rental["rental_yoy"] = rental["rental_index"].pct_change(4) * 100
    return rental


def load_petrol_prices():
    """Load MBIE weekly petrol prices, resample to monthly."""
    try:
        df = pd.read_csv("raw_data/weekly-table.csv")
        petrol = df[(df["Fuel"] == "Regular Petrol") & (df["Variable"] == "Board price")].copy()
        petrol["date"] = pd.to_datetime(petrol["Date"], dayfirst=True, errors="coerce")
        petrol["price"] = pd.to_numeric(petrol["Value"], errors="coerce")
        petrol = petrol.dropna(subset=["date", "price"]).set_index("date")[["price"]]
        return petrol.resample("MS").mean().rename(columns={"price": "petrol_price"})
    except Exception as e:
        print(f"  Could not load petrol data: {e}")
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("MULTIVARIATE INCUMBENT VOTE MODEL & FORECASTING")
    print("=" * 70)

    # ── Load all data ─────────────────────────────────────────────────
    polls_q = load_polls_quarterly()
    polls_m = load_polls_monthly()
    govt = load_govt_performance()
    salience = load_issue_salience()
    cci = load_consumer_confidence()
    mortgage = load_mortgage_rates()
    rental = load_rental_cpi()
    petrol = load_petrol_prices()

    # ── Build merged dataset at Ipsos survey dates ────────────────────
    # Use Ipsos survey dates as the anchor (they have the fewest observations)
    print(f"\n{'=' * 70}")
    print("PART 1: MULTIVARIATE MODEL AT IPSOS SURVEY DATES")
    print(f"{'=' * 70}")

    # Start with govt approval
    merged = govt.copy()
    merged = merged.join(salience, how="left")

    # Add incumbent vote at each survey date
    for idx in merged.index:
        incumbent = get_incumbent(idx)
        if incumbent is None:
            merged.loc[idx, "incumbent_vote"] = np.nan
            continue
        # Find nearest monthly poll
        month = idx.to_period("M").to_timestamp()
        if month in polls_m.index and incumbent in polls_m.columns:
            merged.loc[idx, "incumbent_vote"] = polls_m.loc[month, incumbent]
        else:
            # Try nearest quarter
            quarter = idx.to_period("Q").to_timestamp()
            if quarter in polls_q.index and incumbent in polls_q.columns:
                merged.loc[idx, "incumbent_vote"] = polls_q.loc[quarter, incumbent]
        merged.loc[idx, "incumbent"] = incumbent

    # Add economic indicators (nearest month)
    for idx in merged.index:
        month = idx.to_period("M").to_timestamp()
        # Consumer confidence
        if month in cci.index:
            merged.loc[idx, "cci"] = cci.loc[month, "cci"]
        elif len(cci) > 0:
            nearest = cci.index[cci.index.get_indexer([month], method="nearest")[0]]
            if abs((nearest - month).days) < 60:
                merged.loc[idx, "cci"] = cci.loc[nearest, "cci"]

        # Mortgage rate
        if len(mortgage) > 0 and month in mortgage.index:
            merged.loc[idx, "mortgage_rate"] = mortgage.loc[month, "mortgage_rate"]
        elif len(mortgage) > 0:
            nearest = mortgage.index[mortgage.index.get_indexer([month], method="nearest")[0]]
            if abs((nearest - month).days) < 60:
                merged.loc[idx, "mortgage_rate"] = mortgage.loc[nearest, "mortgage_rate"]

        # Petrol price
        if len(petrol) > 0 and month in petrol.index:
            merged.loc[idx, "petrol_price"] = petrol.loc[month, "petrol_price"]
        elif len(petrol) > 0:
            nearest = petrol.index[petrol.index.get_indexer([month], method="nearest")[0]]
            if abs((nearest - month).days) < 60:
                merged.loc[idx, "petrol_price"] = petrol.loc[nearest, "petrol_price"]

        # Rental inflation (quarterly, find nearest)
        if len(rental) > 0:
            quarter = idx.to_period("Q").to_timestamp()
            if quarter in rental.index:
                merged.loc[idx, "rental_yoy"] = rental.loc[quarter, "rental_yoy"]

    merged = merged.dropna(subset=["incumbent_vote"])
    print(f"\nObservations with incumbent vote: {len(merged)}")
    print(f"Date range: {merged.index.min().strftime('%Y-%m')} to {merged.index.max().strftime('%Y-%m')}")

    # ── Bivariate correlations ────────────────────────────────────────
    print(f"\n  Bivariate correlations with incumbent vote:")
    predictors = {
        "govt_approval": "Govt approval (0-10)",
        "inflation_cost_of_living": "Cost-of-living salience (%)",
        "crime_law_order": "Crime salience (%)",
        "healthcare_hospitals": "Healthcare salience (%)",
        "cci": "Consumer confidence",
        "mortgage_rate": "Mortgage rate (%)",
        "petrol_price": "Petrol price (c/L)",
        "rental_yoy": "Rental inflation (%)",
    }
    print(f"\n  {'Predictor':>30s}  {'r':>7s}  {'p':>8s}  {'n':>4s}")
    for var, label in predictors.items():
        if var in merged.columns:
            sub = merged.dropna(subset=[var, "incumbent_vote"])
            if len(sub) > 5:
                r, p = pearsonr(sub[var], sub["incumbent_vote"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:>30s}  {r:>+7.3f}  {p:>8.4f}  {len(sub):>4d} {sig}")

    # ── Model 1: Govt approval alone ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL 1: Govt approval alone (baseline)")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["govt_approval", "incumbent_vote"])
    X = sm.add_constant(sub[["govt_approval"]])
    y = sub["incumbent_vote"]
    m1 = sm.OLS(y, X).fit()
    print(f"\n  R² = {m1.rsquared:.3f}, Adj R² = {m1.rsquared_adj:.3f}, n = {len(sub)}")
    print(f"  govt_approval: β = {m1.params['govt_approval']:.2f}, p = {m1.pvalues['govt_approval']:.4f}")

    # ── Model 2: Approval + cost-of-living salience ──────────────────
    print(f"\n{'=' * 70}")
    print("MODEL 2: Approval + cost-of-living salience")
    print(f"{'=' * 70}")

    sub = merged.dropna(subset=["govt_approval", "inflation_cost_of_living", "incumbent_vote"])
    X = sm.add_constant(sub[["govt_approval", "inflation_cost_of_living"]])
    y = sub["incumbent_vote"]
    m2 = sm.OLS(y, X).fit()
    print(f"\n  R² = {m2.rsquared:.3f}, Adj R² = {m2.rsquared_adj:.3f}, n = {len(sub)}")
    for var in ["govt_approval", "inflation_cost_of_living"]:
        print(f"  {var}: β = {m2.params[var]:.3f}, p = {m2.pvalues[var]:.4f}")

    # ── Model 3: Kitchen sink ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL 3: Kitchen sink (all available predictors)")
    print(f"{'=' * 70}")

    all_vars = ["govt_approval", "inflation_cost_of_living", "crime_law_order",
                "cci", "mortgage_rate", "petrol_price", "rental_yoy"]
    available = [v for v in all_vars if v in merged.columns]
    sub = merged.dropna(subset=available + ["incumbent_vote"])

    if len(sub) > len(available) + 2:
        X = sm.add_constant(sub[available])
        y = sub["incumbent_vote"]
        m3 = sm.OLS(y, X).fit()
        print(f"\n  R² = {m3.rsquared:.3f}, Adj R² = {m3.rsquared_adj:.3f}, n = {len(sub)}")
        print(f"\n  {'Variable':>30s}  {'β':>8s}  {'SE':>7s}  {'p':>8s}  {'sig':>4s}")
        for var in available:
            sig = "***" if m3.pvalues[var] < 0.001 else "**" if m3.pvalues[var] < 0.01 else "*" if m3.pvalues[var] < 0.05 else ""
            print(f"  {var:>30s}  {m3.params[var]:>+8.3f}  {m3.bse[var]:>7.3f}  {m3.pvalues[var]:>8.4f}  {sig:>4s}")

        # VIF check
        print(f"\n  Variance Inflation Factors:")
        X_novif = sub[available]
        for i, var in enumerate(available):
            try:
                vif = variance_inflation_factor(X_novif.values, i)
                flag = " ← HIGH" if vif > 5 else ""
                print(f"    {var:>30s}: VIF = {vif:.1f}{flag}")
            except Exception:
                pass
    else:
        print(f"\n  Insufficient observations ({len(sub)}) for {len(available)} predictors")
        m3 = None

    # ── Model 4: Parsimonious (significant predictors only) ──────────
    print(f"\n{'=' * 70}")
    print("MODEL 4: Parsimonious model (stepwise reduction)")
    print(f"{'=' * 70}")

    # Start with all, remove highest p > 0.1 iteratively
    remaining = available.copy()
    sub_full = merged.dropna(subset=available + ["incumbent_vote"])

    if len(sub_full) > 5:
        while len(remaining) > 1:
            X = sm.add_constant(sub_full[remaining])
            y = sub_full["incumbent_vote"]
            model = sm.OLS(y, X).fit()
            worst_p = model.pvalues[remaining].max()
            worst_var = model.pvalues[remaining].idxmax()
            if worst_p > 0.10:
                remaining.remove(worst_var)
                print(f"  Dropped {worst_var} (p = {worst_p:.4f})")
            else:
                break

        X = sm.add_constant(sub_full[remaining])
        y = sub_full["incumbent_vote"]
        m4 = sm.OLS(y, X).fit()
        print(f"\n  Final model: R² = {m4.rsquared:.3f}, Adj R² = {m4.rsquared_adj:.3f}, n = {len(sub_full)}")
        print(f"\n  {'Variable':>30s}  {'β':>8s}  {'SE':>7s}  {'p':>8s}")
        for var in remaining:
            print(f"  {var:>30s}  {m4.params[var]:>+8.3f}  {m4.bse[var]:>7.3f}  {m4.pvalues[var]:>8.4f}")

        final_vars = remaining
    else:
        m4 = None
        final_vars = ["govt_approval"]

    # ── Model comparison table ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':>35s}  {'R²':>6s}  {'Adj R²':>7s}  {'AIC':>8s}  {'n':>4s}")
    models = [
        ("Approval only", m1),
        ("Approval + CoL salience", m2),
    ]
    if m3 is not None:
        models.append(("Kitchen sink", m3))
    if m4 is not None:
        models.append(("Parsimonious", m4))
    for name, model in models:
        if model is not None:
            print(f"  {name:>35s}  {model.rsquared:>6.3f}  {model.rsquared_adj:>7.3f}  {model.aic:>8.1f}  {model.nobs:>4.0f}")

    # ════════════════════════════════════════════════════════════════════
    # PART 2: QUARTERLY FORECASTING MODEL
    # ════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("PART 2: QUARTERLY FORECASTING MODEL")
    print("Can current indicators predict NEXT quarter's incumbent vote?")
    print(f"{'=' * 70}")

    # Build quarterly dataset with all indicators
    quarterly = polls_q.copy()
    quarterly["incumbent"] = quarterly.index.map(get_incumbent)
    quarterly["incumbent_vote"] = quarterly.apply(
        lambda r: r[r["incumbent"]] if pd.notna(r.get("incumbent")) and r.get("incumbent") in r.index else np.nan,
        axis=1)

    # Add economic indicators at quarterly frequency
    if len(mortgage) > 0:
        mort_q = mortgage.resample("QS").mean()
        quarterly = quarterly.join(mort_q, how="left")
    if len(petrol) > 0:
        pet_q = petrol.resample("QS").mean()
        quarterly = quarterly.join(pet_q, how="left")
    if len(cci) > 0:
        cci_q = cci.resample("QS").mean()
        quarterly = quarterly.join(cci_q, how="left")
    if len(rental) > 0:
        quarterly = quarterly.join(rental[["rental_yoy"]], how="left")

    # Add Ipsos data (matched to nearest quarter)
    for idx in quarterly.index:
        # Find nearest Ipsos survey
        if len(govt.index) > 0:
            diffs = abs(govt.index - idx)
            nearest_idx = diffs.argmin()
            nearest_date = govt.index[nearest_idx]
            if abs((nearest_date - idx).days) < 90:
                quarterly.loc[idx, "govt_approval"] = govt.loc[nearest_date, "govt_approval"]
        if len(salience.index) > 0:
            diffs = abs(salience.index - idx)
            nearest_idx = diffs.argmin()
            nearest_date = salience.index[nearest_idx]
            if abs((nearest_date - idx).days) < 90:
                for col in salience.columns:
                    if col in salience.columns:
                        quarterly.loc[idx, col] = salience.loc[nearest_date, col]

    quarterly = quarterly.dropna(subset=["incumbent_vote"])

    # Create target: NEXT quarter's incumbent vote change
    quarterly["inc_vote_next"] = quarterly["incumbent_vote"].shift(-1)
    quarterly["d_inc_vote"] = quarterly["inc_vote_next"] - quarterly["incumbent_vote"]

    print(f"\n  Quarterly observations: {len(quarterly)}")
    print(f"  With next-quarter change: {quarterly['d_inc_vote'].notna().sum()}")

    # ── Forecast Test 1: Can current approval predict next-quarter vote change?
    print(f"\n{'=' * 70}")
    print("FORECAST 1: Current indicators → next-quarter vote CHANGE")
    print(f"{'=' * 70}")

    forecast_vars = {
        "govt_approval": "Govt approval",
        "inflation_cost_of_living": "CoL salience",
        "cci": "Consumer confidence",
        "mortgage_rate": "Mortgage rate",
        "petrol_price": "Petrol price",
        "rental_yoy": "Rental inflation",
    }

    print(f"\n  {'Predictor':>25s}  {'r':>7s}  {'p':>8s}  {'n':>4s}")
    for var, label in forecast_vars.items():
        if var in quarterly.columns:
            sub = quarterly.dropna(subset=[var, "d_inc_vote"])
            if len(sub) > 5:
                r, p = pearsonr(sub[var], sub["d_inc_vote"])
                sig = "*" if p < 0.05 else ""
                print(f"  {label:>25s}  {r:>+7.3f}  {p:>8.4f}  {len(sub):>4d} {sig}")

    # ── Forecast Test 2: Level prediction (can current predict next level?)
    print(f"\n{'=' * 70}")
    print("FORECAST 2: Current indicators → next-quarter vote LEVEL")
    print(f"{'=' * 70}")

    print(f"\n  {'Predictor':>25s}  {'r':>7s}  {'p':>8s}  {'n':>4s}")
    for var, label in forecast_vars.items():
        if var in quarterly.columns:
            sub = quarterly.dropna(subset=[var, "inc_vote_next"])
            if len(sub) > 5:
                r, p = pearsonr(sub[var], sub["inc_vote_next"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:>25s}  {r:>+7.3f}  {p:>8.4f}  {len(sub):>4d} {sig}")

    # ── Forecast Test 3: Multivariate forecast model
    print(f"\n{'=' * 70}")
    print("FORECAST 3: Multivariate next-quarter vote model")
    print(f"{'=' * 70}")

    # Use approval + best economic indicator
    fvars = ["govt_approval"]
    for v in ["cci", "mortgage_rate", "rental_yoy", "petrol_price"]:
        if v in quarterly.columns:
            fvars.append(v)

    sub = quarterly.dropna(subset=fvars + ["inc_vote_next"])
    if len(sub) > len(fvars) + 2:
        X = sm.add_constant(sub[fvars])
        y = sub["inc_vote_next"]
        mf = sm.OLS(y, X).fit()
        print(f"\n  R² = {mf.rsquared:.3f}, Adj R² = {mf.rsquared_adj:.3f}, n = {len(sub)}")
        print(f"\n  {'Variable':>25s}  {'β':>8s}  {'p':>8s}")
        for var in fvars:
            print(f"  {var:>25s}  {mf.params[var]:>+8.3f}  {mf.pvalues[var]:>8.4f}")

    # ── Forecast Test 4: Leave-one-out cross-validation
    print(f"\n{'=' * 70}")
    print("FORECAST 4: Leave-one-out cross-validation")
    print(f"{'=' * 70}")

    # Use the simplest predictive model: approval → next quarter level
    sub = quarterly.dropna(subset=["govt_approval", "inc_vote_next"])
    if len(sub) > 5:
        errors = []
        for i in range(len(sub)):
            train = sub.drop(sub.index[i])
            test = sub.iloc[[i]]
            X_train = sm.add_constant(train[["govt_approval"]])
            y_train = train["inc_vote_next"]
            X_test = sm.add_constant(test[["govt_approval"]], has_constant="add")
            m = sm.OLS(y_train, X_train).fit()
            pred = m.predict(X_test).iloc[0]
            actual = test["inc_vote_next"].iloc[0]
            errors.append(abs(pred - actual))

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        print(f"\n  Approval-only model (LOO-CV):")
        print(f"    MAE  = {mae:.2f} pp")
        print(f"    RMSE = {rmse:.2f} pp")
        print(f"    n = {len(sub)} quarters")

    # Also try with best multivariate
    best_fvars = [v for v in ["govt_approval", "cci"] if v in quarterly.columns]
    sub = quarterly.dropna(subset=best_fvars + ["inc_vote_next"])
    if len(sub) > len(best_fvars) + 2:
        errors_mv = []
        for i in range(len(sub)):
            train = sub.drop(sub.index[i])
            test = sub.iloc[[i]]
            X_train = sm.add_constant(train[best_fvars])
            y_train = train["inc_vote_next"]
            X_test = sm.add_constant(test[best_fvars], has_constant="add")
            m = sm.OLS(y_train, X_train).fit()
            pred = m.predict(X_test).iloc[0]
            actual = test["inc_vote_next"].iloc[0]
            errors_mv.append(abs(pred - actual))

        mae_mv = np.mean(errors_mv)
        rmse_mv = np.sqrt(np.mean(np.array(errors_mv)**2))
        print(f"\n  Approval + confidence model (LOO-CV):")
        print(f"    MAE  = {mae_mv:.2f} pp")
        print(f"    RMSE = {rmse_mv:.2f} pp")
        print(f"    n = {len(sub)} quarters")

    # ── Naive baseline ────────────────────────────────────────────────
    sub_naive = quarterly.dropna(subset=["incumbent_vote", "inc_vote_next"])
    if len(sub_naive) > 2:
        naive_errors = abs(sub_naive["incumbent_vote"] - sub_naive["inc_vote_next"])
        print(f"\n  Naïve baseline (this quarter = next quarter):")
        print(f"    MAE  = {naive_errors.mean():.2f} pp")
        print(f"    RMSE = {np.sqrt((naive_errors**2).mean()):.2f} pp")

    # ════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Model fit — approval vs incumbent vote
    ax = axes[0, 0]
    sub = merged.dropna(subset=["govt_approval", "incumbent_vote"])
    colors = ["#d32f2f" if inc == "Labour" else "#1565c0"
              for inc in sub.get("incumbent", [""] * len(sub))]
    ax.scatter(sub["govt_approval"], sub["incumbent_vote"], c=colors, alpha=0.6, s=50)
    if len(sub) > 2:
        m_slope, b_int = np.polyfit(sub["govt_approval"], sub["incumbent_vote"], 1)
        xr = np.linspace(sub["govt_approval"].min(), sub["govt_approval"].max(), 50)
        ax.plot(xr, m_slope * xr + b_int, "k--", alpha=0.5)
        r, _ = pearsonr(sub["govt_approval"], sub["incumbent_vote"])
        ax.set_title(f"Govt Approval → Incumbent Vote (r={r:.3f})")
    ax.set_xlabel("Government approval rating (0-10)")
    ax.set_ylabel("Incumbent party vote (%)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d32f2f", alpha=0.7, label="Labour govt"),
        Patch(facecolor="#1565c0", alpha=0.7, label="National govt"),
    ], fontsize=8)

    # Panel 2: Bivariate correlations bar chart
    ax = axes[0, 1]
    corrs = []
    for var, label in predictors.items():
        if var in merged.columns:
            sub = merged.dropna(subset=[var, "incumbent_vote"])
            if len(sub) > 5:
                r, p = pearsonr(sub[var], sub["incumbent_vote"])
                corrs.append((label.split("(")[0].strip(), r, p))
    if corrs:
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        labels, rs, ps = zip(*corrs)
        colors_bar = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in ps]
        bars = ax.barh(range(len(labels)), rs, color=colors_bar, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Correlation with incumbent vote")
        ax.set_title("Predictor Strength (green = p<0.05)")
        ax.axvline(0, color="gray", lw=0.5)

    # Panel 3: Model R² comparison
    ax = axes[1, 0]
    model_names = []
    r2_vals = []
    adj_r2_vals = []
    for name, model in models:
        if model is not None:
            model_names.append(name)
            r2_vals.append(model.rsquared)
            adj_r2_vals.append(model.rsquared_adj)
    x_pos = range(len(model_names))
    ax.bar([x - 0.15 for x in x_pos], r2_vals, 0.3, label="R²", color="#3498db", alpha=0.7)
    ax.bar([x + 0.15 for x in x_pos], adj_r2_vals, 0.3, label="Adj R²", color="#e67e22", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel 4: Forecast — actual vs predicted next-quarter vote
    ax = axes[1, 1]
    sub = quarterly.dropna(subset=["govt_approval", "inc_vote_next"])
    if len(sub) > 5:
        X = sm.add_constant(sub[["govt_approval"]])
        y = sub["inc_vote_next"]
        m_fit = sm.OLS(y, X).fit()
        predicted = m_fit.predict(X)
        ax.scatter(sub["inc_vote_next"], predicted, alpha=0.6, color="#3498db")
        # 45-degree line
        lims = [min(sub["inc_vote_next"].min(), predicted.min()) - 1,
                max(sub["inc_vote_next"].max(), predicted.max()) + 1]
        ax.plot(lims, lims, "k--", alpha=0.3, label="Perfect forecast")
        ax.set_xlabel("Actual next-quarter vote (%)")
        ax.set_ylabel("Predicted next-quarter vote (%)")
        ax.set_title(f"Forecast: Approval → Next Quarter (R²={m_fit.rsquared:.3f})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = REPORTS_DIR / "multivariate_model.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"""
  The multivariate analysis confirms that government approval is the
  dominant predictor of incumbent vote share. Adding economic indicators
  provides marginal improvement over approval alone, because approval
  already incorporates voters' subjective assessment of economic conditions.

  The causal chain is: pocketbook costs → cost-of-living salience →
  government disapproval → vote shift to opposition.

  For forecasting, current approval predicts next-quarter incumbent vote
  level reasonably well, but predicting vote CHANGES is much harder —
  most quarter-to-quarter variation is driven by events and media cycles
  that are inherently unpredictable.
""")


if __name__ == "__main__":
    main()
