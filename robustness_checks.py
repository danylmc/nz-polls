#!/usr/bin/env python3
"""
Robustness checks addressing adversarial critique:

1. Granger causality test for rental inflation → incumbent vote
2. Newey-West HAC standard errors for all key time-series regressions
   (mortgage, petrol, consumer confidence, rental, approval)

These address the two main statistical gaps identified in the critique:
- Missing Granger test for the strongest pocketbook indicator (rental r=-0.731)
- Autocorrelation-inflated significance in OLS regressions
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = Path(__file__).parent / "raw_data"

# Government periods
GOVERNMENTS = [
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


def load_polls_monthly():
    """Load polls aggregated to monthly means."""
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
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for col in parties:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")[parties].mean()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def load_polls_quarterly():
    """Load polls aggregated to quarterly means."""
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
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for col in parties:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter")[parties].mean()
    quarterly.index = quarterly.index.to_timestamp()
    return quarterly


def add_incumbent(df):
    """Add incumbent party and vote share columns."""
    df["incumbent"] = df.index.map(get_incumbent)
    df["incumbent_vote"] = df.apply(
        lambda r: r[r["incumbent"]]
        if pd.notna(r.get("incumbent")) and r.get("incumbent") in r.index
        else np.nan,
        axis=1,
    )
    return df


def newey_west_regression(y, X, nlags=None):
    """Run OLS with Newey-West HAC standard errors.

    Returns dict with OLS and NW results for comparison.
    """
    X_const = sm.add_constant(X)
    # OLS
    ols = sm.OLS(y, X_const).fit()
    # Newey-West
    if nlags is None:
        nlags = int(np.ceil(4 * (len(y) / 100) ** (2 / 9)))  # Andrews rule of thumb
    nw = sm.OLS(y, X_const).fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
    return ols, nw


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PART 1: RENTAL INFLATION GRANGER CAUSALITY TEST                ║
# ╚═══════════════════════════════════════════════════════════════════╝

def rental_granger_test():
    print("=" * 70)
    print("PART 1: RENTAL INFLATION GRANGER CAUSALITY TEST")
    print("=" * 70)

    # Load rental CPI
    cpi = pd.read_csv(DATA_DIR / "consumers-price-index-december-2025-quarter-index-numbers (1).csv")
    rental = cpi[cpi["Series_reference"] == "CPIQ.SE9041"].copy()
    if len(rental) == 0:
        rental = cpi[cpi["Series_reference"] == "CPIQ.SE904101"].copy()

    rental["date"] = rental["Period"].apply(
        lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 100):02d}-01")
    )
    rental["date"] = rental["date"].dt.to_period("Q").dt.to_timestamp()
    rental = rental.sort_values("date").set_index("date")
    rental = rental[["Data_value"]].rename(columns={"Data_value": "rental_index"})
    rental["rental_yoy"] = rental["rental_index"].pct_change(4) * 100
    rental["rental_qoq"] = rental["rental_index"].pct_change(1) * 100

    # Load polls quarterly
    polls = load_polls_quarterly()
    merged = rental.join(polls, how="inner").dropna(subset=["rental_yoy", "National", "Labour"])
    merged = add_incumbent(merged)

    # First-difference the series for Granger
    merged["d_rental"] = merged["rental_yoy"].diff()
    merged["d_inc"] = merged["incumbent_vote"].diff()

    gc_data = merged[["d_inc", "d_rental"]].dropna()
    print(f"\nGranger test data: {len(gc_data)} quarterly observations")
    print(f"Period: {gc_data.index.min().strftime('%Y-Q%q')} to {gc_data.index.max().strftime('%Y-Q%q')}")

    print(f"\n  Granger causality: rental inflation → incumbent vote changes")
    print(f"  {'Lag':>5s}  {'F-stat':>8s}  {'p-value':>8s}  {'Significant':>12s}")
    try:
        gc = grangercausalitytests(gc_data, maxlag=6, verbose=False)
        for lag, result in gc.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {lag:>5d}  {f_stat:>8.3f}  {p_val:>8.4f}  {sig:>12s}")
    except Exception as e:
        print(f"  Error: {e}")

    # Also test reverse: vote → rental (should not be significant)
    gc_rev_data = merged[["d_rental", "d_inc"]].dropna()
    print(f"\n  Reverse Granger: incumbent vote → rental inflation (should be n.s.)")
    print(f"  {'Lag':>5s}  {'F-stat':>8s}  {'p-value':>8s}")
    try:
        gc_rev = grangercausalitytests(gc_rev_data, maxlag=6, verbose=False)
        for lag, result in gc_rev.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            print(f"  {lag:>5d}  {f_stat:>8.3f}  {p_val:>8.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Also test with QoQ rental change (not YoY)
    merged["d_rental_qoq"] = merged["rental_qoq"].diff()
    gc_qoq = merged[["d_inc", "d_rental_qoq"]].dropna()
    print(f"\n  Granger with QoQ rental change (not YoY):")
    print(f"  {'Lag':>5s}  {'F-stat':>8s}  {'p-value':>8s}")
    try:
        gc2 = grangercausalitytests(gc_qoq, maxlag=4, verbose=False)
        for lag, result in gc2.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {lag:>5d}  {f_stat:>8.3f}  {p_val:>8.4f}  {sig}")
    except Exception as e:
        print(f"  Error: {e}")

    return merged


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  PART 2: NEWEY-WEST HAC STANDARD ERRORS                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

def newey_west_tests():
    print(f"\n\n{'=' * 70}")
    print("PART 2: NEWEY-WEST HAC STANDARD ERRORS")
    print("=" * 70)
    print("\nComparing OLS vs Newey-West significance for key regressions.")
    print("If autocorrelation inflates significance, NW p-values will be larger.\n")

    polls_m = load_polls_monthly()
    polls_m = add_incumbent(polls_m)

    results = []

    # ── 2a. Mortgage rates → incumbent vote (monthly) ──────────────
    print(f"{'─' * 70}")
    print("2a. MORTGAGE RATE → INCUMBENT VOTE (monthly)")
    print(f"{'─' * 70}")
    try:
        mortgage = pd.read_excel(RAW_DIR / "hb20.xlsx", skiprows=4)
        mort_col = mortgage.columns[6]
        mortgage = mortgage[["Series Id", mort_col]].rename(
            columns={"Series Id": "date", mort_col: "mortgage_rate"}
        )
        mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
        mortgage = mortgage.dropna(subset=["date", "mortgage_rate"])
        mortgage["mortgage_rate"] = pd.to_numeric(mortgage["mortgage_rate"], errors="coerce")
        mortgage = mortgage.set_index("date").resample("MS").mean()

        m = polls_m.join(mortgage, how="inner").dropna(subset=["mortgage_rate", "incumbent_vote"])
        print(f"  n = {len(m)}")

        # Levels
        ols, nw = newey_west_regression(m["incumbent_vote"], m["mortgage_rate"])
        r, _ = pearsonr(m["mortgage_rate"], m["incumbent_vote"])
        print(f"\n  LEVELS (r = {r:+.3f}):")
        print(f"    OLS:       coef = {ols.params[1]:+.3f}, SE = {ols.bse[1]:.3f}, p = {ols.pvalues[1]:.4f}")
        print(f"    Newey-West: coef = {nw.params[1]:+.3f}, SE = {nw.bse[1]:.3f}, p = {nw.pvalues[1]:.4f}")
        results.append(("Mortgage→Inc (levels)", r, ols.pvalues[1], nw.pvalues[1], len(m)))

        # Changes
        dm = m[["mortgage_rate", "incumbent_vote"]].diff().dropna()
        if len(dm) > 10:
            ols_d, nw_d = newey_west_regression(dm["incumbent_vote"], dm["mortgage_rate"])
            r_d, _ = pearsonr(dm["mortgage_rate"], dm["incumbent_vote"])
            print(f"\n  FIRST-DIFFERENCES (r = {r_d:+.3f}):")
            print(f"    OLS:       coef = {ols_d.params[1]:+.3f}, SE = {ols_d.bse[1]:.3f}, p = {ols_d.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_d.params[1]:+.3f}, SE = {nw_d.bse[1]:.3f}, p = {nw_d.pvalues[1]:.4f}")
            results.append(("Mortgage→Inc (changes)", r_d, ols_d.pvalues[1], nw_d.pvalues[1], len(dm)))

        # 1-month lagged changes
        dm["d_rate_lag1"] = dm["mortgage_rate"].shift(1)
        dm_lag = dm.dropna()
        if len(dm_lag) > 10:
            ols_l, nw_l = newey_west_regression(dm_lag["incumbent_vote"], dm_lag["d_rate_lag1"])
            r_l, _ = pearsonr(dm_lag["d_rate_lag1"], dm_lag["incumbent_vote"])
            print(f"\n  1-MONTH LAGGED CHANGES (r = {r_l:+.3f}):")
            print(f"    OLS:       coef = {ols_l.params[1]:+.3f}, SE = {ols_l.bse[1]:.3f}, p = {ols_l.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_l.params[1]:+.3f}, SE = {nw_l.bse[1]:.3f}, p = {nw_l.pvalues[1]:.4f}")
            results.append(("Mortgage→Inc (lag-1 Δ)", r_l, ols_l.pvalues[1], nw_l.pvalues[1], len(dm_lag)))
    except Exception as e:
        print(f"  Error: {e}")

    # ── 2b. Petrol prices → incumbent vote (monthly) ───────────────
    print(f"\n{'─' * 70}")
    print("2b. PETROL PRICE → INCUMBENT VOTE (monthly)")
    print(f"{'─' * 70}")
    try:
        pet = pd.read_csv(RAW_DIR / "weekly-table.csv")
        pet = pet[(pet["Fuel"] == "Regular Petrol") & (pet["Variable"] == "Board price")].copy()
        pet["date"] = pd.to_datetime(pet["Date"], errors="coerce")
        pet["petrol_price"] = pd.to_numeric(pet["Value"], errors="coerce")
        pet = pet.dropna(subset=["date", "petrol_price"])
        pet = pet.set_index("date")[["petrol_price"]].resample("MS").mean()

        m = polls_m.join(pet, how="inner").dropna(subset=["petrol_price", "incumbent_vote"])
        print(f"  n = {len(m)}")

        # Levels
        ols, nw = newey_west_regression(m["incumbent_vote"], m["petrol_price"])
        r, _ = pearsonr(m["petrol_price"], m["incumbent_vote"])
        print(f"\n  LEVELS (r = {r:+.3f}):")
        print(f"    OLS:       coef = {ols.params[1]:+.3f}, SE = {ols.bse[1]:.3f}, p = {ols.pvalues[1]:.4f}")
        print(f"    Newey-West: coef = {nw.params[1]:+.3f}, SE = {nw.bse[1]:.3f}, p = {nw.pvalues[1]:.4f}")
        results.append(("Petrol→Inc (levels)", r, ols.pvalues[1], nw.pvalues[1], len(m)))

        # Changes
        dm = m[["petrol_price", "incumbent_vote"]].diff().dropna()
        if len(dm) > 10:
            ols_d, nw_d = newey_west_regression(dm["incumbent_vote"], dm["petrol_price"])
            r_d, _ = pearsonr(dm["petrol_price"], dm["incumbent_vote"])
            print(f"\n  FIRST-DIFFERENCES (r = {r_d:+.3f}):")
            print(f"    OLS:       coef = {ols_d.params[1]:+.3f}, SE = {ols_d.bse[1]:.3f}, p = {ols_d.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_d.params[1]:+.3f}, SE = {nw_d.bse[1]:.3f}, p = {nw_d.pvalues[1]:.4f}")
            results.append(("Petrol→Inc (changes)", r_d, ols_d.pvalues[1], nw_d.pvalues[1], len(dm)))
    except Exception as e:
        print(f"  Error: {e}")

    # ── 2c. Consumer confidence → incumbent vote (monthly) ─────────
    print(f"\n{'─' * 70}")
    print("2c. CONSUMER CONFIDENCE → INCUMBENT VOTE (monthly)")
    print(f"{'─' * 70}")
    try:
        cci = pd.read_csv(DATA_DIR / "consumer_confidence.csv")
        cci["date"] = pd.to_datetime(cci["TIME_PERIOD"], errors="coerce")
        cci = cci.dropna(subset=["date"])
        cci["cci"] = pd.to_numeric(cci["OBS_VALUE"], errors="coerce")
        cci = cci.set_index("date")[["cci"]].resample("MS").mean()

        m = polls_m.join(cci, how="inner").dropna(subset=["cci", "incumbent_vote"])
        print(f"  n = {len(m)}")

        # Levels
        ols, nw = newey_west_regression(m["incumbent_vote"], m["cci"])
        r, _ = pearsonr(m["cci"], m["incumbent_vote"])
        print(f"\n  LEVELS (r = {r:+.3f}):")
        print(f"    OLS:       coef = {ols.params[1]:+.3f}, SE = {ols.bse[1]:.3f}, p = {ols.pvalues[1]:.4f}")
        print(f"    Newey-West: coef = {nw.params[1]:+.3f}, SE = {nw.bse[1]:.3f}, p = {nw.pvalues[1]:.4f}")
        results.append(("CCI→Inc (levels)", r, ols.pvalues[1], nw.pvalues[1], len(m)))

        # Changes
        dm = m[["cci", "incumbent_vote"]].diff().dropna()
        if len(dm) > 10:
            ols_d, nw_d = newey_west_regression(dm["incumbent_vote"], dm["cci"])
            r_d, _ = pearsonr(dm["cci"], dm["incumbent_vote"])
            print(f"\n  FIRST-DIFFERENCES (r = {r_d:+.3f}):")
            print(f"    OLS:       coef = {ols_d.params[1]:+.3f}, SE = {ols_d.bse[1]:.3f}, p = {ols_d.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_d.params[1]:+.3f}, SE = {nw_d.bse[1]:.3f}, p = {nw_d.pvalues[1]:.4f}")
            results.append(("CCI→Inc (changes)", r_d, ols_d.pvalues[1], nw_d.pvalues[1], len(dm)))
    except Exception as e:
        print(f"  Error: {e}")

    # ── 2d. Rental inflation → incumbent vote (quarterly) ──────────
    print(f"\n{'─' * 70}")
    print("2d. RENTAL INFLATION → INCUMBENT VOTE (quarterly)")
    print(f"{'─' * 70}")
    try:
        cpi = pd.read_csv(DATA_DIR / "consumers-price-index-december-2025-quarter-index-numbers (1).csv")
        rental = cpi[cpi["Series_reference"] == "CPIQ.SE9041"].copy()
        if len(rental) == 0:
            rental = cpi[cpi["Series_reference"] == "CPIQ.SE904101"].copy()

        rental["date"] = rental["Period"].apply(
            lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 100):02d}-01")
        )
        rental["date"] = rental["date"].dt.to_period("Q").dt.to_timestamp()
        rental = rental.sort_values("date").set_index("date")
        rental = rental[["Data_value"]].rename(columns={"Data_value": "rental_index"})
        rental["rental_yoy"] = rental["rental_index"].pct_change(4) * 100

        polls_q = load_polls_quarterly()
        m = rental.join(polls_q, how="inner").dropna(subset=["rental_yoy", "National", "Labour"])
        m = add_incumbent(m)
        m = m.dropna(subset=["incumbent_vote"])
        print(f"  n = {len(m)}")

        # Levels
        ols, nw = newey_west_regression(m["incumbent_vote"], m["rental_yoy"])
        r, _ = pearsonr(m["rental_yoy"], m["incumbent_vote"])
        print(f"\n  LEVELS (r = {r:+.3f}):")
        print(f"    OLS:       coef = {ols.params[1]:+.3f}, SE = {ols.bse[1]:.3f}, p = {ols.pvalues[1]:.4f}")
        print(f"    Newey-West: coef = {nw.params[1]:+.3f}, SE = {nw.bse[1]:.3f}, p = {nw.pvalues[1]:.4f}")
        results.append(("Rental→Inc (levels)", r, ols.pvalues[1], nw.pvalues[1], len(m)))

        # Changes
        dm = m[["rental_yoy", "incumbent_vote"]].diff().dropna()
        if len(dm) > 10:
            ols_d, nw_d = newey_west_regression(dm["incumbent_vote"], dm["rental_yoy"])
            r_d, _ = pearsonr(dm["rental_yoy"], dm["incumbent_vote"])
            print(f"\n  FIRST-DIFFERENCES (r = {r_d:+.3f}):")
            print(f"    OLS:       coef = {ols_d.params[1]:+.3f}, SE = {ols_d.bse[1]:.3f}, p = {ols_d.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_d.params[1]:+.3f}, SE = {nw_d.bse[1]:.3f}, p = {nw_d.pvalues[1]:.4f}")
            results.append(("Rental→Inc (changes)", r_d, ols_d.pvalues[1], nw_d.pvalues[1], len(dm)))
    except Exception as e:
        print(f"  Error: {e}")

    # ── 2e. Govt approval → incumbent vote (Ipsos) ─────────────────
    print(f"\n{'─' * 70}")
    print("2e. GOVT APPROVAL → INCUMBENT VOTE (Ipsos)")
    print(f"{'─' * 70}")
    try:
        approval = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
        approval["date"] = pd.to_datetime(approval["date"], errors="coerce")
        approval = approval.dropna(subset=["date"]).set_index("date")
        approval = approval[["mean_score"]].rename(columns={"mean_score": "approval"})

        # Monthly polls
        m = polls_m.join(approval, how="inner").dropna(subset=["approval"])
        m = add_incumbent(m)
        m = m.dropna(subset=["incumbent_vote"])
        print(f"  n = {len(m)}")

        # Levels
        ols, nw = newey_west_regression(m["incumbent_vote"], m["approval"])
        r, _ = pearsonr(m["approval"], m["incumbent_vote"])
        print(f"\n  LEVELS (r = {r:+.3f}):")
        print(f"    OLS:       coef = {ols.params[1]:+.3f}, SE = {ols.bse[1]:.3f}, p = {ols.pvalues[1]:.4f}")
        print(f"    Newey-West: coef = {nw.params[1]:+.3f}, SE = {nw.bse[1]:.3f}, p = {nw.pvalues[1]:.4f}")
        results.append(("Approval→Inc (levels)", r, ols.pvalues[1], nw.pvalues[1], len(m)))

        # Changes
        dm = m[["approval", "incumbent_vote"]].diff().dropna()
        if len(dm) > 10:
            ols_d, nw_d = newey_west_regression(dm["incumbent_vote"], dm["approval"])
            r_d, _ = pearsonr(dm["approval"], dm["incumbent_vote"])
            print(f"\n  FIRST-DIFFERENCES (r = {r_d:+.3f}):")
            print(f"    OLS:       coef = {ols_d.params[1]:+.3f}, SE = {ols_d.bse[1]:.3f}, p = {ols_d.pvalues[1]:.4f}")
            print(f"    Newey-West: coef = {nw_d.params[1]:+.3f}, SE = {nw_d.bse[1]:.3f}, p = {nw_d.pvalues[1]:.4f}")
            results.append(("Approval→Inc (changes)", r_d, ols_d.pvalues[1], nw_d.pvalues[1], len(dm)))
    except Exception as e:
        print(f"  Error: {e}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("SUMMARY: OLS vs NEWEY-WEST SIGNIFICANCE")
    print(f"{'=' * 70}")
    print(f"\n  {'Regression':<30s}  {'r':>6s}  {'OLS p':>8s}  {'NW p':>8s}  {'n':>5s}  {'Survives NW?':>14s}")
    print(f"  {'─' * 75}")
    for name, r, p_ols, p_nw, n in results:
        survives = "Yes***" if p_nw < 0.001 else "Yes**" if p_nw < 0.01 else "Yes*" if p_nw < 0.05 else "No"
        print(f"  {name:<30s}  {r:>+6.3f}  {p_ols:>8.4f}  {p_nw:>8.4f}  {n:>5d}  {survives:>14s}")

    return results


if __name__ == "__main__":
    rental_granger_test()
    newey_west_tests()
