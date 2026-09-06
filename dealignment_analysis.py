#!/usr/bin/env python3
"""
Dealignment Analysis (Dalton 2000, 2008).

Tests whether demographic predictors of vote choice are weakening over time.
The dealignment thesis predicts that social cleavages (class, age, gender,
ethnicity) become less predictive of party vote as societies modernize.

Method: Run logistic regression (National vs Labour) ~ demographics for each
NZES election year. Track pseudo-R² over time. If dealignment is occurring,
the total explanatory power of demographics should be declining.

Data: NZES individual-level survey data (1996-2023, 10 elections).
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
NZES_DB = DATA_DIR / "nzesdb.sqlite"

# Variable mapping per year — from deep_data.py
PARTY_VOTE_RECODE = {
    1996: {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT",
           6: "Christian", 7: "United", 14: "Green"},
    1999: {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT",
           6: "Christian", 7: "Green", 10: "United"},
    2002: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           7: "Alliance", 9: "Progressive", 11: "United Future"},
    2005: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
    2008: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
    2011: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori"},
    2014: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 11: "Conservative"},
    2017: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "Maori", 8: "TOP"},
    2020: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "Maori", 10: "TOP"},
    2023: {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First",
           6: "Maori"},
}

# Year → {var_name: column_name} for demographics
YEAR_CONFIG = {
    1996: {
        "table": "nzes_1996",
        "party_vote": "VOT96P",
        "age": "QAGE",
        "sex": "QSEX",
        "education": "PEDQUAL",
        "maori": "QANCEST",  # Maori ancestry
        "maori_is_ancestry": True,
    },
    1999: {
        "table": "nzes_1999_post",
        "party_vote": "PVOTE",
        "age": "AGE",
        "sex": "SEX",
        "education": "SEDQUAL",
        "maori": "SETHNM",  # Maori ethnicity (1=yes)
    },
    2002: {
        "table": "nzes_2002",
        "party_vote": "WVOT02P",
        "age": "WNAGE",
        "sex": "WSEX",
        "education": "WEDQUAL",
        "maori": "WETHN2",  # NZ Maori (1=yes)
    },
    2005: {
        "table": "nzes_2005",
        "party_vote": "yvot05p",
        "age": "YAGE",
        "sex": "ysex",
        "education": "yeducate",
        "maori": "maord5",  # Maori descent
        "maori_is_descent": True,
    },
    2008: {
        "table": "nzes_2008",
        "party_vote": "zvt08p",
        "age": "ZAGE",
        "sex": "nsex",
        "education": "zeducat",
        "maori": "zmaode",  # Maori descent
        "maori_is_descent": True,
    },
    2011: {
        "table": "nzes_2011",
        "party_vote": "jpartyvote",
        "age": "jage",
        "sex": "jsex",
        "education": "jhqual",
        "maori": "jethnicity_m",  # NZ Maori ethnicity (1=yes)
    },
    2014: {
        "table": "nzes_2014",
        "party_vote": "dpartyvote",
        "age": "dage",
        "sex": "dsex",
        "education": "coldedcons",
        "maori": "dethnicity_m",  # NZ Maori ethnicity
    },
    2017: {
        "table": "nzes_2017",
        "party_vote": "rpartyvote",
        "age": "rsamage",
        "sex": "rnsex",
        "education": "reduc",
        "maori": "rmaorieth",  # NZ Maori ethnicity (1=yes)
    },
    2020: {
        "table": "nzes_2020",
        "party_vote": "E3",
        "age": "lmage",
        "sex": "rollgen",
        "education": "newleducate",
        "maori": "lmaori",  # Identifies as Maori
    },
    2023: {
        "table": "nzes_2023",
        "party_vote": "mpartyvote",
        "age": "mage",
        "sex": "mgender",
        "education": "meducate",
        "maori": "mmaori",  # Maori by descent or identification
    },
}

# Education recode to 4 levels: 0=none/primary, 1=secondary, 2=post-sec/diploma, 3=degree+
EDUCATION_RECODE = {
    1996: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
    1999: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
    2002: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
    2005: {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3},
    2008: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3},
    2011: {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3},
    2014: {0: 0, 1: 1, 2: 2, 3: 3},
    2017: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1,
           10: 2, 11: 2, 12: 3, 13: 3, 14: 3},
    2020: {1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3,
           11: 3, 12: 3},
    2023: {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
}


def extract_year(year, conn):
    """Extract and harmonize data for one NZES election year."""
    config = YEAR_CONFIG[year]
    table = config["table"]

    # Build column list
    cols = [config["party_vote"], config["age"], config["sex"]]
    if config.get("education"):
        cols.append(config["education"])
    if config.get("maori"):
        cols.append(config["maori"])

    # Remove duplicates and filter to existing columns
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing_cols = {r[1] for r in cursor.fetchall()}
    cols = [c for c in cols if c in existing_cols]

    df = pd.read_sql_query(f"SELECT {','.join(cols)} FROM {table}", conn)

    result = pd.DataFrame()
    result["year"] = year

    # Party vote → National vs Labour binary
    pv_col = config["party_vote"]
    recode = PARTY_VOTE_RECODE.get(year, {})
    result["party"] = df[pv_col].apply(
        lambda x: recode.get(int(x)) if pd.notna(x) and not np.isnan(float(x)) and int(x) in recode else None
    )
    result["voted_nat"] = (result["party"] == "National").astype(float)
    result["voted_lab"] = (result["party"] == "Labour").astype(float)
    # Keep only Nat/Lab voters for binary model
    result["nat_vs_lab"] = np.where(
        result["party"] == "National", 1,
        np.where(result["party"] == "Labour", 0, np.nan)
    )

    # Age
    age_col = config["age"]
    if age_col in df.columns:
        result["age"] = pd.to_numeric(df[age_col], errors="coerce")
        # Filter unreasonable ages
        result.loc[(result["age"] < 16) | (result["age"] > 110), "age"] = np.nan

    # Sex → female (0/1)
    sex_col = config["sex"]
    if sex_col in df.columns:
        sex_vals = pd.to_numeric(df[sex_col], errors="coerce")
        result["female"] = np.where(sex_vals == 2, 1, np.where(sex_vals == 1, 0, np.nan))

    # Education → 4-level ordinal
    edu_col = config.get("education")
    if edu_col and edu_col in df.columns:
        edu_vals = pd.to_numeric(df[edu_col], errors="coerce")
        edu_recode = EDUCATION_RECODE.get(year, {})
        result["education"] = edu_vals.apply(
            lambda x: edu_recode.get(int(x)) if pd.notna(x) and int(x) in edu_recode else np.nan
        )

    # Maori → binary
    maori_col = config.get("maori")
    if maori_col and maori_col in df.columns:
        maori_vals = pd.to_numeric(df[maori_col], errors="coerce")
        if config.get("maori_is_ancestry") or config.get("maori_is_descent"):
            # 1=yes, 2=no for ancestry/descent questions
            result["maori"] = np.where(maori_vals == 1, 1, np.where(maori_vals == 2, 0, np.nan))
        else:
            # Ethnicity checkbox: 1=yes
            result["maori"] = np.where(maori_vals == 1, 1, 0)

    return result


def main():
    conn = sqlite3.connect(str(NZES_DB))

    print("=" * 70)
    print("DEALIGNMENT ANALYSIS (Dalton 2000, 2008)")
    print("=" * 70)

    # ── 1. Extract data for all years ──────────────────────────────────
    all_years = {}
    year_stats = []

    for year in sorted(YEAR_CONFIG.keys()):
        df = extract_year(year, conn)
        nat_lab = df.dropna(subset=["nat_vs_lab"])
        all_years[year] = df

        n_total = len(df)
        n_natlab = len(nat_lab)
        n_age = nat_lab["age"].notna().sum()
        n_female = nat_lab["female"].notna().sum()
        n_edu = nat_lab["education"].notna().sum() if "education" in nat_lab.columns else 0
        n_maori = nat_lab["maori"].notna().sum() if "maori" in nat_lab.columns else 0

        year_stats.append({
            "year": year, "n_total": n_total, "n_natlab": n_natlab,
            "n_age": n_age, "n_female": n_female, "n_edu": n_edu, "n_maori": n_maori,
        })

    print(f"\n  {'Year':>6s}  {'Total':>6s}  {'Nat/Lab':>8s}  {'Age':>5s}  {'Female':>7s}  {'Educ':>5s}  {'Maori':>6s}")
    for s in year_stats:
        print(f"  {s['year']:>6d}  {s['n_total']:>6d}  {s['n_natlab']:>8d}  "
              f"{s['n_age']:>5d}  {s['n_female']:>7d}  {s['n_edu']:>5d}  {s['n_maori']:>6d}")

    conn.close()

    # ── 2. Run logistic regressions per year ───────────────────────────
    print(f"\n{'=' * 70}")
    print("2. LOGISTIC REGRESSION: Demographics → National vs Labour")
    print(f"{'=' * 70}")

    results = []

    for year in sorted(YEAR_CONFIG.keys()):
        df = all_years[year]
        nat_lab = df.dropna(subset=["nat_vs_lab"]).copy()
        y = nat_lab["nat_vs_lab"]

        # Determine which predictors are available
        predictors = []
        if nat_lab["age"].notna().mean() > 0.5:
            predictors.append("age")
        if "female" in nat_lab.columns and nat_lab["female"].notna().mean() > 0.5:
            predictors.append("female")
        if "education" in nat_lab.columns and nat_lab["education"].notna().mean() > 0.3:
            predictors.append("education")
        if "maori" in nat_lab.columns and nat_lab["maori"].notna().mean() > 0.3:
            predictors.append("maori")

        # Drop rows with missing predictors
        sub = nat_lab.dropna(subset=predictors + ["nat_vs_lab"])
        if len(sub) < 50:
            print(f"  {year}: Too few observations ({len(sub)}) — skipping")
            continue

        y = sub["nat_vs_lab"]
        X = sub[predictors].astype(float)
        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=0)
            pseudo_r2 = model.prsquared  # McFadden's pseudo-R²
            n = len(sub)
            aic = model.aic

            # Get individual coefficients
            coefs = {}
            for pred in predictors:
                if pred in model.params.index:
                    coefs[pred] = {
                        "coef": model.params[pred],
                        "pval": model.pvalues[pred],
                        "odds": np.exp(model.params[pred]),
                    }

            results.append({
                "year": year,
                "pseudo_r2": pseudo_r2,
                "n": n,
                "aic": aic,
                "predictors": predictors,
                "coefs": coefs,
            })

            pred_str = "+".join(predictors)
            print(f"  {year}: pseudo-R² = {pseudo_r2:.4f}, n = {n:,d}, predictors: {pred_str}")
            for pred, c in coefs.items():
                sig = "***" if c["pval"] < 0.001 else "**" if c["pval"] < 0.01 else "*" if c["pval"] < 0.05 else ""
                print(f"         {pred:>12s}: β = {c['coef']:+.4f}, OR = {c['odds']:.3f}, p = {c['pval']:.4f} {sig}")

        except Exception as e:
            print(f"  {year}: Error — {e}")

    # ── 3. Test for dealignment trend ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("3. DEALIGNMENT TREND: Is demographic pseudo-R² declining?")
    print(f"{'=' * 70}")

    if len(results) >= 5:
        years = [r["year"] for r in results]
        r2s = [r["pseudo_r2"] for r in results]

        r, p = pearsonr(years, r2s)
        print(f"\n  Correlation (year vs pseudo-R²): r = {r:+.3f}, p = {p:.3f}")

        if r < 0 and p < 0.10:
            print(f"  → DEALIGNMENT CONFIRMED: demographic predictive power declining")
        elif r > 0 and p < 0.10:
            print(f"  → REALIGNMENT: demographic predictive power INCREASING")
        else:
            print(f"  → No significant trend in demographic predictive power")

    # ── 4. Individual predictor trends ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("4. INDIVIDUAL PREDICTOR TRENDS")
    print(f"{'=' * 70}")

    for pred in ["age", "female", "education", "maori"]:
        pred_years = []
        pred_coefs = []
        for r in results:
            if pred in r["coefs"]:
                pred_years.append(r["year"])
                pred_coefs.append(r["coefs"][pred]["coef"])

        if len(pred_years) >= 4:
            r_trend, p_trend = pearsonr(pred_years, pred_coefs)
            direction = "→ 0 (weakening)" if abs(pred_coefs[-1]) < abs(pred_coefs[0]) else "away from 0 (strengthening)"
            print(f"\n  {pred:>12s}: trend r = {r_trend:+.3f} (p = {p_trend:.3f})")
            print(f"               {pred_coefs[0]:+.4f} ({pred_years[0]}) → {pred_coefs[-1]:+.4f} ({pred_years[-1]}): {direction}")

    # ── 5. Age effect over time (key dealignment indicator) ────────────
    print(f"\n{'=' * 70}")
    print("5. AGE EFFECT OVER TIME")
    print(f"{'=' * 70}")

    print(f"\n  The age coefficient represents the effect of a 1-year increase in age")
    print(f"  on the log-odds of voting National vs Labour.")
    print(f"  Positive = older voters more likely to vote National.\n")

    for r in results:
        if "age" in r["coefs"]:
            c = r["coefs"]["age"]
            # Per-decade effect
            or_decade = np.exp(c["coef"] * 10)
            print(f"  {r['year']}: β = {c['coef']:+.4f}, OR per decade = {or_decade:.3f}, p = {c['pval']:.4f}")

    # ── 6. Reduced model (age + female only — available all years) ─────
    print(f"\n{'=' * 70}")
    print("6. REDUCED MODEL: Age + Gender only (consistent across years)")
    print(f"{'=' * 70}")

    conn = sqlite3.connect(str(NZES_DB))
    reduced_results = []

    for year in sorted(YEAR_CONFIG.keys()):
        df = all_years[year]
        sub = df.dropna(subset=["nat_vs_lab", "age", "female"])
        if len(sub) < 50:
            continue

        y = sub["nat_vs_lab"]
        X = sm.add_constant(sub[["age", "female"]].astype(float))

        try:
            model = sm.Logit(y, X).fit(disp=0)
            reduced_results.append({
                "year": year,
                "pseudo_r2": model.prsquared,
                "n": len(sub),
                "age_coef": model.params.get("age", np.nan),
                "female_coef": model.params.get("female", np.nan),
                "age_p": model.pvalues.get("age", np.nan),
                "female_p": model.pvalues.get("female", np.nan),
            })
        except Exception:
            pass

    conn.close()

    print(f"\n  {'Year':>6s}  {'pseudo-R²':>10s}  {'n':>6s}  {'Age β':>8s}  {'p':>8s}  {'Female β':>10s}  {'p':>8s}")
    for r in reduced_results:
        print(f"  {r['year']:>6d}  {r['pseudo_r2']:>10.4f}  {r['n']:>6d}  "
              f"{r['age_coef']:>+7.4f}  {r['age_p']:>8.4f}  "
              f"{r['female_coef']:>+9.4f}  {r['female_p']:>8.4f}")

    if len(reduced_results) >= 5:
        years_r = [r["year"] for r in reduced_results]
        r2s_r = [r["pseudo_r2"] for r in reduced_results]
        r_trend, p_trend = pearsonr(years_r, r2s_r)
        print(f"\n  Reduced model trend (year vs pseudo-R²): r = {r_trend:+.3f}, p = {p_trend:.3f}")

    # ── 7. Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("7. SUMMARY")
    print(f"{'=' * 70}")

    if results:
        first = results[0]
        last = results[-1]
        print(f"\n  Full model pseudo-R²:")
        print(f"    {first['year']}: {first['pseudo_r2']:.4f}")
        print(f"    {last['year']}: {last['pseudo_r2']:.4f}")
        change = last['pseudo_r2'] - first['pseudo_r2']
        pct = (change / first['pseudo_r2']) * 100 if first['pseudo_r2'] > 0 else 0
        print(f"    Change: {change:+.4f} ({pct:+.1f}%)")

    if reduced_results:
        first_r = reduced_results[0]
        last_r = reduced_results[-1]
        print(f"\n  Reduced model (age+gender) pseudo-R²:")
        print(f"    {first_r['year']}: {first_r['pseudo_r2']:.4f}")
        print(f"    {last_r['year']}: {last_r['pseudo_r2']:.4f}")
        change_r = last_r['pseudo_r2'] - first_r['pseudo_r2']
        pct_r = (change_r / first_r['pseudo_r2']) * 100 if first_r['pseudo_r2'] > 0 else 0
        print(f"    Change: {change_r:+.4f} ({pct_r:+.1f}%)")

    print(f"\n  Dalton (2000) predicts dealignment: demographics should explain LESS")
    print(f"  of the vote over time as social cleavages weaken.")

    # ── 8. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Pseudo-R² over time (full & reduced models)
    ax = axes[0, 0]
    if results:
        ax.plot([r["year"] for r in results], [r["pseudo_r2"] for r in results],
                "o-", color="#2c3e50", linewidth=2, markersize=6, label="Full model")
    if reduced_results:
        ax.plot([r["year"] for r in reduced_results], [r["pseudo_r2"] for r in reduced_results],
                "s--", color="#e74c3c", linewidth=1.5, markersize=5, label="Age + gender only")
    ax.set_xlabel("Election year")
    ax.set_ylabel("McFadden's pseudo-R²")
    ax.set_title("Demographic Predictive Power Over Time")
    ax.legend(fontsize=9)

    # Panel 2: Age coefficient over time
    ax = axes[0, 1]
    age_years = [r["year"] for r in results if "age" in r["coefs"]]
    age_coefs = [r["coefs"]["age"]["coef"] for r in results if "age" in r["coefs"]]
    if age_years:
        ax.plot(age_years, age_coefs, "o-", color="#3498db", linewidth=2, markersize=6)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Election year")
        ax.set_ylabel("Age coefficient (log-odds)")
        ax.set_title("Age → National Vote (per year of age)")

    # Panel 3: Education coefficient over time
    ax = axes[1, 0]
    edu_years = [r["year"] for r in results if "education" in r["coefs"]]
    edu_coefs = [r["coefs"]["education"]["coef"] for r in results if "education" in r["coefs"]]
    if edu_years:
        ax.plot(edu_years, edu_coefs, "o-", color="#27ae60", linewidth=2, markersize=6)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Election year")
        ax.set_ylabel("Education coefficient (log-odds)")
        ax.set_title("Education → National Vote (per level)")

    # Panel 4: Maori coefficient over time
    ax = axes[1, 1]
    maori_years = [r["year"] for r in results if "maori" in r["coefs"]]
    maori_coefs = [r["coefs"]["maori"]["coef"] for r in results if "maori" in r["coefs"]]
    if maori_years:
        ax.plot(maori_years, maori_coefs, "o-", color="#e67e22", linewidth=2, markersize=6)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Election year")
        ax.set_ylabel("Maori coefficient (log-odds)")
        ax.set_title("Maori Identity → National Vote")

    fig.suptitle("Dealignment Test: Are Demographics Less Predictive of Vote Over Time?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    outpath = REPORTS_DIR / "dealignment.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
