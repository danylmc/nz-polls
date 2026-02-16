#!/usr/bin/env python3
"""
Economic Voting Extensions

Tests three hypotheses from the political science literature:
1A. Retrospective vs Prospective Voting (Fiorina 1981)
1B. Myopic vs Cumulative Retrospection (Achen & Bartels 2016)
1C. Cost of Ruling (Paldam 1991)

Uses existing polls_with_economics.csv + fresh NZES extraction for prospective
economic assessments.
"""

import os
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")
NZES_DB = DATA_DIR / "nzesdb.sqlite"

PARTY_COLORS = {
    "National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
    "ACT": "#FDE401", "NZ First": "#000000",
}

# From deep_data.py
GOVERNMENT_TIMELINE = [
    ("1990-11-02", "1996-12-15", "National"),
    ("1996-12-16", "1999-12-05", "National"),
    ("1999-12-06", "2008-11-19", "Labour"),
    ("2008-11-19", "2017-10-26", "National"),
    ("2017-10-26", "2023-11-27", "Labour"),
    ("2023-11-27", "2030-01-01", "National"),
]

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

INCUMBENTS = {
    1996: "National", 1999: "National", 2002: "Labour", 2005: "Labour",
    2008: "Labour", 2011: "National", 2014: "National", 2017: "National",
    2020: "Labour", 2023: "Labour",
}

# Government periods with coalition classification
GOVERNMENT_PERIODS = [
    {"start": "1990-11-02", "end": "1996-12-15", "party": "National",
     "type": "single", "partners": 0, "label": "Bolger (Nat alone)"},
    {"start": "1996-12-16", "end": "1999-12-05", "party": "National",
     "type": "coalition", "partners": 1, "label": "Shipley (Nat-NZF)"},
    {"start": "1999-12-06", "end": "2008-11-19", "party": "Labour",
     "type": "coalition", "partners": 2, "label": "Clark (Lab-All/Prog/NZF)"},
    {"start": "2008-11-19", "end": "2017-10-26", "party": "National",
     "type": "coalition", "partners": 3, "label": "Key/English (Nat-ACT-UF-Maori)"},
    {"start": "2017-10-26", "end": "2020-10-17", "party": "Labour",
     "type": "coalition", "partners": 2, "label": "Ardern (Lab-NZF-Grn)"},
    {"start": "2020-10-17", "end": "2023-11-27", "party": "Labour",
     "type": "single", "partners": 0, "label": "Ardern/Hipkins (Lab majority)"},
    {"start": "2023-11-27", "end": "2030-01-01", "party": "National",
     "type": "coalition", "partners": 2, "label": "Luxon (Nat-ACT-NZF)"},
]

# Prospective economy variables in NZES
PROSPECTIVE_VARS = {
    1990: {"table": "nzes_1990", "retro": None, "prosp": "FFINNZ",
           "party_vote": None, "dk_code": 6},
    2002: {"table": "nzes_2002", "retro": "WECSTAT", "prosp": "WECNZFT",
           "party_vote": "WVOT02P", "dk_code": 9},
    2005: {"table": "nzes_2005", "retro": "yecstat", "prosp": "ycoufut",
           "party_vote": "yvot05p", "dk_code": 9},
    2008: {"table": "nzes_2008", "retro": "zretcon", "prosp": "zprscon",
           "party_vote": "zvt08p", "dk_code": 9},
}

# Party vote recode for prospective analysis years
PARTY_VOTE_RECODE = {
    2002: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           7: "Alliance", 9: "Progressive", 11: "United Future"},
    2005: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
    2008: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 8: "Progressive"},
}


def get_incumbent_party(date):
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
    for start, end, party in GOVERNMENT_TIMELINE:
        if start <= date_str <= end:
            return party
    return None


def get_government_info(date):
    date_str = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
    for gov in GOVERNMENT_PERIODS:
        if gov["start"] <= date_str <= gov["end"]:
            return gov
    return None


# ─── Data Loading ───────────────────────────────────────────────────────────────

def load_polls():
    df = pd.read_csv(DATA_DIR / "polls_with_economics.csv", parse_dates=["date"])
    df = df.dropna(subset=["incumbent_support"])
    print(f"Loaded {len(df)} polls with incumbent support data")
    return df


def load_nzes():
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    print(f"Loaded {len(df)} NZES respondents")
    return df


def extract_prospective_data():
    """Extract retrospective + prospective economic assessments from NZES SQLite."""
    print("\nExtracting prospective economy data from NZES...")
    conn = sqlite3.connect(str(NZES_DB))
    all_rows = []

    for year, spec in PROSPECTIVE_VARS.items():
        if year == 1990:
            # 1990 has no party vote or retrospective — skip for logistic analysis
            continue

        table = spec["table"]
        prosp_col = spec["prosp"]
        retro_col = spec["retro"]
        pv_col = spec["party_vote"]
        dk = spec["dk_code"]

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"  {year} ({table}): {len(df)} raw rows")

        inc_party = INCUMBENTS.get(year)
        if not inc_party:
            continue

        recode = PARTY_VOTE_RECODE.get(year, {})

        for _, raw in df.iterrows():
            row = {"election_year": year}

            # Party vote
            if pv_col and pv_col in raw.index:
                pv = raw[pv_col]
                if pd.notna(pv):
                    pv_int = int(pv)
                    party = recode.get(pv_int)
                    if party:
                        row["party_vote"] = party
                        row["vote_incumbent"] = 1 if party == inc_party else 0

            # Prospective assessment (1=lot better ... 5=lot worse)
            if prosp_col and prosp_col in raw.index:
                val = raw[prosp_col]
                if pd.notna(val):
                    v = float(val)
                    if 1 <= v <= 5:
                        # Invert so higher = more optimistic
                        row["prospective"] = 6 - v

            # Retrospective assessment
            if retro_col and retro_col in raw.index:
                val = raw[retro_col]
                if pd.notna(val):
                    v = float(val)
                    if 1 <= v <= 5:
                        row["retrospective"] = 6 - v

            # Education for sophistication interaction
            edu_cols = {"2002": "WEDUCAT", "2005": "yeducat", "2008": "zeducat"}
            edu_col = edu_cols.get(str(year))
            if edu_col and edu_col in raw.index:
                val = raw[edu_col]
                if pd.notna(val):
                    v = int(val)
                    if 0 <= v <= 20:
                        row["education"] = v

            all_rows.append(row)

    conn.close()
    result = pd.DataFrame(all_rows)
    print(f"  Total extracted: {len(result)} rows with prospective data")
    return result


# ─── 1A. Retrospective vs Prospective Voting ───────────────────────────────────

def analyze_retro_vs_prospective(prosp_df):
    """Compare retrospective vs prospective economic voting."""
    print("\n" + "=" * 70)
    print("1A. Retrospective vs Prospective Voting (Fiorina 1981)")
    print("=" * 70)

    results = []
    years = sorted(prosp_df["election_year"].unique())

    for year in years:
        yr = prosp_df[prosp_df["election_year"] == year].copy()

        # Need vote_incumbent, retrospective, and prospective
        has_retro = yr["retrospective"].notna().sum() > 30
        has_prosp = yr["prospective"].notna().sum() > 30
        has_vote = yr["vote_incumbent"].notna().sum() > 30

        if not has_vote:
            print(f"  {year}: Insufficient vote data, skipping")
            continue

        year_results = {"year": year}

        # Retrospective-only model
        if has_retro:
            valid = yr[["vote_incumbent", "retrospective"]].dropna()
            if len(valid) > 30:
                try:
                    X = sm.add_constant(valid[["retrospective"]])
                    model = Logit(valid["vote_incumbent"], X).fit(disp=0, maxiter=100)
                    year_results["retro_r2"] = model.prsquared
                    year_results["retro_coef"] = model.params.get("retrospective", np.nan)
                    year_results["retro_p"] = model.pvalues.get("retrospective", np.nan)
                    year_results["retro_n"] = len(valid)
                    print(f"  {year} Retrospective: R²={model.prsquared:.4f}, "
                          f"β={model.params.get('retrospective', 0):.3f}, n={len(valid)}")
                except Exception as e:
                    print(f"  {year} Retrospective error: {e}")

        # Prospective-only model
        if has_prosp:
            valid = yr[["vote_incumbent", "prospective"]].dropna()
            if len(valid) > 30:
                try:
                    X = sm.add_constant(valid[["prospective"]])
                    model = Logit(valid["vote_incumbent"], X).fit(disp=0, maxiter=100)
                    year_results["prosp_r2"] = model.prsquared
                    year_results["prosp_coef"] = model.params.get("prospective", np.nan)
                    year_results["prosp_p"] = model.pvalues.get("prospective", np.nan)
                    year_results["prosp_n"] = len(valid)
                    print(f"  {year} Prospective:   R²={model.prsquared:.4f}, "
                          f"β={model.params.get('prospective', 0):.3f}, n={len(valid)}")
                except Exception as e:
                    print(f"  {year} Prospective error: {e}")

        # Combined model
        if has_retro and has_prosp:
            valid = yr[["vote_incumbent", "retrospective", "prospective"]].dropna()
            if len(valid) > 30:
                try:
                    X = sm.add_constant(valid[["retrospective", "prospective"]])
                    model = Logit(valid["vote_incumbent"], X).fit(disp=0, maxiter=100)
                    year_results["combined_r2"] = model.prsquared
                    year_results["combined_retro_coef"] = model.params.get("retrospective", np.nan)
                    year_results["combined_prosp_coef"] = model.params.get("prospective", np.nan)
                    year_results["combined_retro_p"] = model.pvalues.get("retrospective", np.nan)
                    year_results["combined_prosp_p"] = model.pvalues.get("prospective", np.nan)
                    year_results["combined_n"] = len(valid)
                    print(f"  {year} Combined:      R²={model.prsquared:.4f}, n={len(valid)}")
                except Exception as e:
                    print(f"  {year} Combined error: {e}")

        # Education interaction (sophistication moderator)
        if has_prosp and "education" in yr.columns:
            valid = yr[["vote_incumbent", "prospective", "education"]].dropna()
            if len(valid) > 50:
                valid["edu_high"] = (valid["education"] >= valid["education"].median()).astype(float)
                valid["prosp_x_edu"] = valid["prospective"] * valid["edu_high"]
                try:
                    X = sm.add_constant(valid[["prospective", "edu_high", "prosp_x_edu"]])
                    model = Logit(valid["vote_incumbent"], X).fit(disp=0, maxiter=100)
                    year_results["interaction_coef"] = model.params.get("prosp_x_edu", np.nan)
                    year_results["interaction_p"] = model.pvalues.get("prosp_x_edu", np.nan)
                    print(f"  {year} Prosp×Edu:     β={model.params.get('prosp_x_edu', 0):.3f}, "
                          f"p={model.pvalues.get('prosp_x_edu', 1):.4f}")
                except Exception as e:
                    print(f"  {year} Interaction error: {e}")

        results.append(year_results)

    results_df = pd.DataFrame(results)

    # Visualization: pseudo-R² comparison
    if not results_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: R² comparison
        ax = axes[0]
        valid = results_df.dropna(subset=["retro_r2", "prosp_r2"])
        if not valid.empty:
            x = np.arange(len(valid))
            w = 0.3
            ax.bar(x - w, valid["retro_r2"], w, label="Retrospective", color="#D82A20", alpha=0.8)
            ax.bar(x, valid["prosp_r2"], w, label="Prospective", color="#00529F", alpha=0.8)
            if "combined_r2" in valid.columns:
                comb = valid["combined_r2"].fillna(0)
                ax.bar(x + w, comb, w, label="Combined", color="#098137", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(y)) for y in valid["year"]])
            ax.set_ylabel("McFadden Pseudo-R²")
            ax.set_title("Model Fit: Retrospective vs Prospective")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: Coefficient comparison
        ax = axes[1]
        if "retro_coef" in results_df.columns and "prosp_coef" in results_df.columns:
            valid = results_df.dropna(subset=["retro_coef", "prosp_coef"])
            if not valid.empty:
                ax.plot(valid["year"], valid["retro_coef"], "s-", color="#D82A20",
                        label="Retrospective β", markersize=7)
                ax.plot(valid["year"], valid["prosp_coef"], "o-", color="#00529F",
                        label="Prospective β", markersize=7)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_ylabel("Logit Coefficient")
                ax.set_title("Coefficient Comparison")
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        plt.suptitle("Retrospective vs Prospective Economic Voting in NZ",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / "evx_retro_vs_prospective.png", dpi=150)
        plt.close()

    return results_df


# ─── 1B. Myopic vs Cumulative Retrospection ───────────────────────────────────

def analyze_myopia(polls_df):
    """Test whether voters weight recent economic conditions more than earlier ones."""
    print("\n" + "=" * 70)
    print("1B. Myopic vs Cumulative Retrospection (Achen & Bartels 2016)")
    print("=" * 70)

    df = polls_df.copy()
    df["year_decimal"] = df["date"].dt.year + df["date"].dt.dayofyear / 365.25

    # Assign government period to each poll
    df["gov_start"] = None
    df["gov_label"] = None
    for gov in GOVERNMENT_PERIODS:
        mask = (df["date"] >= pd.Timestamp(gov["start"])) & \
               (df["date"] <= pd.Timestamp(gov["end"]))
        df.loc[mask, "gov_start"] = pd.Timestamp(gov["start"])
        df.loc[mask, "gov_label"] = gov["label"]

    df["months_in_office"] = (df["date"] - pd.to_datetime(df["gov_start"])).dt.days / 30.44

    results = {}

    # --- Election-level analysis: early vs late-term economics ---
    print("\n  Election-level: early vs late-term economics")
    election_rows = []
    for year, res in ELECTION_RESULTS.items():
        if year < 1996:
            continue
        inc = res["incumbent"]
        inc_vote = res.get(inc, np.nan)

        # Find government start
        elec_date = pd.Timestamp(f"{year}-10-01")
        gov = get_government_info(elec_date)
        if not gov:
            continue
        gov_start = pd.Timestamp(gov["start"])
        term_length_q = max(1, int((elec_date - gov_start).days / 91))

        # Get quarterly economics
        econ = pd.read_csv(DATA_DIR / "quarterly_economics.csv",
                           parse_dates=["quarter"], index_col="quarter")
        term_econ = econ[(econ.index >= gov_start) & (econ.index <= elec_date)]

        if len(term_econ) < 4:
            continue

        n_q = len(term_econ)
        early_n = min(4, n_q // 3)
        late_n = min(4, n_q // 3)

        row = {
            "year": year, "incumbent": inc, "incumbent_vote": inc_vote,
            "gov_type": gov["type"],
        }

        if "gdp_growth_qoq" in term_econ.columns:
            row["early_gdp"] = term_econ["gdp_growth_qoq"].iloc[:early_n].mean()
            row["late_gdp"] = term_econ["gdp_growth_qoq"].iloc[-late_n:].mean()
            row["full_term_gdp"] = term_econ["gdp_growth_qoq"].mean()

        if "inflation_yoy" in term_econ.columns:
            row["early_infl"] = term_econ["inflation_yoy"].iloc[:early_n].mean()
            row["late_infl"] = term_econ["inflation_yoy"].iloc[-late_n:].mean()

        election_rows.append(row)

    elec_df = pd.DataFrame(election_rows)
    results["election_level"] = elec_df

    if not elec_df.empty and "late_gdp" in elec_df.columns and "early_gdp" in elec_df.columns:
        valid = elec_df.dropna(subset=["incumbent_vote", "late_gdp", "early_gdp"])
        if len(valid) >= 5:
            r_late, p_late = stats.pearsonr(valid["incumbent_vote"], valid["late_gdp"])
            r_early, p_early = stats.pearsonr(valid["incumbent_vote"], valid["early_gdp"])
            r_full, p_full = stats.pearsonr(valid["incumbent_vote"], valid["full_term_gdp"])
            print(f"    Late-term GDP vs inc vote:  r={r_late:.3f}, p={p_late:.4f}")
            print(f"    Early-term GDP vs inc vote: r={r_early:.3f}, p={p_early:.4f}")
            print(f"    Full-term GDP vs inc vote:  r={r_full:.3f}, p={p_full:.4f}")
            results["r_late"] = r_late
            results["r_early"] = r_early
            results["r_full"] = r_full

    # --- Poll-level analysis: recency weighting ---
    print("\n  Poll-level: lag decay (recency weighting)")
    lag_cols = ["gdp_growth_qoq", "gdp_growth_qoq_lag1",
                "gdp_growth_qoq_lag2", "gdp_growth_qoq_lag4"]
    available = [c for c in lag_cols if c in df.columns]

    if len(available) >= 3:
        valid = df[["incumbent_support"] + available].dropna()
        if len(valid) > 50:
            # Individual lag correlations
            lag_results = []
            for col in available:
                r, p = stats.pearsonr(valid["incumbent_support"], valid[col])
                lag_label = col.replace("gdp_growth_qoq", "").replace("_lag", "t-").strip()
                if not lag_label:
                    lag_label = "t"
                lag_results.append({"lag": lag_label, "col": col, "r": r, "p": p})
                print(f"    {lag_label}: r={r:.4f}, p={p:.4f}")
            results["lag_decay"] = pd.DataFrame(lag_results)

            # OLS with all lags
            X = sm.add_constant(valid[available])
            model = sm.OLS(valid["incumbent_support"], X).fit(cov_type="HC1")
            results["lag_model"] = model
            print(f"    Full lag model R²={model.rsquared:.4f}")
            for var in available:
                print(f"      {var}: β={model.params[var]:.3f}, p={model.pvalues[var]:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Early vs late-term GDP scatter
    ax = axes[0]
    if not elec_df.empty:
        valid = elec_df.dropna(subset=["incumbent_vote", "late_gdp", "early_gdp"])
        if not valid.empty:
            ax.scatter(valid["late_gdp"], valid["incumbent_vote"],
                       color="#D82A20", s=80, zorder=5, label="Late-term GDP")
            for _, row in valid.iterrows():
                ax.annotate(str(int(row["year"])),
                           (row["late_gdp"], row["incumbent_vote"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
            if len(valid) >= 3:
                z = np.polyfit(valid["late_gdp"], valid["incumbent_vote"], 1)
                x_range = np.linspace(valid["late_gdp"].min(), valid["late_gdp"].max(), 50)
                ax.plot(x_range, np.poly1d(z)(x_range), "--", color="#D82A20", alpha=0.5)
    ax.set_xlabel("Late-Term GDP Growth (q/q %)")
    ax.set_ylabel("Incumbent Vote Share (%)")
    ax.set_title("Late-Term Economy vs Election Result")
    ax.grid(True, alpha=0.3)

    # Panel 2: Early-term GDP scatter
    ax = axes[1]
    if not elec_df.empty:
        valid = elec_df.dropna(subset=["incumbent_vote", "early_gdp"])
        if not valid.empty:
            ax.scatter(valid["early_gdp"], valid["incumbent_vote"],
                       color="#00529F", s=80, zorder=5, label="Early-term GDP")
            for _, row in valid.iterrows():
                ax.annotate(str(int(row["year"])),
                           (row["early_gdp"], row["incumbent_vote"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
            if len(valid) >= 3:
                z = np.polyfit(valid["early_gdp"], valid["incumbent_vote"], 1)
                x_range = np.linspace(valid["early_gdp"].min(), valid["early_gdp"].max(), 50)
                ax.plot(x_range, np.poly1d(z)(x_range), "--", color="#00529F", alpha=0.5)
    ax.set_xlabel("Early-Term GDP Growth (q/q %)")
    ax.set_ylabel("Incumbent Vote Share (%)")
    ax.set_title("Early-Term Economy vs Election Result")
    ax.grid(True, alpha=0.3)

    # Panel 3: Lag decay plot
    ax = axes[2]
    if "lag_decay" in results:
        ld = results["lag_decay"]
        ax.bar(range(len(ld)), ld["r"],
               color=["#D82A20", "#ff7f0e", "#2ca02c", "#00529F"][:len(ld)],
               alpha=0.8)
        ax.set_xticks(range(len(ld)))
        ax.set_xticklabels(ld["lag"])
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Correlation with Incumbent Support")
        ax.set_title("Lag Decay: Recency Weighting")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Myopic vs Cumulative Retrospection (Achen & Bartels 2016)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "evx_myopia.png", dpi=150)
    plt.close()

    return results


# ─── 1C. Cost of Ruling ────────────────────────────────────────────────────────

def analyze_cost_of_ruling(polls_df):
    """Test whether incumbent fatigue is steeper for coalition governments."""
    print("\n" + "=" * 70)
    print("1C. Cost of Ruling (Paldam 1991)")
    print("=" * 70)

    df = polls_df.copy()

    # Assign government info to each poll
    rows = []
    for _, poll in df.iterrows():
        gov = get_government_info(poll["date"])
        if gov is None:
            continue
        row = poll.to_dict()
        row["gov_start"] = pd.Timestamp(gov["start"])
        row["gov_label"] = gov["label"]
        row["gov_type"] = gov["type"]
        row["is_coalition"] = 1 if gov["type"] == "coalition" else 0
        row["n_partners"] = gov["partners"]
        row["tenure_months"] = (poll["date"] - pd.Timestamp(gov["start"])).days / 30.44
        row["incumbent_is_national"] = 1 if gov["party"] == "National" else 0
        rows.append(row)

    gdf = pd.DataFrame(rows)
    gdf = gdf.dropna(subset=["incumbent_support", "tenure_months"])
    print(f"  {len(gdf)} polls with government assignment")

    results = {}

    # Base model: incumbent_support ~ tenure + incumbent_is_national
    valid = gdf[["incumbent_support", "tenure_months", "incumbent_is_national"]].dropna()
    X = sm.add_constant(valid[["tenure_months", "incumbent_is_national"]])
    base_model = sm.OLS(valid["incumbent_support"], X).fit(cov_type="HC1")
    results["base_model"] = base_model
    print(f"\n  Base model (R²={base_model.rsquared:.4f}):")
    print(f"    Tenure: β={base_model.params['tenure_months']:.4f}/month, "
          f"p={base_model.pvalues['tenure_months']:.4f}")

    # Extended model: tenure * is_coalition
    valid2 = gdf[["incumbent_support", "tenure_months", "is_coalition",
                   "incumbent_is_national"]].dropna()
    valid2["tenure_x_coalition"] = valid2["tenure_months"] * valid2["is_coalition"]
    X2 = sm.add_constant(valid2[["tenure_months", "is_coalition",
                                  "tenure_x_coalition", "incumbent_is_national"]])
    coalition_model = sm.OLS(valid2["incumbent_support"], X2).fit(cov_type="HC1")
    results["coalition_model"] = coalition_model
    print(f"\n  Coalition interaction model (R²={coalition_model.rsquared:.4f}):")
    for var in ["tenure_months", "is_coalition", "tenure_x_coalition"]:
        print(f"    {var}: β={coalition_model.params[var]:.4f}, "
              f"p={coalition_model.pvalues[var]:.4f}")

    # Economic interaction: tenure * GDP
    if "gdp_growth_qoq" in gdf.columns:
        valid3 = gdf[["incumbent_support", "tenure_months", "gdp_growth_qoq",
                       "incumbent_is_national"]].dropna()
        valid3["tenure_x_gdp"] = valid3["tenure_months"] * valid3["gdp_growth_qoq"]
        X3 = sm.add_constant(valid3[["tenure_months", "gdp_growth_qoq",
                                      "tenure_x_gdp", "incumbent_is_national"]])
        econ_model = sm.OLS(valid3["incumbent_support"], X3).fit(cov_type="HC1")
        results["econ_model"] = econ_model
        print(f"\n  Economic interaction model (R²={econ_model.rsquared:.4f}):")
        for var in ["tenure_months", "gdp_growth_qoq", "tenure_x_gdp"]:
            print(f"    {var}: β={econ_model.params[var]:.4f}, "
                  f"p={econ_model.pvalues[var]:.4f}")

    # Per-government fatigue rates
    print("\n  Per-government fatigue rates:")
    gov_results = []
    for gov in GOVERNMENT_PERIODS:
        gov_polls = gdf[gdf["gov_label"] == gov["label"]].copy()
        if len(gov_polls) < 20:
            continue
        valid_g = gov_polls[["incumbent_support", "tenure_months"]].dropna()
        if len(valid_g) < 20:
            continue
        r, p = stats.pearsonr(valid_g["incumbent_support"], valid_g["tenure_months"])
        # Slope via OLS
        X_g = sm.add_constant(valid_g[["tenure_months"]])
        m = sm.OLS(valid_g["incumbent_support"], X_g).fit()
        slope = m.params["tenure_months"]
        gov_results.append({
            "label": gov["label"], "type": gov["type"],
            "n": len(valid_g), "r": r, "p": p,
            "slope_per_month": slope,
            "slope_per_year": slope * 12,
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {gov['label']:<40} slope={slope*12:+.2f}pp/yr  r={r:.3f} {sig}  n={len(valid_g)}")

    results["gov_rates"] = pd.DataFrame(gov_results)

    # Compare coalition vs single-party fatigue rates
    if not results["gov_rates"].empty:
        gr = results["gov_rates"]
        coalition_rates = gr[gr["type"] == "coalition"]["slope_per_year"]
        single_rates = gr[gr["type"] == "single"]["slope_per_year"]
        if len(coalition_rates) > 0 and len(single_rates) > 0:
            print(f"\n  Mean fatigue rate (coalition): {coalition_rates.mean():.2f} pp/year")
            print(f"  Mean fatigue rate (single):    {single_rates.mean():.2f} pp/year")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Per-government fatigue curves
    ax = axes[0]
    colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(GOVERNMENT_PERIODS)))
    for i, gov in enumerate(GOVERNMENT_PERIODS):
        gov_polls = gdf[gdf["gov_label"] == gov["label"]].copy()
        if len(gov_polls) < 10:
            continue
        gov_polls = gov_polls.sort_values("tenure_months")
        # Rolling average for smoothing
        rolling = gov_polls.set_index("tenure_months")["incumbent_support"].rolling(
            window=10, min_periods=5, center=True).mean()
        linestyle = "-" if gov["type"] == "coalition" else "--"
        ax.plot(rolling.index, rolling.values, linestyle=linestyle,
                color=colors_cycle[i], linewidth=1.5, label=gov["label"][:25], alpha=0.8)
    ax.set_xlabel("Months in Office")
    ax.set_ylabel("Incumbent Support (%)")
    ax.set_title("Fatigue Curves by Government\n(solid=coalition, dashed=single-party)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Fatigue rate comparison
    ax = axes[1]
    if not results["gov_rates"].empty:
        gr = results["gov_rates"].sort_values("slope_per_year")
        colors = ["#D82A20" if t == "coalition" else "#00529F"
                  for t in gr["type"]]
        bars = ax.barh(range(len(gr)), gr["slope_per_year"], color=colors, alpha=0.7)
        ax.set_yticks(range(len(gr)))
        ax.set_yticklabels(gr["label"], fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Fatigue Rate (pp/year)")
        ax.set_title("Fatigue Rate by Government\n(red=coalition, blue=single-party)")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Cost of Ruling: Incumbent Fatigue in NZ (Paldam 1991)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "evx_cost_of_ruling.png", dpi=150)
    plt.close()

    return results


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(retro_prosp_df, myopia_results, cost_results):
    """Generate markdown report for economic voting extensions."""
    report = []
    report.append("# Economic Voting Extensions\n")
    report.append("*Three hypotheses from the economic voting literature, tested on NZ data*\n")

    # ── 1A: Retrospective vs Prospective ──
    report.append("## 1A. Retrospective vs Prospective Voting (Fiorina 1981)\n")
    report.append("**Question**: Do NZ voters look backward or forward when punishing/rewarding incumbents?\n")
    report.append("The \"peasant\" model predicts retrospective voting dominates; the \"banker's\" model "
                  "predicts prospective voting is stronger, especially among sophisticated voters.\n")

    if retro_prosp_df is not None and not retro_prosp_df.empty:
        report.append("### Pseudo-R² Comparison\n")
        report.append("| Year | Retro R² | Prosp R² | Combined R² | Retro β | Prosp β | Prosp×Edu β | n |")
        report.append("|------|----------|----------|-------------|---------|---------|-------------|---|")
        for _, row in retro_prosp_df.iterrows():
            def fmt(val, p_val=None):
                if pd.isna(val):
                    return "—"
                sig = ""
                if p_val is not None and pd.notna(p_val):
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                return f"{val:.4f}{sig}"

            rr2 = fmt(row.get("retro_r2"))
            pr2 = fmt(row.get("prosp_r2"))
            cr2 = fmt(row.get("combined_r2"))
            rc = fmt(row.get("retro_coef"), row.get("retro_p"))
            pc = fmt(row.get("prosp_coef"), row.get("prosp_p"))
            ic = fmt(row.get("interaction_coef"), row.get("interaction_p"))
            n = int(row.get("retro_n", row.get("prosp_n", 0)))
            report.append(f"| {int(row['year'])} | {rr2} | {pr2} | {cr2} | {rc} | {pc} | {ic} | {n} |")

        # Interpretation
        has_both = retro_prosp_df.dropna(subset=["retro_r2", "prosp_r2"])
        if not has_both.empty:
            retro_wins = (has_both["retro_r2"] > has_both["prosp_r2"]).sum()
            prosp_wins = (has_both["prosp_r2"] > has_both["retro_r2"]).sum()
            report.append(f"\n**Finding**: Retrospective model wins in {retro_wins} of "
                          f"{len(has_both)} elections; prospective wins in {prosp_wins}.")
            avg_retro = has_both["retro_r2"].mean()
            avg_prosp = has_both["prosp_r2"].mean()
            if avg_retro > avg_prosp * 1.2:
                report.append("NZ voters are predominantly backward-looking, consistent with "
                              "the \"peasant\" model (Fiorina 1981).\n")
            elif avg_prosp > avg_retro * 1.2:
                report.append("NZ voters are predominantly forward-looking, consistent with "
                              "the \"banker's\" model (MacKuen et al. 1992).\n")
            else:
                report.append("Both perspectives contribute roughly equally — NZ voters "
                              "use both backward and forward-looking assessments.\n")

    report.append(f"\n![Retro vs Prospective](../graphs/evx_retro_vs_prospective.png)\n")

    # ── 1B: Myopia ──
    report.append("## 1B. Myopic vs Cumulative Retrospection (Achen & Bartels 2016)\n")
    report.append("**Question**: Do voters evaluate the whole government term or just the most recent economy?\n")

    if myopia_results:
        # Election-level results
        elec_df = myopia_results.get("election_level")
        if elec_df is not None and not elec_df.empty:
            report.append("### Election-Level Analysis\n")
            report.append("| Year | Incumbent | Inc. Vote | Early GDP | Late GDP | Full GDP |")
            report.append("|------|-----------|-----------|-----------|----------|----------|")
            for _, row in elec_df.iterrows():
                eg = f"{row.get('early_gdp', np.nan):.2f}" if pd.notna(row.get('early_gdp')) else "—"
                lg = f"{row.get('late_gdp', np.nan):.2f}" if pd.notna(row.get('late_gdp')) else "—"
                fg = f"{row.get('full_term_gdp', np.nan):.2f}" if pd.notna(row.get('full_term_gdp')) else "—"
                iv = f"{row.get('incumbent_vote', np.nan):.1f}%" if pd.notna(row.get('incumbent_vote')) else "—"
                report.append(f"| {int(row['year'])} | {row['incumbent']} | {iv} | {eg} | {lg} | {fg} |")

            r_late = myopia_results.get("r_late")
            r_early = myopia_results.get("r_early")
            r_full = myopia_results.get("r_full")
            if r_late is not None and r_early is not None:
                report.append(f"\n- Late-term GDP correlation: r={r_late:.3f}")
                report.append(f"- Early-term GDP correlation: r={r_early:.3f}")
                report.append(f"- Full-term GDP correlation: r={r_full:.3f}")
                if abs(r_late) > abs(r_early):
                    report.append("\n**Finding**: Late-term economy is a stronger predictor than early-term, "
                                  "supporting the myopia thesis (Achen & Bartels 2016).\n")
                else:
                    report.append("\n**Finding**: Early-term economy predicts as well or better than late-term, "
                                  "suggesting voters are more cumulative than myopic.\n")

        # Lag decay
        lag_decay = myopia_results.get("lag_decay")
        if lag_decay is not None:
            report.append("### Poll-Level Lag Decay\n")
            report.append("| Lag | r with Incumbent Support | p-value |")
            report.append("|-----|-------------------------|---------|")
            for _, row in lag_decay.iterrows():
                sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
                report.append(f"| {row['lag']} | {row['r']:.4f}{sig} | {row['p']:.4f} |")

    report.append(f"\n![Myopia](../graphs/evx_myopia.png)\n")

    # ── 1C: Cost of Ruling ──
    report.append("## 1C. Cost of Ruling (Paldam 1991)\n")
    report.append("**Question**: Is incumbent fatigue steeper for coalition governments? "
                  "Does it interact with economic conditions?\n")

    if cost_results:
        # Base model
        base = cost_results.get("base_model")
        if base:
            report.append("### Base Model: Incumbent Support ~ Tenure + Party\n")
            report.append(f"R² = {base.rsquared:.4f}\n")
            report.append("| Variable | β | SE | p |")
            report.append("|----------|---|----|----|")
            for var in base.params.index:
                sig = "***" if base.pvalues[var] < 0.001 else "**" if base.pvalues[var] < 0.01 else "*" if base.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {base.params[var]:.4f} | {base.bse[var]:.4f} | {base.pvalues[var]:.4f}{sig} |")

        # Coalition model
        coal = cost_results.get("coalition_model")
        if coal:
            report.append("\n### Coalition Interaction Model\n")
            report.append(f"R² = {coal.rsquared:.4f}\n")
            report.append("| Variable | β | SE | p |")
            report.append("|----------|---|----|----|")
            for var in coal.params.index:
                sig = "***" if coal.pvalues[var] < 0.001 else "**" if coal.pvalues[var] < 0.01 else "*" if coal.pvalues[var] < 0.05 else ""
                report.append(f"| {var} | {coal.params[var]:.4f} | {coal.bse[var]:.4f} | {coal.pvalues[var]:.4f}{sig} |")

            txc_p = coal.pvalues.get("tenure_x_coalition", 1)
            txc_b = coal.params.get("tenure_x_coalition", 0)
            if txc_p < 0.05:
                direction = "steeper" if txc_b < 0 else "shallower"
                report.append(f"\n**Finding**: Coalition governments experience significantly {direction} "
                              f"fatigue (interaction β={txc_b:.4f}, p={txc_p:.4f}).\n")
            else:
                report.append(f"\n**Finding**: No significant difference in fatigue rates between coalition "
                              f"and single-party governments (interaction p={txc_p:.4f}).\n")

        # Economic interaction
        econ = cost_results.get("econ_model")
        if econ:
            report.append("### Economic Interaction Model\n")
            report.append(f"R² = {econ.rsquared:.4f}\n")
            txg_p = econ.pvalues.get("tenure_x_gdp", 1)
            txg_b = econ.params.get("tenure_x_gdp", 0)
            report.append(f"- Tenure × GDP interaction: β={txg_b:.4f}, p={txg_p:.4f}")
            if txg_p < 0.05:
                report.append(f"- **Fatigue accelerates during bad economic times** (significant interaction).\n")
            else:
                report.append(f"- Fatigue rate does not significantly vary with economic conditions.\n")

        # Per-government rates
        gov_rates = cost_results.get("gov_rates")
        if gov_rates is not None and not gov_rates.empty:
            report.append("### Per-Government Fatigue Rates\n")
            report.append("| Government | Type | Fatigue (pp/yr) | r | p | n |")
            report.append("|------------|------|-----------------|---|---|---|")
            for _, row in gov_rates.iterrows():
                sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
                report.append(f"| {row['label']} | {row['type']} | {row['slope_per_year']:+.2f} | "
                              f"{row['r']:.3f}{sig} | {row['p']:.4f} | {int(row['n'])} |")

    report.append(f"\n![Cost of Ruling](../graphs/evx_cost_of_ruling.png)\n")

    # Write report
    report_text = "\n".join(report)
    report_path = REPORT_DIR / "economic_voting_extensions.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Economic Voting Extensions")
    print("=" * 70)

    polls_df = load_polls()
    prosp_df = extract_prospective_data()

    retro_prosp_results = analyze_retro_vs_prospective(prosp_df)
    myopia_results = analyze_myopia(polls_df)
    cost_results = analyze_cost_of_ruling(polls_df)

    generate_report(retro_prosp_results, myopia_results, cost_results)

    print("\n" + "=" * 70)
    print("Script 1 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
