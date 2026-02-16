#!/usr/bin/env python3
"""
Electoral Behavior Extensions

Tests two hypotheses from the political science literature:
3A. Dealignment (Dalton 2000; Ford & Jennings 2020)
3B. Strategic Voting under MMP (Cox 1997; Gschwend 2007)

Uses harmonized NZES data + fresh SQLite extraction for preferred party,
electorate vote, and split-ticket analysis.
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

INCUMBENTS = {
    1996: "National", 1999: "National", 2002: "Labour", 2005: "Labour",
    2008: "Labour", 2011: "National", 2014: "National", 2017: "National",
    2020: "Labour", 2023: "Labour",
}

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

# Most-liked party recode (different from party vote in 1999)
MOST_LIKED_RECODE = {
    1999: {1: "National", 2: "Labour", 3: "NZ First", 4: "Alliance", 5: "ACT",
           7: "Green", 8: "Christian", 10: "United"},
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

# Electorate vote recode (same as party vote for most years)
ELECTORATE_RECODE = {
    1999: {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT",
           7: "Green"},
    2011: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori"},
    2014: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "United Future", 7: "Maori", 11: "Conservative"},
    2017: {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
           6: "Maori", 8: "TOP"},
    2020: {0: "Nonvote", 1: "Labour", 2: "National", 3: "Green", 4: "NZ First",
           5: "ACT", 6: "Maori"},
    2023: {0: "Nonvote", 1: "Labour", 2: "National", 3: "Green", 4: "ACT",
           5: "NZ First", 6: "Maori"},
}

# Variables for strategic voting extraction
STRATEGIC_VARS = {
    1999: {"table": "nzes_1999_post", "most_liked": "PTYFIR",
           "party_vote": "PVOTE", "elec_vote": "EVOTE"},
    2011: {"table": "nzes_2011", "most_liked": "jmostlike",
           "party_vote": "jpartyvote", "elec_vote": "jelecvote"},
    2014: {"table": "nzes_2014", "most_liked": "dmostlike",
           "party_vote": "dpartyvote", "elec_vote": "delecvote"},
    2017: {"table": "nzes_2017", "most_liked": "rmostlike",
           "party_vote": "rpartyvote", "elec_vote": "relecvote"},
    2020: {"table": "nzes_2020", "most_liked": "B2",
           "party_vote": "lvpartyvote", "elec_vote": "lvelecvote"},
    2023: {"table": "nzes_2023", "most_liked": "B2",
           "party_vote": "mpartyvote", "elec_vote": "melecvote"},
}

MAJOR_PARTIES = {"National", "Labour"}
MINOR_PARTIES = {"Green", "NZ First", "ACT", "Maori", "United Future",
                  "Alliance", "TOP", "Conservative"}


# ─── Data Loading ───────────────────────────────────────────────────────────────

def load_nzes():
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    print(f"Loaded {len(df)} NZES respondents")
    return df


def extract_strategic_data():
    """Extract most-liked party, party vote, and electorate vote from SQLite."""
    print("\nExtracting strategic voting data from NZES...")
    conn = sqlite3.connect(str(NZES_DB))
    all_rows = []

    for year, spec in sorted(STRATEGIC_VARS.items()):
        table = spec["table"]
        ml_col = spec["most_liked"]
        pv_col = spec["party_vote"]
        ev_col = spec["elec_vote"]

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"  {year} ({table}): {len(df)} rows")

        ml_recode = MOST_LIKED_RECODE.get(year, {})
        pv_recode = PARTY_VOTE_RECODE.get(year, {})
        ev_recode = ELECTORATE_RECODE.get(year, pv_recode)

        for _, raw in df.iterrows():
            row = {"election_year": year}

            # Most-liked party
            if ml_col in raw.index:
                val = raw[ml_col]
                if pd.notna(val):
                    try:
                        v = int(float(val))
                    except (ValueError, OverflowError):
                        v = None
                    if v is not None and v in ml_recode:
                        row["most_liked"] = ml_recode[v]

            # Party vote
            if pv_col in raw.index:
                val = raw[pv_col]
                if pd.notna(val):
                    try:
                        v = int(float(val))
                    except (ValueError, OverflowError):
                        v = None
                    if v is not None and v in pv_recode:
                        row["party_vote"] = pv_recode[v]

            # Electorate vote
            if ev_col in raw.index:
                val = raw[ev_col]
                if pd.notna(val):
                    try:
                        v = int(float(val))
                    except (ValueError, OverflowError):
                        v = None
                    if v is not None and v in ev_recode:
                        row["elec_vote"] = ev_recode[v]

            all_rows.append(row)

    conn.close()
    result = pd.DataFrame(all_rows)
    print(f"  Total extracted: {len(result)} rows")
    return result


# ─── 3A. Dealignment ───────────────────────────────────────────────────────────

def analyze_dealignment(nzes_df):
    """Is the total explanatory power of demographics on vote choice declining?"""
    print("\n" + "=" * 70)
    print("3A. Dealignment (Dalton 2000; Ford & Jennings 2020)")
    print("=" * 70)

    major = nzes_df[nzes_df["party_vote"].isin(["National", "Labour"])].copy()
    major["vote_nat"] = (major["party_vote"] == "National").astype(float)

    # Ensure numeric types for all potential predictors
    for col in ["female", "education", "income_bracket", "ethnicity_maori"]:
        if col in major.columns:
            major[col] = pd.to_numeric(major[col], errors="coerce")

    years = sorted(major["election_year"].unique())
    full_results = []
    individual_results = []

    for year in years:
        yr = major[major["election_year"] == year].copy()

        # Standardize age within year
        if "age" in yr.columns and yr["age"].notna().sum() > 30:
            age_mean = yr["age"].mean()
            age_std = yr["age"].std()
            if age_std > 0:
                yr["age_z"] = (yr["age"] - age_mean) / age_std
            else:
                yr["age_z"] = 0.0

        # Full model with all available predictors
        predictors = []
        for p in ["age_z", "female", "education", "income_bracket", "ethnicity_maori"]:
            if p in yr.columns and yr[p].notna().sum() > 30:
                predictors.append(p)

        if not predictors:
            continue

        valid = yr[["vote_nat"] + predictors].dropna()
        if len(valid) < 50:
            continue

        try:
            X = sm.add_constant(valid[predictors])
            model = Logit(valid["vote_nat"], X).fit(disp=0, maxiter=100)
            full_results.append({
                "year": year,
                "n": len(valid),
                "pseudo_r2": model.prsquared,
                "predictors": ", ".join(predictors),
                "n_predictors": len(predictors),
            })
            print(f"  {year}: Full model R²={model.prsquared:.4f} "
                  f"(n={len(valid)}, predictors: {', '.join(predictors)})")
        except Exception as e:
            print(f"  {year}: Full model error: {e}")

        # Individual predictor R² (decomposition)
        for pred in predictors:
            pred_valid = yr[["vote_nat", pred]].dropna()
            if len(pred_valid) < 50:
                continue
            try:
                X = sm.add_constant(pred_valid[[pred]])
                model = Logit(pred_valid["vote_nat"], X).fit(disp=0, maxiter=100)
                individual_results.append({
                    "year": year,
                    "predictor": pred,
                    "pseudo_r2": model.prsquared,
                    "coef": model.params.get(pred, np.nan),
                    "p": model.pvalues.get(pred, np.nan),
                    "n": len(pred_valid),
                })
            except Exception:
                pass

    full_df = pd.DataFrame(full_results)
    indiv_df = pd.DataFrame(individual_results)

    # Test: is R² trending downward?
    results = {"full": full_df, "individual": indiv_df}
    if len(full_df) >= 4:
        r, p = stats.pearsonr(full_df["year"], full_df["pseudo_r2"])
        results["trend_r"] = r
        results["trend_p"] = p
        print(f"\n  Trend in full model R²: r={r:.3f}, p={p:.4f}")
        if r < 0:
            print("  → Demographics becoming LESS predictive (dealignment)")
        else:
            print("  → Demographics becoming MORE predictive (no dealignment)")

    # Replacement test: add ideology and leader evals where available
    print("\n  Testing replacement hypothesis (ideology + leaders):")
    replace_results = []
    for year in years:
        yr = major[major["election_year"] == year].copy()

        # Standardize age
        if "age" in yr.columns and yr["age"].notna().sum() > 30:
            age_mean = yr["age"].mean()
            age_std = yr["age"].std()
            if age_std > 0:
                yr["age_z"] = (yr["age"] - age_mean) / age_std

        demo_preds = [p for p in ["age_z", "female", "education"]
                      if p in yr.columns and yr[p].notna().sum() > 30]
        all_preds = list(demo_preds)

        # Add ideology if available
        if "lr_self" in yr.columns and yr["lr_self"].notna().sum() > 30:
            all_preds.append("lr_self")

        if not demo_preds or len(all_preds) == len(demo_preds):
            continue

        valid = yr[["vote_nat"] + all_preds].dropna()
        if len(valid) < 50:
            continue

        try:
            # Demographics only
            X_demo = sm.add_constant(valid[demo_preds])
            m_demo = Logit(valid["vote_nat"], X_demo).fit(disp=0, maxiter=100)

            # Demographics + ideology
            X_all = sm.add_constant(valid[all_preds])
            m_all = Logit(valid["vote_nat"], X_all).fit(disp=0, maxiter=100)

            replace_results.append({
                "year": year,
                "demo_r2": m_demo.prsquared,
                "full_r2": m_all.prsquared,
                "added_r2": m_all.prsquared - m_demo.prsquared,
                "n": len(valid),
            })
            print(f"    {year}: Demo R²={m_demo.prsquared:.4f}, "
                  f"+Ideology R²={m_all.prsquared:.4f}, "
                  f"added={m_all.prsquared - m_demo.prsquared:.4f}")
        except Exception:
            pass

    results["replacement"] = pd.DataFrame(replace_results)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Full model R² over time
    ax = axes[0]
    if not full_df.empty:
        ax.plot(full_df["year"], full_df["pseudo_r2"], "ko-", markersize=6)
        if len(full_df) >= 3:
            z = np.polyfit(full_df["year"], full_df["pseudo_r2"], 1)
            x_range = np.linspace(full_df["year"].min(), full_df["year"].max(), 50)
            ax.plot(x_range, np.poly1d(z)(x_range), "--", color="gray", alpha=0.5)
        ax.set_ylabel("McFadden Pseudo-R²")
        ax.set_xlabel("Election Year")
        ax.set_title("Total Demographic Explanatory Power")
        ax.grid(True, alpha=0.3)

    # Panel 2: Individual predictor R² decomposition
    ax = axes[1]
    if not indiv_df.empty:
        pred_colors = {"age_z": "#D82A20", "female": "#ff7f0e",
                       "education": "#098137", "income_bracket": "#00529F",
                       "ethnicity_maori": "#B2001A"}
        for pred in indiv_df["predictor"].unique():
            sub = indiv_df[indiv_df["predictor"] == pred].sort_values("year")
            ax.plot(sub["year"], sub["pseudo_r2"], "o-",
                    color=pred_colors.get(pred, "#333"),
                    label=pred.replace("_", " ").title(), markersize=5)
        ax.set_ylabel("Individual Pseudo-R²")
        ax.set_xlabel("Election Year")
        ax.set_title("Predictor Decomposition")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Panel 3: Replacement test
    ax = axes[2]
    if replace_results:
        rep_df = pd.DataFrame(replace_results)
        x = np.arange(len(rep_df))
        w = 0.3
        ax.bar(x - w/2, rep_df["demo_r2"], w, label="Demographics only",
               color="#00529F", alpha=0.7)
        ax.bar(x + w/2, rep_df["full_r2"], w, label="+ Ideology",
               color="#098137", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(y)) for y in rep_df["year"]], rotation=45)
        ax.set_ylabel("Pseudo-R²")
        ax.set_title("Replacement Hypothesis\n(Does ideology fill the gap?)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Dealignment in NZ Elections (Dalton 2000)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "eb_dealignment.png", dpi=150)
    plt.close()

    return results


# ─── 3B. Strategic Voting under MMP ────────────────────────────────────────────

def analyze_strategic_voting(strat_df):
    """Test for strategic desertion and split-ticket voting."""
    print("\n" + "=" * 70)
    print("3B. Strategic Voting under MMP (Cox 1997; Gschwend 2007)")
    print("=" * 70)

    results = {}

    # ─── Part i: Strategic Desertion ───
    print("\n  Part i: Strategic Desertion")
    print("  " + "-" * 50)

    desertion_by_year = []
    desertion_by_party = []

    years = sorted(strat_df["election_year"].unique())
    for year in years:
        yr = strat_df[strat_df["election_year"] == year].copy()
        valid = yr[yr["most_liked"].notna() & yr["party_vote"].notna()].copy()
        if len(valid) < 50:
            continue

        valid["strategic_desertion"] = (valid["most_liked"] != valid["party_vote"]).astype(int)
        overall_rate = valid["strategic_desertion"].mean() * 100

        # Exclude nonvoters
        valid_voters = valid[valid["party_vote"] != "Nonvote"]
        if len(valid_voters) < 50:
            continue
        valid_voters["strategic_desertion"] = (
            valid_voters["most_liked"] != valid_voters["party_vote"]
        ).astype(int)
        voter_rate = valid_voters["strategic_desertion"].mean() * 100

        desertion_by_year.append({
            "year": year, "n": len(valid_voters),
            "desertion_rate": voter_rate,
        })
        print(f"    {year}: Desertion rate = {voter_rate:.1f}% (n={len(valid_voters)})")

        # Per-party desertion rates
        for party in valid_voters["most_liked"].unique():
            party_fans = valid_voters[valid_voters["most_liked"] == party]
            if len(party_fans) < 10:
                continue
            deserted = (party_fans["party_vote"] != party).mean() * 100

            # Where did they go?
            loyal = party_fans[party_fans["party_vote"] == party]
            deserters = party_fans[party_fans["party_vote"] != party]
            if len(deserters) > 0:
                dest = deserters["party_vote"].value_counts()
                top_dest = dest.index[0] if len(dest) > 0 else "—"
            else:
                top_dest = "—"

            is_minor = party in MINOR_PARTIES
            desertion_by_party.append({
                "year": year, "party": party,
                "n_prefer": len(party_fans),
                "desertion_pct": deserted,
                "top_destination": top_dest,
                "is_minor": is_minor,
            })

    results["desertion_by_year"] = pd.DataFrame(desertion_by_year)
    results["desertion_by_party"] = pd.DataFrame(desertion_by_party)

    # Summary: minor vs major party desertion
    dbp = pd.DataFrame(desertion_by_party)
    if not dbp.empty:
        minor = dbp[dbp["is_minor"] == True]
        major = dbp[dbp["is_minor"] == False]
        if not minor.empty and not major.empty:
            print(f"\n    Mean desertion (minor parties): {minor['desertion_pct'].mean():.1f}%")
            print(f"    Mean desertion (major parties): {major['desertion_pct'].mean():.1f}%")

    # ─── Part ii: Split-Ticket Voting ───
    print("\n  Part ii: Split-Ticket Voting")
    print("  " + "-" * 50)

    split_by_year = []
    split_combos_all = []

    for year in years:
        yr = strat_df[strat_df["election_year"] == year].copy()
        valid = yr[yr["party_vote"].notna() & yr["elec_vote"].notna()].copy()
        # Exclude nonvoters
        valid = valid[(valid["party_vote"] != "Nonvote") & (valid["elec_vote"] != "Nonvote")]
        if len(valid) < 50:
            continue

        valid["split_ticket"] = (valid["party_vote"] != valid["elec_vote"]).astype(int)
        split_rate = valid["split_ticket"].mean() * 100

        split_by_year.append({
            "year": year, "n": len(valid),
            "split_rate": split_rate,
        })
        print(f"    {year}: Split-ticket rate = {split_rate:.1f}% (n={len(valid)})")

        # Most common split-ticket combinations
        splits = valid[valid["split_ticket"] == 1]
        if len(splits) > 10:
            combos = splits.groupby(["elec_vote", "party_vote"]).size().reset_index(name="count")
            combos = combos.sort_values("count", ascending=False).head(5)
            for _, combo in combos.iterrows():
                split_combos_all.append({
                    "year": year,
                    "electorate": combo["elec_vote"],
                    "party": combo["party_vote"],
                    "count": combo["count"],
                    "pct_of_splits": combo["count"] / len(splits) * 100,
                })

    results["split_by_year"] = pd.DataFrame(split_by_year)
    results["split_combos"] = pd.DataFrame(split_combos_all)

    # Test: is split-ticket rate increasing over time?
    sby = pd.DataFrame(split_by_year)
    if len(sby) >= 4:
        r, p = stats.pearsonr(sby["year"], sby["split_rate"])
        results["split_trend_r"] = r
        results["split_trend_p"] = p
        print(f"\n    Split-ticket trend: r={r:.3f}, p={p:.4f}")
        if r > 0:
            print("    → Split-ticket voting is INCREASING (voters learning MMP)")
        else:
            print("    → No evidence of increasing split-ticket voting")

    # ─── Part iii: Profile of Strategic Voters ───
    print("\n  Part iii: Profile of Strategic Voters")
    print("  " + "-" * 50)

    # Merge with harmonized NZES for demographics
    nzes = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    profile_results = []

    for year in years:
        yr_strat = strat_df[strat_df["election_year"] == year].copy()
        yr_nzes = nzes[nzes["election_year"] == year].copy()

        # Match by row position (same survey)
        if len(yr_strat) != len(yr_nzes):
            continue

        yr_strat = yr_strat.reset_index(drop=True)
        yr_nzes = yr_nzes.reset_index(drop=True)

        # Add demographics
        for col in ["age", "female", "education", "lr_self"]:
            if col in yr_nzes.columns:
                yr_strat[col] = yr_nzes[col]

        valid = yr_strat[yr_strat["most_liked"].notna() &
                         yr_strat["party_vote"].notna() &
                         (yr_strat["party_vote"] != "Nonvote")].copy()
        valid["strategic"] = (valid["most_liked"] != valid["party_vote"]).astype(int)

        if len(valid) < 50 or valid["strategic"].sum() < 10:
            continue

        loyalists = valid[valid["strategic"] == 0]
        strategists = valid[valid["strategic"] == 1]

        row = {"year": year, "n_loyal": len(loyalists), "n_strategic": len(strategists)}

        for col in ["age", "education", "lr_self"]:
            if col in valid.columns:
                loyal_vals = loyalists[col].dropna()
                strat_vals = strategists[col].dropna()
                if len(loyal_vals) > 10 and len(strat_vals) > 10:
                    t, p = stats.ttest_ind(strat_vals, loyal_vals)
                    row[f"{col}_loyal"] = loyal_vals.mean()
                    row[f"{col}_strategic"] = strat_vals.mean()
                    row[f"{col}_t"] = t
                    row[f"{col}_p"] = p

        # Ideological extremity
        if "lr_self" in valid.columns:
            valid["lr_extreme"] = abs(valid["lr_self"] - 5)
            loyal_ext = loyalists["lr_self"].dropna().apply(lambda x: abs(x - 5))
            strat_ext = strategists["lr_self"].dropna().apply(lambda x: abs(x - 5))
            if len(loyal_ext) > 10 and len(strat_ext) > 10:
                t, p = stats.ttest_ind(strat_ext, loyal_ext)
                row["extremity_loyal"] = loyal_ext.mean()
                row["extremity_strategic"] = strat_ext.mean()
                row["extremity_t"] = t
                row["extremity_p"] = p

        profile_results.append(row)

    results["profiles"] = pd.DataFrame(profile_results)

    if profile_results:
        print("    Demographics: Strategic vs Loyal voters")
        for _, row in pd.DataFrame(profile_results).iterrows():
            print(f"    {int(row['year'])}:")
            for col in ["age", "education", "lr_self"]:
                if f"{col}_loyal" in row and pd.notna(row.get(f"{col}_loyal")):
                    sig = "*" if row.get(f"{col}_p", 1) < 0.05 else ""
                    print(f"      {col}: loyal={row[f'{col}_loyal']:.2f}, "
                          f"strategic={row[f'{col}_strategic']:.2f} {sig}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Desertion rate over time
    ax = axes[0, 0]
    dby = pd.DataFrame(desertion_by_year)
    if not dby.empty:
        ax.plot(dby["year"], dby["desertion_rate"], "ko-", markersize=6)
        ax.set_ylabel("Desertion Rate (%)")
        ax.set_xlabel("Election Year")
        ax.set_title("Strategic Desertion Rate Over Time\n(Preferred party ≠ actual vote)")
        ax.grid(True, alpha=0.3)

    # Panel 2: Desertion by party
    ax = axes[0, 1]
    if not dbp.empty:
        party_colors = {"National": "#00529F", "Labour": "#D82A20", "Green": "#098137",
                        "ACT": "#FDE401", "NZ First": "#000000", "Maori": "#B2001A",
                        "Alliance": "#FF6B6B", "TOP": "#999999", "Conservative": "#663399",
                        "United Future": "#800080"}
        for party in dbp["party"].unique():
            sub = dbp[dbp["party"] == party].sort_values("year")
            if len(sub) >= 2:
                ax.plot(sub["year"], sub["desertion_pct"], "o-",
                        color=party_colors.get(party, "#666"),
                        label=party, markersize=5, alpha=0.8)
        ax.set_ylabel("Desertion Rate (%)")
        ax.set_xlabel("Election Year")
        ax.set_title("Desertion Rate by Preferred Party")
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    # Panel 3: Split-ticket rate over time
    ax = axes[1, 0]
    if not sby.empty:
        ax.plot(sby["year"], sby["split_rate"], "ko-", markersize=6)
        if len(sby) >= 3:
            z = np.polyfit(sby["year"], sby["split_rate"], 1)
            x_range = np.linspace(sby["year"].min(), sby["year"].max(), 50)
            ax.plot(x_range, np.poly1d(z)(x_range), "--", color="gray", alpha=0.5)
        ax.set_ylabel("Split-Ticket Rate (%)")
        ax.set_xlabel("Election Year")
        ax.set_title("Split-Ticket Voting Over Time\n(Party vote ≠ electorate vote)")
        ax.grid(True, alpha=0.3)

    # Panel 4: Top split-ticket combinations (latest year)
    ax = axes[1, 1]
    combos_df = pd.DataFrame(split_combos_all)
    if not combos_df.empty:
        latest_year = combos_df["year"].max()
        latest = combos_df[combos_df["year"] == latest_year].sort_values("count", ascending=True)
        if not latest.empty:
            labels = [f"Elec: {r['electorate']}\nParty: {r['party']}"
                      for _, r in latest.iterrows()]
            ax.barh(range(len(latest)), latest["count"], color="#00529F", alpha=0.7)
            ax.set_yticks(range(len(latest)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Count")
            ax.set_title(f"Top Split-Ticket Combos ({int(latest_year)})")
            ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Strategic Voting Under MMP in NZ (Cox 1997; Gschwend 2007)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "eb_strategic_voting.png", dpi=150)
    plt.close()

    return results


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(dealign_results, strategic_results):
    """Generate markdown report."""
    report = []
    report.append("# Electoral Behavior Extensions\n")
    report.append("*Two hypotheses about dealignment and strategic voting in NZ*\n")

    # ── 3A: Dealignment ──
    report.append("## 3A. Dealignment (Dalton 2000; Ford & Jennings 2020)\n")
    report.append("**Question**: Is the total explanatory power of demographics "
                  "on vote choice declining?\n")

    if dealign_results:
        full_df = dealign_results.get("full")
        if full_df is not None and not full_df.empty:
            report.append("### Full Demographic Model R² by Election\n")
            report.append("| Year | Pseudo-R² | n | Predictors |")
            report.append("|------|-----------|---|-----------|")
            for _, row in full_df.iterrows():
                report.append(f"| {int(row['year'])} | {row['pseudo_r2']:.4f} | "
                              f"{int(row['n'])} | {row['predictors']} |")

            trend_r = dealign_results.get("trend_r")
            trend_p = dealign_results.get("trend_p")
            if trend_r is not None:
                report.append(f"\nTrend in R² over time: r={trend_r:.3f}, p={trend_p:.4f}")
                if trend_r < -0.3:
                    report.append("\n**Finding**: Demographics are becoming less predictive of vote choice, "
                                  "consistent with the dealignment thesis (Dalton 2000).\n")
                elif trend_r > 0.3:
                    report.append("\n**Finding**: Demographics are becoming MORE predictive, suggesting "
                                  "realignment rather than dealignment in NZ.\n")
                else:
                    report.append("\n**Finding**: No clear trend — demographic predictive power is "
                                  "relatively stable across elections.\n")

        # Individual decomposition
        indiv = dealign_results.get("individual")
        if indiv is not None and not indiv.empty:
            report.append("### Individual Predictor R² Decomposition\n")
            pivot = indiv.pivot(index="year", columns="predictor", values="pseudo_r2")
            report.append("| Year | " + " | ".join(pivot.columns) + " |")
            report.append("|------" + "|-----" * len(pivot.columns) + "|")
            for year, row in pivot.iterrows():
                vals = " | ".join([f"{v:.4f}" if pd.notna(v) else "—" for v in row])
                report.append(f"| {int(year)} | {vals} |")

        # Replacement hypothesis
        replacement = dealign_results.get("replacement")
        if replacement is not None and not replacement.empty:
            report.append("\n### Replacement Hypothesis: Demographics + Ideology\n")
            report.append("| Year | Demo-only R² | + Ideology R² | Added R² |")
            report.append("|------|-------------|--------------|----------|")
            for _, row in replacement.iterrows():
                report.append(f"| {int(row['year'])} | {row['demo_r2']:.4f} | "
                              f"{row['full_r2']:.4f} | {row['added_r2']:.4f} |")

    report.append(f"\n![Dealignment](../graphs/eb_dealignment.png)\n")

    # ── 3B: Strategic Voting ──
    report.append("## 3B. Strategic Voting Under MMP (Cox 1997; Gschwend 2007)\n")
    report.append("**Question**: Do voters strategically desert minor parties? "
                  "Do they split tickets?\n")

    if strategic_results:
        # Desertion over time
        dby = strategic_results.get("desertion_by_year")
        if dby is not None and not dby.empty:
            report.append("### Strategic Desertion (Preferred ≠ Actual Vote)\n")
            report.append("| Year | n | Desertion Rate |")
            report.append("|------|---|---------------|")
            for _, row in dby.iterrows():
                report.append(f"| {int(row['year'])} | {int(row['n'])} | {row['desertion_rate']:.1f}% |")

        # By party
        dbp = strategic_results.get("desertion_by_party")
        if dbp is not None and not dbp.empty:
            report.append("\n### Desertion by Preferred Party\n")
            report.append("| Year | Party | n | Desertion % | Top Destination |")
            report.append("|------|-------|---|------------|----------------|")
            # Show only parties with n >= 20
            for _, row in dbp[dbp["n_prefer"] >= 20].iterrows():
                report.append(f"| {int(row['year'])} | {row['party']} | "
                              f"{int(row['n_prefer'])} | {row['desertion_pct']:.1f}% | "
                              f"{row['top_destination']} |")

            # Minor vs major summary
            minor = dbp[dbp["is_minor"] == True]
            major_p = dbp[dbp["is_minor"] == False]
            if not minor.empty and not major_p.empty:
                report.append(f"\nMean desertion rate — Minor parties: "
                              f"**{minor['desertion_pct'].mean():.1f}%**, "
                              f"Major parties: **{major_p['desertion_pct'].mean():.1f}%**")
                if minor["desertion_pct"].mean() > major_p["desertion_pct"].mean():
                    report.append("\n**Finding**: Minor party supporters desert at higher rates, "
                                  "consistent with strategic voting theory (Cox 1997).\n")

        # Split-ticket voting
        sby = strategic_results.get("split_by_year")
        if sby is not None and not sby.empty:
            report.append("\n### Split-Ticket Voting (Party Vote ≠ Electorate Vote)\n")
            report.append("| Year | n | Split-Ticket Rate |")
            report.append("|------|---|------------------|")
            for _, row in sby.iterrows():
                report.append(f"| {int(row['year'])} | {int(row['n'])} | {row['split_rate']:.1f}% |")

            trend_r = strategic_results.get("split_trend_r")
            trend_p = strategic_results.get("split_trend_p")
            if trend_r is not None:
                report.append(f"\nTrend: r={trend_r:.3f}, p={trend_p:.4f}")
                if trend_r > 0.3:
                    report.append("\n**Finding**: Split-ticket voting is increasing over time, "
                                  "suggesting voters are learning to use MMP strategically.\n")
                else:
                    report.append("\n**Finding**: No clear increase in split-ticket voting — "
                                  "MMP learning may have plateaued.\n")

        # Top combos
        combos = strategic_results.get("split_combos")
        if combos is not None and not combos.empty:
            report.append("\n### Most Common Split-Ticket Combinations\n")
            report.append("| Year | Electorate Vote | Party Vote | Count |")
            report.append("|------|----------------|------------|-------|")
            # Top 3 per year
            for year in sorted(combos["year"].unique()):
                yr_combos = combos[combos["year"] == year].nlargest(3, "count")
                for _, row in yr_combos.iterrows():
                    report.append(f"| {int(row['year'])} | {row['electorate']} | "
                                  f"{row['party']} | {int(row['count'])} |")

        # Voter profiles
        profiles = strategic_results.get("profiles")
        if profiles is not None and not profiles.empty:
            report.append("\n### Profile of Strategic Voters\n")
            report.append("| Year | n Loyal | n Strategic | Age (L/S) | Education (L/S) | L-R (L/S) |")
            report.append("|------|---------|-------------|-----------|-----------------|-----------|")
            for _, row in profiles.iterrows():
                age_s = f"{row.get('age_loyal', np.nan):.1f}/{row.get('age_strategic', np.nan):.1f}" if pd.notna(row.get("age_loyal")) else "—"
                edu_s = f"{row.get('education_loyal', np.nan):.1f}/{row.get('education_strategic', np.nan):.1f}" if pd.notna(row.get("education_loyal")) else "—"
                lr_s = f"{row.get('lr_self_loyal', np.nan):.1f}/{row.get('lr_self_strategic', np.nan):.1f}" if pd.notna(row.get("lr_self_loyal")) else "—"
                report.append(f"| {int(row['year'])} | {int(row['n_loyal'])} | "
                              f"{int(row['n_strategic'])} | {age_s} | {edu_s} | {lr_s} |")

    report.append(f"\n![Strategic Voting](../graphs/eb_strategic_voting.png)\n")

    # Write report
    report_text = "\n".join(report)
    report_path = REPORT_DIR / "electoral_behavior_extensions.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Electoral Behavior Extensions")
    print("=" * 70)

    nzes_df = load_nzes()
    strat_df = extract_strategic_data()

    dealign_results = analyze_dealignment(nzes_df)
    strategic_results = analyze_strategic_voting(strat_df)

    generate_report(dealign_results, strategic_results)

    print("\n" + "=" * 70)
    print("Script 3 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
