#!/usr/bin/env python3
"""
Issue & Valence Politics

Tests three hypotheses from the political science literature:
2A. Thermostatic Policy Responsiveness (Wlezien 1995)
2B. Issue Ownership (Petrocik 1996; Egan 2013)
2C. Valence Politics (Stokes 1963; Clarke et al. 2004)

Heavy NZES extraction for leader thermometers, issue coding, competence ratings.
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

# Government policy direction for thermostatic test
GOVT_DIRECTION = [
    (1990, 1999, "National", "right"),
    (1999, 2008, "Labour", "left"),
    (2008, 2017, "National", "right"),
    (2017, 2023, "Labour", "left"),
    (2023, 2026, "National", "right"),
]

# Party vote recodes from deep_data.py
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

# Leader thermometer variable mapping
LEADER_VARS = {
    1996: {"table": "nzes_1996", "pm_like": "QBOLGER", "opp_like": "QCLARK",
           "party_vote": "VOT96P", "pm_party": "National", "opp_party": "Labour",
           "dk_code": 99, "reverse": False},
    1999: {"table": "nzes_1999_post", "pm_like": "SHIPLEY", "opp_like": "SCLARK",
           "party_vote": "PVOTE", "pm_party": "National", "opp_party": "Labour",
           "dk_code": 99, "reverse": True},  # 0=like, 10=dislike in 1999
    2005: {"table": "nzes_2005", "pm_like": "yclark", "opp_like": "ybrash",
           "party_vote": "yvot05p", "pm_party": "Labour", "opp_party": "National",
           "dk_code": 99, "reverse": False},
    2008: {"table": "nzes_2008", "pm_like": "zclklik", "opp_like": "zkeylik",
           "party_vote": "zvt08p", "pm_party": "Labour", "opp_party": "National",
           "dk_code": 99, "reverse": False},
    2011: {"table": "nzes_2011", "pm_like": "jkeylike", "opp_like": "jgofflike",
           "party_vote": "jpartyvote", "pm_party": "National", "opp_party": "Labour",
           "dk_code": 99, "reverse": False},
    2014: {"table": "nzes_2014", "pm_like": "dkeylike", "opp_like": "dcunliffelike",
           "party_vote": "dpartyvote", "pm_party": "National", "opp_party": "Labour",
           "pm_comp": "dkeycomp", "opp_comp": "dcuncomp",
           "dk_code": 99, "reverse": False},
    2017: {"table": "nzes_2017", "pm_like": "renglishlike", "opp_like": "rardernlike",
           "party_vote": "rpartyvote", "pm_party": "National", "opp_party": "Labour",
           "pm_comp": "renglishcomp", "opp_comp": "rarderncomp",
           "dk_code": 99, "reverse": False},
    2020: {"table": "nzes_2020", "pm_like": "B4_2", "opp_like": "B4_1",
           "party_vote": "E3", "pm_party": "Labour", "opp_party": "National",
           "dk_code": 11, "reverse": False},
    2023: {"table": "nzes_2023", "pm_like": "B3b", "opp_like": "B3a",
           "party_vote": "mpartyvote", "pm_party": "Labour", "opp_party": "National",
           "pm_comp": "B7b", "opp_comp": "B7a",
           "dk_code": 99, "reverse": False},
}

# Issue variable mapping
ISSUE_VARS = {
    1999: {"table": "nzes_1999_post", "issue_col": "CISSUE", "party_vote": "PVOTE",
           "type": "coded"},
    2002: {"table": "nzes_2002", "issue_col": "WISSUE", "party_vote": "WVOT02P",
           "type": "coded"},
    2005: {"table": "nzes_2005", "issue_col": "YISSUE", "party_vote": "yvot05p",
           "type": "coded"},
    2014: {"table": "nzes_2014", "issue_col": "dissuegen", "party_vote": "dpartyvote",
           "type": "coded"},
    2017: {"table": "nzes_2017", "issue_col": "rimpissue", "party_vote": "rpartyvote",
           "type": "coded"},
    2020: {"table": "nzes_2020", "issue_col": "lissue1", "party_vote": "E3",
           "type": "coded"},
    2023: {"table": "nzes_2023", "issue_col": "mllcat", "party_vote": "mpartyvote",
           "type": "coded"},
}

# Issue ownership: map raw codes to broad categories, then assign ownership
# Each year has its own coding scheme
ISSUE_CODE_MAP = {
    1999: {
        1: "economy", 2: "education", 3: "health", 4: "tax",
        5: "crime", 6: "housing", 7: "environment", 8: "immigration",
        9: "poverty", 10: "employment", 11: "defence",
        12: "transport", 13: "welfare", 14: "treaty",
    },
    2002: {
        1: "health", 2: "education", 3: "economy", 4: "law and order",
        5: "environment", 6: "tax", 7: "immigration", 8: "employment",
        9: "welfare", 10: "housing", 11: "defence", 12: "treaty",
        13: "transport",
    },
    2005: {
        1: "health", 2: "education", 3: "economy", 4: "law and order",
        5: "environment", 6: "tax", 7: "immigration", 8: "employment",
        9: "welfare", 10: "housing", 11: "defence", 12: "treaty",
        13: "transport",
    },
    2014: {
        1: "economy", 2: "inequality", 3: "cost of living", 4: "health",
        5: "education", 6: "tax", 7: "law and order", 8: "environment",
        9: "employment", 10: "foreign ownership", 11: "housing",
        12: "poverty", 13: "welfare", 14: "immigration", 15: "treaty",
    },
    2017: {
        1: "economy", 2: "housing", 3: "health", 4: "education",
        5: "poverty", 6: "environment", 7: "immigration", 8: "law and order",
        9: "tax", 10: "inequality", 11: "employment",
    },
    2020: {
        1: "economy", 2: "housing", 3: "health", 4: "education",
        5: "poverty", 6: "environment", 7: "immigration", 8: "law and order",
        9: "tax", 10: "inequality", 11: "employment",
        20: "covid", 21: "covid",
    },
    2023: {
        1: "cost of living", 2: "economy", 3: "education", 4: "environment",
        5: "health", 6: "housing", 7: "identity", 8: "immigration",
        9: "law and order", 10: "treaty", 11: "moral", 14: "poverty",
        15: "tax",
    },
}

NATIONAL_ISSUES = {"economy", "tax", "crime", "law and order", "defence",
                    "immigration", "cost of living"}
LABOUR_ISSUES = {"health", "education", "housing", "poverty", "inequality",
                  "environment", "employment", "welfare"}


# ─── Data Loading ───────────────────────────────────────────────────────────────

def load_nzes():
    df = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")
    print(f"Loaded {len(df)} NZES respondents")
    return df


# ─── 2A. Thermostatic Policy Responsiveness ─────────────────────────────────────

def analyze_thermostatic(nzes_df):
    """Test: when government policy moves left, does the public shift right?"""
    print("\n" + "=" * 70)
    print("2A. Thermostatic Policy Responsiveness (Wlezien 1995)")
    print("=" * 70)

    # Mean L-R self-placement per election year
    years = sorted(nzes_df[nzes_df["lr_self"].notna()]["election_year"].unique())
    lr_data = []
    for year in years:
        yr = nzes_df[(nzes_df["election_year"] == year) & nzes_df["lr_self"].notna()]
        if len(yr) < 50:
            continue

        # Find government direction
        gov_dir = None
        for start, end, party, direction in GOVT_DIRECTION:
            if start <= year <= end:
                gov_dir = direction
                break

        lr_data.append({
            "year": year,
            "mean_lr": yr["lr_self"].mean(),
            "median_lr": yr["lr_self"].median(),
            "sd_lr": yr["lr_self"].std(),
            "n": len(yr),
            "gov_direction": gov_dir,
        })
        print(f"  {year}: mean L-R = {yr['lr_self'].mean():.2f} (n={len(yr)}), "
              f"gov direction = {gov_dir}")

    lr_df = pd.DataFrame(lr_data)

    # Test: do shifts go in the opposite direction to government?
    results = {"lr_data": lr_df}
    if len(lr_df) >= 4:
        lr_df["lr_shift"] = lr_df["mean_lr"].diff()
        lr_df["expected_shift"] = lr_df["gov_direction"].map(
            {"right": "left", "left": "right"})

        # Under right government, public should shift left (negative shift)
        # Under left government, public should shift right (positive shift)
        lr_df["thermostatic_correct"] = False
        for i in range(1, len(lr_df)):
            shift = lr_df.iloc[i]["lr_shift"]
            gov = lr_df.iloc[i]["gov_direction"]
            if gov == "right" and shift < 0:
                lr_df.iloc[i, lr_df.columns.get_loc("thermostatic_correct")] = True
            elif gov == "left" and shift > 0:
                lr_df.iloc[i, lr_df.columns.get_loc("thermostatic_correct")] = True

        correct_count = lr_df["thermostatic_correct"].sum()
        total_shifts = lr_df["lr_shift"].notna().sum()
        print(f"\n  Thermostatic correct: {correct_count} / {total_shifts} shifts")
        results["correct_count"] = correct_count
        results["total_shifts"] = total_shifts

    # Within-election test: governing party voters vs opposition
    print("\n  Within-election L-R by vote choice:")
    within_results = []
    for year in years:
        yr = nzes_df[(nzes_df["election_year"] == year) &
                     nzes_df["lr_self"].notna() &
                     nzes_df["party_vote"].notna()]
        inc = INCUMBENTS.get(year)
        if not inc or len(yr) < 50:
            continue

        inc_voters = yr[yr["party_vote"] == inc]["lr_self"]
        opp_party = "Labour" if inc == "National" else "National"
        opp_voters = yr[yr["party_vote"] == opp_party]["lr_self"]

        if len(inc_voters) > 20 and len(opp_voters) > 20:
            within_results.append({
                "year": year,
                "inc_mean_lr": inc_voters.mean(),
                "opp_mean_lr": opp_voters.mean(),
                "gap": inc_voters.mean() - opp_voters.mean(),
                "n_inc": len(inc_voters),
                "n_opp": len(opp_voters),
            })
            print(f"    {year}: Inc voters L-R={inc_voters.mean():.2f}, "
                  f"Opp voters L-R={opp_voters.mean():.2f}")

    results["within_election"] = pd.DataFrame(within_results)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Mean L-R over time with government shading
    ax = axes[0]
    if not lr_df.empty:
        ax.plot(lr_df["year"], lr_df["mean_lr"], "ko-", markersize=6)
        ax.fill_between(lr_df["year"],
                        lr_df["mean_lr"] - lr_df["sd_lr"] / np.sqrt(lr_df["n"]),
                        lr_df["mean_lr"] + lr_df["sd_lr"] / np.sqrt(lr_df["n"]),
                        alpha=0.2, color="gray")
        # Government direction shading
        for start, end, party, direction in GOVT_DIRECTION:
            color = "#D82A20" if direction == "left" else "#00529F"
            ax.axvspan(start, min(end, 2024), alpha=0.08, color=color)
        ax.axhline(5, color="gray", linewidth=0.5, linestyle="--", label="Midpoint")
        ax.set_ylabel("Mean Left-Right Self-Placement")
        ax.set_xlabel("Election Year")
        ax.set_title("Public L-R Position Under Changing Governments\n"
                      "(blue=right gov, red=left gov)")
        ax.grid(True, alpha=0.3)

    # Panel 2: Gov vs opposition voter L-R
    ax = axes[1]
    if within_results:
        wr = pd.DataFrame(within_results)
        ax.plot(wr["year"], wr["inc_mean_lr"], "s-", color="#00529F",
                label="Incumbent voters", markersize=6)
        ax.plot(wr["year"], wr["opp_mean_lr"], "o-", color="#D82A20",
                label="Opposition voters", markersize=6)
        ax.axhline(5, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Mean Left-Right Self-Placement")
        ax.set_xlabel("Election Year")
        ax.set_title("L-R Position by Vote Choice")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Thermostatic Policy Responsiveness in NZ (Wlezien 1995)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "val_thermostatic.png", dpi=150)
    plt.close()

    return results


# ─── 2B. Issue Ownership ────────────────────────────────────────────────────────

def analyze_issue_ownership():
    """Test: do parties benefit when their 'owned' issues are salient?"""
    print("\n" + "=" * 70)
    print("2B. Issue Ownership (Petrocik 1996; Egan 2013)")
    print("=" * 70)

    conn = sqlite3.connect(str(NZES_DB))
    all_results = []
    salience_data = []

    for year, spec in sorted(ISSUE_VARS.items()):
        table = spec["table"]
        issue_col = spec["issue_col"]
        pv_col = spec["party_vote"]
        code_map = ISSUE_CODE_MAP.get(year, {})

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"\n  {year} ({table}): {len(df)} rows")

        # Check column exists
        if issue_col not in df.columns:
            print(f"    WARNING: {issue_col} not found in {table}")
            continue

        recode = PARTY_VOTE_RECODE.get(year, {})
        inc = INCUMBENTS.get(year)

        rows = []
        for _, raw in df.iterrows():
            row = {"election_year": year}

            # Party vote
            if pv_col in raw.index:
                pv = raw[pv_col]
                if pd.notna(pv):
                    pv_int = int(pv)
                    party = recode.get(pv_int)
                    if party:
                        row["party_vote"] = party
                        row["vote_national"] = 1 if party == "National" else (
                            0 if party == "Labour" else np.nan)
                        if inc:
                            row["vote_incumbent"] = 1 if party == inc else 0

            # Issue
            issue_val = raw[issue_col]
            if pd.notna(issue_val):
                try:
                    issue_int = int(float(issue_val))
                except (ValueError, OverflowError):
                    issue_int = None
                if issue_int and issue_int in code_map:
                    issue_name = code_map[issue_int]
                    row["issue"] = issue_name
                    row["national_issue"] = 1 if issue_name in NATIONAL_ISSUES else 0
                    row["labour_issue"] = 1 if issue_name in LABOUR_ISSUES else 0

            rows.append(row)

        yr_df = pd.DataFrame(rows)
        valid = yr_df[yr_df["issue"].notna() & yr_df["party_vote"].notna()]
        print(f"    Valid issue + vote: {len(valid)}")

        if len(valid) < 50:
            continue

        # Issue salience breakdown
        issue_counts = valid["issue"].value_counts()
        total = len(valid)
        nat_issue_pct = valid["national_issue"].mean() * 100
        lab_issue_pct = valid["labour_issue"].mean() * 100

        salience_data.append({
            "year": year, "n": total,
            "pct_national_issues": nat_issue_pct,
            "pct_labour_issues": lab_issue_pct,
            "top_issue": issue_counts.index[0],
            "top_issue_pct": issue_counts.iloc[0] / total * 100,
        })
        print(f"    Nat-owned issues: {nat_issue_pct:.1f}%, Lab-owned: {lab_issue_pct:.1f}%")
        print(f"    Top issue: {issue_counts.index[0]} ({issue_counts.iloc[0]/total*100:.1f}%)")

        # Cross-tabulation: issue ownership × vote
        nat_lab = valid[valid["party_vote"].isin(["National", "Labour"])].copy()
        if len(nat_lab) < 30:
            continue

        # Among those naming a National-owned issue, % voting National
        nat_issue_voters = nat_lab[nat_lab["national_issue"] == 1]
        lab_issue_voters = nat_lab[nat_lab["labour_issue"] == 1]

        nat_vote_among_nat_issue = (nat_issue_voters["party_vote"] == "National").mean() * 100 if len(nat_issue_voters) > 10 else np.nan
        nat_vote_among_lab_issue = (lab_issue_voters["party_vote"] == "National").mean() * 100 if len(lab_issue_voters) > 10 else np.nan

        year_result = {
            "year": year, "n": len(nat_lab),
            "nat_vote_nat_issue": nat_vote_among_nat_issue,
            "nat_vote_lab_issue": nat_vote_among_lab_issue,
        }

        if pd.notna(nat_vote_among_nat_issue) and pd.notna(nat_vote_among_lab_issue):
            gap = nat_vote_among_nat_issue - nat_vote_among_lab_issue
            year_result["ownership_gap"] = gap
            print(f"    Nat vote when Nat issue: {nat_vote_among_nat_issue:.1f}%, "
                  f"when Lab issue: {nat_vote_among_lab_issue:.1f}%, gap: {gap:+.1f}pp")

        # Logistic regression: P(vote Nat) ~ named_national_issue
        try:
            X = sm.add_constant(nat_lab[["national_issue"]])
            model = Logit(nat_lab["vote_national"].dropna(),
                         X.loc[nat_lab["vote_national"].dropna().index]).fit(disp=0)
            year_result["issue_coef"] = model.params.get("national_issue", np.nan)
            year_result["issue_p"] = model.pvalues.get("national_issue", np.nan)
            year_result["issue_r2"] = model.prsquared
        except Exception:
            pass

        all_results.append(year_result)

    conn.close()

    results_df = pd.DataFrame(all_results)
    salience_df = pd.DataFrame(salience_data)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Issue salience over time
    ax = axes[0]
    if not salience_df.empty:
        ax.bar(salience_df["year"] - 0.8, salience_df["pct_national_issues"],
               width=1.5, color="#00529F", alpha=0.7, label="Nat-owned issues")
        ax.bar(salience_df["year"] + 0.8, salience_df["pct_labour_issues"],
               width=1.5, color="#D82A20", alpha=0.7, label="Lab-owned issues")
        ax.set_ylabel("% of Respondents")
        ax.set_title("Issue Salience by Ownership")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Ownership gap over time
    ax = axes[1]
    if not results_df.empty and "ownership_gap" in results_df.columns:
        valid = results_df.dropna(subset=["ownership_gap"])
        if not valid.empty:
            ax.bar(valid["year"], valid["ownership_gap"], color="#098137", alpha=0.7, width=2)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_ylabel("Ownership Gap (pp)")
            ax.set_title("Issue Ownership Effect\n(Nat vote % on Nat issues minus Lab issues)")
            ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Nat vote by issue type
    ax = axes[2]
    if not results_df.empty:
        valid = results_df.dropna(subset=["nat_vote_nat_issue", "nat_vote_lab_issue"])
        if not valid.empty:
            ax.plot(valid["year"], valid["nat_vote_nat_issue"], "s-",
                    color="#00529F", label="Nat-owned issue", markersize=6)
            ax.plot(valid["year"], valid["nat_vote_lab_issue"], "o-",
                    color="#D82A20", label="Lab-owned issue", markersize=6)
            ax.axhline(50, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylabel("National Vote Share (%)")
            ax.set_title("National Vote by Issue Named")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Issue Ownership in NZ Elections (Petrocik 1996)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "val_issue_ownership.png", dpi=150)
    plt.close()

    return {"results": results_df, "salience": salience_df}


# ─── 2C. Valence Politics ───────────────────────────────────────────────────────

def analyze_valence():
    """Test: does perceived competence explain more than ideological proximity?"""
    print("\n" + "=" * 70)
    print("2C. Valence Politics (Stokes 1963; Clarke et al. 2004)")
    print("=" * 70)

    conn = sqlite3.connect(str(NZES_DB))
    nzes = pd.read_csv(DATA_DIR / "nzes_harmonized.csv")

    all_results = []
    competence_results = []

    for year, spec in sorted(LEADER_VARS.items()):
        table = spec["table"]
        pm_col = spec["pm_like"]
        opp_col = spec["opp_like"]
        pv_col = spec["party_vote"]
        dk = spec["dk_code"]
        reverse = spec["reverse"]
        pm_party = spec["pm_party"]

        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"\n  {year} ({table}): {len(df)} rows")

        recode = PARTY_VOTE_RECODE.get(year, {})
        inc = INCUMBENTS.get(year)

        rows = []
        for _, raw in df.iterrows():
            row = {"election_year": year}

            # Party vote
            if pv_col in raw.index:
                pv = raw[pv_col]
                if pd.notna(pv):
                    pv_int = int(pv)
                    party = recode.get(pv_int)
                    if party:
                        row["party_vote"] = party
                        row["vote_national"] = 1 if party == "National" else (
                            0 if party == "Labour" else np.nan)
                        if inc:
                            row["vote_incumbent"] = 1 if party == inc else 0

            # Leader thermometers (0-10 scale)
            for col_name, var_name in [("pm_like", pm_col), ("opp_like", opp_col)]:
                if var_name in raw.index:
                    val = raw[var_name]
                    if pd.notna(val):
                        v = float(val)
                        if v == dk or v > 10:
                            continue
                        if 0 <= v <= 10:
                            if reverse:
                                v = 10 - v  # Fix 1999 reverse coding
                            row[col_name] = v

            # Competence ratings (1-4, lower = more competent)
            pm_comp_col = spec.get("pm_comp")
            opp_comp_col = spec.get("opp_comp")
            if pm_comp_col and pm_comp_col in raw.index:
                val = raw[pm_comp_col]
                if pd.notna(val):
                    v = float(val)
                    if 1 <= v <= 4:
                        row["pm_comp"] = 5 - v  # Invert so higher = more competent

            if opp_comp_col and opp_comp_col in raw.index:
                val = raw[opp_comp_col]
                if pd.notna(val):
                    v = float(val)
                    if 1 <= v <= 4:
                        row["opp_comp"] = 5 - v

            # L-R self and party placements from harmonized data
            nzes_match = nzes[(nzes["election_year"] == year)]
            # We'll get these from the harmonized dataset after merging

            rows.append(row)

        yr_df = pd.DataFrame(rows)

        # Compute valence: PM like minus Opp like
        if "pm_like" in yr_df.columns and "opp_like" in yr_df.columns:
            yr_df["leader_valence"] = yr_df["pm_like"] - yr_df["opp_like"]

        # Get ideology from harmonized data
        yr_nzes = nzes[nzes["election_year"] == year].copy()

        # Ideology proximity: |self - incumbent party LR| minus |self - opposition party LR|
        if inc == "National":
            lr_inc_col = "lr_national"
            lr_opp_col = "lr_labour"
        else:
            lr_inc_col = "lr_labour"
            lr_opp_col = "lr_national"

        if lr_inc_col in yr_nzes.columns and lr_opp_col in yr_nzes.columns and "lr_self" in yr_nzes.columns:
            yr_nzes["ideo_proximity"] = (
                abs(yr_nzes["lr_self"] - yr_nzes[lr_opp_col]) -
                abs(yr_nzes["lr_self"] - yr_nzes[lr_inc_col])
            )
            # Positive = closer to incumbent

            # Merge ideology into leader data (by index position — same survey)
            # Since both come from same table, align by matching lengths
            if len(yr_df) == len(yr_nzes):
                yr_df["ideo_proximity"] = yr_nzes["ideo_proximity"].values
                yr_df["lr_self"] = yr_nzes["lr_self"].values if "lr_self" in yr_nzes.columns else np.nan

        # Model comparison: vote_incumbent ~ valence vs ideology
        year_result = {"year": year, "pm_party": pm_party}

        # Filter to Nat/Lab voters with at least valence
        valid_base = yr_df[yr_df["vote_incumbent"].notna() &
                           yr_df["party_vote"].isin(["National", "Labour"])].copy()

        # Model A: Ideology only
        valid_a = valid_base[["vote_incumbent", "ideo_proximity"]].dropna()
        if len(valid_a) > 50:
            try:
                X = sm.add_constant(valid_a[["ideo_proximity"]])
                model = Logit(valid_a["vote_incumbent"], X).fit(disp=0, maxiter=100)
                year_result["ideo_r2"] = model.prsquared
                year_result["ideo_coef"] = model.params.get("ideo_proximity", np.nan)
                year_result["ideo_p"] = model.pvalues.get("ideo_proximity", np.nan)
                year_result["ideo_n"] = len(valid_a)
                print(f"    Ideology-only: R²={model.prsquared:.4f}, n={len(valid_a)}")
            except Exception as e:
                print(f"    Ideology model error: {e}")

        # Model B: Valence (leader evaluation) only
        valid_b = valid_base[["vote_incumbent", "leader_valence"]].dropna()
        if len(valid_b) > 50:
            try:
                X = sm.add_constant(valid_b[["leader_valence"]])
                model = Logit(valid_b["vote_incumbent"], X).fit(disp=0, maxiter=100)
                year_result["valence_r2"] = model.prsquared
                year_result["valence_coef"] = model.params.get("leader_valence", np.nan)
                year_result["valence_p"] = model.pvalues.get("leader_valence", np.nan)
                year_result["valence_n"] = len(valid_b)
                print(f"    Valence-only:  R²={model.prsquared:.4f}, n={len(valid_b)}")
            except Exception as e:
                print(f"    Valence model error: {e}")

        # Model C: Both
        valid_c = valid_base[["vote_incumbent", "ideo_proximity", "leader_valence"]].dropna()
        if len(valid_c) > 50:
            try:
                X = sm.add_constant(valid_c[["ideo_proximity", "leader_valence"]])
                model = Logit(valid_c["vote_incumbent"], X).fit(disp=0, maxiter=100)
                year_result["combined_r2"] = model.prsquared
                year_result["combined_ideo_coef"] = model.params.get("ideo_proximity", np.nan)
                year_result["combined_val_coef"] = model.params.get("leader_valence", np.nan)
                year_result["combined_ideo_p"] = model.pvalues.get("ideo_proximity", np.nan)
                year_result["combined_val_p"] = model.pvalues.get("leader_valence", np.nan)
                year_result["combined_n"] = len(valid_c)
                print(f"    Combined:      R²={model.prsquared:.4f}, n={len(valid_c)}")
            except Exception as e:
                print(f"    Combined model error: {e}")

        all_results.append(year_result)

        # Competence analysis (where available)
        if "pm_comp" in yr_df.columns and "opp_comp" in yr_df.columns:
            yr_df["comp_valence"] = yr_df["pm_comp"] - yr_df["opp_comp"]
            valid_comp = yr_df[yr_df["vote_incumbent"].notna() &
                               yr_df["comp_valence"].notna() &
                               yr_df["party_vote"].isin(["National", "Labour"])].copy()
            if len(valid_comp) > 30:
                try:
                    X = sm.add_constant(valid_comp[["comp_valence"]])
                    model = Logit(valid_comp["vote_incumbent"], X).fit(disp=0, maxiter=100)
                    competence_results.append({
                        "year": year,
                        "comp_r2": model.prsquared,
                        "comp_coef": model.params.get("comp_valence", np.nan),
                        "comp_p": model.pvalues.get("comp_valence", np.nan),
                        "n": len(valid_comp),
                    })
                    print(f"    Competence:    R²={model.prsquared:.4f}, n={len(valid_comp)}")
                except Exception:
                    pass

    conn.close()

    results_df = pd.DataFrame(all_results)
    comp_df = pd.DataFrame(competence_results)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: R² comparison across years
    ax = axes[0]
    if not results_df.empty:
        valid = results_df.dropna(subset=["valence_r2"])
        if not valid.empty:
            x = np.arange(len(valid))
            w = 0.25
            ideo_r2 = valid["ideo_r2"].fillna(0)
            val_r2 = valid["valence_r2"].fillna(0)
            comb_r2 = valid["combined_r2"].fillna(0) if "combined_r2" in valid.columns else pd.Series([0]*len(valid))

            ax.bar(x - w, ideo_r2, w, label="Ideology", color="#098137", alpha=0.8)
            ax.bar(x, val_r2, w, label="Leader Valence", color="#D82A20", alpha=0.8)
            ax.bar(x + w, comb_r2, w, label="Combined", color="#00529F", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(y)) for y in valid["year"]], rotation=45)
            ax.set_ylabel("McFadden Pseudo-R²")
            ax.set_title("Model Fit: Ideology vs Valence")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: Valence coefficient over time
    ax = axes[1]
    if not results_df.empty:
        valid = results_df.dropna(subset=["valence_coef"])
        if not valid.empty:
            ax.plot(valid["year"], valid["valence_coef"], "o-", color="#D82A20",
                    label="Leader valence β", markersize=6)
            if "ideo_coef" in valid.columns:
                ideo_valid = valid.dropna(subset=["ideo_coef"])
                if not ideo_valid.empty:
                    ax.plot(ideo_valid["year"], ideo_valid["ideo_coef"], "s-",
                            color="#098137", label="Ideology proximity β", markersize=6)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylabel("Logit Coefficient")
            ax.set_title("Coefficient Evolution")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    # Panel 3: Competence vs thermometer comparison
    ax = axes[2]
    if not comp_df.empty:
        # Compare competence R² vs thermometer R² for same years
        merged = comp_df.merge(results_df[["year", "valence_r2"]].dropna(),
                               on="year", how="inner")
        if not merged.empty:
            x = np.arange(len(merged))
            w = 0.3
            ax.bar(x - w/2, merged["valence_r2"], w, label="Thermometer", color="#D82A20", alpha=0.8)
            ax.bar(x + w/2, merged["comp_r2"], w, label="Competence", color="#00529F", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(y)) for y in merged["year"]])
            ax.set_ylabel("McFadden Pseudo-R²")
            ax.set_title("Thermometer vs Competence Ratings")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "Competence data\navailable for\n2014, 2017, 2023 only",
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_title("Thermometer vs Competence")
    else:
        ax.set_visible(False)

    plt.suptitle("Valence Politics in NZ Elections (Stokes 1963)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "val_valence_politics.png", dpi=150)
    plt.close()

    return {"results": results_df, "competence": comp_df}


# ─── Report Generation ──────────────────────────────────────────────────────────

def generate_report(thermo_results, issue_results, valence_results):
    """Generate markdown report."""
    report = []
    report.append("# Issue & Valence Politics\n")
    report.append("*Three hypotheses about how voters evaluate parties beyond left-right ideology*\n")

    # ── 2A: Thermostatic ──
    report.append("## 2A. Thermostatic Policy Responsiveness (Wlezien 1995)\n")
    report.append("**Question**: When government policy moves left, does the public shift right "
                  "(and vice versa)?\n")

    if thermo_results:
        lr_df = thermo_results.get("lr_data")
        if lr_df is not None and not lr_df.empty:
            report.append("### Mean Left-Right Self-Placement by Election\n")
            report.append("| Year | Mean L-R | SD | n | Gov Direction | Shift |")
            report.append("|------|----------|----|----|---------------|-------|")
            for _, row in lr_df.iterrows():
                shift = f"{row.get('lr_shift', np.nan):+.2f}" if pd.notna(row.get("lr_shift")) else "—"
                report.append(f"| {int(row['year'])} | {row['mean_lr']:.2f} | {row['sd_lr']:.2f} | "
                              f"{int(row['n'])} | {row.get('gov_direction', '—')} | {shift} |")

            correct = thermo_results.get("correct_count", 0)
            total = thermo_results.get("total_shifts", 0)
            if total > 0:
                report.append(f"\nThermostatic shifts in expected direction: **{correct}/{total}**\n")
                if correct > total / 2:
                    report.append("**Finding**: Evidence of thermostatic responsiveness — the public "
                                  "drifts opposite to government policy direction, consistent with Wlezien (1995).\n")
                else:
                    report.append("**Finding**: Limited evidence of thermostatic responsiveness. "
                                  "Public L-R placement remains relatively stable regardless of government direction.\n")

        within = thermo_results.get("within_election")
        if within is not None and not within.empty:
            report.append("### Within-Election: Incumbent vs Opposition Voter L-R\n")
            report.append("| Year | Inc Voter L-R | Opp Voter L-R | Gap |")
            report.append("|------|---------------|---------------|-----|")
            for _, row in within.iterrows():
                report.append(f"| {int(row['year'])} | {row['inc_mean_lr']:.2f} | "
                              f"{row['opp_mean_lr']:.2f} | {row['gap']:+.2f} |")

    report.append(f"\n![Thermostatic](../graphs/val_thermostatic.png)\n")

    # ── 2B: Issue Ownership ──
    report.append("## 2B. Issue Ownership (Petrocik 1996; Egan 2013)\n")
    report.append("**Question**: Do parties benefit when their 'owned' issues are salient?\n")
    report.append(f"National-owned issues: {', '.join(sorted(NATIONAL_ISSUES))}\n")
    report.append(f"Labour-owned issues: {', '.join(sorted(LABOUR_ISSUES))}\n")

    if issue_results:
        salience = issue_results.get("salience")
        if salience is not None and not salience.empty:
            report.append("### Issue Salience\n")
            report.append("| Year | n | Nat Issues % | Lab Issues % | Top Issue |")
            report.append("|------|---|-------------|-------------|-----------|")
            for _, row in salience.iterrows():
                report.append(f"| {int(row['year'])} | {int(row['n'])} | "
                              f"{row['pct_national_issues']:.1f}% | {row['pct_labour_issues']:.1f}% | "
                              f"{row['top_issue']} ({row['top_issue_pct']:.0f}%) |")

        results = issue_results.get("results")
        if results is not None and not results.empty:
            report.append("\n### Issue Ownership Effect\n")
            report.append("| Year | Nat Vote (Nat Issue) | Nat Vote (Lab Issue) | Gap | Logit β | p | R² |")
            report.append("|------|---------------------|---------------------|-----|---------|---|----|")
            for _, row in results.iterrows():
                nni = f"{row.get('nat_vote_nat_issue', np.nan):.1f}%" if pd.notna(row.get("nat_vote_nat_issue")) else "—"
                nli = f"{row.get('nat_vote_lab_issue', np.nan):.1f}%" if pd.notna(row.get("nat_vote_lab_issue")) else "—"
                gap = f"{row.get('ownership_gap', np.nan):+.1f}pp" if pd.notna(row.get("ownership_gap")) else "—"
                coef = f"{row.get('issue_coef', np.nan):.3f}" if pd.notna(row.get("issue_coef")) else "—"
                p = f"{row.get('issue_p', np.nan):.4f}" if pd.notna(row.get("issue_p")) else "—"
                r2 = f"{row.get('issue_r2', np.nan):.4f}" if pd.notna(row.get("issue_r2")) else "—"
                report.append(f"| {int(row['year'])} | {nni} | {nli} | {gap} | {coef} | {p} | {r2} |")

            # Summary
            valid_gaps = results["ownership_gap"].dropna()
            if not valid_gaps.empty:
                mean_gap = valid_gaps.mean()
                positive = (valid_gaps > 0).sum()
                report.append(f"\n**Finding**: Issue ownership gap averages {mean_gap:+.1f}pp across "
                              f"{len(valid_gaps)} elections. The gap is positive (confirming theory) in "
                              f"{positive}/{len(valid_gaps)} elections.\n")

    report.append(f"\n![Issue Ownership](../graphs/val_issue_ownership.png)\n")

    # ── 2C: Valence ──
    report.append("## 2C. Valence Politics (Stokes 1963; Clarke et al. 2004)\n")
    report.append("**Question**: Does perceived leader quality explain more than ideological proximity?\n")

    if valence_results:
        vr = valence_results.get("results")
        if vr is not None and not vr.empty:
            report.append("### Model Comparison: Pseudo-R² Across Elections\n")
            report.append("| Year | Ideology R² | Valence R² | Combined R² | n |")
            report.append("|------|------------|------------|-------------|---|")
            for _, row in vr.iterrows():
                ir2 = f"{row.get('ideo_r2', np.nan):.4f}" if pd.notna(row.get("ideo_r2")) else "—"
                vr2 = f"{row.get('valence_r2', np.nan):.4f}" if pd.notna(row.get("valence_r2")) else "—"
                cr2 = f"{row.get('combined_r2', np.nan):.4f}" if pd.notna(row.get("combined_r2")) else "—"
                n_val = row.get("combined_n", row.get("valence_n", row.get("ideo_n", 0)))
                n = int(n_val) if pd.notna(n_val) else 0
                report.append(f"| {int(row['year'])} | {ir2} | {vr2} | {cr2} | {n} |")

            # Summary
            has_both = vr.dropna(subset=["ideo_r2", "valence_r2"])
            if not has_both.empty:
                val_wins = (has_both["valence_r2"] > has_both["ideo_r2"]).sum()
                report.append(f"\n**Finding**: Leader valence (thermometer ratings) outperforms ideological "
                              f"proximity in **{val_wins}/{len(has_both)}** elections. ")
                avg_val = has_both["valence_r2"].mean()
                avg_ideo = has_both["ideo_r2"].mean()
                report.append(f"Average R²: Valence={avg_val:.4f}, Ideology={avg_ideo:.4f}. ")
                if avg_val > avg_ideo * 1.5:
                    report.append("NZ elections are predominantly valence-driven, consistent with "
                                  "Clarke et al. (2004).\n")
                else:
                    report.append("Both factors contribute, though valence tends to dominate.\n")

        comp = valence_results.get("competence")
        if comp is not None and not comp.empty:
            report.append("\n### Competence Ratings (Where Available)\n")
            report.append("| Year | Competence R² | Competence β | p | n |")
            report.append("|------|--------------|-------------|---|---|")
            for _, row in comp.iterrows():
                sig = "***" if row["comp_p"] < 0.001 else "**" if row["comp_p"] < 0.01 else "*" if row["comp_p"] < 0.05 else ""
                report.append(f"| {int(row['year'])} | {row['comp_r2']:.4f} | "
                              f"{row['comp_coef']:.3f}{sig} | {row['comp_p']:.4f} | {int(row['n'])} |")

    report.append(f"\n![Valence Politics](../graphs/val_valence_politics.png)\n")

    # Write report
    report_text = "\n".join(report)
    report_path = REPORT_DIR / "valence_and_issues.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Issue & Valence Politics")
    print("=" * 70)

    nzes_df = load_nzes()

    thermo_results = analyze_thermostatic(nzes_df)
    issue_results = analyze_issue_ownership()
    valence_results = analyze_valence()

    generate_report(thermo_results, issue_results, valence_results)

    print("\n" + "=" * 70)
    print("Script 2 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
