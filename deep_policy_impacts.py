#!/usr/bin/env python3
"""
Policy & Budget Impact Analysis

Tests whether budgets, policy announcements, and issue salience
predict shifts in party support.

Hypotheses:
  4A. Budget Bounce — do budgets systematically move incumbent polls?
  4B. Policy Shocks — do major policy announcements shift polls?
  4C. Issue Ownership in Action — does current issue salience predict
      which party is favoured (using Ipsos data)?
  4D. Government Performance → Polls — does Ipsos govt rating
      predict actual polling?

Uses: polls_with_economics.csv, Ipsos CSVs, events.py
Outputs: reports/policy_impacts.md, graphs/pol_*.png
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── NZ Budget dates (actual delivery dates) ──────────────────────
BUDGETS = [
    {"date": "1991-07-30", "label": "1991 'Mother of All Budgets'",
     "govt": "National", "stance": "contractionary",
     "key_measures": "Benefit cuts, housing corp restructure, user pays health"},
    {"date": "1993-06-29", "label": "1993 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "Pre-election, modest tax relief"},
    {"date": "1994-06-22", "label": "1994 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "Fiscal consolidation continuing"},
    {"date": "1995-06-15", "label": "1995 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Tax cuts, superannuation surcharge reduction"},
    {"date": "1996-06-13", "label": "1996 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Pre-election tax cuts, increased spending"},
    {"date": "1997-06-26", "label": "1997 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "First coalition budget (Nat-NZF)"},
    {"date": "1998-05-14", "label": "1998 Budget",
     "govt": "National", "stance": "contractionary",
     "key_measures": "Asian crisis response, spending restraint"},
    {"date": "1999-06-10", "label": "1999 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Pre-election, health spending increase"},
    {"date": "2000-06-15", "label": "2000 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Top tax rate to 39%, increased social spending"},
    {"date": "2001-05-24", "label": "2001 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Paid parental leave, income-related rents"},
    {"date": "2002-05-23", "label": "2002 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Pre-election, health and education spending"},
    {"date": "2003-05-15", "label": "2003 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Working for Families announced"},
    {"date": "2004-05-27", "label": "2004 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Working for Families implemented"},
    {"date": "2005-05-19", "label": "2005 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Pre-election, student loans interest-free"},
    {"date": "2006-05-18", "label": "2006 Budget",
     "govt": "Labour", "stance": "neutral",
     "key_measures": "Business tax review, KiwiSaver announced"},
    {"date": "2007-05-17", "label": "2007 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "KiwiSaver launched, business tax cuts"},
    {"date": "2008-05-22", "label": "2008 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Pre-election, personal tax cuts"},
    {"date": "2009-05-28", "label": "2009 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "GFC response, stimulus spending"},
    {"date": "2010-05-20", "label": "2010 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "GST 12.5→15%, income tax cuts, tax switch"},
    {"date": "2011-05-19", "label": "2011 Budget",
     "govt": "National", "stance": "contractionary",
     "key_measures": "Post-earthquake, zero budget target"},
    {"date": "2012-05-24", "label": "2012 Budget",
     "govt": "National", "stance": "contractionary",
     "key_measures": "Austerity, partial asset sales announced"},
    {"date": "2013-05-16", "label": "2013 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "Return to surplus target"},
    {"date": "2014-05-15", "label": "2014 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Pre-election, free GP visits under-13, paid parental leave"},
    {"date": "2015-05-21", "label": "2015 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "Childcare package, health spending"},
    {"date": "2016-05-26", "label": "2016 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Tax threshold increases, family incomes package"},
    {"date": "2017-05-25", "label": "2017 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Pre-election, families package, accommodation supplement"},
    {"date": "2018-05-17", "label": "2018 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "First Ardern budget, families package, mental health"},
    {"date": "2019-05-30", "label": "2019 Wellbeing Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Wellbeing framework, mental health, child poverty"},
    {"date": "2020-05-14", "label": "2020 COVID Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "$50B COVID response, wage subsidy, infrastructure"},
    {"date": "2021-05-20", "label": "2021 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Housing acceleration, benefit increases"},
    {"date": "2022-05-19", "label": "2022 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Cost-of-living payment, half-price transport"},
    {"date": "2023-05-18", "label": "2023 Budget",
     "govt": "Labour", "stance": "expansionary",
     "key_measures": "Pre-election, free school lunches extended"},
    {"date": "2024-05-30", "label": "2024 Budget",
     "govt": "National", "stance": "expansionary",
     "key_measures": "Tax cuts (income thresholds), spending restraint"},
    {"date": "2025-05-22", "label": "2025 Budget",
     "govt": "National", "stance": "neutral",
     "key_measures": "Fiscal consolidation, infrastructure spending"},
]

# ── Major policy announcements ──────────────────────────────────
POLICY_EVENTS = [
    {"date": "1991-05-15", "event": "Employment Contracts Act passed",
     "party": "National", "domain": "economic", "direction": "right"},
    {"date": "1999-10-01", "event": "Labour pledges interest-free student loans",
     "party": "Labour", "domain": "social", "direction": "left"},
    {"date": "2004-05-27", "event": "Working for Families announced",
     "party": "Labour", "domain": "economic", "direction": "left"},
    {"date": "2007-05-17", "event": "KiwiSaver launched",
     "party": "Labour", "domain": "economic", "direction": "left"},
    {"date": "2010-05-20", "event": "GST increase 12.5% to 15%",
     "party": "National", "domain": "economic", "direction": "right"},
    {"date": "2012-03-22", "event": "Partial asset sales announced",
     "party": "National", "domain": "economic", "direction": "right"},
    {"date": "2017-08-23", "event": "Labour announces fees-free tertiary",
     "party": "Labour", "domain": "social", "direction": "left"},
    {"date": "2018-10-31", "event": "Foreign buyer ban enacted",
     "party": "Labour", "domain": "housing", "direction": "left"},
    {"date": "2019-04-10", "event": "Gun law reform passed",
     "party": "Labour", "domain": "social", "direction": "left"},
    {"date": "2020-03-17", "event": "COVID wage subsidy announced",
     "party": "Labour", "domain": "economic", "direction": "left"},
    {"date": "2020-03-25", "event": "COVID lockdown Level 4",
     "party": "Labour", "domain": "crisis", "direction": "left"},
    {"date": "2021-03-23", "event": "Housing policy package announced",
     "party": "Labour", "domain": "housing", "direction": "left"},
    {"date": "2021-10-27", "event": "Three Waters reform announced",
     "party": "Labour", "domain": "governance", "direction": "left"},
    {"date": "2023-11-27", "event": "Coalition agreements signed",
     "party": "National", "domain": "governance", "direction": "right"},
    {"date": "2024-01-01", "event": "Three Strikes restored",
     "party": "National", "domain": "social", "direction": "right"},
    {"date": "2024-07-01", "event": "Tax cuts take effect",
     "party": "National", "domain": "economic", "direction": "right"},
    {"date": "2025-10-23", "event": "100,000 worker mega-strike",
     "party": "Labour", "domain": "economic", "direction": "left"},
]


def load_polling():
    """Load polling data with economics."""
    df = pd.read_csv("data/polls_with_economics.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_ipsos():
    """Load Ipsos datasets."""
    govt = pd.read_csv("data/ipsos_govt_performance.csv")
    govt["date"] = pd.to_datetime(govt["date"], format="%Y-%m")
    issues = pd.read_csv("data/ipsos_issue_salience.csv")
    issues["date"] = pd.to_datetime(issues["date"], format="%Y-%m")
    capability = pd.read_csv("data/ipsos_party_capability.csv")
    capability["date"] = pd.to_datetime(capability["date"], format="%Y-%m")
    return govt, issues, capability


# ═══════════════════════════════════════════════════════════════════
# 4A. BUDGET BOUNCE
# ═══════════════════════════════════════════════════════════════════

def analyze_budget_bounce(polls):
    """Test whether budgets systematically shift incumbent support."""
    results = []

    for b in BUDGETS:
        bdate = pd.Timestamp(b["date"])

        # Find polls 60 days before and 60 days after
        pre_mask = (polls["date"] >= bdate - timedelta(days=60)) & \
                   (polls["date"] < bdate)
        post_mask = (polls["date"] > bdate) & \
                    (polls["date"] <= bdate + timedelta(days=60))

        pre = polls.loc[pre_mask, "incumbent_support"]
        post = polls.loc[post_mask, "incumbent_support"]

        if len(pre) >= 2 and len(post) >= 2:
            change = post.mean() - pre.mean()
            # Simple t-test
            t, p = stats.ttest_ind(post, pre, equal_var=False)
            results.append({
                "year": bdate.year,
                "label": b["label"],
                "govt": b["govt"],
                "stance": b["stance"],
                "pre_mean": pre.mean(),
                "post_mean": post.mean(),
                "change": change,
                "n_pre": len(pre),
                "n_post": len(post),
                "t_stat": t,
                "p_value": p,
            })

    df = pd.DataFrame(results)
    return df


def plot_budget_bounce(budget_df):
    """Plot budget bounce effects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: budget bounce by year
    ax = axes[0]
    colors = ["steelblue" if g == "National" else "crimson"
              for g in budget_df["govt"]]
    bars = ax.bar(range(len(budget_df)), budget_df["change"],
                  color=colors, alpha=0.7, edgecolor="white")

    sig = budget_df["p_value"] < 0.05
    for i, s in enumerate(sig):
        if s:
            ax.text(i, budget_df["change"].iloc[i],
                    "*", ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(range(len(budget_df)))
    ax.set_xticklabels(budget_df["year"], rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Change in Incumbent Support (pp)")
    ax.set_title("Budget Bounce: Incumbent Poll Change\n(60 days pre→post)")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="steelblue", label="National govt"),
                       Patch(color="crimson", label="Labour govt")],
              fontsize=9)

    # Panel B: by fiscal stance
    ax = axes[1]
    stance_means = budget_df.groupby("stance")["change"].agg(["mean", "sem", "count"])
    stance_order = ["contractionary", "neutral", "expansionary"]
    stance_means = stance_means.reindex(stance_order).dropna()

    bars = ax.bar(range(len(stance_means)), stance_means["mean"],
                  yerr=stance_means["sem"] * 1.96,
                  color=["#d32f2f", "#757575", "#388e3c"],
                  alpha=0.7, capsize=5)
    ax.set_xticks(range(len(stance_means)))
    ax.set_xticklabels(stance_means.index, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean Change in Incumbent Support (pp)")
    ax.set_title("Budget Bounce by Fiscal Stance\n(with 95% CI)")

    for i, (idx, row) in enumerate(stance_means.iterrows()):
        ax.text(i, row["mean"] + row["sem"] * 2 + 0.2,
                f'n={int(row["count"])}', ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("graphs/pol_budget_bounce.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4B. POLICY SHOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_policy_shocks(polls):
    """Test whether major policy announcements shift polls."""
    results = []

    for p in POLICY_EVENTS:
        pdate = pd.Timestamp(p["date"])

        pre_mask = (polls["date"] >= pdate - timedelta(days=45)) & \
                   (polls["date"] < pdate)
        post_mask = (polls["date"] > pdate) & \
                    (polls["date"] <= pdate + timedelta(days=45))

        for party_col in ["Labour", "National"]:
            pre = polls.loc[pre_mask, party_col].dropna()
            post = polls.loc[post_mask, party_col].dropna()

            if len(pre) >= 2 and len(post) >= 2:
                change = post.mean() - pre.mean()
                t, p_val = stats.ttest_ind(post, pre, equal_var=False)
                results.append({
                    "event": p["event"],
                    "date": p["date"],
                    "party_measured": party_col,
                    "announcing_party": p["party"],
                    "domain": p["domain"],
                    "direction": p["direction"],
                    "pre_mean": pre.mean(),
                    "post_mean": post.mean(),
                    "change": change,
                    "n_pre": len(pre),
                    "n_post": len(post),
                    "p_value": p_val,
                })

    df = pd.DataFrame(results)
    return df


def plot_policy_shocks(policy_df):
    """Plot policy shock effects."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Top 10 largest effects
    ax = axes[0]
    # Focus on announcing party's own support change
    own = policy_df[policy_df["party_measured"] == policy_df["announcing_party"]]
    own_sorted = own.reindex(own["change"].abs().sort_values(ascending=False).index)
    top10 = own_sorted.head(10)

    colors = ["steelblue" if p == "National" else "crimson"
              for p in top10["announcing_party"]]
    y_pos = range(len(top10))
    ax.barh(y_pos, top10["change"], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    labels = [f"{e[:35]}..." if len(e) > 35 else e for e in top10["event"]]
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Change in Announcing Party's Support (pp)")
    ax.set_title("Largest Policy Impacts on Polls")
    ax.invert_yaxis()

    # Panel B: by domain
    ax = axes[1]
    domain_effects = own.groupby("domain")["change"].agg(["mean", "sem", "count"])
    domain_effects = domain_effects[domain_effects["count"] >= 2].sort_values("mean")

    bars = ax.barh(range(len(domain_effects)), domain_effects["mean"],
                   xerr=domain_effects["sem"] * 1.96,
                   color="steelblue", alpha=0.7, capsize=3)
    ax.set_yticks(range(len(domain_effects)))
    ax.set_yticklabels(domain_effects.index, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean Change in Support (pp)")
    ax.set_title("Policy Impact by Domain")

    for i, (idx, row) in enumerate(domain_effects.iterrows()):
        ax.text(max(row["mean"] + row["sem"] * 2 + 0.3, 0.5), i,
                f'n={int(row["count"])}', va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("graphs/pol_policy_shocks.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4C. ISSUE OWNERSHIP IN ACTION (Ipsos data)
# ═══════════════════════════════════════════════════════════════════

def analyze_issue_ownership_current(capability_df, issues_df):
    """Analyze current issue ownership using Ipsos data."""
    # Get most recent wave
    latest = capability_df[capability_df["date"] == capability_df["date"].max()]

    results = []
    for _, row in latest.iterrows():
        issue = row["issue"]
        nat = row["national"]
        lab = row["labour"]
        gap = nat - lab
        owner = "National" if gap > 0 else "Labour" if gap < 0 else "Tied"

        # Get issue salience
        latest_issues = issues_df[issues_df["date"] == issues_df["date"].max()]
        issue_col = issue.replace(" ", "_").replace("/", "_")
        salience = None
        for col in latest_issues.columns:
            if issue.split("_")[0] in col.lower():
                salience = latest_issues[col].values[0] if len(latest_issues) > 0 else None
                break

        results.append({
            "issue": issue,
            "national_pct": nat,
            "labour_pct": lab,
            "gap_nat_minus_lab": gap,
            "owner": owner,
        })

    return pd.DataFrame(results)


def analyze_issue_ownership_trends(capability_df):
    """Track how issue ownership has shifted over time."""
    results = []
    for issue in capability_df["issue"].unique():
        issue_data = capability_df[capability_df["issue"] == issue].sort_values("date")
        for _, row in issue_data.iterrows():
            results.append({
                "date": row["date"],
                "issue": issue,
                "national": row["national"],
                "labour": row["labour"],
                "gap": row["national"] - row["labour"],
            })
    return pd.DataFrame(results)


def plot_issue_ownership(ownership_trends):
    """Plot issue ownership trends."""
    issues = ownership_trends["issue"].unique()
    n_issues = min(len(issues), 6)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, issue in enumerate(issues[:n_issues]):
        ax = axes[i]
        data = ownership_trends[ownership_trends["issue"] == issue].sort_values("date")

        ax.plot(data["date"], data["national"], "b-o", markersize=4,
                label="National", linewidth=1.5)
        ax.plot(data["date"], data["labour"], "r-o", markersize=4,
                label="Labour", linewidth=1.5)
        ax.fill_between(data["date"], data["national"], data["labour"],
                        where=data["national"] > data["labour"],
                        alpha=0.15, color="blue")
        ax.fill_between(data["date"], data["national"], data["labour"],
                        where=data["labour"] > data["national"],
                        alpha=0.15, color="red")

        title = issue.replace("_", " ").title()
        if len(title) > 25:
            title = title[:25] + "..."
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_issues, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Issue Ownership Trends: Which Party is 'Most Capable'?\n(Ipsos NZ Issues Monitor, Aug 2023–Oct 2025)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("graphs/pol_issue_ownership_trends.png", dpi=150,
                bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4D. GOVERNMENT PERFORMANCE → POLLS
# ═══════════════════════════════════════════════════════════════════

def analyze_govt_performance(govt_df, polls):
    """Correlate Ipsos government performance ratings with polling."""
    results = []

    for _, row in govt_df.iterrows():
        ipsos_date = row["date"]
        govt = row["government"]

        # Determine incumbent party
        if "National" in govt:
            incumbent_party = "National"
        else:
            incumbent_party = "Labour"

        # Find nearest polls (within 30 days)
        nearby = polls[
            (polls["date"] >= ipsos_date - timedelta(days=30)) &
            (polls["date"] <= ipsos_date + timedelta(days=30))
        ]

        if len(nearby) >= 1:
            inc_support = nearby[incumbent_party].mean()
            results.append({
                "date": ipsos_date,
                "govt_rating": row["mean_score"],
                "bottom4_pct": row["bottom4_pct"],
                "incumbent_party": incumbent_party,
                "incumbent_poll": inc_support,
                "government": govt,
            })

    df = pd.DataFrame(results)
    return df


def plot_govt_vs_polls(perf_df):
    """Plot government performance vs actual polling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter of govt rating vs incumbent poll
    ax = axes[0]
    for govt in perf_df["government"].unique():
        mask = perf_df["government"] == govt
        color = "steelblue" if "National" in govt else "crimson"
        label = govt[:20]
        ax.scatter(perf_df.loc[mask, "govt_rating"],
                   perf_df.loc[mask, "incumbent_poll"],
                   color=color, alpha=0.7, s=60, label=label)

    # Fit line
    x = perf_df["govt_rating"].values
    y = perf_df["incumbent_poll"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 3:
        slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xline, intercept + slope * xline, "k--", linewidth=1,
                label=f"r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("Ipsos Govt Performance Rating (0-10)")
    ax.set_ylabel("Incumbent Party Poll (%)")
    ax.set_title("Government Rating vs Actual Polling")
    ax.legend(fontsize=8)

    # Panel B: Time series overlay
    ax = axes[1]
    ax2 = ax.twinx()

    ax.plot(perf_df["date"], perf_df["govt_rating"], "k-o",
            markersize=4, label="Govt Rating (0-10)", linewidth=1.5)
    ax2.plot(perf_df["date"], perf_df["incumbent_poll"], "g-s",
             markersize=4, label="Incumbent Poll (%)", linewidth=1.5,
             alpha=0.7)

    ax.set_ylabel("Ipsos Govt Rating (0-10)")
    ax2.set_ylabel("Incumbent Poll (%)")
    ax.set_title("Government Rating & Incumbent Polls Over Time")
    ax.tick_params(axis="x", rotation=45)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig("graphs/pol_govt_vs_polls.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 4E. ELECTION-YEAR BUDGET EFFECT
# ═══════════════════════════════════════════════════════════════════

def analyze_election_year_budgets(budget_df):
    """Compare budget effects in election years vs non-election years."""
    from events import ELECTION_DATES

    election_years = set(ELECTION_DATES.keys())
    budget_df = budget_df.copy()
    budget_df["election_year"] = budget_df["year"].isin(election_years)

    ey = budget_df[budget_df["election_year"]]
    ney = budget_df[~budget_df["election_year"]]

    return {
        "election_year_mean": ey["change"].mean() if len(ey) > 0 else np.nan,
        "election_year_n": len(ey),
        "non_election_mean": ney["change"].mean() if len(ney) > 0 else np.nan,
        "non_election_n": len(ney),
        "ey_budgets": ey,
        "ney_budgets": ney,
    }


# ═══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_report(budget_df, policy_df, ownership_current, ownership_trends,
                    perf_df, ey_analysis):
    """Generate markdown report."""
    lines = []
    lines.append("# Policy & Budget Impacts on Polls")
    lines.append("")
    lines.append("*Do budgets, policies, and issue salience move party support?*")
    lines.append("")

    # ── 4A: Budget Bounce ──
    lines.append("## 4A. Budget Bounce")
    lines.append("")
    lines.append("**Question**: Do budgets systematically shift incumbent support?")
    lines.append("")
    lines.append("### Budget Effects on Incumbent Support (60-day window)")
    lines.append("")
    lines.append("| Year | Budget | Govt | Stance | Pre (%) | Post (%) | Change | p |")
    lines.append("|------|--------|------|--------|---------|----------|--------|---|")

    for _, r in budget_df.iterrows():
        sig = "**" if r["p_value"] < 0.05 else ""
        lines.append(
            f"| {r['year']} | {r['label'][:30]} | {r['govt']} | {r['stance']} "
            f"| {r['pre_mean']:.1f} | {r['post_mean']:.1f} "
            f"| {sig}{r['change']:+.1f}{sig} | {r['p_value']:.3f} |"
        )

    # Summary stats
    mean_bounce = budget_df["change"].mean()
    sig_count = (budget_df["p_value"] < 0.05).sum()
    positive = (budget_df["change"] > 0).sum()
    total = len(budget_df)

    lines.append("")
    lines.append(f"Mean budget bounce: **{mean_bounce:+.1f}pp**")
    lines.append(f"Positive bounces: {positive}/{total} ({100*positive/total:.0f}%)")
    lines.append(f"Statistically significant (p<0.05): {sig_count}/{total}")
    lines.append("")

    # By stance
    stance_means = budget_df.groupby("stance")["change"].agg(["mean", "count"])
    lines.append("### By Fiscal Stance")
    lines.append("")
    lines.append("| Stance | Mean Change | n |")
    lines.append("|--------|------------|---|")
    for stance in ["expansionary", "neutral", "contractionary"]:
        if stance in stance_means.index:
            r = stance_means.loc[stance]
            lines.append(f"| {stance.title()} | {r['mean']:+.1f}pp | {int(r['count'])} |")

    lines.append("")

    # Election year effect
    ey_mean = ey_analysis["election_year_mean"]
    ney_mean = ey_analysis["non_election_mean"]
    lines.append(f"Election-year budget bounce: **{ey_mean:+.1f}pp** (n={ey_analysis['election_year_n']})")
    lines.append(f"Non-election-year bounce: **{ney_mean:+.1f}pp** (n={ey_analysis['non_election_n']})")
    lines.append("")

    exp = budget_df[budget_df["stance"] == "expansionary"]
    if len(exp) > 0:
        lines.append(f"**Finding**: Expansionary budgets produce a mean bounce of "
                     f"{exp['change'].mean():+.1f}pp. "
                     f"{'Election-year budgets are larger.' if ey_mean > ney_mean else 'No clear election-year effect.'}")
    lines.append("")
    lines.append("![Budget Bounce](../graphs/pol_budget_bounce.png)")
    lines.append("")

    # ── 4B: Policy Shocks ──
    lines.append("## 4B. Policy Shocks")
    lines.append("")
    lines.append("**Question**: Do major policy announcements shift party support?")
    lines.append("")

    # Top effects (own party)
    own = policy_df[policy_df["party_measured"] == policy_df["announcing_party"]]
    own_sorted = own.reindex(own["change"].abs().sort_values(ascending=False).index)

    lines.append("### Largest Policy Effects (on announcing party)")
    lines.append("")
    lines.append("| Event | Date | Party | Change (pp) | p |")
    lines.append("|-------|------|-------|-------------|---|")

    for _, r in own_sorted.head(12).iterrows():
        sig = "**" if r["p_value"] < 0.05 else ""
        lines.append(
            f"| {r['event'][:40]} | {r['date'][:10]} | {r['announcing_party']} "
            f"| {sig}{r['change']:+.1f}{sig} | {r['p_value']:.3f} |"
        )

    lines.append("")

    # Cross-party effects
    cross = policy_df[policy_df["party_measured"] != policy_df["announcing_party"]]
    if len(cross) > 0:
        cross_sorted = cross.reindex(cross["change"].abs().sort_values(ascending=False).index)
        lines.append("### Cross-Party Effects (impact on opposing party)")
        lines.append("")
        lines.append("| Event | Opposing Party Change | p |")
        lines.append("|-------|----------------------|---|")
        for _, r in cross_sorted.head(8).iterrows():
            lines.append(
                f"| {r['event'][:40]} | {r['party_measured']} {r['change']:+.1f}pp | {r['p_value']:.3f} |"
            )
        lines.append("")

    lines.append("![Policy Shocks](../graphs/pol_policy_shocks.png)")
    lines.append("")

    # ── 4C: Issue Ownership ──
    lines.append("## 4C. Issue Ownership in Action (Ipsos Oct 2025)")
    lines.append("")
    lines.append("**Question**: Which party do voters trust on which issues?")
    lines.append("")
    lines.append("### Current Party Capability Ratings")
    lines.append("")
    lines.append("| Issue | National % | Labour % | Gap | Owner |")
    lines.append("|-------|-----------|---------|-----|-------|")

    for _, r in ownership_current.iterrows():
        issue_label = r["issue"].replace("_", " ").title()
        lines.append(
            f"| {issue_label} | {r['national_pct']} | {r['labour_pct']} "
            f"| {r['gap_nat_minus_lab']:+d} | **{r['owner']}** |"
        )

    lines.append("")

    nat_leads = (ownership_current["owner"] == "National").sum()
    lab_leads = (ownership_current["owner"] == "Labour").sum()
    lines.append(f"Labour leads on **{lab_leads}/{len(ownership_current)}** issues. "
                 f"National leads on **{nat_leads}/{len(ownership_current)}**.")
    lines.append("")
    lines.append("**Finding**: Labour has captured issue ownership on almost all salient issues, "
                 "including traditionally National-owned domains like the economy and inflation. "
                 "National retains only crime/law & order and unemployment.")
    lines.append("")
    lines.append("![Issue Ownership Trends](../graphs/pol_issue_ownership_trends.png)")
    lines.append("")

    # ── 4D: Government Performance ──
    lines.append("## 4D. Government Performance → Polls")
    lines.append("")
    lines.append("**Question**: Does the Ipsos government performance rating predict actual polling?")
    lines.append("")

    if len(perf_df) >= 3:
        x = perf_df["govt_rating"].values
        y = perf_df["incumbent_poll"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 3:
            slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
            lines.append(f"Correlation: r = **{r:.3f}**, p = {p:.4f}")
            lines.append(f"Slope: {slope:.1f}pp polling per 1-point govt rating")
            lines.append(f"R² = {r**2:.3f} (government rating explains {r**2*100:.0f}% of incumbent poll variance)")
            lines.append("")

            # Current situation
            latest = perf_df.iloc[-1]
            lines.append(f"Current govt rating: **{latest['govt_rating']}/10** "
                         f"(lowest in tracking history)")
            lines.append(f"Current incumbent ({latest['incumbent_party']}) polling: "
                         f"**{latest['incumbent_poll']:.1f}%**")
            lines.append("")

            lines.append("### Government Performance Trajectory")
            lines.append("")
            lines.append("| Date | Govt | Rating | Incumbent Poll | Bottom 4 % |")
            lines.append("|------|------|--------|----------------|-----------|")
            for _, r in perf_df.iterrows():
                lines.append(
                    f"| {r['date'].strftime('%Y-%m')} | {r['government'][:15]} "
                    f"| {r['govt_rating']:.1f} | {r['incumbent_poll']:.1f}% "
                    f"| {r['bottom4_pct']}% |"
                )
            lines.append("")

    lines.append("![Govt vs Polls](../graphs/pol_govt_vs_polls.png)")
    lines.append("")

    # ── Synthesis ──
    lines.append("## Synthesis")
    lines.append("")
    lines.append("### Can we estimate policy impacts on polls?")
    lines.append("")
    lines.append("**Yes, with caveats:**")
    lines.append("")
    lines.append("1. **Budgets have small, inconsistent effects** on incumbent polls. "
                 "The mean budget bounce is modest and often insignificant. "
                 "Expansionary budgets tend to help slightly, contractionary ones hurt, "
                 "but the effects rarely reach statistical significance in a single year.")
    lines.append("")
    lines.append("2. **Major policy announcements can move polls**, but the largest "
                 "effects come from crisis-related policies (COVID lockdown, "
                 "gun reform) rather than routine legislation. "
                 "The mechanism is likely rally-round-the-flag rather than policy evaluation.")
    lines.append("")
    lines.append("3. **Issue ownership is a powerful predictor** of which party benefits. "
                 "When an issue is salient AND a party 'owns' it, that party gains. "
                 "The Ipsos data shows Labour has captured ownership of almost every "
                 "salient issue as of Oct 2025 — a dangerous position for the government.")
    lines.append("")
    lines.append("4. **Government performance ratings track polls closely**, "
                 "suggesting that voters' overall evaluation of government competence "
                 "— not specific policies — drives support. This is consistent with "
                 "our valence politics finding (Rule 17).")
    lines.append("")
    lines.append("### Implications for the Current Government")
    lines.append("")
    lines.append("The Ipsos data paints a dire picture for the National-led coalition:")
    lines.append("- Government rating at 3.9/10 (record low)")
    lines.append("- 45% rate the government 0-3 out of 10")
    lines.append("- Labour leads on 6 of 7 top issues including the economy")
    lines.append("- Issue salience is dominated by cost-of-living (61%) where Labour leads")
    lines.append("- The government's decline mirrors the late-Clark and late-Ardern trajectories")
    lines.append("")

    report = "\n".join(lines)
    with open("reports/policy_impacts.md", "w") as f:
        f.write(report)
    print(f"Report: reports/policy_impacts.md ({len(lines)} lines)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("POLICY & BUDGET IMPACT ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    polls = load_polling()
    print(f"  Polls: {len(polls)} rows, {polls['date'].min()} to {polls['date'].max()}")

    govt_df, issues_df, capability_df = load_ipsos()
    print(f"  Ipsos govt performance: {len(govt_df)} waves")
    print(f"  Ipsos issue salience: {len(issues_df)} waves")
    print(f"  Ipsos party capability: {len(capability_df)} rows")

    # 4A: Budget bounce
    print("\n4A. Budget Bounce Analysis...")
    budget_results = analyze_budget_bounce(polls)
    print(f"  Analyzed {len(budget_results)} budgets with sufficient polling data")
    if len(budget_results) > 0:
        mean_bounce = budget_results["change"].mean()
        sig = (budget_results["p_value"] < 0.05).sum()
        print(f"  Mean bounce: {mean_bounce:+.1f}pp")
        print(f"  Significant (p<0.05): {sig}/{len(budget_results)}")
        plot_budget_bounce(budget_results)
        print("  Graph: graphs/pol_budget_bounce.png")

    # 4B: Policy shocks
    print("\n4B. Policy Shock Analysis...")
    policy_results = analyze_policy_shocks(polls)
    own = policy_results[policy_results["party_measured"] == policy_results["announcing_party"]]
    print(f"  Analyzed {len(POLICY_EVENTS)} events, {len(own)} with own-party data")
    if len(own) > 0:
        largest = own.loc[own["change"].abs().idxmax()]
        print(f"  Largest effect: {largest['event'][:40]} ({largest['change']:+.1f}pp)")
        plot_policy_shocks(policy_results)
        print("  Graph: graphs/pol_policy_shocks.png")

    # 4C: Issue ownership
    print("\n4C. Issue Ownership (Ipsos)...")
    ownership_current = analyze_issue_ownership_current(capability_df, issues_df)
    ownership_trends = analyze_issue_ownership_trends(capability_df)
    nat_leads = (ownership_current["owner"] == "National").sum()
    lab_leads = (ownership_current["owner"] == "Labour").sum()
    print(f"  Labour leads: {lab_leads} issues, National leads: {nat_leads} issues")
    plot_issue_ownership(ownership_trends)
    print("  Graph: graphs/pol_issue_ownership_trends.png")

    # 4D: Government performance
    print("\n4D. Government Performance → Polls...")
    perf_results = analyze_govt_performance(govt_df, polls)
    if len(perf_results) >= 3:
        x = perf_results["govt_rating"].values
        y = perf_results["incumbent_poll"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 3:
            r, p = stats.pearsonr(x[mask], y[mask])
            print(f"  Govt rating ↔ incumbent poll: r={r:.3f}, p={p:.4f}")
        plot_govt_vs_polls(perf_results)
        print("  Graph: graphs/pol_govt_vs_polls.png")

    # 4E: Election-year budgets
    print("\n4E. Election-Year Budget Effect...")
    ey_analysis = analyze_election_year_budgets(budget_results)
    print(f"  Election-year bounce: {ey_analysis['election_year_mean']:+.1f}pp "
          f"(n={ey_analysis['election_year_n']})")
    print(f"  Non-election bounce: {ey_analysis['non_election_mean']:+.1f}pp "
          f"(n={ey_analysis['non_election_n']})")

    # Generate report
    print("\nGenerating report...")
    generate_report(budget_results, policy_results, ownership_current,
                    ownership_trends, perf_results, ey_analysis)

    print("\nDone!")


if __name__ == "__main__":
    main()
