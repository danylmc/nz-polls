#!/usr/bin/env python3
"""
NZ Polling Data Analysis

Statistical analysis of polling data to find predictors and correlates of polling shifts.

Analyses performed:
1. Polling Volatility by Election Cycle
2. Mean Reversion (autocorrelation)
3. National-Labour Zero-Sum correlation
4. Third Party Squeeze
5. MMP Transition Effect
6. Momentum Effects
7. Economic Voting (with external data)
8. Event Studies (leadership changes, crises)
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf

from events import (
    ELECTION_DATES, INCUMBENTS, LEADERSHIP_CHANGES, CRISES, SCANDALS,
    get_all_events, days_to_election, get_election_cycle
)
from economic_scraper import load_economic_data


# Configuration
DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
GRAPHS_DIR = Path(__file__).parent / "graphs"


def load_all_polling_data() -> pd.DataFrame:
    """Load all polling data into a single DataFrame"""
    all_polls = []

    for year in sorted(ELECTION_DATES.keys()):
        filepath = DATA_DIR / f"{year}_polling.json"
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        for poll in data.get("polls", []):
            if not poll.get("date"):
                continue

            parties = poll.get("parties", {})
            if not parties:
                continue

            row = {
                "date": poll["date"],
                "election_year": year,
                "pollster": poll.get("pollster"),
                "sample_size": poll.get("sample_size"),
                "National": parties.get("National"),
                "Labour": parties.get("Labour"),
                "Green": parties.get("Green"),
                "ACT": parties.get("ACT"),
                "NZ First": parties.get("NZ First"),
                "Te Pāti Māori": parties.get("Te Pāti Māori"),
            }
            all_polls.append(row)

    df = pd.DataFrame(all_polls)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add days to election
    df["days_to_election"] = df.apply(
        lambda r: days_to_election(r["date"].strftime("%Y-%m-%d"), r["election_year"]),
        axis=1
    )

    # Add incumbent party
    df["incumbent"] = df["election_year"].map(INCUMBENTS)

    # Calculate minor party total
    df["minor_parties"] = df[["Green", "ACT", "NZ First", "Te Pāti Māori"]].sum(axis=1, skipna=True)

    # Calculate incumbent support
    df["incumbent_support"] = df.apply(
        lambda r: r["National"] if r["incumbent"] == "National" else r["Labour"],
        axis=1
    )

    return df


def load_pm_polling_data() -> pd.DataFrame:
    """Load all PM polling data"""
    all_polls = []

    for year in sorted(ELECTION_DATES.keys()):
        filepath = DATA_DIR / f"{year}_pm_polling.json"
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        for poll in data.get("polls", []):
            if not poll.get("date"):
                continue

            candidates = poll.get("candidates", {})
            if not candidates:
                continue

            row = {
                "date": poll["date"],
                "election_year": year,
                "pollster": poll.get("pollster"),
            }
            row.update(candidates)
            all_polls.append(row)

    if not all_polls:
        return pd.DataFrame()

    df = pd.DataFrame(all_polls)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================================
# ANALYSIS 1: Polling Volatility by Election Cycle
# ============================================================================

def analyze_volatility(df: pd.DataFrame) -> Dict:
    """Calculate polling volatility (std dev) by election cycle"""
    results = {
        "description": "Polling volatility (standard deviation) by election cycle",
        "by_cycle": {},
        "by_party": {},
    }

    parties = ["National", "Labour", "Green", "ACT", "NZ First"]

    for year in sorted(df["election_year"].unique()):
        cycle_df = df[df["election_year"] == year]

        cycle_stats = {}
        for party in parties:
            values = cycle_df[party].dropna()
            if len(values) > 2:
                cycle_stats[party] = {
                    "std": round(values.std(), 2),
                    "mean": round(values.mean(), 2),
                    "n": len(values),
                }
        results["by_cycle"][int(year)] = cycle_stats

    # Aggregate by party across all cycles
    for party in parties:
        values = df[party].dropna()
        if len(values) > 2:
            results["by_party"][party] = {
                "overall_std": round(values.std(), 2),
                "overall_mean": round(values.mean(), 2),
            }

    # Test if volatility increases near election day
    df_valid = df.dropna(subset=["days_to_election", "National", "Labour"])

    # Split into early (>90 days) vs late (<90 days)
    early = df_valid[df_valid["days_to_election"] > 90]
    late = df_valid[(df_valid["days_to_election"] <= 90) & (df_valid["days_to_election"] >= 0)]

    if len(early) > 10 and len(late) > 10:
        early_vol = early["National"].std()
        late_vol = late["National"].std()
        _, p_value = stats.levene(early["National"].dropna(), late["National"].dropna())

        results["election_proximity_effect"] = {
            "early_volatility_national": round(early_vol, 2),
            "late_volatility_national": round(late_vol, 2),
            "levene_p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "interpretation": "Higher volatility near election" if late_vol > early_vol else "Lower volatility near election"
        }

    return results


# ============================================================================
# ANALYSIS 2: Mean Reversion (Autocorrelation)
# ============================================================================

def analyze_mean_reversion(df: pd.DataFrame) -> Dict:
    """Analyze autocorrelation and mean reversion in polling"""
    results = {
        "description": "Mean reversion analysis - do extreme polls regress to mean?",
        "autocorrelation": {},
        "extreme_poll_reversion": {},
    }

    for party in ["National", "Labour"]:
        # Sort by date and calculate changes
        party_data = df[["date", party]].dropna().sort_values("date")
        values = party_data[party].values

        if len(values) < 20:
            continue

        # Calculate autocorrelation
        try:
            ac = acf(values, nlags=5, fft=True)
            results["autocorrelation"][party] = {
                "lag1": round(float(ac[1]), 3),
                "lag2": round(float(ac[2]), 3),
                "lag3": round(float(ac[3]), 3),
            }
        except Exception:
            pass

        # Analyze extreme poll reversion
        mean_val = values.mean()
        std_val = values.std()

        # Find polls > 1.5 std from mean
        extreme_high = np.where(values > mean_val + 1.5 * std_val)[0]
        extreme_low = np.where(values < mean_val - 1.5 * std_val)[0]

        reversions = []
        for idx in list(extreme_high) + list(extreme_low):
            if idx + 1 < len(values):
                current = values[idx]
                next_val = values[idx + 1]
                deviation = current - mean_val
                next_deviation = next_val - mean_val
                # Reversion = moving back toward mean
                reverted = (deviation > 0 and next_deviation < deviation) or (deviation < 0 and next_deviation > deviation)
                reversions.append(reverted)

        if reversions:
            reversion_rate = sum(reversions) / len(reversions)
            results["extreme_poll_reversion"][party] = {
                "n_extreme_polls": len(reversions),
                "reversion_rate": round(reversion_rate, 3),
                "interpretation": "Mean reversion observed" if reversion_rate > 0.5 else "No clear mean reversion"
            }

    return results


# ============================================================================
# ANALYSIS 3: National-Labour Zero-Sum Correlation
# ============================================================================

def analyze_zero_sum(df: pd.DataFrame) -> Dict:
    """Test if National and Labour are negatively correlated (zero-sum)"""
    results = {
        "description": "National-Labour zero-sum hypothesis - are they negatively correlated?",
    }

    # Level correlation
    df_valid = df.dropna(subset=["National", "Labour"])
    if len(df_valid) > 10:
        corr, p_val = pearsonr(df_valid["National"], df_valid["Labour"])
        results["level_correlation"] = {
            "pearson_r": round(corr, 3),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "interpretation": "Strong negative correlation (zero-sum)" if corr < -0.5 else "Weak/no zero-sum relationship"
        }

    # Change correlation (more meaningful)
    df_sorted = df.sort_values("date").dropna(subset=["National", "Labour"])
    df_sorted["National_change"] = df_sorted["National"].diff()
    df_sorted["Labour_change"] = df_sorted["Labour"].diff()

    changes = df_sorted.dropna(subset=["National_change", "Labour_change"])
    if len(changes) > 10:
        corr, p_val = pearsonr(changes["National_change"], changes["Labour_change"])
        results["change_correlation"] = {
            "pearson_r": round(corr, 3),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "interpretation": "Changes are inversely related" if corr < -0.3 else "Changes not clearly inversely related"
        }

    return results


# ============================================================================
# ANALYSIS 4: Third Party Squeeze
# ============================================================================

def analyze_third_party_squeeze(df: pd.DataFrame) -> Dict:
    """Test if minor parties lose support as election approaches"""
    results = {
        "description": "Third party squeeze - do minor parties lose support near elections?",
    }

    df_valid = df.dropna(subset=["days_to_election", "minor_parties"])
    df_valid = df_valid[df_valid["days_to_election"] >= 0]

    if len(df_valid) < 20:
        return results

    # Regression: minor_party_support ~ days_to_election
    X = sm.add_constant(df_valid["days_to_election"])
    y = df_valid["minor_parties"]

    try:
        model = sm.OLS(y, X).fit()

        results["regression"] = {
            "coefficient": round(model.params["days_to_election"], 4),
            "p_value": round(model.pvalues["days_to_election"], 4),
            "r_squared": round(model.rsquared, 3),
            "significant": model.pvalues["days_to_election"] < 0.05,
        }

        # Positive coefficient = more support when further from election
        if model.params["days_to_election"] > 0 and model.pvalues["days_to_election"] < 0.05:
            results["interpretation"] = "Third party squeeze confirmed - minor parties lose support as election approaches"
        else:
            results["interpretation"] = "No significant third party squeeze detected"

    except Exception as e:
        results["error"] = str(e)

    # Compare support >90 days vs <30 days
    early = df_valid[df_valid["days_to_election"] > 90]["minor_parties"]
    late = df_valid[df_valid["days_to_election"] < 30]["minor_parties"]

    if len(early) > 5 and len(late) > 5:
        t_stat, p_val = stats.ttest_ind(early, late)
        results["early_vs_late"] = {
            "early_mean": round(early.mean(), 2),
            "late_mean": round(late.mean(), 2),
            "difference": round(early.mean() - late.mean(), 2),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_val, 4),
        }

    return results


# ============================================================================
# ANALYSIS 5: MMP Transition Effect
# ============================================================================

def analyze_mmp_effect(df: pd.DataFrame) -> Dict:
    """Compare polling patterns pre-1996 (FPP) vs post-1996 (MMP)"""
    results = {
        "description": "MMP transition effect - compare variance before and after 1996",
    }

    # Pre-MMP: 1993 only in our data
    pre_mmp = df[df["election_year"] == 1993]
    post_mmp = df[df["election_year"] >= 1996]

    if len(pre_mmp) < 5 or len(post_mmp) < 20:
        results["note"] = "Insufficient pre-MMP data for robust comparison"
        return results

    for party in ["National", "Labour"]:
        pre_vals = pre_mmp[party].dropna()
        post_vals = post_mmp[party].dropna()

        if len(pre_vals) > 2 and len(post_vals) > 10:
            # Levene's test for equality of variances
            stat, p_val = stats.levene(pre_vals, post_vals)

            results[f"{party.lower()}_variance"] = {
                "pre_mmp_std": round(pre_vals.std(), 2),
                "post_mmp_std": round(post_vals.std(), 2),
                "levene_stat": round(stat, 3),
                "p_value": round(p_val, 4),
                "significant_difference": p_val < 0.05,
            }

    # Minor party emergence under MMP
    post_mmp_minor = post_mmp["minor_parties"].dropna()
    if len(post_mmp_minor) > 10:
        results["minor_party_growth"] = {
            "mean_minor_support": round(post_mmp_minor.mean(), 2),
            "std_minor_support": round(post_mmp_minor.std(), 2),
            "note": "MMP enabled significant minor party representation"
        }

    return results


# ============================================================================
# ANALYSIS 6: Momentum Effects
# ============================================================================

def analyze_momentum(df: pd.DataFrame) -> Dict:
    """Test if polling movement predicts future movement"""
    results = {
        "description": "Momentum effects - does movement predict future movement?",
    }

    for party in ["National", "Labour"]:
        df_sorted = df[["date", party]].dropna().sort_values("date")
        df_sorted["change"] = df_sorted[party].diff()
        df_sorted["prev_change"] = df_sorted["change"].shift(1)

        valid = df_sorted.dropna(subset=["change", "prev_change"])

        if len(valid) < 20:
            continue

        # Correlation between consecutive changes
        corr, p_val = pearsonr(valid["prev_change"], valid["change"])

        results[party] = {
            "change_autocorrelation": round(corr, 3),
            "p_value": round(p_val, 4),
            "significant": p_val < 0.05,
        }

        # Interpretation
        if corr > 0.1 and p_val < 0.05:
            results[party]["interpretation"] = "Momentum effect - gains predict more gains"
        elif corr < -0.1 and p_val < 0.05:
            results[party]["interpretation"] = "Reversal effect - gains predict losses"
        else:
            results[party]["interpretation"] = "No significant momentum pattern"

    return results


# ============================================================================
# ANALYSIS 7: Economic Voting
# ============================================================================

def analyze_economic_voting(df: pd.DataFrame) -> Dict:
    """Regression: incumbent support ~ economic indicators"""
    results = {
        "description": "Economic voting - do economic conditions affect incumbent support?",
    }

    econ_data = load_economic_data()
    if not econ_data:
        results["error"] = "No economic data available"
        return results

    econ_df = pd.DataFrame(econ_data)

    # Aggregate polling by year
    yearly = df.groupby("election_year").agg({
        "incumbent_support": "mean",
        "incumbent": "first",
    }).reset_index()

    # Merge with economic data
    merged = yearly.merge(econ_df, left_on="election_year", right_on="year", how="inner")

    if len(merged) < 5:
        results["note"] = "Insufficient overlapping data for regression"
        return results

    # Simple correlations
    for econ_var in ["gdp_growth", "unemployment_rate", "cpi_inflation"]:
        if econ_var in merged.columns:
            valid = merged.dropna(subset=["incumbent_support", econ_var])
            if len(valid) > 3:
                corr, p_val = pearsonr(valid["incumbent_support"], valid[econ_var])
                results[f"{econ_var}_correlation"] = {
                    "pearson_r": round(corr, 3),
                    "p_value": round(p_val, 4),
                    "n": len(valid),
                }

    # Multiple regression if we have enough data
    valid_cols = ["incumbent_support", "gdp_growth", "unemployment_rate", "cpi_inflation"]
    valid_merged = merged.dropna(subset=valid_cols)

    if len(valid_merged) >= 5:
        X = valid_merged[["gdp_growth", "unemployment_rate", "cpi_inflation"]]
        X = sm.add_constant(X)
        y = valid_merged["incumbent_support"]

        try:
            model = sm.OLS(y, X).fit()
            results["multiple_regression"] = {
                "r_squared": round(model.rsquared, 3),
                "gdp_coef": round(model.params.get("gdp_growth", 0), 3),
                "gdp_pval": round(model.pvalues.get("gdp_growth", 1), 4),
                "unemployment_coef": round(model.params.get("unemployment_rate", 0), 3),
                "unemployment_pval": round(model.pvalues.get("unemployment_rate", 1), 4),
                "inflation_coef": round(model.params.get("cpi_inflation", 0), 3),
                "inflation_pval": round(model.pvalues.get("cpi_inflation", 1), 4),
            }
        except Exception as e:
            results["regression_error"] = str(e)

    return results


# ============================================================================
# ANALYSIS 8: Event Studies
# ============================================================================

def analyze_events(df: pd.DataFrame) -> Dict:
    """Measure polling changes around key events"""
    results = {
        "description": "Event studies - polling changes around leadership changes, crises, scandals",
        "leadership_changes": [],
        "crises": [],
        "scandals": [],
    }

    def analyze_event(event: Dict, window_days: int = 30) -> Optional[Dict]:
        """Analyze polling before and after an event"""
        event_date = datetime.strptime(event["date"], "%Y-%m-%d")

        before = df[(df["date"] >= event_date - pd.Timedelta(days=window_days)) &
                    (df["date"] < event_date)]
        after = df[(df["date"] > event_date) &
                   (df["date"] <= event_date + pd.Timedelta(days=window_days))]

        if len(before) < 2 or len(after) < 2:
            return None

        result = {
            "date": event["date"],
            "event": event["event"],
            "polls_before": len(before),
            "polls_after": len(after),
        }

        # Analyze party affected (if specified)
        party = event.get("party")
        if party and party in ["National", "Labour"]:
            before_mean = before[party].mean()
            after_mean = after[party].mean()

            if pd.notna(before_mean) and pd.notna(after_mean):
                change = after_mean - before_mean
                result[f"{party.lower()}_before"] = round(before_mean, 1)
                result[f"{party.lower()}_after"] = round(after_mean, 1)
                result[f"{party.lower()}_change"] = round(change, 1)

        # Also check both major parties for all events
        for p in ["National", "Labour"]:
            if f"{p.lower()}_change" not in result:
                before_mean = before[p].mean()
                after_mean = after[p].mean()
                if pd.notna(before_mean) and pd.notna(after_mean):
                    result[f"{p.lower()}_before"] = round(before_mean, 1)
                    result[f"{p.lower()}_after"] = round(after_mean, 1)
                    result[f"{p.lower()}_change"] = round(after_mean - before_mean, 1)

        return result

    # Analyze leadership changes
    for event in LEADERSHIP_CHANGES:
        analysis = analyze_event(event)
        if analysis:
            results["leadership_changes"].append(analysis)

    # Analyze crises
    for event in CRISES:
        analysis = analyze_event(event)
        if analysis:
            results["crises"].append(analysis)

    # Analyze scandals
    for event in SCANDALS:
        analysis = analyze_event(event)
        if analysis:
            results["scandals"].append(analysis)

    # Summarize findings
    leadership_effects = [e for e in results["leadership_changes"] if "labour_change" in e or "national_change" in e]
    if leadership_effects:
        party_changes = []
        for e in leadership_effects:
            party = "Labour" if "labour_change" in e else "National"
            change = e.get(f"{party.lower()}_change", 0)
            party_changes.append(change)
        results["leadership_summary"] = {
            "mean_change": round(np.mean(party_changes), 2) if party_changes else None,
            "n_events": len(leadership_effects),
        }

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_all_analyses() -> Dict:
    """Run all analyses and return results"""
    print("Loading polling data...")
    df = load_all_polling_data()
    print(f"Loaded {len(df)} polls from {df['election_year'].nunique()} election cycles")

    results = {
        "metadata": {
            "total_polls": len(df),
            "election_cycles": sorted(df["election_year"].unique().tolist()),
            "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "analyses": {}
    }

    print("\n1. Analyzing volatility...")
    results["analyses"]["volatility"] = analyze_volatility(df)

    print("2. Analyzing mean reversion...")
    results["analyses"]["mean_reversion"] = analyze_mean_reversion(df)

    print("3. Analyzing National-Labour zero-sum...")
    results["analyses"]["zero_sum"] = analyze_zero_sum(df)

    print("4. Analyzing third party squeeze...")
    results["analyses"]["third_party_squeeze"] = analyze_third_party_squeeze(df)

    print("5. Analyzing MMP effect...")
    results["analyses"]["mmp_effect"] = analyze_mmp_effect(df)

    print("6. Analyzing momentum effects...")
    results["analyses"]["momentum"] = analyze_momentum(df)

    print("7. Analyzing economic voting...")
    results["analyses"]["economic_voting"] = analyze_economic_voting(df)

    print("8. Analyzing events...")
    results["analyses"]["events"] = analyze_events(df)

    return results


def generate_report(results: Dict) -> str:
    """Generate markdown report from analysis results"""
    report = []
    report.append("# NZ Polling Analysis: Findings Report")
    report.append("")
    report.append(f"**Analysis Date:** {results['metadata']['analysis_timestamp'][:10]}")
    report.append(f"**Total Polls Analyzed:** {results['metadata']['total_polls']}")
    report.append(f"**Date Range:** {results['metadata']['date_range']}")
    report.append(f"**Election Cycles:** {', '.join(map(str, results['metadata']['election_cycles']))}")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")

    analyses = results["analyses"]

    # Zero-sum finding
    zs = analyses.get("zero_sum", {})
    if "change_correlation" in zs:
        r = zs["change_correlation"]["pearson_r"]
        p = zs["change_correlation"]["p_value"]
        report.append(f"- **National-Labour Zero-Sum:** {'Confirmed' if r < -0.3 and p < 0.05 else 'Not confirmed'} (r = {r}, p = {p:.4f})")

    # Third party squeeze
    tps = analyses.get("third_party_squeeze", {})
    if "regression" in tps:
        sig = tps["regression"]["significant"]
        report.append(f"- **Third Party Squeeze:** {'Confirmed' if sig else 'Not detected'}")

    # Mean reversion
    mr = analyses.get("mean_reversion", {})
    if "extreme_poll_reversion" in mr:
        for party, data in mr["extreme_poll_reversion"].items():
            report.append(f"- **Mean Reversion ({party}):** {data['reversion_rate']*100:.0f}% of extreme polls revert")

    report.append("")

    # Detailed Findings
    report.append("## Detailed Findings")
    report.append("")

    # 1. Volatility
    report.append("### 1. Polling Volatility by Election Cycle")
    report.append("")
    vol = analyses.get("volatility", {})
    if "by_party" in vol:
        report.append("**Overall Volatility (Standard Deviation):**")
        report.append("")
        report.append("| Party | Mean | Std Dev |")
        report.append("|-------|------|---------|")
        for party, stats in vol["by_party"].items():
            report.append(f"| {party} | {stats['overall_mean']:.1f}% | {stats['overall_std']:.1f}% |")
        report.append("")

    if "election_proximity_effect" in vol:
        epe = vol["election_proximity_effect"]
        report.append(f"**Election Proximity Effect:** {epe['interpretation']}")
        report.append(f"- Early campaign volatility: {epe['early_volatility_national']:.1f}%")
        report.append(f"- Late campaign volatility: {epe['late_volatility_national']:.1f}%")
        report.append(f"- Statistical significance: p = {epe['levene_p_value']:.4f}")
        report.append("")

    # 2. Mean Reversion
    report.append("### 2. Mean Reversion Analysis")
    report.append("")
    mr = analyses.get("mean_reversion", {})
    if "autocorrelation" in mr:
        report.append("**Autocorrelation (persistence of polling levels):**")
        report.append("")
        for party, ac in mr["autocorrelation"].items():
            report.append(f"- {party}: Lag-1 = {ac['lag1']:.2f}, Lag-2 = {ac['lag2']:.2f}")
        report.append("")
        report.append("*Positive autocorrelation indicates polling levels persist over time.*")
        report.append("")

    if "extreme_poll_reversion" in mr:
        report.append("**Extreme Poll Reversion:**")
        for party, data in mr["extreme_poll_reversion"].items():
            report.append(f"- {party}: {data['reversion_rate']*100:.0f}% of outlier polls regress toward mean ({data['n_extreme_polls']} extreme polls)")
        report.append("")

    # 3. Zero-Sum
    report.append("### 3. National-Labour Zero-Sum Relationship")
    report.append("")
    zs = analyses.get("zero_sum", {})
    if "level_correlation" in zs:
        lc = zs["level_correlation"]
        report.append(f"**Level Correlation:** r = {lc['pearson_r']:.3f} (p = {lc['p_value']:.6f})")
    if "change_correlation" in zs:
        cc = zs["change_correlation"]
        report.append(f"**Change Correlation:** r = {cc['pearson_r']:.3f} (p = {cc['p_value']:.6f})")
        report.append("")
        report.append(f"*Interpretation:* {cc['interpretation']}")
    report.append("")

    # 4. Third Party Squeeze
    report.append("### 4. Third Party Squeeze Effect")
    report.append("")
    tps = analyses.get("third_party_squeeze", {})
    if "regression" in tps:
        reg = tps["regression"]
        report.append(f"**Regression: Minor Party Support ~ Days to Election**")
        report.append(f"- Coefficient: {reg['coefficient']:.4f} (per day)")
        report.append(f"- p-value: {reg['p_value']:.4f}")
        report.append(f"- R-squared: {reg['r_squared']:.3f}")
        report.append("")

    if "early_vs_late" in tps:
        evl = tps["early_vs_late"]
        report.append(f"**Early vs Late Campaign:**")
        report.append(f"- Minor party support (>90 days out): {evl['early_mean']:.1f}%")
        report.append(f"- Minor party support (<30 days out): {evl['late_mean']:.1f}%")
        report.append(f"- Difference: {evl['difference']:.1f}% (p = {evl['p_value']:.4f})")
    report.append("")

    if "interpretation" in tps:
        report.append(f"*Interpretation:* {tps['interpretation']}")
        report.append("")

    # 5. MMP Effect
    report.append("### 5. MMP Transition Effect")
    report.append("")
    mmp = analyses.get("mmp_effect", {})
    for party in ["national", "labour"]:
        key = f"{party}_variance"
        if key in mmp:
            data = mmp[key]
            report.append(f"**{party.title()}:** Pre-MMP std = {data['pre_mmp_std']:.1f}%, Post-MMP std = {data['post_mmp_std']:.1f}%")

    if "minor_party_growth" in mmp:
        mpg = mmp["minor_party_growth"]
        report.append(f"")
        report.append(f"**Minor Party Growth under MMP:** Mean combined support = {mpg['mean_minor_support']:.1f}%")
    report.append("")

    # 6. Momentum
    report.append("### 6. Momentum Effects")
    report.append("")
    mom = analyses.get("momentum", {})
    for party in ["National", "Labour"]:
        if party in mom:
            data = mom[party]
            report.append(f"**{party}:** Change autocorrelation = {data['change_autocorrelation']:.3f} (p = {data['p_value']:.4f})")
            report.append(f"- {data['interpretation']}")
    report.append("")

    # 7. Economic Voting
    report.append("### 7. Economic Voting")
    report.append("")
    ev = analyses.get("economic_voting", {})
    if "gdp_growth_correlation" in ev:
        report.append("**Correlations with Incumbent Support:**")
        report.append("")
        for var in ["gdp_growth", "unemployment_rate", "cpi_inflation"]:
            key = f"{var}_correlation"
            if key in ev:
                data = ev[key]
                sig = "**" if data["p_value"] < 0.05 else ""
                report.append(f"- {var.replace('_', ' ').title()}: r = {sig}{data['pearson_r']:.3f}{sig} (p = {data['p_value']:.4f}, n = {data['n']})")
        report.append("")

    if "multiple_regression" in ev:
        mr = ev["multiple_regression"]
        report.append(f"**Multiple Regression (R² = {mr['r_squared']:.3f}):**")
        report.append(f"- GDP Growth: coef = {mr['gdp_coef']:.3f}, p = {mr['gdp_pval']:.4f}")
        report.append(f"- Unemployment: coef = {mr['unemployment_coef']:.3f}, p = {mr['unemployment_pval']:.4f}")
        report.append(f"- Inflation: coef = {mr['inflation_coef']:.3f}, p = {mr['inflation_pval']:.4f}")
    report.append("")

    # 8. Event Studies
    report.append("### 8. Event Studies")
    report.append("")
    events = analyses.get("events", {})

    if events.get("leadership_changes"):
        report.append("**Leadership Changes:**")
        report.append("")
        report.append("| Date | Event | National Change | Labour Change |")
        report.append("|------|-------|-----------------|---------------|")
        for e in events["leadership_changes"]:
            nat = e.get("national_change", "N/A")
            lab = e.get("labour_change", "N/A")
            nat_str = f"{nat:+.1f}%" if isinstance(nat, (int, float)) else nat
            lab_str = f"{lab:+.1f}%" if isinstance(lab, (int, float)) else lab
            report.append(f"| {e['date']} | {e['event'][:35]} | {nat_str} | {lab_str} |")
        report.append("")

    if events.get("crises"):
        report.append("**Crises:**")
        report.append("")
        for e in events["crises"]:
            nat = e.get("national_change")
            lab = e.get("labour_change")
            if nat is not None or lab is not None:
                changes = []
                if nat is not None:
                    changes.append(f"National {nat:+.1f}%")
                if lab is not None:
                    changes.append(f"Labour {lab:+.1f}%")
                report.append(f"- **{e['event']}** ({e['date']}): {', '.join(changes)}")
        report.append("")

    # Comparison with Literature
    report.append("## Comparison with Political Science Literature")
    report.append("")
    report.append("### Expected vs. Observed Findings")
    report.append("")
    report.append("| Hypothesis | Expected | Observed | Confirmed? |")
    report.append("|------------|----------|----------|------------|")

    # Zero-sum
    zs_confirmed = "change_correlation" in zs and zs["change_correlation"]["pearson_r"] < -0.3
    report.append(f"| National-Labour negative correlation | Strong negative | r = {zs.get('change_correlation', {}).get('pearson_r', 'N/A')} | {'Yes' if zs_confirmed else 'Partial'} |")

    # Third party squeeze
    tps_confirmed = tps.get("regression", {}).get("significant", False)
    report.append(f"| Third party squeeze near elections | Support declines | Coef = {tps.get('regression', {}).get('coefficient', 'N/A')} | {'Yes' if tps_confirmed else 'No'} |")

    # Mean reversion
    mr_rate = mr.get("extreme_poll_reversion", {}).get("National", {}).get("reversion_rate")
    mr_confirmed = mr_rate is not None and mr_rate > 0.5
    report.append(f"| Mean reversion of outliers | Outliers regress | {mr_rate*100 if mr_rate else 'N/A'}% revert | {'Yes' if mr_confirmed else 'Unclear'} |")

    report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*Report generated by analysis.py*")

    return "\n".join(report)


def main():
    """Main function"""
    print("NZ Polling Data Analysis")
    print("=" * 50)

    # Run analyses
    results = run_all_analyses()

    # Save raw results
    REPORTS_DIR.mkdir(exist_ok=True)
    results_path = REPORTS_DIR / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # Generate and save report
    print("\nGenerating report...")
    report = generate_report(results)
    report_path = REPORTS_DIR / "findings.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)

    analyses = results["analyses"]

    print("\n1. Zero-Sum Hypothesis:")
    if "change_correlation" in analyses.get("zero_sum", {}):
        cc = analyses["zero_sum"]["change_correlation"]
        print(f"   National-Labour change correlation: r = {cc['pearson_r']:.3f} (p = {cc['p_value']:.4f})")

    print("\n2. Third Party Squeeze:")
    if "regression" in analyses.get("third_party_squeeze", {}):
        tps = analyses["third_party_squeeze"]["regression"]
        print(f"   Days-to-election coefficient: {tps['coefficient']:.4f} (p = {tps['p_value']:.4f})")

    print("\n3. Mean Reversion:")
    for party, data in analyses.get("mean_reversion", {}).get("extreme_poll_reversion", {}).items():
        print(f"   {party}: {data['reversion_rate']*100:.0f}% of outliers revert")

    print("\n4. Event Effects:")
    events = analyses.get("events", {})
    if events.get("leadership_changes"):
        print(f"   Leadership changes analyzed: {len(events['leadership_changes'])}")
    if events.get("crises"):
        print(f"   Crises analyzed: {len(events['crises'])}")


if __name__ == "__main__":
    main()
