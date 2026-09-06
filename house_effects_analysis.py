#!/usr/bin/env python3
"""
Pollster House Effects Analysis.

Calculates systematic bias for each pollster relative to election results.
Tests whether NZ pollsters systematically favour certain parties (Jackman 2005).
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Actual election results (party vote %)
# Source: NZ Electoral Commission
ELECTION_RESULTS = {
    1996: {"National": 33.84, "Labour": 28.19, "NZ First": 13.35, "ACT": 6.10, "Green": None},
    1999: {"National": 30.50, "Labour": 38.74, "NZ First": 4.26, "ACT": 7.04, "Green": 5.16},
    2002: {"National": 20.93, "Labour": 41.26, "NZ First": 10.38, "ACT": 7.14, "Green": 7.00},
    2005: {"National": 39.10, "Labour": 41.10, "NZ First": 5.72, "ACT": 1.51, "Green": 5.30},
    2008: {"National": 44.93, "Labour": 33.99, "NZ First": 4.07, "ACT": 3.65, "Green": 6.72},
    2011: {"National": 47.31, "Labour": 27.48, "NZ First": 6.59, "ACT": 1.07, "Green": 11.06},
    2014: {"National": 47.04, "Labour": 25.13, "NZ First": 8.66, "ACT": 0.69, "Green": 10.70},
    2017: {"National": 44.45, "Labour": 36.89, "NZ First": 7.20, "ACT": 0.50, "Green": 6.27},
    2020: {"National": 25.58, "Labour": 50.01, "NZ First": 2.60, "ACT": 7.58, "Green": 7.86},
    2023: {"National": 38.93, "Labour": 26.91, "NZ First": 6.08, "ACT": 8.64, "Green": 11.60},
}


def load_all_polls():
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            year = data.get("election_year")
            for p in data.get("polls", []):
                row = {
                    "date": p["date"],
                    "election_year": year,
                    "pollster": p.get("pollster"),
                }
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def main():
    polls = load_all_polls()
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]

    # Ensure numeric
    for p in parties:
        polls[p] = pd.to_numeric(polls[p], errors="coerce")

    # ── 1. House effects: pollster bias relative to election result ────
    # For each poll in the final 2 weeks, calculate error vs result
    print("=" * 70)
    print("POLLSTER HOUSE EFFECTS ANALYSIS")
    print("=" * 70)

    # First, calculate final poll errors for each election
    final_errors = []
    for year, results in sorted(ELECTION_RESULTS.items()):
        election_date = pd.to_datetime(f"{year}-{'11-06' if year == 1993 else '10-12' if year == 1996 else '01-01'}")
        # Get actual election date from events
        from events import ELECTION_DATES
        edate_str = ELECTION_DATES.get(year)
        if not edate_str:
            continue
        edate = pd.to_datetime(edate_str)

        # Get polls from this election cycle, final 60 days (broader for house effects)
        cycle_polls = polls[polls["election_year"] == year].copy()
        cycle_polls["days_to_election"] = (edate - cycle_polls["date"]).dt.days
        final_polls = cycle_polls[(cycle_polls["days_to_election"] >= 0) &
                                   (cycle_polls["days_to_election"] <= 60)]

        for _, poll in final_polls.iterrows():
            pollster = poll.get("pollster")
            if not pollster or pd.isna(pollster):
                pollster = "Unknown"
            for party in parties:
                actual = results.get(party)
                polled = poll.get(party)
                if actual is not None and pd.notna(polled):
                    final_errors.append({
                        "year": year,
                        "pollster": pollster,
                        "party": party,
                        "polled": polled,
                        "actual": actual,
                        "error": polled - actual,
                        "days_out": poll["days_to_election"],
                    })

    errors_df = pd.DataFrame(final_errors)
    print(f"\nTotal poll-party observations (final 60 days): {len(errors_df)}")
    print(f"Elections covered: {sorted(errors_df['year'].unique())}")

    # ── 2. Average error by party (systematic bias) ───────────────────
    print(f"\n--- SYSTEMATIC BIAS BY PARTY (mean error = polled - actual) ---")
    print(f"  Positive = pollsters overestimate; Negative = underestimate")
    print(f"\n  {'Party':>12s}  {'Mean Error':>10s}  {'Std':>7s}  {'t':>7s}  {'p':>8s}  {'n':>5s}")
    for party in parties:
        sub = errors_df[errors_df["party"] == party]["error"].dropna()
        if len(sub) > 5:
            mean_err = sub.mean()
            std_err = sub.std()
            t_stat, p_val = ttest_1samp(sub, 0)
            print(f"  {party:>12s}  {mean_err:>+10.2f}  {std_err:>7.2f}  {t_stat:>7.2f}  {p_val:>8.4f}  {len(sub):>5d}")

    # ── 3. House effects by pollster ──────────────────────────────────
    print(f"\n--- HOUSE EFFECTS BY POLLSTER (National bias, final 60 days) ---")
    print(f"  Positive = overestimates National; Negative = underestimates National")
    pollster_counts = errors_df[errors_df["party"] == "National"].groupby("pollster").size()
    active_pollsters = pollster_counts[pollster_counts >= 5].index

    house_effects = []
    for pollster in sorted(active_pollsters):
        sub = errors_df[(errors_df["pollster"] == pollster)]
        for party in ["National", "Labour"]:
            party_sub = sub[sub["party"] == party]["error"]
            if len(party_sub) >= 3:
                house_effects.append({
                    "pollster": pollster,
                    "party": party,
                    "mean_error": party_sub.mean(),
                    "std_error": party_sub.std(),
                    "n": len(party_sub),
                    "elections": sub["year"].nunique(),
                })

    he_df = pd.DataFrame(house_effects)
    print(f"\n  {'Pollster':<35s}  {'Party':>10s}  {'Bias':>7s}  {'Std':>7s}  {'n':>4s}  {'Elections':>9s}")
    for _, row in he_df.sort_values("mean_error", ascending=False).iterrows():
        print(f"  {row['pollster']:<35s}  {row['party']:>10s}  {row['mean_error']:>+7.2f}  {row['std_error']:>7.2f}  {row['n']:>4.0f}  {row['elections']:>9.0f}")

    # ── 4. Final 2-week polls vs results (poll accuracy) ──────────────
    print(f"\n--- FINAL POLL ACCURACY (last 14 days before election) ---")
    final_14d = errors_df[errors_df["days_out"] <= 14]
    print(f"  Observations: {len(final_14d)}")

    print(f"\n  {'Party':>12s}  {'Mean Error':>10s}  {'MAE':>7s}  {'RMSE':>7s}  {'n':>5s}")
    for party in parties:
        sub = final_14d[final_14d["party"] == party]["error"].dropna()
        if len(sub) > 3:
            mae = sub.abs().mean()
            rmse = np.sqrt((sub**2).mean())
            print(f"  {party:>12s}  {sub.mean():>+10.2f}  {mae:>7.2f}  {rmse:>7.2f}  {len(sub):>5d}")

    # ── 5. Shy Tory effect: does National systematically outperform? ──
    print(f"\n--- SHY TORY TEST (final 14 days) ---")
    nat_errors = final_14d[final_14d["party"] == "National"]["error"].dropna()
    lab_errors = final_14d[final_14d["party"] == "Labour"]["error"].dropna()
    if len(nat_errors) > 5:
        t, p = ttest_1samp(nat_errors, 0)
        print(f"  National: mean error = {nat_errors.mean():+.2f}pp (t={t:.2f}, p={p:.4f})")
        print(f"    → {'Polls underestimate National (shy Tory)' if nat_errors.mean() < 0 else 'Polls overestimate National'}")
    if len(lab_errors) > 5:
        t, p = ttest_1samp(lab_errors, 0)
        print(f"  Labour:   mean error = {lab_errors.mean():+.2f}pp (t={t:.2f}, p={p:.4f})")

    # ── 6. Convergence: do polls get more accurate closer to election? ─
    print(f"\n--- CONVERGENCE: Error vs Days Out ---")
    for party in ["National", "Labour"]:
        sub = errors_df[errors_df["party"] == party].copy()
        sub["abs_error"] = sub["error"].abs()
        if len(sub) > 20:
            from scipy.stats import pearsonr
            r, p = pearsonr(sub["days_out"], sub["abs_error"])
            print(f"  {party}: correlation(days_out, |error|) = {r:.3f}, p = {p:.4f}")
            print(f"    → {'Polls converge toward result' if r > 0 else 'No convergence detected'}")

    # ── 7. By-election accuracy ───────────────────────────────────────
    print(f"\n--- ACCURACY BY ELECTION (final 14 days, National) ---")
    for year in sorted(ELECTION_RESULTS.keys()):
        sub = final_14d[(final_14d["year"] == year) & (final_14d["party"] == "National")]
        if len(sub) > 0:
            actual = ELECTION_RESULTS[year]["National"]
            mean_polled = sub["polled"].mean()
            err = mean_polled - actual
            print(f"  {year}: polled {mean_polled:.1f}%, actual {actual:.1f}%, error {err:+.1f}pp ({len(sub)} polls)")

    # ── 8. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: House effects by pollster (National bias)
    ax1 = axes[0]
    nat_he = he_df[he_df["party"] == "National"].sort_values("mean_error")
    nat_he = nat_he[nat_he["n"] >= 5]
    if len(nat_he) > 0:
        colors = ["#d32f2f" if x < 0 else "#1565c0" for x in nat_he["mean_error"]]
        ax1.barh(range(len(nat_he)), nat_he["mean_error"], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(nat_he)))
        ax1.set_yticklabels(nat_he["pollster"], fontsize=8)
        ax1.axvline(0, color="black", lw=0.5)
        ax1.set_xlabel("Mean error (polled - actual, pp)")
        ax1.set_title("Pollster House Effects: National Party Bias (final 60 days, n≥5)")

    # Panel 2: Error by election (National)
    ax2 = axes[1]
    election_errors = []
    for year in sorted(ELECTION_RESULTS.keys()):
        sub = final_14d[(final_14d["year"] == year) & (final_14d["party"] == "National")]
        if len(sub) > 0:
            election_errors.append({
                "year": year,
                "mean_error": sub["error"].mean(),
                "n": len(sub),
            })
    ee_df = pd.DataFrame(election_errors)
    if len(ee_df) > 0:
        colors = ["#d32f2f" if x < 0 else "#1565c0" for x in ee_df["mean_error"]]
        ax2.bar(ee_df["year"].astype(str), ee_df["mean_error"], color=colors, alpha=0.7)
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_xlabel("Election Year")
        ax2.set_ylabel("Mean National error (pp)")
        ax2.set_title("National Party: Final Poll Error by Election (last 14 days)")

    plt.tight_layout()
    outpath = REPORTS_DIR / "pollster_house_effects.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
