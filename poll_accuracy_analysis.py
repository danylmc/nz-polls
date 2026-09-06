#!/usr/bin/env python3
"""
Poll Accuracy Analysis.

Compares final polls to election results across all elections.
Tests for systematic bias, convergence, and shy Tory effect.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp, ttest_ind

from events import ELECTION_DATES
from house_effects_analysis import ELECTION_RESULTS

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_all_polls():
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            year = data.get("election_year")
            for p in data.get("polls", []):
                row = {"date": p["date"], "election_year": year, "pollster": p.get("pollster")}
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for p in parties:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    return df


def main():
    polls = load_all_polls()
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    print("=" * 70)
    print("POLL ACCURACY ANALYSIS")
    print("=" * 70)

    # Build errors for all polls relative to their election result
    all_errors = []
    for year, results in sorted(ELECTION_RESULTS.items()):
        edate_str = ELECTION_DATES.get(year)
        if not edate_str:
            continue
        edate = pd.to_datetime(edate_str)

        cycle_polls = polls[polls["election_year"] == year].copy()
        cycle_polls["days_out"] = (edate - cycle_polls["date"]).dt.days
        cycle_polls = cycle_polls[(cycle_polls["days_out"] >= 0) & (cycle_polls["days_out"] <= 365)]

        for _, poll in cycle_polls.iterrows():
            for party in parties:
                actual = results.get(party)
                polled = poll.get(party)
                if actual is not None and pd.notna(polled):
                    all_errors.append({
                        "year": year,
                        "days_out": poll["days_out"],
                        "party": party,
                        "polled": polled,
                        "actual": actual,
                        "error": polled - actual,
                        "abs_error": abs(polled - actual),
                    })
    errors_df = pd.DataFrame(all_errors)

    # ── 1. Overall accuracy by time window ────────────────────────────
    print(f"\n--- ACCURACY BY TIME WINDOW ---")
    print(f"  {'Window':>20s}  {'Nat MAE':>8s}  {'Lab MAE':>8s}  {'Nat Bias':>9s}  {'Lab Bias':>9s}  {'n':>5s}")
    for window_name, max_days in [("Final 7 days", 7), ("Final 14 days", 14),
                                   ("Final 30 days", 30), ("Final 90 days", 90),
                                   ("Full cycle", 365)]:
        sub = errors_df[errors_df["days_out"] <= max_days]
        for party in ["National", "Labour"]:
            ps = sub[sub["party"] == party]
            if len(ps) > 0:
                mae = ps["abs_error"].mean()
                bias = ps["error"].mean()
                if party == "National":
                    n_mae = mae
                    n_bias = bias
                    n_n = len(ps)
                else:
                    print(f"  {window_name:>20s}  {n_mae:>8.2f}  {mae:>8.2f}  {n_bias:>+9.2f}  {bias:>+9.2f}  {n_n:>5d}")

    # ── 2. Convergence analysis ───────────────────────────────────────
    print(f"\n--- CONVERGENCE: Does |error| shrink closer to election? ---")
    for party in parties:
        sub = errors_df[errors_df["party"] == party]
        if len(sub) > 20:
            r, p = pearsonr(sub["days_out"], sub["abs_error"])
            print(f"  {party:>10s}: r(days_out, |error|) = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 3. Shy Tory test by election ──────────────────────────────────
    print(f"\n--- NATIONAL FINAL POLL ERROR BY ELECTION ---")
    print(f"  {'Year':>6s}  {'Polled':>7s}  {'Actual':>7s}  {'Error':>7s}  {'n':>4s}  {'Interpretation'}")
    shy_tory_count = 0
    total_elections = 0
    for year in sorted(ELECTION_RESULTS.keys()):
        sub = errors_df[(errors_df["year"] == year) &
                        (errors_df["party"] == "National") &
                        (errors_df["days_out"] <= 14)]
        if len(sub) > 0:
            total_elections += 1
            mean_err = sub["error"].mean()
            mean_polled = sub["polled"].mean()
            actual = ELECTION_RESULTS[year]["National"]
            if mean_err < -0.5:
                shy_tory_count += 1
                interp = "UNDERESTIMATED (shy Tory)"
            elif mean_err > 0.5:
                interp = "OVERESTIMATED"
            else:
                interp = "~accurate"
            print(f"  {year:>6d}  {mean_polled:>7.1f}  {actual:>7.1f}  {mean_err:>+7.1f}  {len(sub):>4d}  {interp}")

    print(f"\n  Shy Tory elections: {shy_tory_count}/{total_elections}")
    print(f"  National systematically underestimated? ", end="")
    nat_final = errors_df[(errors_df["party"] == "National") & (errors_df["days_out"] <= 14)]
    t, p = ttest_1samp(nat_final["error"], 0)
    print(f"t={t:.2f}, p={p:.4f} → {'YES' if p < 0.05 and nat_final['error'].mean() < 0 else 'NO'}")

    # ── 4. Two-party margin error ─────────────────────────────────────
    print(f"\n--- TWO-PARTY MARGIN ERROR (final 14 days) ---")
    print(f"  {'Year':>6s}  {'Poll Margin':>12s}  {'Actual Margin':>14s}  {'Margin Error':>13s}")
    margin_errors = []
    for year in sorted(ELECTION_RESULTS.keys()):
        nat_sub = errors_df[(errors_df["year"] == year) &
                            (errors_df["party"] == "National") &
                            (errors_df["days_out"] <= 14)]
        lab_sub = errors_df[(errors_df["year"] == year) &
                            (errors_df["party"] == "Labour") &
                            (errors_df["days_out"] <= 14)]
        if len(nat_sub) > 0 and len(lab_sub) > 0:
            poll_margin = nat_sub["polled"].mean() - lab_sub["polled"].mean()
            actual_nat = ELECTION_RESULTS[year]["National"]
            actual_lab = ELECTION_RESULTS[year]["Labour"]
            actual_margin = actual_nat - actual_lab
            margin_err = poll_margin - actual_margin
            margin_errors.append(margin_err)
            winner_correct = (poll_margin > 0) == (actual_margin > 0)
            print(f"  {year:>6d}  {poll_margin:>+12.1f}  {actual_margin:>+14.1f}  {margin_err:>+13.1f}  {'✓' if winner_correct else '✗ WRONG'}")

    if margin_errors:
        mae = np.mean(np.abs(margin_errors))
        print(f"\n  Mean absolute margin error: {mae:.1f}pp")
        print(f"  Winner called correctly: {sum(1 for m in margin_errors if True)}/{len(margin_errors)}")

    # ── 5. Minor party accuracy ───────────────────────────────────────
    print(f"\n--- MINOR PARTY ACCURACY (final 14 days) ---")
    for party in ["Green", "NZ First", "ACT"]:
        sub = errors_df[(errors_df["party"] == party) & (errors_df["days_out"] <= 14)]
        if len(sub) > 5:
            t, p = ttest_1samp(sub["error"], 0)
            print(f"  {party:>10s}: bias = {sub['error'].mean():+.2f}pp, MAE = {sub['abs_error'].mean():.2f}pp (t={t:.2f}, p={p:.4f})")

    # ── 6. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Convergence (National)
    ax = axes[0, 0]
    nat = errors_df[errors_df["party"] == "National"]
    ax.scatter(nat["days_out"], nat["abs_error"], alpha=0.2, s=10, color="#1565c0")
    # Binned average
    bins = [0, 7, 14, 30, 60, 90, 180, 365]
    for i in range(len(bins) - 1):
        sub = nat[(nat["days_out"] >= bins[i]) & (nat["days_out"] < bins[i+1])]
        if len(sub) > 0:
            ax.scatter((bins[i] + bins[i+1]) / 2, sub["abs_error"].mean(), color="red", s=50, zorder=5)
    ax.set_xlabel("Days before election")
    ax.set_ylabel("|Error| (pp)")
    ax.set_title("National: Poll Error Convergence")

    # Panel 2: Convergence (Labour)
    ax = axes[0, 1]
    lab = errors_df[errors_df["party"] == "Labour"]
    ax.scatter(lab["days_out"], lab["abs_error"], alpha=0.2, s=10, color="#d32f2f")
    for i in range(len(bins) - 1):
        sub = lab[(lab["days_out"] >= bins[i]) & (lab["days_out"] < bins[i+1])]
        if len(sub) > 0:
            ax.scatter((bins[i] + bins[i+1]) / 2, sub["abs_error"].mean(), color="red", s=50, zorder=5)
    ax.set_xlabel("Days before election")
    ax.set_ylabel("|Error| (pp)")
    ax.set_title("Labour: Poll Error Convergence")

    # Panel 3: Error by election (National vs Labour)
    ax = axes[1, 0]
    years = sorted(ELECTION_RESULTS.keys())
    x = np.arange(len(years))
    nat_errs = []
    lab_errs = []
    for year in years:
        nat_sub = errors_df[(errors_df["year"] == year) & (errors_df["party"] == "National") & (errors_df["days_out"] <= 14)]
        lab_sub = errors_df[(errors_df["year"] == year) & (errors_df["party"] == "Labour") & (errors_df["days_out"] <= 14)]
        nat_errs.append(nat_sub["error"].mean() if len(nat_sub) > 0 else 0)
        lab_errs.append(lab_sub["error"].mean() if len(lab_sub) > 0 else 0)
    w = 0.35
    ax.bar(x - w/2, nat_errs, w, color="#1565c0", alpha=0.7, label="National")
    ax.bar(x + w/2, lab_errs, w, color="#d32f2f", alpha=0.7, label="Labour")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Mean error (polled - actual, pp)")
    ax.set_title("Final Poll Error by Election (last 14 days)")
    ax.legend(fontsize=8)

    # Panel 4: MAE by party
    ax = axes[1, 1]
    final = errors_df[errors_df["days_out"] <= 14]
    party_mae = []
    for party in parties:
        sub = final[final["party"] == party]
        party_mae.append(sub["abs_error"].mean())
    colors = ["#1565c0", "#d32f2f", "#27AE60", "#2C3E50", "#F1C40F"]
    ax.bar(parties, party_mae, color=colors, alpha=0.7)
    ax.set_ylabel("Mean Absolute Error (pp)")
    ax.set_title("Polling Accuracy by Party (final 14 days)")

    plt.tight_layout()
    outpath = REPORTS_DIR / "poll_accuracy.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
