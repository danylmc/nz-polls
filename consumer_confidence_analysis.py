#!/usr/bin/env python3
"""
Consumer confidence vs incumbent party support.

OECD Composite Consumer Confidence Index (amplitude-adjusted, monthly, 1990-present)
vs party vote polling. Tests whether subjective confidence mediates between economic
conditions and voting intention.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"


def get_incumbent_at_date(date):
    if date < datetime(1999, 11, 27): return 'National'
    if date < datetime(2008, 11, 8): return 'Labour'
    if date < datetime(2017, 10, 26): return 'National'
    if date < datetime(2023, 11, 27): return 'Labour'
    return 'National'


def load_consumer_confidence():
    """Load OECD consumer confidence index."""
    df = pd.read_csv(DATA_DIR / "consumer_confidence.csv")
    # Columns: TIME_PERIOD, OBS_VALUE
    cc = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    cc["date"] = pd.to_datetime(cc["TIME_PERIOD"])
    cc["cci"] = pd.to_numeric(cc["OBS_VALUE"], errors="coerce")
    cc = cc.dropna(subset=["cci"]).sort_values("date")
    cc = cc.set_index("date")[["cci"]]
    # Resample to month-start for alignment
    cc = cc.resample("MS").mean()
    return cc


def load_all_polls():
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            for p in data.get("polls", []):
                row = {"date": p["date"], "pollster": p.get("pollster")}
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def main():
    cci = load_consumer_confidence()
    polls_raw = load_all_polls()

    polls_raw["incumbent"] = polls_raw["date"].apply(get_incumbent_at_date)
    polls_raw["inc_vote"] = polls_raw.apply(
        lambda r: pd.to_numeric(r.get(get_incumbent_at_date(r["date"])), errors="coerce"), axis=1
    )
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for p in parties:
        polls_raw[p] = pd.to_numeric(polls_raw[p], errors="coerce")

    polls_raw["month"] = polls_raw["date"].dt.to_period("M")
    monthly_polls = polls_raw.groupby("month").agg(
        inc_vote=("inc_vote", "mean"),
        **{p: (p, "mean") for p in parties}
    ).reset_index()
    monthly_polls["date"] = monthly_polls["month"].dt.to_timestamp()
    monthly_polls = monthly_polls.set_index("date")

    merged = cci.join(monthly_polls, how="inner").dropna(subset=["cci", "inc_vote"])
    merged["incumbent"] = merged.index.to_series().apply(get_incumbent_at_date)
    print(f"Consumer confidence: {cci.index.min():%Y-%m} to {cci.index.max():%Y-%m}")
    print(f"Matched months: {len(merged)} ({merged.index.min():%Y-%m} to {merged.index.max():%Y-%m})")

    # ── 1. Level correlation ──────────────────────────────────────────
    r_level, p_level = pearsonr(merged["cci"], merged["inc_vote"])
    print(f"\n=== LEVEL CORRELATION ===")
    print(f"  r = {r_level:.3f}, p = {p_level:.4f}, n = {len(merged)}")

    # ── 2. Change correlation ─────────────────────────────────────────
    merged["d_cci"] = merged["cci"].diff()
    merged["d_vote"] = merged["inc_vote"].diff()
    changes = merged.dropna(subset=["d_cci", "d_vote"])
    r_change, p_change = pearsonr(changes["d_cci"], changes["d_vote"])
    print(f"\n=== CHANGE CORRELATION (month-on-month) ===")
    print(f"  r = {r_change:.3f}, p = {p_change:.4f}, n = {len(changes)}")

    # ── 3. Lagged correlations ────────────────────────────────────────
    print(f"\n=== LAGGED CORRELATIONS (Δcci_t → Δvote_t+lag) ===")
    print(f"  {'Lag':>4s}  {'r':>7s}  {'p':>8s}")
    best_lag, best_r, best_p = 0, 0, 1
    lag_results = []
    for lag in range(0, 13):
        if lag > 0:
            lagged = pd.DataFrame({
                "d_cci": changes["d_cci"].values[:-lag],
                "d_vote": changes["d_vote"].values[lag:]
            }).dropna()
        else:
            lagged = changes[["d_cci", "d_vote"]].dropna()
        if len(lagged) < 10:
            continue
        r, p = pearsonr(lagged["d_cci"], lagged["d_vote"])
        lag_results.append((lag, r, p, len(lagged)))
        print(f"  {lag:>4d}  {r:>7.3f}  {p:>8.4f}")
        if abs(r) > abs(best_r):
            best_lag, best_r, best_p = lag, r, p
    print(f"\n  Best lag: {best_lag} months (r={best_r:.3f}, p={best_p:.4f})")

    # ── 4. Quarterly change ───────────────────────────────────────────
    merged["d_cci_3m"] = merged["cci"].diff(3)
    merged["d_vote_3m"] = merged["inc_vote"].diff(3)
    q_changes = merged.dropna(subset=["d_cci_3m", "d_vote_3m"])
    r_q, p_q = pearsonr(q_changes["d_cci_3m"], q_changes["d_vote_3m"])
    print(f"\n=== QUARTERLY CHANGE CORRELATION (3-month Δ) ===")
    print(f"  r = {r_q:.3f}, p = {p_q:.4f}, n = {len(q_changes)}")

    # ── 5. By government period ───────────────────────────────────────
    print(f"\n=== BY GOVERNMENT PERIOD (quarterly Δ) ===")
    for inc in ["Labour", "National"]:
        sub = q_changes[q_changes["incumbent"] == inc]
        if len(sub) > 10:
            r, p = pearsonr(sub["d_cci_3m"], sub["d_vote_3m"])
            print(f"  {inc}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 6. Granger causality ──────────────────────────────────────────
    print(f"\n=== GRANGER CAUSALITY (confidence → vote, max 6 lags) ===")
    gc_data = merged[["d_vote", "d_cci"]].dropna()
    try:
        gc = grangercausalitytests(gc_data, maxlag=6, verbose=False)
        for lag, result in gc.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 7. Party-level effects ────────────────────────────────────────
    print(f"\n=== PARTY-LEVEL CORRELATIONS (quarterly Δ) ===")
    for party in parties:
        merged[f"d_{party}_3m"] = merged[party].diff(3)
    pq = merged.dropna(subset=["d_cci_3m"])
    for party in parties:
        col = f"d_{party}_3m"
        sub = pq.dropna(subset=[col])
        if len(sub) > 10:
            r, p = pearsonr(sub["d_cci_3m"], sub[col])
            print(f"  {party:>10s}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 8. CCI vs approval (does confidence predict approval?) ────────
    try:
        approval = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
        approval["date"] = pd.to_datetime(approval["date"])
        approval = approval.set_index("date").resample("MS").last()
        merged_appr = merged.join(approval[["mean_score"]], how="inner").dropna(
            subset=["cci", "mean_score"]
        )
        if len(merged_appr) > 10:
            r_ca, p_ca = pearsonr(merged_appr["cci"], merged_appr["mean_score"])
            print(f"\n=== CCI vs GOVT APPROVAL ===")
            print(f"  r = {r_ca:.3f}, p = {p_ca:.4f}, n = {len(merged_appr)}")

            # Mediation: CCI → approval → vote
            if len(merged_appr) > 15:
                med = merged_appr.dropna(subset=["cci", "mean_score", "inc_vote"])
                X_a = sm.add_constant(med["cci"])
                mod_a = sm.OLS(med["mean_score"], X_a).fit()
                a = mod_a.params.iloc[1]
                se_a = mod_a.bse.iloc[1]

                X_b = sm.add_constant(med[["cci", "mean_score"]])
                mod_b = sm.OLS(med["inc_vote"], X_b).fit()
                b = mod_b.params["mean_score"]
                se_b = mod_b.bse["mean_score"]

                X_c = sm.add_constant(med["cci"])
                mod_c = sm.OLS(med["inc_vote"], X_c).fit()
                c = mod_c.params.iloc[1]

                indirect = a * b
                direct = mod_b.params["cci"]
                pct_mediated = (indirect / c * 100) if c != 0 else float("nan")

                sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
                sobel_z = indirect / sobel_se
                from scipy.stats import norm
                sobel_p = 2 * (1 - norm.cdf(abs(sobel_z)))

                print(f"\n=== MEDIATION: CCI → Approval → Vote ===")
                print(f"  n = {len(med)}")
                print(f"  Total effect (CCI → vote):       {c:.4f}, p = {mod_c.pvalues.iloc[1]:.4f}")
                print(f"  Direct effect (controlling appr): {direct:.4f}, p = {mod_b.pvalues['cci']:.4f}")
                print(f"  Indirect (CCI→appr→vote):        {indirect:.4f}")
                print(f"  % mediated: {pct_mediated:.1f}%")
                print(f"  Sobel z = {sobel_z:.2f}, p = {sobel_p:.4f}")
    except Exception as e:
        print(f"\n  Approval analysis skipped: {e}")

    # ── 9. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Time series
    ax1 = axes[0]
    ax1r = ax1.twinx()
    ax1.plot(merged.index, merged["inc_vote"], color="steelblue", alpha=0.8, label="Incumbent vote %")
    ax1r.plot(merged.index, merged["cci"], color="#27AE60", alpha=0.8, label="Consumer confidence")
    ax1.set_ylabel("Incumbent vote %", color="steelblue")
    ax1r.set_ylabel("Consumer confidence index", color="#27AE60")
    ax1.set_title("Consumer Confidence vs Incumbent Party Polling")
    govt_periods = [
        ("1993-01", "1999-11", "National"), ("1999-11", "2008-11", "Labour"),
        ("2008-11", "2017-10", "National"), ("2017-10", "2023-11", "Labour"),
        ("2023-11", "2026-03", "National")
    ]
    for s, e, party in govt_periods:
        color = "#d32f2f" if party == "Labour" else "#1565c0"
        try:
            ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.05, color=color)
        except Exception:
            pass
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # Panel 2: Scatter (quarterly changes)
    ax2 = axes[1]
    colors_map = q_changes["incumbent"].map({"Labour": "#d32f2f", "National": "#1565c0"})
    ax2.scatter(q_changes["d_cci_3m"], q_changes["d_vote_3m"], c=colors_map, alpha=0.4, s=20)
    m, b = np.polyfit(q_changes["d_cci_3m"], q_changes["d_vote_3m"], 1)
    x_range = np.linspace(q_changes["d_cci_3m"].min(), q_changes["d_cci_3m"].max(), 50)
    ax2.plot(x_range, m * x_range + b, "k--", alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel("3-month change in consumer confidence")
    ax2.set_ylabel("3-month change in incumbent vote (pp)")
    ax2.set_title(f"Quarterly CCI Change vs Incumbent Vote Change (r={r_q:.3f}, p={p_q:.4f})")
    ax2.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d32f2f", label="Labour govt"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", label="National govt"),
    ], fontsize=9)

    # Panel 3: Lagged correlation bar chart
    ax3 = axes[2]
    lags_df = pd.DataFrame(lag_results, columns=["lag", "r", "p", "n"])
    bar_colors = ["darkgreen" if pv < 0.05 else "gray" for pv in lags_df["p"]]
    ax3.bar(lags_df["lag"], lags_df["r"], color=bar_colors, alpha=0.7)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_xlabel("Lag (months)")
    ax3.set_ylabel("Correlation (r)")
    ax3.set_title("Lagged Correlation: Δ Consumer Confidence → Δ Incumbent Vote")
    ax3.legend(handles=[
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="darkgreen", label="p < 0.05"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", label="p ≥ 0.05"),
    ], fontsize=9)

    plt.tight_layout()
    outpath = REPORTS_DIR / "consumer_confidence_vs_incumbent.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
