#!/usr/bin/env python3
"""
Petrol prices vs incumbent party support — full analysis.

Extends pocketbook voting analysis to petrol prices (the most visible daily cost).
Matches the depth of mortgage_analysis.py: Granger causality, lagged correlations,
by-government-period splits, opposition benefit, minor party insulation, rate-spike episodes.
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

from events import ELECTION_DATES, INCUMBENTS

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = Path(__file__).parent / "raw_data"
REPORTS_DIR = Path(__file__).parent / "reports"


def get_incumbent_at_date(date):
    """Determine incumbent party at a given date."""
    if date < datetime(2008, 11, 8): return 'Labour'
    if date < datetime(2017, 10, 26): return 'National'
    if date < datetime(2023, 11, 27): return 'Labour'
    return 'National'


def load_petrol_prices():
    """Load weekly regular petrol board price from MBIE data, resample to monthly."""
    df = pd.read_csv(RAW_DIR / "weekly-table.csv")
    mask = (df["Fuel"] == "Regular Petrol") & (df["Variable"] == "Board price")
    petrol = df[mask][["Date", "Value"]].copy()
    petrol["Date"] = pd.to_datetime(petrol["Date"])
    petrol["petrol_price"] = pd.to_numeric(petrol["Value"], errors="coerce")
    petrol = petrol.dropna(subset=["petrol_price"]).sort_values("Date").set_index("Date")
    return petrol[["petrol_price"]].resample("MS").mean()


def load_all_polls():
    """Load all polling data into a DataFrame with party columns."""
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
    petrol = load_petrol_prices()
    polls_raw = load_all_polls()

    # Add incumbent info
    polls_raw["incumbent"] = polls_raw["date"].apply(get_incumbent_at_date)
    polls_raw["inc_vote"] = polls_raw.apply(
        lambda r: r.get(get_incumbent_at_date(r["date"])), axis=1
    )
    polls_raw["inc_vote"] = pd.to_numeric(polls_raw["inc_vote"], errors="coerce")

    # Monthly aggregation for each party + incumbent
    polls_raw["month"] = polls_raw["date"].dt.to_period("M")
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for p in parties:
        polls_raw[p] = pd.to_numeric(polls_raw[p], errors="coerce")

    monthly_polls = polls_raw.groupby("month").agg(
        inc_vote=("inc_vote", "mean"),
        **{p: (p, "mean") for p in parties}
    ).reset_index()
    monthly_polls["date"] = monthly_polls["month"].dt.to_timestamp()
    monthly_polls = monthly_polls.set_index("date")

    # Merge with petrol
    merged = petrol.join(monthly_polls, how="inner").dropna(subset=["petrol_price", "inc_vote"])
    merged["incumbent"] = merged.index.to_series().apply(get_incumbent_at_date)
    print(f"Petrol data: {petrol.index.min():%Y-%m} to {petrol.index.max():%Y-%m}")
    print(f"Matched months: {len(merged)} ({merged.index.min():%Y-%m} to {merged.index.max():%Y-%m})")

    # ── 1. Level correlation ──────────────────────────────────────────
    r_level, p_level = pearsonr(merged["petrol_price"], merged["inc_vote"])
    print(f"\n=== LEVEL CORRELATION ===")
    print(f"  r = {r_level:.3f}, p = {p_level:.4f}, n = {len(merged)}")

    # ── 2. Change correlation (month-on-month) ────────────────────────
    merged["d_petrol"] = merged["petrol_price"].diff()
    merged["d_vote"] = merged["inc_vote"].diff()
    changes = merged.dropna(subset=["d_petrol", "d_vote"])
    r_change, p_change = pearsonr(changes["d_petrol"], changes["d_vote"])
    print(f"\n=== CHANGE CORRELATION (month-on-month) ===")
    print(f"  r = {r_change:.3f}, p = {p_change:.4f}, n = {len(changes)}")

    # ── 3. Lagged correlations ────────────────────────────────────────
    print(f"\n=== LAGGED CORRELATIONS (Δpetrol_t → Δvote_t+lag) ===")
    print(f"  {'Lag':>4s}  {'r':>7s}  {'p':>8s}")
    best_lag, best_r, best_p = 0, 0, 1
    lag_results = []
    for lag in range(0, 13):
        if lag > 0:
            lagged = pd.DataFrame({
                "d_petrol": changes["d_petrol"].values[:-lag],
                "d_vote": changes["d_vote"].values[lag:]
            }).dropna()
        else:
            lagged = changes[["d_petrol", "d_vote"]].dropna()
        if len(lagged) < 10:
            continue
        r, p = pearsonr(lagged["d_petrol"], lagged["d_vote"])
        lag_results.append((lag, r, p, len(lagged)))
        print(f"  {lag:>4d}  {r:>7.3f}  {p:>8.4f}")
        if abs(r) > abs(best_r):
            best_lag, best_r, best_p = lag, r, p
    print(f"\n  Best lag: {best_lag} months (r={best_r:.3f}, p={best_p:.4f})")

    # ── 4. Quarterly change (3-month Δ) ───────────────────────────────
    merged["d_petrol_3m"] = merged["petrol_price"].diff(3)
    merged["d_vote_3m"] = merged["inc_vote"].diff(3)
    q_changes = merged.dropna(subset=["d_petrol_3m", "d_vote_3m"])
    r_q, p_q = pearsonr(q_changes["d_petrol_3m"], q_changes["d_vote_3m"])
    print(f"\n=== QUARTERLY CHANGE CORRELATION (3-month Δ) ===")
    print(f"  r = {r_q:.3f}, p = {p_q:.4f}, n = {len(q_changes)}")

    # ── 5. By government period ───────────────────────────────────────
    print(f"\n=== BY GOVERNMENT PERIOD (monthly Δ) ===")
    for inc in ["Labour", "National"]:
        sub = changes[changes["incumbent"] == inc]
        if len(sub) > 10:
            r, p = pearsonr(sub["d_petrol"], sub["d_vote"])
            print(f"  {inc}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 6. Granger causality ──────────────────────────────────────────
    print(f"\n=== GRANGER CAUSALITY (petrol → vote, max 6 lags) ===")
    gc_data = merged[["d_vote", "d_petrol"]].dropna()
    try:
        gc = grangercausalitytests(gc_data, maxlag=6, verbose=False)
        for lag, result in gc.items():
            f_stat = result[0]["ssr_ftest"][0]
            p_val = result[0]["ssr_ftest"][1]
            print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # ── 7. Government vs Opposition Bloc ──────────────────────────────
    print(f"\n=== GOVERNMENT vs OPPOSITION BLOC (quarterly Δ) ===")
    # Calculate govt bloc and opp bloc
    for idx, row in merged.iterrows():
        inc = get_incumbent_at_date(idx)
        if inc == "Labour":
            merged.loc[idx, "govt_bloc"] = (row.get("Labour", 0) or 0) + (row.get("Green", 0) or 0)
            merged.loc[idx, "opp_bloc"] = (row.get("National", 0) or 0) + (row.get("ACT", 0) or 0)
        else:
            merged.loc[idx, "govt_bloc"] = (row.get("National", 0) or 0) + (row.get("ACT", 0) or 0)
            merged.loc[idx, "opp_bloc"] = (row.get("Labour", 0) or 0) + (row.get("Green", 0) or 0)

    merged["d_govt_3m"] = merged["govt_bloc"].diff(3)
    merged["d_opp_3m"] = merged["opp_bloc"].diff(3)
    bloc_q = merged.dropna(subset=["d_petrol_3m", "d_govt_3m", "d_opp_3m"])
    if len(bloc_q) > 10:
        r_govt, p_govt = pearsonr(bloc_q["d_petrol_3m"], bloc_q["d_govt_3m"])
        r_opp, p_opp = pearsonr(bloc_q["d_petrol_3m"], bloc_q["d_opp_3m"])
        print(f"  Govt bloc:  r = {r_govt:.3f}, p = {p_govt:.4f}, n = {len(bloc_q)}")
        print(f"  Opp bloc:   r = {r_opp:.3f}, p = {p_opp:.4f}, n = {len(bloc_q)}")

    # ── 8. Party-level effects (split by incumbency) ──────────────────
    print(f"\n=== PARTY-LEVEL EFFECTS (quarterly Δ, by incumbency) ===")
    for party in parties:
        merged[f"d_{party}_3m"] = merged[party].diff(3)

    party_q = merged.dropna(subset=["d_petrol_3m"])
    for party in parties:
        col = f"d_{party}_3m"
        sub_govt = party_q[party_q["incumbent"] == party].dropna(subset=[col])
        sub_opp = party_q[party_q["incumbent"] != party].dropna(subset=[col])
        r_g = r_o = p_g = p_o = float("nan")
        n_g = len(sub_govt)
        n_o = len(sub_opp)
        if n_g > 10:
            r_g, p_g = pearsonr(sub_govt["d_petrol_3m"], sub_govt[col])
        if n_o > 10:
            r_o, p_o = pearsonr(sub_opp["d_petrol_3m"], sub_opp[col])
        print(f"  {party:>10s}  In Govt: r={r_g:>7.3f} p={p_g:>7.4f} n={n_g:>3d}  |  In Opp: r={r_o:>7.3f} p={p_o:>7.4f} n={n_o:>3d}")

    # ── 9. Petrol spike episodes ──────────────────────────────────────
    # Define spike as >30 c/L rise over 6 months
    merged["d_petrol_6m"] = merged["petrol_price"].diff(6)
    spikes = merged[merged["d_petrol_6m"] > 30].copy()
    print(f"\n=== PETROL PRICE SPIKE EPISODES (>30 c/L rise over 6 months) ===")
    print(f"  {len(spikes)} qualifying months")
    if len(spikes) > 0:
        print(f"  Date range: {spikes.index.min():%Y-%m} to {spikes.index.max():%Y-%m}")
        for party in parties:
            col = f"d_{party}_3m"
            if col in spikes.columns:
                vals = spikes[col].dropna()
                if len(vals) > 0:
                    print(f"  {party:>10s}: mean 3-month polling change = {vals.mean():+.1f}pp")

    # ── 10. YoY change correlation ────────────────────────────────────
    merged["petrol_yoy"] = merged["petrol_price"].pct_change(12) * 100
    merged["vote_yoy"] = merged["inc_vote"].diff(12)
    yoy = merged.dropna(subset=["petrol_yoy", "vote_yoy"])
    if len(yoy) > 10:
        r_yoy, p_yoy = pearsonr(yoy["petrol_yoy"], yoy["vote_yoy"])
        print(f"\n=== YoY CHANGE CORRELATION ===")
        print(f"  r = {r_yoy:.3f}, p = {p_yoy:.4f}, n = {len(yoy)}")

    # ── 11. Mediation through Ipsos approval ──────────────────────────
    try:
        approval = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
        approval["date"] = pd.to_datetime(approval["date"])
        approval = approval.set_index("date").resample("MS").last()
        merged_med = merged.join(approval[["mean_score"]], how="inner").dropna(
            subset=["petrol_price", "inc_vote", "mean_score"]
        )
        if len(merged_med) > 15:
            import statsmodels.api as sm
            # Path a: petrol → approval
            X_a = sm.add_constant(merged_med["petrol_price"])
            mod_a = sm.OLS(merged_med["mean_score"], X_a).fit()
            a = mod_a.params.iloc[1]
            se_a = mod_a.bse.iloc[1]
            # Path b: approval → vote (controlling for petrol)
            X_b = sm.add_constant(merged_med[["petrol_price", "mean_score"]])
            mod_b = sm.OLS(merged_med["inc_vote"], X_b).fit()
            b = mod_b.params["mean_score"]
            se_b = mod_b.bse["mean_score"]
            # Total effect
            X_c = sm.add_constant(merged_med["petrol_price"])
            mod_c = sm.OLS(merged_med["inc_vote"], X_c).fit()
            c = mod_c.params.iloc[1]
            # Indirect
            indirect = a * b
            direct = mod_b.params["petrol_price"]
            pct_mediated = (indirect / c * 100) if c != 0 else float("nan")
            # Sobel test
            sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
            sobel_z = indirect / sobel_se
            sobel_p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(sobel_z)))
            print(f"\n=== MEDIATION: Petrol → Approval → Vote ===")
            print(f"  n = {len(merged_med)}")
            print(f"  Total effect (petrol → vote):    {c:.4f}, p = {mod_c.pvalues.iloc[1]:.4f}")
            print(f"  Direct effect (controlling appr): {direct:.4f}, p = {mod_b.pvalues['petrol_price']:.4f}")
            print(f"  Indirect (petrol→appr→vote):     {indirect:.4f}")
            print(f"  % mediated: {pct_mediated:.1f}%")
            print(f"  Sobel z = {sobel_z:.2f}, p = {sobel_p:.4f}")
    except Exception as e:
        print(f"\n  Mediation analysis skipped: {e}")

    # ── 12. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Panel 1: Time series (dual axis)
    ax1 = axes[0]
    ax1r = ax1.twinx()
    ax1.plot(merged.index, merged["inc_vote"], color="steelblue", alpha=0.8, label="Incumbent vote %")
    ax1r.plot(merged.index, merged["petrol_price"], color="#E67E22", alpha=0.8, label="Regular petrol (c/L)")
    ax1r.invert_yaxis()
    ax1.set_ylabel("Incumbent vote %", color="steelblue")
    ax1r.set_ylabel("Petrol price c/L (inverted)", color="#E67E22")
    ax1.set_title("Petrol Price vs Incumbent Party Polling")
    # Shade government periods
    govt_periods = [
        ("2005-01", "2008-11", "Labour"), ("2008-11", "2017-10", "National"),
        ("2017-10", "2023-11", "Labour"), ("2023-11", "2026-03", "National")
    ]
    for s, e, party in govt_periods:
        color = "#d32f2f" if party == "Labour" else "#1565c0"
        ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.05, color=color)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # Panel 2: Scatter (quarterly changes)
    ax2 = axes[1]
    colors_map = q_changes["incumbent"].map({"Labour": "#d32f2f", "National": "#1565c0"})
    ax2.scatter(q_changes["d_petrol_3m"], q_changes["d_vote_3m"], c=colors_map, alpha=0.4, s=20)
    m, b = np.polyfit(q_changes["d_petrol_3m"], q_changes["d_vote_3m"], 1)
    x_range = np.linspace(q_changes["d_petrol_3m"].min(), q_changes["d_petrol_3m"].max(), 50)
    ax2.plot(x_range, m * x_range + b, "k--", alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel("3-month change in petrol price (c/L)")
    ax2.set_ylabel("3-month change in incumbent vote (pp)")
    ax2.set_title(f"Quarterly Petrol Change vs Incumbent Vote Change (r={r_q:.3f}, p={p_q:.4f})")
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
    ax3.set_title("Lagged Correlation: Δ Petrol Price → Δ Incumbent Vote")
    ax3.legend(handles=[
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="darkgreen", label="p < 0.05"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", label="p ≥ 0.05"),
    ], fontsize=9)

    plt.tight_layout()
    outpath = REPORTS_DIR / "petrol_vs_incumbent.png"
    outpath.parent.mkdir(exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
