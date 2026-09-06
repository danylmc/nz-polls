#!/usr/bin/env python3
"""
PM Preference Lead/Lag Analysis.

Tests the presidentialisation thesis (Poguntke & Webb 2005):
Does preferred PM lead or lag party vote share?
Cross-correlates PM preference with party vote at various lags.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Map PM candidates to their party
PM_PARTY = {
    "Clark": "Labour", "Goff": "Labour", "Shearer": "Labour",
    "Cunliffe": "Labour", "Little": "Labour", "Ardern": "Labour",
    "Hipkins": "Labour",
    "English": "National", "Key": "National", "Brash": "National",
    "Collins": "National", "Luxon": "National", "Bridges": "National",
    "Muller": "National",
    "Peters": "NZ First",
}


def load_pm_polls():
    """Load all PM preference polls, extracting party leader scores."""
    rows = []
    for f in sorted(DATA_DIR.glob("*_pm_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            year = data.get("election_year")
            for p in data.get("polls", []):
                date = p.get("date")
                candidates = p.get("candidates", {})
                # Get National and Labour leader scores
                nat_score = lab_score = None
                for name, score in candidates.items():
                    party = PM_PARTY.get(name)
                    if party == "National" and nat_score is None:
                        nat_score = score
                    elif party == "Labour" and lab_score is None:
                        lab_score = score
                if date and (nat_score is not None or lab_score is not None):
                    rows.append({
                        "date": date,
                        "election_year": year,
                        "nat_pm": nat_score,
                        "lab_pm": lab_score,
                    })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def load_party_polls():
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        with open(f) as fh:
            data = json.load(fh)
            for p in data.get("polls", []):
                row = {"date": p["date"]}
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ["National", "Labour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    pm = load_pm_polls()
    party = load_party_polls()
    print(f"PM polls: {len(pm)} ({pm['date'].min():%Y-%m} to {pm['date'].max():%Y-%m})")
    print(f"Party polls: {len(party)} ({party['date'].min():%Y-%m} to {party['date'].max():%Y-%m})")

    # Resample both to monthly
    pm["month"] = pm["date"].dt.to_period("M")
    pm_monthly = pm.groupby("month").agg(
        nat_pm=("nat_pm", "mean"),
        lab_pm=("lab_pm", "mean"),
    ).reset_index()
    pm_monthly["date"] = pm_monthly["month"].dt.to_timestamp()
    pm_monthly = pm_monthly.set_index("date").drop(columns=["month"])

    party["month"] = party["date"].dt.to_period("M")
    party_monthly = party.groupby("month").agg(
        National=("National", "mean"),
        Labour=("Labour", "mean"),
    ).reset_index()
    party_monthly["date"] = party_monthly["month"].dt.to_timestamp()
    party_monthly = party_monthly.set_index("date").drop(columns=["month"])

    merged = pm_monthly.join(party_monthly, how="inner").dropna()
    print(f"Matched months: {len(merged)}")

    # ── 1. Level correlations ─────────────────────────────────────────
    print(f"\n=== LEVEL CORRELATIONS ===")
    for pm_col, pv_col, label in [
        ("nat_pm", "National", "National PM → National Vote"),
        ("lab_pm", "Labour", "Labour PM → Labour Vote"),
    ]:
        sub = merged.dropna(subset=[pm_col, pv_col])
        if len(sub) > 10:
            r, p = pearsonr(sub[pm_col], sub[pv_col])
            print(f"  {label}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 2. Change correlations ────────────────────────────────────────
    merged["d_nat_pm"] = merged["nat_pm"].diff()
    merged["d_lab_pm"] = merged["lab_pm"].diff()
    merged["d_National"] = merged["National"].diff()
    merged["d_Labour"] = merged["Labour"].diff()
    changes = merged.dropna(subset=["d_nat_pm", "d_National"])

    print(f"\n=== CHANGE CORRELATIONS ===")
    for dpm, dpv, label in [
        ("d_nat_pm", "d_National", "ΔNat PM → ΔNat Vote"),
        ("d_lab_pm", "d_Labour", "ΔLab PM → ΔLab Vote"),
    ]:
        sub = changes.dropna(subset=[dpm, dpv])
        if len(sub) > 10:
            r, p = pearsonr(sub[dpm], sub[dpv])
            print(f"  {label}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 3. Cross-correlation at lags (the key test) ───────────────────
    # Positive lag = PM leads party vote; Negative lag = PM lags party vote
    print(f"\n=== CROSS-CORRELATION: National PM preference → National party vote ===")
    print(f"  Positive lag: PM change at t predicts vote change at t+lag (PM LEADS)")
    print(f"  Negative lag: vote change at t predicts PM change at t+lag (VOTE LEADS)")

    for dpm, dpv, label in [
        ("d_nat_pm", "d_National", "National"),
        ("d_lab_pm", "d_Labour", "Labour"),
    ]:
        sub = changes.dropna(subset=[dpm, dpv])
        print(f"\n  --- {label} ---")
        print(f"  {'Lag':>4s}  {'r':>7s}  {'p':>8s}  {'Direction':>12s}")
        best_lag, best_r = 0, 0
        for lag in range(-6, 7):
            if lag > 0:
                x = sub[dpm].values[:-lag]
                y = sub[dpv].values[lag:]
            elif lag < 0:
                x = sub[dpm].values[-lag:]
                y = sub[dpv].values[:lag]
            else:
                x = sub[dpm].values
                y = sub[dpv].values
            if len(x) < 10:
                continue
            r, p = pearsonr(x, y)
            direction = "PM leads" if lag > 0 else ("contemporaneous" if lag == 0 else "Vote leads")
            print(f"  {lag:>4d}  {r:>7.3f}  {p:>8.4f}  {direction:>12s}")
            if abs(r) > abs(best_r):
                best_lag, best_r = lag, r
        if best_lag > 0:
            print(f"  → Peak at lag {best_lag}: PM preference LEADS party vote by {best_lag} months")
        elif best_lag < 0:
            print(f"  → Peak at lag {best_lag}: Party vote LEADS PM preference by {-best_lag} months")
        else:
            print(f"  → Peak at lag 0: contemporaneous relationship")

    # ── 4. PM margin vs party vote margin ─────────────────────────────
    pm_margin = merged["nat_pm"] - merged["lab_pm"]
    vote_margin = merged["National"] - merged["Labour"]
    sub = pd.DataFrame({"pm_margin": pm_margin, "vote_margin": vote_margin}).dropna()
    if len(sub) > 10:
        r, p = pearsonr(sub["pm_margin"], sub["vote_margin"])
        print(f"\n=== PM MARGIN vs VOTE MARGIN ===")
        print(f"  r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")
        # Regression: how much does 1pp PM lead translate to party vote?
        import statsmodels.api as sm
        X = sm.add_constant(sub["pm_margin"])
        mod = sm.OLS(sub["vote_margin"], X).fit()
        print(f"  Regression: vote_margin = {mod.params.iloc[0]:.2f} + {mod.params.iloc[1]:.3f} × pm_margin")
        print(f"  R² = {mod.rsquared:.3f}")
        print(f"  A 1pp PM preference lead ≈ {mod.params.iloc[1]:.2f}pp party vote lead")

    # ── 5. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: National PM vs National vote (time series)
    ax = axes[0, 0]
    ax.plot(merged.index, merged["nat_pm"], color="#1565c0", alpha=0.7, label="Preferred PM")
    ax.plot(merged.index, merged["National"], color="#1565c0", alpha=0.7, linestyle="--", label="Party vote")
    ax.set_title("National: PM Preference vs Party Vote")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    # Panel 2: Labour PM vs Labour vote
    ax = axes[0, 1]
    ax.plot(merged.index, merged["lab_pm"], color="#d32f2f", alpha=0.7, label="Preferred PM")
    ax.plot(merged.index, merged["Labour"], color="#d32f2f", alpha=0.7, linestyle="--", label="Party vote")
    ax.set_title("Labour: PM Preference vs Party Vote")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)

    # Panel 3: Cross-correlation (National)
    ax = axes[1, 0]
    nat_lags = []
    sub = changes.dropna(subset=["d_nat_pm", "d_National"])
    for lag in range(-6, 7):
        if lag > 0:
            x = sub["d_nat_pm"].values[:-lag]
            y = sub["d_National"].values[lag:]
        elif lag < 0:
            x = sub["d_nat_pm"].values[-lag:]
            y = sub["d_National"].values[:lag]
        else:
            x = sub["d_nat_pm"].values
            y = sub["d_National"].values
        if len(x) >= 10:
            r, p = pearsonr(x, y)
            nat_lags.append((lag, r, p))
    nl = pd.DataFrame(nat_lags, columns=["lag", "r", "p"])
    bar_colors = ["darkgreen" if pv < 0.05 else "gray" for pv in nl["p"]]
    ax.bar(nl["lag"], nl["r"], color=bar_colors, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlabel("Lag (months, + = PM leads)")
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Cross-correlation: Δ National PM → Δ National Vote")

    # Panel 4: PM margin vs vote margin scatter
    ax = axes[1, 1]
    ax.scatter(sub.index.to_series().apply(lambda d: pm_margin.get(d, np.nan)),
               sub.index.to_series().apply(lambda d: vote_margin.get(d, np.nan)),
               alpha=0.3, s=15, color="#2C3E50")
    # Simpler: use the sub df from above
    ax.scatter(sub["pm_margin"] if "pm_margin" in sub.columns else [],
               sub["vote_margin"] if "vote_margin" in sub.columns else [],
               alpha=0.3, s=15, color="#2C3E50")
    sub2 = pd.DataFrame({"pm_margin": pm_margin, "vote_margin": vote_margin}).dropna()
    ax.scatter(sub2["pm_margin"], sub2["vote_margin"], alpha=0.3, s=15, color="#2C3E50")
    m, b = np.polyfit(sub2["pm_margin"], sub2["vote_margin"], 1)
    xr = np.linspace(sub2["pm_margin"].min(), sub2["pm_margin"].max(), 50)
    ax.plot(xr, m * xr + b, "k--", alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("PM preference margin (Nat leader - Lab leader, pp)")
    ax.set_ylabel("Party vote margin (National - Labour, pp)")
    ax.set_title(f"PM Margin vs Vote Margin (r={pearsonr(sub2['pm_margin'], sub2['vote_margin'])[0]:.3f})")

    plt.tight_layout()
    outpath = REPORTS_DIR / "pm_preference_leadlag.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
