#!/usr/bin/env python3
"""
Rental Price Index vs Polling Analysis.

~33% of NZ households rent. Rents are a direct pocketbook cost for a
politically distinct demographic (younger, more urban, more Labour-leaning).
May predict Labour-specific effects that mortgage rates miss.

Tests:
1. Does rising rent inflation correlate with incumbent polling decline?
2. Is the rental effect distinct from mortgage rate effects?
3. Are renters a Labour-specific channel (vs mortgage = National homeowner)?

Data: Stats NZ CPI "Actual rentals for housing" (quarterly, 1999-2025)
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Government periods
GOVERNMENTS = [
    ("1999-11-27", "2002-07-27", "Labour"),
    ("2002-07-27", "2005-09-17", "Labour"),
    ("2005-09-17", "2008-11-08", "Labour"),
    ("2008-11-08", "2011-11-26", "National"),
    ("2011-11-26", "2014-09-20", "National"),
    ("2014-09-20", "2017-09-23", "National"),
    ("2017-10-26", "2020-10-17", "Labour"),
    ("2020-10-17", "2023-10-14", "Labour"),
    ("2023-11-27", "2026-03-01", "National"),
]


def load_rental_cpi():
    """Load quarterly rental CPI index."""
    cpi = pd.read_csv(DATA_DIR / "consumers-price-index-december-2025-quarter-index-numbers (1).csv")
    rental = cpi[cpi["Series_reference"] == "CPIQ.SE9041"].copy()  # Use SE9041 (main series)
    if len(rental) == 0:
        rental = cpi[cpi["Series_reference"] == "CPIQ.SE904101"].copy()

    # Parse period (e.g., "2021.03" → datetime) and align to quarter start
    rental["date"] = rental["Period"].apply(
        lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 100):02d}-01")
    )
    # Align to quarter start for joining with polls
    rental["date"] = rental["date"].dt.to_period("Q").dt.to_timestamp()
    rental = rental.sort_values("date").set_index("date")
    rental = rental[["Data_value"]].rename(columns={"Data_value": "rental_index"})

    # Year-on-year % change
    rental["rental_yoy"] = rental["rental_index"].pct_change(4) * 100
    # Quarter-on-quarter % change
    rental["rental_qoq"] = rental["rental_index"].pct_change(1) * 100

    return rental


def load_polls_quarterly():
    """Load polling data aggregated to quarterly means."""
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
    parties = ["National", "Labour", "Green", "NZ First", "ACT"]
    for col in parties:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly = df.groupby("quarter")[parties].mean()
    quarterly.index = quarterly.index.to_timestamp()
    return quarterly


def get_incumbent(date):
    """Return incumbent party for a given date."""
    for start, end, party in GOVERNMENTS:
        if pd.to_datetime(start) <= date <= pd.to_datetime(end):
            return party
    return None


def main():
    rental = load_rental_cpi()
    polls = load_polls_quarterly()

    print("=" * 70)
    print("RENTAL PRICE INDEX vs POLLING ANALYSIS")
    print("=" * 70)

    print(f"\nRental CPI data: {rental.index.min().strftime('%Y-%m')} to {rental.index.max().strftime('%Y-%m')}")
    print(f"Rental index range: {rental['rental_index'].min():.0f} to {rental['rental_index'].max():.0f}")

    # Merge rental with polling
    merged = rental.join(polls, how="inner").dropna(subset=["rental_yoy", "National", "Labour"])
    print(f"Merged quarterly observations: {len(merged)}")

    # Add incumbent info
    merged["incumbent"] = merged.index.map(get_incumbent)
    merged["incumbent_vote"] = merged.apply(
        lambda r: r[r["incumbent"]] if pd.notna(r["incumbent"]) and r["incumbent"] in r.index else np.nan,
        axis=1
    )

    # Changes
    for col in ["National", "Labour", "Green", "NZ First", "ACT"]:
        merged[f"d_{col}"] = merged[col].diff()
    merged["d_inc"] = merged["incumbent_vote"].diff()

    # ── 1. Level correlations ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("1. LEVEL CORRELATIONS: Rental inflation vs party support")
    print(f"{'=' * 70}")

    print(f"\n  {'Party':>10s}  {'r (rental YoY)':>15s}  {'p':>8s}  {'n':>5s}")
    for party in ["National", "Labour", "Green", "NZ First", "ACT"]:
        sub = merged.dropna(subset=["rental_yoy", party])
        if len(sub) > 10:
            r, p = pearsonr(sub["rental_yoy"], sub[party])
            print(f"  {party:>10s}  {r:>+15.3f}  {p:>8.4f}  {len(sub):>5d}")

    # ── 2. Incumbent-specific effect ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("2. RENTAL INFLATION → INCUMBENT SUPPORT")
    print(f"{'=' * 70}")

    inc_sub = merged.dropna(subset=["rental_yoy", "incumbent_vote"])
    if len(inc_sub) > 10:
        r, p = pearsonr(inc_sub["rental_yoy"], inc_sub["incumbent_vote"])
        print(f"\n  Overall: r = {r:+.3f}, p = {p:.4f}, n = {len(inc_sub)}")

    # By government
    for party in ["Labour", "National"]:
        sub = merged[merged["incumbent"] == party].dropna(subset=["rental_yoy", "incumbent_vote"])
        if len(sub) > 5:
            r, p = pearsonr(sub["rental_yoy"], sub["incumbent_vote"])
            print(f"  {party} in govt: r = {r:+.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 3. Change correlations ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("3. CHANGE CORRELATIONS: Δrental → Δvote")
    print(f"{'=' * 70}")

    merged["d_rental_yoy"] = merged["rental_yoy"].diff()
    print(f"\n  {'Party':>10s}  {'r (Δrent→Δvote)':>16s}  {'p':>8s}  {'n':>5s}")
    for party in ["National", "Labour", "Green", "NZ First", "ACT"]:
        col = f"d_{party}"
        sub = merged.dropna(subset=["d_rental_yoy", col])
        if len(sub) > 10:
            r, p = pearsonr(sub["d_rental_yoy"], sub[col])
            print(f"  {party:>10s}  {r:>+16.3f}  {p:>8.4f}  {len(sub):>5d}")

    # ── 4. Compare rental vs mortgage effects ────────────────────────
    print(f"\n{'=' * 70}")
    print("4. RENTAL vs MORTGAGE: Which pocketbook indicator is stronger?")
    print(f"{'=' * 70}")

    # Load mortgage data if available
    try:
        mortgage = pd.read_excel("raw_data/hb20.xlsx", skiprows=4)
        # Get the 2-year fixed rate
        mort_col = mortgage.columns[6]  # 2yr fixed
        mortgage = mortgage[["Series Id", mort_col]].rename(columns={"Series Id": "date", mort_col: "mortgage_rate"})
        mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
        mortgage = mortgage.dropna(subset=["date", "mortgage_rate"])
        mortgage = mortgage.set_index("date")
        mortgage["mortgage_rate"] = pd.to_numeric(mortgage["mortgage_rate"], errors="coerce")

        # Resample to quarterly
        mort_q = mortgage.resample("QS").mean()

        # Merge with our data
        compare = merged.join(mort_q, how="inner").dropna(subset=["rental_yoy", "mortgage_rate", "incumbent_vote"])

        if len(compare) > 10:
            r_rent, p_rent = pearsonr(compare["rental_yoy"], compare["incumbent_vote"])
            r_mort, p_mort = pearsonr(compare["mortgage_rate"], compare["incumbent_vote"])
            print(f"\n  Rental YoY → incumbent vote: r = {r_rent:+.3f}, p = {p_rent:.4f}")
            print(f"  Mortgage rate → incumbent vote: r = {r_mort:+.3f}, p = {p_mort:.4f}")
            print(f"\n  → {'Mortgage' if abs(r_mort) > abs(r_rent) else 'Rental'} is the stronger predictor")

            # Correlation between the two indicators
            r_rm, p_rm = pearsonr(compare["rental_yoy"], compare["mortgage_rate"])
            print(f"  Rental-mortgage correlation: r = {r_rm:+.3f}, p = {p_rm:.4f}")
    except Exception as e:
        print(f"\n  Could not load mortgage data: {e}")

    # ── 5. Rental inflation timeline ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("5. RENTAL INFLATION TIMELINE (YoY %)")
    print(f"{'=' * 70}")

    recent = rental[rental.index >= "2017-01-01"]
    print(f"\n  {'Quarter':>10s}  {'Index':>7s}  {'YoY %':>7s}")
    for date, row in recent.iterrows():
        if pd.notna(row["rental_yoy"]):
            print(f"  {date.strftime('%Y-Q%q'):>10s}  {row['rental_index']:>7.0f}  {row['rental_yoy']:>+6.1f}%")

    # ── 6. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Rental inflation time series with party polling
    ax = axes[0, 0]
    ax.plot(merged.index, merged["rental_yoy"], color="#E67E22", linewidth=2, label="Rental CPI YoY %")
    ax.set_ylabel("Rental inflation (%)", color="#E67E22")
    axr = ax.twinx()
    axr.plot(merged.index, merged["Labour"], color="#d32f2f", linewidth=1.5, alpha=0.7, label="Labour")
    axr.plot(merged.index, merged["National"], color="#1565c0", linewidth=1.5, alpha=0.7, label="National")
    axr.set_ylabel("Party vote %")
    ax.set_title("Rental Inflation vs Party Support")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Panel 2: Scatter — rental inflation vs Labour
    ax = axes[0, 1]
    sub = merged.dropna(subset=["rental_yoy", "Labour"])
    ax.scatter(sub["rental_yoy"], sub["Labour"], color="#d32f2f", alpha=0.5)
    if len(sub) > 2:
        m, b = np.polyfit(sub["rental_yoy"], sub["Labour"], 1)
        xr = np.linspace(sub["rental_yoy"].min(), sub["rental_yoy"].max(), 50)
        ax.plot(xr, m * xr + b, "k--", alpha=0.5)
        r, _ = pearsonr(sub["rental_yoy"], sub["Labour"])
        ax.set_title(f"Rental Inflation vs Labour (r={r:.3f})")
    ax.set_xlabel("Rental CPI YoY %")
    ax.set_ylabel("Labour vote %")

    # Panel 3: Scatter — rental inflation vs incumbent
    ax = axes[1, 0]
    sub = merged.dropna(subset=["rental_yoy", "incumbent_vote"])
    colors = ["#d32f2f" if inc == "Labour" else "#1565c0"
              for inc in sub["incumbent"]]
    ax.scatter(sub["rental_yoy"], sub["incumbent_vote"], c=colors, alpha=0.5)
    if len(sub) > 2:
        r, _ = pearsonr(sub["rental_yoy"], sub["incumbent_vote"])
        ax.set_title(f"Rental Inflation vs Incumbent (r={r:.3f})")
    ax.set_xlabel("Rental CPI YoY %")
    ax.set_ylabel("Incumbent vote %")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d32f2f", alpha=0.7, label="Labour govt"),
        Patch(facecolor="#1565c0", alpha=0.7, label="National govt"),
    ], fontsize=8)

    # Panel 4: Rental index time series
    ax = axes[1, 1]
    ax.plot(rental.index, rental["rental_index"], color="#E67E22", linewidth=2)
    ax.fill_between(rental.index, rental["rental_index"], alpha=0.1, color="#E67E22")
    ax.set_ylabel("CPI Rental Index")
    ax.set_title("NZ Rental Price Index (Stats NZ)")
    # Mark government changes
    for start, _, party in GOVERNMENTS:
        ax.axvline(pd.to_datetime(start), ls="--", color="gray", lw=0.5, alpha=0.5)

    plt.tight_layout()
    outpath = REPORTS_DIR / "rental_vs_polling.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
