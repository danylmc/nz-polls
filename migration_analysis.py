#!/usr/bin/env python3
"""
Net migration vs NZ First and major party polling.

Tests Hangartner et al. (2019): Do migration surges predict NZ First gains
and/or shifts in major party support?
Annual World Bank data (1960-2024) vs annual mean polling.
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


def load_migration():
    """Load World Bank net migration data."""
    with open(DATA_DIR / "net_migration.json") as f:
        raw = json.load(f)
    # Structure: [metadata, [records]]
    records = raw[1] if isinstance(raw, list) else raw
    rows = []
    if isinstance(records, list):
        for r in records:
            if r.get("value") is not None:
                rows.append({"year": int(r["date"]), "net_migration": r["value"]})
    df = pd.DataFrame(rows).sort_values("year").set_index("year")
    return df


def load_annual_polls():
    """Load polling data aggregated to annual means."""
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
    for p in parties:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")[parties].mean()
    return annual


def main():
    migration = load_migration()
    polls = load_annual_polls()

    merged = migration.join(polls, how="inner").dropna(subset=["net_migration", "NZ First"])
    print(f"Net migration data: {migration.index.min()} to {migration.index.max()}")
    print(f"Matched years: {len(merged)} ({merged.index.min()} to {merged.index.max()})")

    parties = ["National", "Labour", "Green", "NZ First", "ACT"]

    # ── 1. Level correlations ─────────────────────────────────────────
    print(f"\n=== LEVEL CORRELATIONS (net migration vs annual mean poll) ===")
    for party in parties:
        sub = merged.dropna(subset=[party])
        if len(sub) > 5:
            r, p = pearsonr(sub["net_migration"], sub[party])
            print(f"  {party:>10s}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 2. Change correlations ────────────────────────────────────────
    print(f"\n=== CHANGE CORRELATIONS (Δmigration vs Δpoll) ===")
    merged["d_mig"] = merged["net_migration"].diff()
    for party in parties:
        merged[f"d_{party}"] = merged[party].diff()
    changes = merged.dropna(subset=["d_mig"])

    for party in parties:
        col = f"d_{party}"
        sub = changes.dropna(subset=[col])
        if len(sub) > 5:
            r, p = pearsonr(sub["d_mig"], sub[col])
            print(f"  {party:>10s}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

    # ── 3. Migration vs NZ First (the key test) ──────────────────────
    print(f"\n=== NZ FIRST SPECIFIC ===")
    sub = merged.dropna(subset=["NZ First"])
    r, p = pearsonr(sub["net_migration"], sub["NZ First"])
    print(f"  Level: r = {r:.3f}, p = {p:.4f}")

    # Lagged: does migration in year t predict NZ First in year t+1?
    nzf_data = merged[["net_migration", "NZ First"]].dropna()
    for lag in [0, 1, 2]:
        if lag > 0:
            x = nzf_data["net_migration"].values[:-lag]
            y = nzf_data["NZ First"].values[lag:]
        else:
            x = nzf_data["net_migration"].values
            y = nzf_data["NZ First"].values
        if len(x) > 5:
            r, p = pearsonr(x, y)
            print(f"  Lag {lag} year: r = {r:.3f}, p = {p:.4f}")

    # ── 4. Migration vs Ipsos immigration salience ────────────────────
    try:
        salience = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
        # Check if immigration is tracked (it may not be in the 10 issues)
        if "race_relations" in salience.columns:
            salience["date"] = pd.to_datetime(salience["date"])
            salience["year"] = salience["date"].dt.year
            sal_annual = salience.groupby("year")["race_relations"].mean()
            mig_sal = migration.join(sal_annual.rename("race_sal"), how="inner").dropna()
            if len(mig_sal) > 5:
                r, p = pearsonr(mig_sal["net_migration"], mig_sal["race_sal"])
                print(f"\n  Migration vs race relations salience: r = {r:.3f}, p = {p:.4f}, n = {len(mig_sal)}")
    except Exception as e:
        print(f"\n  Salience analysis skipped: {e}")

    # ── 5. High migration periods vs NZ First support ─────────────────
    print(f"\n=== HIGH vs LOW MIGRATION PERIODS ===")
    median_mig = merged["net_migration"].median()
    high = merged[merged["net_migration"] > median_mig]
    low = merged[merged["net_migration"] <= median_mig]
    print(f"  High migration years (>{median_mig:.0f}): NZ First mean = {high['NZ First'].mean():.1f}%")
    print(f"  Low migration years (≤{median_mig:.0f}):  NZ First mean = {low['NZ First'].mean():.1f}%")

    # ── 6. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Migration time series with NZ First
    ax = axes[0, 0]
    axr = ax.twinx()
    ax.bar(merged.index, merged["net_migration"] / 1000, color="#3498DB", alpha=0.6, label="Net migration (000s)")
    axr.plot(merged.index, merged["NZ First"], color="#2C3E50", linewidth=2, label="NZ First %")
    ax.set_ylabel("Net migration (000s)", color="#3498DB")
    axr.set_ylabel("NZ First %", color="#2C3E50")
    ax.set_title("Net Migration vs NZ First Support")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Panel 2: Scatter migration vs NZ First
    ax = axes[0, 1]
    ax.scatter(merged["net_migration"] / 1000, merged["NZ First"], alpha=0.6, color="#2C3E50")
    m, b = np.polyfit(merged["net_migration"] / 1000, merged["NZ First"], 1)
    xr = np.linspace(merged["net_migration"].min() / 1000, merged["net_migration"].max() / 1000, 50)
    ax.plot(xr, m * xr + b, "k--", alpha=0.7)
    ax.set_xlabel("Net migration (000s)")
    ax.set_ylabel("NZ First %")
    r_nzf = pearsonr(merged["net_migration"], merged["NZ First"])
    ax.set_title(f"Migration vs NZ First (r={r_nzf[0]:.3f})")

    # Panel 3: Migration vs National and Labour
    ax = axes[1, 0]
    ax.scatter(merged["net_migration"] / 1000, merged["National"], alpha=0.5, color="#1565c0", label="National")
    ax.scatter(merged["net_migration"] / 1000, merged["Labour"], alpha=0.5, color="#d32f2f", label="Labour")
    ax.set_xlabel("Net migration (000s)")
    ax.set_ylabel("Party vote %")
    ax.set_title("Migration vs Major Party Support")
    ax.legend(fontsize=8)

    # Panel 4: Migration bar chart by year with color coding
    ax = axes[1, 1]
    colors = ["#27AE60" if v > 0 else "#E74C3C" for v in merged["net_migration"]]
    ax.bar(merged.index, merged["net_migration"] / 1000, color=colors, alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net migration (000s)")
    ax.set_title("NZ Net Migration (positive = inflow)")
    ax.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    outpath = REPORTS_DIR / "migration_vs_nzfirst.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
