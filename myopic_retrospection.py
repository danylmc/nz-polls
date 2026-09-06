#!/usr/bin/env python3
"""
Myopic Retrospection Analysis (Achen & Bartels 2016).

Tests whether NZ voters overweight election-year economic conditions
and ignore earlier performance. Compares:
  - Full-term average GDP growth as predictor of incumbent vote change
  - Election-year-only GDP growth as predictor
  - Early-term vs late-term growth

Literature predicts: election-year economy matters more than early-term,
meaning voters are "myopic" retrospectors who only evaluate recent conditions.

Data: Stats NZ quarterly real GDP + election results (1996-2023).
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

# Election results (party vote %)
ELECTION_RESULTS = {
    1996: {"National": 33.84, "Labour": 28.19, "date": "1996-10-12"},
    1999: {"National": 30.50, "Labour": 38.74, "date": "1999-11-27"},
    2002: {"National": 20.93, "Labour": 41.26, "date": "2002-07-27"},
    2005: {"National": 39.10, "Labour": 41.10, "date": "2005-09-17"},
    2008: {"National": 44.93, "Labour": 33.99, "date": "2008-11-08"},
    2011: {"National": 47.31, "Labour": 27.48, "date": "2011-11-26"},
    2014: {"National": 47.04, "Labour": 25.13, "date": "2014-09-20"},
    2017: {"National": 44.45, "Labour": 36.89, "date": "2017-09-23"},
    2020: {"National": 25.58, "Labour": 50.01, "date": "2020-10-17"},
    2023: {"National": 38.93, "Labour": 26.91, "date": "2023-10-14"},
}

# Government terms: (start_date, election_date, incumbent_party, prev_election_year)
TERMS = [
    ("1996-12-01", "1999-11-27", "National", 1996),   # Bolger/Shipley
    ("1999-12-10", "2002-07-27", "Labour", 1999),      # Clark I
    ("2002-08-15", "2005-09-17", "Labour", 2002),      # Clark II
    ("2005-10-17", "2008-11-08", "Labour", 2005),      # Clark III
    ("2008-11-19", "2011-11-26", "National", 2008),    # Key I
    ("2011-12-14", "2014-09-20", "National", 2011),    # Key II
    ("2014-10-20", "2017-09-23", "National", 2014),    # Key/English
    ("2017-10-26", "2020-10-17", "Labour", 2017),      # Ardern I
    ("2020-11-06", "2023-10-14", "Labour", 2020),      # Ardern/Hipkins
]


def load_gdp_quarterly():
    """Load quarterly real GDP (chain volume, seasonally adjusted)."""
    gdp = pd.read_csv(DATA_DIR / "gross-domestic-product-september-2025-quarter.csv")
    real = gdp[gdp["Series_reference"] == "SNEQ.SG02RSC00B15"].copy()
    real["date"] = real["Period"].apply(
        lambda x: pd.to_datetime(f"{int(x)}-{int((x % 1) * 100):02d}-01")
    )
    real = real.sort_values("date").set_index("date")
    real["gdp"] = pd.to_numeric(real["Data_value"], errors="coerce")
    # Quarter-on-quarter growth %
    real["gdp_qoq"] = real["gdp"].pct_change() * 100
    # Year-on-year growth %
    real["gdp_yoy"] = real["gdp"].pct_change(4) * 100
    return real[["gdp", "gdp_qoq", "gdp_yoy"]].dropna()


def load_unemployment():
    """Load unemployment data if available."""
    try:
        econ_files = list(DATA_DIR.glob("economic/unemployment*.csv"))
        if econ_files:
            df = pd.read_csv(econ_files[0])
            return df
    except Exception:
        pass
    return None


def main():
    gdp = load_gdp_quarterly()

    print("=" * 70)
    print("MYOPIC RETROSPECTION ANALYSIS (Achen & Bartels 2016)")
    print("=" * 70)
    print(f"\nGDP data: {gdp.index.min().strftime('%Y-Q%q')} to {gdp.index.max().strftime('%Y-Q%q')}")
    print(f"Observations: {len(gdp)}")

    # ── 1. Build term-level dataset ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("1. TERM-LEVEL ECONOMIC CONDITIONS vs INCUMBENT VOTE CHANGE")
    print(f"{'=' * 70}")

    rows = []
    for start_str, elec_str, incumbent, prev_year in TERMS:
        start = pd.to_datetime(start_str)
        elec = pd.to_datetime(elec_str)
        elec_year = elec.year

        # Get incumbent vote at this election and previous
        if elec_year in ELECTION_RESULTS and prev_year in ELECTION_RESULTS:
            inc_vote_now = ELECTION_RESULTS[elec_year][incumbent]
            inc_vote_prev = ELECTION_RESULTS[prev_year][incumbent]
            inc_change = inc_vote_now - inc_vote_prev
        else:
            continue

        # GDP during the term
        term_gdp = gdp[(gdp.index >= start) & (gdp.index <= elec)]
        if len(term_gdp) < 4:
            continue

        # Full-term average quarterly growth
        full_term_growth = term_gdp["gdp_qoq"].mean()

        # Election year: last 4 quarters before election
        elec_year_start = elec - pd.DateOffset(months=12)
        late_gdp = term_gdp[term_gdp.index >= elec_year_start]
        late_growth = late_gdp["gdp_qoq"].mean() if len(late_gdp) > 0 else np.nan

        # Early term: everything before the election year
        early_gdp = term_gdp[term_gdp.index < elec_year_start]
        early_growth = early_gdp["gdp_qoq"].mean() if len(early_gdp) > 0 else np.nan

        # Cumulative growth over the term
        cumul_growth = ((1 + term_gdp["gdp_qoq"] / 100).prod() - 1) * 100

        # Election-year cumulative
        late_cumul = ((1 + late_gdp["gdp_qoq"] / 100).prod() - 1) * 100 if len(late_gdp) > 0 else np.nan

        # YoY growth at election quarter
        elec_q_yoy = term_gdp["gdp_yoy"].iloc[-1] if len(term_gdp) > 0 else np.nan

        rows.append({
            "election": elec_year,
            "incumbent": incumbent,
            "inc_vote_prev": inc_vote_prev,
            "inc_vote_now": inc_vote_now,
            "inc_change": inc_change,
            "full_term_avg_qoq": full_term_growth,
            "late_term_avg_qoq": late_growth,
            "early_term_avg_qoq": early_growth,
            "cumul_growth": cumul_growth,
            "late_cumul_growth": late_cumul,
            "elec_q_yoy": elec_q_yoy,
            "n_quarters": len(term_gdp),
        })

    df = pd.DataFrame(rows)

    print(f"\n  {'Election':>8s}  {'Incumbent':>10s}  {'ΔVote':>6s}  {'Full Q':>7s}  {'Late Q':>7s}  {'Early Q':>7s}  {'ElecYoY':>8s}")
    for _, r in df.iterrows():
        print(f"  {r['election']:>8.0f}  {r['incumbent']:>10s}  {r['inc_change']:>+5.1f}%  "
              f"{r['full_term_avg_qoq']:>+6.2f}%  {r['late_term_avg_qoq']:>+6.2f}%  "
              f"{r['early_term_avg_qoq']:>+6.2f}%  {r['elec_q_yoy']:>+7.2f}%")

    # ── 2. Correlation tests ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("2. MYOPIC vs CUMULATIVE RETROSPECTION")
    print(f"{'=' * 70}")

    tests = [
        ("Full-term avg QoQ growth", "full_term_avg_qoq"),
        ("Election-year avg QoQ growth", "late_term_avg_qoq"),
        ("Early-term avg QoQ growth", "early_term_avg_qoq"),
        ("Election-quarter YoY growth", "elec_q_yoy"),
        ("Full-term cumulative growth", "cumul_growth"),
        ("Election-year cumulative growth", "late_cumul_growth"),
    ]

    print(f"\n  Predictor → Incumbent vote change:")
    print(f"  {'Predictor':>35s}  {'r':>8s}  {'p':>8s}  {'n':>4s}")
    results = {}
    for label, col in tests:
        sub = df.dropna(subset=[col, "inc_change"])
        if len(sub) >= 5:
            r, p = pearsonr(sub[col], sub["inc_change"])
            results[col] = (r, p, len(sub))
            sig = " *" if p < 0.10 else " **" if p < 0.05 else ""
            print(f"  {label:>35s}  {r:>+7.3f}  {p:>8.3f}  {len(sub):>4d}{sig}")

    # ── 3. Head-to-head: late vs early ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("3. HEAD-TO-HEAD: LATE vs EARLY TERM GROWTH")
    print(f"{'=' * 70}")

    sub = df.dropna(subset=["late_term_avg_qoq", "early_term_avg_qoq", "inc_change"])
    if len(sub) >= 5:
        r_late, p_late = pearsonr(sub["late_term_avg_qoq"], sub["inc_change"])
        r_early, p_early = pearsonr(sub["early_term_avg_qoq"], sub["inc_change"])
        print(f"\n  Late-term growth → ΔVote:   r = {r_late:+.3f}, p = {p_late:.3f}")
        print(f"  Early-term growth → ΔVote:  r = {r_early:+.3f}, p = {p_early:.3f}")
        print(f"\n  → {'MYOPIC (late > early)' if abs(r_late) > abs(r_early) else 'CUMULATIVE (early > late)'}")

        # Williams' test approximation for comparing dependent correlations
        r_12, _ = pearsonr(sub["late_term_avg_qoq"], sub["early_term_avg_qoq"])
        n = len(sub)
        # Steiger's z test for comparing correlated correlations
        z_late = np.arctanh(r_late)
        z_early = np.arctanh(r_early)
        # Approximate test
        diff = abs(r_late) - abs(r_early)
        print(f"  Correlation difference: {diff:+.3f}")
        print(f"  (Note: n={n} elections — too small for formal comparison test)")

    # ── 4. Spearman rank correlations (robust to small N) ─────────────
    print(f"\n{'=' * 70}")
    print("4. RANK CORRELATIONS (Spearman — robust to small N)")
    print(f"{'=' * 70}")

    print(f"\n  {'Predictor':>35s}  {'ρ':>8s}  {'p':>8s}")
    for label, col in tests:
        sub = df.dropna(subset=[col, "inc_change"])
        if len(sub) >= 5:
            rho, p = spearmanr(sub[col], sub["inc_change"])
            sig = " *" if p < 0.10 else " **" if p < 0.05 else ""
            print(f"  {label:>35s}  {rho:>+7.3f}  {p:>8.3f}{sig}")

    # ── 5. Also test with incumbent vote level ────────────────────────
    print(f"\n{'=' * 70}")
    print("5. GDP GROWTH → INCUMBENT VOTE LEVEL (not change)")
    print(f"{'=' * 70}")

    print(f"\n  {'Predictor':>35s}  {'r':>8s}  {'p':>8s}")
    for label, col in [("Election-quarter YoY growth", "elec_q_yoy"),
                       ("Late-term avg QoQ growth", "late_term_avg_qoq"),
                       ("Full-term avg QoQ growth", "full_term_avg_qoq")]:
        sub = df.dropna(subset=[col, "inc_vote_now"])
        if len(sub) >= 5:
            r, p = pearsonr(sub[col], sub["inc_vote_now"])
            print(f"  {label:>35s}  {r:>+7.3f}  {p:>8.3f}")

    # ── 6. Summary & interpretation ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("6. SUMMARY & INTERPRETATION")
    print(f"{'=' * 70}")

    # Check which is stronger
    late_result = results.get("late_term_avg_qoq", (0, 1, 0))
    early_result = results.get("early_term_avg_qoq", (0, 1, 0))
    full_result = results.get("full_term_avg_qoq", (0, 1, 0))
    yoy_result = results.get("elec_q_yoy", (0, 1, 0))

    print(f"\n  Achen & Bartels (2016) predict voters are myopic — they overweight")
    print(f"  the election-year economy and ignore earlier performance.")
    print(f"\n  NZ results (n=9 elections, 1999-2023):")
    print(f"    Full-term GDP → ΔVote:     r = {full_result[0]:+.3f} (p = {full_result[1]:.3f})")
    print(f"    Late-term GDP → ΔVote:     r = {late_result[0]:+.3f} (p = {late_result[1]:.3f})")
    print(f"    Early-term GDP → ΔVote:    r = {early_result[0]:+.3f} (p = {early_result[1]:.3f})")
    print(f"    Election-Q YoY → ΔVote:    r = {yoy_result[0]:+.3f} (p = {yoy_result[1]:.3f})")

    if abs(late_result[0]) > abs(early_result[0]):
        print(f"\n  → MYOPIC retrospection: late-term growth (|r|={abs(late_result[0]):.3f}) predicts")
        print(f"    incumbent fate better than early-term (|r|={abs(early_result[0]):.3f})")
    else:
        print(f"\n  → CUMULATIVE retrospection: early-term growth (|r|={abs(early_result[0]):.3f}) predicts")
        print(f"    incumbent fate better than late-term (|r|={abs(late_result[0]):.3f})")

    any_sig = any(r[1] < 0.10 for r in results.values())
    if not any_sig:
        print(f"\n  CAVEAT: No correlation reaches significance at p<0.10.")
        print(f"  With only 9 elections, statistical power is very low.")
        print(f"  This is consistent with our earlier finding that macro-economic")
        print(f"  fundamentals (GDP, unemployment, CPI) have no significant")
        print(f"  relationship with NZ incumbent polling (Analysis 5).")

    # ── 7. Visualization ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Late-term growth vs vote change
    ax = axes[0, 0]
    sub = df.dropna(subset=["late_term_avg_qoq", "inc_change"])
    colors = ["#d32f2f" if inc == "Labour" else "#1565c0" for inc in sub["incumbent"]]
    ax.scatter(sub["late_term_avg_qoq"], sub["inc_change"], c=colors, s=80, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(f"{r['election']:.0f}", (r["late_term_avg_qoq"], r["inc_change"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    r_val = results.get("late_term_avg_qoq", (0,))[0]
    ax.set_title(f"Election-Year GDP Growth vs Vote Change\n(r={r_val:+.3f})", fontsize=10)
    ax.set_xlabel("Late-term avg quarterly GDP growth (%)")
    ax.set_ylabel("Incumbent vote change (pp)")

    # Panel 2: Early-term growth vs vote change
    ax = axes[0, 1]
    sub = df.dropna(subset=["early_term_avg_qoq", "inc_change"])
    colors = ["#d32f2f" if inc == "Labour" else "#1565c0" for inc in sub["incumbent"]]
    ax.scatter(sub["early_term_avg_qoq"], sub["inc_change"], c=colors, s=80, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(f"{r['election']:.0f}", (r["early_term_avg_qoq"], r["inc_change"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    r_val = results.get("early_term_avg_qoq", (0,))[0]
    ax.set_title(f"Early-Term GDP Growth vs Vote Change\n(r={r_val:+.3f})", fontsize=10)
    ax.set_xlabel("Early-term avg quarterly GDP growth (%)")
    ax.set_ylabel("Incumbent vote change (pp)")

    # Panel 3: Full-term growth vs vote change
    ax = axes[1, 0]
    sub = df.dropna(subset=["full_term_avg_qoq", "inc_change"])
    colors = ["#d32f2f" if inc == "Labour" else "#1565c0" for inc in sub["incumbent"]]
    ax.scatter(sub["full_term_avg_qoq"], sub["inc_change"], c=colors, s=80, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(f"{r['election']:.0f}", (r["full_term_avg_qoq"], r["inc_change"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    r_val = results.get("full_term_avg_qoq", (0,))[0]
    ax.set_title(f"Full-Term GDP Growth vs Vote Change\n(r={r_val:+.3f})", fontsize=10)
    ax.set_xlabel("Full-term avg quarterly GDP growth (%)")
    ax.set_ylabel("Incumbent vote change (pp)")

    # Panel 4: Late vs early comparison bar chart
    ax = axes[1, 1]
    labels = [f"{r['election']:.0f}" for _, r in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35
    sub = df.dropna(subset=["late_term_avg_qoq", "early_term_avg_qoq"])
    ax.bar(x[:len(sub)] - width/2, sub["early_term_avg_qoq"], width,
           label="Early term", color="#3498DB", alpha=0.7)
    ax.bar(x[:len(sub)] + width/2, sub["late_term_avg_qoq"], width,
           label="Late term (election year)", color="#E74C3C", alpha=0.7)
    ax.set_xticks(x[:len(sub)])
    ax.set_xticklabels([f"{e:.0f}" for e in sub["election"]], fontsize=8)
    ax.set_ylabel("Avg quarterly GDP growth (%)")
    ax.set_title("Early vs Late Term GDP Growth by Election")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", ls="--", lw=0.5)

    # Add party color patches
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#d32f2f", alpha=0.7, label="Labour govt"),
        Patch(facecolor="#1565c0", alpha=0.7, label="National govt"),
    ], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Myopic Retrospection Test: Do Voters Only See Election-Year Economy?",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    outpath = REPORTS_DIR / "myopic_retrospection.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
