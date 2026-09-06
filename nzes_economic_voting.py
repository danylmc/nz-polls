#!/usr/bin/env python3
"""
NZES Individual-Level Economic Voting Analysis.

Tests Fiorina (1981) vs MacKuen et al. (1992):
  - Retrospective: Do voters punish/reward based on past economic performance?
  - Prospective: Do voters choose based on expected future economic conditions?
  - Sociotropic vs egotropic: Country economy vs personal finances?

Also tests whether economic perceptions are filtered through partisan lenses
(perceptual screen hypothesis).
"""

import sqlite3
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"

DB_PATH = DATA_DIR / "nzesdb.sqlite"

# ── Variable mapping per NZES year ─────────────────────────────────────
# Each entry: (table, retro_national, retro_personal, prospective, party_vote, incumbent_party)
# None = not available in that year
YEAR_CONFIG = OrderedDict([
    (1990, {
        "table": "nzes_1990",
        "retro_national": "FINNZ",       # NZ economy got better/worse (1=lot better..5=lot worse)
        "retro_personal": "FINHOU",      # household finances
        "prospective": "FFINNZ",         # NZ economy will get better/worse
        "party_vote": "VOT90E",
        "incumbent": "Labour",
        "vote_recode": {1: "Labour", 2: "National", 6: "Green"},
        "econ_scale": "better_to_worse",  # 1=lot better, 5=lot worse
        "econ_dk": [6],
    }),
    (1993, {
        "table": "nzes_1993",
        "retro_national": "NECDIR",      # right direction / wrong direction
        "retro_personal": None,
        "prospective": "NCFUT",          # economy will get better/worse
        "party_vote": "VOT93E",
        "incumbent": "National",
        "vote_recode": {1: "Labour", 2: "National", 3: "Alliance", 4: "NZ First"},
        "econ_scale": "right_wrong",     # 1=right, 2=wrong
        "econ_dk": [3],
        "prosp_scale": "better_to_worse",
        "prosp_dk": [6],
    }),
    (1996, {
        "table": "nzes_1996",
        "retro_national": "ECSAT",       # economy very good..very bad
        "retro_personal": None,
        "prospective": "PCFUT",
        "party_vote": "VOT96P",
        "incumbent": "National",
        "vote_recode": {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT"},
        "econ_scale": "good_to_bad",     # 1=very good..5=very bad
        "econ_dk": [887],
        "prosp_scale": "better_to_worse",
        "prosp_dk": [0, 887],
    }),
    (1999, {
        "table": "nzes_1999_post",
        "retro_national": "CECON",       # economy good/bad
        "retro_personal": None,
        "prospective": "SCFUT",
        "retro_change": "CECCH",         # economy got better/worse (change)
        "party_vote": "PVOTE",
        "incumbent": "National",
        "vote_recode": {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT", 7: "Green"},
        "econ_scale": "good_to_bad_reorder",  # 1=very good,2=good,3=bad,4=very bad,5=neither
        "econ_dk": [88, 99],
        "prosp_scale": "better_to_worse",
        "prosp_dk": [9],
    }),
    (2005, {
        "table": "nzes_2005",
        "retro_national": "ycounow",     # country economy now
        "retro_personal": None,
        "prospective": "ycoufut",
        "party_vote": "yvot05p",
        "incumbent": "Labour",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
        "prosp_scale": "better_to_worse",
        "prosp_dk": [9],
    }),
    (2008, {
        "table": "nzes_2008",
        "retro_national": "zretcon",     # retrospective economy
        "retro_personal": None,
        "prospective": "zprscon",        # prospective economy
        "party_vote": "zvotp",
        "incumbent": "Labour",
        "vote_recode": None,  # Need to figure out coding
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
        "prosp_scale": "better_to_worse",
        "prosp_dk": [9],
    }),
    (2011, {
        "table": "nzes_2011",
        "retro_national": "jecostate",
        "retro_personal": None,
        "prospective": None,
        "party_vote": "jpartyvote",
        "incumbent": "National",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
    }),
    (2014, {
        "table": "nzes_2014",
        "retro_national": "decostate",
        "retro_personal": None,
        "prospective": None,
        "party_vote": "dpartyvote",
        "incumbent": "National",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
    }),
    (2017, {
        "table": "nzes_2017",
        "retro_national": "recostate",
        "retro_personal": None,
        "prospective": None,
        "party_vote": "rpartyvote",
        "incumbent": "National",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
    }),
    (2020, {
        "table": "nzes_2020",
        "retro_national": "C17_2",       # country economy last year
        "retro_personal": "C17_1",       # household finances last year
        "prospective": None,
        "party_vote": "lvpartyvote",
        "incumbent": "Labour",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [6],
    }),
    (2023, {
        "table": "nzes_2023",
        "retro_national": "A14",
        "retro_personal": None,
        "prospective": None,
        "party_vote": "mpartyvote",
        "incumbent": "Labour",
        "vote_recode": {1: "Labour", 2: "National", 3: "Green", 5: "NZ First", 4: "ACT"},
        "econ_scale": "better_to_worse",
        "econ_dk": [9],
    }),
])


def standardise_econ(series, scale_type, dk_values):
    """Convert economy perception to standard -2 to +2 scale (positive = better)."""
    s = series.copy().astype(float)
    # Remove DK/NA
    for dk in dk_values:
        s = s.replace(dk, np.nan)

    if scale_type == "better_to_worse":
        # 1=lot better..5=lot worse → flip to +2..−2
        s = s.replace({1: 2, 2: 1, 3: 0, 4: -1, 5: -2})
    elif scale_type == "good_to_bad":
        # 1=very good..5=very bad → +2..−2
        s = s.replace({1: 2, 2: 1, 3: 0, 4: -1, 5: -2})
    elif scale_type == "right_wrong":
        # 1=right direction, 2=wrong → +1, -1
        s = s.replace({1: 1, 2: -1})
    elif scale_type == "good_to_bad_reorder":
        # 1=very good, 2=good, 3=bad, 4=very bad, 5=neither
        s = s.replace({1: 2, 2: 1, 5: 0, 3: -1, 4: -2})

    return s


def load_year_data(year, config):
    """Load and standardise data for one NZES year."""
    conn = sqlite3.connect(DB_PATH)
    table = config["table"]

    # Build column list
    cols = [config["party_vote"]]
    if config["retro_national"]:
        cols.append(config["retro_national"])
    if config.get("retro_personal"):
        cols.append(config["retro_personal"])
    if config.get("prospective"):
        cols.append(config["prospective"])
    if config.get("retro_change"):
        cols.append(config["retro_change"])

    col_str = ", ".join([f'"{c}"' for c in cols])
    df = pd.read_sql(f'SELECT {col_str} FROM "{table}"', conn)
    conn.close()

    result = pd.DataFrame()

    # Party vote → incumbent vote (1/0)
    vote_col = config["party_vote"]
    if config["vote_recode"]:
        result["party"] = df[vote_col].map(config["vote_recode"])
    else:
        # For 2008, zvotp is binary (0=didn't vote, 1=voted) - need different approach
        return None

    result["voted_incumbent"] = (result["party"] == config["incumbent"]).astype(float)
    result["voted_national"] = (result["party"] == "National").astype(float)
    result["voted_labour"] = (result["party"] == "Labour").astype(float)

    # Retrospective national economy
    if config["retro_national"]:
        result["retro"] = standardise_econ(
            df[config["retro_national"]],
            config["econ_scale"],
            config["econ_dk"]
        )

    # Retrospective personal
    if config.get("retro_personal"):
        pers_dk = config.get("pers_dk", config["econ_dk"])
        result["retro_personal"] = standardise_econ(
            df[config["retro_personal"]],
            config["econ_scale"],
            pers_dk
        )

    # Prospective
    if config.get("prospective"):
        prosp_scale = config.get("prosp_scale", config["econ_scale"])
        prosp_dk = config.get("prosp_dk", config["econ_dk"])
        result["prospective"] = standardise_econ(
            df[config["prospective"]],
            prosp_scale,
            prosp_dk
        )

    # Change retrospective (1999 only)
    if config.get("retro_change"):
        result["retro_change"] = standardise_econ(
            df[config["retro_change"]],
            "better_to_worse",
            config.get("econ_dk", [])
        )

    result["year"] = year
    result["incumbent"] = config["incumbent"]

    return result


def main():
    print("=" * 70)
    print("NZES INDIVIDUAL-LEVEL ECONOMIC VOTING ANALYSIS")
    print("Fiorina (1981) retrospective vs MacKuen et al. (1992) prospective")
    print("=" * 70)

    all_data = []
    retro_results = []
    prosp_results = []

    # ── 1. Year-by-year retrospective voting ─────────────────────────
    print(f"\n{'=' * 70}")
    print("1. RETROSPECTIVE ECONOMIC VOTING BY ELECTION")
    print(f"{'=' * 70}")
    print(f"\n  {'Year':>6s}  {'Incumbent':>10s}  {'Retro→Inc r':>12s}  {'p':>8s}  {'n':>6s}  {'Better→Inc%':>12s}  {'Worse→Inc%':>11s}")

    for year, config in YEAR_CONFIG.items():
        df = load_year_data(year, config)
        if df is None:
            continue

        sub = df.dropna(subset=["retro", "voted_incumbent"])
        sub = sub[sub["party"].notna()]

        if len(sub) < 50:
            continue

        r, p = pearsonr(sub["retro"], sub["voted_incumbent"])
        retro_results.append({"year": year, "r": r, "p": p, "n": len(sub),
                              "incumbent": config["incumbent"]})

        # Vote share by perception
        better = sub[sub["retro"] > 0]["voted_incumbent"].mean() * 100
        worse = sub[sub["retro"] < 0]["voted_incumbent"].mean() * 100

        print(f"  {year:>6d}  {config['incumbent']:>10s}  {r:>+12.3f}  {p:>8.4f}  {len(sub):>6d}  {better:>11.1f}%  {worse:>10.1f}%")

        all_data.append(df)

    retro_df = pd.DataFrame(retro_results)
    print(f"\n  Mean |r| across elections: {retro_df['r'].abs().mean():.3f}")
    print(f"  Elections with p < 0.05: {(retro_df['p'] < 0.05).sum()}/{len(retro_df)}")
    print(f"  All positive? {(retro_df['r'] > 0).all()} (retro better → more incumbent votes)")

    # ── 2. Retrospective vs Prospective (years with both) ────────────
    print(f"\n{'=' * 70}")
    print("2. RETROSPECTIVE vs PROSPECTIVE: HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 70}")

    prosp_years = [y for y, c in YEAR_CONFIG.items() if c.get("prospective")]
    print(f"\n  Years with BOTH retrospective and prospective: {prosp_years}")
    print(f"\n  {'Year':>6s}  {'Retro→Inc r':>12s}  {'Prosp→Inc r':>12s}  {'Winner':>15s}")

    for year in prosp_years:
        config = YEAR_CONFIG[year]
        df = load_year_data(year, config)
        if df is None:
            continue

        sub = df.dropna(subset=["retro", "prospective", "voted_incumbent"])
        sub = sub[sub["party"].notna()]
        if len(sub) < 50:
            continue

        r_retro, p_retro = pearsonr(sub["retro"], sub["voted_incumbent"])
        r_prosp, p_prosp = pearsonr(sub["prospective"], sub["voted_incumbent"])

        winner = "Retrospective" if abs(r_retro) > abs(r_prosp) else "Prospective"
        print(f"  {year:>6d}  {r_retro:>+12.3f}  {r_prosp:>+12.3f}  {winner:>15s}")

        prosp_results.append({
            "year": year, "r_retro": r_retro, "r_prosp": r_prosp,
            "p_retro": p_retro, "p_prosp": p_prosp, "n": len(sub)
        })

    if prosp_results:
        prosp_df = pd.DataFrame(prosp_results)
        retro_wins = (prosp_df["r_retro"].abs() > prosp_df["r_prosp"].abs()).sum()
        print(f"\n  Retrospective wins: {retro_wins}/{len(prosp_df)} elections")
        print(f"  Mean |r| retrospective: {prosp_df['r_retro'].abs().mean():.3f}")
        print(f"  Mean |r| prospective:   {prosp_df['r_prosp'].abs().mean():.3f}")

    # ── 3. Sociotropic vs Egotropic (years with both) ────────────────
    print(f"\n{'=' * 70}")
    print("3. SOCIOTROPIC vs EGOTROPIC (national economy vs personal finances)")
    print(f"{'=' * 70}")

    socio_years = [y for y, c in YEAR_CONFIG.items() if c.get("retro_personal")]
    print(f"\n  Years with both sociotropic and egotropic: {socio_years}")

    for year in socio_years:
        config = YEAR_CONFIG[year]
        df = load_year_data(year, config)
        if df is None:
            continue

        sub = df.dropna(subset=["retro", "retro_personal", "voted_incumbent"])
        sub = sub[sub["party"].notna()]
        if len(sub) < 50:
            continue

        r_soc, p_soc = pearsonr(sub["retro"], sub["voted_incumbent"])
        r_ego, p_ego = pearsonr(sub["retro_personal"], sub["voted_incumbent"])

        print(f"\n  {year}: Sociotropic r={r_soc:+.3f} (p={p_soc:.4f}), Egotropic r={r_ego:+.3f} (p={p_ego:.4f})")
        print(f"  → {'Sociotropic' if abs(r_soc) > abs(r_ego) else 'Egotropic'} dominates")

    # ── 4. Partisan perceptual screen ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("4. PARTISAN PERCEPTUAL SCREEN")
    print("   Do incumbent voters see the economy more positively?")
    print(f"{'=' * 70}")

    print(f"\n  {'Year':>6s}  {'Inc voters retro':>16s}  {'Opp voters retro':>16s}  {'Gap':>6s}  {'p':>8s}")

    pooled_inc = []
    pooled_opp = []

    for year, config in YEAR_CONFIG.items():
        df = load_year_data(year, config)
        if df is None:
            continue

        sub = df.dropna(subset=["retro", "voted_incumbent"])
        sub = sub[sub["party"].notna()]
        if len(sub) < 50:
            continue

        inc_mean = sub[sub["voted_incumbent"] == 1]["retro"].mean()
        opp_mean = sub[sub["voted_incumbent"] == 0]["retro"].mean()
        gap = inc_mean - opp_mean

        pooled_inc.extend(sub[sub["voted_incumbent"] == 1]["retro"].dropna().tolist())
        pooled_opp.extend(sub[sub["voted_incumbent"] == 0]["retro"].dropna().tolist())

        from scipy.stats import ttest_ind
        t, p = ttest_ind(
            sub[sub["voted_incumbent"] == 1]["retro"].dropna(),
            sub[sub["voted_incumbent"] == 0]["retro"].dropna()
        )
        print(f"  {year:>6d}  {inc_mean:>+16.2f}  {opp_mean:>+16.2f}  {gap:>+6.2f}  {p:>8.4f}")

    pooled_gap = np.mean(pooled_inc) - np.mean(pooled_opp)
    print(f"\n  Pooled gap: {pooled_gap:+.2f} scale points (incumbent voters always see economy more positively)")

    # ── 5. Asymmetry test: does economic pain hurt more than gain helps?
    print(f"\n{'=' * 70}")
    print("5. ASYMMETRY: Does economic pain hurt incumbents more than gain helps?")
    print(f"{'=' * 70}")

    all_df = pd.concat(all_data, ignore_index=True)
    valid = all_df.dropna(subset=["retro", "voted_incumbent"])
    valid = valid[valid["party"].notna()]

    # Split into negative perception vs positive
    neg = valid[valid["retro"] < 0]
    pos = valid[valid["retro"] > 0]
    neut = valid[valid["retro"] == 0]

    inc_neg = neg["voted_incumbent"].mean() * 100
    inc_pos = pos["voted_incumbent"].mean() * 100
    inc_neut = neut["voted_incumbent"].mean() * 100

    print(f"\n  Economy got worse: {inc_neg:.1f}% voted incumbent (n={len(neg)})")
    print(f"  Economy stayed same: {inc_neut:.1f}% voted incumbent (n={len(neut)})")
    print(f"  Economy got better: {inc_pos:.1f}% voted incumbent (n={len(pos)})")

    pain_effect = inc_neut - inc_neg
    gain_effect = inc_pos - inc_neut
    print(f"\n  Pain effect (neutral→worse): {pain_effect:+.1f}pp drop")
    print(f"  Gain effect (neutral→better): {gain_effect:+.1f}pp gain")
    print(f"  Asymmetry ratio: {pain_effect/gain_effect:.2f}x (>1 = pain hurts more than gain helps)")

    # ── 6. Cross-election trend: is economic voting strengthening? ───
    print(f"\n{'=' * 70}")
    print("6. TREND: Is economic voting getting stronger over time?")
    print(f"{'=' * 70}")

    if len(retro_df) > 3:
        r, p = pearsonr(retro_df["year"], retro_df["r"].abs())
        print(f"\n  Correlation of year with |retro→incumbent r|: r={r:.3f}, p={p:.4f}")
        print(f"  → {'Strengthening' if r > 0 and p < 0.1 else 'Weakening' if r < 0 and p < 0.1 else 'No significant trend'}")

    # ── 7. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Retrospective correlation by year
    ax = axes[0, 0]
    colors = ["#d32f2f" if config["incumbent"] == "Labour" else "#1565c0"
              for _, config in YEAR_CONFIG.items() if _ in retro_df["year"].values]
    ax.bar(retro_df["year"].astype(str), retro_df["r"], color=colors, alpha=0.7)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("r (retro economy → incumbent vote)")
    ax.set_title("Retrospective Economic Voting by Election")
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color="black", lw=0.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#d32f2f", alpha=0.7, label="Labour incumbent"),
        Patch(facecolor="#1565c0", alpha=0.7, label="National incumbent"),
    ], fontsize=8)

    # Panel 2: Retro vs Prospective comparison
    ax = axes[0, 1]
    if prosp_results:
        prosp_df = pd.DataFrame(prosp_results)
        x = np.arange(len(prosp_df))
        w = 0.35
        ax.bar(x - w/2, prosp_df["r_retro"].abs(), w, color="#2C3E50", alpha=0.7, label="Retrospective")
        ax.bar(x + w/2, prosp_df["r_prosp"].abs(), w, color="#E67E22", alpha=0.7, label="Prospective")
        ax.set_xticks(x)
        ax.set_xticklabels(prosp_df["year"].astype(str))
        ax.set_ylabel("|r| (economy → incumbent vote)")
        ax.set_title("Retrospective vs Prospective |r|")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    # Panel 3: Incumbent vote rate by economic perception (pooled)
    ax = axes[1, 0]
    labels = ["Lot worse", "Little worse", "Same", "Little better", "Lot better"]
    values = [-2, -1, 0, 1, 2]
    rates = []
    ns = []
    for v in values:
        sub = valid[valid["retro"] == v]
        rate = sub["voted_incumbent"].mean() * 100 if len(sub) > 0 else 0
        rates.append(rate)
        ns.append(len(sub))
    bar_colors = ["#E74C3C", "#E67E22", "#95A5A6", "#27AE60", "#2ECC71"]
    bars = ax.bar(labels, rates, color=bar_colors, alpha=0.7)
    ax.set_ylabel("% voted for incumbent (%)")
    ax.set_title("Incumbent Vote by Economic Perception (all elections pooled)")
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'n={n}', ha='center', va='bottom', fontsize=7)

    # Panel 4: Partisan perceptual screen
    ax = axes[1, 1]
    years_list = []
    inc_means = []
    opp_means = []
    for year, config in YEAR_CONFIG.items():
        df = load_year_data(year, config)
        if df is None:
            continue
        sub = df.dropna(subset=["retro", "voted_incumbent"])
        sub = sub[sub["party"].notna()]
        if len(sub) < 50:
            continue
        years_list.append(str(year))
        inc_means.append(sub[sub["voted_incumbent"] == 1]["retro"].mean())
        opp_means.append(sub[sub["voted_incumbent"] == 0]["retro"].mean())

    x = np.arange(len(years_list))
    ax.plot(x, inc_means, "o-", color="#27AE60", label="Incumbent voters", linewidth=2)
    ax.plot(x, opp_means, "s-", color="#E74C3C", label="Opposition voters", linewidth=2)
    ax.fill_between(x, inc_means, opp_means, alpha=0.1, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(years_list, rotation=45)
    ax.set_ylabel("Mean economic perception (-2 to +2)")
    ax.set_title("Partisan Perceptual Screen")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = REPORTS_DIR / "nzes_economic_voting.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
