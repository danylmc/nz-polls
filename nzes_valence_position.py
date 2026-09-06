#!/usr/bin/env python3
"""
NZES Valence vs Position Voting Analysis.

Tests Stokes (1963) / Clarke et al. (2009):
  Are NZ elections decided by ideology (position) or perceived competence (valence)?

Position model: voters choose the party closest to their L-R position.
Valence model: voters choose the party whose leader they rate as more competent.

Method: Logistic regression of National vs Labour vote on:
  1. L-R distance to party (|self - party placement|)
  2. Leader competence differential (govt leader comp - opp leader comp)
Compare pseudo-R² contributions to determine relative importance.

Years available for full test: 2014, 2017, 2020, 2023 (have both L-R + competence)
Years available for position-only: 1990-2023 (have L-R placement in all years)
"""

import sqlite3
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
DB_PATH = DATA_DIR / "nzesdb.sqlite"

# ── Variable mapping ───────────────────────────────────────────────────
YEAR_CONFIG = OrderedDict([
    (2014, {
        "table": "nzes_2014",
        "self_lr": "dslflr",
        "nat_lr": "dnatlr",
        "lab_lr": "dlablr",
        "nat_leader_comp": "dkeycomp",      # Key competence
        "lab_leader_comp": "dcuncomp",       # Cunliffe competence
        "party_vote": "dpartyvote",
        "vote_nat": 2, "vote_lab": 1,
        "lr_dk": [99], "comp_dk": [9],
        "nat_leader": "Key", "lab_leader": "Cunliffe",
    }),
    (2017, {
        "table": "nzes_2017",
        "self_lr": "rselflr",
        "nat_lr": "rnationallr",
        "lab_lr": "rlabourlr",
        "nat_leader_comp": "renglishcomp",   # English competence
        "lab_leader_comp": "rarderncomp",     # Ardern competence
        "party_vote": "rpartyvote",
        "vote_nat": 2, "vote_lab": 1,
        "lr_dk": [99], "comp_dk": [9],
        "nat_leader": "English", "lab_leader": "Ardern",
    }),
    (2020, {
        "table": "nzes_2020",
        "self_lr": "B8",
        "nat_lr": "B7_2",
        "lab_lr": "B7_1",
        "nat_leader_comp": "B9_1",           # Collins competence
        "lab_leader_comp": "B10_1",          # Ardern competence
        "party_vote": "lvpartyvote",
        "vote_nat": 2, "vote_lab": 1,
        "lr_dk": [11], "comp_dk": [5],
        "nat_leader": "Collins", "lab_leader": "Ardern",
    }),
    (2023, {
        "table": "nzes_2023",
        "self_lr": "B6",
        "nat_lr": "B5b",
        "lab_lr": "B5a",
        "nat_leader_comp": "B7a",            # Luxon competence
        "lab_leader_comp": "B7b",            # Hipkins competence
        "party_vote": "mpartyvote",
        "vote_nat": 2, "vote_lab": 1,
        "lr_dk": [99], "comp_dk": [9],
        "nat_leader": "Luxon", "lab_leader": "Hipkins",
    }),
])

# Extended years for position-only analysis (L-R self + party, no competence)
LR_ONLY_CONFIG = OrderedDict([
    (1990, {"table": "nzes_1990", "self_lr": "SCALEP", "nat_lr": "SCALENT", "lab_lr": "SCALELB",
            "party_vote": "VOT90E", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99]}),
    (1993, {"table": "nzes_1993", "self_lr": "NLSCALE", "nat_lr": "LSCNAT", "lab_lr": "LSCLAB",
            "party_vote": "VOT93E", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99]}),
    (1996, {"table": "nzes_1996", "self_lr": "CSCAL", "nat_lr": "CSCNAT", "lab_lr": "CSCLAB",
            "party_vote": "VOT96P", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99, 887]}),
    (1999, {"table": "nzes_1999_post", "self_lr": "SCALE", "nat_lr": "SCNAT", "lab_lr": "SCLAB",
            "party_vote": "PVOTE", "vote_nat": 3, "vote_lab": 2, "lr_dk": [99]}),
    (2002, {"table": "nzes_2002", "self_lr": "WLRSCLE", "nat_lr": "WNTSCL", "lab_lr": "WLBSCL",
            "party_vote": "WVOT02P", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99]}),
    (2005, {"table": "nzes_2005", "self_lr": "ylscale", "nat_lr": "ynatscl", "lab_lr": "ylabscl",
            "party_vote": "yvot05p", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99]}),
    (2008, {"table": "nzes_2008", "self_lr": "zscale", "nat_lr": "zsclnat", "lab_lr": "zscllab",
            "party_vote": "zvotp", "vote_nat": None, "vote_lab": None, "lr_dk": [99]}),
    (2011, {"table": "nzes_2011", "self_lr": "jslflr", "nat_lr": "jnatlr", "lab_lr": "jlablr",
            "party_vote": "jpartyvote", "vote_nat": 2, "vote_lab": 1, "lr_dk": [99]}),
])


def load_data(year, config, need_comp=False):
    """Load NZES data for valence/position analysis."""
    conn = sqlite3.connect(DB_PATH)
    table = config["table"]

    cols = [config["party_vote"], config["self_lr"], config["nat_lr"], config["lab_lr"]]
    if need_comp:
        cols.extend([config["nat_leader_comp"], config["lab_leader_comp"]])

    col_str = ", ".join([f'"{c}"' for c in cols])
    df = pd.read_sql(f'SELECT {col_str} FROM "{table}"', conn)
    conn.close()

    result = pd.DataFrame()

    # Party vote → National (1) vs Labour (0)
    vote_col = config["party_vote"]
    result["voted_nat"] = np.where(
        df[vote_col] == config["vote_nat"], 1,
        np.where(df[vote_col] == config["vote_lab"], 0, np.nan)
    )

    # L-R self placement (0-10)
    for var_key, out_key in [("self_lr", "self_lr"), ("nat_lr", "nat_lr"), ("lab_lr", "lab_lr")]:
        s = df[config[var_key]].astype(float)
        for dk in config["lr_dk"]:
            s = s.replace(dk, np.nan)
        # Filter to 0-10 range
        s = s.where((s >= 0) & (s <= 10))
        result[out_key] = s

    # L-R distance to National and Labour
    result["dist_nat"] = (result["self_lr"] - result["nat_lr"]).abs()
    result["dist_lab"] = (result["self_lr"] - result["lab_lr"]).abs()
    # Proximity: closer to National = positive
    result["lr_proximity"] = result["dist_lab"] - result["dist_nat"]

    if need_comp:
        # Competence (1=very well..4=not at all → flip so higher=more competent)
        for var_key, out_key in [("nat_leader_comp", "nat_comp"), ("lab_leader_comp", "lab_comp")]:
            s = df[config[var_key]].astype(float)
            for dk in config.get("comp_dk", []):
                s = s.replace(dk, np.nan)
            # Flip: 1→4, 2→3, 3→2, 4→1 (higher = more competent)
            s = 5 - s
            s = s.where((s >= 1) & (s <= 4))
            result[out_key] = s

        # Competence differential: National leader minus Labour leader
        result["comp_diff"] = result["nat_comp"] - result["lab_comp"]

    return result


def main():
    print("=" * 70)
    print("NZES VALENCE vs POSITION VOTING ANALYSIS")
    print("Stokes (1963) / Clarke et al. (2009)")
    print("=" * 70)

    # ── 1. Position-only analysis across all years ───────────────────
    print(f"\n{'=' * 70}")
    print("1. POSITION (L-R PROXIMITY) → VOTE ACROSS ALL YEARS")
    print(f"{'=' * 70}")

    all_lr_configs = {}
    all_lr_configs.update(LR_ONLY_CONFIG)
    all_lr_configs.update(YEAR_CONFIG)

    lr_results = []
    print(f"\n  {'Year':>6s}  {'Proximity→Nat r':>16s}  {'p':>8s}  {'n':>6s}  {'Mean self-LR':>13s}")

    for year, config in sorted(all_lr_configs.items()):
        if config.get("vote_nat") is None:
            continue  # Skip 2008 (binary vote variable)
        df = load_data(year, config, need_comp=False)
        sub = df.dropna(subset=["voted_nat", "lr_proximity"])
        if len(sub) < 50:
            continue

        r, p = pearsonr(sub["lr_proximity"], sub["voted_nat"])
        mean_lr = sub["self_lr"].mean()
        lr_results.append({"year": year, "r": r, "p": p, "n": len(sub), "mean_lr": mean_lr})
        print(f"  {year:>6d}  {r:>+16.3f}  {p:>8.4f}  {len(sub):>6d}  {mean_lr:>13.1f}")

    lr_df = pd.DataFrame(lr_results)
    print(f"\n  Mean |r| across elections: {lr_df['r'].abs().mean():.3f}")
    print(f"  All significant? {(lr_df['p'] < 0.001).all()}")

    # ── 2. Full valence vs position model (2014-2023) ────────────────
    print(f"\n{'=' * 70}")
    print("2. VALENCE vs POSITION: HEAD-TO-HEAD (2014-2023)")
    print(f"{'=' * 70}")

    comparison_results = []
    print(f"\n  {'Year':>6s}  {'Position r':>11s}  {'Valence r':>10s}  {'Winner':>12s}  {'Leaders':>30s}")

    for year, config in YEAR_CONFIG.items():
        df = load_data(year, config, need_comp=True)
        sub = df.dropna(subset=["voted_nat", "lr_proximity", "comp_diff"])
        if len(sub) < 50:
            continue

        r_pos, p_pos = pearsonr(sub["lr_proximity"], sub["voted_nat"])
        r_val, p_val = pearsonr(sub["comp_diff"], sub["voted_nat"])

        winner = "Position" if abs(r_pos) > abs(r_val) else "Valence"
        leaders = f"{config['nat_leader']} vs {config['lab_leader']}"
        comparison_results.append({
            "year": year, "r_position": r_pos, "r_valence": r_val,
            "p_position": p_pos, "p_valence": p_val, "n": len(sub),
            "leaders": leaders
        })
        print(f"  {year:>6d}  {r_pos:>+11.3f}  {r_val:>+10.3f}  {winner:>12s}  {leaders:>30s}")

    comp_df = pd.DataFrame(comparison_results)
    pos_wins = (comp_df["r_position"].abs() > comp_df["r_valence"].abs()).sum()
    print(f"\n  Position wins: {pos_wins}/{len(comp_df)}")
    print(f"  Mean |r| position: {comp_df['r_position'].abs().mean():.3f}")
    print(f"  Mean |r| valence:  {comp_df['r_valence'].abs().mean():.3f}")

    # ── 3. Logistic regression for pseudo-R² comparison ──────────────
    print(f"\n{'=' * 70}")
    print("3. LOGISTIC REGRESSION: PSEUDO-R² COMPARISON")
    print(f"{'=' * 70}")

    from statsmodels.discrete.discrete_model import Logit
    import statsmodels.api as sm

    print(f"\n  {'Year':>6s}  {'Position R²':>12s}  {'Valence R²':>11s}  {'Combined R²':>12s}  {'Dominant':>10s}")

    for year, config in YEAR_CONFIG.items():
        df = load_data(year, config, need_comp=True)
        sub = df.dropna(subset=["voted_nat", "lr_proximity", "comp_diff"])
        if len(sub) < 50:
            continue

        y = sub["voted_nat"].values

        # Position-only model
        X_pos = sm.add_constant(sub["lr_proximity"].values)
        try:
            mod_pos = Logit(y, X_pos).fit(disp=0)
            r2_pos = mod_pos.prsquared
        except Exception:
            continue

        # Valence-only model
        X_val = sm.add_constant(sub["comp_diff"].values)
        mod_val = Logit(y, X_val).fit(disp=0)
        r2_val = mod_val.prsquared

        # Combined model
        X_both = sm.add_constant(sub[["lr_proximity", "comp_diff"]].values)
        mod_both = Logit(y, X_both).fit(disp=0)
        r2_both = mod_both.prsquared

        dominant = "Position" if r2_pos > r2_val else "Valence"
        print(f"  {year:>6d}  {r2_pos:>12.3f}  {r2_val:>11.3f}  {r2_both:>12.3f}  {dominant:>10s}")

    # ── 4. Competence gap and election outcomes ──────────────────────
    print(f"\n{'=' * 70}")
    print("4. COMPETENCE GAP AND ELECTION OUTCOMES")
    print(f"{'=' * 70}")

    for year, config in YEAR_CONFIG.items():
        df = load_data(year, config, need_comp=True)
        sub = df.dropna(subset=["nat_comp", "lab_comp"])

        nat_mean = sub["nat_comp"].mean()
        lab_mean = sub["lab_comp"].mean()
        gap = nat_mean - lab_mean

        # Election results
        election_results = {
            2014: ("National won", 47.0, 25.1),
            2017: ("Labour won", 44.4, 36.9),
            2020: ("Labour won", 25.6, 50.0),
            2023: ("National won", 38.1, 26.9),
        }
        outcome, nat_result, lab_result = election_results.get(year, ("?", 0, 0))

        print(f"\n  {year}: {config['nat_leader']} comp={nat_mean:.2f}, {config['lab_leader']} comp={lab_mean:.2f}, gap={gap:+.2f}")
        print(f"         Result: {outcome} (Nat {nat_result}%, Lab {lab_result}%)")
        print(f"         Competence leader: {'National' if gap > 0 else 'Labour'} → {'CORRECT' if (gap > 0 and nat_result > lab_result) or (gap < 0 and lab_result > nat_result) else 'WRONG'}")

    # ── 5. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Position correlation across all years
    ax = axes[0, 0]
    ax.bar(lr_df["year"].astype(str), lr_df["r"], color="#3498DB", alpha=0.7)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("r (L-R proximity → National vote)")
    ax.set_title("Positional Voting Strength by Election")
    ax.tick_params(axis='x', rotation=45)

    # Panel 2: Position vs Valence comparison
    ax = axes[0, 1]
    if len(comp_df) > 0:
        x = np.arange(len(comp_df))
        w = 0.35
        ax.bar(x - w/2, comp_df["r_position"].abs(), w, color="#3498DB", alpha=0.7, label="Position (L-R)")
        ax.bar(x + w/2, comp_df["r_valence"].abs(), w, color="#E67E22", alpha=0.7, label="Valence (competence)")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["year"].astype(str))
        ax.set_ylabel("|r| with National vote")
        ax.set_title("Position vs Valence: Correlation Strength")
        ax.legend(fontsize=8)

    # Panel 3: L-R distribution of voters by vote choice (pooled recent elections)
    ax = axes[1, 0]
    all_voters = []
    for year, config in YEAR_CONFIG.items():
        df = load_data(year, config, need_comp=False)
        df["year"] = year
        all_voters.append(df)
    pooled = pd.concat(all_voters, ignore_index=True)
    nat_voters = pooled[pooled["voted_nat"] == 1]["self_lr"].dropna()
    lab_voters = pooled[pooled["voted_nat"] == 0]["self_lr"].dropna()
    bins = np.arange(-0.5, 11.5, 1)
    ax.hist(nat_voters, bins=bins, alpha=0.5, density=True, color="#1565c0", label=f"National (n={len(nat_voters)})")
    ax.hist(lab_voters, bins=bins, alpha=0.5, density=True, color="#d32f2f", label=f"Labour (n={len(lab_voters)})")
    ax.set_xlabel("Left-Right Self-Placement (0=Left, 10=Right)")
    ax.set_ylabel("Density")
    ax.set_title("L-R Distribution by Vote (2014-2023 pooled)")
    ax.legend(fontsize=8)

    # Panel 4: Competence differential → vote probability
    ax = axes[1, 1]
    all_comp = []
    for year, config in YEAR_CONFIG.items():
        df = load_data(year, config, need_comp=True)
        all_comp.append(df)
    pooled_comp = pd.concat(all_comp, ignore_index=True)
    valid = pooled_comp.dropna(subset=["voted_nat", "comp_diff"])
    # Bin comp_diff and plot probability
    valid["comp_bin"] = pd.cut(valid["comp_diff"], bins=[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    grouped = valid.groupby("comp_bin", observed=True)["voted_nat"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 10]
    midpoints = [interval.mid for interval in grouped.index]
    ax.plot(midpoints, grouped["mean"] * 100, "o-", color="#2C3E50", linewidth=2)
    for mp, row in zip(midpoints, grouped.itertuples()):
        ax.annotate(f'n={row.count}', (mp, row.mean * 100 + 2), fontsize=7, ha="center")
    ax.set_xlabel("Competence differential (Nat leader - Lab leader)")
    ax.set_ylabel("% voted National")
    ax.set_title("Competence Gap → Vote Probability (pooled)")
    ax.axhline(50, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)

    plt.tight_layout()
    outpath = REPORTS_DIR / "nzes_valence_position.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
