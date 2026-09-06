#!/usr/bin/env python3
"""
NZES Strategic Voting Under MMP Analysis.

Tests Cox (1997): Do voters desert minor parties near the 5% threshold?
Under MMP, parties below 5% get no list seats (unless winning an electorate).
Rational voters should desert hopeless parties, concentrating support on viable ones.

Tests:
1. Split-ticket voting: electorate vs party vote differences
2. "Wasted vote" fear: do voters whose preferred party is small vote for someone else?
3. Party liked most vs party voted for: strategic desertion rates
4. Aggregate test: does minor party support drop near elections (squeeze)?
"""

import sqlite3
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
DB_PATH = DATA_DIR / "nzesdb.sqlite"


def recode_party(value, year_recode):
    """Map numeric vote code to party name."""
    return year_recode.get(value)


# ── Year configurations ───────────────────────────────────────────────
YEAR_CONFIG = OrderedDict([
    (1996, {
        "table": "nzes_1996",
        "party_vote": "VOT96P",
        "elec_vote": "VOT96E",
        "most_liked": None,
        "party_recode": {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT",
                         6: "Christian", 7: "United"},
        "elec_recode": {1: "Labour", 2: "National", 3: "NZ First", 4: "Alliance", 5: "ACT",
                        6: "Christian", 7: "United"},
    }),
    (1999, {
        "table": "nzes_1999_post",
        "party_vote": "PVOTE",
        "elec_vote": "EVOTE",
        "most_liked": None,
        "party_recode": {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT",
                         6: "Christian", 7: "Green", 10: "United"},
        "elec_recode": {1: "Alliance", 2: "Labour", 3: "National", 4: "NZ First", 5: "ACT",
                        6: "Christian", 7: "Green", 10: "United"},
    }),
    (2002, {
        "table": "nzes_2002",
        "party_vote": "WVOT02P",
        "elec_vote": "WVOT02E",
        "most_liked": "WPTYLKM",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Other", 7: "Alliance", 8: "Christian", 9: "Progressive", 11: "United"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                        6: "Other", 7: "Alliance", 8: "Christian", 9: "Progressive", 11: "United"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Other", 7: "Alliance", 8: "Christian", 9: "Progressive", 11: "United"},
    }),
    (2005, {
        "table": "nzes_2005",
        "party_vote": "yvot05p",
        "elec_vote": "yvot05e",
        "most_liked": "yplkmst",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 8: "Progressive", 9: "Other"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                        6: "United", 7: "Maori", 8: "Progressive", 9: "Other"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 8: "Progressive", 9: "Other"},
    }),
    (2011, {
        "table": "nzes_2011",
        "party_vote": "jpartyvote",
        "elec_vote": "jelecvote",
        "most_liked": "jmostlike",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 10: "Other"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                        6: "United", 7: "Maori", 10: "Other"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 10: "Other"},
    }),
    (2014, {
        "table": "nzes_2014",
        "party_vote": "dpartyvote",
        "elec_vote": "delecvote",
        "most_liked": "dmostlike",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 10: "Other", 11: "Conservative"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                        6: "United", 7: "Maori", 10: "Other", 11: "Conservative"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "United", 7: "Maori", 10: "Other", 11: "Conservative"},
    }),
    (2017, {
        "table": "nzes_2017",
        "party_vote": "rpartyvote",
        "elec_vote": "relecvote",
        "most_liked": "rmostlike",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Maori", 8: "TOP", 10: "Other"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                        6: "Maori", 8: "TOP", 10: "Other"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Maori", 8: "TOP", 10: "Other"},
    }),
    (2020, {
        "table": "nzes_2020",
        "party_vote": "lvpartyvote",
        "elec_vote": "E5",
        "most_liked": "B2",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Maori", 7: "Other", 10: "TOP"},
        "elec_recode": None,  # Need to check format
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "NZ First", 5: "ACT",
                         6: "Maori", 7: "Other", 10: "TOP"},
    }),
    (2023, {
        "table": "nzes_2023",
        "party_vote": "mpartyvote",
        "elec_vote": "melecvote",
        "most_liked": "B2",
        "party_recode": {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First",
                         6: "Maori", 7: "Other"},
        "elec_recode": {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First",
                        6: "Maori", 7: "Other"},
        "liked_recode": {1: "Labour", 2: "National", 3: "Green", 4: "ACT", 5: "NZ First",
                         6: "Maori", 7: "Other"},
    }),
])

# Actual election results (party vote %) for each election
ELECTION_RESULTS = {
    1996: {"Labour": 28.2, "National": 33.8, "NZ First": 13.4, "Alliance": 10.1, "ACT": 6.1},
    1999: {"Labour": 38.7, "National": 30.5, "NZ First": 4.3, "Alliance": 7.7, "ACT": 7.0, "Green": 5.2},
    2002: {"Labour": 41.3, "National": 20.9, "Green": 7.0, "NZ First": 10.4, "ACT": 7.1, "United": 6.7},
    2005: {"Labour": 41.1, "National": 39.1, "Green": 5.3, "NZ First": 5.7, "ACT": 1.5, "United": 2.7, "Maori": 2.1},
    2008: {"Labour": 34.0, "National": 44.9, "Green": 6.7, "NZ First": 4.1, "ACT": 3.7, "Maori": 2.4},
    2011: {"Labour": 27.5, "National": 47.3, "Green": 11.1, "NZ First": 6.6, "ACT": 1.1},
    2014: {"Labour": 25.1, "National": 47.0, "Green": 10.7, "NZ First": 8.7, "ACT": 0.7, "Conservative": 4.0},
    2017: {"Labour": 36.9, "National": 44.4, "Green": 6.3, "NZ First": 7.2, "ACT": 0.5, "TOP": 2.4},
    2020: {"Labour": 50.0, "National": 25.6, "Green": 7.9, "NZ First": 2.6, "ACT": 7.6},
    2023: {"Labour": 26.9, "National": 38.1, "Green": 11.6, "NZ First": 6.1, "ACT": 8.6},
}

MINOR_PARTIES = ["Green", "NZ First", "ACT", "Alliance", "United", "Maori", "TOP", "Conservative"]


def main():
    conn = sqlite3.connect(DB_PATH)

    print("=" * 70)
    print("NZES STRATEGIC VOTING UNDER MMP")
    print("Cox (1997): Do voters desert minor parties near the 5% threshold?")
    print("=" * 70)

    # ── 1. Split-ticket voting rates ─────────────────────────────────
    print(f"\n{'=' * 70}")
    print("1. SPLIT-TICKET VOTING: Electorate vs Party Vote")
    print(f"{'=' * 70}")

    print(f"\n  {'Year':>6s}  {'Split rate':>11s}  {'n':>6s}  {'Major→Major':>12s}  {'Minor→Major':>12s}  {'Major→Minor':>12s}")

    split_results = []
    for year, config in YEAR_CONFIG.items():
        if config.get("elec_recode") is None:
            continue

        table = config["table"]
        pv = config["party_vote"]
        ev = config["elec_vote"]

        df = pd.read_sql(f'SELECT "{pv}", "{ev}" FROM "{table}"', conn)
        df["party"] = df[pv].map(config["party_recode"])
        df["elec"] = df[ev].map(config["elec_recode"])

        valid = df.dropna(subset=["party", "elec"])
        if len(valid) < 50:
            continue

        # Split-ticket = different party for electorate vs party vote
        valid = valid.copy()
        valid["split"] = valid["party"] != valid["elec"]
        split_rate = valid["split"].mean() * 100

        # Categorise splits
        major = {"Labour", "National"}
        valid["party_type"] = valid["party"].apply(lambda x: "Major" if x in major else "Minor")
        valid["elec_type"] = valid["elec"].apply(lambda x: "Major" if x in major else "Minor")

        # Minor party vote → major electorate vote (strategic)
        minor_pv = valid[valid["party_type"] == "Minor"]
        minor_to_major = (minor_pv["elec_type"] == "Major").mean() * 100 if len(minor_pv) > 0 else 0

        major_pv = valid[valid["party_type"] == "Major"]
        major_to_major = (major_pv["elec_type"] == "Major").mean() * 100 if len(major_pv) > 0 else 0

        major_to_minor = (major_pv["elec_type"] == "Minor").mean() * 100 if len(major_pv) > 0 else 0

        split_results.append({
            "year": year, "split_rate": split_rate, "n": len(valid),
            "minor_to_major": minor_to_major
        })
        print(f"  {year:>6d}  {split_rate:>10.1f}%  {len(valid):>6d}  {major_to_major:>11.1f}%  {minor_to_major:>11.1f}%  {major_to_minor:>11.1f}%")

    split_df = pd.DataFrame(split_results)
    print(f"\n  Mean split-ticket rate: {split_df['split_rate'].mean():.1f}%")
    print(f"  Minor→Major electorate rate: {split_df['minor_to_major'].mean():.1f}%")
    print(f"  (This is the strategic channel: minor party voters lend electorate vote to major party)")

    # ── 2. "Party liked most" vs actual party vote ───────────────────
    print(f"\n{'=' * 70}")
    print("2. STRATEGIC DESERTION: Party Liked Most vs Actual Party Vote")
    print(f"{'=' * 70}")

    print(f"\n  {'Year':>6s}  {'Party':>12s}  {'Liked %':>8s}  {'Voted %':>8s}  {'Desertion':>10s}  {'n liked':>8s}")

    desertion_data = []
    for year, config in YEAR_CONFIG.items():
        if config.get("most_liked") is None or config.get("liked_recode") is None:
            continue

        table = config["table"]
        df = pd.read_sql(f'SELECT "{config["party_vote"]}", "{config["most_liked"]}" FROM "{table}"', conn)
        df["voted"] = df[config["party_vote"]].map(config["party_recode"])
        df["liked"] = df[config["most_liked"]].map(config["liked_recode"])

        valid = df.dropna(subset=["voted", "liked"])
        if len(valid) < 50:
            continue

        total = len(valid)
        for party in MINOR_PARTIES:
            liked_party = valid[valid["liked"] == party]
            voted_party = valid[valid["voted"] == party]
            if len(liked_party) < 5:
                continue

            liked_pct = len(liked_party) / total * 100
            voted_pct = len(voted_party) / total * 100
            # Of those who liked this party, how many voted for someone else?
            deserted = liked_party[liked_party["voted"] != party]
            desertion_rate = len(deserted) / len(liked_party) * 100

            # Actual election result for context
            actual = ELECTION_RESULTS.get(year, {}).get(party, None)

            desertion_data.append({
                "year": year, "party": party, "liked_pct": liked_pct,
                "voted_pct": voted_pct, "desertion_rate": desertion_rate,
                "n_liked": len(liked_party), "actual": actual
            })

            print(f"  {year:>6d}  {party:>12s}  {liked_pct:>7.1f}%  {voted_pct:>7.1f}%  {desertion_rate:>9.1f}%  {len(liked_party):>8d}")

    if desertion_data:
        des_df = pd.DataFrame(desertion_data)

        # ── 3. Does desertion rate correlate with party size? ────────
        print(f"\n{'=' * 70}")
        print("3. DESERTION AND PARTY SIZE: Do smaller parties lose more voters?")
        print(f"{'=' * 70}")

        des_with_actual = des_df.dropna(subset=["actual"])
        if len(des_with_actual) > 5:
            from scipy.stats import pearsonr
            r, p = pearsonr(des_with_actual["actual"], des_with_actual["desertion_rate"])
            print(f"\n  r(actual vote share, desertion rate) = {r:.3f}, p = {p:.4f}")
            print(f"  → {'Smaller parties have HIGHER desertion (strategic voting confirmed)' if r < 0 and p < 0.05 else 'No significant relationship with party size'}")

            # Near-threshold test: parties around 5%
            near_thresh = des_with_actual[(des_with_actual["actual"] >= 2) & (des_with_actual["actual"] <= 8)]
            far_above = des_with_actual[des_with_actual["actual"] > 8]
            if len(near_thresh) > 3 and len(far_above) > 3:
                from scipy.stats import ttest_ind
                t, p = ttest_ind(near_thresh["desertion_rate"], far_above["desertion_rate"])
                print(f"\n  Near threshold (2-8%) mean desertion: {near_thresh['desertion_rate'].mean():.1f}%")
                print(f"  Above threshold (>8%) mean desertion: {far_above['desertion_rate'].mean():.1f}%")
                print(f"  t-test: t={t:.2f}, p={p:.4f}")

        # ── 4. Where do deserters go? ────────────────────────────────
        print(f"\n{'=' * 70}")
        print("4. WHERE DO MINOR PARTY DESERTERS GO?")
        print(f"{'=' * 70}")

        for year, config in YEAR_CONFIG.items():
            if config.get("most_liked") is None or config.get("liked_recode") is None:
                continue

            table = config["table"]
            df = pd.read_sql(f'SELECT "{config["party_vote"]}", "{config["most_liked"]}" FROM "{table}"', conn)
            df["voted"] = df[config["party_vote"]].map(config["party_recode"])
            df["liked"] = df[config["most_liked"]].map(config["liked_recode"])

            valid = df.dropna(subset=["voted", "liked"])

            # Aggregate: where do all minor party deserters go?
            minor_liked = valid[valid["liked"].isin(MINOR_PARTIES)]
            deserted = minor_liked[minor_liked["voted"] != minor_liked["liked"]]
            if len(deserted) < 10:
                continue

            dest = deserted["voted"].value_counts(normalize=True) * 100
            top_dests = dest.head(5)
            print(f"\n  {year}: Minor party deserters went to:")
            for party, pct in top_dests.items():
                print(f"    {party:>12s}: {pct:.1f}%")

    # ── 5. Split-ticket direction ────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("5. SPLIT-TICKET DIRECTION: Party vote minor, electorate major?")
    print(f"{'=' * 70}")

    print(f"\n  Under MMP, voters can split: party vote for preferred minor party,")
    print(f"  electorate vote for viable major party candidate. This is rational.")

    for year, config in YEAR_CONFIG.items():
        if config.get("elec_recode") is None:
            continue

        table = config["table"]
        df = pd.read_sql(f'SELECT "{config["party_vote"]}", "{config["elec_vote"]}" FROM "{table}"', conn)
        df["party"] = df[config["party_vote"]].map(config["party_recode"])
        df["elec"] = df[config["elec_vote"]].map(config["elec_recode"])

        valid = df.dropna(subset=["party", "elec"])
        major = {"Labour", "National"}

        # Minor party voters
        minor_voters = valid[valid["party"].isin(MINOR_PARTIES)]
        if len(minor_voters) < 10:
            continue

        elec_major = (minor_voters["elec"].isin(major)).mean() * 100
        elec_same = (minor_voters["elec"] == minor_voters["party"]).mean() * 100
        print(f"\n  {year}: Minor party voters' electorate choices:")
        print(f"    Same party: {elec_same:.1f}%  |  Major party: {elec_major:.1f}%  |  Other minor: {100-elec_same-elec_major:.1f}%")

    # ── 6. Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Split-ticket rate over time
    ax = axes[0, 0]
    ax.plot(split_df["year"], split_df["split_rate"], "o-", color="#2C3E50", linewidth=2)
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Split-ticket rate (%)")
    ax.set_title("Split-Ticket Voting Rate Under MMP")
    ax.axhline(split_df["split_rate"].mean(), ls="--", color="gray", lw=0.5)

    # Panel 2: Desertion rate by actual party size
    ax = axes[0, 1]
    if len(desertion_data) > 0:
        des_df = pd.DataFrame(desertion_data)
        des_with_actual = des_df.dropna(subset=["actual"])
        colors = {"Green": "#27AE60", "NZ First": "#2C3E50", "ACT": "#F1C40F",
                  "Alliance": "#E74C3C", "United": "#9B59B6", "Maori": "#E67E22",
                  "TOP": "#3498DB", "Conservative": "#8E44AD"}
        for _, row in des_with_actual.iterrows():
            c = colors.get(row["party"], "gray")
            ax.scatter(row["actual"], row["desertion_rate"], color=c, s=40, alpha=0.7)
            if row["desertion_rate"] > 60 or row["actual"] < 2:
                ax.annotate(f"{row['party'][:3]}{row['year']%100:02d}",
                           (row["actual"], row["desertion_rate"]), fontsize=6)
        ax.set_xlabel("Actual election result (%)")
        ax.set_ylabel("Desertion rate (%)")
        ax.set_title("Strategic Desertion vs Party Size")
        ax.axvline(5, ls="--", color="red", lw=1, label="5% threshold")
        ax.legend(fontsize=8)

    # Panel 3: Minor→Major electorate rate over time
    ax = axes[1, 0]
    ax.bar(split_df["year"].astype(str), split_df["minor_to_major"], color="#E67E22", alpha=0.7)
    ax.set_ylabel("% of minor party voters who gave electorate vote to major")
    ax.set_title("Strategic Electorate Lending")
    ax.tick_params(axis='x', rotation=45)

    # Panel 4: Desertion rate by party (averaged across elections)
    ax = axes[1, 1]
    if len(desertion_data) > 0:
        des_df = pd.DataFrame(desertion_data)
        party_avg = des_df.groupby("party")["desertion_rate"].mean().sort_values()
        party_avg = party_avg[party_avg.index.isin(["Green", "NZ First", "ACT", "Alliance", "TOP", "Conservative"])]
        bar_colors = [colors.get(p, "gray") for p in party_avg.index]
        ax.barh(party_avg.index, party_avg.values, color=bar_colors, alpha=0.7)
        ax.set_xlabel("Mean desertion rate (%)")
        ax.set_title("Average Strategic Desertion by Party")

    plt.tight_layout()
    outpath = REPORTS_DIR / "nzes_strategic_voting.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()
    conn.close()


if __name__ == "__main__":
    main()
