#!/usr/bin/env python3
"""
Where the preferred-PM vote sits, 2006-2026: the two major-party leaders, every
other named leader, and the undecided/none residual.

Restricted to the four continuous polling houses. The Post/Freshwater asks a
forced two-way choice (its two numbers total ~80) and Herald-DigiPoll ran on a
method that also totalled ~80, so both are excluded — pooling them would swamp
the trend.

Input: data/pm_polling_all.csv (written by pm_scraper.py)
Output: reports/pm_composition.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pm_history_graph import LEADER_PARTY

ROOT = Path(__file__).parent

START = "2006-01-01"

# Reference palette slots 1-3 — the three that validate on the all-pairs list.
# colour, label, vertical nudge (points) so the end labels don't collide
SERIES = {
    "majors": ("#2a78d6", "National + Labour\nleaders", 14),
    "minors": ("#eb6834", "All other\nparty leaders", 0),
    "residual": ("#1baf7a", "Undecided\nor none", -10),
}

CORE_HOUSES = {
    "1 News (Colmar Brunton / Kantar / Verian)": ("colmar", "kantar", "verian"),
    "Reid Research": ("reid",),
    "Curia / Taxpayers' Union": ("curia", "taxpayer"),
    "Talbot Mills": ("talbot",),
}

# Short notes for the two spikes a general reader would otherwise puzzle over
NOTES = [
    ("2016-06-01", 70, "Ardern replaces\nLittle"),
    ("2019-10-01", 77, "Covid"),
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"


def house(pollster: str):
    name = str(pollster).lower()
    for label, keys in CORE_HOUSES.items():
        if any(k in name for k in keys):
            return label
    return None


def build() -> pd.DataFrame:
    """One row per poll: the three shares, which sum to 100."""
    df = pd.read_csv(ROOT / "data" / "pm_polling_all.csv", parse_dates=["date"])
    df["party"] = df["candidate"].map(LEADER_PARTY)
    df["house"] = df["pollster"].map(house)
    df = df.dropna(subset=["house"])
    df = df[df["date"] >= START]

    rows = []
    for (date, pollster), poll in df.groupby(["date", "pollster"]):
        majors = poll[poll["party"].isin(["National", "Labour"])]
        if majors["party"].nunique() < 2:
            continue  # can't read a two-horse race out of a one-horse table
        major_share = majors.groupby("party")["percent"].max().sum()
        named = poll["percent"].sum()
        rows.append({
            "date": date,
            "majors": major_share,
            "minors": named - major_share,
            "residual": 100 - named,
        })

    out = pd.DataFrame(rows).sort_values("date").set_index("date")
    return out[out["residual"] > -5]


def main():
    df = build()
    baseline = df.loc[df.index.year < 2023, "majors"].mean()

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(baseline, color=SERIES["majors"][0], linewidth=1,
               linestyle=(0, (5, 5)), alpha=0.5, zorder=2)
    ax.text(pd.Timestamp("2023-03-01"), baseline + 1.5,
            f"Their average up to 2022: {baseline:.0f}%",
            fontsize=10, color=SERIES["majors"][0], va="bottom")

    for date, y, text in NOTES:
        ax.annotate(text, xy=(pd.Timestamp(date), y), ha="center", va="top",
                    fontsize=9.5, color=TEXT_MUTED, linespacing=1.4, zorder=3)

    for key, (colour, label, label_dy) in SERIES.items():
        line = df[key].rolling(7, center=True, min_periods=3).mean()
        ax.plot(line.index, line.values, color=colour, linewidth=2.8,
                solid_capstyle="round", zorder=4)
        ax.annotate(f"{label}\n{df[key].tail(6).mean():.0f}%",
                    xy=(df.index[-1], df[key].tail(6).mean()),
                    textcoords="offset points", xytext=(14, label_dy),
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    color=colour, linespacing=1.4, zorder=5)

    ax.set_title("New Zealanders are running out of enthusiasm for both big parties' leaders",
                 fontsize=16, fontweight="bold", color=TEXT_PRIMARY, loc="left", pad=34)
    ax.text(0, 1.015,
            "Who voters name as their preferred Prime Minister, 2006–2026. "
            "Each line is a rolling average of published polls.",
            transform=ax.transAxes, fontsize=11, color=TEXT_SECONDARY, va="bottom")

    ax.set_ylim(0, 80)
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp("2029-06-01"))
    ax.set_yticks(range(0, 81, 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2008, 2027, 3)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=11, length=0)

    fig.text(0.11, 0.02,
             "Polls by 1 News (Colmar Brunton/Kantar/Verian), Reid Research, "
             "Curia/Taxpayers' Union and Talbot Mills.",
             ha="left", fontsize=9, color=TEXT_MUTED)

    out = ROOT / "reports" / "pm_composition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out} ({len(df)} polls from {df.index.min():%b %Y}, "
          f"baseline {baseline:.1f})")


if __name__ == "__main__":
    main()
