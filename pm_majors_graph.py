#!/usr/bin/env python3
"""
Combined preferred-PM support for the National and Labour leaders, 2006-2026.

Restricted to the four continuous polling houses. The Post/Freshwater asks a
forced two-way choice (its two numbers total ~80) and Herald-DigiPoll ran on a
method that also totalled ~80, so both are excluded — pooling them would swamp
the trend.

Input: data/pm_polling_all.csv (written by pm_scraper.py)
Output: reports/pm_majors.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pm_history_graph import LEADER_PARTY

ROOT = Path(__file__).parent

START = "2006-01-01"
# Black and white: no hue at all sidesteps the party-colour problem entirely.
INK = "#111111"

CORE_HOUSES = {
    "1 News (Colmar Brunton / Kantar / Verian)": ("colmar", "kantar", "verian"),
    "Reid Research": ("reid",),
    "Curia / Taxpayers' Union": ("curia", "taxpayer"),
    "Talbot Mills": ("talbot",),
}

# Short notes for the two spikes a general reader would otherwise puzzle over
NOTES = [
    ("2016-08-01", 72, "Ardern replaces\nLittle"),
    ("2019-09-01", 78, "Covid"),
]

# Prime ministers over the window. English and Hipkins held office under a year
# each, so they are shown but not named — labelling them is what makes the band
# unreadable, and leaving them out entirely would misdate Key's and Ardern's terms.
PMS = [
    ("Clark", "2006-01-01", "2008-11-19"),
    ("Key", "2008-11-19", "2016-12-12"),
    ("", "2016-12-12", "2017-10-26"),      # English
    ("Ardern", "2017-10-26", "2023-01-25"),
    ("", "2023-01-25", "2023-11-27"),      # Hipkins
    ("Luxon", "2023-11-27", "2026-09-01"),
]

PM_BAND = (4, 11)  # y-range the band occupies

TEXT_PRIMARY = "#0b1a2b"
TEXT_SECONDARY = "#3f5570"
TEXT_MUTED = "#6b7f96"
SURFACE = "#dbe8f7"   # soft blue panel
GRID = "#ffffff"


def house(pollster: str):
    name = str(pollster).lower()
    for label, keys in CORE_HOUSES.items():
        if any(k in name for k in keys):
            return label
    return None


def build() -> pd.Series:
    """Per poll, the National leader's share plus the Labour leader's share."""
    df = pd.read_csv(ROOT / "data" / "pm_polling_all.csv", parse_dates=["date"])
    df["party"] = df["candidate"].map(LEADER_PARTY)
    df["house"] = df["pollster"].map(house)
    df = df.dropna(subset=["house"])
    df = df[(df["date"] >= START) & df["party"].isin(["National", "Labour"])]

    rows = []
    for (date, pollster), poll in df.groupby(["date", "pollster"]):
        if poll["party"].nunique() < 2:
            continue  # can't read a two-horse race out of a one-horse table
        rows.append({"date": date,
                     "combined": poll.groupby("party")["percent"].max().sum()})

    return pd.DataFrame(rows).sort_values("date").set_index("date")["combined"]


def main():
    series = build()
    baseline = series[series.index.year < 2023].mean()
    line = series.rolling(7, center=True, min_periods=3).mean()
    latest = series.tail(6).mean()

    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (name, begin, end) in enumerate(PMS):
        x0, x1 = pd.Timestamp(begin), pd.Timestamp(end)
        ax.fill_between([x0, x1], PM_BAND[0], PM_BAND[1], color=GRID,
                        alpha=0.85 if i % 2 == 0 else 0.55, linewidth=0, zorder=2)
        if name:
            ax.text(x0 + (x1 - x0) / 2, sum(PM_BAND) / 2, name, ha="center",
                    va="center", fontsize=11, fontweight="bold",
                    color=TEXT_SECONDARY, zorder=3)
    ax.text(pd.Timestamp(START), PM_BAND[1] + 1.5, "Prime minister",
            fontsize=9.5, color=TEXT_MUTED, va="bottom", zorder=3)

    ax.axhline(baseline, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (5, 5)),
               alpha=0.6, zorder=2)
    ax.text(pd.Timestamp("2023-06-01"), baseline + 1.5,
            f"Average up to 2022: {baseline:.0f}%",
            fontsize=10.5, color=TEXT_SECONDARY, va="bottom")

    for date, y, text in NOTES:
        ax.annotate(text, xy=(pd.Timestamp(date), y), ha="center", va="top",
                    fontsize=10, color=TEXT_MUTED, linespacing=1.4, zorder=3)

    ax.plot(line.index, line.values, color=INK, linewidth=3,
            solid_capstyle="round", zorder=4)
    ax.annotate(f"{latest:.0f}%", xy=(series.index[-1], latest),
                textcoords="offset points", xytext=(14, 0),
                ha="left", va="center", fontsize=15, fontweight="bold",
                color=INK, zorder=5)

    ax.set_title("Fewer New Zealanders than ever want either big party's leader "
                 "as Prime Minister",
                 fontsize=16.5, fontweight="bold", color=TEXT_PRIMARY,
                 loc="left", pad=34)
    ax.text(0, 1.015,
            "Combined share naming the National or the Labour leader as their "
            "preferred PM, 2006–2026. Rolling average of published polls.",
            transform=ax.transAxes, fontsize=11.5, color=TEXT_SECONDARY, va="bottom")

    ax.set_ylim(0, 80)
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp("2027-09-01"))
    ax.set_yticks(range(0, 81, 20))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2008, 2027, 3)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, alpha=0.9, linewidth=1.2)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=11.5, length=0)

    fig.text(0.11, 0.02,
             "Polls by 1 News (Colmar Brunton/Kantar/Verian), Reid Research, "
             "Curia/Taxpayers' Union and Talbot Mills. The two unnamed terms are "
             "Bill English (2016–17) and Chris Hipkins (2023).",
             ha="left", fontsize=9, color=TEXT_MUTED)

    out = ROOT / "reports" / "pm_majors.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out} ({len(series)} polls, baseline {baseline:.1f}, "
          f"latest {latest:.1f})")


if __name__ == "__main__":
    main()
