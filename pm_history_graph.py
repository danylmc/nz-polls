#!/usr/bin/env python3
"""
Preferred Prime Minister ratings, 1997-2026.

Two panels: major-party leaders (the actual PM race) and minor-party leaders.
Colour carries the party; a direct label carries the leader.
Input: data/pm_polling_all.csv (written by pm_scraper.py)
Output: reports/pm_preference_history.png
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).parent

# Validated categorical palette (dataviz reference instance), assigned to parties
PARTY_COLOURS = {
    "National": "#2a78d6",
    "Labour": "#e34948",
    "NZ First": "#4a3aa7",
    "Green": "#008300",
    "ACT": "#eda100",
    "Alliance": "#e87ba4",
}

LEADER_PARTY = {
    "Bolger": "National", "Shipley": "National", "English": "National",
    "Brash": "National", "Key": "National", "Bridges": "National",
    "Muller": "National", "Collins": "National", "Luxon": "National",
    "Moore": "Labour", "Clark": "Labour", "Goff": "Labour",
    "Shearer": "Labour", "Cunliffe": "Labour", "Little": "Labour",
    "Ardern": "Labour", "Hipkins": "Labour",
    "Peters": "NZ First",
    "Prebble": "ACT", "Seymour": "ACT",
    "Norman": "Green", "Turei": "Green", "Shaw": "Green",
    "Davidson": "Green", "Swarbrick": "Green",
    "Anderton": "Alliance",
}

MAJOR_PARTIES = {"National", "Labour"}

ELECTIONS = [
    "1999-11-27", "2002-07-27", "2005-09-17", "2008-11-08", "2011-11-26",
    "2014-09-20", "2017-09-23", "2020-10-17", "2023-10-14",
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"

# Label offsets (dx in days, dy in points) for leaders whose peak labels collide
LABEL_NUDGE = {
    "Bolger": (-500, 6), "Moore": (0, -14), "Shipley": (-200, 8),
    "English": (-700, 6), "Brash": (-400, 8), "Goff": (200, 6),
    "Shearer": (-300, 8), "Cunliffe": (250, 6), "Little": (-450, 8),
    "Bridges": (-350, 8), "Muller": (150, 6), "Collins": (300, 6),
    "Turei": (-300, 6), "Norman": (-350, 6), "Prebble": (-250, 8),
}


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "pm_polling_all.csv", parse_dates=["date"])
    df["party"] = df["candidate"].map(LEADER_PARTY)
    return df.dropna(subset=["party"])


def active_window(sub: pd.DataFrame) -> pd.DataFrame:
    """Trim the flat near-zero tails a leader picks up from being listed in a
    cycle's table before or after they actually led the party."""
    threshold = max(1.0, 0.2 * sub["percent"].max())
    live = sub[sub["percent"] >= threshold]
    if live.empty:
        return sub
    return sub[(sub["date"] >= live["date"].min()) & (sub["date"] <= live["date"].max())]


def smooth(series: pd.Series) -> pd.Series:
    """Rolling mean across a leader's own polls — enough to read the trend
    without hiding the honeymoons, which are the interesting part.
    Gaps longer than nine months are broken rather than interpolated across."""
    line = series.rolling(5, center=True, min_periods=1).mean()
    gaps = line.index.to_series().diff() > pd.Timedelta(days=270)
    if gaps.any():
        breaks = pd.Series(float("nan"),
                           index=line.index[gaps] - pd.Timedelta(days=1))
        line = pd.concat([line, breaks]).sort_index()
    return line


def draw_panel(ax, df, leaders, title, subtitle, floor=0.0):
    """`floor` drops readings too low to be a real contest — a sitting party
    leader never polls under ~2% for preferred PM, so anything below it is a
    backbencher or ex-leader still carried in the pollster's name list."""
    ax.set_facecolor(SURFACE)

    for leader in leaders:
        sub = active_window(df[df["candidate"] == leader].sort_values("date"))
        sub = sub[sub["percent"] >= floor]
        if sub.empty:
            continue
        colour = PARTY_COLOURS[LEADER_PARTY[leader]]

        ax.plot(sub["date"], sub["percent"], "o", markersize=2.5,
                color=colour, alpha=0.18, markeredgewidth=0, zorder=2)

        line = smooth(sub.set_index("date")["percent"])
        ax.plot(line.index, line.values, color=colour, linewidth=2,
                solid_capstyle="round", zorder=3)

        peak_idx = line.idxmax()
        dx, dy = LABEL_NUDGE.get(leader, (0, 7))
        ax.annotate(
            leader,
            xy=(peak_idx + pd.Timedelta(days=dx), line.max()),
            textcoords="offset points", xytext=(0, dy),
            ha="center", va="bottom", fontsize=8.5, fontweight="semibold",
            color=TEXT_SECONDARY, zorder=5,
        )

    for election in ELECTIONS:
        ax.axvline(pd.Timestamp(election), color=TEXT_MUTED, linewidth=0.7,
                   alpha=0.35, zorder=1)

    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_PRIMARY,
                 loc="left", pad=26)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
            color=TEXT_SECONDARY, va="bottom")

    ax.set_ylabel("% naming them preferred PM", fontsize=9, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_xlim(pd.Timestamp("1997-01-01"), pd.Timestamp("2027-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main():
    df = load()
    peaks = df.groupby("candidate")["percent"].max()

    majors = [c for c in peaks.index
              if LEADER_PARTY[c] in MAJOR_PARTIES and peaks[c] >= 5]
    minors = [c for c in peaks.index
              if LEADER_PARTY[c] not in MAJOR_PARTIES and peaks[c] >= 3]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 11), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.22},
    )
    fig.patch.set_facecolor(SURFACE)

    draw_panel(ax1, df, majors, "Who New Zealanders want as Prime Minister",
               "Major-party leaders. Dots are individual polls; the line is a "
               "5-poll rolling average. Vertical rules mark general elections.",
               floor=1.5)
    ax1.set_ylim(0, 75)

    # Wikipedia carries no preferred-PM table for the 2005 cycle
    for ax in (ax1, ax2):
        ax.axvspan(pd.Timestamp("2002-09-01"), pd.Timestamp("2005-11-01"),
                   color=TEXT_MUTED, alpha=0.06, zorder=0, linewidth=0)
    ax1.text(pd.Timestamp("2004-02-01"), 68, "no preferred-PM\npolling published\nfor the 2005 cycle",
             ha="center", va="top", fontsize=8.5, color=TEXT_MUTED, linespacing=1.4)

    draw_panel(ax2, df, minors, "Minor-party leaders",
               "Same scale of question, a tenth of the numbers — note the y-axis.",
               floor=0.4)
    ax2.set_ylim(0, 20)

    parties = ["National", "Labour", "NZ First", "Green", "ACT", "Alliance"]
    handles = [Line2D([0], [0], color=PARTY_COLOURS[p], linewidth=2.5, label=p)
               for p in parties]
    ax1.legend(handles=handles, loc="upper left", frameon=False, ncol=6,
               fontsize=9, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0, 1.20), handlelength=1.6, columnspacing=1.6)

    fig.text(0.5, 0.045,
             f"{len(df)} leader-readings from {df.date.min():%b %Y} to "
             f"{df.date.max():%b %Y} · Source: Wikipedia preferred-PM polling tables",
             ha="center", fontsize=8.5, color=TEXT_MUTED)

    out = ROOT / "reports" / "pm_preference_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
