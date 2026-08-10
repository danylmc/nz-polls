#!/usr/bin/env python3
"""
Where the preferred-PM vote sits: the two major-party leaders, every other named
leader, and the undecided/none residual, 1997-2026.

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

from pm_history_graph import LEADER_PARTY, ELECTIONS

ROOT = Path(__file__).parent

# Reference palette slots 1-3 — the three that validate on the all-pairs list
# colour, legend label, vertical nudge (points) for the end label
SERIES = {
    "majors": ("#2a78d6", "National + Labour leaders", 14),
    "minors": ("#eb6834", "All other party leaders", 0),
    "residual": ("#1baf7a", "Undecided / none / not named", -16),
}

CORE_HOUSES = {
    "1 News (Colmar Brunton / Kantar / Verian)": ("colmar", "kantar", "verian"),
    "Reid Research": ("reid",),
    "Curia / Taxpayers' Union": ("curia", "taxpayer"),
    "Talbot Mills": ("talbot",),
}

BASELINE_END = 2023  # long-run major-party average is taken over 1997-2022

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"


def break_gaps(line: pd.Series, days: int = 270) -> pd.Series:
    """Don't draw a straight line across the years nobody polled."""
    gaps = line.index.to_series().diff() > pd.Timedelta(days=days)
    if not gaps.any():
        return line
    breaks = pd.Series(float("nan"), index=line.index[gaps] - pd.Timedelta(days=1))
    return pd.concat([line, breaks]).sort_index()


def house(pollster: str):
    name = str(pollster).lower()
    for label, keys in CORE_HOUSES.items():
        if any(k in name for k in keys):
            return label
    return None


def build() -> pd.DataFrame:
    """One row per poll: the three shares, which must sum to 100."""
    df = pd.read_csv(ROOT / "data" / "pm_polling_all.csv", parse_dates=["date"])
    df["party"] = df["candidate"].map(LEADER_PARTY)
    df["house"] = df["pollster"].map(house)
    df = df.dropna(subset=["house"])

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
    baseline = df.loc[df.index.year < BASELINE_END, "majors"].mean()

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for election in ELECTIONS:
        ax.axvline(pd.Timestamp(election), color=TEXT_MUTED, linewidth=0.7,
                   alpha=0.3, zorder=1)

    # No preferred-PM polling published for the 2005 cycle
    ax.axvspan(pd.Timestamp("2002-09-01"), pd.Timestamp("2005-11-01"),
               color=TEXT_MUTED, alpha=0.06, zorder=0, linewidth=0)
    ax.text(pd.Timestamp("2004-02-01"), 4, "no polling\npublished",
            ha="center", va="bottom", fontsize=8, color=TEXT_MUTED, linespacing=1.4)

    ax.axhline(baseline, color=SERIES["majors"][0], linewidth=1, linestyle=(0, (4, 4)),
               alpha=0.55, zorder=2)
    ax.text(pd.Timestamp("1997-06-01"), baseline + 1.2,
            f"1997–2022 average for the two majors: {baseline:.0f}%",
            fontsize=8.5, color=SERIES["majors"][0], va="bottom")

    for key, (colour, label, label_dy) in SERIES.items():
        ax.plot(df.index, df[key], "o", markersize=3, color=colour,
                alpha=0.16, markeredgewidth=0, zorder=3)
        line = break_gaps(df[key].rolling(7, center=True, min_periods=3).mean())
        ax.plot(line.index, line.values, color=colour, linewidth=2.2,
                solid_capstyle="round", zorder=4, label=label)
        ax.annotate(f"{label}\n{df[key].tail(6).mean():.0f}%",
                    xy=(df.index[-1], df[key].tail(6).mean()),
                    textcoords="offset points", xytext=(12, label_dy),
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    color=colour, linespacing=1.4, zorder=5)

    ax.set_title("The two major-party leaders have never held so little of the "
                 "preferred-PM vote",
                 fontsize=15, fontweight="bold", color=TEXT_PRIMARY, loc="left", pad=52)
    ax.text(0, 1.015,
            "Share of respondents naming each group as preferred PM. Dots are individual "
            "polls; lines are a 7-poll rolling average.\nThe three shares sum to 100 by "
            "construction. Vertical rules mark general elections.",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY,
            va="bottom", linespacing=1.5)

    ax.set_ylabel("% of respondents", fontsize=9.5, color=TEXT_SECONDARY)
    ax.set_ylim(0, 80)
    ax.set_xlim(pd.Timestamp("1997-01-01"), pd.Timestamp("2028-06-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5,
              labelcolor=TEXT_SECONDARY, handlelength=1.6)

    fig.text(0.5, 0.005,
             f"{len(df)} polls from 1 News (Colmar Brunton/Kantar/Verian), Reid Research, "
             "Curia/Taxpayers' Union and Talbot Mills. The Post/Freshwater (forced two-way "
             "choice) and Herald-DigiPoll are excluded — both total ~80% on two names.",
             ha="center", fontsize=8, color=TEXT_MUTED)

    out = ROOT / "reports" / "pm_composition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out} ({len(df)} polls, majors baseline {baseline:.1f})")


if __name__ == "__main__":
    main()
