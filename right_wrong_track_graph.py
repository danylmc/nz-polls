#!/usr/bin/env python3
"""
Government approval: "Right direction / Wrong direction" net rating.

Data source: Wikipedia "Government approval rating" tables, which exist ONLY
on the 2020 and 2023 NZ general election pages (scraped by
approval_scraper.py). Together these cover Oct 2017 - Oct 2023 with a small
gap around the 2020 election. No structured right/wrong-track series exists
on Wikipedia before 2017 or after 2023, so this graph covers 2017-2023 only
-- not back to 1996 as originally requested.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Diverging pair: positive (net right-direction) vs negative (net wrong-direction)
COLOR_POSITIVE = "#2E7D6B"   # teal-green
COLOR_NEGATIVE = "#B3492B"   # burnt orange-red
COLOR_LINE = "#3D4A5C"       # muted ink for the net line
COLOR_GRID = "#D8DCE1"

GOVERNMENTS = [
    ("2017-10-26", "2020-10-17", "Labour-NZF-Green", "#FDECEC"),
    ("2020-10-17", "2023-10-14", "Labour (majority)", "#FDF6E3"),
    ("2023-10-14", "2026-07-27", "National-ACT-NZF", "#E8F0FB"),
]


def load_data():
    path = Path(__file__).parent / "data" / "govt_approval_right_wrong_track.json"
    with open(path) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    df = load_data()

    fig, ax = plt.subplots(figsize=(14, 7))

    for start, end, label, color in GOVERNMENTS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=color, zorder=0)

    ax.axhline(0, color="#888888", linewidth=1, zorder=1)

    net = df["net"].values
    dates = df["date"].values

    ax.fill_between(dates, net, 0, where=(net >= 0), color=COLOR_POSITIVE, alpha=0.35, interpolate=True, zorder=2)
    ax.fill_between(dates, net, 0, where=(net < 0), color=COLOR_NEGATIVE, alpha=0.35, interpolate=True, zorder=2)
    ax.plot(dates, net, color=COLOR_LINE, linewidth=2, marker="o", markersize=4, zorder=3)

    # Direct labels for the extremes
    max_i = df["net"].idxmax()
    min_i = df["net"].idxmin()
    ax.annotate(
        f"+{df.loc[max_i,'net']:.0f} ({df.loc[max_i,'date'].strftime('%b %Y')})\nCovid rally",
        xy=(df.loc[max_i, "date"], df.loc[max_i, "net"]),
        xytext=(10, 10), textcoords="offset points", fontsize=9, color=COLOR_POSITIVE, fontweight="bold",
    )
    ax.annotate(
        f"{df.loc[min_i,'net']:.0f} ({df.loc[min_i,'date'].strftime('%b %Y')})",
        xy=(df.loc[min_i, "date"], df.loc[min_i, "net"]),
        xytext=(10, -15), textcoords="offset points", fontsize=9, color=COLOR_NEGATIVE, fontweight="bold",
    )

    # Government-era labels along the top
    ymax = df["net"].max() * 1.12
    for start, end, label, _ in GOVERNMENTS:
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        if pd.Timestamp(start) > df["date"].max() or pd.Timestamp(end) < df["date"].min():
            continue
        ax.text(mid, ymax, label, ha="center", va="bottom", fontsize=9, color="#555555")

    ax.set_xlim(df["date"].min() - pd.Timedelta(days=20), df["date"].max() + pd.Timedelta(days=20))
    ax.set_ylim(df["net"].min() * 1.15, ymax * 1.08)

    ax.set_title("NZ Government Approval: Right Direction minus Wrong Direction (Net)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Net rating (percentage points)", fontsize=11)
    ax.set_xlabel(
        "Source: Wikipedia 'Government approval rating' tables (Roy Morgan, 1News-Colmar Brunton, Talbot Mills, "
        "Guardian Essential, The Post-Freshwater Strategy, Horizon, Taxpayers' Union-Curia).\n"
        "Coverage limited to Oct 2017 - Oct 2023: this table does not exist on Wikipedia's 1993-2017 or 2026 election pages.",
        fontsize=8.5, color="#666666",
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    output_path = Path(__file__).parent / "reports" / "govt_approval_right_wrong_track.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved graph to {output_path}")


if __name__ == "__main__":
    main()
