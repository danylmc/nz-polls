#!/usr/bin/env python3
"""Plot a dated snapshot of support for the traditional governing party pair."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/purple_vote_snapshot_2026-10-06.csv"
OUTPUT = ROOT / "reports/purple_vote_snapshot_2026-10-06.png"


def main():
    polls = pd.read_csv(INPUT)
    required = {"country", "fieldwork_end", "red_party", "red_pct", "blue_party", "blue_pct", "purple_pct", "source_url"}
    if not required.issubset(polls.columns):
        raise ValueError(f"Missing columns: {required - set(polls.columns)}")
    if len(polls) != polls.country.nunique():
        raise ValueError("Each country must have one snapshot")
    if ((polls.red_pct + polls.blue_pct - polls.purple_pct).abs() > 0.001).any():
        raise ValueError("Purple vote must equal the two primary vote shares")
    if ((polls[["red_pct", "blue_pct", "purple_pct"]] < 0) | (polls[["red_pct", "blue_pct", "purple_pct"]] > 100)).any().any():
        raise ValueError("Vote shares must be within 0–100")

    # The NZ series is maintained by this project. Check the snapshot against it.
    local = pd.read_csv(ROOT / "reports/weighted_poll_average.csv")
    nz = polls.loc[polls.country == "New Zealand"]
    if len(nz) != 1:
        raise ValueError("Expected one New Zealand row")
    local_day = local.loc[local.date == nz.iloc[0].fieldwork_end]
    if len(local_day) != 1:
        raise ValueError("No matching date in the local NZ polling average")
    if abs(nz.iloc[0].red_pct - local_day.iloc[0].Labour) > 0.001 or abs(nz.iloc[0].blue_pct - local_day.iloc[0].National) > 0.001:
        raise ValueError("NZ snapshot is out of sync with the local polling average")

    polls = polls.sort_values("purple_pct")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(polls))
    ax.barh(y, polls.red_pct, color="#d1495b", label="Traditional red party")
    ax.barh(y, polls.blue_pct, left=polls.red_pct, color="#3777b6", label="Traditional blue party")
    ax.set_yticks(list(y), polls.country)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of voting intention / primary vote (%)")
    ax.set_title("'Purple vote': combined support for traditional governing parties")
    for i, row in enumerate(polls.itertuples()):
        ax.text(row.purple_pct + 1, i, f"{row.purple_pct:g}%", va="center", fontweight="bold")
        ax.text(99, i, row.fieldwork_end, va="center", ha="right", fontsize=9, color="#555555")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "NZ: local poll average. Peers: one poll each. Australia uses first preferences, not two-party preferred.", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
