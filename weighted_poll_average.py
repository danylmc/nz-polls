#!/usr/bin/env python3
"""
Rolling weighted average of NZ polls, 2023 election to today.
Produces two charts (major / minor parties) styled after graphs/listenergraphaesthetic.png,
plus a tidy CSV of the underlying series.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ELECTION_DATE = datetime(2023, 10, 14)
WINDOW_DAYS = 30
DEFAULT_SAMPLE_SIZE = 1000
HOUSE_EFFECT_LOOKBACK_DAYS = 730
HOUSE_EFFECT_PEER_DAYS = 30
HOUSE_EFFECT_PRIOR_POLLS = 10

MAJOR_PARTIES = ["National", "Labour"]
MINOR_PARTIES = ["Green", "ACT", "NZ First", "Te Pāti Māori", "TOP"]

PARTY_COLORS = {
    "National": "#00529F",
    "Labour": "#D82A20",
    "Green": "#098137",
    "ACT": "#FDE401",
    "NZ First": "#000000",
    "Te Pāti Māori": "#B2001A",
    "TOP": "#32DAC3",
}
LABEL_COLORS = {**PARTY_COLORS, "ACT": "#9C8300", "TOP": "#0E8C80"}  # darker text variants for contrast on light bg

BG_COLOR = "#DDEEF7"
GRID_COLOR = "#B9D3E0"


def load_polls():
    """Load all polls from election day 2023-10-14 to today."""
    polls = []
    for year in (2023, 2026):
        data = json.load(open(f"data/{year}_polling.json"))
        for p in data["polls"]:
            end_date = datetime.strptime(p["date"], "%Y-%m-%d")
            start_date = datetime.strptime(p.get("fieldwork_start") or p["date"], "%Y-%m-%d")
            fieldwork_midpoint = start_date + (end_date - start_date) / 2
            if end_date < ELECTION_DATE:
                continue
            pollster = p.get("pollster")
            if pollster == "Labour–Talbot Mills":
                pollster = "Talbot Mills"
            polls.append({
                "date": fieldwork_midpoint,
                "sample_size": p.get("sample_size") or DEFAULT_SAMPLE_SIZE,
                "parties": p["parties"],
                "pollster": pollster,
            })
    polls.sort(key=lambda p: p["date"])
    return polls


def election_result():
    """Find the 2023 result in either cycle file.

    Wikipedia includes the previous election result in the current-cycle table,
    so the chart can still be generated if an older-cycle scrape is unavailable.
    """
    for year in (2023, 2026):
        data = json.load(open(f"data/{year}_polling.json"))
        for poll in data.get("polls", []):
            if poll.get("date") == ELECTION_DATE.strftime("%Y-%m-%d"):
                return poll["parties"]
    raise ValueError("2023 election result not found in polling data")


def rolling_weighted_average(polls, parties, end_date):
    """For each day from election to end_date, compute a weighted average per
    party using a centered Gaussian kernel (sigma = WINDOW_DAYS, support out
    to 3*sigma) over nearby polls, weighted by sample size. Centered (uses
    polls on both sides of each date, not just trailing) so a single poll's
    influence is spread across the weeks around it instead of snapping the
    line the instant it's published — this is a retrospective chart of a
    completed series, so using "future" polls relative to each date is fine
    everywhere except the last few days, where it naturally falls back to a
    trailing average since no later polls exist yet."""
    sigma = WINDOW_DAYS / 2
    support = 3 * sigma
    rows = []
    day = ELECTION_DATE
    while day <= end_date:
        window_polls = [
            p for p in polls
            if abs((day - p["date"]).total_seconds() / 86400) <= support
        ]
        row = {"date": day}
        for party in parties:
            weighted_sum = 0.0
            weight_total = 0.0
            for p in window_polls:
                if party not in p["parties"]:
                    continue
                offset_days = (day - p["date"]).total_seconds() / 86400
                closeness = np.exp(-0.5 * (offset_days / sigma) ** 2)
                weight = p["sample_size"] * closeness
                weighted_sum += weight * p["parties"][party]
                weight_total += weight
            row[party] = weighted_sum / weight_total if weight_total > 0 else np.nan
        rows.append(row)
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def estimate_house_effects(polls, parties, end_date):
    """Estimate shrunk pollster-party effects against nearby competing polls.

    Each poll is compared with a Gaussian/sample-weighted average from other
    pollsters within 30 days. Pollster means are partially pooled toward zero,
    then centred per party so the adjustment does not define any one pollster
    as the truth.
    """
    history_start = end_date - timedelta(days=HOUSE_EFFECT_LOOKBACK_DAYS)
    history = [
        p for p in polls
        if history_start <= p["date"] <= end_date
        and p.get("pollster")
        and "election result" not in p["pollster"].lower()
    ]
    residuals = defaultdict(list)
    peer_sigma = HOUSE_EFFECT_PEER_DAYS / 2

    for poll in history:
        peers = [
            peer for peer in history
            if peer["pollster"] != poll["pollster"]
            and abs((peer["date"] - poll["date"]).total_seconds() / 86400)
            <= HOUSE_EFFECT_PEER_DAYS
        ]
        for party in parties:
            if party not in poll["parties"]:
                continue
            values, weights = [], []
            for peer in peers:
                if party not in peer["parties"]:
                    continue
                days = (peer["date"] - poll["date"]).total_seconds() / 86400
                weight = peer["sample_size"] * np.exp(-0.5 * (days / peer_sigma) ** 2)
                values.append(peer["parties"][party])
                weights.append(weight)
            if values:
                local_average = np.average(values, weights=weights)
                residuals[(poll["pollster"], party)].append(
                    poll["parties"][party] - local_average
                )

    effects = {}
    for key, values in residuals.items():
        n = len(values)
        if n < 5:
            continue
        shrinkage = n / (n + HOUSE_EFFECT_PRIOR_POLLS)
        effects[key] = float(np.mean(values) * shrinkage)

    # Centre each party's effects, weighted by the evidence behind each estimate.
    for party in parties:
        keys = [key for key in effects if key[1] == party]
        if not keys:
            continue
        centre = np.average(
            [effects[key] for key in keys],
            weights=[len(residuals[key]) for key in keys],
        )
        for key in keys:
            effects[key] -= centre
    return effects


def adjust_for_house_effects(polls, effects):
    adjusted = []
    for poll in polls:
        parties = {
            party: value - effects.get((poll.get("pollster"), party), 0.0)
            for party, value in poll["parties"].items()
        }
        adjusted.append({**poll, "parties": parties})
    return adjusted


SUBTITLE = "Smoothed, sample-size weighted average of all major published polls"
YEAR_CHECKPOINTS = [
    datetime(2026, 1, 1),
    datetime(2026, 5, 15),
    datetime(2026, 9, 1),
]


def style_axes(ax, title):
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color="#1a1a1a", pad=52)
    ax.annotate(
        SUBTITLE,
        xy=(0, 1), xycoords="axes fraction",
        xytext=(0, 34), textcoords="offset points",
        fontsize=9.5, color="#555555", ha="left", va="bottom",
    )
    ax.yaxis.set_label_position("left")
    ax.set_ylabel("Weighted poll average (%)", fontsize=8, color="#333333")


def plot_chart(df, parties, title, out_path, midpoint_date, result):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax, title)

    last_date = df["date"].iloc[-1]
    x_pad = timedelta(days=45)

    for party in parties:
        color = PARTY_COLORS.get(party, "#777777")
        ax.plot(df["date"], df[party], color=color, linewidth=2.5, solid_capstyle="round", label=party)

        last_val = df[party].iloc[-1]
        start_val = result.get(party)
        if pd.notna(last_val):
            change_txt = ""
            if start_val is not None:
                change = last_val - start_val
                sign = "+" if change >= 0 else ""
                change_txt = f" ({sign}{change:.1f}pp)"
            ax.annotate(
                f"{party}: {last_val:.1f}%{change_txt}",
                xy=(last_date, last_val),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=LABEL_COLORS.get(party, color),
            )

        # 2023 election result, marked at the left edge of the series
        if start_val is not None:
            ax.plot(ELECTION_DATE, start_val, marker="o", markersize=6, color=color,
                     markeredgecolor=BG_COLOR, markeredgewidth=1.2, zorder=5)
            ax.annotate(
                f"{start_val:.1f}%",
                xy=(ELECTION_DATE, start_val),
                xytext=(-8, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=8,
                fontweight="bold",
                color=LABEL_COLORS.get(party, color),
            )

    ax.axvline(ELECTION_DATE, color="#555555", linewidth=1, linestyle="--", alpha=0.8)
    ax.annotate(
        "14 Oct 2023\n(election)",
        xy=(mdates.date2num(ELECTION_DATE), 1.0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(-6, 4),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
        ha="right",
        va="bottom",
    )

    # Midpoint vertical line + a dot on each series with its value beside it
    ax.axvline(midpoint_date, color="#555555", linewidth=1, linestyle="--", alpha=0.8)
    mid_row = df.iloc[(df["date"] - midpoint_date).abs().argsort().iloc[0]]
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    ax.annotate(
        midpoint_date.strftime("%d %b %Y"),
        xy=(mdates.date2num(midpoint_date), 1.0),
        xycoords=trans,
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
        ha="left",
        va="bottom",
    )
    for party in parties:
        val = mid_row[party]
        if pd.isna(val):
            continue
        color = PARTY_COLORS.get(party, "#777777")
        ax.plot(midpoint_date, val, marker="o", markersize=6, color=color,
                 markeredgecolor=BG_COLOR, markeredgewidth=1.2, zorder=5)
        ax.annotate(
            f"{val:.1f}%",
            xy=(midpoint_date, val),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=LABEL_COLORS.get(party, color),
        )

    ax.set_xlim(ELECTION_DATE - timedelta(days=55), last_date + x_pad)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)

    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False,
                        fontsize=9, ncol=len(parties), handlelength=1.5, columnspacing=1.2)
    for text, party in zip(legend.get_texts(), parties):
        text.set_color(LABEL_COLORS.get(party, "#777777"))
        text.set_fontweight("bold")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_year_checkpoints(df, out_path, pollsters):
    """Show the two 4.5-month phases of 2026 as three weighted checkpoints."""
    checkpoints = df[df["date"].isin(YEAR_CHECKPOINTS)].set_index("date")
    if len(checkpoints) != len(YEAR_CHECKPOINTS):
        missing = [d.strftime("%Y-%m-%d") for d in YEAR_CHECKPOINTS if d not in checkpoints.index]
        raise ValueError(f"Missing checkpoint dates: {', '.join(missing)}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        "How party support shifted during 2026",
        x=0.08, y=0.975, ha="left", fontsize=18, fontweight="bold", color="#1a1a1a",
    )
    fig.text(
        0.08, 0.935,
        "House-effect-adjusted averages at the start, 4½-month split, and ninth calendar month",
        ha="left", fontsize=10.5, color="#444444",
    )

    panels = [
        (axes[0], MAJOR_PARTIES, "Major parties"),
        (axes[1], MINOR_PARTIES, "Minor parties"),
    ]
    for ax, parties, panel_title in panels:
        ax.set_facecolor(BG_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.set_ylabel("Weighted average (%)", fontsize=8.5, color="#333333")
        ax.set_title(panel_title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.axvspan(YEAR_CHECKPOINTS[0], YEAR_CHECKPOINTS[1], color="#FFFFFF", alpha=0.18)
        ax.axvspan(YEAR_CHECKPOINTS[1], YEAR_CHECKPOINTS[2], color="#8FC5DF", alpha=0.10)

        for party in parties:
            values = checkpoints.loc[YEAR_CHECKPOINTS, party].to_numpy()
            color = PARTY_COLORS[party]
            label = "Opportunity" if party == "TOP" else party
            ax.plot(
                YEAR_CHECKPOINTS, values, color=color, linewidth=2.6,
                marker="o", markersize=7, markeredgecolor=BG_COLOR,
                markeredgewidth=1.2, solid_capstyle="round", label=label,
            )

            for i, (date, value) in enumerate(zip(YEAR_CHECKPOINTS, values)):
                # Separate labels where lines meet or cross.
                major_offsets = {
                    "National": [-15, -15, 24],
                    "Labour": [9, 9, -26],
                }
                minor_offsets = {
                    "Green": [9, -15, 9],
                    "ACT": [-15, -15, -15],
                    "NZ First": [-15, 9, -15],
                    "Te Pāti Māori": [9, -15, 9],
                    "TOP": [-15, 9, -15],
                }
                offsets = major_offsets if panel_title == "Major parties" else minor_offsets
                offset = offsets[party][i]
                ax.annotate(
                    f"{value:.1f}", xy=(date, value), xytext=(0, offset),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=LABEL_COLORS[party],
                )

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False,
            fontsize=9, ncol=len(parties), handlelength=1.6, columnspacing=1.2,
        )

    axes[-1].set_xticks(YEAR_CHECKPOINTS)
    axes[-1].set_xticklabels(["1 Jan\nStart", "15 May\n4½-month split", "1 Sep\nCurrent"])
    axes[-1].set_xlim(YEAR_CHECKPOINTS[0] - timedelta(days=20),
                      YEAR_CHECKPOINTS[-1] + timedelta(days=20))
    axes[-1].set_ylim(0.5, 13.5)
    fig.text(
        0.08, 0.038,
        "Polling series combined: " + "  ·  ".join(pollsters),
        ha="left", fontsize=8.5, color="#444444", fontweight="bold",
    )
    fig.text(
        0.08, 0.018,
        "Fieldwork-midpoint, Gaussian recency and sample-size weighting. Pollster-party effects use two years of nearby polls with partial pooling.",
        ha="left", fontsize=8.5, color="#555555",
    )
    fig.tight_layout(rect=(0.05, 0.07, 0.98, 0.91), h_pad=3.3)
    fig.savefig(out_path, dpi=600, facecolor=BG_COLOR)
    vector_path = Path(out_path).with_suffix(".svg")
    fig.savefig(vector_path, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Saved {vector_path}")


def main():
    polls = load_polls()
    result = election_result()
    local_today = datetime.now(ZoneInfo("Pacific/Auckland")).date()
    today = datetime.combine(local_today, datetime.min.time())
    midpoint = ELECTION_DATE + (today - ELECTION_DATE) / 2

    all_parties = MAJOR_PARTIES + MINOR_PARTIES
    df = rolling_weighted_average(polls, all_parties, today)
    house_effects = estimate_house_effects(polls, all_parties, today)
    adjusted_polls = adjust_for_house_effects(polls, house_effects)
    adjusted_df = rolling_weighted_average(adjusted_polls, all_parties, today)

    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/weighted_poll_average.csv", index=False)
    adjusted_df.to_csv("reports/weighted_poll_average_house_adjusted.csv", index=False)
    print(f"Saved reports/weighted_poll_average.csv ({len(df)} rows)")

    plot_chart(df, MAJOR_PARTIES, "Weighted poll average: major parties (2023 election – today)",
               "reports/poll_trend_major.png", midpoint, result)
    plot_chart(df, MINOR_PARTIES, "Weighted poll average: minor parties (2023 election – today)",
               "reports/poll_trend_minor.png", midpoint, result)
    checkpoint_pollsters = sorted({
        poll["pollster"]
        for poll in polls
        if poll.get("pollster")
        and "election result" not in poll["pollster"].lower()
        and any(
            abs((poll["date"] - checkpoint).total_seconds() / 86400)
            <= 3 * (WINDOW_DAYS / 2)
            for checkpoint in YEAR_CHECKPOINTS
        )
    })
    plot_year_checkpoints(
        adjusted_df,
        "reports/poll_shift_2026_three_points.png",
        checkpoint_pollsters,
    )


if __name__ == "__main__":
    main()
