#!/usr/bin/env python3
"""Time-weighted poll average for the current calendar year.

Reuses the Gaussian recency / sample-size weighting and house-effect
adjustment from weighted_poll_average.py, then draws only this year's days,
with each individual poll as a faint dot behind the line.
"""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import weighted_poll_average as w


def panel(ax, df, polls, parties, title, year_start, today, show_x):
    ax.set_facecolor(w.BG_COLOR)
    ax.grid(True, color=w.GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9, labelbottom=show_x)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=w.LABEL_INK)
    ax.set_ylabel("Weighted average (%)", fontsize=8, color="#333333")

    first = df[df["date"] >= year_start]
    for party in parties:
        color = w.PARTY_COLORS[party]
        pts = [(p["date"], p["parties"][party]) for p in polls
               if party in p["parties"] and year_start <= p["date"] <= today
               and "election result" not in (p.get("pollster") or "").lower()]
        ax.scatter(*zip(*pts), s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
        ax.plot(first["date"], first[party], color=color, linewidth=2.6,
                solid_capstyle="round", zorder=3)

    ax.set_xlim(year_start - timedelta(days=5), today + timedelta(days=62))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    return first


def end_labels(ax, fig, first, parties, today, start_vals):
    lo, hi = ax.get_ylim()
    pts = ax.get_position().height * fig.get_figheight() * 72
    gap = 13 * (hi - lo) / pts
    finals = {p: first[p].iloc[-1] for p in parties}
    for party, y in w.stagger_labels(finals, gap).items():
        ch = finals[party] - start_vals[party]
        ax.annotate(f"{party} {finals[party]:.1f}% ({ch:+.1f}pp)",
                    xy=(today, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold", color=w.LABEL_INK)
        ax.plot(today, finals[party], "o", color=w.PARTY_COLORS[party],
                markeredgecolor=w.BG_COLOR, markersize=6, zorder=5)


def main():
    today = datetime.combine(datetime.now(ZoneInfo("Pacific/Auckland")).date(), datetime.min.time())
    year_start = datetime(today.year, 1, 1)
    polls = w.load_polls()
    parties = w.MAJOR_PARTIES + w.MINOR_PARTIES
    adjusted = w.adjust_for_house_effects(polls, w.estimate_house_effects(polls, parties, today))
    df = w.rolling_weighted_average(adjusted, parties, today)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.25]})
    fig.set_facecolor(w.BG_COLOR)
    f1 = panel(a1, df, adjusted, w.MAJOR_PARTIES, "Major parties", year_start, today, False)
    f2 = panel(a2, df, adjusted, w.MINOR_PARTIES, "Minor parties", year_start, today, True)
    fig.suptitle(f"Time-weighted poll average, {today.year}", x=0.07, ha="left",
                 fontsize=17, fontweight="bold", color=w.LABEL_INK, y=0.985)
    fig.text(0.07, 0.945, "Line: house-effect-adjusted average, weighted by sample size and recency "
             "(30-day Gaussian). Dots: individual polls. Change since 1 Jan in brackets.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    for ax, f, ps in ((a1, f1, w.MAJOR_PARTIES), (a2, f2, w.MINOR_PARTIES)):
        end_labels(ax, fig, f, ps, today, {p: f[p].iloc[0] for p in ps})
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = Path("reports") / f"weighted_poll_average_{today.year}.png"
    fig.savefig(out, dpi=200, facecolor=w.BG_COLOR)
    print("Saved", out)


if __name__ == "__main__":
    main()
