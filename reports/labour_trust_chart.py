#!/usr/bin/env python3
"""Listener chart: Labour's vote since the 2023 election against its lead on
Ipsos "best party to manage" ratings. Companion to graphics/nationaldecline.png
for the 21 Oct 2026 column on why Labour is falling while trusted on the issues.

Panel a: house-effect-adjusted, sample-weighted poll averages (weighted_poll_average.py).
Panel b: Ipsos NZ Issues Monitor capability, Labour vs National, three top issues.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import weighted_poll_average as w  # noqa: E402

TODAY = datetime(2026, 10, 21)
OUT = Path("graphics/labour_trust_listener.png")

PANEL, INK, MUTED, LEADER = "#dceaf7", "#1a1a1a", "#666666", "#8a8a8a"
LAB, NAT = w.PARTY_COLORS["Labour"], w.PARTY_COLORS["National"]
ISSUES = [("healthcare_hospitals", "Healthcare"),
          ("inflation_cost_of_living", "Cost of living"),
          ("the_economy", "The economy")]

for f in (Path.home() / ".local/share/fonts/lato").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Lato", "axes.facecolor": PANEL, "figure.facecolor": "white",
                     "axes.grid": True, "grid.color": "white", "grid.linewidth": 1.4,
                     "axes.axisbelow": True, "xtick.color": INK, "ytick.color": INK})


def style(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=11)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")


def latest_published():
    """Publication date of the newest poll (load_polls dates polls by fieldwork midpoint)."""
    polls = json.load(open("data/2026_polling.json"))["polls"]
    return datetime.strptime(max(p["date"] for p in polls), "%Y-%m-%d")


def averages():
    polls = w.load_polls()
    parties = w.MAJOR_PARTIES + w.MINOR_PARTIES
    adj = w.adjust_for_house_effects(polls, w.estimate_house_effects(polls, parties, TODAY))
    last_poll = max(p["date"] for p in polls)
    df = w.rolling_weighted_average(adj, parties, last_poll)
    return df.set_index("date"), last_poll


def main():
    avg, last_poll = averages()
    cap = pd.read_csv("data/ipsos_party_capability.csv")
    cap = cap[cap["date"] >= "2023-09"].copy()
    cap["when"] = pd.to_datetime(cap["date"] + "-15")

    fig = plt.figure(figsize=(10, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.42, wspace=0.12,
                          left=0.085, right=0.86, top=0.835, bottom=0.115)
    a = fig.add_subplot(gs[0, :])

    # Panel a: party vote
    style(a)
    lab, nat = avg["Labour"], avg["National"]
    a.plot(nat.index, nat, color=NAT, lw=2.2, alpha=0.35, solid_capstyle="round", zorder=2)
    a.plot(lab.index, lab, color=LAB, lw=3, solid_capstyle="round", zorder=3)
    a.plot(w.ELECTION_DATE, 26.9, "o", color=LAB, ms=7, zorder=4)
    a.annotate("Election\n26.9%", (w.ELECTION_DATE, 26.9), xytext=(10, -44), textcoords="offset points",
               ha="left", fontsize=11.5, fontweight="bold", color=LAB, linespacing=1.0)

    peak = lab.idxmax()
    a.plot(peak, lab[peak], "o", color="white", mec=LAB, mew=2.5, ms=9, zorder=5)
    a.annotate(f"Peak {lab[peak]:.1f}%, {peak:%-d %b %Y}:\nnine days before National's confidence vote",
               (peak, lab[peak]), xytext=(peak - timedelta(days=330), 38.4),
               fontsize=11, fontweight="bold", color=LAB, va="center",
               arrowprops=dict(arrowstyle="-", color=LEADER, lw=0.9, shrinkA=0, shrinkB=6))

    end = lab.index[-1]
    a.plot(end, lab.iloc[-1], "o", color=LAB, ms=7, zorder=5)
    a.plot(end, nat.iloc[-1], "o", color=NAT, alpha=0.45, ms=6, zorder=4)
    a.annotate(f"Labour {lab.iloc[-1]:.1f}%", (end, lab.iloc[-1]), xytext=(9, -7),
               textcoords="offset points", va="center", fontsize=12, fontweight="bold", color=LAB)
    a.annotate(f"National {nat.iloc[-1]:.1f}%", (end, nat.iloc[-1]), xytext=(9, 7),
               textcoords="offset points", va="center", fontsize=11, color=MUTED)

    a.set_ylim(20, 41)
    a.yaxis.set_major_locator(plt.MultipleLocator(5))
    a.set_xlim(w.ELECTION_DATE - timedelta(days=25), end + timedelta(days=20))
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a.xaxis.set_major_formatter(lambda v, _: mdates.num2date(v).strftime("%b\n%Y"))
    a.set_ylabel("Party vote", fontsize=12, fontweight="bold", color=INK)
    a.text(0, 1.04, "a  Labour party vote, weighted poll average (National for comparison)",
           transform=a.transAxes, fontsize=14.5, fontweight="bold", color=INK)

    # Panel b: best party to manage, small multiples
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    for i, (ax, (issue, label)) in enumerate(zip(axes, ISSUES)):
        style(ax)
        d = cap[cap["issue"] == issue].sort_values("when")
        ax.plot(d["when"], d["national"], color=NAT, lw=2.6, marker="o", ms=4.5, zorder=3)
        ax.plot(d["when"], d["labour"], color=LAB, lw=2.6, marker="o", ms=4.5, zorder=4)
        last = d.iloc[-1]
        ax.annotate(f"{last['labour']:.0f}%", (last["when"], last["labour"]), xytext=(6, 4 if issue == "the_economy" else 0),
                    textcoords="offset points", va="center", fontsize=11.5, fontweight="bold", color=LAB)
        ax.annotate(f"{last['national']:.0f}%", (last["when"], last["national"]), xytext=(6, -6 if issue == "the_economy" else 0),
                    textcoords="offset points", va="center", fontsize=11.5, fontweight="bold", color=NAT)
        ax.set_ylim(15, 56)
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax.set_xlim(datetime(2023, 8, 1), datetime(2026, 12, 15))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.text(0.04, 0.91, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Best party to manage", fontsize=12, fontweight="bold", color=INK)
    axes[0].text(0.04, 0.10, "Labour", transform=axes[0].transAxes, fontsize=11.5, fontweight="bold", color=LAB)
    axes[0].text(0.30, 0.10, "National", transform=axes[0].transAxes, fontsize=11.5, fontweight="bold", color=NAT)
    axes[0].text(0, 1.08, "b  Which party is best to manage the issue? (Ipsos survey waves)",
                 transform=axes[0].transAxes, fontsize=14.5, fontweight="bold", color=INK)

    x0 = 0.085
    fig.text(x0, 0.955, "Labour is still trusted on the issues. Its vote is falling anyway",
             fontsize=21, fontweight="bold", color=INK)
    fig.text(x0, 0.905, f"Labour rose to {lab[peak]:.0f}% by April 2026, then slid back almost to its election result. "
             "Through the slide,\nvoters went on rating it well ahead of National on healthcare and the cost of living",
             fontsize=12.5, color=MUTED, va="center", linespacing=1.35)
    fig.text(x0, 0.03, "Source: Wikipedia poll tables (all public polls), Ipsos NZ Issues Monitor (share naming each party best to manage the issue)\n"
             "Lines in a: house-effect-adjusted, sample-weighted averages (30-day Gaussian). "
             f"Latest poll {latest_published():%-d %b %Y}", fontsize=9.5, color=MUTED, linespacing=1.4)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=480)
    fig.savefig(OUT.with_suffix(".svg"))
    print("Saved", OUT, "| Labour peak", peak.date(), round(lab[peak], 1),
          "| end", end.date(), "Lab", round(lab.iloc[-1], 1), "Nat", round(nat.iloc[-1], 1))


if __name__ == "__main__":
    main()
