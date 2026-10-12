#!/usr/bin/env python3
"""Listener-style arrow chart: each party's change since the 2023 election
and since 1 January 2026, both ending at the latest poll average.

Reads the 2023 result row from data/2023_polling.json and the house-adjusted
weighted average from reports/weighted_poll_average_house_adjusted.csv
(run weighted_poll_average.py first to refresh it). Writes
graphics/party_change_arrows_listener.svg; render the PNG with the
listener-charts render.py.
"""

import json
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data/2023_polling.json"
AVERAGE = ROOT / "reports/weighted_poll_average_house_adjusted.csv"
OUTPUT = ROOT / "graphics/party_change_arrows_listener.svg"

YEAR_START = "2026-01-01"
PARTIES = ["National", "Labour", "Green", "NZ First", "ACT", "TOP", "Te Pāti Māori"]
LABELS = {"Green": "Greens"}

UP, DOWN = "#3f7f4f", "#d6404f"
UP_TEXT, DOWN_TEXT = "#3f7f4f", "#c0303f"

W = 1200
PANEL_X, PANEL_W = 40, 1120
AXIS_X0, AXIS_X1, AXIS_MAX = 360, 920, 40  # 0% and 40% positions
ROW_H, LANE_GAP = 92, 34
TOP = 214  # first row's top edge


def x_of(value):
    return AXIS_X0 + (AXIS_X1 - AXIS_X0) * value / AXIS_MAX


def load():
    polls = json.loads(RESULTS.read_text(encoding="utf-8"))["polls"]
    result = next(p for p in polls if "election result" in p["pollster"].lower())
    avg = pd.read_csv(AVERAGE, parse_dates=["date"]).set_index("date")
    latest = avg.iloc[-1]
    rows = []
    for party in PARTIES:
        # Round first so every printed change equals the difference of printed values.
        now = round(float(latest[party]), 1)
        election = round(float(result["parties"][party]), 1)
        january = round(float(avg.loc[YEAR_START, party]), 1)
        rows.append({"party": party, "now": now, "election": election, "january": january})
    rows.sort(key=lambda r: -r["now"])
    return rows, latest.name


def fmt_change(delta):
    if abs(delta) < 0.05:
        return "0.0"
    return f"{'+' if delta > 0 else '−'}{abs(delta):.1f}"


def arrow(start, end, y):
    colour = UP if end > start else DOWN
    x0, x1 = x_of(start), x_of(end)
    sign = 1 if x1 > x0 else -1
    head = 13
    base = x1 - sign * head
    parts = [f'<circle cx="{x0:.1f}" cy="{y}" r="5.5" fill="#fff" stroke="{colour}" stroke-width="3"/>']
    if abs(x1 - x0) > head + 6:
        parts.insert(0, f'<line x1="{x0:.1f}" x2="{base:.1f}" y1="{y}" y2="{y}" '
                        f'stroke="{colour}" stroke-width="6"/>')
    # The head goes on top so even a one-point change shows its direction.
    parts.append(f'<polygon points="{x1:.1f},{y} {base:.1f},{y - 8.5} {base:.1f},{y + 8.5}" fill="{colour}"/>')
    return "\n".join(parts)


def build(rows, as_of):
    n = len(rows)
    panel_top = 170
    panel_bottom = TOP + n * ROW_H + 8
    height = panel_bottom + 78
    day = f"{as_of.day} {as_of:%B %Y}"

    nat = next(r for r in rows if r["party"] == "National")
    lab = next(r for r in rows if r["party"] == "Labour")
    title = "National is down 10 points since 2023, and Labour has slumped this year"
    desc = (
        "Arrow chart of party vote. For each party, one arrow runs from the 2023 election "
        "result to the current poll average and a second runs from the 1 January 2026 "
        f"poll average to the current average ({day}). "
        + " ".join(
            f"{r['party']}: {r['now']}% now, {fmt_change(r['now'] - r['election'])} points "
            f"since the election ({r['election']}%), {fmt_change(r['now'] - r['january'])} "
            f"since 1 January ({r['january']}%)."
            for r in rows
        )
    )

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" '
        f'height="{height}" role="img" aria-labelledby="t d">',
        f'<title id="t">{escape(title)}</title>',
        f'<desc id="d">{escape(desc)}</desc>',
        "<style>text{font-family:Lato,'Helvetica Neue',Helvetica,Arial,sans-serif;fill:#1a1a1a}"
        ".h{font-size:34px;font-weight:700}.s{font-size:19px;fill:#666}.src{font-size:15px;fill:#666}"
        ".ax{font-size:17px;fill:#444}.hd{font-size:17px;font-weight:700;fill:#444}"
        ".p{font-size:23px;font-weight:700}.lane{font-size:17px;fill:#555}"
        ".now{font-size:26px;font-weight:700}.up{font-size:21px;font-weight:700;fill:" + UP_TEXT + "}"
        ".dn{font-size:21px;font-weight:700;fill:" + DOWN_TEXT + "}.g{stroke:#fff;stroke-width:1.5}</style>",
        f'<rect width="{W}" height="{height}" fill="#fff"/>',
        f'<text class="h" x="40" y="54">{escape(title)}</text>',
        '<text class="s" x="40" y="90">Party vote, %. Each party has two arrows, both ending at today\'s poll average:</text>',
        '<text class="s" x="40" y="116">the top one starts at the 2023 election result, the bottom one at 1 January 2026.</text>',
        f'<text class="s" x="40" y="142"><tspan fill="{UP_TEXT}" font-weight="700">Green arrows</tspan> = gained, '
        f'<tspan fill="{DOWN_TEXT}" font-weight="700">red arrows</tspan> = lost. Changes are in percentage points.</text>',
        f'<rect x="{PANEL_X}" y="{panel_top}" width="{PANEL_W}" height="{panel_bottom - panel_top}" fill="#dceaf7"/>',
    ]

    # Column headings and axis
    head_y = panel_top + 26
    out.append(f'<text class="hd" x="1000" y="{head_y}" text-anchor="middle">Now</text>')
    out.append(f'<text class="hd" x="1140" y="{head_y}" text-anchor="end">Change</text>')
    for v in range(0, AXIS_MAX + 1, 10):
        x = x_of(v)
        out.append(f'<line class="g" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 4}" y2="{panel_bottom - 8}"/>')
        for ty in (head_y, panel_bottom + 24):
            out.append(f'<text class="ax" x="{x:.1f}" y="{ty}" text-anchor="middle">{v}%</text>')

    for i, r in enumerate(rows):
        top = TOP + i * ROW_H
        if i:
            out.append(f'<line class="g" x1="{PANEL_X}" x2="{PANEL_X + PANEL_W}" y1="{top}" y2="{top}"/>')
        lane1 = top + (ROW_H - LANE_GAP) / 2
        lane2 = lane1 + LANE_GAP
        mid = (lane1 + lane2) / 2
        out.append(f'<text class="p" x="60" y="{mid + 8:.0f}">{escape(LABELS.get(r["party"], r["party"]))}</text>')
        out.append(f'<text class="lane" x="{AXIS_X0 - 22}" y="{lane1 + 5.5:.0f}" text-anchor="end">Since election</text>')
        out.append(f'<text class="lane" x="{AXIS_X0 - 22}" y="{lane2 + 5.5:.0f}" text-anchor="end">Since 1 January</text>')
        out.append(arrow(r["election"], r["now"], lane1))
        out.append(arrow(r["january"], r["now"], lane2))
        out.append(f'<text class="now" x="1000" y="{mid + 9:.0f}" text-anchor="middle">{r["now"]:.1f}%</text>')
        for lane, start in ((lane1, r["election"]), (lane2, r["january"])):
            d = round(r["now"] - start, 1)
            cls = "up" if d > 0 else "dn"
            out.append(f'<text class="{cls}" x="1140" y="{lane + 7:.0f}" text-anchor="end">{fmt_change(d)}</text>')

    out.append(
        f'<text class="src" x="40" y="{height - 26}">Source: 2023 official election result; '
        f"published polls compiled on Wikipedia | Poll figures are a house-effect-adjusted, "
        f"time-weighted average as at {day}</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main():
    rows, as_of = load()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(build(rows, as_of), encoding="utf-8")
    for r in rows:
        print(f"{r['party']:15} election {r['election']:5.1f}  1 Jan {r['january']:5.1f}  now {r['now']:5.1f}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
