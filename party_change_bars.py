#!/usr/bin/env python3
"""Listener-style bar chart: each party's change in party vote since the 2023
election and since 1 January 2026, in two side-by-side panels on one scale.

Bars use party colours in both panels; direction is shown only by which side
of zero a bar sits on.
Writes graphics/party_change_bars_listener.svg; render the PNG with the
listener-charts render.py.
"""

from html import escape
from pathlib import Path

from party_change_arrows import fmt_change, load

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "graphics/party_change_bars_listener.svg"

LABELS = {"Green": "Greens"}
# Party colours, adjusted where the brand colour vanishes on the pale-blue
# panel (ACT, TOP) or is too close to another party's (Te Pāti Māori vs Labour).
COLOURS = {
    "National": "#00529f",
    "Labour": "#d82a20",
    "Green": "#098137",
    "NZ First": "#1a1a1a",
    "ACT": "#d4b000",
    "TOP": "#17a593",
    "Te Pāti Māori": "#6e0a1e",
}

W = 1200
PANEL_X, PANEL_W = 40, 1120
LO, HI = -12, 8  # shared scale, percentage points
SUBPANELS = [(300, 410, "Since the 2023 election", "election"),
             (735, 410, "Since 1 January 2026", "january")]
ROW_H, BAR_H = 72, 36


def build(rows, as_of):
    day = f"{as_of.day} {as_of:%B %Y}"
    panel_top = 192
    first_row = panel_top + 66
    panel_bottom = first_row + len(rows) * ROW_H + 6
    height = panel_bottom + 80

    headline = ["National is down 10 points since 2023. Labour has slumped this year,",
                "but is still polling above its 2023 election result"]
    title = " ".join(headline)
    desc = (
        "Two bar charts of the change in each party's vote, in percentage points: one since "
        "the 2023 election result, one since the 1 January 2026 poll average, both measured "
        f"to the poll average at {day}. "
        + " ".join(
            f"{r['party']} ({r['now']}% now): {fmt_change(r['now'] - r['election'])} since the "
            f"election, {fmt_change(r['now'] - r['january'])} since 1 January."
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
        ".ax{font-size:17px;fill:#444}.hd{font-size:21px;font-weight:700}.hm{font-size:17px;fill:#555}"
        ".p{font-size:23px;font-weight:700}.lv{font-size:17px;fill:#555}"
        ".v{font-size:21px;font-weight:700}.g{stroke:#fff;stroke-width:1.5}</style>",
        f'<rect width="{W}" height="{height}" fill="#fff"/>',
        f'<text class="h" x="40" y="54">{escape(headline[0])}</text>',
        f'<text class="h" x="40" y="96">{escape(headline[1])}</text>',
        '<text class="s" x="40" y="132">Change in each party\'s share of the vote, percentage points. '
        'Bars to the right are gains,</text>',
        f'<text class="s" x="40" y="158">bars to the left are losses. Today\'s figures are the poll average at {day}.</text>',
        f'<rect x="{PANEL_X}" y="{panel_top}" width="{PANEL_W}" height="{panel_bottom - panel_top}" fill="#dceaf7"/>',
        f'<text class="hm" x="60" y="{panel_top + 38}">Party vote now</text>',
    ]

    for x0, width, heading, _ in SUBPANELS:
        scale = width / (HI - LO)

        def x_of(v, x0=x0, scale=scale):
            return x0 + (v - LO) * scale

        out.append(f'<text class="hd" x="{x0 + 6}" y="{panel_top + 38}">{heading}</text>')
        for v in range(-10, HI + 1, 5):
            x = x_of(v)
            if v:
                out.append(f'<line class="g" x1="{x:.1f}" x2="{x:.1f}" y1="{first_row - 6}" y2="{panel_bottom - 6}"/>')
            tick = f"{v:+d}".replace("-", "−") if v else "0"
            out.append(f'<text class="ax" x="{x:.1f}" y="{panel_bottom + 24}" text-anchor="middle">{tick}</text>')

    # White gutter between the two panels.
    gutter = (SUBPANELS[0][0] + SUBPANELS[0][1] + SUBPANELS[1][0]) / 2
    out.append(f'<line x1="{gutter:.1f}" x2="{gutter:.1f}" y1="{panel_top}" y2="{panel_bottom}" '
               f'stroke="#fff" stroke-width="4"/>')

    for i, r in enumerate(rows):
        top = first_row + i * ROW_H
        mid = top + ROW_H / 2
        if i:
            out.append(f'<line class="g" x1="{PANEL_X}" x2="{PANEL_X + PANEL_W}" y1="{top}" y2="{top}"/>')
        out.append(f'<text class="p" x="60" y="{mid - 2:.0f}">{escape(LABELS.get(r["party"], r["party"]))}</text>')
        out.append(f'<text class="lv" x="60" y="{mid + 22:.0f}">{r["now"]:.1f}%</text>')
        for x0, width, _, key in SUBPANELS:
            scale = width / (HI - LO)
            zero = x0 + -LO * scale
            d = round(r["now"] - r[key], 1)
            end = zero + d * scale
            colour = COLOURS[r["party"]]
            left = min(zero, end)
            out.append(f'<rect x="{left:.1f}" y="{mid - BAR_H / 2:.1f}" width="{abs(end - zero):.1f}" '
                       f'height="{BAR_H}" fill="{colour}"/>')
            label_x, anchor = (end + 8, "start") if d >= 0 else (end - 8, "end")
            out.append(f'<text class="v" x="{label_x:.1f}" y="{mid + 7.5:.1f}" text-anchor="{anchor}">'
                       f'{fmt_change(d)}</text>')

    # Zero lines go last so they sit on top of the bars.
    for x0, width, *_ in SUBPANELS:
        zero = x0 + -LO * width / (HI - LO)
        out.append(f'<line x1="{zero:.1f}" x2="{zero:.1f}" y1="{first_row - 6}" y2="{panel_bottom - 6}" '
                   f'stroke="#1a1a1a" stroke-width="2"/>')

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
    print(OUTPUT)


if __name__ == "__main__":
    main()
