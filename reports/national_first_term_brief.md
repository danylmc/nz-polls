# National and Luxon against other first terms — data brief

Companion to `coalition_feature_brief.md`. Last updated 2026-10-19 (chart, autumn 2024, winter–spring 2026 and model sections added).
Script: `reports/first_term_comparison.py`. Sources: `data/*_polling.json`, `data/pm_polling_all.csv`,
`data/govt_approval_right_wrong_track.json` (2017–23), `data/right_wrong_direction_2026.csv`
(new: 63 polls from the cached 2026 Wikipedia page plus the June 2026 Freshwater poll the table omits).
Chart: `reports/national_luxon_mood_listener.{svg,png}` (`national_luxon_mood_chart.py`).
Model: `reports/mood_leader_party_model.py` → `mood_leader_party_monthly.csv`.

---

## The argument in one paragraph

Every first-term government since 1990 gained in the polls except two: Bolger's in 1990–93
and Luxon's. Clark, Key and Ardern each sat 7–14 points above their election result by the
30-month mark. National under Luxon sat 8 points below. Luxon is also the only new PM since
1999 who got no bump in preferred PM after winning. He began lower than any of them and never
built the lead over his rival that the others had. On right/wrong direction his government never had the
honeymoon the previous government got. The national mood is now as sour as it was in the
months before Labour was thrown out. Within the term there are five distinct falls. Only two hit
the party, the leader and the national mood together: autumn 2024 and the summer of 2024–25.

---

## 1. First-term governments: party vote

Governing party's change from its election result (LOWESS trend, all pollsters, unadjusted
so that every cycle is treated the same way)

| Government | Election | 12 months | 24 months | 30 months | Next election |
|---|---|---|---|---|---|
| Bolger (Nat) 1990–93 | 47.8 | **−21.4** | −16.9 | −18.0 | −12.8 (won, 1-seat majority) |
| Clark (Lab) 1999–2002 | 38.7 | +7.3 | +7.5 | +14.4 | +2.5 (won) |
| Key (Nat) 2008–11 | 44.9 | +9.1 | +6.7 | +7.7 | +2.4 (won) |
| Ardern (Lab) 2017–20 | 36.9 | +6.7 | +4.5 | +13.0 | +13.1 (won, Covid) |
| **Luxon (Nat) 2023–26** | **38.1** | **−2.6** | **−6.5** | **−8.3** | — |

- Luxon now: ~28.6 raw trend; 26.7 after removing pollster house effects. That is −9.5 to −11.4.
- Across **all 12 terms since 1990**, only two governing parties were further down at 30
  months: Bolger 1990–93 (−18.0) and Labour 2020–23 (−15.7, then lost 23 points). Next is
  Labour 2005–08 at −6.0, the term that ended Clark.
- **Bolger is the precedent.** He fell much harder and much faster (to ~26% within a year:
  the Mother of All Budgets and the superannuation surtax), recovered to ~37 by polling day,
  and held on with a one-seat majority. FPTP, one pollster, a different world.
  But it is the only case of a first-term government this far behind that survived.
- Shape matters. Bolger's collapse came in year one and then he recovered. Luxon's has been slow and steady
  and is at its lowest at the end of the term. No first-term government has gone into an election at its low point.

## 2. First-term PMs: preferred PM (1 News series only)

The 1 News poll (Colmar Brunton → Kantar → Verian) is the only one that runs through every
cycle, so this compares like with like. No PM data exists for Bolger.

| PM | Last 1 News poll before winning | First 6 months | Peak in term | Low in term | 30–36 months |
|---|---|---|---|---|---|
| Clark | 30 | 43 | 52 | 37 | 50 |
| Key | 40 | 51 | 57 | 45 | 55 |
| Ardern | 31 | 39 | 63 | 36 | 56 |
| **Luxon** | **25** | **25** | **28** | **16** | **18** |

- **No winner's bump.** Clark, Key and Ardern each gained 8–13 points in the 1 News rating after
  winning. Luxon gained nothing.
- **Lead over the next-best person in the same poll** (6-month averages):
  Key +40 to +47 all term; Clark +20 to +32; Ardern +15 rising to +42; **Luxon +10 → +8 → +6
  → +3 → +1 → +1**. He began closer to his rival than any of them and the gap has closed to almost nothing.
- Caveat: few polls in the older cycles (Clark 6 in-term 1 News polls). Treat bins as indicative.

## 3. Right/wrong direction (Roy Morgan, the only pollster asking it throughout)

Net right direction, 6-month averages by months since election

| Term | 0–6 | 6–12 | 12–18 | 18–24 | 24–30 | 30–36 |
|---|---|---|---|---|---|---|
| Ardern 2017–20 | +42 | — | — | — | +38 | +55 |
| Ardern/Hipkins 2020–23 | +48 | +26 | 0 | −12 | −14 | −20 |
| **Luxon 2023–26** | **−7** | **−9** | **−11** | **−12** | **−17** | **−19** |

- **The change of government barely lifted the mood.** Roy Morgan had net −24 in Aug 2023,
  −7 just before the election, and −7 to −10 in Luxon's first months. Labour's terms each
  opened at +40 or better. Across all pollsters, net direction was positive only in a brief window in
  Jan–Feb 2024 (RM +1, Curia +4.5, Talbot Mills +2).
- **Now:** RM −26.5 (Sep 2026), Freshwater −29, Reid −21.5. In Roy Morgan terms this is worse than in the
  final months of the Labour government it replaced (−24 in Aug 2023).
- Labour 2020–23 started +48 and fell nearly 70 points. Luxon started near −7 and fell about 15.
  The previous government lost the public's goodwill; this one never had it.

## 4. Within the term: when did they fall?

Steepest non-overlapping 90-day falls in each trend (house-adjusted). Each is tested by comparing
raw polls just before and just after, with pollster fixed effects.
*Sig* = p < 0.05 on that test. The National vote's normal 90-day change is about ±1.

| Episode | National vote | Luxon pref PM | Net right direction | What fell |
|---|---|---|---|---|
| **Autumn 2024** (Feb → May 2024) | −1.7 trend; −2.1, p=0.007 | −1.5 trend; −2.5, p<0.001 | −3.8 trend; −7.6, p=0.11 (Roy Morgan −2 → −19, Feb → Mar) | **All three** (mood marginal) |
| **Summer 2024–25** (Oct 2024 → Feb 2025) | −2.8 trend; −4.2, p=0.001 | −2.4 trend; −4.5, p<0.001 | −3.3 trend; −9.3, p=0.04 | **All three** (the main break) |
| Winter 2025 (May → Aug 2025) | −0.8, n.s. | −0.9 trend; −2.5, p=0.02 | −2.2, n.s. | Luxon only |
| Summer 2026 (Jan → Apr 2026) | −2.4 trend; −3.5, p=0.005 | flat | −1.5, n.s. | National only |
| Winter–spring 2026 (Jul → Oct 2026) | −3.3 trend; −1.3, p=0.003 | flat (22.2) | **−5.6 trend; −11.4, p<0.001** | National + mood |

**New in this brief: autumn 2024.** The honeymoon ended in about March 2024. Direction went from
net positive in Jan–Feb to RM −19 by late March. National fell from ~38 to ~36 and Luxon from ~29 to ~27.6. Smaller
than the later falls, but it is the first time all three moved together. *Events to check against
Knowledge Basket:* Luxon's accommodation allowance row (Feb 2024), recession confirmed
(Q4 2023 GDP, ~21 Mar 2024), public service job cuts (Mar–Apr 2024), Fast-track Bill.

**Latest fall: winter–spring 2026.** The direction measure fell faster than at any other point this term,
from about −17 to −25, after the August coup. National fell with it. Luxon's own rating did not.
It sits at the end of the series, where the trend is least reliable, and rests on only 6 direction polls.

**How the three relate** (correlation of 90-day changes in the trends)
- National vote and direction: **r = +0.59**. The party moves with the national mood.
- Luxon and National: r = +0.22. Luxon and direction: r = −0.28.
- **Luxon's rating stopped tracking anything after mid-2025.** It has sat at 22–23 in the trend
  through the summer 2026 slide, two leadership challenges and the worst direction numbers of the
  term. Either he had reached a floor of committed supporters, or the problem was never mainly him.

---

## Suggested shape (~900–1,200 words, standalone or folded into the main brief)

1. **Hook:** Bolger 1990. A first-term National government at 26%, written off. It won anyway.
   Or open with the other three: every other first term since has gained.
2. **The anomaly:** 3 of 4 MMP-era first terms rose 7–14 points. Luxon's fell 8–11.
   Only Bolger and Labour 2020–23 did worse.
3. **No honeymoon:** no winner's bump in preferred PM, a single-digit lead over his rival, and
   right/wrong direction that barely moved with the change of government.
4. **Five falls:** autumn 2024 and the summer of 2024–25 hit everything at once. After
   that the leader and the party come apart.
5. **The decoupling:** National now tracks the national mood and Luxon tracks nothing. That
   bears on whether the coups were aimed at the right target.
6. **Close:** the Bolger precedent cuts both ways. He recovered from a year-one collapse.
   Luxon's low point is now, three weeks out (election 7 November). (Don't claim "no government has ever recovered
   from a late slide": not tested.)

## Caveats

- Older cycles: no pollster labels before 2017 and far fewer polls. Bolger's numbers come from one
  pollster under FPTP. Cross-term figures are unadjusted trends; Luxon's adjusted figure is ~2 points lower.
- Ardern's first term includes the Covid rally; her 30-month figure is inflated.
- Clark's term ran only 32 months (snap election, Jul 2002).
- Preferred PM formats differ by pollster (Curia prompts more names), so the cross-term PM
  comparison uses 1 News only. The within-term Luxon trend is house-adjusted across all pollsters.
- Right/wrong direction before 2023 is Roy Morgan-heavy, with a gap from 2018 to 2019.
- Late 2026 has few polls; describe end points as "recent polls".
