# Where the coalition went wrong — data brief

Extended column, ~1500 words. Last updated 2026-10-14.
Sources: `data/2026_polling.json`, `data/pm_polling_all.csv` (refreshed 14 Oct), Ipsos
CSVs, Stats NZ CPI, RBNZ hb20, BIS real house prices, OECD CCI, `column/data/econ/`,
Knowledge Basket. Decline analysis: `reports/coalition_declines.py`.

---

## The argument in one paragraph

National's fall happened in two phases with different causes. **Summer 2024–25** was
the real break. National and Luxon fell together, votes crossed to Labour, and National
lost ground on every issue at once. It started in the same week of October 2024 that
Willis declared "the era of crushing price rises is now over" and a poll found 51% thought
Luxon out of touch. **2026** is a party problem, not a Luxon problem. National kept falling
while his personal rating held, and the votes went to coalition partner NZ First and to
TOP rather than to Labour. In between, the government repeated the error: in August 2025
Willis dismissed critics as "merchants of misery", GDP then fell twice as far as forecast,
and approval hit its term low. Underneath it all, cost-of-living concern never fell and
real house prices had their worst term since at least 1990.

---

## 1. The arc

- National (pollster-adjusted trend): 37.5 (end-2023) → 36.1 (Sep 2024) → 32.4 (Jan 2025)
  → ~32 through 2025 → 29.4 (Apr 2026) → 26.7 (Oct 2026)
- Roughly −0.25pp/month: twice the average incumbent decay (−0.11), well short of
  Labour 2020–23 (−0.56). Worse than normal, not a collapse.
- Luxon preferred PM (trend): 31.2 → 28.5 (Mar 2024) → 24.3 (Mar 2025) → flat at 22–23 since mid-2025
- Ipsos govt rating (0–10): 4.7 (Sep 2023) → 4.7 (Oct 2024) → 4.2 (Feb 2025) → **3.9 (Oct 2025, term low)** → 4.1 (Sep 2026)
- Govt bloc (Nat+ACT+NZF) still ~48 vs left bloc ~44–46
- National + Labour combined: 56.6 this quarter, **the lowest since 1996** (56.0, the first
  MMP year, when NZ First surged). 2026 Q3 (58.9) is the third-lowest quarter on record.

**Steepest falls** (LOWESS trend after removing pollster house effects)

| | National | Luxon (preferred PM) |
|---|---|---|
| 1st | Sep 2024 → Jan 2025: −3.7 (36.1 → 32.4) | Nov 2024 → Mar 2025: −3.0 (27.4 → 24.3) |
| 2nd | Jun → Oct 2026: −3.2 (29.9 → 26.7)* | Nov 2023 → Mar 2024: −2.7 (honeymoon fade) |
| 3rd | Dec 2025 → Apr 2026: −2.8 (32.3 → 29.4) | none after mid-2025 |

\*The steepest *60-day* fall is 13 Aug → 12 Oct 2026 (−2.9), starting the day after the
August coup. It's at the series end-point, where the trend is least reliable, with only
4 polls in Q4.

---

## 2. The break: Sep 2024 – Jan 2025

**Where the votes went** (adjusted trend, 1 Sep 2024 → 31 Jan 2025)
- National −3.9, Labour +2.4, ACT +1.0, Te Pāti Māori +0.7; Greens, NZ First, TOP flat
- Govt bloc −3.2, left bloc +2.7. This is the **only period of the term when votes crossed
  the aisle**. In 2026 they stayed inside the tent.

**Two steps, not a slide**
- Step 1, Sep → Oct 2024: Curia 39.0 → 34.9; Roy Morgan 37.5 → 31.0 → 28.5 (Nov)
- Step 2, Dec 2024 → Jan/Feb 2025: Curia 34.2 → 29.6; Verian 37 (4 Dec) → 34 (7 Feb)
- Both flanks gained around the Treaty Principles Bill: ACT 13.0 in Curia and RM (Dec);
  Te Pāti Māori 7–9 in RM, Verian and Labour–Talbot Mills (Nov–Dec)

**Luxon fell, Hipkins didn't rise**
- Verian: Luxon 28 → 25 → 24 → 22 (Aug 2024 → Feb 2025); Hipkins flat at 15–18
- Curia: Luxon 32.7 → 20.7 (Sep 2024 → Feb 2025)
- First lost head-to-heads: Freshwater late Nov 2024 (Hipkins 42–41); Labour–Talbot
  Mills 28 Nov; Talbot Mills Jan 2025; Curia 10 Mar 2025

**National lost ground on every issue at once**
- Salience-weighted National capability lead (6 Ipsos issues): +7.6 (Aug 2024) → +2.9
  (Oct) → +0.2 (Feb 2025), and negative every wave since
- Of the −7.4 change Aug → Feb, **−7.0 came from National's lead falling on each issue**;
  only −0.4 from the issue mix changing
- Economy 15 → 7, cost of living 11 → 4, crime 19 → 14, housing 4 → −2,
  health −5 → −13, unemployment −2 → −10
- Meanwhile cost-of-living concern was *falling* (56 → 50) and crime concern too (28 → 25).
  The usual government-killers were easing.
- Reads as a general loss of credibility. Caveat: capability tracks vote intention
  closely (r = 0.82–0.94), so it is partly the vote fall showing up again.
- Competence ratings fell first (by Oct 2024); the overall rating stayed at 4.7 and only
  fell by Feb 2025.

**The economy in the window: better on paper, worse in jobs**
- Mortgage 2yr fixed 6.5% → 5.7%; OCR cuts from Aug 2024 (50bp Oct and Nov)
- CPI back in the target band (2.2%); consumer confidence *rising* (CCI 97.6 Aug → 98.6 Dec)
- But unemployment 4.7% → 5.1%; unemployment concern 8% → 13% (May → Aug 2024);
  GDP −1.2% in Q3 2024 (released 19 Dec)
- Fits rule 6: under this government, rate cuts read as a sign of recession, not relief

---

## 3. Timeline of the break (Knowledge Basket; publication dates)

**Step 1, the "wealthy and out of touch" month**
- 23–28 Sep 2024: Willis orders public servants back to the office ("Don't make me tell you again")
- 1 Oct: Post reports Luxon avoided tax on his Wellington apartment sale only because his
  government changed the bright-line test. He sold rental properties on 23 Sep, 11 Oct and 4 Dec.
- Early Oct: Luxon says he is **"wealthy and sorted"**. It topped nominations for Massey's
  Quote of the Year 2024. *Original date and context still to pin down.*
- 16 Oct: Willis: **"The era of crushing price rises is now over."**
- Same week: Verian poll has **51% saying Luxon is out of touch**; preferred PM 25 (−3)
- 21 Oct: Bayly "loser" incident, no action. ~1 Nov: I Am Hope / youth mental health row.

**Step 2**
- 5 Nov: Luxon apologises in Parliament over Verrall's sister-in-law (heated-tobacco row)
- 14 Nov: Treaty Principles Bill first reading; hīkoi follows
- Early Dec: Freshwater has Hipkins ahead as preferred PM; 48% say wrong direction
  (Post, 8 Dec: "Why has Christopher Luxon proven to be so hard to like?")
- 11 Dec: Ferries: Willis says **"I've delivered"**, though details are scarce. Post: "Willis'
  masterclass in incompetence". (Reversed Mar 2025, when Peters reordered rail ferries.)
- 17 Dec: HYEFU: no surplus in sight; Willis adopts OBEGALx and is accused of "creative accounting"
- 19 Dec: Q3 GDP fall confirmed; Luxon says he'll skip Waitangi
- 19 Jan 2025: reshuffle; Reti loses health
- 23–25 Jan: "Going for growth" state of the nation ("Surplus is dead! Long live growth!", SST)
- 29 Jan: Treasury contradicts Luxon's "three years of recession that Labour created"

**After: Luxon kept falling while National held**
- 11 Feb: Question Time "own goal"; 24 Feb: Bayly resigns; 4 Mar: "go make a Marmite
  sandwich"; 10 Mar: Hipkins overtakes in Curia; 11 Mar: "pretty lame-o"

---

## 4. The repeat: "merchants of misery" (Aug 2025)

- 20 Aug 2025 (OCR cut day): Willis says households are listening to **"merchants of misery"**
  and opposition "doomsayers": "Do we talk ourselves into an ongoing funk, or do we look
  ahead and recognize that things will get better?" (RNZ). Earlier that month she attacked
  the "commentariat" for "glass-half-empty economics". The Herald (21 Aug) reports she called
  negative talk about the economy "unpatriotic".
- 18 Sep 2025: June-quarter GDP falls about twice as far as forecast. Herald column:
  "Despite Luxon's Government last month castigating economists as 'merchants of misery',
  the economy worsened by twice as much as any had forecast."
- Ipsos rating 4.2 (Aug) → **3.9 (Oct 2025), the term low**; rated poorly 41% → 45%
- National's vote barely moved (31.9 → 31.5); NZ First 8.5 → 9.8
- **Pattern:** Oct 2024 declares victory over inflation, then the vote falls. Aug 2025
  dismisses the doomsayers, then GDP falls and approval bottoms. Both times good news was
  announced before voters felt it.

---

## 5. The background grind

**Cost of living never stopped being the issue**
- CPI: 5.6% (Sep 2023) → 2.2% (late 2024) → back to 3.1% (Dec 2025 – Mar 2026)
- Mortgage 2yr fixed: 7.5% → 5.3%
- Ipsos cost-of-living concern: 59% (Feb 2024) → 50% (Feb 2025 low) → **63% (Sep 2026, term high)**
- Unemployment 4.4% → 5.4%; GDP fell in 2024 Q2, 2024 Q3 and 2025 Q2

**Losing the issues National owns** (Ipsos, National v Labour)

| Issue | Sep 2023 | Sep 2026 |
|---|---|---|
| Economy | 41–26 | 28–30 |
| Cost of living | 35–27 | 21–31 |
| Healthcare | 30–29 | 21–38 |

Labour first led on cost of living in May 2025 (32–31).

**House prices: worst term for owners since at least 1990**

| Term | Govt | Real | Nominal* |
|---|---|---|---|
| 2011–14 | National | +16.1% | +20.2% |
| 2014–17 | National | +32.9% | +36.5% |
| 2017–20 | Labour | +20.9% | +27.4% |
| 2020–23 | Labour | −6.1% | +11.7% |
| 2023–Q3 2025 | National | **−7.5%** | **−3.1%** |

- The only term with a *nominal* fall; real prices are ~27% below the 2021 peak
- Understated: the BIS data ends Q3 2025, and the 2026 slump continues
- The first National government whose home-owning base watched its main asset shrink
- \*Nominal is derived (real × CPI); check against REINZ/RBNZ before printing
- Caveat: across 1993–2025 real house prices don't predict incumbent polling (r = −0.07),
  and housing concern fell (29% → 19%). Use as context, not causation.

---

## 6. 2026: a party problem, not a Luxon problem

- National 32 → 27 in 2026; Luxon's personal rating flat at 22–23 throughout
- NZ First 6.4 (early 2024) → 12.4 peak (Apr–May 2026) → ~10.5; TOP ~2 → 6.3
- Labour peaked at 34 (Dec 2025 – Mar 2026), now ~29; Greens 9.4 → 14.8
- **The coups were a symptom, not a cause.** The April confidence vote (Tue 14 Apr 2026)
  followed National's slide to ~30 with NZ First at 11–12. Penk's challenge (Wed 12 Aug
  2026) followed a Curia poll with National at 31.
- Luxon after the August coup is mixed by pollster: Verian 17 → 19, 20, 20; Reid
  20.0 → 19.9, 17.1; Curia 23.2 → 21.2, 21.2; Freshwater (head-to-head) 37 → 35

**Event tests** (35 days before vs after; National's normal 35-day swing SD = 1.2pp,
90% of weeks between −2.5 and +1.3)

| Event | National | Labour | NZ First | Notes |
|---|---|---|---|---|
| Pay equity changes (6 May 2025) | −1.4 (p=0.35) | +0.6 | −0.3 | same-pollster Nat −1.2 |
| Budget 2025 (22 May 2025) | −0.5 (p=0.82) | +0.6 | −0.6 | can't be separated from pay equity |
| April confidence vote (14 Apr 2026) | −1.1 (p=0.49) | +0.5 | +0.9 | same-pollster Nat −1.3 |
| August coup (12 Aug 2026) | −0.9 (p=0.33) | −2.5 (p=0.27) | −0.2 | Greens +2.6 (p=0.06), ACT +1.4 |

- None is distinguishable from noise; Ipsos approval flat across all of them
- National dipped ~1pp after each one: consistent with slow erosion, proves nothing
- The August window's biggest move was on the left (Labour −2.5, Greens +2.6)
- Pay equity likely hit women and public-sector workers more than headline share; can't
  test until NZES 2026 (due 2027)

---

## 7. Suggested shape (~1500 words, 9 paragraphs)

1. **Hook** (~150): "merchants of misery" (Aug 2025), or the 1996 parallel (the last time
   the big two were this weak, Peters chose the government; now he's in it and gaining)
2. **The arc** (~150): the fall, benchmarked: worse than normal decay, not a collapse
3. **One week in October 2024** (~200): "era of crushing price rises is over", "wealthy and
   sorted", tax-free property sales, 51% out of touch. The break begins.
4. **Anatomy of the break** (~200): lost on every issue at once; votes crossed to Labour;
   Luxon's own numbers fell with the party's
5. **Better on paper** (~200): inflation down, rates down, confidence up, yet jobs and GDP
   worsened and cost-of-living concern never fell. Rate cuts read as recession.
6. **The repeat** (~150): "merchants of misery", GDP twice as bad as forecast, approval term low
7. **The owners' recession** (~150): worst real-terms term for house prices since at least
   1990, the first nominal fall, and National's base
8. **Inside the tent** (~200): 2026's losses went to NZ First and TOP; the coups chased a
   leader whose numbers had stopped falling
9. **Close** (~100): big two at a 30-year low; wry and open

---

## Caveats

- Q4 2026 has only 4 polls; describe "now" figures as "recent polls"
- Ipsos: 12 waves this term. The 33rd edition revised 2018–24 approval down 0.1–0.3; CSV
  keeps the as-published figures.
- Timing ≠ cause. In the summer of 2024–25, rate cuts, recession data, the Treaty
  Principles Bill and Luxon's gaffes all overlap.
- Our models say approval follows cost-of-living concern more than any single decision;
  frame "went wrong" as what was in the government's control and what wasn't
- Still to verify: the original date and context of "wealthy and sorted"; nominal house prices
