# Party, PM and national mood: two slumps and a model — data brief

For the column on where National went wrong. Companion to `coalition_feature_brief.md`.
Last updated 2026-10-19. Election: **7 November 2026**.

- Chart: `reports/national_luxon_mood_listener.png` / `.svg` (script `national_luxon_mood_chart.py`)
- Model: `reports/mood_leader_party_model.py` → `reports/mood_leader_party_monthly.csv`
  (run with the latest MBIE `weekly-table.csv` as its argument; the repo copy ends Feb 2026)
- Direction polls: `data/right_wrong_direction_2026.csv` (Wikipedia table, plus the June 2026
  Freshwater poll that the table omits)
- News: Knowledge Basket; `/mnt/d/data/column/news/2026/`

---

## The chart

**"National has shed more than a quarter of its vote since the election."** Three stacked panels from the
14 Oct 2023 election to Oct 2026 (`reports/national_luxon_mood_listener.png`/`.svg`):
National's vote (38.1% result → ~27% trend, −29%), Luxon as preferred PM (~27% → 20%) with three of his
lines marked ("I'm entitled to the entitlement", "Wealthy and sorted", "Make a Marmite sandwich", and
"Who cares what the government thinks or does?": Rotorua Business Chamber, Fri 31 Jul 2026, apologised Mon 3 Aug),
and wrong direction (~48% → ~56%, honeymoon low 42% in Feb 2024; the 56% peak under Labour in Aug 2023
has now been matched, but neither is annotated). Events now include the pay-equity Budget (22 May 2025). Four events are marked on
National's line: tax-cut Budget, Treaty Principles Bill, confidence vote, Penk challenge. There is one shading
style, labelled in place with values: Recession (2024 Q2–Q3: GDP −0.5%, then −1.2%), Sharp GDP fall
(2025 Q2: −0.7% on current data, −0.9% as reported 18 Sep 2025), Petrol spike ($2.55 → $3.51/L,
20 Feb → 10 Apr 2026) and Petrol spike ($3.00 → $3.54/L, 4 Sep → 9 Oct 2026; MBIE regular 91 weekly
board price, low to peak).

Notes from the earlier four-panel version, kept for the column but not charted:

1. **National's party vote:** ~27% when Luxon took over (Nov 2021), a peak of ~37% (early 2024), **~26% now**.
   Events marked: tax-cut Budget (30 May 2024), Treaty Principles Bill first reading (14 Nov 2024),
   confidence vote (14 Apr 2026), Penk challenge (12 Aug 2026).
2. **Luxon as preferred PM:** peaked at ~27% around the election and is 19% now. Flat since mid-2025. His lines
   are marked on the trend:

   | Date | Line | Luxon trend at the time |
   |---|---|---|
   | 29 Feb – 1 Mar 2024 | "I'm entitled to the entitlement" (accommodation allowance) | 25 |
   | 29 Apr 2024 | Poll coverage "frothy and sensationalist" | 24 |
   | 1 Oct 2024 | "Let me be clear: I'm wealthy, I'm — you know — sorted" (Newstalk ZB, Heather du Plessis-Allan; runner-up, Massey Quote of the Year) | 26 → falls to 21 by Mar 2025 |
   | 4 Mar 2025 | Unhappy with school lunches? "make a Marmite sandwich and put an apple in a bag" | 21 |
   | 11 Mar 2025 | Wellington councils "pretty lame-o" | 21 |

   After March 2025 there is nothing of that kind. His rating had reached its floor by then.
3. **Wrong direction** (wrong only): ~56% mid-2023, ~43% in the honeymoon, **~56% now**.
4. **Ipsos, best party to manage the economy, National v Labour** (from Feb 2023). Willis's lines:

   | Date | Line | National's economy rating |
   |---|---|---|
   | ~24 Sep 2024 | Public servants back to the office: "Don't make me tell you again" | 42 (Aug) |
   | 16 Oct 2024 | "The era of crushing price rises is now over" | 40 (Oct) → 36 (Feb 2025) |
   | 11 Dec 2024 | Ferries: "I've delivered" (reversed Mar 2025) | 36 (Feb 2025) |
   | 20 Aug 2025 | Households listening to "merchants of misery" | 32 (Aug) → **29 (Oct)**: Labour ahead 33–29 for the first time |

   The series: 41 at the 2023 election → **47 peak (Feb 2024)** → 40 (May 2024, the autumn-2024 drop, the biggest
   between waves) → 36 (Feb 2025) → 29 (Oct 2025) → 28 v Labour 30 (Sep 2026). National has led
   only once since Oct 2025 (May 2026, 33–28).

**Shading:** grey for quarters when real GDP shrank (2022 Q1, 2023 Q1, 2024 Q2–Q3, 2025 Q2: current Stats NZ
vintage). White for petrol spikes (Mar–Jul 2022, Ukraine; Jul–Oct 2023, the fuel-tax cut ending; Mar–Jun 2026
and Sep–Oct 2026, Iran war).

- **The recession that ended the honeymoon was later revised away.** On 21 Mar 2024 Stats NZ reported
  falls of −0.3% and −0.1% for Q3 and Q4 2023. Current data shows +0.1% and +0.5%. The only technical
  recession on today's numbers is Q2–Q3 2024 (−0.5%, −1.2%), reported in Sep and Dec 2024. Voters reacted
  to the news at the time, not the revision. That's worth a line in the column.

**What else could go on (not added, to keep it legible):** pay equity changes (6 May 2025), the OCR rise
(Jul 2026), Smokefree repeal (Feb 2024), the hīkoi (19 Nov 2024), the 2025 and 2026 Budgets. None moved
the polls measurably in event tests (main brief §6). The **Treaty Principles Bill** is on as an event. Its window
(Nov–Dec 2024) is when ACT and Te Pāti Māori both rose and votes crossed to Labour. But it overlaps "wealthy
and sorted", the bright-line story and the Q3 GDP fall, so its own effect can't be separated out.

---

## 1. How the three interact: the model

**Short answer:** national mood drives the party vote. The PM's personal rating runs on its own clock.

Monthly series, Nov 2020 – Oct 2026, covering both the Labour 2020–23 and National 2023–26 governments. Each is
house-adjusted (month + pollster fixed effects). "Vote" is the governing party, "PM" the sitting PM,
and "mood" is net right direction.

| Test | Result |
|---|---|
| Mood → next month's vote (Granger) | **p = 0.001** (lag 1), p = 0.002 (lag 2) |
| Vote → next month's mood | p = 0.21 — no |
| PM rating → vote, or vote → PM rating | p = 0.57–0.75 — no |
| Mood → PM rating, or PM rating → mood | p = 0.48–0.60 — no |

Under National, using 3-month changes (HAC standard errors):
- Change in mood alone explains **29%** of the change in National's vote (p < 0.001)
- Change in Luxon's rating alone explains 2% (p = 0.48)
- Together: mood is unchanged, and Luxon adds nothing (coefficient −0.13, p = 0.32)
- Over the whole term the three *levels* move together (r = 0.56–0.68), because all three trend down.
  In the short run, Luxon's rating does not move with the other two.

**Petrol is the outside driver in this term.** Correlation with 3-month changes: National vote
r = −0.58, mood r = −0.34, Luxon r = −0.17. The 2026 run (monthly, house-adjusted):

| Month | Petrol (c/L) | Net right direction | National | Luxon |
|---|---|---|---|---|
| Feb 2026 | 256 | −14 | 31.0 | 19.3 |
| Apr 2026 | **344** | **−23** | **27.8** | 18.0 |
| Jul 2026 | 298 | −14 | 30.4 | 19.2 |
| Oct 2026 | **351** | **−25** | **26.6** | 19.4 |

The mood and National's vote rose and fell twice with the pump price. Luxon's rating didn't move.
October's 354c (9 Oct) is a nominal record in MBIE's weekly series. The previous high was 314c in July 2022.

**How this fits the literature and our earlier findings**
- It extends the causal chain we already found (Rule 1: conditions → salience → approval → vote).
  Right/wrong direction is a sociotropic, backward-looking summary judgement of the same kind as
  the Ipsos government rating. In this data it leads the vote by about a month.
- **Macropartisanship** (MacKuen, Erikson & Stimson 1989; *The Macro Polity*, 2002): aggregate party
  support moves with economic sentiment and evaluations of government.
- **Leader effects** are real but modest in the comparative literature (Aarts, Blais & Schmitt 2011;
  Bittner 2011). The "presidentialisation" thesis (Poguntke & Webb 2005) was not confirmed in NZ
  (Analysis 17: PM preference moves with the party vote at the same time, without leading it).
- **A two-clock reading** (my interpretation, not a tested model): the PM's rating is a slow
  *stock*. It forms early, then gets marked down in steps by personal episodes. The mood is a fast *flow*,
  fed by prices and jobs. The party vote is mostly flow. **All three fall together only when a personal
  episode and an economic hit arrive at once.** That happened twice: autumn 2024 and summer 2024–25. In
  2026 the damage was all flow. Both leadership challenges went after the stock.

Caveats: about 70 monthly observations pooled across two governments, with gaps of up to 2 months interpolated.
Roy Morgan's fieldwork runs for a month, so part of the one-month lead could be timing. The 2020–23
short-run changes are weak (levels track closely, r = 0.93); the result is driven by 2023–26.

---

## 2. Autumn 2024: the honeymoon ends

**The shape:** mood and Luxon turned first (Feb–Mar). The party vote followed (Apr–May). There was a partial
recovery after the 30 May Budget.

| Date | What happened | Polls |
|---|---|---|
| 27 Nov 2023 | Government sworn in | |
| Late Jan – early Feb 2024 | Peak honeymoon | RM net direction +1 (28 Jan); Curia +4.5, Talbot Mills +2 (7–10 Feb): **the only positive readings of the term**. National ~38–41 |
| Feb 2024 | Benefits indexed to inflation, not wages; public service told to cut 6.5–7.5%; Seymour talks of pruning school lunches; Smokefree repeal under urgency | Ipsos govt rating 4.6 (late Feb), "on par with Labour (4.7) just before the election" (Grant Duncan, Herald 28 Mar) |
| **29 Feb – 1 Mar** | **Accommodation allowance.** Luxon is claiming $52,000 a year to live in his own mortgage-free Wellington apartment. Defends it ("I'm entitled to the entitlement"), then U-turns within 12 hours and repays $13,000. Herald: "the first PM in more than 30 years" to make that choice. Stuff (9 Apr): "arguably the most glaring gaffe in his premiership". | |
| 10 Mar | | Talbot Mills: Luxon 24 (from 27); net direction −8 |
| **21 Mar** | **Recession confirmed**: GDP −0.3% (Sep qtr) and −0.1% (Dec qtr 2023). Westpac: GDP per person down ~4% from its mid-2022 peak | |
| 24 Mar | | **RM net direction −19** (from −2 a month earlier) |
| 27 Mar | Herald column: "Luxon's honeymoon feels more like a shotgun wedding … no elected Prime Minister has ever had a worse honeymoon." | |
| Mar–Apr | Public-sector job cuts in the headlines; interest deductibility restored for landlords; Fast-track Bill | |
| 29 Apr | | **1 News–Verian: National 36 (−2), Labour 30**; on these numbers the coalition couldn't govern. Luxon 23. Economic pessimism jumps. Luxon calls the coverage "frothy and sensationalist" |
| 10–19 May | | Talbot Mills: National 35, Luxon 25 = Hipkins 25. RM: National 33.5 |
| 30 May | Budget: tax cuts | July: Curia National 37.6, Luxon 34.5, direction +13 (Curia runs high on all three) |

Claire Trevett (Herald, 30 Apr): "There has been no major catastrophe … all those 100-day plans
and three-month action plans seem to have got him precisely nowhere in the polls." She pointed to the
spike in economic pessimism and to job cuts as "the most visible" price of the tax cuts.

**Reading:** a personal episode that fed the "out of touch" frame (the allowance), followed three weeks later
by an economic hit (recession, falling per-capita income), while the government was visibly cutting. It is the
same mix that broke the government again in Oct–Dec 2024.

*To verify:* the exact date the allowance story broke (Herald Sat 2 Mar says "within 12 hours of news";
U-turn on Friday 1 Mar), and the date the Smokefree repeal passed.

---

## 3. 2026: two fuel shocks, and a coup between them

**The shape:** two dips in mood and National's vote, each lining up with a petrol spike from the Middle East
war. Luxon's rating was flat through both, and through two leadership challenges.

**Shock 1 (Mar–Apr 2026):** petrol 256c (Feb) → 344c (Apr). Net direction −14 → −23; National 31 → 28.
The April confidence vote (Tue 14 Apr) came at the bottom of this dip. National and the mood recovered
by July as petrol eased with the ceasefire.

**Between the shocks (May–Jul):** June Freshwater: 57% wrong direction vs 27% right, "the worst since
before this Government was elected". Since mid-2025 the share whose finances are "troubled or impacted"
rose from 35% to 45% (Post/Waikato Times, 20 Jun). Anneke Smith (Post, 15 Jul) on National at 28.7 in Reid: "hard to
imagine what could possibly be acting as such a sustained handbrake … if not Luxon himself or the
economy."

**Shock 2 (Aug–Oct 2026):**

| Date | What happened | Polls |
|---|---|---|
| ~8 Jul | **RBNZ raises the OCR** 25bp to 2.5% as war-driven inflation picks up; banks pass it on in full. Labour's Edmonds: "quick to take credit when mortgage rates came down" | Reid 9 Jul: net direction −12 |
| Late Jul | Brent crude briefly back above US$100 (Houthi attacks); fuel support kept in place | RM 26 Jul: −15 |
| 5 Aug | Stuff Wealth of the Nation: 4 in 10 worse off than last year; 46% name inflation and daily costs as their top voting issue | |
| **12 Aug** | **Penk's challenge.** Caucus keeps Luxon; Penk sacked; Stanford denies involvement | |
| 21–24 Aug | | Reid: net direction **−21**; Luxon's "lowest approval rating to date" (net −23); 45% say National was wrong to keep him vs 37% right. RM 23 Aug: −18.5 |
| 31 Aug | Renewed US–Iran hostilities lift oil | |
| 3–15 Sep | | Curia (8 Sep) and Freshwater (15 Sep): **both major parties below 30%, a first** |
| Sep | National's supermarket intervention ("Comrade Luxon", Herald 18 Sep); MMP referendum; tax-cut hints | Freshwater 11 Sep: net direction −29; RM 27 Sep: **−26.5** |
| 18 Sep – 9 Oct | **Petrol 306c → 354c** (record); diesel $3.19 (29 Sep) | |
| 27 Sep | Our Tomorrow/Kantar: two-thirds say politicians don't care about people like them; 67% think NZ's best years are behind it | |
| 6 Oct | National promises to cut foreign aid to fund Pharmac | Reid: **National 25.9** ("National crashes"); net direction −21.5 |

**Reading:**
- The mood fell in August, before the September petrol surge. That month had the OCR rise, the oil
  scare and the coup. National's vote held at 29–31 through August and fell in September–October as
  petrol hit records.
- The coup itself moved nothing measurable (main brief: National −0.9, p = 0.33). Luxon's rating was 19–20
  before and after.
- In 2026 voters punished the party and stayed sour on the country. They didn't mark the leader down
  any further: his rating had been at its floor since mid-2025.

---

## Suggested shape (~1,000–1,200 words)

1. **Hook:** the chart's headline. National is back where it was the week Luxon took over, and
   three in five voters say the country is going the wrong way.
2. **The honeymoon that wasn't** (autumn 2024): the allowance, the recession, the shotgun-wedding
   column. Three months of good mood, then gone.
3. **What actually moves National:** mood leads the vote by about a month; Luxon's rating doesn't lead
   anything. Petrol twice in 2026.
4. **Two clocks:** a slow leader stock and a fast economic flow. All three fall only when both hit at
   once (autumn 2024, summer 2024–25).
5. **2026:** the caucus went after the stock while the flow did the damage. The coup moved nothing; the
   fuel price moved everything.
6. **Close:** an incumbent whose fate is set at the pump, three weeks out, with petrol at a record.

## Caveats

- Q4 2026 has few polls; describe end points as "recent polls"
- Direction polls are Roy Morgan-heavy. Curia runs well above others on right direction (house effect);
  the trends adjust for this but single-poll quotes should name the pollster.
- Preferred PM excludes Freshwater, which asks a head-to-head question (Luxon 35–37 vs Hipkins 44–45)
- Petrol is MBIE's regular 91 board price (nominal, before any fuel support); "record" is nominal and
  only within MBIE's weekly series
- Autumn 2024 direction test is marginal (p = 0.11) once Curia's high May reading is included;
  the Roy Morgan series alone shows −2 → −19 from February to March
- The two-clock framing is an interpretation that fits the tests; it is not itself tested
