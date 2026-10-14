# Cost of Living and the Vote: Which Issues Decide New Zealand Elections?

**Date:** 15 October 2026
**Script:** `reports/panel_deep_dive.py` (all new numbers below); earlier results from `approval_drivers.py`, `mortgage_analysis.py`, `petrol_vs_incumbent.py`, `household_income_analysis.py`
**Data:**
- NZ Election Study 2023 (n=1,989)
- Ipsos Issues Monitor (34 waves, Feb 2018–Sep 2026)
- Ipsos party capability (16 waves, Feb 2023–Sep 2026)
- Stats NZ CPI (to the Dec 2025 quarter)
- 1,020 published party-vote polls (pollster-identified from the 2020 cycle)

---

## Summary

The question: cost of living, health, housing, unemployment and fuel prices all rank highly in issues polls. Which of them actually decides votes, and how does cost of living move them?

1. **Ranking high in an issues poll doesn't make an issue decisive.** In 2023, 69% of voters said the economy or inflation affected their vote. Among 2020 Labour voters, those who stayed with Labour said so as often (66%) as those who left (64%). An issue nearly everyone names can't explain why some voters moved and others didn't.
2. **Cost of living decided where defectors went, not whether they defected.** Naming it as the top issue didn't raise the odds of leaving Labour (odds ratio 0.96, n.s.). But those who left went to National at nearly twice the rate of other defectors (19% vs 11%), and to the Greens at less than half the rate (4% vs 10%). A move across the blocs changes who governs; a move within a bloc doesn't. That's where the issue's electoral force lies.
3. **No party owns cost of living, so it works as a referendum on whoever governs.** Among 2023 voters whose top issue was cost of living, 32% named National as best on it, 27% Labour, and 24% no party or don't know. The economy, by contrast, was owned by National (69%).
4. **Concern ratchets on the price level and pulses with the grocery bill.** It jumped from about 25% to over 50% in early 2022 and hasn't been below 50% since, even as inflation fell from 7.2% to 2.2%. Above that floor, it has tracked food inflation (r=+0.73) more closely than headline CPI (r=+0.47) since 2022.
5. **The issue has worked differently under each government.**
   - Under Labour (2021–23), cost-of-living concern tracked approval almost perfectly (r=-0.91).
   - Under National (2023–26), it has been a constant (55–63%) and doesn't track the slide (r=-0.35, n.s.).
   - National's decline lines up instead with rising concern about the economy and jobs, and falling concern about crime, the issue it owned in 2023.
6. **The final step, from punishing the government to rewarding the opposition, has weakened.**
   - Over summer 2024–25, votes moved straight from National to Labour.
   - Since early 2025 the movement has stayed within each bloc: National to NZ First, and Labour to the Greens and TOP.
   - Labour leads on cost of living (31% to 21%) but polls 29%.
   - National's capability ratings track its vote closely (r=0.92); Labour's track it much less closely (r=0.66).

---

## 1. Why salience isn't power

"Most important issue" questions measure how many people think about an issue. They don't measure whether the issue changes anyone's vote (Wlezien 2005). For an issue to move votes between parties, more has to be true than that people care about it:

| Condition | What it means | Literature |
|---|---|---|
| **Salience** | People think about it | Wlezien (2005) |
| **Attribution** | People hold the government responsible | Fiorina (1981); Clarke et al. (2009) |
| **Ownership** | One party is credibly better on it, or none is | Petrocik (1996); Bélanger & Meguid (2008) |
| **Direction** | It moves voters *between* blocs, not within them | Specific to MMP |

Bélanger and Meguid's point matters most here: ownership only influences vote choice when the issue is also salient. The fourth condition is specific to MMP. A swing from Labour to the Greens can change a party's seat count without changing the government.

Cost of living meets all four conditions. Health, the second-biggest concern, meets the first, partly meets the second, and fails the fourth.

---

## 2. Evidence from the 2023 Election Study

The NZES 2023 asked respondents for their single most important issue (free text, coded into topics), which party would be best at dealing with it, their 2023 party vote (validated where available) and their recalled 2020 vote. Results below are weighted (`vwgt`).

### 2.1 Top issue, defection from Labour, and destination

Rows are respondents who voted Labour in 2020, grouped by the issue they named as most important in 2023.

| Top issue (2023) | % of all voters naming it | n (2020 Lab voters) | Left Labour | → National | → Nat/ACT/NZF | → Green | Best party: Nat | Lab | None/DK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cost of living / inflation | 17 | 143 | 56 | **19** | 27 | **4** | 32 | 27 | 24 |
| Economy | 16 | 103 | 61 | 23 | 27 | 5 | **69** | 14 | 7 |
| Health | 7 | 103 | 61 | 11 | **16** | 8 | 27 | 30 | 15 |
| Poverty / inequality | 6 | 82 | 55 | 5 | 7 | 9 | 10 | 40 | 3 |
| Housing | 6 | 54 | 61 | 10 | 13 | 15 | 22 | 31 | 7 |
| "Time for a change" | 5 | 51 | 78 | 7 | 25 | 37 | 27 | 4 | 3 |
| Crime | 5 | 23 | 81 | 35 | 47 | 1 | 72 | 3 | 2 |
| Climate | 4 | 51 | 76 | 5 | 11 | 47 | 1 | 6 | 4 |
| **All 2020 Labour voters** | | 875 | 58 | 11* | 23 | 10* | | | |

*\* For 2020 Labour voters who did not name cost of living.* "Left Labour" includes not voting (about 20%).

### 2.2 What predicts leaving Labour at all

Weighted logistic regression of leaving Labour (2020 Labour voters):

| Top issue | Odds ratio | p |
|---|---:|---:|
| Crime | 3.03 | 0.047 |
| "Time for a change" | 2.71 | 0.019 |
| Economy | 1.23 | 0.344 |
| Health | 1.13 | 0.608 |
| Housing | 1.13 | 0.670 |
| Cost of living / inflation | 0.96 | 0.817 |
| Wages / employment | 0.25 | 0.018 |

### 2.3 Reading it

- **Self-reported influence doesn't tell switchers from stayers.** "Did the economy or inflation affect your vote?" got 66% yes from Labour stayers and 64% from leavers.
- **Cost of living did not drive defection. It shaped direction.** Defectors who named it went to National rather than the Greens. Because it pushed voters across the blocs, it mattered more to the result than its effect on the defection rate suggests.
- **Health voters stayed on the left.** Only 16% of 2020 Labour voters whose top issue was health moved to the right, against 23% overall. Health concern weakens the government's rating without delivering votes across the blocs.
- **Ownership separates the economy from cost of living.** The economy belonged to National in 2023. Cost of living belonged to no one: a quarter of those naming it could name no party. An unowned issue doesn't reward a party's brand. It punishes whoever governs.

---

## 3. The mechanism: from prices to votes

The project's earlier analyses establish the chain:

| Step | Evidence | Source |
|---|---|---|
| Visible prices → incumbent support | Petrol: Granger-causal, same-month, quarterly r=-0.35. Mortgages: Granger-causal, ~1-month lag, r=-0.31 | Analyses 12, 14 |
| Cost-of-living concern → approval | Concern alone explains 67% of approval variance (2018–26) | Analysis 13 |
| Approval → vote | r=0.857; +1 point on the 0–10 rating ≈ +5.9pp incumbent vote | Analysis 9 |
| Mediation | 70% of the mortgage effect and 51% of the petrol effect pass through approval | Analyses 13, 14 |
| Destination | Historically the main opposition only. Minor parties used to be insulated from these swings | Analyses 12, 14 |

Two new results refine the second step.

### 3.1 The ratchet: concern follows the price level, not the inflation rate

| Ipsos wave | Cost-of-living concern (%) | Headline CPI (y/y %) | Food CPI (y/y %) | Prices vs Dec 2020 (%) | Govt rating |
|---|---:|---:|---:|---:|---:|
| Jun 2021 | 27 | 3.3 | 1.6 | +2.2 | 6.5 |
| Feb 2022 | 53 | 5.9 | 4.1 | +5.9 | 5.7 |
| Feb 2023 | **65** | 7.2 | 10.7 | +13.6 | 5.4 |
| Aug 2023 | 62 | 6.0 | 12.3 | +16.2 | 4.5 |
| Feb 2024 | 59 | 4.7 | 5.7 | +18.9 | 4.6 |
| Aug 2024 | 56 | 3.3 | 0.2 | +20.1 | 4.5 |
| Feb 2025 | **50** | 2.2 | 1.3 | +21.5 | 4.2 |
| Aug 2025 | 60 | 2.7 | 4.2 | +23.3 | 4.2 |
| Oct 2025 | 61 | 3.0 | 4.6 | +24.6 | 3.9 |
| Sep 2026 | **63** | — | — | — | 4.1 |

*Each wave is matched to the latest CPI quarter published before it.*

| Cost-of-living concern vs. | 2018–25 | 2022–25 only |
|---|---:|---:|
| Headline inflation | +0.72 | +0.47 (p=0.08) |
| Food inflation | +0.59 | **+0.73 (p=0.002)** |
| Price level vs Dec 2020 | **+0.90** | +0.04 (p=0.90) |

Across the full period, concern tracks the price level (r=0.90). Within the high-concern era it doesn't (r=0.04). The level set a floor in 2022, and the floor hasn't moved because prices haven't come back down.

This fits Stantcheva (2024): people experience inflation as a loss of what their pay will buy, and they judge prices against what they remember paying. A slower rate of increase doesn't register as relief. In New Zealand, the 2021–23 price shock permanently raised the baseline of concern. Concern was around 25% before 2022. It hasn't fallen below 50% since.

### 3.2 The pulse: food prices

Above the floor, concern moves with food inflation. The cleanest test is 2025:
- Headline inflation moved only from 2.2% to 3.0%, within the Reserve Bank's band.
- Food inflation rose from 1.3% to 4.6%.
- Concern rose from its term low of 50% to 61%.

Groceries now play the role petrol played in 2022: the most visible price, and the one people react to.

---

## 4. The same issue under two governments

Correlation of each issue's concern level with the government's Ipsos rating, within each government's term (*Δ* = wave-to-wave changes):

| Issue | Labour 2021–23 (n=10) | National 2023–26 (n=12) | National, Δ |
|---|---:|---:|---:|
| Cost of living | **-0.91** | -0.35 (n.s.) | +0.27 (n.s.) |
| Economy | -0.39 (n.s.) | **-0.82** | -0.32 (n.s.) |
| Unemployment | +0.87 | **-0.84** | -0.21 (n.s.) |
| Crime | -0.84 | **+0.80** | +0.28 (n.s.) |
| Housing | +0.90 | +0.79 | +0.36 (n.s.) |
| Health | -0.50 (n.s.) | -0.53 (p=0.08) | -0.07 (n.s.) |

Correlations with the incumbent's vote share are similar or stronger: National term, economy -0.94, unemployment -0.92, crime +0.89.

**Interpretation.**
- **Under Labour**, cost of living was the dynamic issue. Concern rose from 27% to 65% while the government's rating fell from 6.5 to 4.5.
- **Under National**, cost-of-living concern has been a constant backdrop. It was high when they took office and is at a record 63% now. The issues that have *moved* during the term are:
  - economy concern, up from 25% to 34%
  - unemployment concern, up from 7% to 21%
  - crime concern, down from 37% to 15%

  The crime decline matters because crime was the issue on which National's 2023 vote rested (section 2.1: 72% of crime-focused voters named National best on it). As crime fades from the agenda, National loses its owned issue. As jobs rise, the agenda moves onto ground where Labour leads (34% to 21% on unemployment).

**Caveat.** These are trends across 10–12 waves. None of the wave-to-wave relationships is significant. The table describes slow shifts in the agenda, not reactions to each wave, and with all series trending, specific correlations shouldn't be read as precise effects.

### Other issues in brief

- **Fuel:** cost of living's sharpest edge. It has the strongest and fastest pocketbook effect in the project's data.
- **Health:** 44%, second-biggest concern, and up 6 points in the latest wave (the largest single-wave move of 2026). Labour owns it (38% to 21%). On current evidence it weakens the government's rating rather than moving votes across the blocs, but it's the issue most likely to gain that power if it keeps rising.
- **Housing:** 19%, a series low. House prices show no net effect on the incumbent's vote, because homeowners' gains and renters' losses cancel (Analysis 18).
- **Household income:** real disposable income per person rose about 2.6% in the year to June 2026 while the government's vote kept falling (Analysis 31). Income statistics don't register. Prices people see do.

---

## 5. Where the punished votes are going

### 5.1 Blocs are stable; the major parties are not

Unweighted quarterly means of published polls:

| Quarter | Nat | Lab | Grn | ACT | NZF | TOP | TPM | Lab+Nat | Right (Nat+ACT+NZF) | Left (Lab+Grn+TPM) | Polls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 Q1 | 38.2 | 25.9 | 12.5 | 9.5 | 6.2 | 2.5 | 3.6 | 64.1 | 53.9 | 42.0 | 9 |
| 2024 Q4 | 33.8 | 29.6 | 11.3 | 9.7 | 6.4 | 1.9 | 5.1 | 63.4 | 49.8 | 45.9 | 11 |
| 2025 Q1 | 32.2 | 31.5 | 11.6 | 9.3 | 6.5 | 1.8 | 4.9 | 63.6 | 47.9 | 48.0 | 10 |
| 2025 Q3 | 32.0 | 33.1 | 11.1 | 8.4 | 8.6 | 1.9 | 3.9 | 65.1 | 49.0 | 48.1 | 11 |
| 2026 Q1 | 31.1 | 33.9 | 10.5 | 7.7 | 10.2 | 2.2 | 2.7 | 65.0 | 49.0 | 47.1 | 12 |
| 2026 Q2 | 29.3 | 32.8 | 10.8 | 7.9 | 12.0 | 3.9 | 2.5 | 62.1 | 49.2 | 46.0 | 12 |
| 2026 Q3 | 29.7 | 29.2 | 12.2 | 8.7 | 10.2 | 6.1 | 2.1 | 58.9 | 48.6 | 43.5 | 14 |
| 2026 Q4 | 27.8 | 28.8 | 14.8 | 9.3 | 10.6 | 6.3 | 1.1 | 56.6 | 47.8 | 44.6 | 4 |

There have been two phases:

1. **Summer 2024–25: movement across the blocs.** The right fell 6 points and the left rose 6. National lost and Labour gained. This is the classic incumbent-punishment pattern from section 3.
2. **2025–26: movement within each bloc.**
   - Since early 2025 the right bloc has held at 48–49%. National's further 4-point loss was matched by NZ First's 4-point gain. A partner within the government growing as the senior partner shrinks is a protest within the coalition.
   - In 2026 Labour fell 5 points. The Greens (+4) and TOP (+4) rose. TOP has no declared bloc, but the timing suggests it draws mainly from the left.

Combined Labour and National support held at 62–65% until mid-2026, then fell to 59% (Q3) and 57% so far in Q4.

### 5.2 The seesaw still works from one poll to the next

Correlation of consecutive changes in National and Labour within the same pollster:

| Period | r | p | n |
|---|---:|---:|---:|
| Labour govt 2020–23 | -0.37 | <0.001 | 114 |
| National govt 2023–25 | -0.18 | 0.13 | 74 |
| National govt 2026 | **-0.47** | 0.004 | 35 |

Short-term poll movements are still zero-sum between the majors. What's new in 2026 is a shared downward drift underneath them.

### 5.3 Competence ratings punish National but don't reward Labour

Ipsos "best party to manage" (first wave Feb 2023 → latest wave Sep 2026, with National's peak):

| Issue | National | Labour | Either major |
|---|---|---|---|
| Cost of living | 34 → 21 (peak 39) | 31 → 31 (peak 36) | 65 → **52** |
| Economy | 39 → 28 (peak 47) | 30 → 30 | 69 → 58 |
| Health | 25 → 21 (peak 33) | 36 → 38 | 61 → 59 |
| Unemployment | 30 → 21 (peak 38) | 27 → 34 | 57 → 55 |

Does each major party's capability rating track its own vote? (16 waves)

| Issue | National: capability ~ vote | Labour: capability ~ vote |
|---|---:|---:|
| Cost of living | **0.92** | 0.66 |
| Economy | **0.93** | 0.57 |
| Health | **0.84** | 0.50 |
| Unemployment | **0.91** | 0.59 |

National's vote moves almost one-for-one with its perceived competence. Labour's moves only loosely with its own. Labour's capability lead has grown, peaking at 36% to 24% on cost of living in October 2025, without a matching vote gain.

On cost of living, only 52% now name either major party as best, the lowest in the series. The pattern fits voters who have withdrawn trust from National without transferring it to Labour, which is the "no reason to come home" account of the minor-party surge. Aggregate polls can't separate this from genuine attraction to the Greens, TOP or NZ First at the individual level.

---

## 6. Implications

1. **Cost of living is the threshold issue, not the swing issue, this term.** It set a high floor of discontent that any government would struggle with. The incremental damage to National has come from economy and jobs concern, plus the fading of crime.
2. **Falling inflation won't rescue the government.** Concern follows prices people remember paying, and lately the grocery bill. Only real income growth large enough to restore pre-2022 purchasing power, or visibly falling prices for groceries and fuel, is likely to lower it.
3. **For Labour, the problem is converting competence into votes, not earning it.** Labour leads on every tracked issue but its vote doesn't follow its ratings. Under MMP, voters who reject National can stay on the right through NZ First.
4. **Watch health and unemployment.** Health has the salience but not yet the cross-bloc pull. Unemployment is rising and owned by Labour. Either could become the swing issue of the campaign.

---

## Limitations

- **NZES evidence is from 2023 only.** It shows how cost of living worked against a Labour government. Applying the mechanism to National is an inference from the capability data. The 2026 NZES won't be available until after the election.
- **Recall and rationalisation.** 2020 vote is recalled in 2023, and voters who switched may name cost of living to justify the switch.
- **Small n and trending series.** The within-term correlations use 10–12 waves of trending data. Wave-to-wave changes are all non-significant.
- **CPI coverage.** The project's Stats NZ file ends at the December 2025 quarter, so the food-inflation link can't yet be checked against the 2026 waves.
- **Poll aggregation.** Bloc totals are unweighted quarterly means without house-effect adjustment. TOP is in neither bloc. The latest quarter has only 4 polls.
- **Pollster identity** exists only from the 2020 cycle, so same-pollster flows can't be compared with earlier governments.

---

## References

- Bélanger, É. & Meguid, B. (2008). Issue salience, issue ownership, and issue-based vote choice. *Electoral Studies* 27(3).
- Clarke, H., Sanders, D., Stewart, M. & Whiteley, P. (2009). *Performance Politics and the British Voter.* Cambridge UP.
- Fiorina, M. (1981). *Retrospective Voting in American National Elections.* Yale UP.
- Petrocik, J. (1996). Issue ownership in presidential elections. *American Journal of Political Science* 40(3).
- Stantcheva, S. (2024). Why do we dislike inflation? *Brookings Papers on Economic Activity.*
- Wlezien, C. (2005). On the salience of political issues: the problem with "most important problem". *Electoral Studies* 24(4).

---

## Reproduce

```bash
python reports/panel_deep_dive.py
```
