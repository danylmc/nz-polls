# Election panel: the deeper answers

*As of 15 October 2026. Companion to `panel_answers_2026-10.md`. The numbers come from `reports/panel_deep_dive.py` and the project's earlier analyses.*

This covers the questions the project's data can speak to: 1 (which issue decides votes), 2 (minor parties), 4 (supermarkets, in part) and 5 (media oxygen). Each answer starts with a version you can say on the panel, followed by the reasoning and evidence.

---

## 1. Which issue has the emotional power to decide votes, and how does cost of living shift them?

### Spoken version

Being top of an issues poll is not what gives an issue power. Cost of living has the power because it combines three things. People feel it every week at the checkout and the pump. They blame whoever is in government for it. And no party owns it, so every election becomes a referendum on the incumbent's record on prices.

Health is now the second-biggest concern, but it works differently. People who care most about health mostly vote Labour or Green already. It deepens doubts about the government without moving many votes across the aisle.

There's also a twist this term. Cost-of-living concern has been stuck between 55% and 63% since the election. It set a high floor of discontent, but it isn't what has pushed National down month by month. That decline lines up with the agenda shifting to the economy and jobs, and away from crime, the issue National campaigned on in 2023.

### Why salience isn't the same as power

**Everyone says the economy affected their vote, so the answer doesn't separate anyone.** In the 2023 NZ Election Study, 69% said the economy or inflation affected their vote. Among people who voted Labour in 2020, those who stayed with Labour in 2023 said so at the same rate (66%) as those who left (64%). An issue that everyone ranks highly can't, by itself, explain why some voters moved and others didn't.

**Cost of living shaped where defectors went, not whether they defected.** Among 2020 Labour voters, naming inflation or cost of living as their single most important issue didn't make leaving Labour any more likely (odds ratio 0.96, not significant). It did change their destination:

| 2020 Labour voters, by top issue in 2023 | Voted National | Voted Green |
|---|---:|---:|
| Cost of living / inflation (n=143) | 19% | 4% |
| Everyone else | 11% | 10% |

This is what makes cost of living decisive. A Labour voter who moves to the Greens doesn't change who governs. A Labour voter who moves to National counts twice in the race between the blocs. Cost of living sends voters across the aisle. The issues that predicted leaving Labour at all were crime (odds ratio 3.0) and "time for a change" (2.7).

*(Caveat: 2020 vote is recalled in 2023, and "most important issue" is stated after the fact, so some switchers may name cost of living to justify a move they made for other reasons.)*

**No party owns cost of living, which makes it a referendum on the government.** Among 2023 voters whose top issue was cost of living, 32% named National as best to deal with it, 27% Labour, and 24% no party or don't know. The economy, by contrast, was owned outright: 69% said National. An issue nobody owns doesn't reward a party for its brand. It punishes whoever holds power. That's why the same issue that sank Labour in 2023 is now working against National.

### How cost of living actually moves votes

The project's evidence fits a four-step chain:

1. **Visible prices.** Petrol moves incumbent support in the same month. Mortgage rates take about a month, as people refix. Both are Granger-causal, meaning past price changes predict later changes in the government's support. Petrol's effect is the stronger of the two (quarterly r=-0.35 against -0.31 for mortgages).
2. **Cost-of-living concern**, which works like a ratchet with a pulse (see below).
3. **Government approval.** Across 2018–26, cost-of-living concern on its own explains about two-thirds of the variation in the government's Ipsos rating. About 70% of the mortgage-rate effect and half the petrol effect pass through approval.
4. **Votes move to the main opposition party**, and historically only to it. Minor parties used to be insulated from these swings. That last step is the one now weakening (see question 2).

**The ratchet: people judge the price level, not the inflation rate.** Concern sat around 25% from 2018 to 2021. It jumped to 53% in February 2022 and hasn't been below 50% since, even though headline inflation fell from 7.2% to 2.2%. Since 2022, concern is unrelated to how far prices have risen since 2020 (r=0.04). The price level set a floor and the floor hasn't moved, because prices never came back down. Getting inflation "back in the band" means prices are rising more slowly, but they're still far above what people remember paying. This matches survey work abroad (e.g. Stantcheva 2024, *Why Do We Dislike Inflation?*): people experience inflation as a loss of what their pay will buy, and they don't credit governments for a slower rate of rises.

**The pulse is the grocery bill.** Above that floor, concern since 2022 moves with food inflation (r=+0.73, p=0.002) more than with headline CPI (r=+0.47, not significant). The clearest case is 2025. Concern hit its low for the term, 50%, in February 2025, when food inflation was 1.3%. It climbed back to 60–61% by August–October as food inflation rose to 4.2–4.6%, while headline inflation only edged from 2.2% to 3.0%. (The Stats NZ CPI file in the project ends at the December 2025 quarter.)

**The same issue behaved differently under the two governments.**

| Correlation with government approval | Labour 2021–23 | National 2023–26 |
|---|---:|---:|
| Cost-of-living concern | **-0.91** | -0.35 (n.s.) |
| Economy concern | -0.39 (n.s.) | **-0.82** |
| Unemployment concern | +0.87 | **-0.84** |
| Crime concern | -0.84 | **+0.80** |

Under Labour, rising cost-of-living concern drove approval down. Under National it has been a constant. It was high when they took office, it's high now, and it doesn't track their slide. What does track the slide:
- concern about the economy rising (25% to 34%)
- concern about unemployment rising (7% to 21%)
- concern about crime falling (37% to 15%). As crime fades, National loses the issue it owned in 2023.

*(Caveat: these are trends over 12 waves. Wave to wave, none of the relationships is significant. Read this as a slow shift in the agenda, not a reaction to each poll.)*

### The other issues

- **Fuel prices** are cost of living's sharpest edge: the most visible price, with the fastest effect, and the strongest single pocketbook signal in the data.
- **Health** (44%, up 6 points in the latest wave, the biggest move of any issue this year) is owned by Labour (Ipsos: Labour 38%, National 21% as best party). In 2023 only 16% of 2020 Labour voters whose top issue was health moved to National, ACT or NZ First, against 23% of all 2020 Labour voters. Health concern hurts the government's rating but mostly reinforces votes already on the left. It's the issue most likely to gain vote-moving power if it keeps rising.
- **Housing** (19%, a series low) splits its own constituency. Homeowners gain from rising prices and renters lose, and the two effects cancel in the polling.
- **Unemployment** is the one rising under this government, and Labour leads on it (34% to 21%). It affects relatively few people directly. Its power comes from signalling that the economy is going wrong, and it's where "recovering but still feels difficult" shows up.

### "Recovering but still feels difficult"

The data supports this framing directly:
- Real household disposable income per person rose about 2.6% in the year to June 2026, and the government's vote kept falling.
- Official economic statistics barely predict polling.
- Consumer confidence and approval predict it strongly.

Because of the ratchet, a recovery will feel real when incomes catch up with the 2021 price level, not when the inflation rate falls.

---

## 2. Are voters moving to minor parties because those parties are attractive, or because the majors give them no reason to come home?

### Spoken version

It's mostly the second, but since early 2025 it has been happening inside each bloc, not across them. The right bloc has held at about 48–49% since then. As National fell, NZ First picked up most of the slack. That's dissatisfaction with National, not with the government's direction. The left bloc dipped this year as Labour fell, and the Greens and TOP took those votes. In both cases voters are leaving the big party in their own camp. The striking number is that Labour now leads National on every major issue, including cost of living, yet polls 29%. Competence gains aren't turning into votes.

### The evidence

**Combined Labour and National support held at 62–65% until mid-2026.** Then it dropped to 59% in the third quarter and 57% so far in the fourth (only 4 polls in the latest quarter). The break is this year, not a slow erosion over the term.

| Quarter | National | Labour | Green | ACT | NZ First | TOP | Right bloc | Left bloc (Lab+Grn+TPM) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 Q1 | 38.2 | 25.9 | 12.5 | 9.5 | 6.2 | 2.5 | 53.9 | 42.0 |
| 2025 Q1 | 32.2 | 31.5 | 11.6 | 9.3 | 6.5 | 1.8 | 47.9 | 48.0 |
| 2026 Q1 | 31.1 | 33.9 | 10.5 | 7.7 | 10.2 | 2.2 | 49.0 | 47.1 |
| 2026 Q3 | 29.7 | 29.2 | 12.2 | 8.7 | 10.2 | 6.1 | 48.6 | 43.5 |
| 2026 Q4 (4 polls) | 27.8 | 28.8 | 14.8 | 9.3 | 10.6 | 6.3 | 47.8 | 44.6 |

- **Two phases.** Over summer 2024–25 the movement was across the blocs: the right fell 6 points, the left rose 6, and voters went straight from National to Labour. Since then it has been within the blocs.
- **Right bloc:** since early 2025 National has lost about 4 points and NZ First has gained about 4, while the bloc total hasn't moved. A partner within the government growing as the senior partner shrinks is a protest within the coalition.
- **Left bloc:** Labour fell 5 points from early 2026. The Greens (+4) and TOP (+4) rose. TOP has no declared bloc, but the timing suggests it is drawing mainly from the left.

**The seesaw still works from one poll to the next.** Within the same pollster, a National fall still comes with a Labour rise (r=-0.47 in 2026, against -0.37 under Labour in 2020–23). What's new isn't a broken seesaw. It's a slow leak from both ends of it.

**Rejection shows up most clearly in the capability data.** On cost of living:
- National's rating as best party fell from a peak of 39% to 21%.
- Labour's rose from 23% to 36% by late 2025, then slipped to 31%.
- Only 52% now name either major party, down from 65% in February 2023 and the lowest yet.

There's also an asymmetry:
- National's capability rating tracks its vote almost exactly (r=0.92 on cost of living).
- Labour's tracks much less closely (r=0.66).
- Voters punish National for losing competence, but don't reward Labour in proportion for gaining it.

**Attraction is real too.** The Greens are back near 15% and TOP has tripled. Aggregate polls can't tell attraction and rejection apart for individual voters. That needs panel data, and the 2026 Election Study won't exist until after the election.

---

## 4. The supermarket debate (data angle only)

**Groceries now set the pulse of cost-of-living concern.** Since 2022, concern has tracked food inflation (r=0.73) more closely than headline inflation (r=0.47), and the 2025 rise in concern followed food prices while headline inflation stayed within the target band. Supermarket prices are the most visible cost of living, so every party has a reason to be seen acting on them. The party positions have converged because the politics of the issue favours whoever looks active on prices, not because ideology has converged.

For business, visibility creates the exposure. Any sector whose prices people see weekly becomes the face of cost of living, whoever governs. Petrol was that sector in 2022; groceries are now.

---

## 5. What is the media giving too much or too little oxygen to?

The useful distinction is between how many people name an issue and how many votes it moves.

- **Too little: the economy and jobs.** These are the issues whose rise lines up with this government's decline. Unemployment concern has gone from 3% (2022) to 21%, and Labour leads on it. If media coverage folds them into "cost of living and housing", it misses where the change is happening.
- **Too little: health.** It's the fastest-rising concern (+6 points in the latest wave) and second overall. On current evidence it hurts the government's rating more than it moves votes, but that could change.
- **Too much: inflation figures as good news for the government.** A falling inflation rate doesn't lower concern, because people judge prices against what they remember. Stories that frame "inflation back in the band" as a turning point for the government don't fit the data.
- **Too much: Labour against National as the main story.** The blocs have been stable. The movement is within them: National to NZ First, and Labour to the Greens and TOP. That's where the coalition arithmetic will be decided.
- **Too much: single polls.** About two-thirds of extreme polls revert toward the average in the next reading.

---

## If challenged

- **NZES numbers are from 2023 only.** They show how cost of living worked against Labour. The claim that it now works the same way against National is an inference from the ownership data, not a measurement.
- **The within-term correlations come from 10–12 Ipsos waves of trending series.** Wave-to-wave changes show no significant relationships, so these describe a direction of travel, not precise effects.
- **The food-inflation link** covers 15 waves from 2022 to 2025, and the CPI file ends at the December 2025 quarter.
- **The bloc totals** use unweighted quarterly averages of published polls. TOP is left out of both blocs.
