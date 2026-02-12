# The Five Tribes of New Zealand Politics

## A feature summary based on cluster analysis of the New Zealand Election Study, 1990-2023

---

## The Big Picture

New Zealand's electorate divides into five recognisable political tribes, identified through cluster analysis of 40 attitude variables measured in the New Zealand Election Study. These aren't demographic boxes -- a machine learning classifier using age, gender, education, ethnicity, and home ownership can only predict which tribe someone belongs to with 29% accuracy, barely above the 26% you'd get from always guessing the largest tribe. The tribes are genuinely attitudinal: they capture how people think about politics, not who they are on paper.

The five tribes, and what has happened to them over three decades, tell a story about how New Zealand politics has changed -- and how much it hasn't.

---

## The Five Tribes (2023)

### 1. Alienated Conservatives (17%)
**The informed cynics.** Older, male, often immigrant -- only 65% born in New Zealand, the lowest of any tribe. Lower-middle class, modestly resourced, and distinguished by the lowest institutional trust of any group. They distrust parliament, government, and the courts more than any other tribe, and are also among the most cynical about the political system -- they strongly agree that MPs are out of touch (raw mean 1.96/5), that a person like them has no say (1.95), that big interests run government (2.10), and that politicians don't care (1.93). Distrust and cynicism go together here: this is comprehensive disillusionment. Many have immigrant backgrounds, which may inform comparative scepticism about NZ institutions. They vote National (38%) as the least-bad option, not out of institutional faith, and their 20% non-vote rate — highest of any tribe — reflects genuine disengagement.

### 2. Educated Progressives (21%)
**The educated liberal class.** University-educated, high-income, overwhelmingly NZ-born Pakeha professionals. The "champagne socialist" tribe, and the label fits descriptively if not pejoratively: they can afford to vote on values because their material needs are met. They support redistribution, Treaty rights, and climate action not because they personally benefit, but because they believe these things are right. They trust institutions (second-most trusting tribe after the Middle New Zealand) and feel the most personally capable of influencing politics — they disagree most strongly with statements like "a person like me has no say" (raw mean 3.23/5). Trust and efficacy go together: the system works for them, they're part of it, and they want it to go further. Split between Labour (39%) and Green (32%).

### 3. Precariat Left (18%)
**The working-class left.** The youngest tribe (average age 42), most female, with 38% Maori and 10% Pacific. This is the housing crisis tribe: only 11% own their home outright (vs 34% nationally), 11% are in social housing, and 18% report below-average income. Their support for Treaty settlements and co-governance is material, not abstract -- these policies affect their communities directly. They distrust institutions (second only to the Alienated Conservatives) and are also the most cynical about the political system — they strongly agree that government is run by big interests (-0.81), that MPs lose touch (-0.65), and that they have no say (raw mean 2.01/5). Distrust and cynicism are earned here, not abstract. Same votes as the Educated Progressives, very different motivations and very different relationships with the system.

### 4. Middle New Zealand (26%)
**The silent majority -- and the tribe elections are won and lost in.** Average on almost every demographic indicator: middle-aged, middle-income, middle-class, suburban homeowners. Their defining feature is institutional trust -- they trust parliament, government, and courts more than any other tribe. And they also feel part of the system, disagreeing with statements like "politics is too complicated" and "a person like me has no say" (raw mean 3.10/5). Trust and efficacy go together: they believe the system works and feel included in it. They voted 39% National and 23% Labour in 2023, but these are performance voters who swing on vibes and cost of living, not ideology. Soft clustering confirms they're the most "porous" tribe, sharing boundary voters with all four others. They are the contested centre of New Zealand politics.

### 5. Establishment Right (18%)
**The propertied class.** The oldest (56), most male (57%), most Pakeha (79%), and wealthiest tribe. Over a quarter earn top-bracket income; 55% own their home outright. Notably, they're not the most educated -- university rates are slightly below average. These are people who built wealth through business, property, and trades rather than credentials. Their conservatism is straightforwardly material: they oppose redistribution because they'd be redistributed from, and oppose co-governance because the current system has served them well. The ideological core of the centre-right: 64% National, 17% ACT.

---

## The Key Findings

### 1. What unites us is more interesting than what divides us

All five tribes agree on several things. They all distrust social media. They all think politicians are out of touch. They all believe courts should check government power. They all think people are generally self-interested. These democratic values and institutional cynicism items have near-zero between-tribe variance (eta-squared below 0.02), meaning tribe membership explains virtually none of the disagreement.

The consensus items share a theme: they're judgements about the political *system* rather than specific *policies*. New Zealanders across the spectrum share a common, mildly cynical view of how democracy works -- they just disagree about what to do about it.

### 2. What divides us most is money

Income redistribution is the dominant dividing line in New Zealand politics -- by a large margin. The two redistribution questions ("Government should reduce income differences" and "Income differences are too large") have eta-squared values around 0.69, meaning tribe membership explains roughly 70% of the variation in responses. This is about 70 times more divisive than social media trust, and nearly three times more divisive than Treaty/co-governance attitudes.

The top two questions alone explain more between-tribe variance than the bottom 30 combined. If you could ask New Zealanders only one question to sort them politically, it would be about whether the government should reduce income inequality.

### 3. Treaty politics is the primary axis of political contestation

This sounds contradictory to the finding above, but it isn't. When you look at the full attitude space via principal component analysis, the first dimension (explaining 20.8% of total attitude variance) is dominated by Treaty and co-governance items, not economic redistribution. The second dimension (11.5%) captures institutional trust versus scepticism.

The resolution: redistribution attitudes are the sharpest dividing line *between the five tribes*, but Treaty/co-governance attitudes are the broadest organising dimension across the entire attitude space. Treaty positions connect to more other attitudes than any other single topic -- they're correlated with views on redistribution, climate, unions, gender equity, and more. Someone's position on co-governance tells you more about their overall political worldview than almost any other single question.

### 4. The "left but anti-Maori" voter has disappeared

The most striking change between 1990 and 2023 is the disappearance of a political type that was once a quarter of the electorate. In 1990, the largest cluster was the "Centre-Left Working Class" (25%): economically progressive, pro-union, pro-redistribution -- but opposed to Maori political voice and sceptical of the Treaty. These were classic blue-collar Labour voters whose economic leftism didn't extend to identity politics.

By 2023, this combination has largely vanished. Economic progressivism and support for Treaty/co-governance now load on the same principal component -- they've become part of the same political package. The 1990 voters who were economically left but culturally conservative on Maori issues appear to have been absorbed into either the Middle New Zealand (if they drifted right) or the Precariat Left (if they came to accept Maori political aspirations). The pro-Treaty constituency grew from one small cluster (Left Progressives, 14% in 1990) to two large ones (Educated Progressives 21% + Precariat Left 18% in 2023).

### 5. The biggest gap in New Zealand politics is between older men and younger women

Single demographic variables understate the real divides. The widest consistent attitudinal gap isn't between left and right voters, or between university graduates and non-graduates, or between Maori and Pakeha. It's between men over 40 and women under 40: a 1.27-point gap on the 0-10 left-right scale in 2023, and it's widening. Older men (mean 5.88) sit almost 1.3 points to the right of younger women (4.61).

This compounds two separate effects -- age and gender -- that reinforce each other. And while single demographic gaps fluctuate (the gender gap spiked during COVID-19 then moderated), the age-gender interaction has widened steadily: 1.19 points in 2017, 1.19 in 2020, 1.27 in 2023.

### 6. Young people aren't becoming conservative as they age

The common assumption that people drift rightward as they get older isn't supported by the data. Tracking birth cohorts across 2017-2023, the 1980s cohort shifted right by just 0.06 points over six years; the 1990s+ cohort by just 0.05. Meanwhile, a 1.78-point generational gradient separates the most right-leaning cohort (1940s, mean 6.33) from the most left-leaning (1990s+, mean 4.55).

If cohort effects persist -- if today's young voters remain roughly where they are -- generational replacement will gradually shift the electorate leftward as the 1940s-1960s cohorts are replaced by 1980s-1990s+ cohorts. This is not inevitable (period effects like a major economic crisis could shift everyone rightward), but the current trajectory is clear.

### 7. The housing divide is real and growing

Home owners are consistently more right-leaning than renters, and the gap is compounded by age. Four distinct political groups emerge when you cross age with housing tenure:

- **Older owners**: Furthest right
- **Older renters**: Centre-right
- **Younger owners**: Centre
- **Younger renters**: Furthest left

As home ownership rates decline among young New Zealanders, the renter population grows and becomes more politically distinctive. Housing tenure is becoming a political cleavage in its own right, not just a proxy for wealth or age.

### 8. The Maori-non-Maori gap is widening fast

The left-right gap between Maori and non-Maori voters more than doubled between 2017 and 2023: from 0.34 points to 0.80 points. This happened because Maori moved substantially leftward (from 5.16 to 4.65 on the 0-10 scale) while non-Maori attitudes stayed put.

Simultaneously, Labour's Maori vote share collapsed from 57% to 28%, with the emergence of Te Pati Maori (10% among Maori) fracturing the Maori centre-left vote. Maori voters are moving left in attitudes but away from Labour as a party -- a structural realignment, not a rightward shift.

Within the Maori electorate itself, there's a 60/40 split: 60% are Progressive Maori (left-leaning, Labour/TPM/Green voters) and 40% are Moderate Maori (centrist, with 32% not voting and 26% voting National). The Moderate Maori tribe is older, more male, and far less engaged with electoral politics.

### 9. Education polarisation is a gradient, not a cliff

University graduates are 0.7-1.0 points more left-leaning than non-graduates, and the gap is widening. But the finer-grained picture (available from 2017) reveals a smooth gradient rather than a sharp threshold at university level. Each step -- no qualification to trade/diploma to bachelor to postgraduate -- shifts attitudes leftward by a roughly equal amount. Postgraduates are meaningfully more left than bachelor-degree holders on both left-right self-placement and redistribution.

University education also partly equalises gendered political attitudes: the gender gap is larger among non-graduates. Higher education may dampen gendered political socialisation.

### 10. The tribal structure is getting sharper, not blurrier

Despite the fuzzy boundaries between tribes (44% of respondents have ambiguous assignments, sitting between two or more tribes), the five-tribe structure is becoming more distinct over time. Eta-squared -- the proportion of left-right variance explained by tribe membership -- rose from 80.1% in 2017 to 81.5% in 2020 to 82.1% in 2023. The tribes are pulling further apart on the left-right dimension.

The broad architecture is also recognisable across 33 years. A right-wing economic elite, a conservative social base, a centrist swing group, and a progressive intelligentsia are all visible in both 1990 and 2023. What changed is who belongs to each group and what binds them together -- especially the fusion of economic and Treaty/identity dimensions into a single left-right package.

---

## The Dog That Didn't Bark: Immigration

### Immigration is not a polarising issue in New Zealand politics

Given the prominence of immigration in recent political debate, one of the most surprising findings is how little it divides the electorate along conventional political lines.

The NZES asks a consistent question across 2017-2023: "Should the number of immigrants allowed into NZ be increased or reduced?" (1 = increased a lot, 5 = reduced a lot). Immigration attitudes were not among the 40 clustering variables -- and it turns out there's a good reason: they barely correlate with the attitude dimensions that define the five tribes.

**Immigration doesn't map onto left vs right.** The correlation between immigration attitudes and left-right self-placement is just r = 0.12 -- very weak. For comparison, redistribution correlates r = 0.45 with left-right. And immigration attitudes correlate essentially zero (r = 0.01) with redistribution attitudes. Knowing whether someone wants to reduce inequality tells you almost nothing about whether they want to reduce immigration.

The left-right breakdown shows why:

| L-R position | Mean immigration attitude | Interpretation |
|-------------|--------------------------|----------------|
| Left (0-3) | 2.85 | Lean toward increase |
| Centre (4-6) | 3.20 | About the same |
| Right (7-10) | 3.18 | About the same |

Left-wing voters are mildly more pro-immigration, but centre and right voters are essentially indistinguishable on the question. This is fundamentally different from the UK or Europe, where immigration is a major structuring dimension of politics.

**By party**, only NZ First voters stand out. In 2017, they averaged 4.32 (firmly "reduce") -- far above any other party's voters. In 2023, NZ First voters remained the most anti-immigration at 3.61. Everyone else clusters together: Green voters are the most pro-immigration (2.65 in 2023), but Labour (3.22), National (3.15), and ACT (3.23) are nearly identical.

**New Zealand has become sharply more pro-immigration over time:**

| Year | Mean | Shift |
|------|------|-------|
| 2017 | 3.61 | -- |
| 2020 | 3.66 | +0.05 (stable) |
| 2023 | 3.14 | -0.52 (large pro-immigration shift) |

In 2017 and 2020, the average New Zealander leaned toward reducing immigration. By 2023, the average had moved to roughly neutral. The share wanting immigration "increased a lot" or "a little" rose from 13% (2017) to 29% (2023). This shift may reflect the post-COVID labour shortages and border closures that made the costs of *less* immigration tangible.

**The demographic cross-cuts are interesting:**
- **Age**: Under 30 = 2.81 (pro-increase) vs 60+ = 3.29 (lean reduce) -- age matters more than left-right position
- **Maori vs non-Maori**: Maori are slightly *more* anti-immigration (3.31) than non-Maori (3.05) -- the opposite of what you might expect given Maori voters' overall leftward lean
- **Gender gap**: Tiny (male 3.18, female 3.11)

**Correlations with other attitude items are uniformly weak.** The strongest is with support for stiffer sentences (r = -0.18) -- those who want to reduce immigration also tend toward punitive justice. But even this is modest. Immigration attitudes don't load onto the same dimensions that define the five tribes. They cross-cut the tribal structure rather than reinforcing it.

**Why this matters:** In the UK, Brexit was fundamentally an immigration question that realigned politics along a new axis. In New Zealand, no comparable realignment has occurred. Immigration is a valence issue (most people have mild preferences) rather than a position issue (where it sharply divides). The one exception is NZ First, whose anti-immigration stance is distinctive -- but this maps more onto a populist-nationalist dimension than onto the economic left-right axis that structures the five tribes. If immigration ever does become a sharply polarising issue in New Zealand, it would likely create a new cross-cutting dimension rather than reinforcing the existing tribal structure.

---

## Contradictions and Tensions in the Findings

Several findings sit in tension with each other, though none are outright contradictions:

1. **Tribes explain 82% of left-right variance, but 44% of people sit on tribe boundaries.** These are compatible: the tribe *centroids* are pulling apart (high eta-squared), but individual voters are spread along gradients rather than clustered in tight groups. The tribes are useful ideal types, not natural kinds. Think of them like climate zones rather than national borders.

2. **Treaty politics is the "primary axis" (PC1), but redistribution is the "most divisive" (highest eta-squared).** These measure different things. PC1 captures the dimension that organises the most total variation across all 40 attitude variables. Eta-squared measures how sharply the five tribes diverge on each individual question. Treaty attitudes organise the whole space; redistribution attitudes most sharply separate the tribes that exist within that space.

3. **Bootstrap stability is moderate (ARI = 0.67), while 44% of respondents sit on tribe boundaries.** These are compatible: the five-tribe solution mostly replicates across resamples (ARI of 0.67 is in the "moderate" range), but many individual respondents sit near boundaries and can shuffle between adjacent tribes. Educated Progressives are the most stable tribe (mean respondent stability 0.92), while the Middle New Zealand are the least stable (0.74), consistent with their centrist, swing-voter positioning. The tribes are reliable groupings, even though the dividing lines between them are gradients rather than walls.

4. **Scale direction matters: trust and efficacy items are easily misread.** The trust items (A11) are scored 1=Trust a lot to 4=Don't trust at all, so positive z-scores mean *more distrust*. The efficacy items (G12) are cynical statements ("no say", "too complicated", "MPs out of touch") where 1=Strongly agree and 5=Strongly disagree. Positive z-scores mean more *disagreement* with cynical statements (= higher efficacy); negative z-scores mean more *agreement* with cynicism (= lower efficacy). Trust and efficacy go together: the Alienated Conservatives (Tribe 1) have the highest distrust (+0.62) *and* highest cynicism (-0.62, agreeing with cynical statements). The Middle New Zealand (Tribe 4) have the highest trust (-0.39) *and* lowest cynicism (+0.31, disagreeing with cynical statements). An earlier version of these reports had these interpretations inverted.

5. **The 1990 comparison is suggestive but not formally aligned.** Different clustering variables (52 vs 40), different imputation methods (median vs MICE), different sample compositions, and no survey weights for 1990. The parallels between 1990 and 2023 tribes are interpretive, not statistical. The reports acknowledge this but readers should treat the 1990-2023 comparisons as informed speculation rather than measured change.

6. **Maori Tribe 1 ("Moderate Maori") shows positive z-scores on co-governance items (C9b +0.81, C12i +0.81), which should indicate opposition to co-governance.** But the tribe is described as "Moderate" rather than "conservative" or "anti-co-governance." The z-scores are relative to the Maori subsample (not the full population), and even the "moderate" Maori tribe is substantially more pro-co-governance than the overall population mean. The naming is reasonable but the raw z-scores within the Maori subsample could mislead if read without this context.

---

## The Story in One Paragraph

New Zealand's electorate splits into five political tribes defined primarily by attitudes to income redistribution and, increasingly, Treaty/co-governance issues -- two dimensions that were once separate but have fused into a single political package over 33 years. The largest tribe (26%) is the Middle New Zealand: non-ideological swing voters who trust institutions and feel the system works, and who decide elections. The biggest gap in our politics is between older men and younger women (1.27 points on a 10-point scale, widening). Young people aren't drifting right as they age. Home ownership is becoming a political identity. Maori voters are diverging rapidly from non-Maori. And underneath all the disagreement, all five tribes share a common, mildly cynical view of democracy: they agree politicians are out of touch, social media is untrustworthy, and courts should check government power. They just disagree, profoundly, about whether the government should be doing more to reduce inequality.

---

*Based on analysis of the New Zealand Election Study (1990, 2017, 2020, 2023). Total sample across all years: ~11,276 respondents. Full methodology, individual tribe profiles, and 27 analysis notebooks available in the project repository.*
