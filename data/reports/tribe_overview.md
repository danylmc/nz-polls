# NZES2023 Voter Tribes - Overview

Analysis of 1,989 New Zealand voters from the 2023 Election Study, clustered into 5 tribes based on 40 political attitude variables.

## Methodological Notes

The NZES oversamples Māori (34.9% of raw sample vs ~17% of NZ population). All statistics in this report use survey weights (`vwgt`) to produce population-representative estimates. Unweighted sample sizes (n) are provided to indicate statistical precision.

**Tribe boundaries are fuzzy, not sharp.** Soft clustering analysis (softmax membership probabilities) shows that the average respondent has only a 54.1% probability of belonging to their assigned tribe. 44% of respondents have ambiguous assignments (max probability below 0.5), and only 5% have highly confident assignments (above 0.8). Despite this, the five tribes explain 82% of left-right attitude variance (eta-squared = 0.821 in 2023), and this proportion is increasing over time. The tribes are best understood as useful ideal types that capture real attitudinal structure, even though individual voters shade between them.

## The Five Tribes at a Glance

| Tribe | Name | n | % | L-R | Top Party | Class Profile |
|-------|------|---|---|-----|-----------|---------------|
| 1 | Alienated Conservatives | 344 | 17% | 6.4 | National (38%) | Lower-middle class, often immigrant |
| 2 | Educated Progressives | 410 | 21% | 2.6 | Labour (39%) | Upper-middle class professionals |
| 3 | Precariat Left | 358 | 18% | 3.8 | Labour (37%) | Working class, Māori/Pacific |
| 4 | Middle New Zealand | 523 | 26% | 5.8 | National (39%) | Middle class suburbanites |
| 5 | Establishment Right | 354 | 18% | 7.4 | National (64%) | Upper class, propertied |

## Tribe Summaries

### Tribe 1: Alienated Conservatives (17%)

**The informed cynics**: Older, male, often immigrant, middle-class voters who distrust institutions but feel politically capable.

- **Demographics**: Age 52, 54% male, only 65% born in NZ (most immigrant)
- **Economics**: Lower-middle class - fewer in top income bracket (11% vs 16%), below-average home ownership
- **Politics**: Centre-right (L-R 6.4), National voters, lowest institutional trust of any tribe
- **Key trait**: Highest distrust of parliament/government/courts AND most cynical about the political system — comprehensive disillusionment, reflected in the highest non-vote rate

---

### Tribe 2: Educated Progressives (21%)

**The educated liberal class**: University-educated, high-income, partnered professionals whose progressivism is ideological rather than material.

- **Demographics**: Age 47, 56% female, 77% NZ European, 80% born in NZ
- **Economics**: Upper-middle class - highest university education (22%), highest top income bracket (22%), fewest below-average (4%)
- **Politics**: Left (L-R 2.6), Labour/Green voters, high institutional trust, highest political efficacy
- **Key trait**: The "champagne socialist" demographic - they can afford to vote on values because their material needs are met

---

### Tribe 3: Precariat Left (18%)

**The working-class left**: Young, female, Māori and Pacific, renters and social housing tenants whose politics reflect material need.

- **Demographics**: Age 42 (youngest), 57% female, 38% Māori, 10% Pacific
- **Economics**: Working class - only 11% own outright (vs 34%), 11% in social housing (vs 3%), 18% below-average income
- **Housing crisis tribe**: 46% have partner (lowest) - economic constraints limit family formation
- **Politics**: Left (L-R 3.8), Labour/Green/TPM voters, high institutional distrust, high cynicism
- **Key trait**: Support for Treaty/co-governance is material, not abstract - these policies affect their communities directly

---

### Tribe 4: Middle New Zealand (26%)

**The silent majority**: Older, female, middle-class suburbanites who trust institutions and feel the system works.

- **Demographics**: Age 53, 58% female, 73% NZ European - the median New Zealander
- **Economics**: Solidly middle class - average on almost every indicator (income, education, home ownership)
- **Politics**: Centre-right (L-R 5.8), swing voters (39% National, 23% Labour), highest institutional trust
- **Key trait**: The swing voter tribe - trusts the system and feels part of it. Votes on performance. Elections are won and lost here.
- **Boundary tribe**: Soft clustering confirms this is the most "porous" tribe, sharing significant boundary populations with all four other tribes. Its central position in attitudinal space makes it the primary conduit for voters moving between tribes.

---

### Tribe 5: Establishment Right (18%)

**The propertied class**: Older, male, wealthy, partnered Pākehā homeowners whose conservatism reflects material success.

- **Demographics**: Age 56 (oldest), 57% male, 79% NZ European, only 7% Māori
- **Economics**: Upper class - 27% in top income bracket, 55% own home outright, 81% partnered
- **Politics**: Right (L-R 7.4), National/ACT voters (64%/17%), oppose redistribution and co-governance
- **Key trait**: Self-made rather than credentialed (average education) - built wealth through business/property/trades

---

## The Class Structure of NZ Politics

This analysis reveals a clear class dimension to New Zealand political attitudes:

| Class Position | Tribes | % of Population |
|----------------|--------|-----------------|
| Working class / precarious | Precariat Left (#3) | 18% |
| Lower-middle class | Alienated Conservatives (#1) | 17% |
| Middle class | Middle New Zealand (#4) | 26% |
| Upper-middle class (professional) | Educated Progressives (#2) | 21% |
| Upper class (propertied) | Establishment Right (#5) | 18% |

**Key insight**: The two left-wing tribes (#2 and #3) represent very different class positions. Educated Progressives are wealthy professionals whose progressivism is ideological. Precariat Left are working-class voters whose left politics reflect material need. Same votes, different motivations.

Similarly, the right is split between Alienated Conservatives (lower-middle class, institutional distrust) and Establishment Right (upper class, material self-interest). Middle New Zealand sits in the middle, available to either side depending on circumstances.

**Important caveat**: Demographics cannot predict tribe membership. A Random Forest classifier using age, gender, education, Maori identity, and home ownership achieves only 29.4% accuracy (barely above the 26.1% majority-class baseline). The tribes are genuinely attitudinal groupings that cut across demographic lines -- the class correspondences above are tendencies, not determinisms.

---

## Domain Comparison

Mean z-scores by attitude domain (positive = more conservative/individualist/trusting):

| Domain | T1 | T2 | T3 | T4 | T5 |
|--------|----|----|----|----|-----|
| institutional_trust | +0.62 | -0.34 | +0.32 | -0.39 | +0.05 |
| political_efficacy | -0.62 | +0.42 | -0.64 | +0.31 | +0.32 |
| economic | -0.06 | -0.01 | -0.17 | +0.01 | +0.23 |
| social_cultural | +0.16 | -0.17 | -0.31 | -0.01 | +0.37 |
| democratic_values | -0.14 | +0.22 | +0.04 | -0.03 | -0.11 |

---

*All statistics use survey weights. See individual tribe profiles, methodology.md, and advanced_analysis_summary.md for details.*
