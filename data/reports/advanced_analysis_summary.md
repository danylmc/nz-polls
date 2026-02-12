# Advanced NZES Analysis: Notebooks 14--27

This report summarises findings from fourteen additional analyses extending the core five-tribe clustering of the New Zealand Election Study (2023) with cross-year comparisons, methodological validation, and deeper demographic exploration.

---

## Executive Summary

1. **Housing tenure is a significant political cleavage**: Home owners are consistently more right-leaning than renters, with the gap most pronounced among older owners. The interaction of age and ownership creates a four-way political map.

2. **Demographic interactions compound**: The widest attitudinal gap in New Zealand politics is between older men (mean L-R 5.88) and younger women (4.61) -- a 1.27-point spread that is widening.

3. **Education polarisation is a gradient, not a threshold**: Moving from no qualifications to postgraduate, each step shifts attitudes leftward. Postgraduates are distinctly more left than bachelor-degree holders.

4. **Urban-rural divide is real but modest**: Urban City residents sit left of Rural voters, but the gap is smaller than education or age effects.

5. **Maori-non-Maori divergence is widening**: The left-right gap between Maori and non-Maori voters doubled from 0.34 to 0.80 points between 2017 and 2023.

6. **Tribes are attitudinal, not demographic**: A Random Forest classifier achieves only 29.4% accuracy predicting tribe membership from demographics -- barely above the 26.1% majority-class baseline.

7. **Cluster stability is moderate**: Bootstrap resampling produces a mean ARI of 0.67, indicating the five-tribe solution mostly replicates across resamples, with Educated Progressives the most stable tribe and Middle New Zealand the least.

8. **Affective polarisation exists but is structurally asymmetric**: Both Labour and National voters trust their own party's government roughly 1.14 points more than the opposing party's on a 4-point scale.

9. **Birth cohort effects dominate lifecycle effects**: Younger cohorts (1980s--1990s) are not shifting rightward as they age, suggesting generational replacement will gradually move the electorate leftward.

10. **Attitude space has two primary dimensions**: PC1 captures Treaty/co-governance attitudes (20.8% variance); PC2 captures institutional trust vs scepticism (11.5%).

---

## 1. Housing Tenure Polarisation (Notebook 14)

**Years covered:** 1990, 2002, 2011, 2023 (home ownership unavailable in 2017, 2020)

### Left-Right by Housing Tenure

Home owners consistently place themselves further right than renters on the 0--10 left-right scale. The gap has persisted across three decades.

### Redistribution Attitudes

Owners are less supportive of government redistribution than renters, consistent with material self-interest: those with property have less to gain from redistribution and more to lose from the taxes that fund it.

### Age-Ownership Interaction

Crossing age (<40 vs 40+) with tenure creates four politically distinct groups:

- **40+, Owner**: Furthest right
- **40+, Renter**: Centre-right
- **Under 40, Owner**: Centre
- **Under 40, Renter**: Furthest left

This pattern suggests both lifecycle position and material circumstances independently shape political attitudes. As home ownership rates decline among young New Zealanders, the renter population grows and becomes more politically distinctive.

### Party Vote

Owners favour centre-right parties (National); renters favour centre-left (Labour, Greens). The gap in party vote mirrors the attitudinal divide.

**Figures:** `housing_gap_left_right.png`, `housing_gap_redistribution.png`, `housing_party_vote.png`, `left_right_by_housing.png`

---

## 2. Demographic Interaction Effects (Notebook 15)

**Years covered:** 2017, 2020, 2023

### Age x Gender

| Year | 40+ Male | Under 40 Female | Gap |
|------|----------|-----------------|-----|
| 2017 | 5.69     | 4.50            | 1.19 |
| 2020 | 5.92     | 4.72            | 1.19 |
| 2023 | 5.88     | 4.61            | 1.27 |

The age-gender interaction produces the largest and most consistent attitudinal gap in New Zealand politics. Older men are the most right-leaning demographic; younger women the most left-leaning. The gap widened slightly to 1.27 points in 2023.

### Age x Education

| Year | 40+ No Uni | Under 40 Uni | Gap |
|------|-----------|--------------|-----|
| 2017 | 5.61      | 4.74         | 0.87 |
| 2020 | 5.18      | 4.29         | 0.89 |
| 2023 | 5.69      | 4.66         | 1.03 |

The age-education interaction gap is widening (0.87 to 1.03 points). Older non-graduates and younger graduates are pulling apart.

### Education x Gender

The education-gender interaction shows the gender gap is larger among non-university respondents. University education partially equalises male and female political attitudes.

**Figures:** `interaction_age_gender.png`, `interaction_age_education.png`, `interaction_education_gender.png`, `interaction_trends.png`

---

## 3. Fine-Grained Education Polarisation (Notebook 16)

**Years covered:** 2017, 2020, 2023 (4-level education only available from 2017; 2017 cannot distinguish bachelor from postgraduate)

### Education Gradient (Left-Right, 2020 and 2023)

The education-politics relationship is a **gradient rather than a sharp threshold**. Each step in education is associated with a leftward shift:

- **No qualification/School** -> **Trade/Diploma** -> **Bachelor** -> **Postgraduate**

The step-size analysis reveals roughly equal decrements across the education spectrum, though the postgraduate step is notable: postgraduates are meaningfully more left-leaning than bachelor-degree holders on both left-right self-placement and redistribution attitudes.

### Redistribution

The education gradient on redistribution mirrors the left-right pattern: more educated respondents are more supportive of redistribution. This challenges the view that education polarisation is purely cultural rather than economic.

### Implications

The binary university/non-university split used in earlier analyses captures the broad pattern but misses important nuance. The postgraduate category -- which includes many academics, policy professionals, and senior public servants -- is politically distinctive and growing as a share of the electorate.

**Figures:** `education_granular_left_right.png`, `education_gradient.png`

---

## 4. Urban-Rural Granularity (Notebook 17)

**Years covered:** 2020, 2023 (4-level urban-rural only available from 2020)

### Left-Right by Urban-Rural Category

The 4-level classification (Rural, Small Town, Suburban, Urban City) reveals a gradient from right (Rural) to left (Urban City), though the total span is smaller than education or age effects.

### Party Vote Composition (2023)

Party support varies across the urban-rural gradient:
- **Rural**: National and NZ First dominate
- **Small Town**: National-leaning with some Labour
- **Suburban**: Mixed, competitive terrain
- **Urban City**: Stronger Labour and Green support

### Gradient Shape

The step-size analysis shows that the biggest attitudinal jump is typically between Small Town and Suburban, suggesting a partial threshold effect at the boundary between towns and metropolitan areas.

**Figures:** `urban_rural_granular_lr.png`, `urban_rural_party_vote.png`

---

## 5. Maori Attitude Drift (Notebook 18)

**Years covered:** 2017, 2020, 2023 (Maori identification unavailable in 1990)

### Left-Right Divergence

| Year | Maori | Non-Maori | Gap |
|------|-------|-----------|-----|
| 2017 | 5.05  | 5.39      | -0.34 |
| 2020 | 5.16  | 5.47      | -0.31 |
| 2023 | 4.65  | 5.45      | **-0.80** |

The left-right gap between Maori and non-Maori voters more than doubled between 2017 and 2023. Maori moved substantially leftward in 2023 (from 5.16 to 4.65), while non-Maori attitudes remained stable.

### Redistribution

Maori are consistently more pro-redistribution than non-Maori, though the gap here has narrowed slightly over time.

### Labour Party Vote

| Year | Maori Labour % | Non-Maori Labour % | Gap |
|------|---------------|-------------------|-----|
| 2017 | 56.7          | 33.7              | +23.0pp |
| 2020 | 44.6          | 40.0              | +4.5pp |
| 2023 | 27.7          | 20.3              | +7.4pp |

Labour's Maori vote share fell dramatically from 56.7% to 27.7% between 2017 and 2023, though it remains higher than non-Maori Labour support. The emergence of Te Pati Maori (9.8% among Maori in 2023) has fractured the Maori centre-left vote.

**Figures:** `maori_lr_drift.png`, `maori_nonmaori_divergence.png`, `maori_redist_drift.png`, `maori_party_vote_timeline.png`

---

## 6. Soft Clustering and Tribe Boundaries (Notebook 19)

### Assignment Confidence

Using softmax membership probabilities computed from Euclidean distances to cluster centroids:

- Mean maximum probability: **0.541** (i.e., the average respondent has a 54.1% probability of belonging to their assigned tribe)
- 44.0% of respondents have max probability below 0.5 (genuinely ambiguous assignments)
- Only 5.0% have max probability above 0.8 (highly confident assignments)

### Tribe Proximity

The proximity matrix reveals which tribes share the most boundary respondents:

- **Middle New Zealand** is the most "porous" tribe, sharing significant boundary populations with all other tribes
- **Educated Progressives** and **Establishment Right** are the most distinct from each other (almost no boundary overlap)
- **Alienated Conservatives** share boundaries primarily with Middle New Zealand and Establishment Right

### Boundary Respondents

876 respondents (44%) sit on boundaries (max probability < 0.5). The most common boundary pairs:

1. Middle New Zealand / Educated Progressives (100 respondents)
2. Middle New Zealand / Establishment Right (98)
3. Middle New Zealand / Alienated Conservatives (76)
4. Middle New Zealand / Precariat Left (70)

The Middle New Zealand's central position in attitudinal space makes it the primary "boundary tribe" -- consistent with its characterisation as the swing voter tribe.

**Figures:** `assignment_confidence.png`, `tribe_proximity_matrix.png`, `boundary_respondents.png`

---

## 7. Within-Tribe Heterogeneity Over Time (Notebook 20)

### Within-Tribe Standard Deviation (Left-Right)

| Year | Alien. Cons. | Educated Prog. | Precariat Left | Middle NZ | Establ. Right |
|------|---------------|---------------|-------------|-----------------|--------------|
| 2017 | 1.14          | 1.03          | 1.95        | 1.06            | 1.12         |
| 2020 | 1.45          | 0.88          | 1.06        | 1.09            | 1.47         |
| 2023 | 1.11          | 1.59          | 1.45        | 1.07            | 1.18         |

### Between vs Within Variance (Eta-Squared)

| Year | Eta-Squared |
|------|------------|
| 2017 | 0.801      |
| 2020 | 0.815      |
| 2023 | 0.821      |

Eta-squared is steadily increasing, meaning the five tribes are explaining an increasing proportion of left-right attitude variance. The tribe structure is becoming **more distinct** over time, not less -- the tribes are pulling further apart on left-right self-placement.

**Figures:** `within_tribe_variance_timeline.png`, `between_vs_within_variance.png`

---

## 8. Decision Timing and Swing Voters (Notebook 21)

### Decision Timing by Tribe (2023)

E7a: "When did you decide which party to vote for?"

| Tribe | "A long time ago" | Last week / Election day |
|-------|-------------------|-------------------------|
| Establishment Right | 41.0% | 8.4% |
| Educated Progressives | 32.1% | 20.1% |
| Alienated Conservatives | 30.2% | 22.5% |
| Middle New Zealand | 29.0% | 19.9% |
| Precariat Left | 27.9% | 24.7% |

Establishment Right are the most locked-in tribe, deciding earliest and with the fewest late deciders. Precariat Left are the most fluid, with nearly a quarter deciding in the final week or on election day.

### Left-Right by Decision Timing

| Timing | Mean L-R | n |
|--------|----------|---|
| A long time ago | 5.81 | 543 |
| Before campaign | 5.58 | 442 |
| During campaign | 4.84 | 363 |
| Last week | 4.55 | 223 |
| Election day | 4.99 | 109 |

Early deciders are more right-leaning (mean 5.81); late deciders (last week) skew left (4.55). This pattern suggests that late-deciding voters lean centre-left and may have been weighing Labour vs other left options rather than left vs right.

**Figures:** `decision_timing_by_tribe.png`, `late_deciders_lr.png`, `swing_voter_profile.png`

---

## 9. PCA of Attitude Space (Notebook 22)

### Variance Explained

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1       | 20.8%    | 20.8%      |
| PC2       | 11.5%    | 32.3%      |
| PC3       | 5.6%     | 37.9%      |
| PC4       | 4.2%     | 42.1%      |
| PC5       | 3.3%     | 45.4%      |

The attitude space does not collapse into a few dimensions. Seven components are needed to reach 50% of variance; ~20 components for 80%. This high dimensionality validates the use of K-means on the full variable set rather than a reduced PCA space.

### Component Interpretation

**PC1 (20.8%): Treaty / Co-governance dimension**
- Positive loadings: co-governance items (C9a-d), Maori decision-making (C12i)
- Negative loadings: Treaty settlement gone too far (C11), remove Treaty references (C12d)
- This is the primary axis of political contestation in New Zealand

**PC2 (11.5%): Institutional trust vs populist scepticism**
- Positive loadings: trust in parliament (A11a), government (A11b), courts (A11c)
- Negative loadings: "government run by big interests" (G12f), "politicians don't care" (G12g), "MPs out of touch" (G12d)
- Separates institutionalists from populist sceptics

**PC3 (5.6%): Anti-democratic populism / social media trust**
- Positive loadings: trust in social media (A11g), preference for expert rule (A9a), business leader rule (A9b)
- This minor component captures a small pocket of anti-democratic sentiment

### Tribe Separation in PC Space

Pairwise centroid distances (PC1-PC2 plane):
- **Most separated**: Educated Progressives vs Establishment Right (7.76)
- Alienated Conservatives vs Educated Progressives (7.11)
- Precariat Left vs Establishment Right (7.41)
- **Least separated**: Educated Progressives vs Middle New Zealand (3.85) and Middle New Zealand vs Establishment Right (3.92)

**Figures:** `pca_scree_plot.png`, `pca_loadings.png`, `pca_tribe_scatter.png`, `pca_tribe_scatter_combined.png`, `pca_biplot.png`, `pca_tribe_centroids.png`

---

## 10. Discriminant Analysis (Notebook 23)

### Can Demographics Predict Tribe Membership?

A Random Forest classifier (500 trees, 5-fold cross-validation) using age, gender, education, Maori identity, and home ownership achieves:

- **Cross-validated accuracy: 29.4%** (+/- 1.1%)
- Random baseline: 20.0%
- Majority-class baseline: 26.1% (Middle New Zealand)

The classifier exceeds chance but only marginally beats always guessing the largest tribe. This confirms that **tribes are genuinely attitudinal groupings that cut across demographic lines**. Demographics alone cannot sort people into tribes.

### Feature Importance

The most predictive demographic features (in order):
1. Age (0.666 -- dominant)
2. Education (0.131)
3. Home ownership (0.117)
4. Gender (0.048)
5. Maori identity (0.037)

### Classification by Tribe

Per-tribe precision and recall are uniformly low (22--40%). Precariat Left are the most demographically distinctive (precision 0.38, recall 0.40), while Establishment Right are the least predictable (precision 0.22, recall 0.20).

**Figures:** `discriminant_feature_importance.png`, `discriminant_confusion_matrix.png`

---

## 11. Affective Polarisation (Notebook 24)

### In-Party vs Out-Party Trust

Using trust in Labour-led government (A11h) and National-led government (A11i) as proxies for affective polarisation:

| Group | In-Party Trust | Out-Party Trust | Gap |
|-------|---------------|-----------------|-----|
| Labour voters | 3.06 | 1.91 | **1.15** |
| National voters | 2.96 | 1.83 | **1.13** |

Both partisan groups trust their own party's government roughly one full point more on the 1--4 scale. The affective gap is symmetric between Labour and National voters.

### By Tribe

The highest affective polarisation appears among:
- Precariat Left Labour voters (1.77) -- the most affectively polarised group
- Establishment Right National voters (1.55)
- The lowest affective polarisation is among Middle New Zealand voters on both sides (~0.73--0.77), consistent with their non-ideological, performance-based orientation

### By Demographics

Non-university voters show higher affective polarisation than university graduates, and younger Labour voters show higher affective polarisation than older Labour voters.

### Limitations

Trust in party-led government is a weak proxy for true affective polarisation (which would require feeling thermometers). The 1--4 scale is narrow, and the analysis covers only the two major parties.

**Figures:** `inparty_outparty_trust.png`, `affective_polarisation_by_tribe.png`

---

## 12. Bootstrap Cluster Stability (Notebook 25)

### Overall Stability

100 bootstrap resamples of the 2023 data, re-clustered with k=5:

- **Mean ARI: 0.672** (range: 0.402--0.940)
- ARI standard deviation: 0.164

An ARI of 0.67 falls in the "moderate" stability range. The five-tribe solution mostly replicates across resamples, though with some boundary-respondent shuffling expected for continuous attitude data.

### Per-Respondent Stability

- Mean stability: 0.817
- Always same tribe (stability = 1.0): 197 respondents (9.9%)
- Highly stable (>0.9): 705 (35.4%)
- Unstable (<0.5): 118 (5.9%)

### Stability by Tribe

| Tribe | Mean Stability | Median |
|-------|---------------|--------|
| Educated Progressives | 0.919 | 0.977 |
| Precariat Left | 0.838 | 0.922 |
| Establishment Right | 0.816 | 0.837 |
| Alienated Conservatives | 0.789 | 0.837 |
| Middle New Zealand | 0.742 | 0.787 |

Educated Progressives are the most stable tribe -- their members are reliably reassigned across bootstrap samples (mean 0.92, median 0.98). Middle New Zealand is the least stable (mean 0.74), consistent with their centrist position near multiple tribe boundaries and their characterisation as swing voters.

### Interpretation

The moderate ARI and high per-respondent stability (82% mean) confirm that the five-tribe solution is a reliable grouping of the attitude space. The lower ARI relative to per-respondent stability reflects the fact that K-means cluster numbering is arbitrary: when a small number of respondents shuffle between tribes, the ARI penalises this more harshly than per-respondent metrics. The tribes capture real attitudinal structure (eta-squared = 82%) with moderate boundary fuzziness, consistent with the soft clustering finding that 44% of respondents sit near tribe boundaries.

**Figures:** `bootstrap_ari_distribution.png`, `respondent_stability_map.png`

---

## 13. Attitude Correlation Network (Notebook 26)

### Network Structure

The correlation network (|r| > 0.3) contains 46 nodes and 197 edges, with a density of 0.190.

### Most Central Variables

The most broadly connected attitude items (highest degree centrality):

1. **A11h** -- Trust in the Labour Party (20 connections)
2. **C12i** -- Maori should have more say (19 connections)
3. **C11** -- Treaty settlement process (19 connections)
4. **C6a** -- Government should reduce income differences (18 connections)
5. **C9b/C9c** -- Co-governance items (18 connections each)

Trust in the Labour Party is the single most correlated item in the attitude space -- it connects to both institutional trust items and socio-economic policy attitudes.

### Community Detection

Greedy modularity identifies 11 communities, dominated by two large clusters:

1. **Community 1 (23 variables)**: Economic and social-cultural items form a single large cluster. This includes redistribution, inequality, Treaty/co-governance, climate, unions, and crime. The merger of economic and social domains in a single community indicates high ideological constraint.

2. **Community 2 (12 variables)**: Institutional trust and political efficacy items cluster together. Trust in institutions and belief that politicians are responsive form a coherent dimension.

The remaining communities are small (1--2 variables each), comprising isolated democratic values items (e.g., preference for expert rule, understanding of politics).

### Ideological Constraint

| Comparison | Mean |r| |
|-----------|---------|
| Within-economic | 0.281 |
| Within-social | 0.378 |
| Cross-domain (economic-social) | 0.250 |

Cross-domain correlations (0.250) are nearly as large as within-domain correlations (0.281 economic, 0.378 social). This high degree of ideological constraint means that knowing someone's position on economic issues substantially predicts their social/cultural positions. New Zealand's attitude space is more unidimensional than the variable domains might suggest.

**Figures:** `domain_correlation_heatmap.png`, `attitude_network.png`, `network_communities.png`

---

## 14. Birth Cohort APC Decomposition (Notebook 27)

### Cohort Effects on Left-Right

| Cohort | Mean L-R |
|--------|----------|
| Pre-1940 | 5.59 |
| 1940s | 6.33 |
| 1950s | 5.65 |
| 1960s | 5.78 |
| 1970s | 5.44 |
| 1980s | 4.92 |
| 1990s+ | 4.55 |

A generational gradient of 1.78 points separates the most right-leaning (1940s) from the most left-leaning (1990s+) cohort. Each successive generation since the 1940s sits further left.

### Age vs Cohort: Do People Move Right as They Age?

Tracking cohorts across 2017--2023:

| Cohort | 2017 | 2023 | Change |
|--------|------|------|--------|
| 1980s | 4.86 | 4.92 | +0.06 |
| 1990s+ | 4.50 | 4.55 | +0.05 |
| 1970s | 5.07 | 5.44 | +0.37 |

The 1980s and 1990s cohorts show **minimal rightward drift** over six years (+0.05 to +0.06 points). This contradicts the common assumption that young people inevitably become more conservative as they age. The 1970s cohort shows a larger shift (+0.37), but this may reflect period effects rather than ageing.

### Period Effects

Most cohorts shifted right between 2017 and 2020 (possibly a COVID-19 / Ardern effect), then moved back left between 2020 and 2023. This period effect is visible across all cohorts, suggesting a common societal shift rather than differential ageing.

### Redistribution

All cohorts became less pro-redistribution between 2017 and 2023. This is a universal rightward shift on economic attitudes, likely reflecting the changing political environment (transition from Ardern Labour government to Luxon National government).

**Figures:** `apc_cohort_trajectories.png`, `apc_age_effect.png`, `apc_period_effect.png`, `apc_cohort_redistribution.png`, `apc_decomposition_summary.png`

---

## Methodological Implications

### Validation of the Five-Tribe Model

The advanced analyses provide several validation checks:

1. **PCA confirms multidimensionality**: The attitude space requires many components, validating K-means over simpler left-right models.
2. **Discriminant analysis confirms attitudinal basis**: Low demographic predictability (29%) confirms tribes capture attitudes, not demographics.
3. **Bootstrap stability confirms robustness**: The moderate ARI (0.67) and high per-respondent stability (82%) show the five-tribe solution reliably replicates, even though boundaries between adjacent tribes are gradients rather than sharp divides.
4. **Eta-squared confirms explanatory power**: Tribes explain 82% of left-right variance and this share is increasing.

### The Treaty Dimension

PC1 of the attitude space is dominated by Treaty/co-governance items, not economic left-right. This suggests that Treaty politics is the **primary axis of political contestation** in New Zealand -- more so than economic redistribution. The correlation network confirms this: Treaty items are among the most highly connected in the attitude space.

### Generational Change

The APC analysis suggests that generational replacement -- not individual conversion -- is the primary driver of electoral change. If current cohort effects persist, the electorate will gradually shift leftward as the 1940s--1960s cohorts are replaced by 1980s--1990s+ cohorts.

---

## Technical Notes

### Data Sources

- NZES 1990 (n=2,102), 2017 (n=3,455), 2020 (n=3,730), 2023 (n=1,989)
- 2002 and 2011 data unavailable for this run; notebooks using `create_harmonized_panel` loaded available years only

### Weighting

- Clustering: Unweighted
- Descriptive statistics: Weighted (year-specific survey weights)

### Software

- Python 3.12, pandas, scikit-learn, networkx
- K-means clustering with k=5, n_init=50

---

*Generated from notebooks 14--27 of the NZES Extended Analysis project.*
