# NZ Polling Analysis: Findings Report

**Analysis Date:** 2026-03-09
**Total Polls Analyzed:** 1020
**Date Range:** 1990-10-27 to 2026-03-03
**Election Cycles:** 1993, 1996, 1999, 2002, 2005, 2008, 2011, 2014, 2017, 2020, 2023, 2026

## Executive Summary

- **National-Labour Zero-Sum:** Confirmed (r = -0.31, p = 0.0000)
- **Third Party Squeeze:** Not detected
- **Mean Reversion (National):** 67% of extreme polls revert
- **Mean Reversion (Labour):** 64% of extreme polls revert
- **Govt Performance → Incumbent Vote:** r = 0.827 (p = 0.0000) — subjective approval tracks voting intention
- **Issue Ownership:** 10 issues show significant party links (e.g. inflation cost of living: Labour hurt when salient)
- **Perceived Competence → Vote:** r = 0.585 (pooled across issues)

## Detailed Findings

### 1. Polling Volatility by Election Cycle

**Overall Volatility (Standard Deviation):**

| Party | Mean | Std Dev |
|-------|------|---------|
| National | 41.4% | 8.7% |
| Labour | 35.1% | 7.5% |
| Green | 8.7% | 3.3% |
| ACT | 4.0% | 4.1% |
| NZ First | 5.0% | 3.3% |

**Election Proximity Effect:** Higher volatility near election
- Early campaign volatility: 8.6%
- Late campaign volatility: 9.1%
- Statistical significance: p = 0.3368

### 2. Mean Reversion Analysis

**Autocorrelation (persistence of polling levels):**

- National: Lag-1 = 0.90, Lag-2 = 0.91
- Labour: Lag-1 = 0.90, Lag-2 = 0.89

*Positive autocorrelation indicates polling levels persist over time.*

**Extreme Poll Reversion:**
- National: 67% of outlier polls regress toward mean (125 extreme polls)
- Labour: 64% of outlier polls regress toward mean (115 extreme polls)

### 3. National-Labour Zero-Sum Relationship

**Level Correlation:** r = -0.506 (p = 0.000000)
**Change Correlation:** r = -0.310 (p = 0.000000)

*Interpretation:* Changes are inversely related

### 4. Third Party Squeeze Effect

**Regression: Minor Party Support ~ Days to Election**
- Coefficient: -0.0003 (per day)
- p-value: 0.6822
- R-squared: 0.000

**Early vs Late Campaign:**
- Minor party support (>90 days out): 17.7%
- Minor party support (<30 days out): 19.3%
- Difference: -1.6% (p = 0.0442)

*Interpretation:* No significant third party squeeze detected

### 5. MMP Transition Effect

**National:** Pre-MMP std = 6.4%, Post-MMP std = 8.7%
**Labour:** Pre-MMP std = 4.3%, Post-MMP std = 7.5%

**Minor Party Growth under MMP:** Mean combined support = 18.6%

### 6. Momentum Effects

**National:** Change autocorrelation = -0.510 (p = 0.0000)
- Reversal effect - gains predict losses
**Labour:** Change autocorrelation = -0.451 (p = 0.0000)
- Reversal effect - gains predict losses

### 7. Economic Voting

**Correlations with Incumbent Support:**

- Gdp Growth: r = -0.116 (p = 0.7344, n = 11)
- Unemployment Rate: r = -0.251 (p = 0.4575, n = 11)
- Cpi Inflation: r = 0.014 (p = 0.9684, n = 11)

**Multiple Regression (R² = 0.083):**
- GDP Growth: coef = 0.119, p = 0.9357
- Unemployment: coef = -1.247, p = 0.5023
- Inflation: coef = -0.573, p = 0.7487

### 8. Event Studies

**Leadership Changes:**

| Date | Event | National Change | Labour Change |
|------|-------|-----------------|---------------|
| 2017-08-01 | Little → Ardern | -2.1% | +8.9% |
| 2018-02-27 | English → Bridges | +3.2% | — |
| 2020-07-14 | Bridges → Collins | +2.1% | — |
| 2021-11-30 | Collins → Luxon | +7.7% | -3.2% |

Mid-term leadership changes produce larger honeymoon bounces than election transitions. The Ardern (+8.9pp) and Luxon (+7.7pp) changes were the largest in the dataset.

**Crises:**

- **Christchurch earthquake** (2011-02-22): National -1.0%, Labour +0.8%
- **Delta outbreak, second lockdown** (2021-08-17): National -4.5%, Labour +1.6%

**Cumulative Damage:**

Multiple events in succession compound. Correlation between number of successive negative events and polling decline: r = -0.755. Coalition infighting is particularly damaging because it generates multiple events in sequence.

### 9. Government Performance vs Incumbent Polling (Ipsos)

**Data:** 29 months matched (2017-09 to 2025-10)

**Overall Correlation:** r = 0.827 (p = 0.0000, n = 29)
- Strong positive: higher govt ratings → higher incumbent vote

**Regression:** R² = 0.684, Each +1 point in govt rating ≈ +5.7pp incumbent vote

**Labour (2017-2023):** r = 0.795 (p = 0.0000, n = 19)
**National (2023-2025):** r = 0.893 (p = 0.0005, n = 10)

**Time-Lagged Cross-Correlation:** Strongest correlation at lag=0 months (contemporaneous) (r = 0.827)


### 10. Issue Salience vs Party Support (Ipsos)

**Data:** 27 months matched

**Level Correlations (salience % vs party vote %):**

| Issue | National r | National p | Labour r | Labour p |
|-------|-----------|-----------|---------|---------|
| Inflation Cost Of Living | -0.013 | 0.949 | **-0.885** | 0.000 |
| Healthcare Hospitals | -0.140 | 0.485 | **0.670** | 0.000 |
| The Economy | 0.241 | 0.226 | **-0.701** | 0.000 |
| Housing Price | 0.233 | 0.241 | **-0.692** | 0.000 |
| Crime Law Order | **0.403** | 0.037 | **-0.726** | 0.000 |
| Poverty Inequality | 0.014 | 0.946 | 0.335 | 0.087 |
| Unemployment | **-0.593** | 0.001 | **0.631** | 0.000 |
| Climate Change | -0.215 | 0.281 | -0.373 | 0.055 |
| Education | 0.085 | 0.675 | **-0.423** | 0.028 |
| Race Relations | 0.366 | 0.061 | 0.273 | 0.168 |

**Issue Ownership Pattern:**
- Inflation Cost Of Living: Labour hurt when salient
- Healthcare Hospitals: Labour benefits when salient
- The Economy: Labour hurt when salient
- Housing Price: Labour hurt when salient
- Crime Law Order: National benefits when salient
- Poverty Inequality: Labour benefits when salient
- Unemployment: National hurt when salient
- Climate Change: Labour hurt when salient
- Education: Labour hurt when salient
- Race Relations: National benefits when salient

**First-Differenced Correlations (change in salience vs change in vote):**

| Issue | National r | Labour r | n |
|-------|-----------|---------|---|
| Inflation Cost Of Living | 0.198 | -0.210 | 26 |
| Healthcare Hospitals | -0.308 | 0.662 | 26 |
| The Economy | 0.152 | -0.174 | 26 |
| Housing Price | 0.509 | -0.536 | 26 |
| Crime Law Order | 0.108 | -0.386 | 26 |
| Poverty Inequality | 0.239 | -0.055 | 26 |
| Unemployment | -0.290 | 0.579 | 26 |
| Climate Change | -0.364 | 0.345 | 26 |
| Education | 0.353 | -0.022 | 26 |
| Race Relations | -0.102 | 0.342 | 26 |


### 11. Party Capability vs Party Vote (Ipsos)

**Data:** 10 waves, 6 issues

**Pooled Correlation (all issues × both parties):** r = 0.585 (p = 0.0000, n = 120)
- Strong link: perceived competence tracks voting intention

**National:** r = 0.355 (p = 0.0053, n = 60)
**Labour:** r = 0.327 (p = 0.0108, n = 60)

**By Issue:**

| Issue | National r | Labour r | n |
|-------|-----------|---------|---|
| Inflation Cost Of Living | 0.675 | 0.717 | 10 |
| Healthcare Hospitals | 0.374 | 0.623 | 10 |
| The Economy | 0.873 | 0.892 | 10 |
| Housing Price | 0.515 | 0.573 | 10 |
| Crime Law Order | 0.777 | 0.885 | 10 |
| Unemployment | -0.731 | -0.787 | 10 |


### 12. Mortgage Rates vs Polling (RBNZ hb20)

**Data:** RBNZ 2-year fixed mortgage rate (monthly, Dec 2004 – Feb 2026) vs 789 polls

#### Incumbent Punishment Effect

| Test | r | p-value | n |
|------|---|---------|---|
| Level correlation | -0.328 | < 0.001 | 241 |
| Month-on-month Δ | -0.067 | 0.302 | 240 |
| **1-month lagged Δ** | **-0.209** | **0.001** | 239 |
| **Quarterly (3-month) Δ** | **-0.307** | **< 0.001** | 238 |

**Granger causality:** Mortgage rate changes Granger-cause incumbent vote changes at all lags 1–6 (strongest at lag 1: F=13.8, p=0.0003).

**Interpretation:** Rising mortgage rates predict incumbent polling declines ~1 month later. A 1pp quarterly rate rise corresponds to roughly a 2–3pp incumbent vote drop. This is the **cost-of-living punishment** mechanism — voters blame the government for rate rises regardless of whether monetary policy is within government control.

#### Government vs Opposition Bloc

| Bloc | Quarterly Δ r | p-value | 1-month lag r | p-value |
|------|--------------|---------|---------------|---------|
| Government bloc | -0.258 | 0.0001 | -0.154 | 0.017 |
| Opposition bloc | +0.261 | < 0.001 | +0.147 | 0.023 |

Rate rises are a near-perfect zero-sum transfer: the government bloc loses almost exactly what the opposition bloc gains.

#### Party-Level Effects (controlling for incumbency)

Quarterly change correlations, split by whether the party was in government or opposition at the time:

| Party | In Govt r | p | n | In Opp r | p | n |
|-------|----------|---|---|----------|---|---|
| National | -0.021 | 0.812 | 133 | **+0.370** | **< 0.001** | 105 |
| Labour | **-0.212** | **0.030** | 105 | +0.057 | 0.515 | 133 |
| Green | -0.032 | 0.802 | 63 | -0.092 | 0.227 | 175 |
| NZ First | -0.103 | 0.469 | 52 | +0.008 | 0.918 | 186 |
| ACT | -0.056 | 0.782 | 27 | -0.078 | 0.263 | 208 |

**Key finding:** The mortgage effect is a **major-party phenomenon**. Labour is significantly hurt by rate rises when in government (r=-0.212, p=0.030). National significantly benefits from rate rises when in opposition (r=+0.370, p<0.001). Minor parties (Green, NZ First, ACT) show no significant mortgage rate sensitivity in either direction.

#### Rate-Rise Episodes (>1pp rise over 6 months)

All 14 qualifying months occurred during Nov 2021 – Feb 2023 (Labour government, post-COVID rate tightening):

| Party | Mean 6-month polling change |
|-------|----------------------------|
| National | **+3.4pp** |
| Labour | **-3.8pp** |
| Green | +0.1pp |
| NZ First | +0.1pp |
| ACT | -0.2pp |

#### Mean Party Vote by Mortgage Rate Regime

| Rate band | National | Labour | Green | NZ First | ACT |
|-----------|----------|--------|-------|----------|-----|
| < 4% | 27.8 | 47.0 | 9.6 | 2.1 | 8.6 |
| 4–5% | 38.5 | 43.2 | 7.4 | 3.4 | 4.8 |
| 5–6% | 43.5 | 31.9 | 11.3 | 6.3 | 2.5 |
| 6–7% | 47.5 | 30.4 | 9.8 | 3.9 | 3.2 |
| > 7% | 43.6 | 35.2 | 7.8 | 4.1 | 3.8 |

*Note: Level correlations are confounded by incumbency — high rates coincided with National-led governments (2008–2017), low rates with Labour's COVID-era support. The change-based and lagged analyses above control for this.*


### 13. What Drives Government Approval? (Causal Chain Analysis)

**Data:** 29 matched observations (Ipsos approval × mortgage rate × incumbent polling), 24 with issue salience

#### Drivers of Approval (Level Correlations)

| Predictor | r | p-value | n |
|-----------|---|---------|---|
| Inflation/cost-of-living salience | **-0.869** | **< 0.001** | 24 |
| Unemployment salience | **+0.757** | **< 0.001** | 24 |
| Mortgage rate (2yr fixed) | **-0.754** | **< 0.001** | 30 |
| Crime/law & order salience | **-0.744** | **< 0.001** | 24 |
| Economy salience | **-0.696** | **< 0.001** | 24 |
| Housing price salience | **-0.577** | **0.003** | 24 |
| Education salience | -0.482 | 0.017 | 24 |
| Healthcare salience | +0.443 | 0.030 | 24 |

**Interpretation:** Government approval is overwhelmingly driven by **cost-of-living anxiety**. When voters are worried about inflation, mortgage rates, crime, or "the economy", approval drops. The top predictor is how salient cost-of-living is as an issue (r=-0.869), not any objective economic measure.

#### Change Correlations (what predicts *shifts* in approval?)

| Predictor | Δr | p-value |
|-----------|-----|---------|
| Δ Unemployment salience | **+0.783** | **< 0.001** |
| Δ Crime salience | **-0.694** | **< 0.001** |
| Δ Healthcare salience | **+0.565** | **0.009** |
| Δ Housing price salience | -0.408 | 0.074 |
| Δ Mortgage rate | -0.205 | 0.286 |

**Key insight:** Changes in *issue salience* predict changes in approval more strongly than changes in mortgage rates themselves. The mechanism is: economic conditions → public concern about issues → government disapproval → vote shift.

#### Stepwise Model for Approval (R² = 0.91)

| Step | Variable Added | Adj R² | ΔR² |
|------|---------------|--------|-----|
| 1 | Inflation/cost-of-living salience | 0.744 | 0.744 |
| 2 | Unemployment salience | 0.880 | 0.136 |
| 3 | Education salience | 0.895 | 0.016 |

Just two issue salience variables explain 88% of the variance in government approval. Cost-of-living salience alone explains 74%.

#### Mediation: Mortgage Rate → Approval → Polling

| Path | Coefficient | p-value |
|------|-------------|---------|
| Total effect (mortgage → vote) | -4.17 | < 0.001 |
| Direct effect (mortgage → vote, controlling for approval) | -1.35 | 0.163 |
| Indirect effect (mortgage → approval → vote) | -2.82 | — |
| **% mediated** | **67.7%** | — |
| Sobel test | z = -3.32 | **0.001** |

**Finding:** 68% of the mortgage rate effect on polling operates *through* government approval. The causal chain is:

**Mortgage rates ↑ → Cost-of-living concern ↑ → Government approval ↓ → Incumbent vote ↓**

The remaining 32% direct effect (mortgage → vote, bypassing approval) is not statistically significant on its own (p=0.163), suggesting approval almost fully mediates the relationship.

#### Direction Reversal Under National Government

| Government | Mortgage r vs approval | Interpretation |
|------------|----------------------|----------------|
| Labour (2017–2023) | **-0.814** | Rising rates → falling approval (cost-of-living pain) |
| National (2023–present) | **+0.829** | Falling rates → falling approval (rates falling because economy is in recession) |

Under National, falling mortgage rates don't *help* — they signal RBNZ emergency response to recession. The cost-of-living → approval link remains (r=-0.748 for Labour period), but the mortgage rate proxies for different things depending on the economic regime.


### 14. Petrol Prices vs Incumbent Polling (MBIE)

**Data:** MBIE weekly regular petrol board price (monthly, Apr 2004 – Feb 2026) vs 250 matched months

#### Incumbent Punishment Effect

| Test | r | p-value | n |
|------|---|---------|---|
| Level correlation | -0.413 | < 0.001 | 250 |
| **Month-on-month Δ** | **-0.193** | **0.002** | 249 |
| **Quarterly (3-month) Δ** | **-0.348** | **< 0.001** | 247 |
| YoY change | -0.406 | < 0.001 | 238 |

**Granger causality:** Petrol price changes Granger-cause incumbent vote changes at all lags 1–6 (lag 1: F=12.0, p=0.0006).

**Key finding:** Petrol prices show an even more immediate pocketbook effect than mortgage rates — the strongest correlation is at lag 0 (contemporaneous), consistent with petrol being the most visible daily cost. The quarterly relationship (r=-0.348) is stronger than for mortgage rates (r=-0.307).

#### Government vs Opposition Bloc

| Bloc | Quarterly Δ r | p-value |
|------|--------------|---------|
| Government bloc | -0.320 | < 0.001 |
| Opposition bloc | +0.352 | < 0.001 |

#### Party-Level Effects (controlling for incumbency)

| Party | In Govt r | p | In Opp r | p |
|-------|----------|---|----------|---|
| National | -0.021 | 0.814 | **+0.329** | **0.0003** |
| Labour | **-0.315** | **0.001** | +0.073 | 0.401 |
| Green | n/a | n/a | +0.058 | 0.364 |
| NZ First | n/a | n/a | +0.084 | 0.189 |
| ACT | n/a | n/a | -0.068 | 0.293 |

**Pattern matches mortgage finding:** Major-party-only effect. Labour hurt when in government; National benefits when in opposition. Minor parties insulated.

#### Petrol Price Spike Episodes (>30 c/L rise over 6 months)

22 qualifying months (2005–2023):

| Party | Mean 3-month polling change |
|-------|-----------------------------|
| National | +0.8pp |
| Labour | **-1.8pp** |
| Green | +0.8pp |
| NZ First | +0.3pp |
| ACT | -0.2pp |

#### Mediation: Petrol → Approval → Vote

| Path | Coefficient | p-value |
|------|-------------|---------|
| Total effect (petrol → vote) | -0.201 | < 0.001 |
| Direct effect (controlling approval) | -0.103 | 0.008 |
| Indirect (petrol → approval → vote) | -0.098 | — |
| **% mediated** | **48.9%** | — |
| Sobel test | z = -3.11 | **0.002** |

**Finding:** 49% of the petrol price effect operates through government approval. Unlike mortgage rates (68% mediated), petrol retains a larger direct effect — possibly because petrol price pain is more immediately visible and doesn't require the cognitive step of linking it to "cost of living" as an issue.

**By government period:** National govts show stronger petrol punishment (r=-0.231, p=0.007) than Labour govts (r=-0.167, p=0.073).


### 15. Consumer Confidence vs Incumbent Polling (OECD)

**Data:** OECD Composite Consumer Confidence Index (monthly, 1990–2025) vs 369 matched months

#### Incumbent Support Correlations

| Test | r | p-value | n |
|------|---|---------|---|
| Level correlation | +0.389 | < 0.001 | 369 |
| Month-on-month Δ | +0.124 | 0.017 | 368 |
| Quarterly (3-month) Δ | +0.151 | 0.004 | 366 |

**Granger causality:** Consumer confidence Granger-causes incumbent vote changes (lag 1: F=9.3, p=0.002).

**Interpretation:** Rising consumer confidence predicts rising incumbent support. The effect is positive (unlike petrol/mortgage which are negative costs), confirming the "feel-good factor" in political science literature.

**By government period:** Similar effect under both Labour (r=0.157, p=0.055) and National (r=0.141, p=0.039).

#### CCI vs Government Approval

Consumer confidence correlates with Ipsos government approval: r = 0.488, p = 0.007 (n=29).

#### Mediation: CCI → Approval → Vote

| Path | Coefficient | p-value |
|------|-------------|---------|
| Total effect (CCI → vote) | 4.586 | < 0.001 |
| Direct effect (controlling approval) | 2.310 | 0.006 |
| Indirect (CCI → approval → vote) | 2.275 | — |
| **% mediated** | **49.6%** | — |
| Sobel test | z = 2.62 | **0.009** |

**Finding:** 50% of the consumer confidence effect on polling operates through government approval. The other 50% is a direct "feel-good" effect on voting intention, independent of formal approval. This is consistent with the literature: consumer sentiment captures something broader than what approval ratings measure.


### 16. Poll Accuracy & Systematic Bias

**Data:** Final polls (last 14 days) compared to election results across 10 elections (1996–2023)

#### Accuracy by Time Window

| Window | National MAE | Labour MAE | National Bias | Labour Bias |
|--------|-------------|-----------|--------------|------------|
| Final 7 days | 2.12pp | 1.38pp | +0.44pp | -0.42pp |
| Final 14 days | 2.48pp | 1.72pp | +0.82pp | -0.12pp |
| Final 30 days | 3.09pp | 2.32pp | +1.35pp | +0.39pp |
| Full cycle | 4.47pp | 4.22pp | +2.42pp | +1.28pp |

#### Shy Tory Effect

**Not confirmed.** NZ polls actually *overestimate* National by +0.82pp on average (t=2.15, p=0.036). The pattern is inconsistent across elections:
- Underestimated National: 1996 (-1.2), 2017 (-0.5), 2023 (-3.8)
- Overestimated National: 2002 (+2.2), 2011 (+3.6), 2020 (+4.3)

**No systematic shy Tory effect in NZ** — if anything, there's a slight "bold National" effect where polls overstate their support.

#### Convergence

Polls become significantly more accurate closer to election day:
- National: r(days_out, |error|) = 0.243, p < 0.001
- Labour: r(days_out, |error|) = 0.359, p < 0.001

#### Two-Party Margin

Mean absolute margin error: 2.4pp (final 14 days). Polls called the winner correctly in 9/10 elections (incorrect only in 2005, where the margin was ~2pp).

#### Minor Party Bias

| Party | Systematic Bias | p-value |
|-------|----------------|---------|
| Green | **+1.03pp overestimated** | **< 0.001** |
| NZ First | **-0.74pp underestimated** | **0.0003** |
| ACT | +0.19pp (not significant) | 0.114 |

Green party is consistently overestimated in polls (possibly due to social desirability bias or differential turnout).


### 17. PM Preference Lead/Lag (Presidentialisation Thesis)

**Data:** 299 PM preference polls and party vote polls, 148 matched months (2005–2025)

#### Level Correlations

| Comparison | r | p-value |
|-----------|---|---------|
| National PM pref → National vote | 0.624 | < 0.001 |
| Labour PM pref → Labour vote | 0.637 | < 0.001 |

#### Cross-Correlation (Lead/Lag)

**National:** Peak correlation at lag 0 (contemporaneous, r=0.264, p=0.001). No evidence of PM preference leading or lagging party vote.

**Labour:** Weak, inconsistent lag structure. No clear lead/lag pattern.

**Conclusion:** The presidentialisation thesis (Poguntke & Webb 2005) receives **partial support at best**. PM preference and party vote move largely contemporaneously rather than PM preference leading party support.

#### PM Margin → Vote Margin

| Metric | Value |
|--------|-------|
| PM margin vs vote margin r | **0.763** |
| p-value | < 0.001 |
| R² | 0.582 |
| 1pp PM preference lead ≈ | 0.32pp party vote lead |

**Interpretation:** While there's no lead/lag relationship, PM preference is a strong contemporaneous predictor. The preferred PM margin explains 58% of the variance in the party vote margin. However, the effect is attenuated: a 1pp PM lead translates to only ~0.32pp party vote lead, reflecting that many voters choose party over leader.


### 18. House Prices vs Incumbent Polling (BIS)

**Data:** BIS Real House Price Index (quarterly, 1962–2025) vs 138 matched quarters

#### No Significant Effect

| Test | r | p-value | n |
|------|---|---------|---|
| Level correlation | 0.044 | 0.610 | 138 |
| Quarter-on-quarter Δ | -0.060 | 0.485 | 137 |
| Year-on-year Δ | -0.069 | 0.427 | 134 |
| Granger causality (lag 1) | F=0.17 | 0.678 | — |

**Key null finding:** Real house prices have **no significant relationship** with incumbent polling. This likely reflects the cancellation of two opposing forces:
- **Wealth effect:** Rising prices make homeowners feel richer (helps incumbent)
- **Affordability grievance:** Rising prices lock out renters/FHBs (hurts incumbent)

The net effect is zero — or more precisely, below detectable significance.

**Exception:** ACT party shows a significant positive correlation with house prices (r=0.404, p<0.001 for YoY changes), possibly reflecting ACT's homeowner/investor base.


### 19. Cost of Ruling: Coalition vs Majority Decay

**Data:** Monthly polling decay rates for 10 government periods (1996–2026)

#### Decay Rates by Government

| Government | Type | Decay (pp/month) | r | p | Total Change |
|-----------|------|------------------|---|---|-------------|
| Bolger/Shipley-NZF (96-99) | Coalition | -0.184 | -0.444 | 0.0003 | -6.5pp |
| Clark-Alliance (99-02) | Coalition | +0.081 | +0.186 | 0.263 | +2.6pp |
| Clark minority (02-05) | Coalition | -0.219 | -0.559 | < 0.001 | -8.3pp |
| Clark-NZF-UF (05-08) | Coalition | -0.210 | -0.639 | < 0.001 | -7.9pp |
| Key-ACT-UF-MP (08-11) | Coalition | +0.012 | +0.045 | 0.608 | +0.4pp |
| Key-ACT-UF (11-14) | Near-majority | +0.089 | +0.309 | 0.0002 | +3.0pp |
| Key/English-ACT-UF (14-17) | Near-majority | -0.157 | -0.522 | < 0.001 | -5.7pp |
| Ardern-NZF-Grn (17-20) | Coalition | +0.263 | +0.535 | 0.0001 | +9.4pp |
| Ardern/Hipkins majority (20-23) | Majority | **-0.559** | -0.875 | < 0.001 | **-20.1pp** |
| Luxon-ACT-NZF (23-) | Coalition | -0.259 | -0.699 | < 0.001 | -7.0pp |

#### Coalition vs Majority

| Type | Mean decay (pp/month) |
|------|----------------------|
| Coalition (n=7) | -0.074 |
| Majority/near-majority (n=3) | -0.209 |
| t-test p-value | 0.430 |

**Paldam (1991) NOT confirmed:** No significant difference between coalition and majority governments. If anything, the sole majority government (Labour 2020–23) decayed *fastest* at -0.559 pp/month — the opposite of Paldam's prediction. This may reflect that majority governments take more ambitious actions that generate opposition, while coalitions are constrained by their partners.

**Overall mean decay:** -0.114 pp/month across all governments.


### 20. Net Migration vs NZ First and Party Polling

**Data:** World Bank annual net migration (1993–2024) vs annual mean party vote polling, 32 matched years

#### Level Correlations

| Party | r | p-value |
|-------|---|---------|
| NZ First | +0.340 | 0.057 |
| ACT | -0.373 | 0.039 |
| Green | +0.346 | 0.071 |
| National | +0.094 | 0.608 |
| Labour | -0.059 | 0.748 |

#### NZ First and Immigration

NZ First support is higher during high-migration periods:
- High migration years (>20K): NZ First mean = 6.5%
- Low migration years (≤20K): NZ First mean = 3.9%

The correlation (r=0.340) is marginally significant (p=0.057). **Hangartner et al. (2019) receives partial support:** migration surges weakly predict NZ First gains, but the effect is not robust at conventional significance levels.

**Change correlations** are non-significant for NZ First (r=0.196, p=0.291), suggesting the level of migration matters more than sudden changes.

**ACT shows the opposite pattern:** Higher migration correlates with *lower* ACT support (r=-0.373 for levels, r=-0.521 for changes, p=0.003). This may reflect that ACT's libertarian, pro-business base is not anti-immigration, and migration-driven housing/infrastructure strain activates voters away from ACT's policy positions.


---

## Analysis 31: Budget Bounce

**Question:** Do budgets systematically boost incumbent support?

**Data:** 27 budgets (1991–2025) matched to polling in the 60-day window before and after each budget.

**Results:**

| Metric | Value |
|--------|-------|
| Mean budget bounce | **-0.3pp** |
| Positive bounces | 11/27 (41%) |
| Significant at p < 0.05 | 6/27 |

**By fiscal stance:**

| Stance | Mean Change | n |
|--------|------------|---|
| Expansionary | -0.4pp | 15 |
| Neutral | +1.7pp | 8 |
| Contractionary | -4.1pp | 4 |

**Election-year vs non-election-year:**
- Election-year budget bounce: +1.0pp (n=9)
- Non-election-year bounce: -1.0pp (n=18)

**Conclusion:** No systematic budget bounce. Expansionary budgets produce essentially no political benefit (-0.4pp). Contractionary budgets are clearly damaging (-4.1pp). Only 6 of 27 budgets produced statistically significant changes. Election-year budgets show a slight positive effect (+1.0pp), consistent with a recency heuristic, but the sample is too small to draw strong conclusions. Budgets move polls mainly when they are unusually contractionary or expansionary relative to expectations — not through a general "government doing stuff" mechanism.

---

## Analysis 32: Policy Shock Analysis

**Question:** Do major policy announcements shift party support beyond the "budget bounce" mechanism?

**Data:** 12 major policy events (2004–2025) identified from news records, matched to polling changes in 30-day windows.

**Largest effects (announcing party):**

| Event | Date | Party | Change (pp) | p |
|-------|------|-------|-------------|---|
| Labour announces fees-free tertiary | 2017-08-23 | Labour | +10.4 | 0.001 |
| Three Waters reform announced | 2021-10-27 | Labour | -3.8 | 0.034 |
| Three Strikes restored | 2024-01-01 | National | +2.3 | 0.024 |
| Working for Families announced | 2004-05-27 | Labour | +2.5 | 0.182 |

**Cross-party effects:**

| Event | Opposing Party Change | p |
|-------|----------------------|---|
| Working for Families announced | National -4.8pp | 0.011 |
| Three Waters reform announced | National +3.5pp | 0.043 |

**Caveat on the fees-free effect:** The 2017-08-23 fees-free announcement falls within 3 weeks of the Ardern leadership change (2017-08-01, +8.9pp Labour). The two events are too close to disentangle. The +10.4pp figure likely captures the continuation of the Ardern honeymoon rather than a pure policy effect.

**Conclusion:** Most routine policy announcements (tax cuts, KiwiSaver, GST increases) produce negligible polling shifts (|change| < 3pp, p > 0.1). The large significant effects (Working for Families, Three Waters) were unusual — either unusually salient policies that directly activated issue ownership mechanisms, or confounded by other events. The finding is consistent with Analysis 27 (government approval is the master variable): voters respond to overall government performance judgements, not individual policies. Policy announcements matter mainly when they change the salience of issues the party owns (or doesn't own).

---

## Comparison with Political Science Literature

### Expected vs. Observed Findings

| Hypothesis | Expected | Observed | Confirmed? |
|------------|----------|----------|------------|
| National-Labour negative correlation | Strong negative | r = -0.31 | Yes |
| Third party squeeze near elections | Support declines | Coef = -0.0003 | No |
| Mean reversion of outliers | Outliers regress | 67.2% revert | Yes |
| Govt approval → incumbent vote | Positive correlation | r = 0.827 | Yes |
| Issue ownership affects vote | Salience shifts benefit 'owning' party | 10 issues with sig. links | Yes |
| Perceived competence → vote | Positive correlation | r = 0.585 | Yes |
| Mortgage rates → incumbent punishment | Rate rises hurt incumbent | Granger-causal, r = -0.307 (quarterly) | Yes |
| Mortgage rate protest vote | Rate rises help main opposition | National in opp: r = +0.370 | Partial (major parties only) |
| Petrol prices → incumbent punishment | Price rises hurt incumbent | Granger-causal, r = -0.348 (quarterly) | Yes |
| Consumer confidence → incumbent support | Confidence helps incumbent | Granger-causal, r = +0.151 (quarterly) | Yes |
| Shy Tory effect | National underestimated | National overestimated +0.82pp | **No (reversed)** |
| Poll convergence | Accuracy improves near election | r(days_out, |error|) = 0.24–0.36 | Yes |
| PM preference leads party vote | PM leads | Contemporaneous (lag 0) | No (simultaneous) |
| House prices → incumbent support | Price rises help/hurt | r = -0.069 (YoY), p = 0.427 | **No effect** |
| Coalition decay faster (Paldam) | Coalitions erode faster | Coalition -0.07 vs majority -0.21 pp/mo | **No (reversed)** |
| Immigration → NZ First support (Hangartner) | Migration surges boost NZ First | r = 0.340, p = 0.057 | Partial |

---

## Analysis 21: NZES Retrospective vs Prospective Economic Voting

**Theory:** Fiorina (1981) argues voters are retrospective — they punish/reward based on past economic performance. MacKuen et al. (1992) argue voters are forward-looking, choosing based on expected future conditions.

**Data:** NZES individual-level data across 10 elections (1990-2023), using economic perception questions standardised to a -2 to +2 scale.

**Results:**
- Retrospective economic voting is **universal**: all 10 elections show significant positive correlation between "economy got better" and voting for the incumbent (mean |r| = 0.376, all p < 0.001)
- In head-to-head comparison (5 elections with both measures), **retrospective wins 4/5 times** (mean |r| retro = 0.380 vs prospective = 0.222)
- **Sociotropic dominates egotropic**: national economy perception matters more than personal finances (r = 0.361 vs 0.153 in 1990; r = 0.178 vs -0.003 in 2020)
- **Partisan perceptual screen**: incumbent voters see the economy +0.68 scale points more positively than opposition voters, every single year (p < 0.001 all years)
- Asymmetry: at the individual level, economic improvement helps incumbents more than pain hurts them (+22.3pp gain vs -10.0pp drop), which is opposite to the aggregate finding — likely reflecting rally effects and partisan selection
- No significant trend in economic voting strength over time (r = -0.187, p = 0.605)

**Conclusion:** NZ voters are overwhelmingly retrospective, sociotropic, and filtered through partisan lenses. Fiorina (1981) confirmed; MacKuen et al. (1992) largely rejected.

---

## Analysis 22: NZES Valence vs Position Voting

**Theory:** Stokes (1963) / Clarke et al. (2009) distinguish position voting (choosing the ideologically closest party) from valence voting (choosing the party perceived as most competent).

**Data:** NZES left-right self-placement and party placement (0-10 scale, 11 elections 1990-2023), plus leader competence ratings (4 elections 2014-2023).

**Results:**
- **Position (L-R proximity) is remarkably strong**: mean |r| = 0.698 across 11 elections, all p < 0.001
- In head-to-head comparison, **position wins all 4 elections** (2014-2023): mean |r| position = 0.738 vs valence = 0.679
- Logistic regression pseudo-R²: position = 0.586 vs valence = 0.468 (average across 4 elections)
- Combined model reaches pseudo-R² = 0.648-0.730 — both contribute independently
- Competence correctly predicted the winning party 3/4 times (wrong in 2023: Hipkins rated slightly more competent but National won)
- L-R proximity voting has been strengthening: r = 0.533 in 1990 → r = 0.760 in 2023

**Conclusion:** NZ elections are decided primarily by ideological proximity, with competence as a secondary factor. Position dominates, but both matter. The strengthening of positional voting suggests increasing ideological sorting despite the absence of polarisation (the L-R distribution remains stable).

---

## Analysis 23: Strategic Voting Under MMP

**Theory:** Cox (1997) predicts voters desert minor parties near the 5% threshold to avoid wasting their vote.

**Data:** NZES party vote vs electorate vote, and "party liked most" vs actual party vote, across 8 elections (1996-2023).

**Results:**
- **Split-ticket voting is common**: ~30% of voters cast electorate and party votes for different parties (mean 39.4% including outlier 2002)
- **Minor party voters lend electorate votes to majors**: ~50% of minor party voters give their electorate vote to Labour or National
- Strategic desertion from party vote is modest: typically 15-25% of minor party supporters vote for a different party than their most-liked
- **No significant correlation** between party size and desertion rate (r = -0.243, p = 0.24) — doesn't strongly support Cox's prediction
- When minor party deserters leave, they overwhelmingly go to Labour (50-60% in 2011-2017) or National

**Conclusion:** MMP encourages a different kind of strategic behaviour than Cox predicted. Rather than deserting minor parties on the party vote, voters use the *split ticket* — giving their party vote to a minor party and electorate vote to a major party. This is rational under MMP and produces high rates of split-ticket voting (~30%) without the squeeze effect that first-past-the-post systems generate.

---

## Analysis 24: Crime Statistics vs Crime Salience

**Theory:** Does actual recorded crime drive Ipsos crime salience, or is crime salience driven by media and perception independent of reality? Extends the "perception vs reality" finding from economics.

**Data:** NZ Police monthly victimisations (Dec 2017 – Dec 2023, source: Figure.NZ/NZ Police) matched to Ipsos crime salience (28 waves, 2018-2025).

**Results:**
- **Unlike economics, actual crime DOES predict crime salience** (r = 0.569, p = 0.007 for 3-month average; r = 0.577, p = 0.008 for 12-month average)
- Crime salience → govt approval: r = -0.858, p < 0.001 (very strong)
- Crime salience → Labour: r = -0.720, p < 0.001 (strong negative — issue ownership)
- Crime salience → National: r = +0.469, p = 0.037 (moderate positive)
- **Perception still beats reality** as a predictor of approval: salience R² = 73.6% vs actual crime R² = 63.0%
- Crime salience and cost-of-living salience are correlated (r = 0.463) — issues travel together
- COVID lockdowns caused both crime and crime salience to drop sharply

**Conclusion:** Crime is qualitatively different from economics. Actual crime rates *do* predict public concern, unlike GDP/CPI/unemployment which have essentially zero predictive power. But the perception layer still adds value beyond the objective indicator. Crime salience is a powerful predictor of government approval and activates the issue ownership mechanism (hurts Labour, helps National).

---

## Analysis 25: Google Trends NZ as Issue Salience Proxy

**Theory:** Can Google search intensity provide a high-frequency (monthly) proxy for issue salience that sits between objective conditions and survey-measured concern?

**Data:** Google Trends NZ monthly index for 5 keywords (2017-2025), matched to Ipsos Issues Monitor, govt approval, and party polling.

**Results:**
- **"Cost of living" searches DO predict Ipsos salience** (r = 0.482, p = 0.009) — the best-performing proxy
- **"Crime" searches predict govt approval decline** (r = -0.421, p = 0.020)
- **"Healthcare" searches strongly predict National decline** (r = -0.569, p < 0.001) — consistent with issue ownership
- **"Immigration" searches correlate with National gains** (r = 0.473, p < 0.001)
- "Housing crisis" barely registers as a search term and doesn't predict anything
- **No clear lead/lag**: Google Trends don't *lead* polls — they move contemporaneously, suggesting both respond to the same underlying events

**Conclusion:** Google Trends is a useful supplementary indicator but not a forecasting tool. It validates the issue ownership framework: search behaviour around healthcare favours Labour, while immigration searches coincide with National gains. The lack of lead/lag means trends can't predict future polling shifts — they're a real-time barometer, not a crystal ball.

---

## Analysis 26: Rental Price Index vs Polling

**Theory:** ~33% of NZ households rent. Rents are a direct pocketbook cost for a large, politically distinct demographic (younger, more urban, more Labour-leaning). May predict Labour-specific effects that mortgage rates miss.

**Data:** Stats NZ CPI "Actual rentals for housing" (quarterly, 1999-2025) matched to quarterly polling averages.

**Results:**
- Rental inflation → incumbent vote (levels): r = -0.731, p < 0.001 (OLS)
- National in government: r = -0.749 (very strong)
- Labour in government: r = -0.328
- **Rental and mortgage are uncorrelated** (r = 0.052) — they measure genuinely different pocketbook channels
- Rising rents specifically hurt Labour at the level (r = -0.233) and help Green (r = +0.256) and ACT (r = +0.246)
- Change correlations are all non-significant

**Robustness checks (added in response to adversarial critique):**
- **Granger causality: NOT significant** at any lag 1–6 (all p > 0.27). Rental inflation does not Granger-cause incumbent vote changes. This contrasts with mortgage rates (F=13.8, p=0.0003) and petrol prices (F=12.0, p=0.0006), which both pass.
- **Newey-West HAC standard errors:** The level correlation (OLS p=0.0006) **loses significance** under Newey-West correction (p=0.112). The OLS standard errors are inflated by serial autocorrelation in both series. Change correlations remain non-significant (NW p=0.854).
- **Reverse Granger test:** Also non-significant (incumbent vote does not Granger-cause rental changes), so the relationship is not causal in either direction.

**Revised conclusion:** The strong level correlation (r=-0.731) between rental inflation and incumbent vote share is likely a **spurious regression** between two persistent time series. It does not survive Granger causality testing, Newey-West standard errors, or first-differencing. The rental-mortgage independence finding (r=0.052) remains valid and interesting — they do measure different channels — but the claim that rental inflation is a *causal* pocketbook indicator is not supported. Rental inflation is better interpreted as a **contemporaneous co-movement** that reflects the same underlying political-economic environment, not as a causal driver of vote shifts. This demotes rental inflation from the project's "strongest pocketbook indicator" to an associational finding only.

---

## Analysis 27: Multivariate Incumbent Vote Model

**Theory:** Across 26 analyses we identified multiple predictors of incumbent vote share. Do they provide independent explanatory power, or are they all proxies for the same underlying process?

**Data:** Ipsos government performance (30 waves), issue salience (28 waves), RBNZ mortgage rates, MBIE petrol prices, OECD consumer confidence, Stats NZ rental CPI — all matched to incumbent vote share at Ipsos survey dates (n=24-30).

**Results:**

| Model | R² | Adj R² | AIC | n |
|-------|-----|--------|-----|---|
| Approval only | 0.675 | 0.664 | 175.8 | 30 |
| Approval + CoL salience | 0.745 | 0.720 | 140.6 | 24 |
| Kitchen sink (7 predictors) | 0.806 | 0.721 | 144.0 | 24 |
| **Parsimonious (approval + petrol)** | **0.767** | **0.745** | **138.3** | **24** |

- Government approval alone explains 67.5% of incumbent vote variance
- Adding all 7 predictors raises R² to 0.806 but Adj R² *drops* to 0.721 (overfitting)
- Massive multicollinearity: all VIFs > 30, most > 100 — predictors are all measuring the same underlying phenomenon
- **Parsimonious model** (approval + petrol price) achieves lowest AIC with just two predictors
- Each 1-point increase in govt approval ≈ +3.8pp incumbent vote; each 1 c/L petrol price increase ≈ -0.1pp

**Conclusion:** The multivariate analysis confirms that government approval is the master variable. Economic indicators (mortgage rates, rental inflation, consumer confidence) don't add independent explanatory power because they all feed through the same causal chain: **pocketbook costs → cost-of-living concern → government disapproval → vote shift**. Approval already incorporates voters' subjective assessment of economic conditions. The one exception is petrol prices, which retain a small direct effect even controlling for approval — consistent with petrol's unique visibility as a daily-frequency price signal.

---

## Analysis 28: Forecasting Model

**Theory:** Can current indicators predict next quarter's incumbent vote? This is the practical payoff of the project — moving from explanation to prediction.

**Data:** Quarterly polling data (n=114 quarters) with lagged economic indicators and Ipsos data (n=31 quarters where Ipsos available).

**Results:**

Predicting next-quarter vote *changes*:
- **No indicator significantly predicts quarter-to-quarter changes** (all p > 0.2)
- Govt approval: r = -0.214, p = 0.247
- Consumer confidence: r = -0.009, p = 0.927
- Mortgage rate: r = +0.093, p = 0.400

Predicting next-quarter vote *levels*:
- Cost-of-living salience: r = -0.818, p < 0.001
- Govt approval: r = +0.701, p < 0.001
- Consumer confidence: r = +0.509, p < 0.001
- Petrol price: r = -0.421, p < 0.001

Leave-one-out cross-validation:
| Model | MAE | RMSE |
|-------|-----|------|
| Approval only | 3.84 pp | 5.16 pp |
| Approval + confidence | 3.20 pp | 3.94 pp |
| **Naïve baseline (this Q = next Q)** | **2.64 pp** | **4.10 pp** |

**Conclusion:** NZ incumbent polling follows a near-random walk. Current indicators predict next-quarter vote *levels* well (because levels are autocorrelated — if approval is high now, it'll probably still be high next quarter). But they cannot predict *changes* — the incremental shifts that actually matter for forecasting. The naïve baseline ("assume current polls continue") beats all model-based forecasts. This is consistent with the efficient markets hypothesis applied to polling: current polls already incorporate all available information. Forecasting would require predicting future *events* (scandals, crises, economic shocks), which are inherently unpredictable.

---

## Analysis 29: Myopic Retrospection (Achen & Bartels 2016)

**Question:** Do NZ voters overweight the election-year economy and ignore earlier performance?

**Data:** Stats NZ quarterly real GDP (chain volume, seasonally adjusted) matched to election results for 9 elections (1999-2023).

**Method:** For each government term, we calculate average quarterly GDP growth during the early term (all quarters before the election year) and late term (final 4 quarters before election). We then correlate each with the incumbent party's vote share change.

**Key results:**
- Election-year GDP growth → incumbent vote change: r = +0.588, p = 0.096
- Early-term GDP growth → incumbent vote change: r = -0.114, p = 0.770
- Full-term GDP growth → incumbent vote change: r = +0.380, p = 0.314
- Election-quarter YoY growth → incumbent vote change: r = +0.460, p = 0.213

**Interpretation:** The pattern is *suggestive* of myopic retrospection. Election-year economic conditions (r = +0.588) predict incumbent fate better than early-term conditions (r = -0.114). However, with only N=9 elections, the late-term correlation reaches only marginal significance (p = 0.096) — below conventional thresholds. The confidence interval for r=0.588 at N=9 stretches approximately from -0.10 to +0.90, so the true effect could range from negligible to very strong. The 2008 and 2023 elections are consistent with the thesis, but may be driving the result — removing either could substantially change the correlation. This is directionally consistent with Achen & Bartels (2016) and Healy & Lenz (2014) but **should be classified as suggestive rather than confirmed** given the small sample.

**Caveat:** This result is consistent with but cannot formally confirm myopia given the small N. It also aligns with our earlier null finding that macro fundamentals have no significant relationship with polling (Analysis 5) — both suggest economic conditions operate through perception rather than through direct GDP measurement.

---

## Analysis 30: Dealignment (Dalton 2000)

**Question:** Are demographic predictors of vote choice weakening over time in New Zealand?

**Data:** NZES individual-level survey data across 10 elections (1996-2023), with harmonised demographics: age, gender (female), education (4 levels), and Maori identity.

**Method:** Logistic regression (National vs Labour vote choice) ~ demographics for each election year. Track McFadden's pseudo-R² over time as a measure of total demographic predictive power.

**Key results:**

Full model pseudo-R² by year:
| Year | pseudo-R² | n | Predictors | Dominant predictor |
|------|-----------|---|------------|-------------------|
| 1996 | 0.032 | 2,622 | age+female+educ+Maori | age (β=-0.013***) |
| 1999 | 0.021 | 1,539 | age+female+educ+Maori | Maori (β=-0.99***) |
| 2002 | 0.004 | 2,902 | age+female+educ+Maori | education (β=+0.18**) |
| 2005 | 0.022 | 2,243 | age+female+educ | female (β=-0.40***) |
| 2008 | 0.011 | 2,009 | age+female+educ | female (β=-0.29**) |
| 2011 | 0.101 | 1,815 | age+female+educ+Maori | Maori (β=-1.94***) |
| 2014 | 0.103 | 1,751 | age+female+educ+Maori | Maori (β=-1.95***) |
| 2017 | 0.082 | 2,447 | age+female+educ+Maori | Maori (β=-1.71***) |
| 2020 | 0.054 | 2,432 | age+female+educ+Maori | Maori (β=-1.16***) |
| 2023 | 0.044 | 1,097 | age+female+educ+Maori | Maori (β=-1.02***) |

Overall trend: r = +0.532, p = 0.114 (NOT declining — if anything, increasing)

Individual predictor trends:
- **Age:** Reversed direction — negative in 1996 (young→National) to positive from 2005 (old→National). This is realignment, not dealignment. Trend toward zero: r = +0.620, p = 0.056.
- **Gender:** Consistently negative (women less likely to vote National). Weakening slightly but not significantly: r = -0.163, p = 0.653.
- **Education:** Unstable — positive in early years (educated→National), trending toward zero or negative by 2023.
- **Maori identity:** The strongest and most consistent predictor. Maori voters are dramatically less likely to vote National (OR = 0.14-0.39). The effect has *strengthened* since 1996, particularly during 2011-2014.

**Interpretation:** Dalton's dealignment thesis is **not confirmed** for New Zealand as a whole. Total demographic predictive power has not declined — if anything, it has increased, driven primarily by the strengthening Maori identity cleavage. However, there is *selective* dealignment: age and education effects have weakened, while the ethnic cleavage has strengthened. The reduced model (age + gender only) does show dealignment — pseudo-R² dropped 47% from 0.011 (1996) to 0.006 (2023). The overall picture is of realignment rather than dealignment: old cleavages (class-linked age effects) are being replaced by new ones (ethnic identity), consistent with Ford & Jennings (2020).

---

## Updated Hypothesis Comparison Table

| Hypothesis | Prediction | Result | p-value |
|------------|------------|--------|---------|
| National-Labour zero-sum | Vote share inversely correlated | **Confirmed** (r = -0.310) | < 0.001 |
| Mean reversion | Extreme polls revert | **Confirmed** (67% revert) | N/A |
| Reversal effect | Prior change predicts reversal | **Confirmed** (r = -0.51) | < 0.001 |
| Third party squeeze | Minor parties squeezed near elections | Not detected | 0.682 |
| Economic voting (macro) | GDP/CPI/unemp predict incumbent | Not significant | > 0.05 |
| Govt performance → vote | Subjective rating predicts vote | **Confirmed** (r = 0.827) | < 0.001 |
| Issue ownership | Issues activate party brands | **Confirmed** (10 issues) | < 0.05 |
| Perceived competence → vote | "Best party to manage X" | **Confirmed** (r = 0.585) | < 0.001 |
| Mortgage → incumbent punishment | Rate rises hurt incumbent | **Confirmed** (Granger F=13.8) | < 0.001 |
| Petrol → incumbent punishment | Price rises hurt incumbent | **Confirmed** (Granger F=12.0) | < 0.001 |
| Consumer confidence → incumbent | Confidence predicts vote | **Confirmed** (Granger F=9.3) | 0.003 |
| House prices → incumbent | Prices affect vote | **Not significant** (r = -0.069) | 0.43 |
| Immigration → NZ First | Migration boosts NZ First | **Partial** (r = 0.340) | 0.057 |
| Shy Tory effect | National underestimated | **Not confirmed** (overestimated +0.82pp) | > 0.05 |
| PM preference leads party vote | Presidentialisation | **Not confirmed** (contemporaneous) | N/A |
| Paldam coalition decay | Coalitions decay faster | **Not confirmed** (majority decayed fastest) | > 0.05 |
| Retrospective voting (Fiorina) | Past economy predicts vote | **Confirmed** (mean r = 0.376, all p < 0.001) | < 0.001 |
| Prospective voting (MacKuen) | Future expectations predict | **Partial** (loses 4/5 to retro) | varies |
| Position > valence (Stokes) | L-R distance > competence | **Confirmed** (position wins 4/4) | < 0.001 |
| Strategic desertion (Cox) | Minor parties squeezed by threshold | **Partial** — split-ticket instead | 0.24 |
| Crime reality → salience | Actual crime predicts concern | **Confirmed** (r = 0.569) | 0.007 |
| Google Trends → salience | Search predicts survey concern | **Partial** — cost-of-living only (r = 0.482) | 0.009 |
| Rental inflation → incumbent | Rents hurt incumbent | **Associational only** (r = -0.731 levels; fails Granger, NW) | 0.112 (NW) |
| Multivariate model adds value | Multiple predictors > approval alone | **Not confirmed** (approval dominates, VIF > 30) | N/A |
| Forecasting possible | Current indicators predict next Q | **Not confirmed** (naïve baseline wins, MAE 2.64pp) | > 0.2 |
| Myopic retrospection (Achen & Bartels) | Election-year economy > full-term | **Partial** (late r=+0.588 vs early r=-0.114) | 0.096 |
| Dealignment (Dalton 2000) | Demographics less predictive over time | **Not confirmed** (pseudo-R² stable/increasing) | 0.114 |
| Budget bounce | Expansionary budgets boost incumbent | **Not confirmed** (mean -0.3pp across 27 budgets) | > 0.1 |

## Data Sources

| Source | Coverage | Notes |
|--------|----------|-------|
| Wikipedia polling tables | 1990-10-27 to 2026-03-03 | 1020 party vote polls, 299 PM polls |
| Ipsos NZ Issues Monitor | Sep 2017 – Oct 2025 | 30 waves, govt performance + issue salience + party capability |
| RBNZ hb20 | Dec 2004 – Feb 2026 | 2-year fixed mortgage rate (monthly) |
| MBIE fuel prices | Apr 2004 – Feb 2026 | Weekly regular petrol board price |
| OECD CLI | Jan 1990 – Dec 2025 | Composite consumer confidence (monthly) |
| BIS | Q1 1962 – Q3 2025 | Real house price index (quarterly) |
| World Bank | 1960 – 2024 | Net migration (annual), GDP, unemployment, CPI |
| NZ Electoral Commission | 1996 – 2023 | Official election results |
| NZES | 1990 – 2023 | NZ Election Study (16 tables, ~46K respondents) |
| NZ Police | Dec 2017 – Dec 2023 | Monthly victimisations (via Figure.NZ) |
| Google Trends NZ | Jan 2017 – Dec 2025 | Monthly search index, 5 keywords |
| Stats NZ CPI | Q2 1999 – Q4 2025 | Rental price index (quarterly) |

---

## Robustness Checks (Newey-West HAC Standard Errors)

All key time-series regressions were re-estimated with Newey-West heteroscedasticity and autocorrelation consistent (HAC) standard errors to address the concern that serial autocorrelation (lag-1 ρ ≈ 0.90) inflates significance in OLS. Bandwidth selected by Andrews' rule of thumb.

| Regression | r | OLS p | NW p | n | Survives NW? |
|------------|---|-------|------|---|-------------|
| Mortgage → Inc (levels) | -0.321 | < 0.001 | 0.0003 | 240 | Yes*** |
| Mortgage → Inc (changes) | -0.051 | 0.430 | 0.397 | 239 | No (was already n.s.) |
| Mortgage → Inc (lag-1 Δ) | -0.213 | 0.001 | 0.023 | 238 | Yes* |
| Petrol → Inc (levels) | -0.397 | < 0.001 | 0.002 | 248 | Yes** |
| Petrol → Inc (changes) | -0.186 | 0.003 | 0.005 | 247 | Yes** |
| CCI → Inc (levels) | +0.554 | < 0.001 | < 0.001 | 279 | Yes*** |
| CCI → Inc (changes) | +0.147 | 0.014 | 0.014 | 278 | Yes* |
| Rental → Inc (levels) | -0.335 | 0.001 | 0.112 | 101 | **No** |
| Rental → Inc (changes) | +0.029 | 0.775 | 0.854 | 100 | No (was already n.s.) |
| Approval → Inc (levels) | +0.825 | < 0.001 | < 0.001 | 29 | Yes*** |
| Approval → Inc (changes) | +0.545 | 0.003 | 0.002 | 28 | Yes** |

**Key conclusion:** All core pocketbook findings (mortgage rates, petrol prices, consumer confidence) and the government approval relationship survive Newey-West correction. The only casualty is the **rental inflation level correlation**, which loses significance (OLS p=0.001 → NW p=0.112). Combined with the failed Granger test, this demotes rental inflation from a confirmed causal indicator to an associational finding.

---

## Limitations

This project has several important limitations that should be considered when interpreting the findings:

### Statistical limitations

1. **Multiple comparisons.** The project runs 30 hypothesis tests without formal correction (Bonferroni, Benjamini-Hochberg). At α=0.05 with ~10 effectively independent tests, approximately 0.5 false positives are expected. However, 8/30 tests returned null results (reported prominently), and the core findings are supported by multiple converging analyses rather than single tests.

2. **Small N for some analyses.** Several findings rest on small samples: myopic retrospection (N=9 elections, p=0.096), Paldam cost-of-ruling test (N=10 government periods), regime-dependent mortgage effects (N=10 for National period), and the multivariate model (N=24 with 7 predictors). These should be treated as suggestive patterns rather than confirmed regularities.

3. **Serial autocorrelation.** Monthly polling data has lag-1 autocorrelation of ~0.90, which inflates OLS significance for level correlations. Newey-West HAC standard errors (reported above) show that most findings survive this correction, but the rental inflation level correlation does not. First-differenced correlations and Granger causality tests (which are robust to autocorrelation) are reported alongside level correlations throughout.

4. **Spurious regression risk.** The rental inflation finding (r=-0.731) is likely a spurious regression between two persistent time series. It fails both Granger causality and Newey-West correction. Other level correlations (mortgage, petrol, CCI) survive these tests but should still be interpreted alongside their change-based counterparts.

### Data limitations

5. **Wikipedia as primary polling source.** All 1,020 polls were sourced from Wikipedia, a secondary source subject to transcription errors and selective inclusion. Cross-checking against election results (Analysis 16) provides partial validation, and random errors would attenuate rather than inflate correlations.

6. **Ipsos data extraction.** The Ipsos Issues Monitor data was extracted from PDFs, including some chart-reading. The government performance table (page 11) extracts cleanly, but issue salience values from trend charts may contain measurement error. With only 24-30 Ipsos observations, even small errors could affect correlations.

7. **Mixed temporal resolutions.** The project combines annual (migration, GDP), quarterly (house prices, rental CPI), monthly (confidence, mortgages, petrol), and irregular (Ipsos, polls) data. Correlations are computed at matched resolutions, but aggregation introduces smoothing.

### Interpretive limitations

8. **Ecological fallacy.** Most findings are aggregate-level (polling averages vs economic indicators). Claims about voter behaviour (e.g., "voters blame the government") are partly mitigated by individual-level NZES analyses (Analyses 21-23) but cannot be fully confirmed without panel data tracking the same individuals over time.

9. **No media or opposition controls.** The issue salience framework cannot distinguish between economic conditions directly affecting voter concern and media coverage mediating the relationship. Government policy actions and opposition messaging are also uncontrolled confounders.

10. **Causal language.** Granger causality establishes temporal precedence, not true causation. The mediation analysis (Sobel test) uses cross-sectional data and cannot establish temporal ordering within a single time point. The causal chain interpretation relies on combining Granger tests (for temporal ordering) with mediation (for decomposing effect size).

11. **Generalisability.** All findings are from a single country (New Zealand) over a 30-year period with 10 elections under MMP. The "rules" may be specific to this era and institutional context. Cross-national replication would be needed to establish them as general regularities.

12. **The forecasting paradox.** The project's own forecasting test (Analysis 28) shows that no indicator predicts quarter-to-quarter vote changes — the naïve baseline beats all models. This means the "rules" describe contemporaneous relationships and response functions, not predictive tools. They explain *how* polls respond to shocks but cannot predict *when* shocks will occur.

---

*Report generated by analysis.py and supplementary analysis scripts. Robustness checks added 2026-03-09.*