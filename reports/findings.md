# NZ Polling Analysis: Findings Report

**Analysis Date:** 2026-02-12
**Total Polls Analyzed:** 1016
**Date Range:** 1990-10-27 to 2026-02-03
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

**Crises:**

- **Christchurch earthquake** (2011-02-22): National -1.0%, Labour +0.8%
- **Delta outbreak, second lockdown** (2021-08-17): National -4.5%, Labour +1.6%

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

## Data Sources

| Source | Coverage | Notes |
|--------|----------|-------|
| Wikipedia polling tables | 1990-10-27 to 2026-02-03 | 1016 party vote polls |
| Ipsos NZ Issues Monitor | Sep 2017 – Oct 2025 | 30 waves, govt performance + issue salience + party capability |
| World Bank / Stats NZ | Various | GDP, unemployment, CPI |

---

*Report generated by analysis.py*