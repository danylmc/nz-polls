# Laws of NZ Politics: Statistical Rules from 30 Years of Polling

*Generated: 2026-02-09*

## Summary Table

| # | Rule | Statement | Key Statistic | Strength |
|---|------|-----------|---------------|----------|
| 1 | Incumbent Fatigue | Governing parties lose ~1.6 points/year | slope=-1.56 pts/yr, p=0.0 | Strong |
| 2 | Punctuated Equilibrium | Top 10% of weeks account for 38% of movement | bootstrap p=0.0, kurtosis=8.32 | Strong |
| 3 | Cross-Bloc Tides | Left/right blocs move inversely (r=-0.706) | r=-0.706, p=0.0 | Strong |
| 4 | Leadership Honeymoon | New leaders get ~6.7 point bounce | midterm avg=6.7, election avg=? | Suggestive |
| 5 | Crisis Rally | Incumbent gains ~0.6 points in 30 days | mean boost=0.6, n=3 | Suggestive |
| 6 | Economic Sentiment | Economic conditions predict incumbent support | gdp_growth 12m change: coef=1.145, p=0.0 | Moderate |
| 7 | Election Proximity | incumbent fades in 7/9 elections; Labour settles late; NZF surges +1.2pts; gap narrows in only 4/11 cycles | incumbent fade 7/9, NZF late +1.2pts p=0.0 | Moderate |
| 8 | Minor Party Volatility | Minor parties 3.2x more volatile (CV) than major | major CV=0.212, minor CV=0.687 | Moderate |
| 9 | Post-Election Reset | Winners move +0.1 pts, losers -0.8 pts | n=10 transitions | Suggestive |
| 10 | Thermostatic Model | Deviations self-correct (β=-0.1848) | National β=-0.1848, p=0.0, n=969 | Strong |

## Detailed Findings

### Rule 1: Incumbent Fatigue Law

**Statement:** Governing parties lose support at a measurable rate per year in office.

| Government | Party | N | Start % | Pts/Year | r | p-value |
|------------|-------|---|---------|----------|---|---------|
| Bolger/Shipley 1993-1999 | National | 103 | 38.8 | -0.47 | -0.191 | 0.0539 |
| Clark 1999-2008 | Labour | 264 | 51.7 | -1.91 | -0.774 | 0.0000 |
| Key/English 2008-2017 | National | 351 | 53.2 | -0.98 | -0.568 | 0.0000 |
| Ardern/Hipkins 2017-2023 | Labour | 175 | 53.0 | -3.68 | -0.731 | 0.0000 |
| Luxon 2023- | National | 86 | 37.6 | -3.21 | -0.699 | 0.0000 |

**Pooled estimate (with government fixed effects):** -1.56 points/year (p = 0.0000, R² = 0.677, n = 979)

**Caveats:** Fatigue rate varies across governments; external shocks (COVID, crises) 
confound the time trend. Fixed effects absorb level differences but not within-government shocks.

### Rule 2: Punctuated Equilibrium

**Statement:** Polling moves in sudden bursts, not gradual drift. Most weeks are stable;
a small fraction of weeks accounts for most total movement.

**National** (1841 weeks):

- Movement concentration: top 10% of weeks = 38% of total |Δ| (Gaussian null: 26%, bootstrap p = 0.0000)
- Top 20% of weeks = 57% of total movement
- Regime switching: stable σ = 0.46, shift σ = 2.35 (ratio: 5.1x)
- Fraction of time in shift regime: 47%
- Excess kurtosis: 8.32 (p = 0.0000) — leptokurtic (fat-tailed bursts confirmed)

**Labour** (1841 weeks):

- Movement concentration: top 10% of weeks = 39% of total |Δ| (Gaussian null: 26%, bootstrap p = 0.0000)
- Top 20% of weeks = 58% of total movement
- Regime switching: stable σ = 0.54, shift σ = 2.51 (ratio: 4.7x)
- Fraction of time in shift regime: 38%
- Excess kurtosis: 6.14 (p = 0.0000) — leptokurtic (fat-tailed bursts confirmed)

### Rule 3: Cross-Bloc Tidal Dynamics

**Statement:** Left (Labour+Green) and right (National+ACT+NZFirst) blocs move as
coherent opposing units; bloc-level support is smoother than party-level.

**Volatility comparison (std dev):**

- National_std: 8.71
- Labour_std: 7.47
- Green_std: 4.11
- ACT_std: 4.04
- NZFirst_std: 3.38
- left_bloc_std: 7.02
- right_bloc_std: 6.98
- Left bloc smoother than Labour alone: True
- Right bloc smoother than National alone: True

**Bloc change correlation:** r = -0.706 (p = 0.000000, n = 1015)

**CUSUM tide analysis:**
- Mean bloc difference (Right - Left): +6.9 points
- Number of tide changes: 11
- Average tide duration: 1271 days (3.5 years)
- Current direction: Left-leaning

**Caveats:** Bloc composition is fixed (NZ First classified as right); in practice
NZ First acts as a swing party. Green party data sparse pre-2002.

### Rule 4: Leadership Honeymoon Law

**Statement:** New leaders receive a polling bounce that decays over time.
Mid-term changes produce larger bounces than election transitions.

| Event | Type | Before | After 30d | Bounce | Opponent Δ | Decay half-life |
|-------|------|--------|-----------|--------|------------|-----------------|
| Bolger → Shipley | mid-term | 33.3 | — | — | — | — |
| Election: Shipley → Clark | election | 37.6 | — | — | — | — |
| Election: Clark → Key | election | 48.0 | — | — | — | — |
| Key → English | mid-term | 48.1 | — | — | — | — |
| Little → Ardern | mid-term | 26.2 | 35.3 | +9.0 | -2.6 | 6d |
| Election: Labour-NZF-Green coalitio | election | 36.5 | — | — | — | — |
| Ardern wins majority | election | 47.3 | — | — | — | — |
| Ardern → Hipkins | mid-term | 30.5 | 34.8 | +4.3 | -3.3 | 253d |
| Election: Hipkins → Luxon | election | 35.9 | — | — | — | — |

**Averages:** mid-term bounce = 6.7 pts (n=2), election transition = ? pts (n=?)

**Caveats:** Small n (9 events). Bounces conflated with contemporaneous events.

### Rule 5: Crisis Rally Effect

**Statement:** Crises tend to boost incumbent support, at least in the short term.

| Crisis | Incumbent | Before | +30d | +60d | +90d | +180d |
|--------|-----------|--------|------|------|------|-------|
| Christchurch earthquake | National | 52.4 | -1.7 | +0.1 | +0.1 | +0.3 |
| Christchurch mosque shootings | Labour | 46.2 | +2.5 | +2.5 | +1.3 | +0.4 |
| COVID-19 lockdown begins | Labour | 41.6 | — | +15.1 | +14.7 | +11.9 |
| Delta outbreak, second lockdow | Labour | 42.4 | +1.0 | +1.8 | +0.4 | -1.2 |

**Average 30-day incumbent boost:** 0.6 pts (n=3)

**Caveats:** Only 4 crises; each is unique. COVID lockdown confounds rally with
extensive policy response.

### Rule 6: Economic Sentiment

**Statement:** Testing whether economic conditions predict incumbent support at poll-level (n=979) with various time lags.

**Level effects by lag:**

| Lag | Variable | Coefficient | p-value | Significant |
|-----|----------|-------------|---------|-------------|
| lag_0m | gdp_growth | 0.570 | 0.0000 | Yes |
| lag_0m | unemployment_rate | 3.065 | 0.0000 | Yes |
| lag_0m | cpi_inflation | -1.169 | 0.0000 | Yes |
| lag_12m | gdp_growth | -0.887 | 0.0000 | Yes |
| lag_12m | unemployment_rate | 1.832 | 0.0000 | Yes |
| lag_12m | cpi_inflation | -1.090 | 0.0000 | Yes |
| lag_3m | gdp_growth | 0.189 | 0.1116 | No |
| lag_3m | unemployment_rate | 3.107 | 0.0000 | Yes |
| lag_3m | cpi_inflation | -1.172 | 0.0000 | Yes |
| lag_6m | gdp_growth | -0.208 | 0.1004 | No |
| lag_6m | unemployment_rate | 2.730 | 0.0000 | Yes |
| lag_6m | cpi_inflation | -1.132 | 0.0000 | Yes |
| lag_9m | gdp_growth | -0.618 | 0.0000 | Yes |
| lag_9m | unemployment_rate | 2.265 | 0.0000 | Yes |
| lag_9m | cpi_inflation | -1.109 | 0.0000 | Yes |

**12-month change effects:**

| Variable | Coefficient | p-value | Significant |
|----------|-------------|---------|-------------|
| gdp_growth 12m Δ | 1.145 | 0.0000 | Yes |
| unemployment_rate 12m Δ | -0.168 | 0.5623 | No |
| cpi_inflation 12m Δ | 0.330 | 0.0231 | Yes |


**Caveats:** Economic data is interpolated from annual observations; true monthly/quarterly
data might reveal relationships. Government fixed effects absorb between-government variation.

### Rule 7: Election Proximity Behaviour

**Statement:** Several consistent patterns emerge as elections approach: polling
frequency surges, Labour's numbers settle, incumbents fade, and NZ First picks up
late-deciding voters. But the National-Labour gap does not systematically narrow.

#### Polling Frequency

| Days Out | Polls | Polls/Month |
|----------|-------|-------------|
| 365+ | 564 | 20.3 |
| 180-365 | 152 | 24.6 |
| 90-180 | 93 | 31.0 |
| 30-90 | 90 | 45.0 |
| 0-30 | 117 | 117.0 |

#### Poll-to-Poll Volatility

**National** — mean |change| by proximity:

| Days Out | Mean |Δ| | Median |Δ| | N |
|----------|---------|-----------|---|
| 365+ | 2.91 | 2.5 | 563 |
| 180-365 | 3.32 | 3.0 | 152 |
| 90-180 | 3.08 | 2.1 | 93 |
| 30-90 | 2.98 | 2.45 | 90 |
| 0-30 | 2.76 | 2.2 | 117 |

Late vs early: 2.76 vs 3.0 (p = 0.2886) — no significant settling

**Labour** — mean |change| by proximity:

| Days Out | Mean |Δ| | Median |Δ| | N |
|----------|---------|-----------|---|
| 365+ | 2.72 | 2.0 | 563 |
| 180-365 | 2.84 | 2.15 | 152 |
| 90-180 | 2.51 | 2.0 | 93 |
| 30-90 | 2.9 | 2.5 | 90 |
| 0-30 | 2.09 | 2.0 | 117 |

Late vs early: 2.09 vs 2.72 (p = 0.0033) — settles significantly

#### Incumbent Late Fade

Incumbents lose support in final 90 days in **7/9** elections — campaign scrutiny erodes rather than builds incumbent position.

| Election | Incumbent | 90-180d | 0-90d | Change |
|----------|-----------|---------|-------|--------|
| 1996 | National | 40.0 | 37.7 | -2.3 |
| 1999 | National | 36.5 | 32.5 | -4.0 |
| 2005 | Labour | 43.6 | 40.3 | -3.3 |
| 2008 | Labour | 32.3 | 35.2 | +2.9 |
| 2011 | National | 53.0 | 52.8 | -0.2 |
| 2014 | National | 48.4 | 49.1 | +0.7 |
| 2017 | National | 45.4 | 42.9 | -2.6 |
| 2020 | Labour | 55.0 | 49.4 | -5.6 |
| 2023 | Labour | 32.9 | 27.8 | -5.1 |

#### National-Labour Gap (No Consistent Convergence)

Gap narrows in **4/11** cycles — 
no systematic convergence; behaviour is election-specific.

| Election | Early Gap | Late Gap | Change |
|----------|-----------|----------|--------|
| 1993 | 10.3 | 2.8 | -7.4 (narrows) |
| 1996 | 12.5 | 15.1 | +2.6 (widens) |
| 1999 | 7.8 | 5.8 | -2.0 (narrows) |
| 2002 | 11.8 | 22.9 | +11.1 (widens) |
| 2005 | 15.9 | 3.7 | -12.2 (narrows) |
| 2008 | 10.0 | 12.9 | +2.9 (widens) |
| 2011 | 20.5 | 24.7 | +4.2 (widens) |
| 2014 | 13.9 | 22.8 | +8.9 (widens) |
| 2017 | 18.7 | 9.2 | -9.5 (narrows) |
| 2020 | 5.1 | 17.4 | +12.3 (widens) |
| 2023 | 8.5 | 8.6 | +0.1 (widens) |

#### Minor Party Late Movement

| Party | >180d Mean | <30d Mean | Change | p-value |
|-------|-----------|-----------|--------|---------|
| Green | 8.7 | 9.2 | +0.5 | 0.1413 |
| ACT | 4.1 | 3.7 | -0.4 | 0.2776 |
| NZ First | 4.6 | 5.8 | +1.2 ** | 0.0 |

NZ First's late surge (+1.2 pts, p = 0.0) is consistent with their
role as the party of late-deciding voters who are dissatisfied with both major parties.

#### Total Five-Party Share

Stable at 92.5-93.6% regardless of proximity — the ~7% allocated to other
parties and undecideds does not systematically shrink as elections approach.

**Caveats:** Polling frequency increase means late-campaign bins have more data,
potentially reducing noise. Incumbent fade conflated with natural fatigue trend.

### Rule 8: Minor Party Lifecycle Patterns

**Statement:** Minor parties are systematically more volatile than major parties
and their combined support acts as a dissatisfaction barometer.

**Coefficient of variation (std/mean):**

| Party | Mean % | Std % | CV |
|-------|--------|-------|----|
| National | 41.4 | 8.7 | 0.211 |
| Labour | 35.1 | 7.5 | 0.213 |
| Green | 8.7 | 3.3 | 0.382 |
| ACT | 4.0 | 4.1 | 1.020 |
| NZ First | 5.0 | 3.3 | 0.658 |

Major party avg CV: 0.212, Minor party avg CV: 0.687 — minor parties are **3.2x** more volatile

**Minor total vs incumbent support:** coefficient = -0.277 (p = 0.0000) — counter-cyclical (rises when incumbent falls)

**MMP 5% threshold stability:**

| Party | % polls above 5% | Threshold crossings | N polls |
|-------|-------------------|--------------------:|---------|
| Green | 90% | 67 | 915 |
| ACT | 33% | 49 | 946 |
| NZ First | 44% | 158 | 971 |


### Rule 9: Post-Election Reset

**Statement:** After elections, winner and loser support shifts from final polling levels.

| Election | Winner | Winner Δ | Loser | Loser Δ |
|----------|--------|----------|-------|---------|
| 1993 | National | -2.8 | Labour | -4.2 |
| 1999 | Labour | +13.0 | National | -0.2 |
| 2002 | Labour | +0.3 | National | -0.1 |
| 2005 | Labour | -1.0 | National | +0.3 |
| 2008 | National | +0.1 | Labour | -2.7 |
| 2011 | National | -5.6 | Labour | +2.2 |
| 2014 | National | +0.3 | Labour | +0.4 |
| 2017 | Labour | -3.4 | National | +1.6 |
| 2020 | Labour | -1.0 | National | -4.0 |
| 2023 | National | +1.4 | Labour | -1.5 |

**Average winner change:** +0.1 pts (5 up, 5 down)
**Average loser change:** -0.8 pts

**Caveats:** Post-election polls may reflect actual result rather than independent
opinion shift. Coalition negotiations (1996, 2017) create extended uncertainty.

### Rule 10: Thermostatic Model

**Statement:** Public opinion acts as a thermostat (Wlezien 1995) — when a party's
support rises above its equilibrium, it subsequently corrects downward, and vice versa.
This creates a negative feedback loop that stabilises the political system.

**National — error-correction model:**
- β = -0.1848 (p = 0.0000, n = 969)
- Each 1pt above mean predicts 0.18pt correction next poll
- Thermostatic correction confirmed

**National — asymmetric correction:**
- Above mean: β = -0.2088 (p = 0.0000, n = 517)
- Below mean: β = -0.0860 (p = 0.0205, n = 452)
- Faster correction: from above

**Labour — error-correction model:**
- β = -0.1481 (p = 0.0000, n = 969)
- Each 1pt above mean predicts 0.15pt correction next poll
- Thermostatic correction confirmed

**Labour — asymmetric correction:**
- Above mean: β = -0.1260 (p = 0.0006, n = 474)
- Below mean: β = -0.1387 (p = 0.0004, n = 495)
- Faster correction: from below

**Cross-party thermostat (high incumbent → opposition gains):**
- β = 0.0748 (p = 0.0004, n = 978)
- High incumbent support predicts opposition gains (β=0.075)

**Caveats:** Error-correction within government periods may conflate thermostatic
correction with mean reversion in polling noise. Causal direction ambiguous —
negative feedback could reflect genuine public response or polling methodology.

## Methodology

- **Data**: 1,016 party-vote polls scraped from Wikipedia (1993-2026)
- **Economic data**: World Bank API (GDP growth, unemployment, CPI), annual 1960-2024,
  cubic-spline interpolated to monthly for poll-level matching
- **Weekly series**: National and Labour resampled to weekly frequency via linear interpolation
- **Government fixed effects**: All regressions include government-period dummies to avoid
  confounding between-government variation with within-government trends
- **Regime switching**: 2-regime Markov model (statsmodels) with switching variance
- **Bootstrap**: 10,000 resamples for movement concentration null distribution

## Data Limitations

- **Polling accuracy**: polls have house effects and sampling error (~3% margin)
- **Annual economic data**: spline interpolation adds smoothness that may not reflect
  within-year economic shocks
- **Small N for events**: only 4 mid-term leadership changes, 4 crises, ~11 elections
- **Survivorship in minor parties**: parties that disappeared (Alliance, United Future)
  are not tracked, biasing lifecycle analysis
- **No causal identification**: all findings are correlational

---

*Report generated by nz_rules.py*