# Voter Flow Analysis: Partial Correlations & Compositional Data

## Motivation

Simple pairwise correlations on vote share data are misleading because
party shares are **compositional** (they sum to ~100%). A decline in one
party mechanically inflates others, creating spurious negative correlations.
Three methods address this:

1. **Partial correlations** - control for all other parties
2. **Log-return correlations** - correlate proportional changes Δln(xi),
   which are free from the constant-sum constraint
3. **Variation matrix** (Aitchison, 1986) - var(ln(xi/xj)) measures how
   much the ratio between two parties fluctuates

## 1. Raw Change Correlations (for reference)

Poll-to-poll changes in raw percentage points (n = 880).
These are **biased** by the compositional constraint.

```
                National      Labour       Green         ACT    NZ First
------------------------------------------------------------------------
    National   1.000     -0.333     -0.401     -0.277     -0.417   
      Labour  -0.333      1.000     -0.419     -0.336     -0.122   
       Green  -0.401     -0.419      1.000      0.197      0.110   
         ACT  -0.277     -0.336      0.197      1.000      0.080   
    NZ First  -0.417     -0.122      0.110      0.080      1.000   
```

## 2. Partial Correlations

Each correlation controls for changes in all other parties (n = 880).

```
                National      Labour       Green         ACT    NZ First
------------------------------------------------------------------------
    National   1.000     -0.813***  -0.753***  -0.603***  -0.653***
      Labour  -0.813***   1.000     -0.756***  -0.621***  -0.562***
       Green  -0.753***  -0.756***   1.000     -0.421***  -0.461***
         ACT  -0.603***  -0.621***  -0.421***   1.000     -0.372***
    NZ First  -0.653***  -0.562***  -0.461***  -0.372***   1.000   

Significance: * p<0.05, ** p<0.01, *** p<0.001
```

**Caveat:** All partial correlations are strongly negative. With 5 party
changes that approximately sum to zero, conditioning on the other 3 parties
nearly determines the sum of the remaining 2, forcing a negative correlation.
The relative ordering is informative (National-Labour is the most negative,
ACT-NZ First the least), but absolute values are artifacts of the closure.

## 3. Log-Return Correlations (primary method)

Correlations of Δln(xi) — proportional changes (n = 880).
Unlike raw changes, log-returns **do not** sum to a constant, so these
correlations are free from compositional bias.

Interpretation requires political context (Left bloc: Labour, Green;
Right bloc: National, ACT, NZ First):
- **Within-bloc negative** = direct voter exchange (e.g. Labour ↔ Green)
- **Cross-bloc negative** = opposing tides, not direct flow (e.g. National ↔ Green)
- **Positive** = co-movement from shared external drivers

```
                National      Labour       Green         ACT    NZ First
------------------------------------------------------------------------
    National   1.000     -0.270***  -0.390***  -0.218***  -0.344***
      Labour  -0.270***   1.000     -0.350***  -0.134***  -0.108** 
       Green  -0.390***  -0.350***   1.000      0.160***   0.135***
         ACT  -0.218***  -0.134***   0.160***   1.000      0.090** 
    NZ First  -0.344***  -0.108**    0.135***   0.090**    1.000   

Significance: * p<0.05, ** p<0.01, *** p<0.001
```

## 4. Variation Matrix — var(ln(xi/xj))

Aitchison's standard measure for compositional data. Each cell shows
how much the log-ratio between two parties fluctuates between consecutive
polls. **Low** = stable ratio (co-movement); **High** = volatile ratio
(potential voter exchange or independent shocks).

```
                National      Labour       Green         ACT    NZ First
------------------------------------------------------------------------
    National  0.0000     0.0250     0.1234     0.5912     0.2088   
      Labour  0.0250     0.0000     0.1217     0.5795     0.1895   
       Green  0.1234     0.1217     0.0000     0.5694     0.2294   
         ACT  0.5912     0.5795     0.5694     0.0000     0.6626   
    NZ First  0.2088     0.1895     0.2294     0.6626     0.0000   
```

## 5. Pairwise Relationships (ranked by log-return correlation)

Bloc key: L = Left (Labour, Green), R = Right (National, ACT, NZ First)

| Rank | Party Pair | Bloc | Log-ret r | p-value | Variation | Interpretation |
|------|-----------|------|-----------|---------|-----------|----------------|
| 1 | National <-> Green | cross | -0.390*** | 0.0000 | 0.1234 | Cross-bloc opposing tides |
| 2 | Labour <-> Green | within | -0.350*** | 0.0000 | 0.1217 | Within-bloc voter exchange |
| 3 | National <-> NZ First | within | -0.344*** | 0.0000 | 0.2088 | Within-bloc voter exchange |
| 4 | National <-> Labour | cross | -0.270*** | 0.0000 | 0.0250 | Swing voter corridor |
| 5 | National <-> ACT | within | -0.218*** | 0.0000 | 0.5912 | Within-bloc voter exchange |
| 6 | Labour <-> ACT | cross | -0.134*** | 0.0001 | 0.5795 | Cross-bloc opposing tides |
| 7 | Labour <-> NZ First | cross | -0.108** | 0.0014 | 0.1895 | Cross-bloc opposing tides |
| 8 | ACT <-> NZ First | within | 0.090** | 0.0078 | 0.6626 | Minor party co-movement |
| 9 | Green <-> NZ First | cross | 0.135*** | 0.0001 | 0.2294 | Minor party co-movement |
| 10 | Green <-> ACT | cross | 0.160*** | 0.0000 | 0.5694 | Minor party co-movement |

## 6. Key Findings

### Within-bloc voter exchange (direct competition for same voters):

- **Labour <-> Green**: r = -0.350 (p = 0.0000), τ = 0.1217
  Direct competition within the left bloc. When Labour
  gains, Green tends to lose — consistent with voters shifting
  between ideologically adjacent parties.
- **National <-> NZ First**: r = -0.344 (p = 0.0000), τ = 0.2088
  Direct competition within the right bloc. When National
  gains, NZ First tends to lose — consistent with voters shifting
  between ideologically adjacent parties.
- **National <-> ACT**: r = -0.218 (p = 0.0000), τ = 0.5912
  Direct competition within the right bloc. When National
  gains, ACT tends to lose — consistent with voters shifting
  between ideologically adjacent parties.

### Swing voter corridor:

- **National <-> Labour**: r = -0.270 (p = 0.0000), τ = 0.0250
  The two major parties compete for centrist swing voters.
  The very low variation (τ = 0.0250) indicates their
  combined share is highly stable — the overall left-right
  balance shifts slowly.

### Cross-bloc opposing tides (not direct voter exchange):

These pairs move inversely because they sit on opposite sides
of the left-right divide. When the political environment favours
one bloc, the other loses — but voters typically flow through
intermediaries (e.g., Green → Labour → swing → National), not
directly between ideologically distant parties.

- **National <-> Green**: r = -0.390 (p = 0.0000)
- **Labour <-> ACT**: r = -0.134 (p = 0.0001)
- **Labour <-> NZ First**: r = -0.108 (p = 0.0014)

### Minor party co-movement:

These pairs rise and fall together. When major parties dominate
polling, all minor parties tend to be squeezed simultaneously;
when dissatisfaction with major parties rises, minor parties
benefit collectively.

- **ACT <-> NZ First**: r = 0.090 (p = 0.0078)
- **Green <-> NZ First**: r = 0.135 (p = 0.0001)
- **Green <-> ACT**: r = 0.160 (p = 0.0000)

## Methodological Notes

- **Partial correlations** use the precision matrix (inverse of the
  correlation matrix). With near-compositional data, all partial
  correlations tend negative (Aitchison, 1986). Included for
  completeness; log-return correlations are the primary measure.
- **Log-return correlations** correlate Δln(xi) between consecutive
  polls. Since log-returns do not sum to a constant, these are free
  from compositional bias. This is analogous to correlating asset
  returns in finance — a well-established approach for proportional data.
- **Variation matrix** entries τ(i,j) = var(Δln(xi/xj)). This is
  Aitchison's standard pairwise association measure for compositional
  data and does not suffer from CLR singularity issues.
- Changes are computed **within election cycles only** to avoid
  cross-election discontinuities.
- Only polls reporting all 5 major parties are included.
- Shares of 0% are replaced with 0.01% before taking logs.

---

*Report generated by voter_flows.py*