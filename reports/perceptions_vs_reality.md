# Economic Perceptions vs Reality

*Do voters accurately perceive the economy? Does reality or perception drive voting?*

## 6a. Perception-Reality Gap

| Year | Mean Perception (raw) | Actual GDP (y/y) | Actual Inflation |
|------|----------------------|-----------------|-----------------|
| 1999 | 3.20 | 4.3% | -0.1% |
| 2005 | 2.98 | 2.4% | 2.9% |
| 2008 | 4.24 | 1.1% | 3.9% |
| 2011 | 3.68 | 0.3% | 4.6% |
| 2014 | 2.74 | 2.5% | 1.4% |
| 2017 | 2.85 | 3.6% | 1.8% |
| 2020 | 3.92 | -0.8% | 1.8% |
| 2023 | 4.02 | 2.9% | 6.4% |

![Perception vs Reality](../graphs/perceptions_vs_reality.png)

## 6b. Partisan Perceptual Screen

| Year | Incumbent | Govt Supporters | Opposition | Gap | p-value |
|------|-----------|-----------------|------------|-----|---------|
| 1999 | National | 2.80 | 3.36 | 0.56 | 0.0000** |
| 2005 | Labour | 2.72 | 3.15 | 0.42 | 0.0000** |
| 2008 | Labour | 4.03 | 4.36 | 0.34 | 0.0000** |
| 2011 | National | 3.28 | 3.96 | 0.68 | 0.0000** |
| 2014 | National | 2.16 | 3.17 | 1.01 | 0.0000** |
| 2017 | National | 2.24 | 3.24 | 1.00 | 0.0000** |
| 2020 | Labour | 3.79 | 4.09 | 0.30 | 0.0000** |
| 2023 | Labour | 3.47 | 4.23 | 0.76 | 0.0000** |

![Partisan Filter](../graphs/perceptions_partisan.png)

## 6c. Perception vs Reality: Which Drives Voting?

| Model | Pseudo R² | AIC | Key Finding |
|-------|-----------|-----|-------------|
| Actual Only | 0.0072 | 27094 | |
| Perceived Only | 0.0258 | 17463 | |
| Both | 0.0365 | 17274 | |

### Full Model Coefficients

| Variable | Coefficient | Std Error | p-value |
|----------|-------------|-----------|---------|
| const | 0.142 | 0.095 | 0.1346 |
| econ_change | -0.312 | 0.018 | 0.0000** |
| actual_gdp | 0.083 | 0.011 | 0.0000** |
| actual_inflation | 0.143 | 0.011 | 0.0000** |
| age | 0.008 | 0.001 | 0.0000** |
| female | -0.236 | 0.037 | 0.0000** |

![Model Comparison](../graphs/perceptions_model_comparison.png)
