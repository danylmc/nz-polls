# Adversarial Critique: NZ Polling Analysis Project

*A systematic challenge to the 21 rules, 30 analyses, and overall methodology.*

---

## I. Fundamental Methodological Problems

### 1. The Multiple Comparisons Problem (p-hacking risk)

The project runs **30 hypothesis tests** across overlapping datasets with correlated variables, yet applies no correction for multiple comparisons (Bonferroni, Benjamini-Hochberg, or similar). At α=0.05 with 30 tests, you'd expect ~1.5 spurious significant results by chance alone.

Worse, many tests aren't pre-registered — the hypotheses were formulated *after* exploring the data. The project explicitly describes an iterative workflow: run analysis, find pattern, write it up as a "confirmed hypothesis." This is textbook HARKing (Hypothesising After Results are Known). The impressive hit rate (22/30 confirmed or partially confirmed) may partly reflect the researcher's freedom to define what counts as "confirmed" and which tests to report.

**Most vulnerable findings:** The issue salience correlations (Analysis 10), where 10 issues × 5 parties × 2 methods = 100 correlations were tested, but only the significant ones are highlighted.

### 2. Tiny N for the Most Important Claims

Several headline findings rest on dangerously small samples:

| Finding | N | Effective degrees of freedom |
|---------|---|-----|
| Myopic retrospection (Analysis 29) | 9 elections | 7 |
| Cost of ruling / Paldam (Analysis 19) | 10 govt periods | 8 |
| Economic voting macro (Analysis 7) | 11 elections | 8 |
| Ipsos approval → vote (Analysis 9) | 29 months | 27 |
| Party capability (Analysis 11) | 10 waves | 8 |
| Crime → salience (Analysis 24) | 21 matched obs | 19 |
| Google Trends (Analysis 25) | ~30 months | ~28 |
| Multivariate model (Analysis 27) | 24 obs, 7 predictors | 16 |

With N=9, a correlation of r=0.588 (myopic retrospection) has a confidence interval stretching roughly from -0.10 to +0.90. The finding is directionally suggestive but claiming it as evidence for Achen & Bartels when p=0.096 is generous interpretation. Similarly, rejecting Paldam (1991) based on N=10 government periods (with only 1 majority government as the comparison case) is not statistically defensible.

The multivariate model (Analysis 27) with 7 predictors and N=24 has an observation-to-predictor ratio of 3.4:1 — well below the commonly recommended minimum of 10:1 or even 15:1. The "massive multicollinearity" finding (VIF >30) is real, but it may partly reflect inadequate sample size rather than genuine redundancy.

### 3. Non-Independent Observations

The project treats monthly polling observations as independent data points, but they are clearly autocorrelated (the report itself notes lag-1 autocorrelation of 0.90). This inflates the effective N and deflates p-values for every time-series correlation reported. A correlation computed across 250 monthly observations with ρ₁=0.90 has an effective sample size closer to ~25. Most of the "p < 0.001" results would lose significance under proper correction for serial dependence (Newey-West standard errors, or working with first-differenced data throughout).

The project *sometimes* reports first-differenced correlations, but inconsistently. For rental inflation (Analysis 26), the levels correlation is r=-0.731 (headline finding), but the change correlations are "all non-significant." This should prompt the question: is the levels result spurious, driven by two trending time series? The project dismisses this by noting the level matters more than the change — but this is exactly what you'd expect from a spurious regression.

### 4. Ecological Fallacy

The project constantly moves between aggregate-level findings (polling averages) and individual-level claims (voter behaviour). Rules like "voters blame the government" and "voters are backward-looking" are claims about individual cognition, but most evidence comes from aggregate correlations between economic indicators and polling averages. The ecological fallacy warns that aggregate relationships need not hold at the individual level, and vice versa.

The NZES analyses (21-23) do use individual-level data, but even these are cross-sectional snapshots within each election — they cannot establish within-person change over time, only between-person differences at a single point.

---

## II. Specific Rule Challenges

### Rule 1: "The causal chain is economic conditions → issue salience → government approval → vote intention"

**Challenge:** The mediation analysis (Sobel test) establishing this chain uses cross-sectional data with N=29. Baron & Kenny mediation requires temporal precedence that simultaneous measurement cannot provide. All three variables (mortgage rate, approval, vote) are measured at approximately the same time. The "68% mediated" claim assumes a causal direction that the data cannot confirm — it's equally consistent with:
- Voters who've already decided to oppose the government report lower approval *and* cite cost of living
- Media coverage of bad polls increases cost-of-living coverage, which raises salience

The mediation could run in reverse: opposition voters become more sensitive to cost-of-living information (confirmation bias), which inflates the apparent mediation.

### Rule 4: "Subjective perception beats objective reality"

**Challenge:** This may be a measurement artifact. GDP/CPI/unemployment are measured quarterly or annually at the national level, while approval is measured monthly and captures contemporaneous voter sentiment. The comparison is between a lagged, coarse, national aggregate and a real-time sentiment measure. Of course the sentiment measure correlates better with another sentiment measure (voting intention). This doesn't prove "perception beats reality" — it may just prove "surveys correlate with surveys better than surveys correlate with administrative data at different temporal resolutions."

A fairer test would use high-frequency objective indicators (weekly petrol prices, monthly CPI components) at the same resolution as polling. When the project *does* this (petrol, mortgages, rents), the objective indicators *do* significantly predict polling. The "reality doesn't matter" claim is overstated.

### Rule 6: "The same indicator can mean different things in different regimes"

**Challenge:** This "finding" may be an ad hoc rescue of a failing model. When mortgage rates and polling show the expected negative relationship under Labour (r=-0.814), it's pocketbook punishment. When they show the *opposite* relationship under National (r=+0.829), it's "because falling rates signal recession." But this interpretation is unfalsifiable — any direction of correlation can be explained post hoc. What would *disconfirm* this theory?

The National-period sample is also tiny (10 observations over ~2 years). A correlation of r=+0.829 with N=10 has wide confidence intervals and could easily be driven by one or two unusual months.

### Rule 9: "PM preference tracks party vote but doesn't lead it"

**Challenge:** The "no presidentialisation" conclusion requires accepting a null result from a test with limited statistical power. Cross-correlations between monthly time series are noisy, and the PM preference dataset (299 polls, but only 148 matched months) may lack the temporal resolution to detect lead/lag relationships that operate over weeks rather than months. The absence of evidence is not evidence of absence.

### Rule 17: "Rental inflation is the strongest pocketbook indicator"

**Challenge:** This is the most suspiciously strong correlation in the entire project (r=-0.731). Several red flags:

1. **Spurious regression risk:** Both rental CPI and incumbent vote share are persistent time series. Two trending series will produce high correlations regardless of causal connection. The fact that *change* correlations are "all non-significant" is a major warning sign that this is a levels-levels spurious regression.

2. **Confounding by era:** Rental inflation has been high during the 2020-2025 period, which is also when incumbents have struggled (Labour post-COVID, then National). The correlation may simply reflect that "recent years have both high rents and unpopular governments" without any causal link.

3. **No Granger causality test reported:** Unlike mortgages and petrol (which passed Granger causality), rental inflation was not subjected to this more rigorous temporal test. The absence is conspicuous.

4. **No mediation analysis:** If rents truly work through the same causal chain as other pocketbook indicators, we should see mediation through approval. This wasn't tested.

### Rule 19: "Polling is a random walk"

**Challenge:** This may be the most important finding, but it also undermines the rest of the project. If polling is truly a random walk where no indicator predicts changes, then *all* the correlations in Rules 1-18 are either:
- Level-level relationships between trending time series (spurious)
- Contemporaneous associations that describe but don't predict
- Explaining variance in levels, not changes (useless for forecasting)

The project doesn't adequately reconcile this tension. You can't simultaneously claim "mortgage rates Granger-cause incumbent vote shifts" (Rule 1) and "no indicator predicts quarter-to-quarter vote changes" (Rule 19). The Granger causality tests use monthly data with short lags, while the forecasting test uses quarterly horizons — but if the effect is real and persistent, it should show up at quarterly resolution too.

### Rule 20: "Voters only remember the election year"

**Challenge:** With N=9 and p=0.096, this is not a confirmed finding by any conventional standard. The project labels it "Partial" but then promotes it to a "Rule." The correlation r=+0.588 between election-year GDP and incumbent fate could easily be driven by 2-3 elections (2008 and 2023 are obvious candidates). Remove those and the relationship may vanish. With 9 data points, there's no way to assess robustness.

### Rule 21: "Demographics predict vote through ethnic identity, not class"

**Challenge:** McFadden's pseudo-R² values are extremely low throughout (0.02-0.07). Even the "strongest" demographic model explains only 7% of the variance in vote choice. The finding that demographics don't explain much is correct; the finding that the *composition* of what matters has shifted requires interpreting year-to-year fluctuations in very small pseudo-R² differences (e.g., 0.032 vs 0.044), which may be noise.

The Maori identity finding is more robust, but the variable definition changes across NZES waves (ethnicity vs ancestry vs identity), introducing measurement inconsistency.

---

## III. Data Quality Concerns

### 1. Wikipedia as Primary Source

All 1,020 polls were scraped from Wikipedia. Wikipedia is a secondary source that relies on volunteer editors accurately transcribing polling results. Potential issues:
- **Selective inclusion:** Not all polls may be listed. Obscure pollsters or embarrassing results for editors' preferred parties may be omitted.
- **Transcription errors:** No systematic cross-checking against original pollster releases was performed.
- **Survivor bias:** Polling firms that went out of business may be underrepresented in older data.

### 2. Ipsos Data Extraction

The Ipsos data (crucial for Analyses 9-11, 13, 24-27) was manually extracted from PDFs, including reading values from chart labels. The project itself notes: "Historical values in later editions may differ from original editions (methodology revisions)" and "chart labels come out scrambled." This introduces unknown measurement error into the most important predictor in the entire framework (government approval).

With only 24-30 observations of the key Ipsos variables, even small extraction errors could meaningfully shift correlations.

### 3. No Out-of-Sample Validation

Every finding is in-sample. The project uses the same data to discover patterns and to report them as "rules." The one attempt at out-of-sample validation (Analysis 28, forecasting) showed that models fail — which is exactly what you'd expect when in-sample findings don't generalise.

### 4. Mixed Temporal Resolutions

The project freely mixes annual (migration, GDP), quarterly (house prices, rental CPI), monthly (confidence, mortgages, petrol), and irregular (Ipsos waves, polls) data. Interpolation and aggregation to match these create artificial smoothing that can inflate correlations.

---

## IV. Interpretive Overreach

### 1. Correlation Language vs Causal Claims

The project oscillates between careful correlational language ("associated with") and strong causal claims ("voters blame," "punishment flows," "the mechanism is"). Rules 1-3 are explicitly causal statements derived from observational data. Even Granger causality is merely temporal precedence, not true causation — it could reflect a shared cause (media coverage) that affects both the predictor and the outcome at different speeds.

### 2. Confirmation Bias in Literature Matching

The project frames results as "confirming" or "disconfirming" specific political science theories. But the tests applied are often quite different from what the original theorists proposed:

- **Fiorina (1981):** Tested in NZ using NZES cross-sectional correlations, not panel data tracking the same voters over time. The original theory is about individual learning, not aggregate cross-sections.
- **Achen & Bartels (2016):** Their "myopia" thesis was tested with much larger datasets (US presidential elections 1948-2012, with state-level variation providing hundreds of observations). Testing it with N=9 NZ elections is not comparable.
- **Cox (1997):** Strategic voting theory is about individual decision-making under FPTP. The project tests it under MMP, a fundamentally different electoral system. The "not confirmed" result may simply mean the theory was applied to the wrong institutional context.
- **Paldam (1991):** Tested across hundreds of elections in dozens of democracies. Rejecting it based on 10 NZ government periods (with 1 majority government) is not a meaningful test.

### 3. The "21 Rules" Framing

Calling findings "rules" implies they are stable, generalisable regularities. But most are derived from a single country over a 30-year period with 10 elections. Political science "rules" typically require cross-national or cross-temporal replication. Some of these "rules" may be specific to the 1996-2023 MMP era in New Zealand and break down under different conditions (different media environment, different party system configuration, different economic structure).

### 4. Selective Emphasis

The project emphasises confirmed hypotheses (22/30) and de-emphasises null findings. But several nulls are arguably more important:
- **Macro-economic voting doesn't work** (Analysis 7) undermines the theoretical foundation of pocketbook voting
- **Forecasting doesn't work** (Analysis 28) undermines the practical value of all other findings
- **House prices don't matter** (Analysis 18) contradicts the pocketbook framework for NZ's most politically salient economic issue

The project resolves these tensions through auxiliary hypotheses ("perception matters more than reality," "wealth and affordability cancel out"), but these are unfalsifiable ad hoc explanations.

---

## V. Missing Controls and Confounders

### 1. No Media Controls

The entire "issue salience" framework assumes salience is driven by objective conditions. But issue salience is largely driven by media coverage, which is itself driven by editorial decisions, news cycles, and political strategy. Without controlling for media coverage intensity, the project cannot distinguish between:
- Economic conditions → public concern → polling shift
- Media emphasis on economic conditions → simultaneous concern and polling shift

### 2. No Government Policy Controls

The project doesn't account for government actions that simultaneously affect economic indicators AND polling. For example:
- A government introduces a controversial policy → polling drops AND economic confidence drops
- The correlation between confidence and polling is confounded by the policy, not causal

### 3. No Opposition Strategy Controls

Opposition parties respond to economic conditions by emphasising cost-of-living issues. The issue salience → polling shift relationship may partly reflect effective opposition messaging rather than voter organic response to conditions.

### 4. Global Trends

NZ is a small open economy heavily influenced by global trends. Global commodity prices drive petrol costs; global interest rates influence NZ mortgage rates; global inflation affects CPI. The project doesn't attempt to separate domestic political responsibility from global economic forces — which is ironic given that Rule 2 claims "voters blame the government for things it doesn't control."

---

## VI. The Self-Undermining Problem

The project's most defensible finding — that polling follows a random walk (Rule 19) — logically undermines its other 20 rules. If no indicator predicts future polling changes, then the "rules" describe contemporaneous associations, not dynamic causal processes. They tell you what the world looks like when polls are high or low, but they can't tell you what will make polls go up or down.

This creates a paradox: the project's descriptive framework is rich and internally consistent, but its own forecasting test suggests the framework has no predictive power beyond "polls will probably be about the same next quarter." The 21 rules may be an elaborate description of how political variables co-move, not a set of causal mechanisms that can be intervened upon.

The honest conclusion might be: **we've built a detailed map of what NZ politics looks like, but we haven't found a compass.**

---

## VII. Summary of Weakest and Strongest Findings

### Most Robust (hard to challenge)
1. **National-Labour zero-sum** (large N, simple test, both levels and changes significant)
2. **Poll accuracy and convergence** (direct validation against election results)
3. **Green overestimation / NZ First underestimation** (consistent across elections)
4. **L-R proximity predicts vote** (individual-level, large N, multiple elections)
5. **Retrospective > prospective voting** (individual-level, consistent across elections)

### Most Vulnerable (easy to challenge)
1. **Rental inflation as strongest pocketbook indicator** (likely spurious regression, no Granger test, changes n.s.)
2. **Myopic retrospection** (N=9, p=0.096, promoted to "Rule")
3. **The causal chain** (cross-sectional mediation, reverse causation plausible)
4. **Regime-dependent mortgage effects** (ad hoc, N=10 for National period)
5. **Paldam rejection** (N=10, only 1 majority government as comparison)
6. **Dealignment findings** (pseudo-R² < 0.07, variable definitions change across waves)

---

*This critique does not claim the project's findings are wrong — many are directionally plausible and consistent with international evidence. The critique identifies where the evidence is weaker than presented, where alternative explanations haven't been excluded, and where the framing overstates what the data can support.*
