# Response to Adversarial Critique

*A point-by-point defence of the methodology, findings, and interpretive framework.*

---

## I. Response to Fundamental Methodological Problems

### 1. Multiple Comparisons — Acknowledged but Overstated

The critique correctly identifies that 30 tests without formal correction creates multiple comparison risk. However, several mitigating factors are ignored:

**The tests are not independent.** The critique calculates 1.5 expected false positives assuming 30 independent tests at α=0.05. But our tests are deliberately structured around a single theoretical framework (the causal chain: economic conditions → salience → approval → vote). Many "tests" are simply the same relationship examined from different angles — mortgage rates, petrol prices, and consumer confidence are all measuring pocketbook stress through different channels. The effective number of independent hypotheses is closer to 8-10 (polling dynamics, pocketbook voting, perception vs reality, issue ownership, individual-level voting, forecasting), not 30.

**The HARKing charge is partially unfair.** The research plan (`furtherresearch.md`) was written before most analyses were run, specifying which theories to test and which datasets to use. The plan file in the project explicitly lists priorities A1-A8 and B1-B9 with pre-specified hypotheses from the political science literature (Fiorina, MacKuen, Stokes, Cox, Paldam, Dalton, Achen & Bartels). We didn't go fishing for patterns — we tested specific, published theories against NZ data.

**Eight of 30 tests returned null results, and we reported them prominently.** If we were HARKing, we would have found ways to make these significant or simply not reported them. Instead, the project highlights macro-economic voting (null), house prices (null), Paldam coalition decay (null), forecasting (null), shy Tory (reversed), presidentialisation (null), strategic desertion (not confirmed as predicted), and dealignment (not confirmed). A genuine p-hacking exercise would not produce a 27% null rate.

**Regarding Analysis 10 (issue salience):** The critique claims "100 correlations were tested, but only the significant ones are highlighted." This is incorrect. The findings report includes the *complete* correlation table for all 10 issues × 2 major parties (20 correlations), with both levels and first-differenced versions (40 total). Every correlation is reported, including non-significant ones. The issue ownership interpretation is based on the pattern of which issues help which parties, not on cherry-picked significant results.

### 2. Small N — Acknowledged as a Genuine Limitation

The critique's strongest point. We concede:

- **Myopic retrospection (N=9):** The report already labels this "Partial" and notes p=0.096. The finding is directionally consistent with Achen & Bartels (2016), tested with much larger datasets internationally. We present it as suggestive, not definitive.
- **Paldam test (N=10):** We reject Paldam's prediction, but we don't claim to have *proven* coalitions are better — we claim the NZ data offers no support for the hypothesis that they're worse. That's a meaningful empirical contribution even at N=10.
- **Multivariate model (N=24, 7 predictors):** The report itself identifies the multicollinearity problem and concludes that the kitchen sink model is overfitted. The *recommended* model uses only 2 predictors (approval + petrol) with N=24, giving a ratio of 12:1.

However, the critique overstates the problem for several findings:

- **Ipsos approval → vote (N=29):** With r=0.827, this would require a 95% CI of approximately ±0.15, giving a lower bound around r=0.67. This is not a fragile finding.
- **Granger causality tests (mortgage, petrol, confidence):** These use 238-369 monthly observations, not 29. The small N criticism applies to the Ipsos-based analyses but not to the time-series econometric tests.

### 3. Non-Independent Observations — Partially Addressed in the Analysis

The critique is correct that lag-1 autocorrelation of 0.90 inflates effective N for level correlations. But the project explicitly addresses this in multiple ways:

**First-differenced correlations are reported throughout.** Every major analysis reports both level and change correlations. The mortgage analysis reports contemporaneous Δr=-0.067 (n.s.) alongside the 1-month lagged Δr=-0.209 (p=0.001) and quarterly Δr=-0.307 (p<0.001). The petrol analysis reports Δr=-0.193 (p=0.002). Consumer confidence reports Δr=+0.124 (p=0.017). These first-differenced results survive the autocorrelation critique.

**Granger causality tests are specifically designed for autocorrelated time series.** The Granger tests use lagged first-differences and test whether past changes in X predict future changes in Y *beyond what past changes in Y predict*. The F-statistics (13.8 for mortgage, 12.0 for petrol, 9.3 for confidence) are robust to serial correlation by construction.

**The rental inflation critique has merit.** We acknowledge that the r=-0.731 level correlation between rental CPI and incumbent vote is the most vulnerable to the spurious regression charge. The change correlations are indeed non-significant. However, the level correlation is not between two unit-root trending series in the classical spurious regression sense — rental CPI is I(1) but incumbent vote share is bounded and mean-reverting (as our own mean reversion analysis confirms). The correlation may reflect that persistent periods of high rental inflation coincide with persistent periods of incumbent unpopularity, which is a substantively meaningful finding even if it doesn't imply quarter-to-quarter causation.

### 4. Ecological Fallacy — Partially Addressed by Mixed Methods

The project explicitly uses both aggregate-level time-series data (Analyses 1-20, 24-29) and individual-level NZES survey data (Analyses 21-23). The NZES analyses confirm at the individual level what the aggregate data suggests:

- Individual voters who perceive the economy negatively are more likely to vote against the incumbent (Analysis 21, r=0.376, N=2000+ per election)
- Individual voters choose parties closest to them ideologically (Analysis 22, r=0.698)
- Individual voters split their tickets strategically under MMP (Analysis 23, ~30%)

The aggregate findings are *consistent with* the individual-level findings, which is the strongest evidence available short of panel data. The critique is correct that cross-sectional NZES data cannot establish within-person change — but this is a limitation of all available NZ election surveys, not a flaw specific to this project.

---

## II. Response to Specific Rule Challenges

### Rule 1: The Causal Chain

The critique raises legitimate concerns about cross-sectional mediation. Our response:

**The temporal ordering IS established, just not within the Sobel test.** The Granger causality tests (Analyses 12, 14, 15) establish that mortgage rate changes *temporally precede* incumbent vote changes by 1-3 months. The mediation analysis then decomposes the *size* of this pre-established temporal relationship into direct and indirect paths through approval. The Sobel test is used to quantify the proportion mediated, not to establish temporal precedence.

**The reverse causation story is implausible.** The critique suggests "voters who've already decided to oppose the government report lower approval *and* cite cost of living." But mortgage rates and petrol prices are *objective, externally determined variables*. The RBNZ sets the OCR; global oil markets set petrol prices. Voters cannot cause mortgage rates to rise by disliking the government. The exogeneity of the economic trigger is what makes the causal interpretation defensible.

**Media confounding is possible but doesn't invalidate the finding.** Even if media coverage mediates between economic conditions and voter concern, the chain remains: economic conditions → (media) → salience → approval → vote. The media is a transmission mechanism, not an alternative explanation.

### Rule 4: "Subjective Perception Beats Objective Reality"

The critique makes an excellent point about temporal resolution mismatch. We partially concede:

**The phrasing is too strong.** A fairer statement is: "Survey-measured perceptions of economic conditions predict voting better than administrative economic statistics at their available temporal resolutions." The critique correctly notes that when we *do* use high-frequency objective indicators (petrol, mortgages), they significantly predict polling.

**However, the core finding survives.** GDP, unemployment, and CPI — the macro fundamentals that dominate economic voting literature — genuinely have no relationship with NZ incumbent polling (Analysis 7, all p > 0.05, R²=0.083). This isn't just a resolution problem: even annual GDP growth doesn't predict annual polling changes (Analysis 29, full-term r=+0.380, p=0.314). The finding that *specific, visible* costs matter while *aggregate* indicators don't is itself substantively important.

### Rule 6: Regime-Dependent Mortgage Effects

The critique calls this "ad hoc" and "unfalsifiable." We disagree:

**The finding IS falsifiable.** It predicts that under a future Labour government, rising mortgage rates will hurt Labour, while under a future National government, falling rates will signal recession and hurt National. If a future Labour government sees rising rates *helping* their polling, or a National government sees rate cuts *boosting* their polling, the theory would be falsified.

**The economic logic is straightforward.** The RBNZ cut rates from 5.5% to 3.75% during 2024-2025 specifically because the economy was in recession. Falling rates were *caused by* economic distress. It would be strange if voters celebrated rate cuts that were occurring *because* they were losing their jobs. The regime dependence isn't an ad hoc rescue — it reflects the widely understood fact that rate cuts during recessions are a symptom, not a cure.

**The N=10 concern is valid** but the effect size is very large (r=+0.829). Even with wide confidence intervals, the direction is clear and the economic mechanism is transparent.

### Rule 9: PM Preference Doesn't Lead Party Vote

The critique argues we're accepting a null with limited power. Fair point, but:

**The null is informative.** If presidentialisation were a strong effect (as Poguntke & Webb argue), it should be detectable even with monthly resolution. The cross-correlation at lag 0 is r=0.264, while all leads and lags are weaker. The *pattern* of results — highest at lag 0, declining symmetrically in both directions — is the signature of a contemporaneous relationship, not a failed detection of a lead/lag effect.

**The 2017 Ardern change provides a natural experiment.** The +8.9pp Labour bounce coincided exactly with the leadership change, not with a prior rise in PM preference polling. This is consistent with contemporaneous movement rather than PM popularity leading.

### Rule 17: Rental Inflation — CRITIQUE VALIDATED

**Update:** Following the critique, we ran the missing Granger causality test and Newey-West HAC standard errors. **The critique was correct.** Rental inflation:

1. **Fails Granger causality** at all lags 1-6 (all p > 0.27, N=99 quarters). For comparison, mortgage rates pass at F=13.8, p=0.0003 and petrol at F=12.0, p=0.0006.
2. **Loses significance under Newey-West** (OLS p=0.001 → NW p=0.112). The autocorrelation-corrected standard errors reveal the level correlation was inflated.
3. Change correlations remain non-significant (NW p=0.854).

The finding has been **demoted from "strongest pocketbook indicator" to "associational finding only"** in findings.md and the guide. The level correlation (r=-0.731) likely reflects contemporaneous co-movement between two persistent series, not a causal pocketbook effect.

What survives: the rental-mortgage independence (r=0.052) is a valid finding about separate demographic channels, and the descriptive association between high-rent periods and incumbent unpopularity is real — it just isn't causal in the way we originally claimed.

### Rule 19: Polling as a Random Walk

The critique presents this as "self-undermining" — if polling is a random walk, the other rules are vacuous. We disagree:

**The random walk finding applies to *forecasting*, not to *explanation*.** Rules 1-18 are explanatory: they describe what moves contemporaneously with polls and identify the causal mechanisms. Rule 19 says you can't *predict* future changes from current information. These are logically compatible:

- "When petrol prices rise, incumbent polls fall" (contemporaneous explanation) — confirmed
- "Knowing today's petrol price doesn't tell you where polls will be next quarter" (prediction) — also confirmed

The reconciliation is simple: you can't predict polls because you can't predict *petrol prices, economic shocks, or scandals*. If you could predict that petrol would spike next month, Rules 1-18 would tell you what would happen to polls. The rules describe the *transfer function*, while Rule 19 says the *inputs* are unpredictable.

**This is exactly how efficient markets work.** Stock prices respond to earnings, interest rates, and news — but you can't predict future stock prices because you can't predict future earnings, rates, and news. The existence of a response function doesn't contradict the random walk. Rules 1-18 are the response function; Rule 19 is the unpredictability of future shocks.

**The Granger causality results do NOT contradict Rule 19.** Granger tests ask: "does past X help predict future Y, controlling for past Y?" The mortgage Granger test (F=13.8) says yes for 1-month horizons. The forecasting test (Analysis 28) asks: "can you beat the naïve baseline at quarterly horizons?" These are different questions at different time scales. Short-term (1-month) predictability from specific economic shocks is compatible with quarterly-horizon unpredictability, because the shocks themselves are unpredictable and their effects are absorbed within 1-2 months.

### Rule 20: Myopic Retrospection

**We agree this is among the weakest findings.** The report labels it "Partial" and notes the marginal significance. The "Rule" framing is too strong for N=9 with p=0.096. We would reclassify this as a "suggestive pattern consistent with international evidence" rather than a confirmed rule.

### Rule 21: Dealignment

The critique notes that pseudo-R² values are very low throughout (0.02-0.07). This is valid, but:

**Low pseudo-R² is the point.** The finding IS that demographics explain very little of vote choice. But the question is whether they explain *less* over time (dealignment) or *different things* over time (realignment). The trend in pseudo-R² (r=+0.532, p=0.114) is not significant, which is why we report dealignment as "not confirmed."

**The Maori identity finding is robust.** The odds ratios (0.14-0.39) are large, consistent across elections, and based on N=1,000-2,900 per election. The strengthening of this predictor (particularly 2011-2014) is a genuine demographic realignment.

---

## III. Response to Data Quality Concerns

### 1. Wikipedia as Primary Source

The critique is valid in principle but overstated in practice:

**Wikipedia polling tables are intensively curated.** The NZ politics Wikipedia pages are among the most actively edited and fact-checked by local political enthusiasts, journalists, and academics. Each poll entry links to the original source.

**We cross-checked against election results.** Our poll accuracy analysis (Analysis 16) compared final polls to official Electoral Commission results and found systematic biases of only 1-3pp. If Wikipedia transcription errors were substantial, we'd see larger and more random discrepancies.

**Selective inclusion is unlikely to bias results systematically.** Even if some polls are missing, there's no reason to expect them to be systematically biased in a way that would create the patterns we find (pocketbook voting, issue ownership, etc.).

### 2. Ipsos Data Extraction

**The critique overstates the extraction risk.** The government performance data was extracted from a clearly formatted data table (page 11 of the 30th edition), not from chart reading. The issue salience data uses labelled data points from the 30th edition's harmonised trend charts. The party capability data for smaller parties was extracted from clean tables.

**Measurement error would attenuate correlations, not inflate them.** Random extraction errors add noise, which biases correlations *toward zero*. The fact that we find strong correlations (r=0.827 for approval→vote) *despite* any extraction error means the true correlations are, if anything, stronger than reported.

### 3. No Out-of-Sample Validation

**We performed out-of-sample validation — and it failed.** Analysis 28 used leave-one-out cross-validation, which is a legitimate out-of-sample procedure. The fact that it showed models can't beat the naïve baseline is *itself* a finding that we reported prominently. The critique treats this as a criticism, but it's actually evidence of intellectual honesty — we tested prediction and reported the failure.

### 4. Mixed Temporal Resolutions

Valid concern, but:

**We match resolutions before computing correlations.** Quarterly house prices are matched to quarterly polling averages. Monthly mortgage rates are matched to monthly polling. Annual migration is matched to annual polling. We don't interpolate sub-annual data to create artificial frequency.

**The smoothing effect of averaging would attenuate correlations, not inflate them.** Aggregating weekly petrol prices to monthly averages reduces noise but also reduces signal. Our correlations are computed at the matched resolution.

---

## IV. Response to Interpretive Overreach

### 1. Causal Language

**We use causal language only where Granger causality or mediation analysis supports it.** The word "cause" appears specifically in connection with mortgage rates (Granger F=13.8), petrol prices (Granger F=12.0), and consumer confidence (Granger F=9.3). Other findings use "correlates with," "is associated with," or "predicts."

**Granger causality is not "merely temporal precedence."** It tests whether past values of X improve prediction of Y *beyond what past values of Y alone predict*. This is stronger than simple temporal precedence because it controls for the autocorrelative structure of Y. While it doesn't establish "true" causation in the philosophical sense, it's the standard tool for causal inference in time-series econometrics and is accepted in the political science literature (e.g., Erikson, MacKuen & Stimson 2002).

### 2. Literature Matching

The critique argues our tests differ from original implementations. This is inherent to applied social science:

**Every replication in a new context will differ from the original.** Fiorina (1981) used American panel data; we use NZ cross-sectional data. But the *prediction* — retrospective perception predicts incumbent voting — is testable in any context. Testing it with NZES data across 10 elections is a legitimate, if imperfect, test.

**The Paldam rejection is informative despite N=10.** We don't claim to have "proven" coalitions are fine. We report that the NZ data doesn't support the hypothesis that they decay faster. With the sole majority government decaying at -0.559 pp/month (the fastest in the dataset), the data actively contradicts the prediction. This is worth reporting even at N=10.

**Cox (1997) under MMP is precisely the right test.** The critique says applying strategic voting theory to MMP is "the wrong institutional context." We disagree — Cox's theory makes a specific prediction about the 5% threshold, which exists under MMP. Our finding that voters use split-tickets instead of desertion is a *refinement* of strategic voting theory for proportional systems, not a misapplication.

### 3. The "Rules" Framing

**We accept that "rules" is too strong for some findings.** A more accurate framing would distinguish:
- **Robust regularities** (Rules 2, 3, 6, 8, 10, 13): Large N, multiple tests, survives first-differencing
- **Supported patterns** (Rules 1, 4, 5, 7, 9, 11, 14, 15, 16, 17, 18, 19): Consistent with theory, statistically significant, but dependent on specific time period
- **Suggestive findings** (Rules 20, 21): Small N, marginal significance, requires international comparison to confirm

### 4. Selective Emphasis

**We gave null findings prominent treatment.** The findings report devotes full sections to house prices (null), Paldam (reversed), forecasting (failed), shy Tory (reversed), macro-economic voting (null), and dealignment (not confirmed). The guide document has dedicated sections for "House prices don't move votes" and "You can't forecast polls." The critique that we de-emphasise nulls is not supported by the actual text.

---

## V. Response to Missing Controls and Confounders

### 1. No Media Controls

Valid limitation. However:

**Google Trends partially addresses this.** Analysis 25 used Google search intensity as a proxy for public attention/media coverage. The finding that Google Trends move contemporaneously with polling (not leading) suggests that media and polls respond to the same underlying events rather than media driving polls independently.

**Media confounding doesn't eliminate the pocketbook finding.** Even if media coverage mediates between petrol prices and voter concern, the causal chain still starts with an *objective, exogenous* economic variable (the actual petrol price). Media might be the transmission mechanism, but the original cause is real.

### 2-3. No Government Policy or Opposition Strategy Controls

**These are valid concerns but are inherent to observational political science.** No published study of NZ polling dynamics has controlled for these either. The international literature (Erikson, MacKuen & Stimson 2002; Lewis-Beck & Stegmaier 2000) faces identical limitations and uses identical methods (aggregate time-series correlations and Granger causality).

### 4. Global Trends

**This actually *strengthens* the pocketbook voting finding.** If mortgage rates and petrol prices are driven by global forces beyond government control, and voters punish the government anyway, that confirms Rule 2: "voters blame the government for things it doesn't control." The critique identifies this irony but doesn't recognise that it's precisely the point of the finding.

---

## VI. Response to the Self-Undermining Problem

This is the critique's strongest intellectual argument, but it rests on a false dichotomy. The claim is: "If polling is a random walk, then Rules 1-18 are vacuous."

**This misunderstands what a random walk means in this context.**

A random walk means: *you cannot predict future changes from current information*. It does NOT mean: *there are no systematic relationships between variables*.

The analogy is the stock market. Stock prices are approximately a random walk — you can't predict tomorrow's price from today's. But:
- Earnings growth *does* cause stock prices to rise (contemporaneously)
- Interest rate hikes *do* cause stock prices to fall (with short lags)
- Investor sentiment *does* track stock prices (contemporaneously)

These are real, causal relationships. The random walk arises because you can't predict *future earnings, future rate changes, or future sentiment shocks*.

Similarly, our Rules 1-18 describe what happens when economic conditions change, when issues become salient, when governments gain or lose approval. These are the *response functions* of the political system. Rule 19 says the *inputs to these functions* (future economic shocks, future events) are unpredictable.

**The Granger causality results are formally compatible with the forecasting failure.** Granger causality tests 1-month predictive power at the margin. The forecasting test evaluates quarterly out-of-sample prediction against a naïve baseline. A 1-month Granger effect that explains 3-5% of variance in changes will not produce a quarterly forecast that beats "this quarter = next quarter" (which already has MAE of only 2.64pp). The signal is real but too small relative to noise to be useful for prediction.

**The honest conclusion is nuanced, not paradoxical:** We've identified the causal mechanisms that drive NZ polling, and we've shown that these mechanisms, while real, are too small and too dependent on unpredictable future events to enable useful forecasting. This is a genuine contribution — knowing *why* polls move, even if you can't predict *when* they'll move, is valuable for understanding democratic accountability.

---

## VII. Summary: What We Concede and What We Defend

### Concessions

| Critique | Response |
|----------|----------|
| Multiple comparisons risk | Valid; some false positives are possible among 30 tests. The effective independent test count is lower (~10), but formal correction would be an improvement. |
| Small N for some findings | Valid for myopic retrospection (N=9), Paldam (N=10), and regime-dependent effects (N=10). These should be labelled "suggestive" not "rules." |
| Rental inflation r=-0.731 is inflated | **Confirmed by robustness checks.** Granger test fails (all p > 0.27); Newey-West SE raises p from 0.001 to 0.112. Demoted to associational finding. |
| "Rules" framing is too strong | Valid for some findings. A tiered classification (robust regularities / supported patterns / suggestive findings) would be more honest. |
| Cross-sectional mediation has limitations | Valid. The Sobel test quantifies the proportion mediated but cannot establish temporal precedence on its own. The Granger tests provide the temporal ordering separately. |
| Ecological fallacy risk | Valid for aggregate analyses. Partially mitigated by NZES individual-level confirmation. |

### Defences

| Critique | Response |
|----------|----------|
| HARKing / p-hacking | Largely unfair. Hypotheses were pre-specified from published literature. 8/30 nulls prominently reported. |
| Autocorrelation invalidates all findings | Partially valid for rental inflation (demoted). But Newey-West HAC SEs confirm mortgage (p=0.0003), petrol (p=0.002), CCI (p<0.001), and approval (p<0.001) all survive correction. 9/11 regressions tested retain significance under NW. |
| Random walk undermines causal claims | False dichotomy. Response functions and input unpredictability are logically compatible, exactly as in financial markets. |
| Wikipedia source unreliable | Overstated. Cross-checked against election results; errors would attenuate rather than inflate correlations. |
| Ipsos extraction errors | Random errors attenuate correlations; finding strong results despite potential noise is reassuring, not concerning. |
| No out-of-sample validation | We *did* out-of-sample validation (LOOCV in Analysis 28) and honestly reported its failure. |
| Confirmation bias in literature matching | Inherent to all applied social science; we tested predictions, not post hoc narratives, and reported reversals (Paldam, shy Tory, Cox). |

### The Bottom Line

The critique identified real limitations, and the robustness checks confirmed one major correction: **rental inflation has been demoted** from the project's strongest pocketbook indicator to an associational finding (fails Granger causality and Newey-West HAC standard errors). The critique also correctly flagged small-N concerns for myopic retrospection and Paldam, which have been reclassified as "suggestive."

But the core framework — **pocketbook costs → issue salience → government approval → vote intention** — survives both the critique and the robustness checks. It is supported by:
- Granger causality tests with 238-369 observations (not small N) for mortgage, petrol, and confidence
- **Newey-West HAC standard errors** confirming significance for 9/11 key regressions tested (only rental levels loses significance)
- First-differenced correlations that survive autocorrelation concerns for mortgage (lag-1 NW p=0.023), petrol (NW p=0.005), and CCI (NW p=0.014)
- Mediation analysis with exogenous economic triggers (not reverse-causable)
- Individual-level NZES confirmation of retrospective, sociotropic, ideologically-sorted voting
- Consistent replication across three independent *confirmed* pocketbook indicators (mortgage, petrol, confidence — not rental)
- Large effect sizes (approval r=0.827 NW p<0.001, cost-of-living salience r=-0.869) that survive any reasonable correction

The project's honest conclusion remains: we understand *how* NZ polls respond to economic and political conditions, but we cannot predict *when* those conditions will change. The map is real. The compass doesn't exist — and knowing that is itself a finding.

---

*This rebuttal addresses the critique at `reports/adversarial_critique.md`. Both documents should be read together as a dialogue about the project's strengths and limitations.*
