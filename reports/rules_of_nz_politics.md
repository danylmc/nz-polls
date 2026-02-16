# The Rules of New Zealand Politics

*Derived from 1,016 polls (1990-2025), quarterly GDP/CPI/unemployment/house price data (1962-2025), monthly interest rates/consumer confidence/dairy prices/exchange rates (1973-2025), annual migration (1960-2024), 35,000+ individual survey respondents across 10 elections (NZES 1996-2023), and 30 waves of the Ipsos NZ Issues Monitor (2017-2025)*

---

## The Polling Rules

### Rule 1: The Seesaw

**When one major party goes up, the other goes down.**

National and Labour operate in a zero-sum relationship. At the level of polling
support, their correlation is r = -0.51 (p < 0.001). Even in poll-to-poll *changes*,
the relationship holds at r = -0.32 (p < 0.001). Gains for one major party come
almost exclusively at the expense of the other.

This is not mechanically guaranteed — in theory, both parties could lose support
simultaneously to minor parties, or both could gain at minor parties' expense. But
in practice, NZ politics is structured around a dominant two-party contest where
voters move between National and Labour far more often than they exit the major-party
system entirely.

| Measure | r | p-value |
|---------|---|---------|
| Level correlation | -0.508 | < 0.001 |
| Change correlation | -0.316 | < 0.001 |

---

### Rule 2: The Rubber Band

**Extreme polls snap back. What goes up must come down.**

Approximately 67% of extreme polls (those more than one standard deviation from the
mean) revert toward the mean in the next observation. More strikingly, poll-to-poll
*changes* are anti-persistent: a large gain this month predicts a loss next month.

This reversal effect is strong — the autocorrelation of changes is -0.51 for National
and -0.45 for Labour, both highly significant. This does not mean polls are random;
polling *levels* are highly persistent (lag-1 autocorrelation ~0.90 for both parties).
But the rate of change oscillates. A party that surges 5 points in a month should
expect to give some of that back.

The practical implication: single dramatic polls are usually noise. A party that
hits an unusually high or low reading will almost certainly regress. Sustained shifts
require multiple consecutive polls confirming the new level.

| Party | Level persistence (lag-1) | Change reversal | Extreme poll reversion |
|-------|--------------------------|-----------------|----------------------|
| National | 0.90 | -0.512 | 68% |
| Labour | 0.90 | -0.447 | 66% |

---

### Rule 3: The Honeymoon

**New leaders get a polling bounce.**

Leadership changes produce measurable polling effects, with the magnitude depending
on context and the leader involved. The most dramatic example in the dataset is
Jacinda Ardern's elevation to Labour leader in August 2017: Labour surged +8.9
percentage points and National dropped -2.1 points in the following polls.

The honeymoon effect is well-documented internationally, but NZ's MMP environment
may amplify it — new leaders can attract not just opposition voters but also minor
party supporters. The bounce is temporary: it decays as the new leader faces
scrutiny, which blends into the broader incumbent fatigue pattern (Rule 8 on the
negativity bias, Rule 11 on the cost of ruling).

---

### Rule 4: The Rally

**Crises can boost incumbents, but the effect is weak and unreliable in NZ.**

The "rally round the flag" effect — where national crises temporarily boost incumbent
support — is well-established in presidential systems, particularly the United States.
In NZ, the evidence is mixed:

- **Delta lockdown (August 2021)**: Labour +1.6pp, National -4.5pp — a clear but
  modest rally effect
- **Christchurch earthquake (February 2011)**: National -1.0pp, Labour +0.8pp — the
  incumbent actually *lost* a point

The NZ rally effect appears weaker than the international norm, perhaps because MMP
governments are coalition-based and accountability is diffused, or because NZ's media
environment is less prone to the patriotic framing that drives rallies in presidential
systems. Crises can help incumbents, but it is not a reliable rule — the nature of the
crisis, the government's perceived response, and the political context all mediate the
effect.

---

## The Economic Rules

### Rule 5: The Cost of Living

**Inflation hurts incumbents. Housing costs hurt the most. GDP barely matters.**

The original analysis using annual economic data found no significant economic voting
effect (p > 0.05) — but this was a statistical artifact of testing with only N = 11
elections. Re-testing with quarterly data matched to 1,016 individual polls reveals
that economic voting is highly significant, but only through the inflation channel.

Housing costs (CPI Housing, which measures rents, local authority rates, insurance,
maintenance, and household energy — *not* house prices or capital gains) are the
strongest single predictor of incumbent support. In the horse-race comparison of six
economic indicators (each with controls for government identity and time trend),
housing costs top the rankings by a wide margin:

| Rank | Indicator | Adj R² (with controls) |
|------|-----------|------------------------|
| 1 | Housing costs (CPI) | **0.484** |
| 2 | Consumer confidence | 0.313 |
| 3 | House price growth | 0.190 |
| 4 | CPI inflation | 0.104 |
| 5 | Unemployment rate | 0.095 |
| 6 | GDP growth | 0.070 (not significant) |

After controlling for government identity and time trend, each 1% increase in housing
inflation costs the incumbent approximately **1.2 polling points** (p < 0.001). The
raw correlation (r = -0.62) overstates the effect because Labour governments coincide
with higher housing inflation (5.1% vs 3.3% under National) and tend to have lower
incumbent support. But the correlation holds within both National (r = -0.43) and
Labour (r = -0.55) governments separately.

GDP growth has essentially no relationship with incumbent support (β = -0.02 in the
full model, p = 0.80). GDP is the indicator governments boast about but voters ignore.

---

### Rule 6: The Wealth Effect

**Rising house prices help incumbents. Rising housing costs hurt them. Both operate simultaneously.**

This is the most surprising finding from the extended analysis. House prices and
housing costs — two different measures of "housing" — have **opposite** effects on
incumbent support, and both are statistically significant:

| Variable | β | Direction | p-value |
|----------|---|-----------|---------|
| House price growth (y/y %) | +0.18 | Helps incumbent | < 0.001 |
| CPI Housing inflation (y/y %) | -1.18 | Hurts incumbent | < 0.001 |

In a controlled model with both variables plus government identity and time trend,
the combined R² is **0.52** — these four variables alone explain over half the
variation in incumbent polling.

The wealth effect is asymmetric across governments: house price growth helps Labour
incumbents much more (r = +0.58) than National incumbents (r = +0.14). This likely
reflects that property booms under Labour were unusual and politically salient
(2002-2007, 2020-2021), while under National they are closer to the expected baseline.

The practical implication: the housing market creates a political tug-of-war. A boom
where prices rise faster than rents helps incumbents. A squeeze where rents and rates
rise without capital gains crushes them. The net political effect depends on the
balance — and on whether the incumbent's voters are more likely to be homeowners
(benefiting from prices) or renters (suffering from costs).

---

### Rule 7: The Rate Squeeze

**Higher interest rates hurt incumbents. Rates, rents, and prices form a housing trinity that dominates economic voting.**

The 3-month interbank rate (a proxy for the OCR) is the third-strongest single
predictor of incumbent support (Adj R² = 0.37), after housing costs and consumer
confidence. Each percentage point increase in interest rates costs the incumbent
approximately **2.4 polling points** after controlling for government identity and
time trend (p < 0.001).

When interest rates, housing costs, and house prices are combined in a single
"housing trinity" model, all three are independently significant:

| Variable | β | p-value |
|----------|---|---------|
| Interest rate | -0.81 | < 0.001 |
| Housing costs (CPI) | -2.00 | < 0.001 |
| House price growth | +0.17 | < 0.001 |

This trinity achieves R² = 0.44 — three housing-related variables alone explain
44% of the variation in incumbent polling.

The mechanism is intuitive: rate hikes simultaneously slow house price growth
(removing the wealth effect from Rule 6) and increase mortgage costs for homeowners
while pushing up rents for tenants (amplifying the cost-of-living effect from
Rule 5). A tightening cycle is a political pincer movement.

There is an asymmetry across governments: under Labour, rate increases strongly
hurt incumbent support (r = -0.47). Under National, the relationship is weaker
and rate *increases* even show a slight positive correlation (r = +0.13) —
possibly because National-era rate hikes coincided with economic confidence and
booming house prices that offset the direct cost.

---

### Rule 8: The Negativity Bias

**Economic contractions hurt incumbents. Economic growth does not help them.**

Incumbents poll significantly lower during contraction quarters than during growth
quarters: 39.5% vs 42.4% (t-test p < 0.001, N = 198 contraction polls, N = 813
growth polls). But during growth periods, the relationship between GDP growth and
incumbent support vanishes (r = 0.015, p = 0.68). A growing economy is the
*expected baseline* and earns no political reward.

This asymmetry is one of the most replicated findings in comparative political science
and is attributed to the "negativity bias" in human cognition: losses loom larger than
equivalent gains. For NZ incumbents, the practical implication is stark — growth
doesn't help, but contraction costs roughly 3 percentage points.

However, a subtlety: within contraction quarters, *deeper* contractions do not
produce proportionally worse results for incumbents. The within-contraction
correlation (r = -0.29) runs in the "wrong" direction — more negative GDP is
associated with *higher* incumbent support — because the deepest contraction in
the dataset (2020 Q2, annualized -10%) coincided with Labour polling above 50%
during the COVID rally. Excluding the COVID period, the within-contraction gradient
is not significant (r = -0.12, p = 0.11). The penalty for contraction appears to be
binary (contraction vs growth) rather than proportional to depth.

---

### Rule 9: The Confidence Channel

**Consumer confidence is the second-best predictor of incumbent support — and it bridges perception and reality.**

The OECD Consumer Confidence Index (derived from the Westpac McDermott Miller survey)
correlates with incumbent support at r = +0.47 (p < 0.001) — second only to housing
costs. After controlling for government identity and time trend, each 1-point increase
in the CCI is worth **3.2 polling points** for the incumbent (p < 0.001, R² = 0.32).

Consumer confidence matters because it captures the "felt" economy — the subjective
experience that Rule 10 (the partisan filter) shows is more important than statistics.
The CCI correlates strongly with inflation (r = -0.63) and moderately with GDP growth
(r = +0.29), confirming it picks up real economic signals. But it also incorporates
intangibles: housing stress, job security, cost-of-living anxiety.

The confidence effect holds within both National (r = +0.34) and Labour (r = +0.59)
governments. It is stronger under Labour, which may reflect that Labour's base is
more economically exposed (renters, lower income) and therefore more sensitive to
confidence swings.

However, consumer confidence does **not** function as a leading indicator. Lagged CCI
does not predict *changes* in incumbent support (r = +0.008, p = 0.79). Confidence
and polling move together contemporaneously rather than one leading the other. This
suggests both respond to the same underlying economic conditions rather than confidence
*causing* polling shifts.

The Ipsos government performance rating (Rule 27) outperforms consumer confidence as
a predictor of incumbent support (r = 0.83 vs r = 0.47), suggesting that direct
government evaluation captures more of the subjective assessment pathway than the
economy-focused CCI alone.

---

### Rule 10: The Partisan Filter

**What voters *believe* about the economy matters more than what is *actually* happening. And belief is filtered through partisanship.**

Using NZES survey data (2002-2020), subjective economic assessments explain 3.5 times
more variance in vote choice than actual GDP and inflation data (R² = 0.026 vs
R² = 0.007). When both are included in the same model, perception remains significant
while objective indicators weaken.

More importantly, economic perception is systematically distorted by partisanship.
Government supporters rate the economy 0.5 to 1.0 scale points higher than opposition
supporters — even when experiencing the same actual economic conditions. This "partisan
perceptual screen" is:

- Present in every election year tested (2002-2020)
- Statistically significant at p < 0.001 in all years
- Present under both National and Labour governments
- Not clearly strengthening or weakening over time

This creates a reinforcing feedback loop: partisanship shapes economic perception,
which then reinforces vote choice, which sustains partisanship. It also explains
why economic conditions can matter at the aggregate level (Rule 5) while appearing
weak in individual-level models: the signal is partially absorbed by the partisan
filter before it reaches the vote decision.

Consumer confidence data (Rule 9) provides monthly evidence for this mechanism.
The CCI tracks incumbent support at r = +0.47, suggesting that aggregate sentiment
measures capture the same perception-mediated channel that NZES surveys reveal at
the individual level.

| Model | R² | Key predictor |
|-------|----|---------------|
| Actual economy only | 0.007 | Inflation (weak) |
| Perceived economy only | 0.026 | Subjective assessment |
| Combined | 0.037 | Perception dominates |

---

### Rule 11: The Cost of Ruling

**All governments bleed support over time. Context matters more than coalition structure.**

Every NZ government since 1990 has experienced significant support erosion during its
term, with the sole exceptions of the Bolger government (1990-96, which was recovering
from deeply unpopular early-term reforms) and Ardern's first term (2017-2020, a
combination of honeymoon and COVID rally).

The fatigue rate varies enormously by government:

| Government | Type | Fatigue (pp/yr) | Context |
|------------|------|-----------------|---------|
| Ardern/Hipkins (2020-23) | Single-party | **-6.7** | COVID fatigue, cost-of-living crisis |
| Luxon (2023-) | Coalition | -3.2 | Early-term decline |
| Shipley (1996-99) | Coalition | -2.2 | Coalition collapse with NZ First |
| Clark (1999-2008) | Coalition | -2.0 | Steady erosion over 9 years |
| Key/English (2008-17) | Coalition | -1.0 | Slowest fatigue — sustained personal popularity |
| Bolger (1990-96) | Single-party | +1.1 | Recovery from early-term unpopularity |
| Ardern (2017-20) | Coalition | +3.2 | Honeymoon + COVID rally |

Paldam's (1991) prediction that coalition governments suffer steeper fatigue than
single-party governments is **not confirmed** for NZ (interaction p = 0.15). The
most dramatic fatigue in the dataset belongs to a single-party majority government
(Ardern/Hipkins at -6.7 pp/year), while the slowest belongs to a multi-partner
coalition (Key/English at -1.0 pp/year).

Fatigue does not interact significantly with economic conditions either (tenure × GDP
p = 0.66). The rate of decline appears driven by idiosyncratic factors — leader
quality, scandal, crisis management — rather than structural features of government
type or economic circumstances.

This updates the earlier finding of a flat average fatigue rate of -1.56 pp/year.
That average masked enormous variation: the range runs from +3.2 to -6.7 pp/year.
The political context of each government is far more important than the mechanical
passage of time.

---

### Rule 12: The Banker's Model

**NZ voters look forward, not backward — except during crises.**

When NZ voters punish or reward governments for the economy, they weight their
expectations of the future more heavily than their assessment of the past. Using
NZES data from three elections (2002, 2005, 2008) where both retrospective and
prospective economic assessments are available:

| Year | Retrospective R² | Prospective R² | Winner | Context |
|------|-------------------|----------------|--------|---------|
| 2002 | 0.041 | **0.122** | Prospective | Stable growth |
| 2005 | 0.024 | **0.076** | Prospective | Stable growth |
| 2008 | **0.029** | 0.006 | Retrospective | Global Financial Crisis |

In stable times (2002, 2005), forward-looking assessments predict vote choice 3x
better than backward-looking ones. Voters behave like MacKuen, Erikson & Stimson's
(1992) "bankers," evaluating whether the economy is heading in the right direction.

But during the GFC (2008), the pattern reverses. Retrospective assessments dominate
and prospective assessments become almost irrelevant. When the economy is actively
deteriorating, voters look at what has already gone wrong rather than speculating
about recovery.

The education interaction (do sophisticated voters look forward more, as the
literature predicts?) is not significant in the NZ data — both educated and less
educated voters appear to use the same mix of backward and forward evaluation.

Combined models (including both assessments) outperform either alone, confirming
that voters use both — the question is emphasis, not exclusion.

---

### Rule 13: The Myopic Voter (Weak)

**Voters weight recent economic conditions slightly more than earlier ones, but the evidence is thin.**

Achen & Bartels (2016) argue voters are "myopic," overweighting the election-year
economy while ignoring earlier conditions. The NZ evidence provides weak support:

- Late-term GDP correlates slightly more with election results (r = 0.248) than
  early-term GDP (r = 0.236), but neither is statistically significant
- At the poll level, the 1-quarter lag has the strongest individual correlation
  with incumbent support, but all effects are small (r < 0.06)
- In a distributed lag model, only the 1-quarter lag reaches significance
  (β = 0.33, p = 0.04)

With only 10 elections to work with, NZ lacks the statistical power to confirm or
reject the myopia thesis convincingly. The direction of the evidence is consistent
with Achen & Bartels, but the effects are too small to draw strong conclusions.
Voters may be slightly myopic, but they are not dramatically so.

---

### Rule 14: The 60% Model

**Six economic indicators together explain 61% of the variance in incumbent polling.**

When all six economic indicators (housing costs, consumer confidence, house price
growth, CPI inflation, unemployment, and GDP growth) are combined with controls
for government identity and time trend, the full model achieves R² = 0.61. This
means economic conditions — broadly measured — explain the majority of variation
in incumbent support.

GDP growth is the only indicator that remains insignificant in the full model
(p = 0.80). Every other indicator contributes independently.

| Variable | β | p-value | Interpretation |
|----------|---|---------|----------------|
| Housing costs (CPI) | -1.07 | < 0.001 | Rents/rates/energy hurt |
| Consumer confidence | +1.76 | < 0.001 | Sentiment helps |
| House price growth | +0.24 | < 0.001 | Wealth effect helps |
| CPI inflation | +1.66 | < 0.001 | * |
| Unemployment rate | +3.17 | < 0.001 | * |
| GDP growth | -0.02 | 0.80 | Irrelevant |
| Incumbent is National | +2.07 | 0.03 | National polls higher |
| Time trend | -0.12 | 0.03 | Secular decline |

*Note: The positive signs on CPI inflation and unemployment in the full model are
counterintuitive and result from multicollinearity — these variables are highly
correlated with housing costs and consumer confidence. In univariate models, both
have the expected negative signs. The full model isolates the unique contribution
of each variable conditional on all others, which can produce sign reversals when
predictors are correlated.*

The practical implication: incumbent polling is not random or mysterious. It is
overwhelmingly driven by economic conditions that voters experience directly. The
non-economic residual (39%) is where leadership effects, events, scandals, and
campaign dynamics operate — but they are the minority of the variance, not the
majority.

---

## The Voter Rules

### Rule 15: The Thermostat

**When government policy moves left, the public drifts right. And vice versa.**

NZ voters exhibit strong thermostatic responsiveness. Under right-leaning governments
(National), the public's mean left-right self-placement shifts leftward; under
left-leaning governments (Labour), it shifts rightward. This pattern holds in **7 out
of 8** inter-election shifts in the NZES data (1996-2020):

| Period | Government Direction | Public Shift | Thermostatic? |
|--------|---------------------|-------------|---------------|
| 1996→1999 | National (right) | -0.56 (leftward) | Yes |
| 1999→2002 | Labour (left) | +0.14 (rightward) | Yes |
| 2002→2005 | Labour (left) | +0.07 (rightward) | Yes |
| 2005→2008 | Labour (left) | +0.25 (rightward) | Yes |
| 2008→2011 | National (right) | -0.04 (leftward) | Yes |
| 2011→2014 | National (right) | +0.26 (rightward) | **No** |
| 2014→2017 | National (right) | -0.49 (leftward) | Yes |
| 2017→2020 | Labour (left) | +0.03 (rightward) | Yes |

This resolves a puzzle from the earlier ideology analysis (Rule 20). NZ's mean
left-right self-placement has been remarkably stable at 5.0-5.7 for 30 years. The
thermostatic finding reveals *why*: it is not that nothing is happening, it is that
the public is actively self-correcting against whichever direction the government
pushes. The stability is dynamic equilibrium, not apathy.

The mechanism is consistent with Wlezien's (1995) original theory: voters want policy
to be "about right." When a right-wing government moves policy rightward, marginal
voters feel the status quo has moved too far and adjust their preferences leftward —
and vice versa. This produces the centrist equilibrium observed in Rule 20 as an
emergent property rather than a fixed preference.

---

### Rule 16: The Issue Agenda

**Parties benefit when voters care about "their" issues. The effect is massive.**

Issue ownership theory (Petrocik 1996) predicts that National benefits when voters
care about the economy, tax, and law and order, while Labour benefits when health,
education, and housing are salient. The NZ evidence confirms this in **all 7 elections
tested**, with an average ownership gap of **+27.5 percentage points**.

Among National/Labour voters who named a "most important issue":

| Year | Top Issue | Nat Vote (Nat Issue) | Nat Vote (Lab Issue) | Gap |
|------|-----------|---------------------|---------------------|-----|
| 1999 | Health | 57.7% | 35.0% | +22.7pp |
| 2002 | Health | 34.6% | 28.6% | +5.9pp |
| 2005 | Tax | 44.5% | 39.4% | +5.2pp |
| 2014 | Economy | 82.6% | 41.4% | +41.2pp |
| 2017 | Economy | 73.1% | 34.3% | +38.8pp |
| 2020 | Economy | 61.5% | 19.3% | +42.2pp |
| 2023 | Economy | 69.9% | 33.3% | +36.6pp |

The gap has grown dramatically. In the early MMP elections (1999-2005), the ownership
effect was modest (5-23pp). From 2014 onward it has been enormous (37-42pp). This
likely reflects increasing partisan sorting — voters are more inclined to identify
issues that align with their pre-existing party preference — but the pattern is
unmistakable regardless of interpretation.

The issue salience pattern also matters. Health dominated voter concerns in 1999-2002
(benefiting Labour). Tax was the top issue in 2005 (when Don Brash's tax-cut campaign
nearly won). The economy has been the top issue from 2014 onward — and when it is
salient, National benefits strongly.

This extends the earlier finding about NZ First and immigration (Rule 19). Issue
ownership is not unique to NZ First — it is a fundamental feature of how voters
choose between parties. The election agenda shapes the result.

**Monthly Issue Salience Tracking (Ipsos 2018-2025)**

The NZES data above is cross-sectional — one snapshot per election. The Ipsos NZ Issues
Monitor provides monthly tracking of issue salience (2018-2025, 28 waves), allowing
a test of whether *changes* in issue salience predict *changes* in party support at
higher temporal resolution.

The level correlations between issue salience and party vote share (27 matched months):

| Issue | National r | Labour r | Interpretation |
|-------|-----------|---------|----------------|
| Inflation/Cost of Living | -0.01 | **-0.885** | Strongly hurts Labour |
| Healthcare/Hospitals | -0.14 | **+0.670** | Helps Labour |
| The Economy | +0.24 | **-0.701** | Hurts Labour |
| Housing/Price | +0.23 | **-0.692** | Hurts Labour |
| Crime/Law & Order | **+0.403** | **-0.726** | Helps National, hurts Labour |
| Unemployment | **-0.593** | **+0.631** | Helps Labour, hurts National |
| Climate Change | -0.22 | -0.37 | Weak (hurts both) |

The headline finding: **cost-of-living salience is the single strongest issue-level
predictor of party support** (r = -0.885 with Labour). When voters rank inflation as
a top concern, Labour's vote share collapses. This is consistent with the economic
voting findings (Rules 5-8): cost-of-living pain hurts incumbents, and Labour was
the incumbent for most of the 2018-2023 inflation surge.

First-differenced correlations (change in salience vs change in vote, to avoid
spurious trend correlation) confirm that healthcare and housing shifts predict vote
shifts even in period-to-period changes:

- Healthcare salience increase → Labour vote increase (r = +0.66)
- Housing salience increase → National vote increase (r = +0.51), Labour decrease (r = -0.54)
- Unemployment salience increase → Labour vote increase (r = +0.58)

The Ipsos monthly data thus confirms the NZES cross-sectional pattern with independent
data at much higher temporal resolution. Issue ownership is not just a static feature
of elections — it operates continuously, month by month, as the issue agenda shifts.

---

### Rule 17: The Leader Premium

**Leader evaluations explain more than ideology. NZ elections are personality contests.**

Valence theory (Stokes 1963; Clarke et al. 2004) predicts that voters choose based
on perceived leader quality rather than ideological proximity. The NZ evidence strongly
confirms this. Across 8 elections with both measures available, leader thermometer
ratings outperform left-right ideological proximity as a predictor of vote choice in
**6 out of 8**:

| Year | Leaders | Ideology R² | Valence R² | Combined R² |
|------|---------|------------|------------|-------------|
| 1996 | Bolger/Clark | **0.558** | 0.532 | 0.696 |
| 1999 | Shipley/Clark | **0.482** | 0.459 | 0.626 |
| 2005 | Clark/Brash | 0.598 | **0.705** | 0.782 |
| 2008 | Clark/Key | 0.532 | **0.700** | 0.743 |
| 2011 | Key/Goff | 0.578 | **0.623** | 0.757 |
| 2014 | Key/Cunliffe | 0.617 | **0.644** | 0.777 |
| 2017 | English/Ardern | 0.587 | **0.699** | 0.779 |
| 2020 | Ardern/Collins | 0.504 | **0.647** | 0.715 |

The pattern shifted in 2005. In the first two MMP elections (1996, 1999), ideology
and valence were roughly equal predictors. From 2005 onward, leader evaluations
consistently dominate. This coincides with John Key's emergence — a leader whose
appeal was almost entirely valence-based (personal likability, perceived competence)
rather than ideological positioning.

Both factors contribute — combined models (R² = 0.70-0.78) substantially outperform
either alone. But if forced to choose one predictor for NZ vote choice, "who do you
like more?" beats "who is closer to you ideologically?" in most elections.

Explicit competence ratings (available for 2014, 2017, 2023) also strongly predict
vote choice (R² = 0.40-0.53), but less powerfully than overall leader affect. This
suggests voters' holistic feeling about a leader — warmth, charisma, trustworthiness —
matters more than their narrow assessment of competence alone.

The practical implication is that NZ elections are substantially leader-driven. This
is consistent with the honeymoon effect (Rule 3), the varying fatigue rates across
governments (Rule 11), and the weakness of demographics as predictors (Rule 23).
In a system where ideology and demographics explain relatively little, the leaders
themselves become the primary mechanism through which voters differentiate parties.

---

### Rule 18: The Demographic Floor

**Demographics explain almost nothing about vote choice. Ideology explains nearly everything.**

This is one of the starkest findings in the data. Logistic regression models using
all available demographics (age, gender, education, income, ethnicity) to predict
National vs Labour vote achieve pseudo-R² values of only **0.01 to 0.14** across
10 elections. Adding left-right self-placement to the same model catapults R² to
**0.22-0.42**.

| Year | Demographics Only R² | + Ideology R² | Ideology Added |
|------|---------------------|---------------|----------------|
| 1996 | 0.007 | 0.418 | +0.410 |
| 1999 | 0.013 | 0.364 | +0.352 |
| 2002 | 0.001 | 0.335 | +0.335 |
| 2005 | 0.012 | 0.319 | +0.307 |
| 2008 | 0.009 | 0.225 | +0.217 |
| 2011 | 0.007 | 0.303 | +0.297 |
| 2014 | 0.015 | 0.363 | +0.348 |
| 2017 | 0.018 | 0.349 | +0.330 |
| 2020 | 0.017 | 0.253 | +0.236 |

Demographics alone never explain more than 2% of the variance. Left-right
self-placement adds 22-41 percentage points of explanatory power. NZ voters choose
based on where they see themselves on the political spectrum, not what demographic
box they tick.

The dealignment thesis (Dalton 2000) predicts that demographics should be losing
predictive power over time. In NZ, this is **not confirmed** — but only because
demographics never had much predictive power to begin with. There was nothing to
dealign from. The weak positive trend (r = +0.43, p = 0.25) is driven primarily by
Māori ethnicity becoming a stronger predictor from 2005 onward, following the
Foreshore and Seabed controversy and the rise of Te Pāti Māori.

This finding reframes several other rules. The age reversal (Rule 21), gender gap
(Rule 22), and diploma divide (Rule 23) are all real patterns, but they operate at
the margins of a system where demographics collectively explain 1-2% of vote choice.
The dominant predictors are ideology (Rule 20), leader evaluations (Rule 17), issue
salience (Rule 16), and economic perceptions (Rule 10).

---

### Rule 19: The Strategic Voter

**Minor party supporters strategically desert. But the rate hasn't changed since MMP began.**

Under MMP, voters have incentives to vote strategically — giving their party vote to
a preferred minor party while giving their electorate vote to a major party candidate
who can win the seat. Individual-level NZES data confirms this pattern is widespread:

**Strategic desertion** (preferred party ≠ actual party vote): Minor party supporters
desert their preferred party at **25.8%** — 2.5 times the rate of major party supporters
(10.0%). The most common desertion flows follow bloc logic:

- Green → Labour (11-20% across elections)
- ACT → National (18-24%)
- Māori Party → Labour (23-33%)
- NZ First → Labour or National (depending on the year, 14-16%)

**Split-ticket voting** (party vote ≠ electorate vote): approximately **30%** of voters
split their ticket in every election since 1999, with no significant trend over time
(r = 0.15, p = 0.78). The dominant patterns are:

- Electorate Labour / Party Green (the most common split in every election since 2011)
- Electorate National / Party ACT (rising from 2020 onward)
- Electorate Labour / Party NZ First (declining since 2017)

This reframes the earlier aggregate finding that minor parties don't get "squeezed"
near elections. At the individual level, strategic desertion is real and substantial.
But it is a *constant background feature* of MMP rather than something that intensifies
as polling day approaches. Voters who are going to desert have already decided well
before the campaign. The aggregate polling for minor parties stays flat because the
desertion rate is baked in from the start.

The profile of strategic voters is revealing: they tend to be more ideologically
centrist (closer to 5.0 on the left-right scale) and, in some elections, lower in
education. They are not sophisticates gaming the system — they are moderate voters
who like a minor party but default to the safe major-party option.

---

## The Demographic Rules

### Rule 20: The Centrist Electorate

**NZ voters are centrist and have stayed centrist. The country is not polarizing.**

Despite international trends and domestic commentary about increasing division, the
data shows remarkable ideological stability:

- Mean left-right self-placement has stayed between 5.0 and 5.7 on a 0-10 scale
  across every NZES survey from 1996 to 2020
- The standard deviation of left-right placement is stable at 2.3-2.5 — no
  evidence of bimodal polarization
- The distance between National voters (~7.0) and Labour voters (~4.0) averages
  about 3 scale points and shows no widening trend
- Affective polarization (in-party vs out-party thermometer gap) is moderate at
  ~4.2 on a 0-10 scale, with too few data points to assess trends

This stability is not apathy — it is the product of thermostatic self-correction
(Rule 15). When governments push policy left or right, the public adjusts in the
opposite direction, maintaining the centrist equilibrium as a dynamic process.

NZ has *not* experienced the ideological sorting seen in the US Congress, the
bimodal split observed in US public opinion surveys, or the sharp affective
polarization documented in many Western democracies.

The one exception is the **Green Party**, which voters perceive as moving
leftward over time (from 3.3 in 2011 to 2.2 in 2023 on the left-right scale).
National's and Labour's perceived positions have remained stable.

| Measure | 1996-2002 | 2005-2011 | 2014-2020 | Trend |
|---------|-----------|-----------|-----------|-------|
| Mean L-R placement | 5.0-5.3 | 5.2-5.7 | 5.0-5.2 | Stable |
| SD of L-R placement | 2.3-2.5 | 2.3-2.4 | 2.3-2.5 | Stable |
| Nat-Lab voter distance | ~3.0 | ~3.1 | ~3.2 | Stable |

---

### Rule 21: The Age Reversal

**Young voters shifted from right to left. Old voters shifted from left to right. The gap is now 30 points.**

This is the single largest structural change in NZ politics over the past 30 years.
In 1996, voters aged 18-29 were approximately 11 percentage points *more* likely to
vote National than voters aged 60+. By 2017, the pattern had completely reversed:
voters 60+ were 20 percentage points more likely to vote National than those 18-29.

That is a swing of roughly 30 percentage points in the age gradient over two decades —
larger than the equivalent shift observed in the UK and comparable to the US.

| Year | Age gradient (60+ minus 18-29, National vote share) |
|------|-----------------------------------------------------|
| 1996 | -11pp (young more National) |
| 1999 | -5pp |
| 2002 | +3pp |
| 2005 | +5pp |
| 2008 | +8pp |
| 2011 | +10pp |
| 2014 | +14pp |
| 2017 | +20pp (old more National) |

The likely drivers include housing wealth (older homeowners benefiting from property
appreciation while younger renters face rising costs — see Rules 5-7), climate and
environmental politics (more salient to younger cohorts), and social liberalism on
issues like drug reform and gender identity. The data can demonstrate the pattern but
cannot conclusively prove which mechanism dominates.

This realignment has profound implications for NZ's political future: as older
cohorts are replaced, the age gradient implies a structural headwind for National
unless the party can rebuild its appeal to younger voters — or unless voters shift
rightward as they age (the "lifecycle" hypothesis, which the data cannot yet test
with only 30 years of coverage).

---

### Rule 22: The Gender Gap

**Men vote National at 7-8 percentage points higher rates than women. Always.**

The gender gap in NZ politics is persistent, significant, and remarkably stable.
In every NZES survey from 1996 to 2020, men are more likely to vote National than
women by approximately 7-8 percentage points. The gap does not widen, does not
narrow meaningfully, and does not interact strongly with other variables.

This matches the "modern gender gap" observed in most Western democracies, where
women lean left and men lean right. Unlike some countries (notably the US), NZ's
gender gap shows no trend of increasing over time — it arrived early and has simply
persisted.

The gap temporarily narrowed in 2002 and 2011, both years where National achieved
unusually broad support (a landslide effect that compressed demographic differences).
But it returned to its baseline in subsequent elections.

---

### Rule 23: The Late Diploma Divide

**University-educated voters shifted left after 2017. NZ was 5-7 years behind the international trend.**

In the US and UK, the "diploma divide" — where voters with university degrees shifted
toward left-of-centre parties while those without shifted right — emerged clearly
around 2010-2012. In NZ, the pattern arrived later:

- Before 2017, university-educated voters were *slightly more likely* to vote National
  than those without degrees
- In 2017, the gap reversed: voters without qualifications became marginally more
  National, and degree-holders shifted toward Labour and the Greens

The reversal is real but more modest than in the US or UK, consistent with NZ's
generally lower levels of political polarization (see Rule 20).

Importantly, **income-based class voting has *not* declined** in NZ. Higher-income
voters remain more likely to vote National (r = 0.07 to 0.22 depending on the year),
and there is no clear dealignment trend. The education and income effects are
increasingly decoupled — a pattern also observed internationally, where education
predicts cultural/social preferences while income predicts economic ones.

---

### Rule 24: The Ethnic Anchor

**Māori voters are strongly and persistently Labour-aligned. Te Pāti Māori is growing.**

Ethnic voting is the most stable cleavage in the NZ dataset. Māori voters have
supported Labour at rates 20-40 percentage points higher than European voters in
every election. Pacific voters follow a similar pattern, though data is sparser.

The significant development since 2004 is the growth of Te Pāti Māori, which has
drawn an increasing share of the Māori vote — reaching approximately 20% by 2023.
This growth has come primarily at Labour's expense, fragmenting the left-of-centre
Māori vote rather than expanding it.

European voting patterns are relatively stable, with slightly higher National support
and lower Labour support compared to the overall electorate.

---

## The Structural Rules

### Rule 25: The Migration Backlash

**High immigration boosts NZ First. But only from opposition.**

Net migration is positively correlated with NZ First polling support (r = +0.39,
p < 0.001). This is the expected direction: NZ First capitalises on anti-immigration
sentiment, which is strongest when immigration is visibly high. The correlation
holds within both National (r = +0.38) and Labour (r = +0.39) governments,
ruling out an era confound. Last year's migration also predicts this year's NZ
First support (r = +0.30, p < 0.001), confirming a genuine lagged effect.

The era breakdown reveals an important nuance:

| Period | NZ First role | NZF support | Net migration |
|--------|---------------|-------------|---------------|
| 2015-17 | Opposition | **8.0%** | **71k/yr** |
| 2018-20 | In coalition | 3.1% | 52k/yr |

During 2015-17, record migration and NZ First's opposition status combined to
produce their highest sustained polling. But when NZ First entered government in
2017 and migration remained high, their support collapsed to 3.1%. Being part of
the government presiding over high immigration kills the protest vote.

This is a textbook example of issue ownership (Rule 16): NZ First owns the
immigration issue, benefits when immigration is salient, but cannot capitalise when
they share responsibility for governing.

---

### Rule 26: The Mobile Voter

**A quarter to a third of voters switch party between elections. NZ voters are unusually mobile.**

Across all NZES surveys with previous-vote data, the overall switching rate is
approximately 25-35%, varying by election context:

- Higher in "change" elections: 1999, 2017, 2023
- Lower in status-quo elections: 2002, 2011, 2014

Most switching occurs *within* ideological blocs — Labour to Green, National to ACT,
and vice versa. But cross-bloc switching (right-to-left and left-to-right) accounts
for a non-trivial 15-25% of all flows.

**Who switches?** Compared to loyalists, switchers are:
- Younger
- More ideologically centrist (closer to 5.0 on the left-right scale)
- More likely to hold a university degree
- Responsive to economic perceptions and leader evaluations

**Party retention rates** follow a predictable pattern: National and Labour retain
60-75% of their voters between elections, while minor parties retain only 40-60%.
NZ First has particularly volatile retention, consistent with its role as a
protest/populist vehicle.

This high voter mobility explains NZ's capacity for dramatic election swings. When
a third of the electorate is open to switching and a significant minority will
cross the centre line, large aggregate movements (like Labour's +19pp in 2020 or
-23pp in 2023) become structurally possible without requiring a fundamental
realignment.

---

### Rule 27: The Approval Barometer

**Government performance ratings are the single strongest predictor of incumbent support. Stronger than housing costs, consumer confidence, or any economic indicator.**

The Ipsos NZ Issues Monitor (2017-2025) asks respondents to rate government performance
on a 0-10 scale. This simple measure correlates with incumbent party vote share at
r = **0.827** (p < 0.0001, n = 29 months) — by far the strongest bivariate relationship
found in this project, surpassing housing costs (r = -0.62), consumer confidence
(r = +0.47), and every other economic indicator.

A simple regression of incumbent vote share on the Ipsos mean score achieves R² = **0.684**
— a single survey question explains more than two-thirds of the variation in incumbent
polling. Each +1 point on the government performance scale corresponds to approximately
**+5.7 percentage points** in incumbent party vote share (p < 0.001).

The relationship holds across both government eras in the data:

| Government Era | r | p-value | n |
|----------------|---|---------|---|
| Labour (2017-2023) | 0.795 | < 0.001 | 19 |
| National (2023-2025) | 0.893 | < 0.001 | 10 |

Time-lagged cross-correlation shows the strongest correlation at **lag = 0** (contemporaneous).
Government performance ratings do not lead or lag incumbent polling — both move together,
likely responding to the same underlying conditions. This mirrors the consumer confidence
finding (Rule 9) and is consistent with the perception-mediated channel identified in
Rule 10: voters' subjective evaluation of the government, not objective economic statistics,
drives their vote intention.

The practical interpretation is stark: the Ipsos 0-10 score functions as NZ's closest
equivalent to a US presidential approval rating. When it drops below ~5.0, the incumbent
is in serious trouble. The Ardern government's decline from 7.6 (May 2020, post-COVID
lockdown) to 4.5 (August 2023) maps almost perfectly onto Labour's polling collapse
from the mid-50s to the low-30s. The current National-led government's decline from 4.7
(September 2023) to 3.9 (October 2025) similarly tracks the steady erosion in National's
polling from the high 30s to the low 30s.

Combined with Rule 9 (consumer confidence) and Rule 10 (partisan filter), the approval
barometer completes the subjective evaluation picture: voters experience the economy
through a partisan lens (Rule 10), this feeds into aggregate sentiment measures (Rule 9),
and it crystallises as an overall government performance verdict (Rule 27) that tracks
voting intention at r = 0.83.

---

### Rule 28: The Competence Signal

**When voters think a party is "best to manage" an issue, they vote for that party. The correlation is strongest on the economy.**

The Ipsos party capability question (2023-2025, 10 waves) asks which party is best placed
to manage each of six issues. Across all issues and both major parties combined, the
"best to manage" score correlates with actual party vote share at r = **0.585** (p < 0.0001,
n = 120 issue-party-wave observations).

The strength of the competence-vote link varies dramatically by issue:

| Issue | National r | Labour r | n |
|-------|-----------|---------|---|
| The Economy | 0.873 | 0.892 | 10 |
| Crime/Law & Order | 0.777 | 0.885 | 10 |
| Inflation/Cost of Living | 0.675 | 0.717 | 10 |
| Housing | 0.515 | 0.573 | 10 |
| Healthcare | 0.374 | 0.623 | 10 |
| Unemployment | -0.731 | -0.787 | 10 |

The economy and crime show near-perfect tracking between capability perception and vote
share (r = 0.87-0.89). When voters increasingly see Labour as "best to manage the economy,"
Labour's poll numbers rise almost one-for-one. The same applies to National.

The unemployment anomaly (negative r) deserves attention: both parties' capability scores
on unemployment run *inversely* to their vote share. This may reflect that unemployment
becomes salient precisely when a party is losing support for other reasons, or that voters
attribute unemployment management capability to whichever party they do *not* currently
support (a protest dynamic).

This finding complements Rule 17 (The Leader Premium). Where Rule 17 shows that overall
leader evaluations drive vote choice more than ideology, Rule 28 shows that *issue-specific*
competence perception also tracks voting intention — especially on the economy and crime,
the two issues where National has traditionally held ownership (Rule 16). The mechanism
is consistent: voters choose the party they perceive as competent, and competence
perception is issue-domain-specific.

The 10-wave limitation means these correlations should be treated as indicative rather
than definitive. But the consistency across issues and the high magnitude of the
economy and crime correlations suggest a robust underlying pattern.

---

## Rules Not Confirmed

### MMP Strategic Squeeze — Not Found at Aggregate Level

The shift to MMP increased volatility and created a permanent minor-party sector
(~19% combined support), but the expected "strategic squeeze" — where minor party
support declines as elections approach — does not occur (coefficient: -0.0002 per
day, p = 0.75). NZ voters do not abandon minor parties for tactical major-party
votes as polling day nears. However, individual-level strategic desertion is
substantial (~26% for minor party supporters) — it is simply constant throughout
the cycle rather than concentrated near elections (see Rule 19).

### Dealignment — Not Found (Nothing to Dealign From)

Dalton's (2000) prediction that demographics should lose predictive power over time
is not confirmed in NZ — but only because demographics never explained more than 2%
of the variance (see Rule 18). NZ has not dealigned because it was never strongly
aligned along demographic lines in the first place.

### Myopic Retrospection — Inconclusive

Voters may weight recent economic conditions slightly more than earlier ones, but
the evidence is too thin to confirm (see Rule 13). NZ's small number of elections
limits statistical power for this test.

### Unemployment Asymmetry — Unclear

Unlike GDP (where contraction hurts but growth doesn't help), unemployment does not
show a clean asymmetric pattern. The raw bivariate correlation is confounded: it runs
positive (r = +0.11) because National governments coincide with both lower unemployment
and higher incumbent support. After controlling for government identity, unemployment
is significant in the expected direction (β = -1.6, p < 0.001), but the within-
government results are contradictory (negative under National, positive under Labour),
suggesting the unemployment signal is entangled with government-era effects rather
than operating as an independent voter consideration.

### Dairy Commodity Prices — Trivially Small

The FAO Dairy Price Index is statistically significant (p = 0.005) but the effect
is negligible: β = -0.02 per 1% price change, Adj R² barely above baseline (0.073
vs 0.070). The correlation vanishes within individual governments (National r = -0.06
ns, Labour r = -0.09 ns). NZ's key export commodity does not meaningfully affect
incumbent polling — whatever effects dairy prices have on the economy are too
diffuse and slow to register in political support.

### Exchange Rate — Confounded with Government Era

The NZD/USD shows a raw positive correlation with incumbent support (r = +0.38),
but the within-government analysis reveals opposite signs (National r = +0.67,
Labour r = -0.26). The NZD was generally stronger during the National era (2008-2017),
so the raw correlation captures government identity, not a genuine FX→voting
channel. The import price mechanism is also weak (NZD vs CPI: r = -0.12).

---

## The Big Picture

The 28 rules, taken together, describe a system where:

1. **Subjective evaluation dominates** (Rules 5-14, 27-28). Housing costs, consumer
   confidence, and interest rates collectively explain over 60% of incumbent support
   variation. But the single strongest predictor is even simpler: the Ipsos government
   performance rating (r = 0.83, R² = 0.68). GDP is irrelevant. The economy operates
   through subjective perception, not objective statistics, and perception is filtered
   through partisan identity (Rule 10). Government approval is the ultimate summary
   statistic of that perception.

2. **Leaders matter more than ideology** (Rules 15-17). NZ elections are substantially
   personality-driven. Leader evaluations outperform ideological proximity as vote
   predictors from 2005 onward. This explains why leadership changes produce large
   polling shifts (Rule 3) and why fatigue rates vary so dramatically across
   governments (Rule 11) — the leader *is* the government's political brand.

3. **The issue agenda shapes outcomes** (Rules 16, 28). When voters care about the economy
   and tax, National benefits. When they care about health and housing, Labour
   benefits. The ownership effect has grown from single digits to 40+ percentage
   points since MMP began. Monthly Ipsos tracking confirms this operates continuously,
   not just at elections: cost-of-living salience tracks Labour's decline at r = -0.89.
   Perceived issue competence (Rule 28) tracks vote intention at r = 0.59 pooled,
   rising to r = 0.87-0.89 on the economy specifically. Elections are partly won and
   lost on which party's issues are salient — and which party voters trust to manage them.

4. **The electorate is centrist and self-correcting** (Rules 15, 20). NZ has not
   polarized. The mean left-right position has been stable for 30 years because the
   public thermostically adjusts against whichever direction the government pushes.
   This dynamic equilibrium, not fixed preferences, produces the centrist stability.

5. **Demographics matter less than anywhere else** (Rules 18, 21-24). Age, gender,
   education, income, and ethnicity collectively explain 1-2% of vote choice.
   Ideology adds 30-40%. NZ voters choose based on what they think, not who they
   are. The demographic patterns (age reversal, gender gap, diploma divide) are real
   but marginal — they operate at the edges of a system driven by ideology,
   leaders, and economic perception.

6. **Voters are mobile and strategic** (Rules 19, 26). A quarter to a third of voters
   switch between elections. A quarter of minor-party supporters strategically desert.
   Thirty percent split their ticket. NZ's electorate is fluid, pragmatic, and
   capable of producing dramatic swings — not because of fundamental realignment,
   but because a large pool of centrist voters responds to changing leaders, issues,
   and economic conditions.

---

## Summary Table

| # | Rule | Key statistic | Confidence |
|---|------|---------------|------------|
| 1 | The Seesaw | Nat-Lab r = -0.51 | High (N = 1,016) |
| 2 | The Rubber Band | 67% extreme poll reversion | High (N = 1,016) |
| 3 | The Honeymoon | Ardern +8.9pp | High (multiple events) |
| 4 | The Rally | Delta +1.6pp Lab, ChCh -1.0pp Nat | Low (few events, mixed) |
| 5 | The Cost of Living | Housing costs Adj R² = 0.48 (best economic predictor) | High (N = 870) |
| 6 | The Wealth Effect | House prices β = +0.18; housing costs β = -1.18 (opposing) | High (N = 870) |
| 7 | The Rate Squeeze | Interest rate β = -0.81 in housing trinity (R² = 0.44) | High (N = 870) |
| 8 | The Negativity Bias | Contraction 39.5% vs growth 42.4% (p < 0.001) | High (N = 1,011) |
| 9 | The Confidence Channel | CCI r = +0.47, Adj R² = 0.31 | High (N = 1,015) |
| 10 | The Partisan Filter | 0.5-1.0 pts perception gap, all years p < 0.001 | High (N = 35,000+) |
| 11 | The Cost of Ruling | -1.0 to -6.7 pp/yr; context > coalition type | High (N = 1,015) |
| 12 | The Banker's Model | Prospective R² > Retrospective R² in 2/3 elections | Medium (3 elections) |
| 13 | The Myopic Voter | Late-term r = 0.25 vs early r = 0.24 (weak) | Low (10 elections) |
| 14 | The 60% Model | 6 indicators + controls: R² = 0.61 | High (N = 870) |
| 15 | The Thermostat | 7/8 shifts in predicted direction | High (8 transitions) |
| 16 | The Issue Agenda | +27.5pp ownership gap; Ipsos cost-of-living r = -0.89 with Labour | High (7 elections + 27 months) |
| 17 | The Leader Premium | Valence > ideology in 6/8 elections | High (8 elections) |
| 18 | The Demographic Floor | Demo R² = 0.01-0.02; + ideology R² = 0.22-0.42 | High (9 elections) |
| 19 | The Strategic Voter | Minor party desertion 25.8% vs major 10.0%; 30% split-ticket | High (6 elections) |
| 20 | The Centrist Electorate | Mean L-R stable at 5.0-5.7 | High (8 elections) |
| 21 | The Age Reversal | ~30pp gradient swing, 1996-2017 | High (10 elections) |
| 22 | The Gender Gap | ~7-8pp, every election | High (10 elections) |
| 23 | The Late Diploma Divide | Reversed after 2017 | Medium (fewer elections post-reversal) |
| 24 | The Ethnic Anchor | Māori 20-40pp more Labour | High (10 elections) |
| 25 | The Migration Backlash | NZF vs migration r = +0.39, within both govts | High (N = 924) |
| 26 | The Mobile Voter | 25-35% switching rate | High (10 elections) |
| 27 | The Approval Barometer | Ipsos govt rating r = 0.827 with incumbent vote (R² = 0.68) | High (N = 29 months) |
| 28 | The Competence Signal | Capability score r = 0.585 pooled; economy r = 0.87-0.89 | Medium (10 waves) |

---

## Data Sources

| Source | Coverage | N |
|--------|----------|---|
| Party vote polls (Wikipedia) | 1990-2025 | 1,016 polls |
| Stats NZ GDP (quarterly, seasonally adjusted) | 1987-2025 | 152 quarters |
| Stats NZ CPI (quarterly, component-level) | 1914-2025 | 420+ quarters |
| BIS real house price index (via FRED) | 1962-2025 | 254 quarters |
| OECD unemployment rate (via FRED) | 1986-2025 | 159 quarters |
| OECD consumer confidence index | 1993-2025 | 451 months |
| NZ 3-month interbank rate (OECD via FRED) | 1973-2025 | 625 months |
| FAO Dairy Price Index | 1990-2026 | 433 months |
| NZD/USD exchange rate (FRED) | 1971-2026 | 14,376 days |
| World Bank net migration | 1960-2024 | 65 years |
| NZ Election Study (NZES) | 1996-2023, 10 elections | 35,107 respondents |
| Ipsos NZ Issues Monitor | Sep 2017 – Oct 2025 | 30 waves (govt performance, issue salience, party capability) |

## Methodology

All findings are correlational. No causal identification is attempted. Statistical
methods include Pearson and Spearman correlations (with heteroskedasticity-consistent
standard errors where applicable), OLS regression with distributed lags, logistic
regression for binary vote-choice models, and two-sample t-tests for group comparisons.
Housing cost and house price findings include robustness checks controlling for
incumbent party identity and secular time trends. The full economic model uses
HC1 standard errors to account for heteroskedasticity. NZES survey data is subject
to harmonization limitations as variable names, coding schemes, and value labels
differ across election years.

---

*Generated from analysis of NZ polling, economic, and survey data. February 2026.*
