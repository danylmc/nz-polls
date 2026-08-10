# Claude Code Project Instructions

## Project Goal
Analyze NZ polling data to find predictors and correlates of polling shifts using political science literature and statistical analysis.

## What's Been Done
1. Scraped 1,020 party vote polls from Wikipedia (1993-2026)
2. Scraped 385 Preferred PM polls from Wikipedia (1997-2026; no PM table exists for the 2005 cycle)
3. Fetched economic data from World Bank API (GDP, unemployment, CPI 1961-2024)
4. Created events timeline with leadership changes, crises, scandals
5. Created visualization script with 4 cross-election trend graphs
6. Ran statistical analysis with 12 hypothesis tests (8 polling + 3 Ipsos + 1 mortgage)
7. Generated findings report (`reports/findings.md`)
8. Scraped and extracted Ipsos NZ Issues Monitor data (24 PDFs, 2018-2025)
9. Integrated Ipsos data into analysis: govt performance, issue salience, party capability vs polling
10. Analysed RBNZ mortgage rates (hb20) vs party polling — Granger-causal incumbent punishment effect
11. Petrol prices vs incumbent polling — Granger-causal, stronger than mortgage rates (r=-0.348)
12. Consumer confidence vs incumbent polling — Granger-causal, 50% mediated through approval
13. Pollster house effects & poll accuracy — no shy Tory effect, Green overestimated, polls converge
14. PM preference lead/lag analysis — contemporaneous (no lead), PM margin explains 58% of vote margin
15. House prices vs polling — null finding (wealth effect cancels affordability grievance)
16. Cost of ruling analysis — no coalition vs majority decay difference (Paldam 1991 not confirmed)
17. Net migration vs NZ First — marginal positive association (r=0.340, p=0.057)
18. NZES retrospective vs prospective economic voting — retro wins 4/5, Fiorina (1981) confirmed
19. NZES valence vs position voting — position wins 4/4 elections, L-R proximity r=0.698 mean
20. NZES strategic voting under MMP — split-ticket common (30%), desertion modest (15-25%)
21. Crime statistics vs crime salience — actual crime predicts salience (r=0.569), unlike economics
22. Google Trends NZ as salience proxy — cost-of-living searches predict Ipsos (r=0.482)
23. Rental price index vs polling — strong level correlation (r=-0.731) but fails Granger & Newey-West; demoted to associational
24. Multivariate model — approval dominates (R²=0.675 alone), massive multicollinearity, parsimonious = approval+petrol (R²=0.767)
25. Forecasting model — polling is a random walk; naïve baseline (MAE 2.64pp) beats all models; no indicator predicts vote changes
26. Ipsos visualizations — govt performance vs vote, salience heatmap, party capability trends, salience time series
27. Myopic retrospection — election-year GDP growth (r=+0.588) predicts incumbent fate; early-term growth irrelevant (r=-0.114)
28. Dealignment analysis — no overall dealignment; Maori identity strengthening as predictor; age effect reversed; partial class dealignment
29. Robustness checks — Newey-West HAC SEs for all key regressions + rental Granger test; rental inflation demoted (fails both); all other pocketbook findings survive
30. Adversarial critique and rebuttal — systematic challenge to methodology, specific rules, and data quality; rebuttal addresses each point with concessions and defences

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `scraper.py` | Wikipedia party vote scraper | Complete |
| `pm_scraper.py` | Preferred PM polling scraper (1999–2026 cycles) | Complete |
| `pm_refetch.py` | Wikipedia page fetch/cache via MediaWiki API | Complete |
| `pm_history_graph.py` | Preferred PM ratings 1997–2026 chart | Complete |
| `economic_scraper.py` | World Bank API economic data | Complete |
| `events.py` | Event timeline data | Complete |
| `analysis.py` | Statistical analysis (11 tests) | Complete |
| `mortgage_analysis.py` | Mortgage rate vs polling analysis | Complete |
| `raw_data/hb20.xlsx` | RBNZ 2yr fixed mortgage rates (monthly) | Complete |
| `visualize.py` | Trend graphs | Complete |
| `data/*.json` | Party vote polling data | Complete |
| `data/*_pm_polling.json` | PM preference polling data (per cycle) | Complete |
| `data/pm_polling_all.csv` | All PM polls, long format, deduplicated | Complete |
| `raw_data/wikipedia_polling/*.html` | Cached Wikipedia polling pages | Complete |
| `data/economic/*.csv` | Economic indicators | Complete |
| `data/ipsos_pdfs/*.pdf` | Ipsos Issues Monitor PDFs (24 editions) | Complete |
| `data/ipsos_issue_salience.csv` | Issue importance (28 waves, 10 issues) | Complete |
| `data/ipsos_govt_performance.csv` | Government rating 0-10 (30 waves) | Complete |
| `data/ipsos_party_capability.csv` | Party best to manage issue (10 waves, 6 issues) | Complete |
| `data/ipsos_report_links.json` | Ipsos website scrape / PDF URLs | Complete |
| `petrol_vs_incumbent.py` | Petrol price vs polling analysis | Complete |
| `consumer_confidence_analysis.py` | OECD CCI vs polling analysis | Complete |
| `house_effects_analysis.py` | Pollster bias & poll accuracy | Complete |
| `poll_accuracy_analysis.py` | Final poll error & shy Tory test | Complete |
| `pm_leadlag_analysis.py` | PM preference lead/lag analysis | Complete |
| `house_prices_analysis.py` | BIS house prices vs polling | Complete |
| `cost_of_ruling_analysis.py` | Coalition vs majority decay | Complete |
| `migration_analysis.py` | Net migration vs NZ First | Complete |
| `raw_data/weekly-table.csv` | MBIE weekly petrol prices | Complete |
| `data/consumer_confidence.csv` | OECD consumer confidence (monthly) | Complete |
| `data/house_prices_real.csv` | BIS real house price index (quarterly) | Complete |
| `data/net_migration.json` | World Bank net migration (annual) | Complete |
| `nzes_economic_voting.py` | Retrospective vs prospective voting (NZES) | Complete |
| `nzes_valence_position.py` | Valence vs position voting (NZES) | Complete |
| `nzes_strategic_voting.py` | Strategic voting under MMP (NZES) | Complete |
| `crime_salience_analysis.py` | Crime statistics vs crime salience | Complete |
| `google_trends_analysis.py` | Google Trends NZ as salience proxy | Complete |
| `rental_prices_analysis.py` | Rental CPI vs polling | Complete |
| `data/google_trends_nz.csv` | Google Trends monthly index (5 keywords) | Complete |
| `multivariate_model.py` | Multivariate vote model & forecasting | Complete |
| `ipsos_visualizations.py` | Ipsos data visualizations (4 charts) | Complete |
| `myopic_retrospection.py` | Myopic retrospection (Achen & Bartels 2016) | Complete |
| `dealignment_analysis.py` | Dealignment analysis (Dalton 2000) | Complete |
| `robustness_checks.py` | Newey-West HAC SEs + rental Granger test | Complete |
| `reports/adversarial_critique.md` | Systematic critique of methodology and findings | Complete |
| `reports/rebuttal.md` | Response to adversarial critique | Complete |
| `reports/guide_to_nz_polling.md` | Plain-language guide (19 findings) | Complete |
| `reports/findings.md` | Analysis report (30 analyses) | Complete |

---

## Key Findings Summary

| Hypothesis | Result | p-value |
|------------|--------|---------|
| National-Labour zero-sum | **Confirmed** (r = -0.310) | < 0.001 |
| Mean reversion | **Confirmed** (67% revert) | N/A |
| Reversal effect | **Confirmed** (r = -0.51) | < 0.001 |
| Third party squeeze | Not detected | 0.682 |
| Economic voting (macro) | Not significant | > 0.05 |
| Govt performance → incumbent vote | **Confirmed** (r = 0.827, R² = 0.68) | < 0.001 |
| Issue ownership affects vote | **Confirmed** (10 issues with sig. links) | < 0.05 |
| Perceived competence → vote | **Confirmed** (r = 0.585 pooled) | < 0.001 |
| Mortgage rate → incumbent punishment | **Confirmed** (Granger-causal, quarterly r = -0.307) | < 0.001 |
| Petrol price → incumbent punishment | **Confirmed** (Granger-causal, quarterly r = -0.348) | < 0.001 |
| Consumer confidence → incumbent support | **Confirmed** (Granger-causal, quarterly r = +0.151) | 0.002 |
| Shy Tory effect | **Not confirmed** (polls overestimate National +0.82pp) | 0.036 |
| Poll convergence | **Confirmed** (r = 0.24–0.36 for days_out vs |error|) | < 0.001 |
| PM preference leads party vote | **Not confirmed** (contemporaneous, lag 0 strongest) | — |
| PM margin predicts vote margin | **Confirmed** (r = 0.763, R² = 0.58) | < 0.001 |
| House prices → incumbent support | **Not significant** (r = -0.069) | 0.427 |
| Coalition decay faster (Paldam 1991) | **Not confirmed** (majority decayed faster) | 0.430 |
| Immigration → NZ First (Hangartner) | **Marginal** (r = 0.340) | 0.057 |
| Mortgage rate → opposition benefit | **Confirmed** (National in opp: r = +0.370) | < 0.001 |
| Mortgage rate mediated by approval | **Confirmed** (68% mediated, Sobel z = -3.32) | 0.001 |
| Cost-of-living salience drives approval | **Confirmed** (r = -0.869, R² = 0.74 alone) | < 0.001 |
| Retrospective voting (Fiorina 1981) | **Confirmed** (all 10 elections significant, mean r = 0.376) | < 0.001 |
| Position > valence (Stokes 1963) | **Confirmed** (position wins 4/4 elections, R² = 0.586 vs 0.468) | < 0.001 |
| Strategic desertion (Cox 1997) | **Partial** — split-ticket (30%) instead of desertion | 0.24 |
| Crime reality → salience | **Confirmed** (r = 0.569, unlike economics) | 0.007 |
| Google Trends → Ipsos salience | **Partial** — cost-of-living only (r = 0.482) | 0.009 |
| Rental inflation → incumbent | **Associational only** (r = -0.731 levels; fails Granger & Newey-West p=0.112) | 0.112 (NW) |
| Multivariate model adds value | **Not confirmed** (approval dominates, VIF > 30) | N/A |
| Forecasting beats naïve baseline | **Not confirmed** (naïve MAE 2.64pp beats all models) | > 0.2 |
| Myopic retrospection (Achen & Bartels) | **Partial** (late-term r=+0.588, early r=-0.114) | 0.096 |
| Dealignment (Dalton 2000) | **Not confirmed** (pseudo-R² stable/increasing, Maori strengthening) | 0.114 |

Notable event effects:
- **Ardern honeymoon (2017)**: +8.9% Labour boost
- **Delta lockdown (2021)**: +1.6% Labour, -4.5% National

Notable Ipsos findings:
- Each +1pt govt rating ≈ +5.7pp incumbent vote (strongest single predictor)
- Cost-of-living salience strongly hurts Labour (r = -0.885)
- Healthcare salience helps Labour (r = 0.670); crime helps National (r = 0.403)
- Economy capability perception tracks vote almost 1:1 (r = 0.87–0.89)

Notable mortgage rate findings:
- Rising mortgage rates Granger-cause incumbent vote declines (F=13.8 at lag 1)
- Effect is a zero-sum transfer: govt bloc loses ≈ what opposition bloc gains
- **Major-party phenomenon only** — minor parties (Green, NZ First, ACT) unaffected
- Labour hurt when in govt (r=-0.212); National benefits when in opposition (r=+0.370)
- Rate-rise episodes (2021–23): National +3.4pp, Labour -3.8pp, minor parties flat

Causal chain (Analysis 13):
- **Mortgage rates → cost-of-living concern → govt disapproval → vote shift**
- 68% of mortgage→polling effect is mediated through govt approval (Sobel z=-3.32, p=0.001)
- Cost-of-living salience alone explains 74% of govt approval variance
- Two issue salience vars (cost-of-living + unemployment) explain 88% of approval (R²=0.88)
- Direction flips: under Labour, rate rises hurt; under National, rate *falls* hurt (signal recession)

Notable petrol price findings:
- Petrol prices Granger-cause incumbent vote shifts (F=12.0, p=0.0006 at lag 1)
- Stronger quarterly effect than mortgage rates (r=-0.348 vs r=-0.307)
- Same major-party-only pattern: Labour hurt in govt (r=-0.315), National gains in opp (r=+0.329)
- 49% mediated through approval (vs 68% for mortgages) — petrol has more direct pocketbook impact
- Petrol effect is contemporaneous (lag 0 strongest) vs mortgage rate ~1 month delay

Notable consumer confidence findings:
- Consumer confidence Granger-causes incumbent vote (F=9.3, p=0.002)
- Positive relationship: rising confidence helps incumbent (r=+0.389 levels, +0.151 changes)
- 50% mediated through govt approval; the other 50% is a direct "feel-good" effect
- Confirms "subjective perception beats objective reality" thesis

Notable poll accuracy findings:
- No shy Tory effect in NZ — polls actually overestimate National by +0.82pp
- Green consistently overestimated (+1.03pp), NZ First underestimated (-0.74pp)
- Polls converge: error shrinks significantly closer to election
- Final poll MAE: National 2.48pp, Labour 1.72pp (final 14 days)
- Winner called correctly 9/10 elections (only 2005 wrong, by 2pp)

Notable PM preference findings:
- PM margin vs vote margin: r=0.763, R²=0.58
- 1pp PM preference lead ≈ 0.32pp party vote lead (attenuated)
- Contemporaneous relationship (lag 0), not lead/lag — no presidentialisation

Notable NZES individual-level findings:
- Retrospective voting confirmed in ALL 10 elections (mean r=0.376), prospective loses 4/5
- Sociotropic > egotropic: national economy perception matters more than personal finances
- Partisan perceptual screen: +0.68 scale point gap between incumbent and opposition voters (all years p<0.001)
- L-R positional voting is very strong (mean r=0.698) and strengthening over time
- Position dominates valence (competence) in all 4 tested elections (2014-2023)
- Split-ticket voting common under MMP (~30%), but outright party desertion is modest (15-25%)
- Minor party voters lend electorate votes to majors ~50% of the time

Notable crime & media findings:
- Actual crime DOES predict crime salience (r=0.569) — unlike economics where reality has no power
- Crime salience → govt approval: r=-0.858 (very strong), hurts Labour (r=-0.720), helps National (r=+0.469)
- Google "cost of living" searches predict Ipsos salience (r=0.482), but no lead/lag over polls
- Google "healthcare" searches predict National decline (r=-0.569) — issue ownership in real-time

Notable rental price findings:
- **Rental inflation shows strong level correlation with incumbent vote** (r=-0.731) but fails Granger causality and Newey-West — likely associational, not causal
- Rental and mortgage are completely uncorrelated (r=0.052) — separate channels, different demographics
- Rising rents hurt Labour (r=-0.233) but help both Green (r=+0.256) and ACT (r=+0.246)

Notable multivariate & forecasting findings:
- **Government approval alone explains 67.5% of incumbent vote variance** — adding 6 economic indicators barely helps
- All pocketbook indicators are massively collinear (VIF 30-2500) — they all measure the same underlying process
- Parsimonious model = approval + petrol price (R²=0.767, Adj R²=0.745, lowest AIC)
- **No indicator predicts quarter-to-quarter vote changes** (all p > 0.2) — polling is a random walk
- Naïve baseline (this Q = next Q) MAE = 2.64pp, beats approval-only (3.84pp) and approval+CCI (3.20pp)
- Current indicators predict next-quarter vote *levels* (r=0.70 for approval) but this is just autocorrelation

Notable null findings:
- House prices have NO significant effect on incumbent polling (wealth effect cancels affordability)
- Coalition govts do NOT decay faster than majority govts (Paldam 1991 rejected)
- Net migration → NZ First only marginally significant (r=0.340, p=0.057)
- Google Trends do NOT lead polls — contemporaneous movement only
- Multivariate model does NOT outperform approval alone (Adj R² barely improves)
- Forecasting models do NOT beat the naïve baseline
- Dealignment NOT confirmed overall — demographic pseudo-R² stable/increasing since 1996

Notable myopic retrospection findings:
- Election-year GDP growth correlates with incumbent fate (r=+0.588, p=0.096); early-term growth does not (r=-0.114)
- Pattern is **suggestive** of myopic retrospection (Achen & Bartels 2016) but N=9 with wide CI (~-0.10 to +0.90)
- 2008 and 2023 are consistent but may be driving the result — below conventional significance threshold

Notable dealignment findings:
- **Overall dealignment NOT confirmed** — pseudo-R² of demographics→vote actually increased (0.032→0.044)
- **Age effect reversed**: young→National in 1990s, old→National from 2005+. This is realignment, not dealignment
- **Maori identity is the strongest demographic predictor** (OR=0.14-0.39 for National vote) and strengthening
- Gender and education effects weakening toward zero — partial dealignment on these cleavages
- NZ is experiencing **realignment** (Ford & Jennings 2020) not dealignment (Dalton 2000)

---

## How to Run

```bash
cd /mnt/d/data/nz-polls

# Run core analysis
python analysis.py                      # Main 13 analyses → reports/findings.md
python mortgage_analysis.py             # Mortgage rates vs polling
python petrol_vs_incumbent.py           # Petrol prices vs polling
python consumer_confidence_analysis.py  # Consumer confidence vs polling
python house_effects_analysis.py        # Pollster house effects
python poll_accuracy_analysis.py        # Poll accuracy & shy Tory test
python pm_leadlag_analysis.py           # PM preference lead/lag
python house_prices_analysis.py         # House prices vs polling
python cost_of_ruling_analysis.py       # Coalition vs majority decay
python migration_analysis.py           # Net migration vs NZ First
python nzes_economic_voting.py         # Retro vs prospective voting (NZES)
python nzes_valence_position.py        # Valence vs position voting (NZES)
python nzes_strategic_voting.py        # Strategic voting under MMP (NZES)
python crime_salience_analysis.py      # Crime stats vs crime salience
python google_trends_analysis.py       # Google Trends as salience proxy
python rental_prices_analysis.py       # Rental CPI vs polling
python multivariate_model.py          # Multivariate model & forecasting
python ipsos_visualizations.py        # Ipsos data visualizations
python myopic_retrospection.py        # Myopic retrospection (Achen & Bartels)
python dealignment_analysis.py        # Dealignment analysis (Dalton 2000)

# Generate visualizations
python visualize.py
```

---

## Dependencies
Installed in venv:
- requests, beautifulsoup4, lxml (scraping)
- pandas, numpy (data)
- matplotlib, seaborn (visualization)
- scipy, statsmodels (statistical analysis)

---

## Political Science Literature Context

**Key predictors from literature:**
1. **Economic fundamentals** - GDP, unemployment, inflation affect incumbent support
2. **Leadership honeymoon** - New leaders get polling bounce
3. **Rally-round-flag** - Crises can boost incumbent (short-term)
4. **Incumbent fatigue** - Support erodes after 2+ terms
5. **Strategic voting** - Minor parties squeezed near elections under MMP
6. **Government approval** - Subjective performance ratings predict vote intention
7. **Issue ownership** - Parties benefit when "their" issues become salient
8. **Perceived competence** - "Best party to manage X" tracks voting intention
9. **Pocketbook voting** - Household-level economic pain (e.g. mortgage costs) punishes incumbents

**NZ Findings vs. Literature:**
- National-Labour zero-sum: **Confirmed** as expected
- Leadership honeymoon: **Confirmed** (Ardern 2017 most dramatic)
- Rally effect: **Partial** (Delta lockdown showed incumbent boost)
- Third party squeeze: **Not confirmed** in NZ data
- Economic voting (macro): **Not significant** (possibly due to small N of 11 elections)
- Government approval → vote: **Confirmed** (r = 0.827) — strongest single predictor found
- Issue ownership: **Confirmed** — cost-of-living hurts Labour, crime helps National, healthcare helps Labour
- Perceived competence → vote: **Confirmed** (r = 0.585 pooled) — especially strong on economy and crime
- Pocketbook voting (mortgage rates): **Confirmed** — Granger-causal link, ~1 month delay
- Pocketbook voting (petrol prices): **Confirmed** — Granger-causal, even stronger (r=-0.348 quarterly)
- Consumer confidence → vote: **Confirmed** — Granger-causal, 50% mediated through approval
- Shy Tory effect: **Not confirmed** — NZ polls overestimate National (+0.82pp)
- Presidentialisation (Poguntke & Webb 2005): **Not confirmed** — PM pref contemporaneous with party vote, not leading
- Wealth effect (house prices): **Not detected** — wealth and affordability effects cancel
- Cost of ruling (Paldam 1991): **Not confirmed** — majority govt decayed faster than coalitions
- Immigration → NZ First (Hangartner 2019): **Marginal** (r=0.340, p=0.057)
- Retrospective voting (Fiorina 1981): **Confirmed** — retro wins 4/5 vs prospective, mean r=0.376
- Prospective voting (MacKuen 1992): **Partial** — weaker than retrospective in head-to-head
- Position voting (Stokes 1963): **Confirmed** — L-R proximity wins 4/4 vs competence, mean r=0.698
- Valence voting (Clarke 2009): **Confirmed but secondary** — competence matters but position dominates
- Strategic desertion (Cox 1997): **Not confirmed** — MMP produces split-tickets, not desertion
- Crime drives perception: **Confirmed** — actual crime predicts salience (r=0.569), unlike economics
- Rental pocketbook effect: **Associational only** — strong level correlation (r=-0.731) but fails Granger causality and Newey-West HAC (p=0.112); likely spurious regression
- Myopic retrospection (Achen & Bartels 2016): **Suggestive** — election-year GDP (r=+0.588) outpredicts early-term (r=-0.114), but N=9 with p=0.096 and wide confidence interval (~-0.10 to +0.90); directionally consistent but below significance threshold
- Dealignment (Dalton 2000): **Not confirmed** — demographic pseudo-R² stable/increasing, driven by strengthening Maori identity cleavage
- Realignment (Ford & Jennings 2020): **Consistent** — old cleavages (class-age) weakening, new cleavage (ethnic identity) strengthening

**Generalised Rules of NZ Polling Dynamics:**

1. **The causal chain is: economic conditions → issue salience → government approval → vote intention.** This is the central finding. Objective economic variables (mortgage rates, petrol prices) don't directly move votes — they work by making cost-of-living a salient public concern, which erodes government approval, which shifts voting intention. 68% of the mortgage→vote effect and 49% of the petrol→vote effect are mediated through approval. Cost-of-living salience alone explains 74% of approval variance.
2. **Incumbent punishment is the dominant mechanism.** Voters blame the government for economic pain regardless of whether it controls the cause. Mortgage rates, petrol prices, and inflation all feed into "cost-of-living concern" which is the master variable for government approval.
3. **Punishment flows to the main opposition, not minor parties.** Mortgage rate pain, petrol price spikes, cost-of-living salience, and government disapproval all transfer vote share from the incumbent major party to the opposition major party. Green, NZ First, and ACT are largely insulated from these swings.
4. **Subjective perception beats objective reality.** Government approval ratings (r=0.827), consumer confidence (r=0.389), and "best party to manage X" perceptions (r=0.585) predict vote intention far better than actual GDP, unemployment, or inflation data (all non-significant). Two issue salience variables explain 88% of approval; macro fundamentals explain nearly nothing.
5. **Issue salience is asymmetric.** Different issues activate different party brands. When cost-of-living or crime dominate public concern, it hurts Labour and helps National. When healthcare or unemployment are salient, the reverse occurs. The issue environment is as important as actual performance.
6. **The same indicator can mean different things in different regimes.** Under Labour (2017–23), rising mortgage rates signalled cost-of-living pain and hurt approval (r=-0.814). Under National (2023–), falling mortgage rates signal RBNZ emergency response to recession and are associated with falling approval (r=+0.829). Context determines whether a variable helps or hurts.
7. **More visible costs produce faster effects.** Petrol prices (seen daily) show contemporaneous effects (lag 0). Mortgage rates (noticed at refix time) show ~1 month lag. Consumer confidence (gradual attitude shift) shows ~1 month lag. Visibility determines response speed.
8. **National-Labour is zero-sum.** Changes in National and Labour vote share are inversely correlated (r=-0.31). Government bloc losses almost exactly equal opposition bloc gains. Under MMP, the two-party dynamic persists at the bloc level even as coalition composition changes.
9. **PM preference tracks party vote but doesn't lead it.** The PM margin explains 58% of the party vote margin (r=0.763), but the relationship is contemporaneous — no evidence that leader popularity drives party support or vice versa. A 1pp PM lead translates to only ~0.32pp party vote lead.
10. **NZ polls are accurate but biased toward Green and against NZ First.** Final polls call the winner correctly 90% of the time. Green is systematically overestimated (+1.03pp), NZ First underestimated (-0.74pp). There is no shy Tory effect — National is slightly *overestimated*.
11. **Incumbent decay is universal but not uniform.** Average decay is -0.114 pp/month. The sole majority government (Labour 2020–23) decayed fastest (-0.559 pp/month), disconfirming Paldam's prediction that coalitions erode faster.
12. **House prices are politically neutral at the aggregate level.** The wealth effect (homeowners feel richer) and affordability grievance (renters feel locked out) cancel each other, producing no net effect on incumbent polling.
13. **Voters are backward-looking, not forward-looking.** At the individual level, retrospective economic perceptions predict incumbent voting in all 10 NZES elections (mean r=0.376). Prospective perceptions lose 4/5 head-to-head tests. NZ voters punish past performance (Fiorina 1981), not invest in future competence (MacKuen 1992).
14. **Ideology matters more than competence.** Left-right proximity predicts the National-vs-Labour vote choice with r=0.698 on average, stronger than leader competence (which adds independent value but explains less variance). Positional voting has strengthened since 1990 without the L-R distribution polarising — suggesting ideological sorting without affective polarisation.
15. **MMP encourages split-ticket voting, not minor party desertion.** ~30% of voters cast party and electorate votes for different parties. Minor party voters lend their electorate vote to a major party ~50% of the time. This is rational strategy under MMP and replaces the Cox (1997) squeeze effect seen in FPTP systems.
16. **Crime is the exception to the "perception beats reality" rule.** Unlike economics (where GDP/CPI have no predictive power), actual recorded crime rates DO predict crime salience (r=0.569). But perception still adds value: crime salience explains 74% of govt approval variance, vs 63% for actual crime. Crime salience strongly activates issue ownership — hurts Labour (r=-0.720), helps National (r=+0.469).
17. **Rental inflation tracks incumbent unpopularity but may not cause it (SUGGESTIVE).** Rental CPI year-on-year change shows a strong level correlation with incumbent vote share (r=-0.731), but this finding **fails Granger causality** (no lag significant, all p > 0.27) and **loses significance under Newey-West HAC standard errors** (OLS p=0.001 → NW p=0.112). Change correlations are non-significant. The relationship is likely a contemporaneous co-movement rather than a causal pocketbook effect. Rental and mortgage inflation are uncorrelated (r=0.052), confirming they represent different channels — but only mortgage rates (and petrol prices) survive as confirmed causal indicators.
18. **Government approval is the master variable — everything else is noise once you have it.** A multivariate model with 7 predictors (approval, cost-of-living salience, crime salience, consumer confidence, mortgage rates, petrol prices, rental inflation) has massive multicollinearity (all VIF > 30) because all predictors feed through the same causal chain. Approval alone explains 67.5% of variance; adding six economic variables raises this only marginally. The parsimonious model is just approval + petrol price (R² = 0.767).
19. **Polling is a random walk — you cannot outpredict "assume current polls continue."** No indicator significantly predicts quarter-to-quarter vote *changes* (all p > 0.2). The naïve baseline (this quarter ≈ next quarter) beats all model-based forecasts with MAE of 2.64pp. Current polls already incorporate all available information; forecasting requires predicting unpredictable future events.
20. **Voters may only remember the election year (SUGGESTIVE).** Election-year GDP growth correlates with incumbent fate (r=+0.588, p=0.096) while early-term growth shows no relationship (r=-0.114). This is directionally consistent with Achen & Bartels' (2016) myopic retrospection thesis, but with only N=9 elections the finding is below conventional significance thresholds. The confidence interval is wide (~-0.10 to +0.90) and the result may be driven by 2-3 elections.
21. **Demographics predict vote choice, but through ethnic identity, not class.** The total demographic predictive power for National-vs-Labour vote has not declined since 1996 (Dalton's dealignment thesis not confirmed). But the *composition* has shifted: the age effect reversed direction (young→National in 1990s, old→National from 2005+), education and gender effects weakened, while Maori identity became the dominant predictor (OR=0.14-0.39). This is realignment, not dealignment.

---

---

## Ipsos Issues Monitor Data

24 PDFs downloaded from Ipsos NZ (editions 2-30, missing 1/4/5/9/10/15). Three CSVs extracted:

**`ipsos_issue_salience.csv`** — 28 waves (Sep 2018–Oct 2025), 10 issues tracked. Values from labeled data points on 30th edition trend charts. Note: values represent the 30th edition's harmonized dataset; individual earlier editions may show different values due to methodology revisions.

**`ipsos_govt_performance.csv`** — 30 waves (Sep 2017–Oct 2025). Mean score (0-10), top 4 / neutral / bottom 4 / don't know percentages, plus governing coalition label. Extracted from exact data table on page 11 of the 30th edition.

**`ipsos_party_capability.csv`** — 10 waves (Aug 2023–Oct 2025), 6 issues. For each issue: National, Labour, NZ First, Green, ACT, Te Pāti Māori, Other, Don't Know, None percentages. Party capability question appears to have started with the 21st edition (Aug 2023).

---

## Potential Future Work

1. ~~**Pollster house effects**~~ — Done (Analysis 16: no shy Tory, Green overestimated, polls converge)
2. ~~**Prediction accuracy**~~ — Done (Analysis 16: MAE 2.48pp National, winner correct 9/10)
3. ~~**PM polling vs party vote**~~ — Done (Analysis 17: contemporaneous, no lead/lag, PM margin r=0.763)
4. **Quarterly analysis** - Economic indicators at quarterly resolution vs annual
5. **Additional events** - Budget announcements, policy changes, international events
6. ~~**Ipsos issue salience vs polling**~~ — Done (Analysis 10)
7. ~~**Government performance vs polling**~~ — Done (Analysis 9)
8. ~~**Ipsos visualizations**~~ — Done (4 charts: govt vs vote, salience heatmap, capability trends, salience lines)
9. ~~**Multivariate model**~~ — Done (Analysis 27: approval dominates R²=0.675, kitchen sink VIF>30, parsimonious=approval+petrol)
10. ~~**Forecasting**~~ — Done (Analysis 28: random walk, naïve baseline MAE 2.64pp beats all models)
11. ~~**Mortgage rates vs polling**~~ — Done (Analysis 12)
12. ~~**What drives government approval**~~ — Done (Analysis 13: cost-of-living salience explains 74%, mediation confirmed)
13. ~~**Other pocketbook indicators**~~ — Done (Analysis 14: petrol Granger-causal r=-0.348; Analysis 26: rental r=-0.731 strongest)
14. ~~**Multivariate vote model**~~ — Done (merged with Analysis 27, approval dominates)
15. ~~**Consumer confidence vs polling**~~ — Done (Analysis 15: Granger-causal, 50% mediated through approval)
16. ~~**House prices vs polling**~~ — Done (Analysis 18: null finding, wealth ≈ affordability)
17. ~~**Cost of ruling: coalition vs majority**~~ — Done (Analysis 19: Paldam not confirmed)
18. ~~**Net migration vs NZ First**~~ — Done (Analysis 20: marginal, r=0.340, p=0.057)
19. ~~**Economic retrospection vs prospection**~~ — Done (Analysis 21: retro wins 4/5, mean r=0.376, partisan gap +0.68)
20. ~~**Valence vs position voting**~~ — Done (Analysis 22: position wins 4/4, pseudo-R² 0.586 vs 0.468, L-R strengthening)
21. ~~**Strategic voting under MMP**~~ — Done (Analysis 23: ~30% split-ticket, ~50% minor→major lending, no threshold effect)
22. ~~**Google Trends vs issue salience**~~ — Done (Analysis 24: "cost of living" predicts Ipsos r=0.482, contemporaneous)
23. ~~**Crime statistics vs crime salience**~~ — Done (Analysis 25: actual crime predicts salience r=0.569, unlike economics)
24. ~~**Rental price index vs polling**~~ — Done (Analysis 26: r=-0.731 incumbent, strongest pocketbook, uncorrelated with mortgage)
25. **Comparative Manifesto Project** — thermostatic responsiveness (Wlezien 1995)
26. ~~**Myopic retrospection**~~ — Done (Analysis 29: late-term GDP r=+0.588 vs early-term r=-0.114, Achen & Bartels supported)
27. ~~**Dealignment**~~ — Done (Analysis 30: not confirmed, pseudo-R² stable, Maori identity strengthening, realignment not dealignment)
