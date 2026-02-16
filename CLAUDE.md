# Claude Code Project Instructions

## Project Goal
Analyze NZ polling data to find predictors and correlates of polling shifts using political science literature and statistical analysis.

## What's Been Done
1. Scraped 1,012 party vote polls from Wikipedia (1993-2026)
2. Scraped 299 Preferred PM polls from Wikipedia (2008-2026)
3. Fetched economic data from World Bank API (GDP, unemployment, CPI 1961-2024)
4. Created events timeline with leadership changes, crises, scandals
5. Created visualization script with 4 cross-election trend graphs
6. Ran statistical analysis with 11 hypothesis tests (8 polling + 3 Ipsos)
7. Generated findings report (`reports/findings.md`)
8. Scraped and extracted Ipsos NZ Issues Monitor data (24 PDFs, 2018-2025)
9. Integrated Ipsos data into analysis: govt performance, issue salience, party capability vs polling

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `scraper.py` | Wikipedia party vote scraper | Complete |
| `pm_scraper.py` | Preferred PM polling scraper | Complete |
| `economic_scraper.py` | World Bank API economic data | Complete |
| `events.py` | Event timeline data | Complete |
| `analysis.py` | Statistical analysis (11 tests) | Complete |
| `visualize.py` | Trend graphs | Complete |
| `data/*.json` | Party vote polling data | Complete |
| `data/*_pm_polling.json` | PM preference polling data | Complete |
| `data/economic/*.csv` | Economic indicators | Complete |
| `data/ipsos_pdfs/*.pdf` | Ipsos Issues Monitor PDFs (24 editions) | Complete |
| `data/ipsos_issue_salience.csv` | Issue importance (28 waves, 10 issues) | Complete |
| `data/ipsos_govt_performance.csv` | Government rating 0-10 (30 waves) | Complete |
| `data/ipsos_party_capability.csv` | Party best to manage issue (10 waves, 6 issues) | Complete |
| `data/ipsos_report_links.json` | Ipsos website scrape / PDF URLs | Complete |
| `reports/findings.md` | Analysis report | Complete |

---

## Key Findings Summary

| Hypothesis | Result | p-value |
|------------|--------|---------|
| National-Labour zero-sum | **Confirmed** (r = -0.310) | < 0.001 |
| Mean reversion | **Confirmed** (67% revert) | N/A |
| Reversal effect | **Confirmed** (r = -0.51) | < 0.001 |
| Third party squeeze | Not detected | 0.682 |
| Economic voting | Not significant | > 0.05 |
| Govt performance → incumbent vote | **Confirmed** (r = 0.827, R² = 0.68) | < 0.001 |
| Issue ownership affects vote | **Confirmed** (10 issues with sig. links) | < 0.05 |
| Perceived competence → vote | **Confirmed** (r = 0.585 pooled) | < 0.001 |

Notable event effects:
- **Ardern honeymoon (2017)**: +8.9% Labour boost
- **Delta lockdown (2021)**: +1.6% Labour, -4.5% National

Notable Ipsos findings:
- Each +1pt govt rating ≈ +5.7pp incumbent vote (strongest single predictor)
- Cost-of-living salience strongly hurts Labour (r = -0.885)
- Healthcare salience helps Labour (r = 0.670); crime helps National (r = 0.403)
- Economy capability perception tracks vote almost 1:1 (r = 0.87–0.89)

---

## How to Run

```bash
cd /mnt/d/data/polls
source venv/bin/activate

# Re-scrape data (if needed)
python scraper.py          # Party vote polls
python pm_scraper.py       # PM polls
python economic_scraper.py # Economic data

# Run analysis
python analysis.py         # Generates reports/findings.md

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

**NZ Findings vs. Literature:**
- National-Labour zero-sum: **Confirmed** as expected
- Leadership honeymoon: **Confirmed** (Ardern 2017 most dramatic)
- Rally effect: **Partial** (Delta lockdown showed incumbent boost)
- Third party squeeze: **Not confirmed** in NZ data
- Economic voting: **Not significant** (possibly due to small N of 11 elections)
- Government approval → vote: **Confirmed** (r = 0.827) — strongest single predictor found
- Issue ownership: **Confirmed** — cost-of-living hurts Labour, crime helps National, healthcare helps Labour
- Perceived competence → vote: **Confirmed** (r = 0.585 pooled) — especially strong on economy and crime

---

---

## Ipsos Issues Monitor Data

24 PDFs downloaded from Ipsos NZ (editions 2-30, missing 1/4/5/9/10/15). Three CSVs extracted:

**`ipsos_issue_salience.csv`** — 28 waves (Sep 2018–Oct 2025), 10 issues tracked. Values from labeled data points on 30th edition trend charts. Note: values represent the 30th edition's harmonized dataset; individual earlier editions may show different values due to methodology revisions.

**`ipsos_govt_performance.csv`** — 30 waves (Sep 2017–Oct 2025). Mean score (0-10), top 4 / neutral / bottom 4 / don't know percentages, plus governing coalition label. Extracted from exact data table on page 11 of the 30th edition.

**`ipsos_party_capability.csv`** — 10 waves (Aug 2023–Oct 2025), 6 issues. For each issue: National, Labour, NZ First, Green, ACT, Te Pāti Māori, Other, Don't Know, None percentages. Party capability question appears to have started with the 21st edition (Aug 2023).

---

## Potential Future Work

1. **Pollster house effects** - Do different pollsters systematically favor certain parties?
2. **Prediction accuracy** - How do final polls compare to election results?
3. **PM polling vs party vote** - Does PM preference lead or lag party support?
4. **Quarterly analysis** - Economic indicators at quarterly resolution vs annual
5. **Additional events** - Budget announcements, policy changes, international events
6. ~~**Ipsos issue salience vs polling**~~ — Done (Analysis 10)
7. ~~**Government performance vs polling**~~ — Done (Analysis 9)
8. **Ipsos visualizations** - Time-series plots of govt performance + incumbent vote, issue salience heatmaps
9. **Multivariate model** - Combine govt performance, issue salience, and economic data into a single incumbent vote model
10. **Forecasting** - Can Ipsos govt performance rating predict next-quarter polling movement?
