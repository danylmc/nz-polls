# Claude Code Project Instructions

## Project Goal
Analyze NZ polling data to find predictors and correlates of polling shifts using political science literature and statistical analysis.

## What's Been Done
1. Scraped 1,012 party vote polls from Wikipedia (1993-2026)
2. Scraped 299 Preferred PM polls from Wikipedia (2008-2026)
3. Fetched economic data from World Bank API (GDP, unemployment, CPI 1961-2024)
4. Created events timeline with leadership changes, crises, scandals
5. Created visualization script with 4 cross-election trend graphs
6. Ran statistical analysis with 8 hypothesis tests
7. Generated findings report (`reports/findings.md`)

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `scraper.py` | Wikipedia party vote scraper | Complete |
| `pm_scraper.py` | Preferred PM polling scraper | Complete |
| `economic_scraper.py` | World Bank API economic data | Complete |
| `events.py` | Event timeline data | Complete |
| `analysis.py` | Statistical analysis (8 tests) | Complete |
| `visualize.py` | Trend graphs | Complete |
| `data/*.json` | Party vote polling data | Complete |
| `data/*_pm_polling.json` | PM preference polling data | Complete |
| `data/economic/*.csv` | Economic indicators | Complete |
| `reports/findings.md` | Analysis report | Complete |

---

## Key Findings Summary

| Hypothesis | Result | p-value |
|------------|--------|---------|
| National-Labour zero-sum | **Confirmed** (r = -0.316) | < 0.001 |
| Mean reversion | **Confirmed** (68% revert) | N/A |
| Reversal effect | **Confirmed** (r = -0.51) | < 0.001 |
| Third party squeeze | Not detected | 0.748 |
| Economic voting | Not significant | > 0.05 |

Notable event effects:
- **Ardern honeymoon (2017)**: +8.9% Labour boost
- **Delta lockdown (2021)**: +1.6% Labour, -4.5% National

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

**NZ Findings vs. Literature:**
- National-Labour zero-sum: **Confirmed** as expected
- Leadership honeymoon: **Confirmed** (Ardern 2017 most dramatic)
- Rally effect: **Partial** (Delta lockdown showed incumbent boost)
- Third party squeeze: **Not confirmed** in NZ data
- Economic voting: **Not significant** (possibly due to small N of 11 elections)

---

## Potential Future Work

1. **Pollster house effects** - Do different pollsters systematically favor certain parties?
2. **Prediction accuracy** - How do final polls compare to election results?
3. **PM polling vs party vote** - Does PM preference lead or lag party support?
4. **Quarterly analysis** - Economic indicators at quarterly resolution vs annual
5. **Additional events** - Budget announcements, policy changes, international events
