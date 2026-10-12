# Purple vote across nine democracies, 2026

For the corresponding **election-result** comparison around 2005, 2017 and the latest completed election, see [`purple_vote_election_results.md`](purple_vote_election_results.md).

**Updated 6 October 2026.** This extends the first four-country snapshot to every eligible 2026 national poll in the selected Wikipedia polling tables, plus the NZ polls already stored in this project. The dataset contains **914 poll records**. The chart is [`purple_vote_2026_trends.png`](purple_vote_2026_trends.png); the reproducible poll-level file is [`../data/purple_vote_polls_2026.csv`](../data/purple_vote_polls_2026.csv).

“Purple vote” is the combined polling share for two long-established governing parties. It measures support for those party organisations, **not** the whole left or right bloc. Red/blue is only a visual shorthand; in Ireland both chosen parties are broadly centre-right, and in the Netherlands the former Labour Party now appears within a merged alliance.

| Country | Pair counted | Polls in last 45 days | Mean combined share | Latest poll end |
|---|---|---:|---:|---:|
| Netherlands | GroenLinks–PvdA/PRO + VVD | 3 | **29.2%** | 28 Sep |
| Germany | SPD + CDU/CSU Union | 30 | **32.9%** | 5 Oct |
| Ireland | Fianna Fáil + Fine Gael | 4 | **34.8%** | 4 Oct |
| Austria | SPÖ + ÖVP | 6 | **36.3%** | 22 Sep |
| United Kingdom | Labour + Conservative | 41 | **46.2%** | 2 Oct |
| Australia | Labor + Liberal–National Coalition | 17 | **48.6%** | 2 Oct |
| New Zealand | Labour + National | 6 | **56.4%** | 27 Sep |
| Spain | PSOE + PP | 26 | **58.6%** | 3 Oct |
| Canada | Liberal + Conservative | 18 | **77.5%** | 3 Oct |

The 45-day window is **22 August–6 October 2026**, inclusive. Each published poll is one observation; figures are unweighted arithmetic means of the two party shares in that poll. They are descriptive summaries, not modelled polling averages. Repeated rolling polls may share respondents, and prolific pollsters have more influence. The NZ project’s daily weighted average is a separate series: Labour plus National was **56.8% on 6 October**.

The year-to-date pattern varies. NZ fell from a 66.0% mean in 1 January–14 February polls to 56.4% recently; Germany fell from 40.0% to 32.9%; Australia fell from 52.6% to 48.6%. The UK moved the other way, from 38.3% to 46.2%. These before/after numbers describe polls in two windows, without controlling for pollster mix or campaign events.

## Sources and interpretation

The data come from the current national polling tables for [New Zealand (local copy)](../data/2026_polling.json), [Australia](https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Australian_federal_election), [United Kingdom](https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election), [Canada](https://en.wikipedia.org/wiki/Opinion_polling_for_the_46th_Canadian_federal_election), [Ireland](https://en.wikipedia.org/wiki/Next_Irish_general_election), [Germany](https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_German_federal_election), [Austria](https://en.wikipedia.org/wiki/Next_Austrian_legislative_election), [Spain](https://en.wikipedia.org/wiki/Opinion_polling_for_the_2026_Spanish_general_election), and [the Netherlands](https://en.wikipedia.org/wiki/Next_Dutch_general_election). The extractor is [`../comparative_purple_polls.py`](../comparative_purple_polls.py).

The **Netherlands** is the least clean case: GroenLinks and PvdA merged into the PRO list, so its combined share includes voters for a green party as well as the old Labour party. Its Wikipedia page publishes many seat polls but only a few vote-share polls. Treat its 29.2% as a successor-alliance indicator, not a pure legacy-party measure.

**Ireland** is also a different political cleavage: Fianna Fáil and Fine Gael are historic governing rivals, not an ideological left/right pair. Their low combined share supports a story of fragmentation of the old party system, but not a claim that a traditional centre-left party has collapsed.

**France** is omitted from the numerical comparison. Its legislative polling frequently reports alliances rather than separate Socialist and Republicans party vote shares, while presidential polling measures candidate support in a different contest. A small number formed by mixing those measures would be misleading.

All percentages use stated primary vote or voting intention shares. Australia's two-party-preferred percentages are intentionally excluded. National samples, decided-voter treatments, party alliances, electoral systems, and polling methods differ; the rankings are descriptive rather than precise country effects. Wikipedia tables are compiled from pollster releases and can be revised. Validate any consequential finding against the linked pollster source tables.
