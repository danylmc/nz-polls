# The fastest law in the West, still — structural flaws brief

Column brief, ~1,000–1,500 words. Last updated 2026-10-22.
Sources: web research (linked inline and listed at the end), DPMC ministerial list
(7 Apr 2026), OECD *Government at a Glance 2025*, this repo's polling findings
(`reports/findings.md`, Rules 3, 11, 20), and a new count of select committee
bypass across 2,506 Acts (`urgency_analysis.py`, `data/acts_legislative_history.csv`).

---

## The argument in one paragraph

In 1979 Geoffrey Palmer called New Zealand's Parliament "the fastest law in the West"
and its government an elected dictatorship. MMP in 1996 was meant to be the cure. It
changed **who sits in Parliament**, but not **how much power the executive holds once
it's there**. The country still has one chamber and no supreme constitution, so a
simple majority can change almost anything. Nearly half the governing MPs are on the
executive payroll. Almost one law in five passed this term skipped
select committee entirely, a record, against about one in a hundred under Clark. The rules that bind a government
for three years are written in private coalition talks, and voters blame only the
biggest party for what results. The obvious reform, a four-year term traded for
stronger scrutiny, lost its scrutiny half in select committee and was then shelved.
Meanwhile Wellington controls almost 88% of public spending, so every local problem
becomes national politics. The common thread: power is concentrated, it moves fast,
and accountability is aimed at the wrong place.

---

## 1. One chamber, no brakes

**The precedent.** NZ had an upper house, the Legislative Council, until 1950. Sidney
Holland's National government packed it with appointees pledged to vote for its own
abolition (the "suicide squad"). The Legislative Council Abolition Act took effect on
1 January 1951. National had promised in 1949 to find another safeguard against
hasty legislation. Its Constitutional Reform Committee reported in 1952 and
recommended a Senate: 32 members nominated by party leaders in proportion to their
strength, able to delay bills for two months. It was never created. Seventy-five
years on, the replacement is still pending.
([NZ History](https://nzhistory.govt.nz/legislative-council-abolished);
[Te Ara 1966](https://teara.govt.nz/en/1966/government-parliament/page-4))

**What fills the gap.** Almost nothing is entrenched. The Electoral Act's "reserved
provisions" (parliamentary term, voting age, secret ballot, electoral boundaries)
need 75% of MPs or a referendum to change. But the clause that entrenches them, s268,
is not itself entrenched. The 2022 Three Waters episode showed how casually
entrenchment can be used: a 60% entrenchment clause was slipped into a water bill at
the committee stage late at night, then reversed after an outcry.

**The Bill of Rights doesn't bind Parliament.** When a bill appears to breach the Bill
of Rights Act, the Attorney-General must tell the House (a "section 7 report").
Parliament can pass it anyway, and regularly does. One count found almost 40 such
reports on government bills, and more than 70 in total, over about 25 years to 2017
([Double Aspect, 2017](https://doubleaspect.blog/2017/06/08/the-good-the-bad-and-the-ugly/)).
Courts can declare a law inconsistent with the Act (*Attorney-General v Taylor*,
2018), but can't strike it down. Since 2022 the government must formally respond to
such a declaration, and that is the whole consequence.

**Palmer's own verdict.** His phrase in the 1979 first edition was "the fastest law in
the west". Later retellings, including a [2024 Spinoff interview](https://thespinoff.co.nz/politics/25-06-2024/geoffrey-palmer-on-rapid-reforms-you-should-learn-from-history-not-repeat-it),
give it as "fastest law-maker". Later editions of the book dropped the line
([Scoop, 2007](https://m.scoop.co.nz/stories/PA0704/S00224.htm)). Palmer and Andrew
Butler's 2016 proposal for a written constitution, *A Constitution for Aotearoa New
Zealand*, went nowhere.

---

## 2. The payroll vote: governing MPs who scrutinise themselves

From the DPMC ministerial list of 7 April 2026:

| | National | ACT | NZ First | Total |
|---|---|---|---|---|
| Cabinet | 14 | 3 | 3 | **20** |
| Outside Cabinet | 5 | 2 | 1 | **8** |
| Parliamentary under-secretaries | 0 | 1 | 1 | **2** |
| **Executive roles** | 19 | 6 | 5 | **30** |
| Caucus (2023 election + Port Waikato by-election) | 49 | 11 | 8 | **68** |
| Share of caucus | 39% | 55% | 63% | **44%** |

- 30 of the coalition's 68 MPs hold executive jobs. Add the Speaker, whips and the
  government-held select committee chairs, and most of the rest are also on extra
  pay or office.
- The smaller the partner, the larger the share of its caucus inside the executive.
  Five of NZ First's eight MPs are ministers or an under-secretary.
- Across the whole 123-seat House, 30 MPs (24%) belong to the executive whose
  conduct the House is supposed to scrutinise.
- **MMP made this worse, not better.** List MPs owe their places to party rankings.
  The 2018 waka-jumping law (Electoral Act s55A, no sunset clause) lets a party expel
  a defecting MP from Parliament. National voted against it in 2018 and promised
  repeal, but has not repealed it in government, and repeal isn't in the coalition
  agreements ([Wikipedia](https://en.wikipedia.org/wiki/Waka_jumping);
  [1News 2024](https://www.1news.co.nz/2024/07/17/dark-places-david-seymour-on-using-waka-jumping-law/)).
  Party discipline is stronger than under first-past-the-post.
- Select committees are majority-controlled by the government, chaired mostly by
  government MPs, and rely on advice from the departments whose bills they examine
  ([Spinoff](https://thespinoff.co.nz/books/23-01-2025/why-were-not-ready-to-extend-the-parliamentary-term)).

*Check before filing:* the caucus numbers (any departures since 2023) and the current
ministerial list, since there may have been a reshuffle after April.

---

## 3. Laws made fast

**Our own count (new).** We parsed the legislative history of every public Act
since 2000: 2,506 Acts from legislation.govt.nz (`urgency_fetch.py`,
`urgency_analysis.py`). The measure: Acts that never went to a select committee, so
the public had no chance to make submissions. Appropriation and Imprest Supply Acts,
which skip committee by rule, are excluded.

| Parliament | Government | Acts | Skipped committee | Share | Bill-level share* |
|---|---|---|---|---|---|
| 46th 1999–02 | Clark I (Lab–Alliance) | 209 | 8 | 3.8% | 7.5% |
| 47th 2002–05 | Clark II | 366 | 4 | 1.1% | 1.7% |
| 48th 2005–08 | Clark III | 312 | 4 | 1.3% | 2.1% |
| 49th 2008–11 | Key I | 297 | 29 | 9.8% | 14.1% |
| 50th 2011–14 | Key II | 318 | 10 | 3.1% | 5.8% |
| 51st 2014–17 | Key/English | 266 | 16 | 6.0% | 10.7% |
| 52nd 2017–20 | Ardern I (Lab–NZF) | 201 | 24 | 11.9% | 12.5% |
| 53rd 2020–23 | Ardern/Hipkins (Lab majority) | 195 | 30 | 15.4% | 15.7% |
| **54th 2023–26** | **Luxon (Nat–ACT–NZF)** | **209** | **38** | **18.2%** | **18.6%** |

\*Before about 2020, omnibus bills were split into many Acts at third reading,
which inflates the early Act counts. The bill-level column counts each original
bill once. The 54th is the highest on both measures.

- **The headline:** almost one law in five passed this term skipped select
  committee. Under Clark it was about one in a hundred.
- **The first 100 days** (from swearing-in): 14 of the coalition's first 16 Acts
  skipped committee. The only precedent is Key's first government, at 8 of 9
  (December 2008: the 90-day trial law, national standards, bail and sentencing).
  Every other government skipped committee for two or fewer of its first-100-day
  Acts.
  - The 14: Reserve Bank single mandate, Fair Pay Agreements repeal, clean car
    discount repeal, RMA replacement repeal, 90-day trials, Productivity
    Commission, Three Waters repeal, benefit indexation, Māori Health Authority,
    smokefree generation, legal aid, business payment practices, regional fuel tax,
    and the taxation principles reporting repeal.
- **After the first 100 days**: 24 more. Twelve were introduced on Budget day (2024,
  2025, 2026). The other 12 include the Equal Pay Amendment Act 2025 (the pay
  equity changes), the Climate Change Response (2050 Target) Amendment Act 2025, the
  RMA consent-duration and Auckland housing changes, FamilyBoost, the greyhound
  racing ban amendment, electronic bail monitoring and the adoption amendment.
- **It's a trend across both parties, and that's the structural point.** It ran at
  1–4% under Clark, spiked with Key's first government (Canterbury earthquakes,
  Budget-night tax bills), then climbed through 12% (Ardern I, mostly Covid-19 laws)
  and 15% (the Labour majority) to 18%. Each government treats the last one's peak
  as normal.
- **Cross-check:** Parliamentary Library figures reported by RNZ and The Spinoff
  (August 2026) give 49th 10.0%, 50th 3.1%, 51st 6.1%, 54th 17.4% (first 940 days),
  and 15.5% for the 53rd over the same span. Our figures are 9.8, 3.1, 6.0, 18.2 (full
  term) and 15.4. The one disagreement is the 52nd (theirs 5.5%, ours 11.9%). Our 24
  are all checkable: 11 Covid-19 laws plus Budget-night bills. It's possible the
  Library excluded something
  ([Spinoff, Aug 2026](https://thespinoff.co.nz/politics/03-08-2026/more-laws-rushed-through-parliament-under-urgency-than-any-term-since-2008)).
- Media reports put the coalition's urgency in its first 100 days at 13–14 bills,
  more than any MMP government. Those counts include bills that went to committee
  but had other stages rushed. Transparency International NZ's "57% vs 29%" measures
  bills with *any* stage under urgency. Our data can't see urgency on individual
  stages, only whether committee was skipped, so don't quote the 57% as ours.
- Leader of the House Louise Upston told RNZ that only 10 bills that skipped
  committee weren't Budget-related or first-100-days bills. Our count is 12, to
  September 2026.
- Chart: `reports/urgency_select_committee_listener.png` (Listener style, bars
  coloured by governing party).
- Urgency can shorten or skip the select committee stage, which is the only point
  where the public gets a say on a bill.
- There's no test for urgency. The government doesn't have to show anything is
  actually urgent, and TI NZ's proposal is that it should have to.
- Fast-track: the Fast-track Approvals Act 2024 created a one-stop consenting regime
  with 149 projects listed in the Act itself. The original bill gave three ministers
  the final say. Select committee moved final decisions to expert panels, but
  ministers still control which projects enter the process
  ([RNZ, Mar 2024](https://www.rnz.co.nz/news/political/512259/the-unprecedented-power-the-government-is-handing-three-of-its-ministers-under-its-new-fast-track-approval-bill);
  [MfE](https://environment.govt.nz/acts-and-regulations/acts/fast-track-approvals/fast-track-approvals-process/)).
  This is a better example than it first looks: **select committee worked here**,
  which shows what's lost when it's skipped.

---

## 4. Policy whiplash: the cost of no brakes

Each government undoes the last one's centrepieces, often under urgency:

- Three Waters → Local Water Done Well
- Natural and Built Environment Act and Spatial Planning Act (replacements for the
  RMA): repealed Dec 2023
- Fair Pay Agreements: repealed Dec 2023
- Smokefree generation law: repealed Feb 2024, under urgency
- Te Aka Whai Ora (Māori Health Authority): abolished Feb 2024, under urgency
- Clean car discount: ended Dec 2023
- Auckland light rail: cancelled

These repeals are what you get from a system with one chamber, a three-year term and
no settled constitution. A new government's first job is undoing the old one's, and
infrastructure that takes a decade to plan doesn't survive a three-year cycle.

---

## 5. Three years, and the four-year term that lost its trade

**The time problem (repo data).** Incumbents lose an average of 0.11pp a month in
the polls. Labour 2020–23 lost 0.56pp a month, the fastest
(`cost_of_ruling_analysis.py`). Election-year GDP growth tracks incumbent fate
(r = +0.59) and early-term growth doesn't (r = −0.11), though with N = 9 that is
suggestive, not proven (`myopic_retrospection.py`). The incentive this creates:
legislate fast in year one, then campaign. Short terms and urgency feed each other.

**Voters have said no twice.** In the 1967 and 1990 referendums, roughly two-thirds
voted to keep three-year terms.

**The bill.** The Term of Parliament (Enabling 4-year Term) Legislation Amendment
Bill (2025) offered a deal: a government would get a full four years **only if** it
gave opposition parties a proportional share of select committee seats and chairs.
More power in exchange for more scrutiny is exactly Palmer's logic. The Justice
Committee **removed the variable term and the select committee proportionality
requirements**
([NZ Initiative, Jan 2026](https://www.nzinitiative.org.nz/reports-and-media/opinion/running-out-of-time-for-more-time/)).
In February 2026 the government dropped the referendum: "We won't be progressing with
a referendum on a four-year term at this election" — Paul Goldsmith
([Newstalk ZB](https://www.newstalkzb.co.nz/news/politics/government-scraps-plans-for-referendum-on-four-year-parliamentary-term);
[Wikipedia](https://en.wikipedia.org/wiki/Term_of_Parliament_(Enabling_4-year_Term)_Legislation_Amendment_Bill)).
The trade lost its scrutiny half first, then the rest was shelved.

---

## 6. Coalition agreements: an unwritten constitution written in private

- Three leaders negotiate for a few weeks after the election. Nobody voted for the
  combined package, but it binds the government for three years and outranks the
  manifestos people did vote on.
- **Regulatory Standards Act 2025**: passed its third reading 68–55 on 13 Nov 2025,
  in force from 1 Jan 2026. More than 98% of public submissions opposed it. It
  passed because National and NZ First had agreed to it in coalition deals with ACT
  ([Wikipedia](https://en.wikipedia.org/wiki/Regulatory_Standards_Act_2025);
  [ODT](https://www.odt.co.nz/news/national/regulatory-standards-bill-passes-final-hurdle)).
  ACT won 8.6% of the party vote.
- **Treaty Principles Bill**: the agreements only committed the partners to support
  it to first reading. It was voted down 112–11 at second reading in April 2025. The
  coalition-agreement process still put it on the agenda and dominated a year of
  politics.
- **Power without blame (repo data).** This is the original angle. Voters punish
  incumbents for cost-of-living pain, mortgage rates, petrol prices and falling
  approval, but the punishment moves votes between **National and Labour only**. The
  Greens, NZ First and ACT are statistically unaffected (Rule 3; mortgage and petrol
  analyses). Minor partners get agenda power through coalition agreements and
  executive power through the payroll share (55–63% of their caucuses), yet bear
  none of the economic blame. In 2026 NZ First *rose* from 6.4% to a 12.4% peak while
  its senior partner fell (`coalition_feature_brief.md`).

---

## 7. A centralised state

- Central government accounts for **87.8%** of NZ public spending (OECD 2023 data).
  The UK figure is 80.4%. In Denmark local government spends **64.3%**, and in Sweden
  50.7% ([OECD Government at a Glance 2025](https://www.oecd.org/en/publications/2025/06/government-at-a-glance-2025_70e14c6c/full-report/revenues-and-spending-by-level-of-government_84942f25.html)).
  NZ is consistently described as among the most centralised OECD countries.
- Councils are funded almost entirely by rates, so they bear the cost of growth
  (pipes, roads) but get none of the tax revenue it produces. That gives them every
  reason to resist housing and little ability to fund infrastructure.
- So local questions become national politics: Three Waters, Local Water Done Well,
  each RMA rewrite, Auckland transport. Combined with point 4, the country's
  long-lived assets get planned on a three-year cycle.

---

## 8. A Parliament without the staff to check the executive

- NZ has no parliamentary budget office or independent fiscal institution. A
  2019 bill for one failed for lack of opposition support. Treasury was still
  advising the current Finance Minister on options in 2024–25, but nothing has been
  set up ([Treasury OIA](https://www.treasury.govt.nz/sites/default/files/2025-03/oia-20250050-v2.pdf);
  [IMF PFM blog](https://blog-pfm.imf.org/en/pfmblog/2025/02/an-independent-fiscal-institution-for-new-zealand)).
- Scrutiny Weeks, introduced after the 2023 Standing Orders review, are an
  improvement. They run twice a year, for a week each
  ([Waatea](https://waateanews.com/?p=162319)).
- MPs have very little research capacity of their own. Bills are drafted, explained
  and evaluated by the executive's own officials.

---

## 9. Hollow parties with total control

- National claimed 200,000 members in the mid-1970s and had about a tenth of that
  by 2000. Labour declined in parallel. There is no official data on party
  membership, and the parties don't publish it
  ([Te Ara](https://teara.govt.nz/en/national-party/page-5)).
- A few hundred members pick candidates, and party boards rank the lists that decide
  most of who gets into Parliament. Add the waka-jumping law, and MPs answer to the
  party first.
- The 2023 Independent Electoral Review recommended lowering the threshold to 3.5%,
  abolishing coat-tailing, and tightening donation rules. The government declined
  most of it. The parties that would have to change the rules are the ones the rules
  help. *(Check which recommendations were adopted.)*

---

## Suggested shape (~1,200 words)

1. **Open** (~150): 1950 and the suicide squad, an upper house voting itself out of
   existence on the promise of a replacement that never came. Or open on a recent
   urgency night. (The opening is yours.)
2. **Palmer's diagnosis** (~150): "fastest law in the West", elected dictatorship,
   and MMP as the cure that changed the passengers but not the vehicle.
3. **No brakes** (~200): one chamber, unentrenched entrenchment, a Bill of Rights
   Parliament can ignore.
4. **The payroll** (~200): 30 of 68 governing MPs in the executive (44%), five of
   NZ First's eight. List rankings plus the waka-jumping law make independent
   backbenchers rare.
5. **Speed** (~200): our count. Almost one law in five skipped select committee this
   term, against about one in a hundred under Clark. 14 of the first 16 Acts. A
   trend across both parties, which is why it's structural. Fast-track as proof
   that select committee matters. Run the chart.
6. **Coalition deals and blame** (~200): the Regulatory Standards Act's 98%
   opposition, and the repo finding that minor partners get power while National
   takes the economic blame.
7. **The reform that ate itself** (~150): four-year term, scrutiny half removed,
   referendum shelved.
8. **Close** (~100): Palmer's line, still true, still holding the record.
   Optional turn: a centralised state (87.8%) plus policy whiplash as the bill for all
   of the above.

Points 7–9 (centralisation, capacity, parties) are probably a second column. Each
could carry its own piece.

---

## Data still to build (if wanted)

- ~~**Urgency by government**~~: done for Acts that skipped select committee,
  2000–2026 (section 3). Still missing: urgency on individual stages (needs
  Hansard or Journals of the House), and the Geiringer, Higbee & McLeay database
  (*What's the Hurry?*, 2011) for 1987–99.
- **Payroll share by term**: executive roles as a share of the governing caucus for
  each government since 1996, from DPMC ministerial lists. Tests whether MMP-era
  coalitions really have a higher payroll share.
- **Section 7 reports passed into law**, by government.
- **Listener chart**: the OECD central government spending share, NZ highlighted.

## Facts to verify before filing

- Current caucus sizes and ministerial list (post-April 2026 reshuffles)
- 1967 and 1990 referendum percentages (from memory: roughly two-thirds kept three
  years both times)
- *Taylor* (2018) and the 2022 declarations-of-inconsistency amendment (from memory)
- Which 2023 Electoral Review recommendations were rejected
- The Treaty Principles Bill second reading vote (112–11, April 2025, from memory)
- That waka-jumping repeal hasn't happened since 2024

---

## Sources

- [NZ History — Legislative Council abolished](https://nzhistory.govt.nz/legislative-council-abolished)
- [Te Ara 1966 — Government and Parliament](https://teara.govt.nz/en/1966/government-parliament/page-4)
- [Legislative Council Abolition Act 1950](https://www.legislation.govt.nz/act/public/1950/3/en/latest/)
- [Scoop 2007 — Palmer's "fastest law in the west"](https://m.scoop.co.nz/stories/PA0704/S00224.htm)
- [Spinoff 2024 — Geoffrey Palmer on rapid reforms](https://thespinoff.co.nz/politics/25-06-2024/geoffrey-palmer-on-rapid-reforms-you-should-learn-from-history-not-repeat-it)
- [Double Aspect — section 7 reports](https://doubleaspect.blog/2017/06/08/the-good-the-bad-and-the-ugly/)
- [Ministry of Justice — Bill of Rights compliance reports](https://www.justice.govt.nz/justice-sector-policy/constitutional-issues-and-human-rights/the-bill-of-rights-act/compliance-reports/)
- [DPMC ministerial list, 7 April 2026](https://www.dpmc.govt.nz/sites/default/files/2026-04/ministerial-list-07-apr-2026.pdf)
- [Wikipedia — Waka jumping](https://en.wikipedia.org/wiki/Waka_jumping)
- [1News 2024 — Seymour on the waka-jumping law](https://www.1news.co.nz/2024/07/17/dark-places-david-seymour-on-using-waka-jumping-law/)
- [Spinoff, Aug 2026 — more laws rushed through under urgency than any term since 2008](https://thespinoff.co.nz/politics/03-08-2026/more-laws-rushed-through-parliament-under-urgency-than-any-term-since-2008)
- [legislation.govt.nz — as-enacted Acts with legislative histories](https://www.legislation.govt.nz/)
- [Spinoff — the government's latest time warp](https://thespinoff.co.nz/politics/01-10-2024/the-governments-latest-time-warp)
- [Waikato University — NZ's "brittle" democracy](https://www.waikato.ac.nz/news-events/news/new-news-article-fast-track-laws-parliamentary-urgency-treaty-tension-media-retreat-warning-signs-for-nzs-brittle-democracy/)
- [Transparency International NZ — a positive test for urgency](https://www.transparency.org.nz/blog/time-to-raise-the-bar-the-case-for-a-positive-test-for-urgency-in-new-zealand-parliament)
- [RNZ — fast-track ministerial powers](https://www.rnz.co.nz/news/political/512259/the-unprecedented-power-the-government-is-handing-three-of-its-ministers-under-its-new-fast-track-approval-bill)
- [MfE — fast-track approvals process](https://environment.govt.nz/acts-and-regulations/acts/fast-track-approvals/fast-track-approvals-process/)
- [NZ Initiative — Running out of time for more time](https://www.nzinitiative.org.nz/reports-and-media/opinion/running-out-of-time-for-more-time/)
- [Newstalk ZB — four-year term referendum scrapped](https://www.newstalkzb.co.nz/news/politics/government-scraps-plans-for-referendum-on-four-year-parliamentary-term)
- [Wikipedia — four-year term bill](https://en.wikipedia.org/wiki/Term_of_Parliament_(Enabling_4-year_Term)_Legislation_Amendment_Bill)
- [Spinoff — why we're not ready to extend the term](https://thespinoff.co.nz/books/23-01-2025/why-were-not-ready-to-extend-the-parliamentary-term)
- [Wikipedia — Regulatory Standards Act 2025](https://en.wikipedia.org/wiki/Regulatory_Standards_Act_2025)
- [ODT — Regulatory Standards Bill passes](https://www.odt.co.nz/news/national/regulatory-standards-bill-passes-final-hurdle)
- [OECD Government at a Glance 2025 — spending by level of government](https://www.oecd.org/en/publications/2025/06/government-at-a-glance-2025_70e14c6c/full-report/revenues-and-spending-by-level-of-government_84942f25.html)
- [Treasury OIA 2025 — independent fiscal institution](https://www.treasury.govt.nz/sites/default/files/2025-03/oia-20250050-v2.pdf)
- [IMF PFM blog — an IFI for New Zealand](https://blog-pfm.imf.org/en/pfmblog/2025/02/an-independent-fiscal-institution-for-new-zealand)
- [Waatea — Scrutiny Week](https://waateanews.com/?p=162319)
- [Te Ara — National Party membership](https://teara.govt.nz/en/national-party/page-5)
