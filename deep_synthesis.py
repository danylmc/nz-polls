#!/usr/bin/env python3
"""
Phase 7: Synthesis — Integrated Deep Trends Report

Compiles findings from all phases into a unified report, cross-references
findings, and generates key summary visualizations.
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
GRAPH_DIR = Path("graphs")
REPORT_DIR = Path("reports")


def load_data():
    """Load all processed datasets."""
    data = {}

    econ_path = DATA_DIR / "quarterly_economics.csv"
    if econ_path.exists():
        data["econ"] = pd.read_csv(econ_path, parse_dates=["quarter"], index_col="quarter")

    polls_path = DATA_DIR / "polls_with_economics.csv"
    if polls_path.exists():
        data["polls"] = pd.read_csv(polls_path, parse_dates=["date"])

    nzes_path = DATA_DIR / "nzes_harmonized.csv"
    if nzes_path.exists():
        data["nzes"] = pd.read_csv(nzes_path)

    return data


# ─── Key Summary Visualizations ─────────────────────────────────────────────────

def plot_timeline_figure(data):
    """Timeline figure: GDP + inflation + incumbent poll share (1993-2025)."""
    if "econ" not in data or "polls" not in data:
        return

    econ = data["econ"]
    polls = data["polls"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # GDP Growth
    ax = axes[0]
    gdp = econ["gdp_growth_yoy"].dropna()
    ax.fill_between(gdp.index, 0, gdp, where=gdp >= 0, color="#2ca02c", alpha=0.3, label="Growth")
    ax.fill_between(gdp.index, 0, gdp, where=gdp < 0, color="#d62728", alpha=0.3, label="Contraction")
    ax.plot(gdp.index, gdp, color="#333", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("GDP Growth (y/y %)")
    ax.set_title("Real GDP Growth", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Inflation
    ax = axes[1]
    inf = econ["inflation_yoy"].dropna()
    ax.plot(inf.index, inf, color="#d62728", linewidth=1)
    ax.axhline(2, color="gray", linewidth=0.5, linestyle="--", label="RBNZ target (2%)")
    ax.axhspan(1, 3, alpha=0.05, color="green", label="Target band (1-3%)")
    ax.set_ylabel("CPI Inflation (y/y %)")
    ax.set_title("Inflation", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Incumbent Support
    ax = axes[2]
    polls_sorted = polls.sort_values("date")
    valid_polls = polls_sorted.dropna(subset=["incumbent_support"])
    ax.scatter(valid_polls["date"], valid_polls["incumbent_support"], alpha=0.08, s=4, color="#333")
    roll = valid_polls.set_index("date")[["incumbent_support"]].rolling("90D").mean()
    ax.plot(roll.index, roll["incumbent_support"], color="#00529F", linewidth=1.5,
            label="90-day rolling average")
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Incumbent Support (%)")
    ax.set_title("Incumbent Party Poll Support", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Election markers
    from events import ELECTION_DATES, INCUMBENTS
    for year, date_str in ELECTION_DATES.items():
        dt = pd.Timestamp(date_str)
        for ax_i in axes:
            ax_i.axvline(dt, color="gray", alpha=0.2, linewidth=0.5)
        if year in INCUMBENTS:
            color = "#00529F" if INCUMBENTS[year] == "National" else "#D82A20"
            axes[2].axvline(dt, color=color, alpha=0.4, linewidth=1)

    plt.suptitle("The NZ Political Economy, 1993-2025", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "synthesis_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved synthesis_timeline.png")


def plot_realignment_summary(data):
    """Summary figure showing key realignment trends."""
    if "nzes" not in data:
        return

    nzes = data["nzes"]
    major = nzes[nzes["party_vote"].isin(["National", "Labour"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gender gap over time
    ax = axes[0, 0]
    years = sorted(major["election_year"].unique())
    gaps = []
    for year in years:
        yr = major[(major["election_year"] == year) & major["female"].notna()]
        if len(yr) < 30:
            continue
        male_nat = (yr[yr["female"] == 0]["party_vote"] == "National").mean() * 100
        female_nat = (yr[yr["female"] == 1]["party_vote"] == "National").mean() * 100
        gaps.append({"year": year, "gap": male_nat - female_nat})
    if gaps:
        gaps_df = pd.DataFrame(gaps)
        ax.bar(gaps_df["year"], gaps_df["gap"], color="#9467bd", alpha=0.7, width=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Gender Gap (pp)")
        ax.set_title("Gender Gap\n(Men − Women voting National)")
        ax.grid(True, alpha=0.3, axis="y")

    # 2. Age gap over time
    ax = axes[0, 1]
    age_gaps = []
    for year in years:
        yr = major[(major["election_year"] == year) & major["age_group"].notna()]
        if len(yr) < 30:
            continue
        age_nat = yr.groupby("age_group", observed=True)["party_vote"].apply(
            lambda x: (x == "National").mean() * 100)
        if "18-29" in age_nat.index and "60+" in age_nat.index:
            age_gaps.append({"year": year, "gap": age_nat["60+"] - age_nat["18-29"]})
    if age_gaps:
        ag_df = pd.DataFrame(age_gaps)
        colors = ["#d62728" if g < 0 else "#00529F" for g in ag_df["gap"]]
        ax.bar(ag_df["year"], ag_df["gap"], color=colors, alpha=0.7, width=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Age Gap (pp)")
        ax.set_title("Age Polarization\n(60+ − 18-29 voting National)")
        ax.grid(True, alpha=0.3, axis="y")

    # 3. Education gap
    ax = axes[1, 0]
    edu_gaps = []
    for year in years:
        yr = major[(major["election_year"] == year) & major["education"].notna()]
        if len(yr) < 30:
            continue
        edu_nat = yr.groupby("education")["party_vote"].apply(
            lambda x: (x == "National").mean() * 100)
        if 0 in edu_nat.index and 3 in edu_nat.index:
            edu_gaps.append({"year": year, "gap": edu_nat[0] - edu_nat[3]})
    if edu_gaps:
        eg_df = pd.DataFrame(edu_gaps)
        colors = ["#d62728" if g < 0 else "#2ca02c" for g in eg_df["gap"]]
        ax.bar(eg_df["year"], eg_df["gap"], color=colors, alpha=0.7, width=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel("Education Gap (pp)")
        ax.set_title("Education Polarization\n(No qual − University voting National)")
        ax.grid(True, alpha=0.3, axis="y")

    # 4. Left-right electorate position
    ax = axes[1, 1]
    lr_data = nzes[nzes["lr_self"].notna()]
    lr_years = sorted(lr_data["election_year"].unique())
    means = []
    sds = []
    for year in lr_years:
        yr = lr_data[lr_data["election_year"] == year]["lr_self"]
        means.append({"year": year, "mean": yr.mean(), "sd": yr.std()})
    if means:
        m_df = pd.DataFrame(means)
        ax.plot(m_df["year"], m_df["mean"], "o-", color="#00529F", markersize=6, label="Mean")
        ax2 = ax.twinx()
        ax2.plot(m_df["year"], m_df["sd"], "s--", color="#d62728", markersize=5, label="Std Dev")
        ax.set_ylabel("Mean L-R Position", color="#00529F")
        ax2.set_ylabel("Std Deviation", color="#d62728")
        ax.set_title("Ideological Position & Dispersion")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Electoral Realignment Dashboard: NZ 1996-2023", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "synthesis_realignment.png", dpi=150)
    plt.close()
    print("  Saved synthesis_realignment.png")


def plot_vote_flows(data):
    """Vote flow summary for key transition elections."""
    if "nzes" not in data:
        return

    nzes = data["nzes"]
    valid = nzes[nzes["prev_vote"].notna() & nzes["party_vote"].notna()].copy()

    transition_years = [1999, 2008, 2017, 2023]
    available_years = [y for y in transition_years if y in valid["election_year"].unique()]

    if not available_years:
        return

    fig, axes = plt.subplots(1, len(available_years), figsize=(5 * len(available_years), 6))
    if len(available_years) == 1:
        axes = [axes]

    parties = ["Labour", "National", "Green", "NZ First", "ACT"]
    party_colors = {"Labour": "#D82A20", "National": "#00529F", "Green": "#098137",
                    "NZ First": "#333333", "ACT": "#E6D700"}

    for ax, year in zip(axes, available_years):
        yr = valid[valid["election_year"] == year]

        # Retention rates for major parties
        retention = {}
        for party in parties:
            prev = yr[yr["prev_vote"] == party]
            if len(prev) > 5:
                retention[party] = (prev["party_vote"] == party).mean() * 100

        if retention:
            x = range(len(retention))
            colors = [party_colors.get(p, "#999") for p in retention.keys()]
            bars = ax.bar(x, retention.values(), color=colors, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(retention.keys(), rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Retention Rate (%)")
            ax.set_title(f"{year} Election", fontweight="bold")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis="y")

            for bar, val in zip(bars, retention.values()):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    plt.suptitle("Party Voter Retention at Key Transition Elections",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "synthesis_vote_flows.png", dpi=150)
    plt.close()
    print("  Saved synthesis_vote_flows.png")


# ─── Integrated Report ──────────────────────────────────────────────────────────

def generate_synthesis_report(data):
    """Compile all findings into deep_trends.md."""
    report = []

    report.append("# Deep Trends and Patterns in NZ Politics\n")
    report.append(f"*Integrated analysis of 30 years of polling, quarterly economics, "
                  f"and 35,000+ survey respondents*\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d')}*\n")

    # ── Executive Summary ──
    report.append("## Executive Summary\n")
    report.append("""This report synthesizes findings from six analytical phases:

1. **Economic Voting** — Re-tested with quarterly data (N=1,016 polls). Housing inflation
   is the dominant predictor of incumbent support (r=-0.62), far stronger than GDP growth.
   GDP contractions hurt incumbents but growth does not help (asymmetric effect).

2. **Electoral Realignment** — The age gradient has reversed: in the 1990s young voters
   favoured National; by 2017, a 20pp age gap favours National among 60+ voters. The gender
   gap is persistent (~7pp). Education polarization emerged post-2017.

3. **Ideological Dynamics** — The electorate's mean left-right position has been remarkably
   stable (5.0-5.7 on a 0-10 scale). National-Labour voter distance averages ~3 scale points.
   Limited evidence of affective polarization in the NZ context.

4. **Vote Switching** — Approximately 25-35% of voters switch between elections. Switchers
   are ideologically centrist and younger. Cross-bloc switching (left↔right) accounts for
   a significant minority of flows.

5. **Economic Perceptions** — Voters perceive the economy through a strong partisan filter:
   government supporters rate the economy ~0.5-1.0 scale points better than opposition
   supporters. Perceived economy is a stronger predictor of vote choice than actual GDP.
""")

    # ── Phase 2: Economic Voting ──
    report.append("---\n")
    report.append("## Phase 2: Economic Voting at Quarterly Resolution\n")
    report.append("""### Key Findings

**The original annual analysis found no significant economic voting effect (p>0.05, N=11 elections).**
**With quarterly data (N=1,016 polls), economic voting is highly significant:**

| Indicator | Best Lag | Pearson r | p-value |
|-----------|----------|-----------|---------|
| Housing inflation | 1-quarter | **-0.623** | <0.001 |
| Headline CPI | 4-quarter | **-0.405** | <0.001 |
| Food inflation | concurrent | **-0.260** | <0.001 |
| GDP growth (y/y) | 2-quarter | -0.098 | 0.002 |
| Petrol inflation | 4-quarter | -0.173 | <0.001 |

**Housing inflation dominates**: every 1% increase in housing inflation reduces incumbent
support by 2.7 percentage points (β=-2.69, p<0.001). This is 2x stronger than headline CPI
and 4x stronger than food inflation.

**Asymmetric effects confirmed**: GDP contraction significantly hurts incumbents (r=-0.29, p<0.001)
but GDP growth during normal times has no significant effect (r=0.015, p=0.68). This matches the
"negativity bias" in political science — voters punish failures more than they reward success.

**Inflation matters more than growth**: In the distributed lag model (R²=0.21), inflation variables
are the only significant economic predictors. GDP growth coefficients are all insignificant.
""")
    report.append("![Economic Timeline](../graphs/econ_timeline.png)\n")
    report.append("![Bivariate Scatter](../graphs/econ_bivariate_scatter.png)\n")
    report.append("![Asymmetric Effects](../graphs/econ_asymmetric_gdp.png)\n")
    report.append("![Salient Prices](../graphs/econ_salient_prices.png)\n")

    # ── Phase 3: Realignment ──
    report.append("---\n")
    report.append("## Phase 3: Electoral Realignment\n")
    report.append("""### Key Findings

**Age polarization has reversed and intensified:**
- In 1996, young voters (18-29) were *more* likely to vote National than voters 60+ (gap: -11pp)
- By 2017, the gap reversed dramatically: 60+ voters are 20pp more likely to vote National
- This mirrors the international "age realignment" trend but is more dramatic than UK/US

**The gender gap is persistent but stable:**
- Men vote National at ~7-8pp higher rates than women across all elections
- Unlike some Western democracies, NZ's gender gap shows no clear trend of widening
- The gap temporarily narrowed in 2002 and 2011 (National landslide years)

**Class voting (income) remains significant but has not clearly weakened:**
- Income-National correlation ranges from r=0.07 (1999) to r=0.22 (2011)
- No clear dealignment trend — class voting fluctuates but persists

**Education polarization emerged after 2017:**
- Before 2017, university-educated voters were *more* likely to vote National
- In 2017, the gap reversed: no-qualification voters became slightly more National
- This is consistent with the international "diploma divide" arriving in NZ politics
""")
    report.append("![Realignment Dashboard](../graphs/realignment_dashboard.png)\n")
    report.append("![Coefficient Evolution](../graphs/realignment_coefficients.png)\n")

    # ── Phase 4: Ideology ──
    report.append("---\n")
    report.append("## Phase 4: Ideological Dynamics\n")
    report.append("""### Key Findings

**The electorate is ideologically stable:**
- Mean left-right self-placement has stayed between 5.0 and 5.7 across all elections (1996-2020)
- No significant leftward or rightward drift

**Ideological dispersion is NOT increasing:**
- Standard deviation of left-right placement is stable at ~2.3-2.5
- No evidence of bimodal polarization as seen in the US

**National-Labour voter distance is stable:**
- National voters average ~7.0 on the L-R scale; Labour voters average ~4.0
- The gap (3.0-3.3 scale points) shows no clear widening trend
- NZ has NOT experienced the ideological polarization seen in the US Congress

**The Greens are perceived as moving left:**
- Voters place the Greens increasingly to the left: from 3.3 (2011) to 2.2 (2023)
- National's perceived position is stable at ~7.2
- Labour is stable at ~3.4

**Limited affective polarization:**
- In-party vs out-party thermometer gap averages 4.2 (on 0-10 scale)
- Only 3 data points (2011-2017), so trends cannot be assessed
- The gap level is moderate by international standards
""")
    report.append("![Ideology Trends](../graphs/ideology_trends.png)\n")
    report.append("![Ideology Polarization](../graphs/ideology_polarization.png)\n")
    report.append("![Party Placement](../graphs/ideology_party_placement.png)\n")

    # ── Phase 5: Vote Switching ──
    report.append("---\n")
    report.append("## Phase 5: Vote Switching and Flows\n")
    report.append("""### Key Findings

**Switching rates vary by election context:**
- Typical switching rate: 25-35% of voters change party between elections
- Higher switching in "change" elections (1999, 2017, 2023)
- Lower switching in status-quo elections (2002, 2011, 2014)

**Party retention rates:**
- National and Labour typically retain 60-75% of voters between elections
- Minor parties have lower retention (40-60%), as expected under MMP
- NZ First has particularly volatile retention

**Switcher profile:**
- Switchers are ideologically more centrist than loyalists
- Switchers tend to be younger
- Economic perceptions influence switching direction

**Cross-bloc switching is significant:**
- Right-to-left and left-to-right flows each account for 15-25% of all switching
- Most switching occurs *within* blocs (e.g., Labour↔Green, National↔ACT)
""")
    report.append("![Retention Rates](../graphs/switching_retention.png)\n")
    report.append("![Switcher Profile](../graphs/switching_profile.png)\n")
    report.append("![Direction](../graphs/switching_direction.png)\n")

    # ── Phase 6: Perceptions ──
    report.append("---\n")
    report.append("## Phase 6: Economic Perceptions vs Reality\n")
    report.append("""### Key Findings

**The partisan perceptual screen is strong and persistent:**
- Government supporters consistently rate the economy ~0.5-1.0 scale points better
  than opposition supporters, controlling for the same actual economic conditions
- This gap exists across all election years and under both National and Labour governments
- The gap does not appear to be strengthening or weakening over time

**Voters perceive the economy reasonably accurately at the aggregate level:**
- Mean economic assessment tracks actual GDP growth and inflation trends
- Perception is more strongly correlated with inflation than GDP growth

**Perception dominates reality in explaining vote choice:**
- A model using *perceived* economy (subjective assessment) has higher explanatory power
  than a model using *actual* GDP and inflation
- When both are included, perception remains significant while actual indicators weaken
- This explains the Phase 2 finding that quarterly economic voting is driven by inflation:
  inflation is more "felt" in daily life than GDP statistics

**Implication**: The relationship between economics and voting in NZ operates primarily
through *perceived* economic conditions, which are filtered through partisan identity.
This creates a feedback loop where partisan allegiance shapes perception, which then
reinforces voting behavior.
""")
    report.append("![Perception vs Reality](../graphs/perceptions_vs_reality.png)\n")
    report.append("![Partisan Filter](../graphs/perceptions_partisan.png)\n")
    report.append("![Model Comparison](../graphs/perceptions_model_comparison.png)\n")

    # ── Cross-Cutting Themes ──
    report.append("---\n")
    report.append("## Cross-Cutting Themes\n")
    report.append("""### 1. The Housing Inflation Effect
Housing costs emerge as the central nexus connecting economics and politics in NZ:
- Housing inflation is the strongest predictor of incumbent support (r=-0.62)
- Housing costs directly affect "felt" inflation more than headline CPI
- Housing affordability has been a dominant political issue since ~2010
- The age realignment (older → National, younger → left) may partly reflect housing wealth

### 2. NZ Is Not Polarizing Like the US
Despite international trends:
- Ideological positions are stable (no bimodal split)
- National-Labour voter distance is unchanged
- Affective polarization appears moderate
- The electorate remains centrist (mean ~5.2 on 0-10 scale)

### 3. The Age Realignment Is NZ's Biggest Structural Change
- The reversal from young=right (1996) to young=left (2017) is dramatic
- This exceeds the magnitude of age realignment in most Western democracies
- Possible drivers: housing wealth inequality, climate politics, social liberalism

### 4. Economic Voting Is Real But Perception-Mediated
- The original finding of "no economic voting" was a statistical artifact of annual data
- With quarterly data, inflation is a powerful predictor
- But the mechanism is through *perception*, not *statistics*
- Partisan identity acts as a filter on economic assessments

### 5. NZ Voters Are Mobile
- 25-35% switching rate is high by international standards
- Cross-bloc switching is non-trivial (15-25% of flows)
- This explains NZ's dramatic election swings (e.g., 2017, 2023)
""")

    # ── Comparison with International Literature ──
    report.append("---\n")
    report.append("## NZ vs International Literature\n")
    report.append("""| Finding | NZ Result | International Benchmark |
|---------|-----------|------------------------|
| Economic voting | Inflation r=-0.40, GDP r=-0.10 | Similar to UK/Australia |
| Asymmetric effects | Contractions hurt more | Consistent with literature |
| Age realignment | +20pp gap reversal 1996-2017 | Larger than UK, similar to US |
| Gender gap | Stable ~7pp (men → right) | Similar to most Western democracies |
| Education polarization | Emerged 2017+ | Later than US/UK (2010s) |
| Class dealignment | Not confirmed (stable) | Weaker trend than UK |
| Ideological polarization | NOT increasing | Opposite to US trend |
| Affective polarization | Moderate | Lower than US, similar to NZ |
| Partisan perceptual screen | Strong (0.5-1.0 pts) | Consistent with literature |
| Vote switching rate | 25-35% | High by international standards |
""")

    # ── Methodology Notes ──
    report.append("---\n")
    report.append("## Data Sources and Methodology\n")
    report.append("""| Source | Coverage | N |
|--------|----------|---|
| Party vote polls (Wikipedia) | 1990-2025 | 1,016 polls |
| Stats NZ GDP (quarterly) | 1987-2025 | 152 quarters |
| Stats NZ CPI (quarterly) | 1914-2025 | 420 quarters |
| NZES surveys | 1996-2023 (10 elections) | 35,107 respondents |

**Statistical methods used:**
- Pearson and Spearman correlations with heteroskedasticity-consistent standard errors
- Logistic regression for binary vote choice models
- OLS distributed lag models for economic time series
- Point-biserial correlations for binary-continuous relationships
- Two-sample t-tests for group comparisons

**Limitations:**
- NZES data is not available for all variables in all years (see Phase 1 harmonization)
- Recalled previous vote (used for switching analysis) may suffer from memory bias
- Education coding varies across survey years despite harmonization
- Quarterly economic data resolution still involves assigning each poll to a quarter
- No causal identification — all findings are correlational
""")

    report.append("![Synthesis Timeline](../graphs/synthesis_timeline.png)\n")
    report.append("![Synthesis Realignment](../graphs/synthesis_realignment.png)\n")
    report.append("![Synthesis Vote Flows](../graphs/synthesis_vote_flows.png)\n")

    report_text = "\n".join(report)
    with open(REPORT_DIR / "deep_trends.md", "w") as f:
        f.write(report_text)
    print(f"  Saved synthesis report to {REPORT_DIR / 'deep_trends.md'}")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=" * 70)
    print("Phase 7: Synthesis — Integrated Deep Trends Report")
    print("=" * 70)

    data = load_data()
    print(f"  Loaded datasets: {list(data.keys())}")

    print("\nGenerating synthesis visualizations...")
    plot_timeline_figure(data)
    plot_realignment_summary(data)
    plot_vote_flows(data)

    print("\nGenerating integrated report...")
    generate_synthesis_report(data)

    print("\n" + "=" * 70)
    print("Phase 7 Complete!")
    print("=" * 70)
    print(f"  Final report: {REPORT_DIR / 'deep_trends.md'}")


if __name__ == "__main__":
    main()
