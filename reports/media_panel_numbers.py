#!/usr/bin/env python3
"""
Numbers for the Media and Politics panel answers (reports/panel_answers_2026-10-21_media_politics.md).

1. Business confidence (OECD composite BCI) by government, around changes of
   government, and against incumbent polling and GDP growth, compared with
   consumer confidence.
2. Who commissions New Zealand's published polls, by electoral cycle.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from consumer_confidence_analysis import get_incumbent_at_date, load_consumer_confidence, load_all_polls  # noqa: E402

DATA_DIR = ROOT / "data"

CHANGES_OF_GOVT = {
    "1999 (Nat→Lab)": "1999-11-27",
    "2008 (Lab→Nat)": "2008-11-08",
    "2017 (Nat→Lab)": "2017-10-26",
    "2023 (Lab→Nat)": "2023-11-27",
}


def load_business_confidence():
    df = pd.read_csv(DATA_DIR / "business_confidence.csv")
    bc = pd.DataFrame({"date": pd.to_datetime(df["TIME_PERIOD"]),
                       "bci": pd.to_numeric(df["OBS_VALUE"], errors="coerce")})
    return bc.dropna().sort_values("date").set_index("date").resample("MS").mean()


def load_gdp_growth():
    gdp = pd.read_csv(DATA_DIR / "gross-domestic-product-september-2025-quarter.csv", low_memory=False)
    g = gdp[gdp["Series_reference"] == "SNEQ.SG02RSC31B15PC"][["Period", "Data_value"]].copy()
    year = g["Period"].astype(float).astype(int)
    month = ((g["Period"].astype(float) - year) * 100).round().astype(int)
    g["q"] = pd.PeriodIndex.from_fields(year=year, quarter=(month // 3), freq="Q")
    return g.set_index("q")["Data_value"].astype(float)


def confidence_by_government(conf):
    print("\n=== 1a. Mean confidence index by governing party (100 = long-run average) ===")
    conf = conf.copy()
    conf["govt"] = [get_incumbent_at_date(d) for d in conf.index]
    print(conf.groupby("govt")[["bci", "cci"]].mean().round(1).to_string())

    print("\n=== 1b. Change in confidence: 6 months before vs 6 months after each change of government ===")
    print(f"{'Change':<18}{'BCI before':>11}{'BCI after':>10}{'ΔBCI':>7}{'CCI before':>12}{'CCI after':>10}{'ΔCCI':>7}")
    for label, d in CHANGES_OF_GOVT.items():
        d = pd.Timestamp(d)
        before = conf[(conf.index >= d - pd.DateOffset(months=6)) & (conf.index < d)]
        after = conf[(conf.index >= d) & (conf.index < d + pd.DateOffset(months=6))]
        b, a = before[["bci", "cci"]].mean(), after[["bci", "cci"]].mean()
        print(f"{label:<18}{b.bci:>11.1f}{a.bci:>10.1f}{a.bci - b.bci:>+7.1f}{b.cci:>12.1f}{a.cci:>10.1f}{a.cci - b.cci:>+7.1f}")


def confidence_vs_polling(conf):
    print("\n=== 1c. Confidence vs incumbent major-party vote (quarterly means) ===")
    polls = load_all_polls()
    polls["incumbent_vote"] = polls.apply(
        lambda r: pd.to_numeric(r.get(get_incumbent_at_date(r["date"])), errors="coerce"), axis=1)
    polls = polls.dropna(subset=["incumbent_vote"])
    q = pd.DataFrame({"vote": polls.set_index("date")["incumbent_vote"].resample("QS").mean()})
    q = q.join(conf.resample("QS").mean()).dropna()
    q = q[q.index >= "1996-01-01"]
    d = q.diff().dropna()
    for col in ["bci", "cci"]:
        rl, pl = pearsonr(q[col], q["vote"])
        rc, pc = pearsonr(d[col], d["vote"])
        res = grangercausalitytests(q[["vote", col]].diff().dropna(), maxlag=2, verbose=False)
        gp = min(res[lag][0]["ssr_ftest"][1] for lag in (1, 2))
        print(f"{col.upper()}: levels r={rl:+.3f} (p={pl:.3f}); changes r={rc:+.3f} (p={pc:.3f}); "
              f"Granger {col}→vote best p={gp:.3f}; n={len(q)}")
    print(f"BCI–CCI correlation (quarterly levels): r={pearsonr(q['bci'], q['cci'])[0]:+.3f}")


def confidence_vs_gdp(conf):
    print("\n=== 1d. Confidence vs real GDP growth (q/q, SA) ===")
    gdp = load_gdp_growth()
    cq = conf.resample("QS").mean()
    cq.index = cq.index.to_period("Q")
    df = cq.join(gdp.rename("gdp"), how="inner").dropna()
    df["gdp_next"] = df["gdp"].shift(-1)
    df = df[(df.index >= pd.Period("1991Q1")) & ~df.index.isin(pd.period_range("2020Q1", "2021Q4", freq="Q"))]
    for col in ["bci", "cci"]:
        same = pearsonr(df[col], df["gdp"])
        nxt = pearsonr(df.dropna()[col], df.dropna()["gdp_next"])
        print(f"{col.upper()}: same-quarter growth r={same[0]:+.3f} (p={same[1]:.3f}); "
              f"next-quarter growth r={nxt[0]:+.3f} (p={nxt[1]:.3f}); n={len(df)} (excl. 2020–21)")


MEDIA = ("1 News", "Newshub", "RNZ", "Stuff", "The Post")
ADVOCACY = ("Taxpayers' Union", "Curia")
PARTY_OR_CORPORATE = ("Talbot Mills", "Business NZ", "Labour–", "National–")


def commissioner(pollster):
    if any(pollster.startswith(m) for m in MEDIA):
        return "Media outlet"
    if any(a in pollster for a in ADVOCACY):
        return "Taxpayers' Union / Curia"
    if any(p in pollster for p in PARTY_OR_CORPORATE):
        return "Party or corporate client"
    if "Roy Morgan" in pollster:
        return "Roy Morgan (self-published)"
    return "Other"


def poll_commissioners():
    print("\n=== 2. Who commissions published party-vote polls, by cycle ===")
    for year in (2020, 2023, 2026):
        polls = json.load(open(DATA_DIR / f"{year}_polling.json"))["polls"]
        names = [p.get("pollster") or "" for p in polls if "result" not in (p.get("pollster") or "")]
        counts = pd.Series([commissioner(n) for n in names]).value_counts()
        share = (100 * counts / counts.sum()).round(0).astype(int)
        print(f"{year - 3}–{str(year)[2:]} cycle (n={len(names)}): "
              + "; ".join(f"{k} {counts[k]} ({share[k]}%)" for k in counts.index))


def main():
    conf = load_business_confidence().join(load_consumer_confidence(), how="inner")
    confidence_by_government(conf)
    confidence_vs_polling(conf)
    confidence_vs_gdp(conf)
    poll_commissioners()


if __name__ == "__main__":
    main()
