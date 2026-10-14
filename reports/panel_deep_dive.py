"""Numbers behind reports/panel_answers_2026-10_deep.md.

1. NZES 2023: most important issue vs 2020->2023 vote switching and best party
2. Ipsos: issue salience vs govt approval, within each government's term
3. Cost-of-living salience vs headline inflation, food inflation and the price level
4. Ipsos party capability: who is seen as best on each issue, and whether it tracks the vote
5. Polls since 2023: same-pollster flows, major-party share, bloc totals

Run from the project root: python reports/panel_deep_dive.py
"""
import json, glob, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from approval_drivers import load_incumbent_monthly  # noqa: E402

DATA = ROOT / "data"
PARTY = {0: "Nonvote", 1: "Lab", 2: "Nat", 3: "Grn", 4: "ACT", 5: "NZF", 6: "TPM", 7: "Oth"}


def wm(mask, w):
    return np.average(mask, weights=w) * 100 if len(mask) else np.nan


def nzes_2023():
    print("\n=== 1. NZES 2023: most important issue ===")
    flags = ["minflation", "meconomy", "mhealth", "mequality", "mhousing", "mchange", "mcrime",
             "mclimate", "mtax", "mwages"]
    c = sqlite3.connect(DATA / "nzesdb.sqlite")
    d = pd.read_sql(f"select {','.join(flags)},C3,E6e,mvpartyvote,mpartyvote,mppartyvote,vwgt from nzes_2023", c)
    d["v23"] = d.mvpartyvote.fillna(d.mpartyvote)
    d["w"] = d.vwgt.fillna(d.vwgt.median())
    lab20 = d[(d.mppartyvote == 1) & d.v23.isin(PARTY)]
    rows = []
    for f in flags:
        s, l = d[d[f] == 1], lab20[lab20[f] == 1]
        best = s.C3.map(PARTY | {0: "none", 9: "DK"}).value_counts(normalize=True) * 100
        rows.append(dict(issue=f[1:], named=wm(d[f] == 1, d.w), lab20_n=len(l),
                         lab20_left_Lab=wm(l.v23 != 1, l.w), lab20_to_Nat=wm(l.v23 == 2, l.w),
                         lab20_to_NatActNzf=wm(l.v23.isin([2, 4, 5]), l.w),
                         lab20_to_Grn=wm(l.v23 == 3, l.w),
                         best_Nat=best.get("Nat", 0), best_Lab=best.get("Lab", 0),
                         best_none_dk=best.get("none", 0) + best.get("DK", 0)))
    print(pd.DataFrame(rows).round(0).to_string(index=False))
    other = lab20[lab20.minflation != 1]
    print(f"2020 Labour voters not naming inflation: to Nat {wm(other.v23 == 2, other.w):.0f}%, "
          f"to Grn {wm(other.v23 == 3, other.w):.0f}%, all left Labour {wm(lab20.v23 != 1, lab20.w):.0f}%")
    for k, g in lab20.groupby(lab20.v23 != 1):
        print(f"2020 Labour voters who {'switched' if k else 'stayed'}: "
              f"economy/inflation affected vote {wm(g.E6e == 1, g.w):.0f}%")
    print(f"All respondents: economy/inflation affected vote {wm(d.E6e == 1, d.w):.0f}%")
    X = sm.add_constant(lab20[["minflation", "mhealth", "mhousing", "meconomy", "mwages", "mcrime",
                               "mtax", "mchange"]].fillna(0))
    m = sm.GLM((lab20.v23 != 1).astype(int), X, family=sm.families.Binomial(),
               freq_weights=lab20.w / lab20.w.mean()).fit()
    print("Logit: left Labour ~ top issue (odds ratios)")
    print(pd.DataFrame({"OR": np.exp(m.params), "p": m.pvalues}).round(3).to_string())


def load_ipsos():
    sal = pd.read_csv(DATA / "ipsos_issue_salience.csv", parse_dates=["date"]).set_index("date")
    sal = sal.drop(columns="source_note")
    ap = pd.read_csv(DATA / "ipsos_govt_performance.csv", parse_dates=["date"]).set_index("date")
    return sal.join(ap[["mean_score"]], how="inner")


ISSUES = ["inflation_cost_of_living", "healthcare_hospitals", "the_economy", "housing_price",
          "crime_law_order", "unemployment"]


def within_term(df):
    print("\n=== 2. Salience vs approval within each government's term (levels; Δ = wave-to-wave) ===")
    df = df.copy()
    df["inc"] = load_incumbent_monthly().rolling(3, min_periods=1).mean().reindex(df.index)
    for label, s in [("Labour 2021-23", df["2021-02":"2023-09"]), ("National 2023-26", df["2023-11":])]:
        dd = s.diff()
        print(f"{label} (n={len(s)})")
        for i in ISSUES:
            r, p = pearsonr(s[i], s.mean_score)
            rv, pv = pearsonr(s[i], s.inc)
            x = dd[[i, "mean_score"]].dropna()
            rd, pd_ = pearsonr(x[i], x.mean_score)
            print(f"  {i:26s} approval r={r:+.2f} (p={p:.3f})  vote r={rv:+.2f} (p={pv:.3f})"
                  f"  Δ r={rd:+.2f} (p={pd_:.2f})")


def cpi_series(code):
    cpi = pd.read_csv(DATA / "consumers-price-index-december-2025-quarter-index-numbers (1).csv")
    s = cpi[cpi.Series_reference == code].copy()
    p = s.Period.astype(str)
    s["date"] = pd.to_datetime(p.str[:4] + "-" + p.str[5:7] + "-01")
    return s.set_index("date").Data_value.astype(float)


def col_vs_prices(df):
    print("\n=== 3. Cost-of-living salience vs prices (CPI to Dec 2025 qtr) ===")
    q = pd.DataFrame({"all": cpi_series("CPIQ.SE9A"), "food": cpi_series("CPIQ.SE901")})
    q["headline_infl"] = q["all"].pct_change(4) * 100
    q["food_infl"] = q.food.pct_change(4) * 100
    q["price_level_vs_Dec2020"] = (q["all"] / q.loc["2020-12-01", "all"] - 1) * 100
    m = q.resample("MS").ffill()  # each wave gets the latest published quarter
    w = df[["inflation_cost_of_living", "mean_score"]].join(
        m[["headline_infl", "food_infl", "price_level_vs_Dec2020"]]).dropna()
    print(w["2021-06":].round(1).to_string())
    for c in ["headline_infl", "food_infl", "price_level_vs_Dec2020"]:
        r_all = pearsonr(w[c], w.inflation_cost_of_living)[0]
        post = w["2022-01":]
        r_post, p_post = pearsonr(post[c], post.inflation_cost_of_living)
        print(f"  CoL salience vs {c:24s} 2018-25 r={r_all:+.2f}   2022-25 r={r_post:+.2f} (p={p_post:.3f})")


def capability():
    print("\n=== 4. Ipsos party capability ===")
    c = pd.read_csv(DATA / "ipsos_party_capability.csv", parse_dates=["date"])
    rows = []
    for f in ["2023_polling.json", "2026_polling.json"]:
        for p in json.load(open(DATA / f))["polls"]:
            rows.append(dict(date=p["date"], National=p["parties"].get("National"),
                             Labour=p["parties"].get("Labour")))
    v = pd.DataFrame(rows)
    v["date"] = pd.to_datetime(v.date, errors="coerce")
    v = v.dropna().set_index("date").sort_index().resample("MS").mean().rolling(3, min_periods=1).mean()
    for i in ["inflation_cost_of_living", "the_economy", "healthcare_hospitals", "unemployment"]:
        s = c[c.issue == i].set_index("date")
        s["majors"] = s.national + s.labour
        print(f"{i}: first/latest  Nat {s.national.iloc[0]}->{s.national.iloc[-1]} (peak {s.national.max()}),"
              f"  Lab {s.labour.iloc[0]}->{s.labour.iloc[-1]},  majors {s.majors.iloc[0]}->{s.majors.iloc[-1]}")
        s = s.join(v).dropna(subset=["National"])
        print(f"   capability tracks vote: National r={pearsonr(s.national, s.National)[0]:.2f}, "
              f"Labour r={pearsonr(s.labour, s.Labour)[0]:.2f} (n={len(s)})")


def poll_flows():
    print("\n=== 5. Polls since 2020 (pollster names exist from the 2020 cycle) ===")
    parties = ["National", "Labour", "Green", "ACT", "NZ First", "TOP", "Te Pāti Māori"]
    rows = []
    for f in ["2020_polling.json", "2023_polling.json", "2026_polling.json"]:
        for p in json.load(open(DATA / f))["polls"]:
            rows.append(dict(date=p["date"], pollster=p["pollster"], **p["parties"]))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df.pollster.notna() & ~df.pollster.str.contains("election", case=False)].sort_values("date")
    df["pk"] = df.pollster.str.replace(r"[^A-Za-z]", "", regex=True).str.lower().str[:8]
    for p in parties:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    for label, s in [("Labour govt 2020-23", df[(df.date >= "2020-10-18") & (df.date < "2023-10-14")]),
                     ("National govt 2023-25", df[(df.date >= "2023-11-27") & (df.date < "2026-01-01")]),
                     ("National govt 2026", df[df.date >= "2026-01-01"])]:
        ch = s.groupby("pk")[parties].diff().dropna(subset=["National", "Labour"])
        r, p = pearsonr(ch.National, ch.Labour)
        print(f"{label}: same-pollster Δ National vs Δ Labour r={r:+.2f} (p={p:.3f}, n={len(ch)})")
    t = df[df.date >= "2023-11-27"].copy()
    qt = t.groupby(t.date.dt.to_period("Q"))[parties].mean()
    qt["Lab+Nat"] = qt.National + qt.Labour
    qt["Right (Nat+ACT+NZF)"] = qt.National + qt.ACT + qt["NZ First"]
    qt["Left (Lab+Grn+TPM)"] = qt.Labour + qt.Green + qt["Te Pāti Māori"]
    qt["n"] = t.groupby(t.date.dt.to_period("Q")).size()
    print(qt.round(1).to_string())


if __name__ == "__main__":
    nzes_2023()
    ipsos = load_ipsos()
    within_term(ipsos)
    col_vs_prices(ipsos)
    capability()
    poll_flows()
