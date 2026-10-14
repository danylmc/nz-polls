#!/usr/bin/env python3
"""
Real household disposable income vs incumbent party support.

Question: does growth in real household disposable income per person predict
support for the governing party (the "pocketbook" measure most directly tied to
household living standards)?

Data:
- Household net disposable income (Stats NZ national accounts, experimental,
  quarterly SA, 2016Q2-2026Q2), deflated by CPI, divided by population
- Real gross national disposable income per capita (RGNDI, quarterly SA,
  1991-2026): national rather than household, but a long series
- Quarterly mean party vote (Wikipedia polls), Ipsos government rating

Build data first: /mnt/d/data/.venv/bin/python household_income_fetch.py

Tests, as for mortgage/petrol/rental: level correlation, change correlation,
Granger causality, Newey-West HAC SEs, a within-government (fixed effects)
check, approval mediation, a horse race against CPI inflation, and an
election-level myopic test with RGNDI.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
REPORTS_DIR = BASE / "reports"

GOVERNMENTS = [
    ("1999-11-27", "2002-07-27", "Labour", "Clark 1"),
    ("2002-07-27", "2005-09-17", "Labour", "Clark 2"),
    ("2005-09-17", "2008-11-08", "Labour", "Clark 3"),
    ("2008-11-08", "2011-11-26", "National", "Key 1"),
    ("2011-11-26", "2014-09-20", "National", "Key 2"),
    ("2014-09-20", "2017-09-23", "National", "Key/English 3"),
    ("2017-10-26", "2020-10-17", "Labour", "Ardern 1"),
    ("2020-10-17", "2023-10-14", "Labour", "Ardern/Hipkins 2"),
    ("2023-11-27", "2026-12-31", "National", "Luxon 1"),
]

# Official final party vote (%)
ELECTION_RESULTS = {
    "1999-11-27": {"National": 30.50, "Labour": 38.74},
    "2002-07-27": {"National": 20.93, "Labour": 41.26},
    "2005-09-17": {"National": 39.10, "Labour": 41.10},
    "2008-11-08": {"National": 44.93, "Labour": 33.99},
    "2011-11-26": {"National": 47.31, "Labour": 27.48},
    "2014-09-20": {"National": 47.04, "Labour": 25.13},
    "2017-09-23": {"National": 44.45, "Labour": 36.89},
    "2020-10-17": {"National": 25.58, "Labour": 50.01},
    "2023-10-14": {"National": 38.06, "Labour": 26.91},
}

ELECTIONS = {  # election date -> incumbent party going in
    "2002-07-27": "Labour", "2005-09-17": "Labour", "2008-11-08": "Labour",
    "2011-11-26": "National", "2014-09-20": "National", "2017-09-23": "National",
    "2020-10-17": "Labour", "2023-10-14": "Labour",
}


def government_at(date):
    for start, end, party, name in GOVERNMENTS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return party, name
    return None, None


def load_polls_quarterly():
    rows = []
    for f in sorted(DATA_DIR.glob("*_polling.json")):
        if "_pm_" in f.name:
            continue
        for p in json.load(open(f)).get("polls", []):
            if "election" in (p.get("pollster") or "").lower():
                continue
            rows.append({"date": p["date"], **p.get("parties", {})})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ["National", "Labour"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    q = df.groupby("quarter")[["National", "Labour"]].mean()
    # Assign a quarter to a government by its midpoint
    govs = [government_at(d + pd.Timedelta(days=45)) for d in q.index]
    q["incumbent"] = [g[0] for g in govs]
    q["government"] = [g[1] for g in govs]
    q["incumbent_vote"] = [r[r["incumbent"]] if isinstance(r["incumbent"], str) else np.nan for _, r in q.iterrows()]
    return q


def load_income():
    df = pd.read_csv(DATA_DIR / "household_disposable_income.csv", parse_dates=["quarter"], index_col="quarter")
    df["hh_yoy"] = df["real_hh_ndi_pc"].pct_change(4) * 100
    df["hh_qoq"] = df["real_hh_ndi_pc"].pct_change(1) * 100
    df["rgndi_yoy"] = df["rgndi_pc"].pct_change(4) * 100
    df["rgndi_qoq"] = df["rgndi_pc"].pct_change(1) * 100
    df["inflation_yoy"] = df["cpi"].pct_change(4) * 100
    return df


def load_approval_quarterly():
    a = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    date_col = [c for c in a.columns if "date" in c.lower()][0]
    score_col = [c for c in a.columns if "mean" in c.lower() or "score" in c.lower()][0]
    a["quarter"] = pd.to_datetime(a[date_col]).dt.to_period("Q").dt.to_timestamp()
    return a.groupby("quarter")[score_col].mean().rename("approval")


def nw(y, X, lags=None):
    X = sm.add_constant(X)
    if lags is None:
        lags = int(np.ceil(4 * (len(y) / 100) ** (2 / 9)))
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def fmt_r(x, y):
    r, p = pearsonr(x, y)
    return r, p, f"r = {r:+.3f} (p = {p:.3f}, n = {len(x)})"


def run_block(label, df, growth_yoy, growth_qoq, results):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    d = df.dropna(subset=[growth_yoy, "incumbent_vote"]).copy()
    print(f"Sample: {d.index.min():%Y-%m} to {d.index.max():%Y-%m}, {len(d)} quarters")
    out = {"n": len(d), "start": f"{d.index.min():%Y-%m}", "end": f"{d.index.max():%Y-%m}"}

    # 1. Level: income growth vs incumbent vote level
    r, p, s = fmt_r(d[growth_yoy], d["incumbent_vote"])
    m = nw(d["incumbent_vote"], d[[growth_yoy]])
    print(f"1. Level: yoy growth vs incumbent vote: {s}")
    print(f"   OLS slope {m.params[growth_yoy]:+.3f} pp per 1% growth; Newey-West p = {m.pvalues[growth_yoy]:.3f}")
    out["level"] = {"r": r, "p": p, "slope": m.params[growth_yoy], "nw_p": m.pvalues[growth_yoy]}

    # 2. Within-government: add a dummy per government (removes era differences)
    dummies = pd.get_dummies(d["government"], drop_first=True, dtype=float)
    m_fe = nw(d["incumbent_vote"], pd.concat([d[[growth_yoy]], dummies], axis=1))
    print(f"2. Within-government (FE): slope {m_fe.params[growth_yoy]:+.3f}, Newey-West p = {m_fe.pvalues[growth_yoy]:.3f}")
    out["within_govt"] = {"slope": m_fe.params[growth_yoy], "nw_p": m_fe.pvalues[growth_yoy]}

    # 3. Changes: quarterly change in incumbent vote vs growth (same quarter and lag 1),
    #    excluding quarters where the government changed
    d["d_vote"] = d["incumbent_vote"].diff()
    d["same_gov"] = d["government"].eq(d["government"].shift())
    c = d[d["same_gov"]].dropna(subset=["d_vote", growth_qoq])
    r0, p0, s0 = fmt_r(c[growth_qoq], c["d_vote"])
    c1 = c.assign(lag=c[growth_qoq].shift()).dropna(subset=["lag"])
    r1, p1, s1 = fmt_r(c1["lag"], c1["d_vote"])
    print(f"3. Changes: q/q growth vs q/q incumbent vote change: lag 0 {s0}")
    print(f"                                                     lag 1 {s1}")
    out["changes"] = {"lag0_r": r0, "lag0_p": p0, "lag1_r": r1, "lag1_p": p1, "n": len(c)}

    # 4. Granger: does growth help predict vote changes beyond past vote changes?
    g = c[["d_vote", growth_qoq]].dropna()
    maxlag = 4 if len(g) > 60 else 2
    gc = grangercausalitytests(g, maxlag=maxlag, verbose=False)
    gres = {lag: (gc[lag][0]["ssr_ftest"][0], gc[lag][0]["ssr_ftest"][1]) for lag in gc}
    print("4. Granger (income growth -> vote change): " + ", ".join(f"lag {k}: F = {f:.2f}, p = {pv:.3f}" for k, (f, pv) in gres.items()))
    out["granger"] = {str(k): {"F": f, "p": pv} for k, (f, pv) in gres.items()}

    # 5. Horse race with CPI inflation
    h = d.dropna(subset=["inflation_yoy"])
    m_h = nw(h["incumbent_vote"], h[[growth_yoy, "inflation_yoy"]])
    r_inf, p_inf, s_inf = fmt_r(h["inflation_yoy"], h["incumbent_vote"])
    print(f"5. Inflation alone vs incumbent vote: {s_inf}")
    print(f"   Joint: income slope {m_h.params[growth_yoy]:+.3f} (NW p = {m_h.pvalues[growth_yoy]:.3f}), "
          f"inflation slope {m_h.params['inflation_yoy']:+.3f} (NW p = {m_h.pvalues['inflation_yoy']:.3f})")
    out["horse_race"] = {"income_slope": m_h.params[growth_yoy], "income_nw_p": m_h.pvalues[growth_yoy],
                         "inflation_slope": m_h.params["inflation_yoy"], "inflation_nw_p": m_h.pvalues["inflation_yoy"],
                         "inflation_alone_r": r_inf, "inflation_alone_p": p_inf}

    # 5b. Robustness: drop the COVID quarters (2020Q2-Q4), when income support and
    #     the rally effect move together
    covid = (d.index >= "2020-04-01") & (d.index <= "2020-12-31")
    x = d[~covid]
    r_x, p_x, s_x = fmt_r(x[growth_yoy], x["incumbent_vote"])
    cx = c[~((c.index >= "2020-04-01") & (c.index <= "2020-12-31"))]
    r_cx, p_cx, s_cx = fmt_r(cx[growth_qoq], cx["d_vote"])
    print(f"5b. Excluding 2020Q2-Q4: level {s_x}; changes lag 0 {s_cx}")
    out["ex_covid"] = {"level_r": r_x, "level_p": p_x, "change_r": r_cx, "change_p": p_cx}

    # 6. By incumbent party
    for party in ["Labour", "National"]:
        sub = d[d["incumbent"] == party]
        if len(sub) >= 8:
            rp, pp, sp = fmt_r(sub[growth_yoy], sub["incumbent_vote"])
            print(f"6. {party} in government: {sp}")
            out[f"{party.lower()}_govt"] = {"r": rp, "p": pp, "n": len(sub)}

    results[label] = out
    return d


def myopic_test(income, results):
    """Election-level: RGNDI per capita growth in the year before the election vs
    the incumbent party's vote change since the previous election."""
    print(f"\n{'=' * 70}\nELECTION LEVEL: RGNDI per capita growth before each election\n{'=' * 70}")
    rows = []
    dates = sorted(ELECTION_RESULTS)
    for prev, cur in zip(dates, dates[1:]):
        inc = ELECTIONS[cur]
        q = pd.Timestamp(cur).to_period("Q").to_timestamp()
        # last full quarter before the election
        q_last = q - pd.DateOffset(months=3)
        if q_last not in income.index or q_last - pd.DateOffset(months=12) not in income.index:
            continue
        g = (income.loc[q_last, "rgndi_pc"] / income.loc[q_last - pd.DateOffset(months=12), "rgndi_pc"] - 1) * 100
        dv = ELECTION_RESULTS[cur][inc] - ELECTION_RESULTS[prev][inc]
        rows.append({"election": cur[:4], "incumbent": inc, "rgndi_growth_pre": g, "incumbent_change": dv})
    e = pd.DataFrame(rows).dropna()
    for _, r in e.iterrows():
        print(f"   {r['election']} {r['incumbent']:9s} RGNDI pc growth {r['rgndi_growth_pre']:+5.1f}%   incumbent vote change {r['incumbent_change']:+5.1f}pp")
    r, p, s = fmt_r(e["rgndi_growth_pre"], e["incumbent_change"])
    print(f"   {s}")
    results["election_level"] = {"r": r, "p": p, "n": len(e), "rows": e.to_dict("records")}


def plot(hh, results):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax = axes[0]
    ax.plot(hh.index, hh["real_hh_ndi_pc"] * 4, color="#333", lw=2)
    ax.set_ylabel("Real household disposable\nincome per person (NZ\\$ a year, June 2026 prices)")
    ax.set_title("Real household disposable income per person vs incumbent vote, 2017–2026")
    ax.grid(alpha=0.3)
    ax = axes[1]
    colors = hh["incumbent"].map({"Labour": "#d32f2f", "National": "#1565c0"})
    ax.scatter(hh.index, hh["incumbent_vote"], c=colors, s=18)
    ax.set_ylabel("Incumbent party vote (%)\n(red = Labour govt, blue = National)")
    ax.grid(alpha=0.3)
    lv = results["HOUSEHOLD (2016-2026)"]["level"]
    ax.text(0.01, 0.04, f"yoy income growth vs incumbent vote: r = {lv['r']:+.2f}, Newey-West p = {lv["nw_p"]:.3f}",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    out = REPORTS_DIR / "household_income_vs_incumbent.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def main():
    polls = load_polls_quarterly()
    income = load_income()
    df = income.join(polls, how="inner")
    results = {}

    hh = run_block("HOUSEHOLD (2016-2026)", df, "hh_yoy", "hh_qoq", results)
    run_block("RGNDI PER CAPITA (1999-2026)", df[df.index >= "1999-10-01"], "rgndi_yoy", "rgndi_qoq", results)

    # Approval mediation (household series, Ipsos waves)
    appr = load_approval_quarterly()
    a = hh.join(appr, how="inner").dropna(subset=["approval", "hh_yoy"])
    r_ia, p_ia, s_ia = fmt_r(a["hh_yoy"], a["approval"])
    r_av, p_av, s_av = fmt_r(a["approval"], a["incumbent_vote"])
    print(f"\nAPPROVAL: income growth vs Ipsos govt rating: {s_ia}")
    print(f"          govt rating vs incumbent vote:      {s_av}")
    m = nw(a["incumbent_vote"], a[["hh_yoy", "approval"]])
    print(f"          joint: income NW p = {m.pvalues['hh_yoy']:.3f}, approval NW p = {m.pvalues['approval']:.3f}")
    results["approval"] = {"income_approval_r": r_ia, "income_approval_p": p_ia, "n": len(a),
                           "joint_income_nw_p": m.pvalues["hh_yoy"], "joint_approval_nw_p": m.pvalues["approval"]}

    myopic_test(income, results)
    plot(hh, results)

    with open(REPORTS_DIR / "household_income_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
