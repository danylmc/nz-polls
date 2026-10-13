#!/usr/bin/env python3
"""
What drives government approval? (Analysis 13)

Ipsos government performance rating vs Ipsos issue salience and the RBNZ
2-year fixed mortgage rate, plus mediation of mortgage rate -> approval ->
incumbent vote. Previously computed ad hoc; this script reproduces it.
"""
import json, glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, norm

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = Path(__file__).parent / "raw_data"

SALIENCE = ["inflation_cost_of_living", "unemployment", "crime_law_order", "the_economy",
            "housing_price", "education", "healthcare_hospitals", "poverty_inequality",
            "climate_change", "race_relations"]


def get_incumbent(d):
    if d < datetime(2008, 11, 8): return "Labour"
    if d < datetime(2017, 10, 26): return "National"
    if d < datetime(2023, 11, 27): return "Labour"
    return "National"


def load_incumbent_monthly():
    rows = []
    for f in sorted(glob.glob(str(DATA_DIR / "*_polling.json"))):
        with open(f) as fh:
            for p in json.load(fh).get("polls", []):
                row = {"date": p["date"]}
                row.update(p.get("parties", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["inc_vote"] = pd.to_numeric(df.apply(lambda r: r.get(get_incumbent(r["date"])), axis=1), errors="coerce")
    df = df.dropna(subset=["inc_vote"])
    return df.set_index("date")["inc_vote"].resample("MS").mean()


def load_mortgage_monthly():
    mort = pd.read_excel(RAW_DIR / "hb20.xlsx", header=None, skiprows=5).iloc[:, [0, 6]]
    mort.columns = ["date", "rate_2yr"]
    mort["date"] = pd.to_datetime(mort["date"], errors="coerce")
    mort = mort.dropna()
    return mort.set_index("date")["rate_2yr"].astype(float).resample("MS").last()


def load_ipsos():
    appr = pd.read_csv(DATA_DIR / "ipsos_govt_performance.csv")
    appr["date"] = pd.to_datetime(appr["date"])
    sal = pd.read_csv(DATA_DIR / "ipsos_issue_salience.csv")
    sal["date"] = pd.to_datetime(sal["date"])
    return appr.set_index("date")[["mean_score", "government"]], sal.set_index("date")[SALIENCE]


def stepwise(df, y, candidates, max_steps=3):
    chosen, out = [], []
    prev = 0.0
    for _ in range(max_steps):
        best = None
        for c in candidates:
            if c in chosen: continue
            sub = df.dropna(subset=[y] + chosen + [c])
            m = sm.OLS(sub[y], sm.add_constant(sub[chosen + [c]])).fit()
            if best is None or m.rsquared_adj > best[1]: best = (c, m.rsquared_adj, len(sub))
        if best is None or best[1] <= prev: break
        chosen.append(best[0]); out.append((best[0], best[1], best[1] - prev, best[2])); prev = best[1]
    return out


def main():
    inc = load_incumbent_monthly()
    mort = load_mortgage_monthly()
    appr, sal = load_ipsos()

    df = appr.join(mort, how="left").join(inc.rename("inc_vote"), how="left").join(sal, how="left")
    print(f"Approval waves: {len(appr)}; with mortgage+vote: {df.dropna(subset=['rate_2yr','inc_vote']).shape[0]}; "
          f"with salience: {df.dropna(subset=['inflation_cost_of_living']).shape[0]}")

    print("\n=== DRIVERS OF APPROVAL (levels) ===")
    res = []
    for c in SALIENCE + ["rate_2yr"]:
        sub = df.dropna(subset=["mean_score", c])
        r, p = pearsonr(sub[c], sub["mean_score"])
        res.append((c, r, p, len(sub)))
    for c, r, p, n in sorted(res, key=lambda x: -abs(x[1])):
        print(f"  {c:26s} r = {r:+.3f}  p = {p:.4f}  n = {n}")

    print("\n=== CHANGE CORRELATIONS (wave-to-wave Δ) ===")
    d = df[["mean_score"] + SALIENCE + ["rate_2yr"]].dropna(subset=["inflation_cost_of_living"]).diff()
    for c in ["unemployment", "crime_law_order", "healthcare_hospitals", "housing_price", "inflation_cost_of_living", "rate_2yr"]:
        sub = d.dropna(subset=["mean_score", c])
        r, p = pearsonr(sub[c], sub["mean_score"])
        print(f"  Δ{c:25s} r = {r:+.3f}  p = {p:.4f}  n = {len(sub)}")

    print("\n=== STEPWISE MODEL FOR APPROVAL (salience only) ===")
    for c, adj, dr, n in stepwise(df, "mean_score", SALIENCE):
        print(f"  + {c:26s} adj R² = {adj:.3f}  ΔR² = {dr:+.3f}  n = {n}")

    print("\n=== MEDIATION: mortgage → approval → incumbent vote ===")
    med = df.dropna(subset=["rate_2yr", "mean_score", "inc_vote"])
    mod_a = sm.OLS(med["mean_score"], sm.add_constant(med["rate_2yr"])).fit()
    mod_b = sm.OLS(med["inc_vote"], sm.add_constant(med[["rate_2yr", "mean_score"]])).fit()
    mod_c = sm.OLS(med["inc_vote"], sm.add_constant(med["rate_2yr"])).fit()
    a, se_a = mod_a.params["rate_2yr"], mod_a.bse["rate_2yr"]
    b, se_b = mod_b.params["mean_score"], mod_b.bse["mean_score"]
    c = mod_c.params["rate_2yr"]
    indirect = a * b
    z = indirect / np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    print(f"  n = {len(med)}")
    print(f"  Total effect:    {c:.2f}  p = {mod_c.pvalues['rate_2yr']:.4f}")
    print(f"  Direct effect:   {mod_b.params['rate_2yr']:.2f}  p = {mod_b.pvalues['rate_2yr']:.4f}")
    print(f"  Indirect effect: {indirect:.2f}")
    print(f"  % mediated:      {indirect / c * 100:.1f}%")
    print(f"  Sobel z = {z:.2f}  p = {2 * (1 - norm.cdf(abs(z))):.4f}")

    print("\n=== BY GOVERNMENT ===")
    for label, mask in [("Labour (2017–2023)", df.index < "2023-11-01"), ("National (2023–)", df.index >= "2023-11-01")]:
        sub = df[mask & df.index.to_series().ge("2017-10-01").values]
        for c in ["rate_2yr", "inflation_cost_of_living"]:
            s = sub.dropna(subset=["mean_score", c])
            if len(s) > 3:
                r, p = pearsonr(s[c], s["mean_score"])
                print(f"  {label:20s} {c:26s} r = {r:+.3f}  p = {p:.4f}  n = {len(s)}")


if __name__ == "__main__":
    main()
