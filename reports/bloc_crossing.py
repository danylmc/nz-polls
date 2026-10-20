#!/usr/bin/env python3
"""Do vote switchers cross the left-right divide less than they used to?

NZES 1996-2023: recalled previous party vote vs current party vote, classified into
blocs (Left = Labour, Green, Alliance, Progressive, Mana; Right = National, ACT,
Conservative; NZ First and everyone else = centre/other). Plus like/dislike ratings
of Labour and National (2002-2023) as a measure of affective polarisation.
Aggregate check: per-cycle correlation of poll-to-poll National vs Labour changes.
"""
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "nzesdb.sqlite"

# year: (table, current vote, previous vote, weight, like Labour, like National)
CONFIG = {
    1996: ("nzes_1996", "VOT96P", "PVOT93E", "PWT", None, None),
    1999: ("nzes_1999_post", "VOT99P", "SVOT96P", "NCWT", None, None),
    2002: ("nzes_2002", "WVOT02P", "WVOT99P", "ALZWT", "WLBLK", "WNTLK"),
    2005: ("nzes_2005", "yvot05p", "yvot02p", "YPWT", "ylablk", "ynatlk"),
    2008: ("nzes_2008", "zvt08p", "zvot05p", "ZZWT6", None, None),
    2011: ("nzes_2011", "jpartyvote", "jlastpvote", "jfinwt", "jlablike", "jnatlike"),
    2014: ("nzes_2014", "dpartyvote", "dlastpvote", "dwtfin", "dlablike", "dnatlike"),
    2017: ("nzes_2017", "rpartyvote", "rlastpvote", "rwt", "rlablike", "rnatlike"),
    2020: ("nzes_2020", "E3", "E19", "lfinwt", "B1_1", "B1_2"),
    2023: ("nzes_2023", "mvpartyvote", "mppartyvote", "vwgt", "B1a", "B1b"),
}
LEFT = {"Labour", "Green", "Alliance", "Progressive", "Mana"}
RIGHT = {"National", "ACT", "Conservative"}


def party(label):
    """Normalise a value label to a party name; None for nonvote/DK/refused."""
    s = re.sub(r"^\d+\.\s*", "", str(label)).strip().lower()
    if re.search(r"non ?vote|did not|didn|not vote|none|don|dk|refus|cant|ineligib|missing|dnv|not applicable", s):
        return None
    if "labour" in s:
        return "Labour"
    if "national" in s:
        return "National"
    if "prog green" in s:
        return "Other"
    if "green" in s:
        return "Green"
    if "alliance" in s:
        return "Alliance"
    if "progress" in s:
        return "Progressive"
    if "mana" in s and "maori" not in s:
        return "Mana"
    if re.fullmatch(r"act( nz)?", s):
        return "ACT"
    if s.startswith("cons") or "conservative" in s:
        return "Conservative"
    if re.search(r"nz ?first|new zealand first", s):
        return "NZ First"
    return "Other"


def bloc(p):
    return "Left" if p in LEFT else "Right" if p in RIGHT else "Centre/other"


def wpct(mask, w):
    return np.average(mask, weights=w) * 100 if len(mask) else np.nan


def switching():
    c = sqlite3.connect(DB)
    vl = pd.read_sql("select * from _value_labels", c)
    rows = []
    for year, (t, cur, prev, wt, _, _) in CONFIG.items():
        cols = [cur, prev, wt] + (["mpartyvote"] if year == 2023 else [])
        d = pd.read_sql(f'select {",".join(chr(34) + x + chr(34) for x in cols)} from {t}', c)
        if year == 2023:
            d[cur] = d[cur].fillna(d.mpartyvote)
        for v in (cur, prev):
            labels = vl[(vl.table_name == t) & (vl.column_name == v)].set_index("value").label
            d[v + "_p"] = d[v].map(labels).map(party)
        d["w"] = d[wt].fillna(d[wt].median())
        d = d.dropna(subset=[cur + "_p", prev + "_p"])
        a, b = d[prev + "_p"], d[cur + "_p"]
        ba, bb = a.map(bloc), b.map(bloc)
        sw = a != b
        cross = ((ba == "Left") & (bb == "Right")) | ((ba == "Right") & (bb == "Left"))
        left_nat, left_lab = (a == "National") & sw, (a == "Labour") & sw
        rows.append({
            "year": year, "n": len(d),
            "switched_%": wpct(sw, d.w),
            "crossed_LR_%": wpct(cross, d.w),
            "cross_share_of_switchers_%": wpct(cross[sw], d.w[sw]),
            "Nat_leavers_to_left_%": wpct(bb[left_nat] == "Left", d.w[left_nat]),
            "Nat_leavers_n": int(left_nat.sum()),
            "Lab_leavers_to_right_%": wpct(bb[left_lab] == "Right", d.w[left_lab]),
            "Lab_leavers_n": int(left_lab.sum()),
        })
    return pd.DataFrame(rows).set_index("year")


def affect():
    c = sqlite3.connect(DB)
    vl = pd.read_sql("select * from _value_labels", c)
    rows = []
    for year, (t, cur, _, wt, ll, nl) in CONFIG.items():
        if ll is None:
            continue
        cols = [cur, wt, ll, nl] + (["mpartyvote"] if year == 2023 else [])
        d = pd.read_sql(f'select {",".join(chr(34) + x + chr(34) for x in cols)} from {t}', c)
        if year == 2023:
            d[cur] = d[cur].fillna(d.mpartyvote)
        labels = vl[(vl.table_name == t) & (vl.column_name == cur)].set_index("value").label
        d["v"] = d[cur].map(labels).map(party)
        d["w"] = d[wt].fillna(d[wt].median())
        for x in (ll, nl):
            d[x] = d[x].where(d[x].between(0, 10))
        d = d.dropna(subset=[ll, nl])
        lab, nat = d[d.v == "Labour"], d[d.v == "National"]
        rows.append({
            "year": year,
            "gap_all": np.average((d[ll] - d[nl]).abs(), weights=d.w),
            "Lab_voters_like_Nat": np.average(lab[nl], weights=lab.w),
            "Nat_voters_like_Lab": np.average(nat[ll], weights=nat.w),
            "Lab_voters_warm_Nat_%": wpct(lab[nl] >= 5, lab.w),
            "Nat_voters_warm_Lab_%": wpct(nat[ll] >= 5, nat.w),
            "Lab_voters_hate_Nat_%": wpct(lab[nl] <= 2, lab.w),
            "Nat_voters_hate_Lab_%": wpct(nat[ll] <= 2, nat.w),
        })
    return pd.DataFrame(rows).set_index("year")


def poll_mirror():
    """Per cycle: correlation of consecutive-poll changes in National and Labour, and of
    National's changes with the Labour+Green change (cross-bloc mirror)."""
    rows = []
    for y in range(1999, 2027, 3):
        f = ROOT / "data" / f"{y}_polling.json"
        if not f.exists():
            continue
        df = pd.DataFrame([{"date": p["date"], **p["parties"]} for p in json.load(open(f))["polls"]])
        df["date"] = pd.to_datetime(df.date)
        m = df.set_index("date").sort_index().resample("MS")[["National", "Labour", "Green", "ACT"]].mean().dropna(subset=["National", "Labour"])
        dm = m.diff().dropna(subset=["National", "Labour"]).fillna(0)
        rows.append({
            "cycle": y, "months": len(dm),
            "r_dNat_dLab": dm.National.corr(dm.Labour),
            "r_dNat_dLabGrn": dm.National.corr(dm.Labour + dm.Green),
            "r_dNatACT_dLabGrn": (dm.National + dm.ACT).corr(dm.Labour + dm.Green),
            "sd_dNat": dm.National.std(), "sd_dLab": dm.Labour.std(),
        })
    return pd.DataFrame(rows).set_index("cycle")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("=== Switching between elections (NZES, recalled previous vote, weighted) ===")
    print(switching().round(1).to_string())
    print("\n=== Affective polarisation: like/dislike 0-10 (NZES, weighted) ===")
    print(affect().round(2).to_string())
    print("\n=== Polls: month-to-month National vs Labour changes, per cycle ===")
    print(poll_mirror().round(2).to_string())
