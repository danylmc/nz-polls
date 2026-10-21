#!/usr/bin/env python3
"""
Urgency and select committee bypass by Parliament, 1999-2026.

Parses the legislative histories cached by urgency_fetch.py and classifies every
public Act:

- skipped_sc: never referred to a select committee (no "referral to" / "Reported
  from" in its history). Divided bills inherit from the bill they were split from.
- all_stages_one_go: skipped select committee AND first to third reading within
  7 days, i.e. pushed through in a single urgency sitting (which can span several
  calendar days).
- supply: Appropriation and Imprest Supply Acts, which skip select committee by
  Standing Orders; reported separately and excluded from the headline shares.

Acts are assigned to a Parliament by third reading date. Legislative history
tables start in 2000, so the 46th Parliament (1999-2002) is missing its first weeks.

Outputs:
  data/acts_legislative_history.csv     one row per Act
  reports/urgency_by_parliament.csv     per-Parliament summary
"""

import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CACHE = BASE / "raw_data" / "legislation_history"
ACTS_OUT = BASE / "data" / "acts_legislative_history.csv"
SUMMARY_OUT = BASE / "reports" / "urgency_by_parliament.csv"

# (parliament, election date, government sworn in, government, short label)
PARLIAMENTS = [
    (46, date(1999, 11, 27), date(1999, 12, 10), "Labour–Alliance", "Clark I"),
    (47, date(2002, 7, 27), date(2002, 8, 15), "Labour–Progressive", "Clark II"),
    (48, date(2005, 9, 17), date(2005, 10, 19), "Labour–Progressive", "Clark III"),
    (49, date(2008, 11, 8), date(2008, 11, 19), "National (ACT, UF, Māori)", "Key I"),
    (50, date(2011, 11, 26), date(2011, 12, 14), "National (ACT, UF, Māori)", "Key II"),
    (51, date(2014, 9, 20), date(2014, 10, 8), "National (ACT, UF, Māori)", "Key/English"),
    (52, date(2017, 9, 23), date(2017, 10, 26), "Labour–NZ First (Greens)", "Ardern I"),
    (53, date(2020, 10, 17), date(2020, 11, 6), "Labour majority", "Ardern/Hipkins"),
    (54, date(2023, 10, 14), date(2023, 11, 27), "National–ACT–NZ First", "Luxon"),
]

DATE_RE = re.compile(r"^\s*(\d{1,2} [A-Z][a-z]+ \d{4})\s{2,}(.*)$")
BILL_RE = re.compile(r"Bill (\d+)[–-]")


def fix_mojibake(s):
    if s and ("Ã" in s or "Ä" in s or "Å" in s):
        try:
            return s.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def parse_history(text):
    """Return [(date, event_text)] from a legislative history block."""
    events = []
    for line in text.splitlines():
        if line.strip() == "Legislative history" or not line.strip():
            continue
        m = DATE_RE.match(line)
        if m:
            # first three letters only: the source has typos like "Feburary"
            day, month, year = m.group(1).split()
            d = datetime.strptime(f"{day} {month[:3]} {year}", "%d %b %Y").date()
            events.append([d, m.group(2).strip()])
        elif events:
            events[-1][1] += " " + line.strip()
    return [(d, re.sub(r"\s+", " ", t)) for d, t in events]


def first_date(events, pattern):
    for d, t in events:
        if re.search(pattern, t, re.I):
            return d
    return None


def parliament_for(d):
    current = None
    for p in PARLIAMENTS:
        if d >= p[1]:
            current = p
    return current


def classify(rec):
    row = {"year": rec["year"], "num": rec["num"],
           "title": fix_mojibake(rec.get("title", "")),
           "as_enacted": rec.get("as_enacted"), "error": rec.get("error")}
    hist = fix_mojibake(rec.get("history") or "")
    events = parse_history(hist) if hist else []
    row["has_history"] = bool(events)
    if not events:
        return row
    full = " | ".join(t for _, t in events)
    row["history"] = " | ".join(f"{d.isoformat()}: {t}" for d, t in events)
    row["introduced"] = first_date(events, r"Introduction|Divided from|Reinstated")
    row["first_reading"] = first_date(events, r"first reading")
    row["referred"] = first_date(events, r"referral to|referred to")
    row["reported"] = first_date(events, r"Reported from (?!(the )?committee of the whole)")
    row["second_reading"] = first_date(events, r"second reading")
    row["third_reading"] = first_date(events, r"third reading")
    row["assent"] = first_date(events, r"Royal assent")
    row["divided"] = "Divided from" in full
    m = BILL_RE.search(full)
    row["bill_no"] = int(m.group(1)) if m else None
    row["went_to_sc"] = bool(row["referred"] or row["reported"])
    t = row["title"]
    row["supply"] = bool(re.match(r"(Appropriation|Imprest Supply)\b", t))
    return row


def main():
    recs = [json.loads(p.read_text()) for p in sorted(CACHE.glob("*.json"))]
    df = pd.DataFrame([classify(r) for r in recs])
    df = df[df["has_history"]].copy()
    for c in ["divided", "went_to_sc", "supply"]:
        df[c] = df[c].astype(bool)

    for c in ["introduced", "first_reading", "referred", "reported",
              "second_reading", "third_reading", "assent"]:
        df[c] = pd.to_datetime(df[c])

    ref_date = df["third_reading"].fillna(df["assent"])
    p = ref_date.dt.date.map(parliament_for)
    df["parliament"] = p.map(lambda x: x[0] if x else None)
    df["government"] = p.map(lambda x: x[3] if x else None)
    df["days_since_election"] = [
        (d.date() - x[1]).days if x else None for d, x in zip(ref_date, p)]
    df["days_since_sworn_in"] = [
        (d.date() - x[2]).days if x else None for d, x in zip(ref_date, p)]

    # Divided bills inherit select committee status from the parent bill
    # (same Parliament, same bill number).
    # A divided bill whose parent isn't in the data almost always split off after
    # select committee; treat as having gone to committee.
    parent_sc = (df[~df["divided"]].groupby(["parliament", "bill_no"])["went_to_sc"].any()).to_dict()
    df["went_to_sc"] = [
        parent_sc.get((pa, b), True) if div else sc
        for pa, b, div, sc in zip(df["parliament"], df["bill_no"], df["divided"], df["went_to_sc"])]

    df["skipped_sc"] = ~df["went_to_sc"]
    df["fr_to_tr_days"] = (df["third_reading"] - df["first_reading"]).dt.days
    df["all_stages_one_go"] = df["skipped_sc"] & (df["fr_to_tr_days"] <= 7)
    df["sc_days"] = (df["reported"] - df["referred"]).dt.days
    df["short_sc"] = df["sc_days"] < 42   # under six weeks for public submissions and hearings

    df = df.sort_values(["third_reading", "year", "num"])
    cols = ["year", "num", "title", "parliament", "government", "introduced",
            "first_reading", "referred", "reported", "second_reading",
            "third_reading", "assent", "days_since_election", "days_since_sworn_in", "bill_no", "divided",
            "supply", "went_to_sc", "skipped_sc", "all_stages_one_go",
            "fr_to_tr_days", "sc_days", "short_sc", "history"]
    df[cols].to_csv(ACTS_OUT, index=False)

    # ---- per-Parliament summary ----
    # Bill-level count: before ~2020 omnibus bills were split into many Acts at
    # third reading, inflating early Act counts. Collapse divided Acts into one
    # unit per (Parliament, bill number).
    ns_all = df[~df["supply"]].copy()
    ns_all["bill_key"] = [
        f"{pa}-{int(b)}" if pd.notna(b) else f"act-{y}-{n}"
        for pa, b, y, n in zip(ns_all["parliament"], ns_all["bill_no"], ns_all["year"], ns_all["num"])]
    bills = ns_all.groupby("bill_key").agg(parliament=("parliament", "first"),
                                           skipped=("skipped_sc", "all"))
    rows = []
    for num, elect, sworn, govt, label in PARLIAMENTS:
        d = df[df["parliament"] == num]
        if d.empty:
            continue
        ns = d[~d["supply"]]
        first100 = ns[ns["days_since_sworn_in"] <= 100]
        sc = ns[ns["went_to_sc"] & ns["sc_days"].notna()]
        rows.append({
            "parliament": num, "government": govt, "label": label,
            "election": elect.isoformat(),
            "acts": len(d), "supply_acts": int(d["supply"].sum()),
            "acts_ex_supply": len(ns),
            "skipped_sc": int(ns["skipped_sc"].sum()),
            "skipped_sc_pct": round(100 * ns["skipped_sc"].mean(), 1),
            "all_stages_one_go": int(ns["all_stages_one_go"].sum()),
            "all_stages_one_go_pct": round(100 * ns["all_stages_one_go"].mean(), 1),
            "bills_ex_supply": int((bills["parliament"] == num).sum()),
            "bills_skipped_sc": int(bills.loc[bills["parliament"] == num, "skipped"].sum()),
            "bills_skipped_sc_pct": round(100 * bills.loc[bills["parliament"] == num, "skipped"].mean(), 1),
            "first100_acts": len(first100),
            "first100_skipped_sc": int(first100["skipped_sc"].sum()),
            "median_sc_days": sc["sc_days"].median(),
            "short_sc": int(sc["short_sc"].sum()),
            "short_sc_pct": round(100 * sc["short_sc"].mean(), 1),
            "first_act": d["third_reading"].min().date().isoformat(),
            "last_act": d["third_reading"].max().date().isoformat(),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(SUMMARY_OUT, index=False)

    pd.set_option("display.width", 200)
    print(summ.drop(columns=["election"]).to_string(index=False))
    print(f"\nActs parsed: {len(df)}; without history skipped: "
          f"{len(recs) - len(df)}")


if __name__ == "__main__":
    main()
