import re, sys, json
COLS = {"inflation_cost_of_living":r"inflation|cost of living","healthcare_hospitals":r"healthcare|hospitals","the_economy":r"^(the )?economy",
        "housing_price":r"housing","crime_law_order":r"^crime|law and order","poverty_inequality":r"poverty","unemployment":r"^unemployment",
        "climate_change":r"^climate change","education":r"^education","race_relations":r"race relations"}
res = {}
for ed in sys.argv[2:]:
    L = open(f"{sys.argv[1]}/{ed}.txt", encoding="utf-8").read().splitlines()
    i = next((k for k, l in enumerate(L) if re.search(r"most important issues facing New Zealand", l, re.I) and k > 20), None)
    if i is None: res[ed] = None; continue
    vals = {}
    for l in L[i:i+70]:
        m = re.match(r"^\s*(.{3,80}?)\s{2,}(\d{1,2})%", l)
        if not m: continue
        lab = m.group(1).strip().lower()
        for c, pat in COLS.items():
            if c not in vals and re.search(pat, lab): vals[c] = int(m.group(2)); break
    base = next((l for l in L if re.search(r"Base:", l)), "")
    res[ed] = {"vals": vals, "base": base.strip()[:80]}
json.dump(res, sys.stdout, indent=1)
