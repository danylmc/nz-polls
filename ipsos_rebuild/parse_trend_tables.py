import re, sys, json
MON = {"Mar":"03","Apr":"04","Jan":"01","Dec":"12","Feb":"02","May":"05","Jun":"06","Jul":"07","July":"07","Aug":"08","Sep":"09","Sept":"09","Oct":"10","Nov":"11"}
ROWS = {"National Party":"national","Labour Party":"labour","New Zealand First":"nz_first","Green Party":"green",
        "ACT":"act","Te Pāti Māori":"te_pati_maori","Te Pati Māori":"te_pati_maori","Te Pāti Maori":"te_pati_maori","Te Pati Maori":"te_pati_maori","Māori Party":"te_pati_maori","The Opportunity Party*":"top","Other":"other","None":"none","Issue":"salience"}
ISSUE = {"inflation":"inflation_cost_of_living","healthcare":"healthcare_hospitals","the economy":"the_economy",
         "housing":"housing_price","crime":"crime_law_order","unemployment":"unemployment","petrol":"petrol","taxation":"taxation"}
def parse(path, ed):
    L = open(path, encoding="utf-8").read().splitlines()
    heads = [(i, l) for i, l in enumerate(L) if re.search(r"(issue|spotlight) [–-] ", l, re.I) and len(l.strip()) < 90]
    out = []
    for k, (i, h) in enumerate(heads):
        end = heads[k+1][0] if k+1 < len(heads) else len(L)
        name = re.split(r"[–-] ", h, maxsplit=1)[1].strip().lower()
        iss = next((v for kk, v in ISSUE.items() if name.startswith(kk)), name)
        dl = next((j for j in range(i, end) if len(re.findall(r"[A-Za-z]{3,4} '\d\d", L[j])) >= 3), None)
        if dl is None: continue
        dates = ["20"+y+"-"+MON[m.capitalize()] for m, y in re.findall(r"([A-Za-z]{3,4}) '(\d\d)", L[dl])]
        rec = {}
        for j in range(dl+1, end):
            seg = L[j][85:] if len(L[j]) > 85 else ""
            for lab, key in ROWS.items():
                if seg.strip().startswith(lab + " ") or seg.strip().startswith(lab + "\t"):
                    vals = [int(x) for x in re.findall(r"(\d+)%", seg[seg.index(lab)+len(lab):])]
                    if key not in rec: rec[key] = vals
        for key, vals in rec.items():
            if key == "top" or len(vals) != len(dates):
                if key != "top": print(f"WARN ed{ed} {iss} {key}: {len(vals)} vals vs {len(dates)} dates", file=sys.stderr)
                continue
        for n, d in enumerate(dates):
            row = {"edition": ed, "date": d, "issue": iss}
            for key, vals in rec.items():
                if key == "top": row["top"] = vals[0] if (n == len(dates)-1 and vals) else 0
                elif len(vals) == len(dates): row[key] = vals[n]
            out.append(row)
    return out
rows = []
for ed in sys.argv[2:]:
    rows += parse(f"{sys.argv[1]}/{ed}.txt", int(ed))
json.dump(rows, sys.stdout)
