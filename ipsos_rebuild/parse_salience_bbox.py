import re, sys, json, subprocess, html
COLS = {"inflation_cost_of_living":r"inflation|cost of living","healthcare_hospitals":r"healthcare|hospitals","the_economy":r"^(the )?economy$",
        "housing_price":r"housing","crime_law_order":r"^crime|law (and|&) order","poverty_inequality":r"poverty","unemployment":r"^unemployment$",
        "climate_change":r"^climate change$","education":r"^education$","race_relations":r"race relations"}
def pages(pdf):
    x = subprocess.run(["pdftotext","-bbox",pdf,"-"],capture_output=True,text=True).stdout
    for pg in re.findall(r"<page.*?</page>", x, re.S):
        yield [(float(a),float(b),float(c),float(d),html.unescape(w)) for a,b,c,d,w in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', pg)]
def parse(pdf):
    best = None
    for words in pages(pdf):
        txt = " ".join(w[4] for w in words)
        if not re.search(r"most important issues facing", txt, re.I) or "overpopulation" not in txt.lower(): continue
        if txt.count("%") < 15 or re.search(r"KEY FINDINGS|OVER TIME|NEXT 5 YEARS|TOP-5|TOP 5|NOTEWORTHY|Spotlight", txt[:200], re.I): continue
        pct = [w for w in words if re.fullmatch(r"\d{1,2}%", w[4])]
        lab = [w for w in words if not re.fullmatch(r"\d{1,2}%", w[4])]
        vals = {}
        for pw in pct:
            yc = (pw[1]+pw[3])/2
            row = sorted([w for w in lab if abs((w[1]+w[3])/2 - yc) < 4 and w[2] <= pw[0] + 1], key=lambda w: w[0])
            if not row: continue
            cl = [row[-1]]
            for w in reversed(row[:-1]):
                if cl[0][0] - w[2] < 12: cl.insert(0, w)
                else: break
            text = re.sub(r"\s*/\s*", " / ", " ".join(w[4] for w in cl).lower()).strip()
            for c, pat in COLS.items():
                if c not in vals and re.search(pat, text):
                    vals[c] = int(pw[4][:-1]); break
        if best is None or len(vals) > len(best): best = vals
    return best
out = {f: parse(f) for f in sys.argv[1:]}
json.dump(out, sys.stdout, indent=1)
