import pymupdf, sys, re, json
from collections import defaultdict
MON = {"feb":"02","mar":"03","march":"03","apr":"04","may":"05","jun":"06","june":"06","jul":"07","july":"07","aug":"08","sep":"09","sept":"09","oct":"10","nov":"11"}
out = []
for ed in sys.argv[1:]:
    doc = pymupdf.open(f"ipsos/ed{ed}.pdf")
    for pno, pg in enumerate(doc):
        spans = [s for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
        hdr = []
        for s in spans:
            for m in re.finditer(r"\b([A-Za-z]{3,5})[ ’'‘-]{1,3}(\d{2})\b", s["text"]):
                if m.group(1).lower() in MON:
                    x0 = s["bbox"][0] + (s["bbox"][2]-s["bbox"][0]) * m.start()/max(1,len(s["text"]))
                    hdr.append(((x0 + s["bbox"][2])/2 if len(s["text"]) < 9 else x0+15, s["bbox"][1], "20%s-%s" % (m.group(2), MON[m.group(1).lower()])))
        # header row = most common y among date tokens
        if len(hdr) < 4: continue
        ys = defaultdict(list)
        for h in hdr: ys[round(h[1]/4)].append(h)
        row = max(ys.values(), key=len)
        if len(row) < 4: continue
        row = sorted(row); hy = row[0][1]
        pct = [s for s in spans if re.fullmatch(r"\d{1,2} ?%", s["text"].strip()) and s["bbox"][1] < hy]
        if len(pct) < 20: continue
        series = defaultdict(dict)
        for s in pct:
            xc = (s["bbox"][0]+s["bbox"][2])/2
            d = min(row, key=lambda h: abs(h[0]-xc))
            if abs(d[0]-xc) > 45: continue
            series["#%06x" % s["color"]].setdefault(d[2], int(re.sub(r"\D", "", s["text"])))
        out.append({"ed": ed, "page": pno+1, "waves": [r[2] for r in row], "series": series})
json.dump(out, sys.stdout, indent=1)
