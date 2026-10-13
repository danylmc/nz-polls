import sys, os, json, re
from urllib.parse import urljoin, unquote, quote
from camoufox.sync_api import Camoufox
out = sys.argv[1]
found = {}
with Camoufox(headless=True) as b:
    pg = b.new_page()
    for q in ["issues monitor 2019", "issues monitor 2018", "issues monitor 2020", "issues monitor 2021", "NZ issues monitor"]:
        for page in range(0, 6):
            pg.goto(f"https://www.ipsos.com/en-nz/search?search={quote(q)}&page={page}", timeout=60000); pg.wait_for_timeout(6000)
            for a in pg.query_selector_all("a"):
                h = a.get_attribute("href") or ""; t = (a.inner_text() or "").strip().replace("\n"," ")[:90]
                if h.startswith("/en-nz/") and re.search(r"issues?[- ]monitor", h+" "+t, re.I) and not re.search(r"housing|health|cost-living|inflation", h): found[h] = t
    print("\n".join(f"{h} | {t}" for h, t in sorted(found.items())))
    links = []
    for p in sorted(found):
        if not re.search(r"2018|2019|2020|2021|1st|4th|5th|9th|10th|15th|fuel|housing-grow", p): continue
        u = "https://www.ipsos.com" + p
        try: pg.goto(u, timeout=60000); pg.wait_for_timeout(4000)
        except Exception as e: print("ERR", p, e); continue
        pdfs = sorted({urljoin(u, a.get_attribute("href")) for a in pg.query_selector_all("a") if (a.get_attribute("href") or "").lower().split("?")[0].endswith(".pdf")})
        links.append({"path": p, "url": u, "page_title": pg.title(), "pdfs": pdfs})
        for pdf in pdfs:
            fn = os.path.join(out, os.path.basename(unquote(pdf.split("?")[0])))
            if os.path.exists(fn): continue
            r = pg.request.get(pdf, timeout=120000)
            if r.status == 200 and r.body()[:4] == b"%PDF":
                open(fn, "wb").write(r.body()); print("saved", os.path.basename(fn))
            else: print("FAIL", r.status, pdf)
    json.dump(links, open(os.path.join(out, "_links.json"), "w"), indent=2)
