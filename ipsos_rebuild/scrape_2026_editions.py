import sys, os, json
from urllib.parse import urljoin, unquote
from camoufox.sync_api import Camoufox
out = sys.argv[1]
pages = ["/en-nz/ipsos-new-zealand-31st-issues-monitor", "/en-nz/ipsos-new-zealand-32nd-issues-monitor",
         "/en-nz/ipsos-issues-monitor-july-2026", "/en-nz/34th-ipsos-nz-issues-monitor-sept-2026"]
links = []
with Camoufox(headless=True) as b:
    pg = b.new_page()
    for p in pages:
        u = "https://www.ipsos.com" + p
        pg.goto(u, timeout=60000); pg.wait_for_timeout(5000)
        pdfs = sorted({urljoin(u, a.get_attribute("href")) for a in pg.query_selector_all("a") if (a.get_attribute("href") or "").lower().split("?")[0].endswith(".pdf")})
        date = pg.query_selector("time")
        print(p, "|", pg.title(), "|", date.inner_text() if date else "", "|", pdfs)
        links.append({"path": p, "url": u, "page_title": pg.title(), "pdfs": pdfs})
        for pdf in pdfs:
            r = pg.request.get(pdf, timeout=120000)
            fn = os.path.join(out, os.path.basename(unquote(pdf.split("?")[0])))
            with open(fn, "wb") as f: f.write(r.body())
            print("   saved", fn, r.status, len(r.body()))
json.dump(links, open(os.path.join(out, "_links.json"), "w"), indent=2)
