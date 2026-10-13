# Ipsos Issues Monitor rebuild (Oct 2026)

One-off scripts used to rebuild `data/ipsos_issue_salience.csv` and
`data/ipsos_party_capability.csv` after the original chart-read values
turned out to be wrong. Kept for provenance; they are not a pipeline.

Working layout they expect: a directory `ipsos/` holding `NN.txt`
(`pdftotext -layout` of edition NN) and `edNN.pdf` (symlink to the PDF).

1. `scrape_2026_editions.py`, `scrape_older_editions.py` — Camoufox
   (`/mnt/d/data/.venv`) download from ipsos.com/en-nz. The 31st edition's
   report is published under a "Media Release ..._0.pdf" name.
2. `parse_trend_tables.py ipsos 27 ... 34` — per-issue "most capable party"
   trend tables (editions 27+). Gives capability for Feb 2023 on and
   salience for the six core issues (the "Issue" row). Identical across editions.
3. `parse_salience_text.py` / `parse_salience_bbox.py` — each edition's own
   "most important issues" bar chart. The bbox version matches each % to the
   label words immediately to its left using `pdftotext -bbox` coordinates
   (top bars carry their value inside the bar, which the text layout drops).
   Validated against step 2: 115/128 values recovered, none wrong.
4. `parse_colour_trend_charts.py` (needs PyMuPDF, in `/mnt/d/data/.venv`) —
   over-time line charts; series identified by label colour, matched to
   issues by exact agreement with known waves. Used for Mar 2019, Apr 2019
   and Mar 2020, which have no report of their own.

Hand-read exceptions: the 3rd edition (Oct 2018) uses grouped bars and an
Australia comparison column; its values and Feb 2018 were read from page
coordinates. Sept 2026 economy capability for National (28%) comes from the
report's stated Labour value (30%) and 2-point gap, because the table row is
overlaid by the legend.
