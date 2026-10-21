"""Listener-style chart: share of Acts that skipped select committee, by Parliament, 2000-2026.
Data from urgency_analysis.py (reports/urgency_by_parliament.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

for f in font_manager.findSystemFonts(fontpaths=['/home/danylmc/.local/share/fonts/lato']):
    font_manager.fontManager.addfont(f)

HERE = Path(__file__).parent
INK, MUTED, PANEL, LEADER = '#1a1a1a', '#666666', '#dceaf7', '#999999'
LABOUR, NATIONAL = '#d6404f', '#00529f'
plt.rcParams.update({'font.family': 'Lato', 'axes.facecolor': PANEL, 'figure.facecolor': 'white',
                     'axes.edgecolor': PANEL, 'axes.grid': True, 'grid.color': 'white',
                     'grid.linewidth': 1.2, 'axes.axisbelow': True,
                     'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': 11, 'ytick.labelsize': 11})

s = pd.read_csv(HERE / 'urgency_by_parliament.csv')
years = {46: '1999–02', 47: '2002–05', 48: '2005–08', 49: '2008–11', 50: '2011–14',
         51: '2014–17', 52: '2017–20', 53: '2020–23', 54: '2023–26'}
colours = [LABOUR if 'Labour' in g else NATIONAL for g in s.government]

fig = plt.figure(figsize=(12, 7.2), dpi=100)
ax = fig.add_axes([0.06, 0.15, 0.91, 0.66])
x = range(len(s))
ax.bar(x, s.skipped_sc_pct, color=colours, width=0.68, zorder=2)
for i, v in zip(x, s.skipped_sc_pct):
    ax.text(i, v + 0.4, f'{v:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold',
            color=colours[i])
ax.set_xticks(list(x))
ax.set_xticklabels([f'{l}\n{years[p]}' for l, p in zip(s.label, s.parliament)], fontsize=11)
ax.tick_params(axis='x', length=0, pad=8)
ax.tick_params(axis='y', length=0)
ax.set_ylim(0, 22.5)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_yticklabels(['0', '5', '10', '15', '20%'])
ax.grid(axis='x', visible=False)
ax.axhline(0, color=INK, lw=1.6, zorder=3)
ax.set_xlim(-0.6, len(s) - 0.4)

# legend, top-left inside the panel
ax.add_patch(Rectangle((-0.38, 20.75), 0.2, 0.9, color=LABOUR, zorder=3))
ax.text(-0.1, 21.2, 'Labour-led government', color=INK, fontsize=12, va='center')
ax.add_patch(Rectangle((-0.38, 19.25), 0.2, 0.9, color=NATIONAL, zorder=3))
ax.text(-0.1, 19.7, 'National-led government', color=INK, fontsize=12, va='center')

# callouts
r54 = s[s.parliament == 54].iloc[0]
ax.annotate(f'First 100 days: {int(r54.first100_skipped_sc)} of\n{int(r54.first100_acts)} Acts skipped committee',
            xy=(8 - 0.34, 15.5), xytext=(5.75, 20.2), fontsize=12, color=INK, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=LEADER, lw=1))
ax.annotate('Covid-19 laws', xy=(6 - 0.34, 9.5), xytext=(5.0, 13.2), fontsize=12, color=INK,
            ha='center', va='center', arrowprops=dict(arrowstyle='-', color=LEADER, lw=1))

fig.text(0.06, 0.935, 'Almost one new law in five now skips select committee', fontsize=26,
         fontweight='bold', color=INK, ha='left')
fig.text(0.06, 0.885, 'Share of Acts passed without going to a select committee, so the public '
         'never gets to make submissions,', fontsize=13, color=MUTED, ha='left')
fig.text(0.06, 0.855, 'by Parliament. Excludes Appropriation and Imprest Supply Acts, which skip '
         'committee by rule', fontsize=13, color=MUTED, ha='left')
fig.text(0.06, 0.035, 'Source: legislation.govt.nz legislative histories, 2,506 Acts, analysis by '
         'the author | 1999–02 term from Feb 2000; 2023–26 to Sep 2026',
         fontsize=10.5, color=MUTED, ha='left')

fig.savefig(HERE / 'urgency_select_committee_listener.svg')
fig.savefig(HERE / 'urgency_select_committee_listener.png', dpi=200)
print('saved')
