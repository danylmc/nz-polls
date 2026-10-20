"""Listener-style chart: National party vote, Luxon preferred PM and wrong direction since
the 2023 election, with recessions, petrol spikes, key events and Luxon's worst lines marked."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from statsmodels.nonparametric.smoothers_lowess import lowess

for f in font_manager.findSystemFonts(fontpaths=['/home/danylmc/.local/share/fonts/lato']):
    font_manager.fontManager.addfont(f)

INK, MUTED, PANEL, BLUE, LEADER = '#1a1a1a', '#666666', '#dceaf7', '#00529f', '#999999'
plt.rcParams.update({'font.family': 'Lato', 'axes.facecolor': PANEL, 'figure.facecolor': 'white',
                     'axes.edgecolor': PANEL, 'axes.grid': True, 'grid.color': 'white',
                     'grid.linewidth': 1.2, 'axes.axisbelow': True, 'legend.frameon': False,
                     'xtick.color': INK, 'ytick.color': INK, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
T = pd.Timestamp
ELECTION, END = T('2023-10-14'), T('2026-10-20')
FIT_FROM = T('2023-06-01')   # fit trends on some pre-election polls so the start isn't an edge artefact

# Shaded periods. Recessions: quarters of falling real GDP (Stats NZ, current vintage; 2024 Q2-Q3 is the
# only two-quarter fall). Petrol spikes: MBIE regular 91 weekly board price, low to peak, Iran war 2026.
SHADED = [('2024-04-01', '2024-09-30', 'Recession\nGDP −0.5%,\nthen −1.2%'),
          ('2025-04-01', '2025-06-30', 'Sharp\nGDP fall\n−0.7%'),
          ('2026-03-01', '2026-06-15', 'Petrol spike\n\\$2.55 → \\$3.51/L'),
          ('2026-09-01', '2026-10-20', 'Petrol spike\n\\$3.00 → \\$3.54/L')]
EVENTS = [('2024-05-30', 'Tax-cut\nBudget', 'left'), ('2024-11-14', 'Treaty Principles\nBill', 'left'),
          ('2025-05-22', 'Pay-equity\nBudget', 'left'),
          ('2026-04-14', 'Confidence\nvote', 'right'), ('2026-08-12', 'Penk\nchallenge', 'left')]
# (date, quote, label x, label y, ha)
LUXON = [('2024-02-29', '"I\'m entitled to\nthe entitlement"', '2023-11-20', 33.5, 'left'),
         ('2024-10-01', '"Wealthy and\nsorted"', '2024-10-01', 33.5, 'center'),
         ('2025-03-04', '"Make a Marmite\nsandwich"', '2025-04-25', 29.5, 'left'),
         ('2026-07-31', '"Who cares what the\ngovernment thinks or does?"', '2026-03-20', 30.5, 'center')]


def norm(s):
    return s.fillna('none').str.replace(r'[^A-Za-z]', '', regex=True).str.lower().str[:8]


def trend(df, days=75, anchor=None, anchor_weight=8):
    """House-effect-adjusted LOWESS trend; bandwidth ~ `days` either side.
    `anchor` = (date, value): an election result, entered as `anchor_weight` copies with no house effect,
    so the trend starts from the result. Pollster house effects are still centred on the pollster average."""
    df = df.dropna(subset=['v']).copy(); df['pk'] = norm(df.pollster)
    if anchor:
        df = pd.concat([df, pd.DataFrame({'date': [T(anchor[0])] * anchor_weight, 'v': anchor[1], 'pk': 'election'})])
    df = df.sort_values('date')
    t = (df.date - df.date.min()).dt.days.values.astype(float)
    frac = min(1, 2 * days / (t.max() - t.min()))
    adj = df.v.values.copy()
    for _ in range(5):
        fit = lowess(adj, t, frac=frac, return_sorted=False)
        he = pd.Series(df.v.values - fit, index=df.pk.values).groupby(level=0).mean()
        he = he.drop('election', errors='ignore'); he -= he.mean(); he['election'] = 0.0
        adj = df.v.values - df.pk.map(he).values
    sm = lowess(adj, t, frac=frac)
    s = pd.Series(sm[:, 1], index=df.date.min() + pd.to_timedelta(sm[:, 0], unit='D')).groupby(level=0).mean()
    return s[s.index >= ELECTION]


def at(series, date):
    s = series.sort_index()
    days = lambda idx: (pd.DatetimeIndex(idx) - T('2000-01-01')) / pd.Timedelta(days=1)
    return float(np.interp(days([T(date)])[0], days(s.index), s.values))


# ---- data
polls = []
for cyc in (2023, 2026):
    for p in json.load(open(f'data/{cyc}_polling.json'))['polls']:
        polls.append(dict(date=p['date'], pollster=p['pollster'], v=p['parties'].get('National')))
nat = pd.DataFrame(polls); nat['date'] = pd.to_datetime(nat.date)
nat = nat[(nat.date > ELECTION) & ~nat.pollster.str.contains('election result', na=False)]

pm = pd.read_csv('data/pm_polling_all.csv'); pm['date'] = pd.to_datetime(pm.date)
lux = pm[pm.candidate.str.contains('Luxon') & (pm.date >= FIT_FROM)
         & ~pm.pollster.str.contains('Freshwater')].rename(columns={'percent': 'v'})  # Freshwater asks head-to-head

old = pd.DataFrame(json.load(open('data/govt_approval_right_wrong_track.json'))).rename(columns={'wrong_direction': 'wrong'})
rw = pd.concat([old[['date', 'pollster', 'wrong']],
                pd.read_csv('data/right_wrong_direction_2026.csv')[['date', 'pollster', 'wrong']]])
rw['date'] = pd.to_datetime(rw.date); rw = rw[rw.date >= FIT_FROM].rename(columns={'wrong': 'v'})

t_nat, t_lux, t_wrong = trend(nat, anchor=(ELECTION, 38.08)), trend(lux, 90), trend(rw, 90)

# ---- layout
fig = plt.figure(figsize=(12, 11.8), dpi=100)
L, R = 0.085, 0.905
boxes = [(0.595, 0.265), (0.34, 0.215), (0.09, 0.215)]
axes = [fig.add_axes([L, y, R - L, h]) for y, h in boxes]
for ax in axes:
    for a, b, _ in SHADED:
        ax.axvspan(T(a), T(b), color='white', alpha=0.6, lw=0, zorder=0.5)
    ax.set_xlim(ELECTION - pd.Timedelta(days=12), END + pd.Timedelta(days=80))
    ax.set_xticks(pd.date_range('2024-01-01', END, freq='6MS'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(length=0)
for ax in axes[:2]:
    ax.tick_params(labelbottom=False)


def series(ax, raw, tr, color):
    raw = raw[raw.date >= ELECTION]
    ax.plot(tr.index, tr.values, color=color, lw=3, zorder=3)
    ax.plot(tr.index[-1], tr.values[-1], 'o', color=color, ms=7, zorder=4)
    ax.text(tr.index[-1] + pd.Timedelta(days=12), tr.values[-1], f'{tr.values[-1]:.0f}%',
            color=color, fontweight='bold', fontsize=14, va='center')


# panel 1: National + events + shaded-period labels
ax = axes[0]
series(ax, nat, t_nat, BLUE)
ax.set_ylim(17, 47.5); ax.set_yticks(range(20, 45, 5))
ax.set_ylabel('National party vote (%)', fontweight='bold', fontsize=11, color=INK)
ax.plot(ELECTION, 38.1, 'o', color=BLUE, ms=8, zorder=5)
ax.annotate('Election result:\n38%', xy=(ELECTION, 38.1), xytext=(ELECTION + pd.Timedelta(days=25), 43.0),
            fontsize=11, color=INK, va='center', arrowprops=dict(arrowstyle='-', color=LEADER, lw=0.9))
for a, b, lab in SHADED:
    mid = T(a) + (T(b) - T(a)) / 2
    ax.text(mid, 47.0, lab, fontsize=10, color=MUTED, ha='center', va='top', linespacing=1.15)
for d, lab, side in EVENTS:
    y = at(t_nat, d)
    ax.plot([T(d), T(d)], [17, y - 0.9], color=INK, lw=0.9, zorder=2)
    ax.plot(T(d), y, 'D', ms=6, color=INK, zorder=5)
    ax.text(T(d) + pd.Timedelta(days=-8 if side == 'right' else 8), 17.8, lab, fontsize=10, color=INK,
            va='bottom', ha=side, linespacing=1.1)

# panel 2: Luxon + his lines
ax = axes[1]
series(ax, lux, t_lux, BLUE)
ax.set_ylim(10, 37); ax.set_yticks(range(10, 36, 5))
ax.set_ylabel('Luxon, preferred PM (%)', fontweight='bold', fontsize=11, color=INK)
for d, text, tx, ty, ha in LUXON:
    y = at(t_lux, d)
    ax.plot(T(d), y, 'o', ms=7, mfc='white', mec=INK, mew=1.6, zorder=5)
    ax.annotate(text, xy=(T(d), y), xytext=(T(tx), ty), ha=ha, va='center', fontsize=10.5,
                fontstyle='italic', color=INK, linespacing=1.15, zorder=6,
                arrowprops=dict(arrowstyle='-', color=LEADER, lw=0.9, shrinkA=2, shrinkB=4))

# panel 3: wrong direction
ax = axes[2]
series(ax, rw, t_wrong, INK)
ax.set_ylim(30, 66); ax.set_yticks(range(30, 66, 10))
ax.set_ylabel('Country heading in the\nwrong direction (%)', fontweight='bold', fontsize=11, color=INK)

# ---- titles, source
fig.text(L, 0.962, 'National has shed more than a quarter of its vote since the election', fontsize=25,
         fontweight='bold', color=INK)
fig.text(L, 0.94, "National's party vote, Christopher Luxon's preferred-PM rating and the share of voters saying the country "
                   "is heading in the wrong\ndirection, since the October 2023 election. Lines are trends adjusted for pollster "
                   "house effects;\nNational's line starts from its 38.1% election result. Shaded: recessions and petrol price spikes.",
         fontsize=11.5, color=MUTED, va='top', linespacing=1.4)
fig.text(L, 0.045, "Source: published polls compiled on Wikipedia (Roy Morgan, Taxpayers' Union–Curia, 1 News–Verian, RNZ–Reid "
                  "Research, Talbot Mills, Freshwater and others);\nStats NZ GDP (current vintage); MBIE weekly fuel prices | "
                  "Preferred PM excludes Freshwater (head-to-head question)",
         fontsize=9.5, color=MUTED, va='top', linespacing=1.5)

out = 'reports/national_luxon_mood_listener'
fig.savefig(out + '.svg'); fig.savefig(out + '.png', dpi=200)
print({k: round(float(v.values[-1]), 1) for k, v in [('nat', t_nat), ('lux', t_lux), ('wrong', t_wrong)]})
print('start of trends', {k: round(float(v.values[0]), 1) for k, v in [('nat', t_nat), ('lux', t_lux), ('wrong', t_wrong)]})
print('Luxon at quotes', {d: round(at(t_lux, d), 1) for d, *_ in LUXON})
print('vote lost since election: %.0f%%' % (100 * (1 - t_nat.values[-1] / 38.1)))
