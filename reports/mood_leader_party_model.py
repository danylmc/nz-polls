"""How do national mood (net right direction), the PM's personal rating and the
governing party's vote interact?  Monthly series, Nov 2020 - Oct 2026, built
within each government with month + pollster fixed effects.  Also adds petrol
(MBIE board price) as an outside driver for 2023-26."""
import json, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests

PERIODS = [  # (label, start, end, governing party, PM name pattern)
    ('Labour 2020-23', '2020-10-18', '2023-10-14', 'Labour', 'Ardern|Hipkins'),
    ('National 2023-26', '2023-10-15', '2026-10-31', 'National', 'Luxon'),
]
PM_SWITCH = pd.Period('2023-01', 'M')   # Ardern -> Hipkins
PETROL = sys.argv[1] if len(sys.argv) > 1 else 'raw_data/weekly-table.csv'


def norm(s):
    return s.fillna('none').str.replace(r'[^A-Za-z]', '', regex=True).str.lower().str[:8]


def monthly_fe(df):
    """Month effects from v ~ C(month) + C(pollster): a house-adjusted monthly series."""
    df = df.dropna(subset=['v']).copy()
    df['month'] = df.date.dt.to_period('M').astype(str); df['pk'] = norm(df.pollster)
    m = smf.ols('v ~ C(month) + C(pk) - 1', data=df).fit()
    s = m.params.filter(like='C(month)')
    s.index = pd.PeriodIndex([i.split('[')[1].rstrip(']') for i in s.index], freq='M')
    # re-level so the mean equals the raw mean (pollster effects average to zero)
    return s - s.mean() + df.v.mean()


# ---- load
polls = []
for cyc in (2023, 2026):
    for p in json.load(open(f'data/{cyc}_polling.json'))['polls']:
        polls.append(dict(date=p['date'], pollster=p['pollster'], **p['parties']))
polls = pd.DataFrame(polls); polls['date'] = pd.to_datetime(polls.date)
polls = polls[~polls.pollster.str.contains('election result', na=False)]
pm = pd.read_csv('data/pm_polling_all.csv'); pm['date'] = pd.to_datetime(pm.date)
old = pd.DataFrame(json.load(open('data/govt_approval_right_wrong_track.json')))
old = old.rename(columns={'right_direction': 'right', 'wrong_direction': 'wrong'})
new = pd.read_csv('data/right_wrong_direction_2026.csv')
rw = pd.concat([old[['date', 'pollster', 'right', 'wrong']], new[['date', 'pollster', 'right', 'wrong']]])
rw['date'] = pd.to_datetime(rw.date); rw['v'] = rw.right - rw.wrong

pet = pd.read_csv(PETROL)
pet = pet[(pet.Fuel == 'Regular Petrol') & (pet.Variable == 'Board price')]
pet['date'] = pd.to_datetime(pet.Date)
petrol = pet.set_index('date').Value.resample('MS').mean(); petrol.index = petrol.index.to_period('M')

frames = []
for label, a, b, party, pmname in PERIODS:
    inwin = lambda d: d[(d.date >= a) & (d.date <= b)]
    vote = monthly_fe(inwin(polls).rename(columns={party: 'v'}))
    lead = monthly_fe(inwin(pm[pm.candidate.str.contains(pmname)]).rename(columns={'percent': 'v'}))
    mood = monthly_fe(inwin(rw))
    f = pd.DataFrame({'vote': vote, 'pm': lead, 'mood': mood})
    f = f.reindex(pd.period_range(f.index.min(), f.index.max(), freq='M'))
    f['n_missing'] = f.isna().sum(axis=1)
    f = f.interpolate(limit=2, limit_area='inside')
    f['period'] = label
    frames.append(f)
M = pd.concat(frames)
M['petrol'] = petrol.reindex(M.index)
M.index.name = 'month'
M.to_csv('reports/mood_leader_party_monthly.csv')

print('Monthly series (house-adjusted). Rows with interpolated gaps flagged by n_missing>0')
print(M.round(1).to_string())

# ---- 1. levels and changes
print('\n=== Correlations: levels | month-to-month changes | 3-month changes')
for label, g in M.groupby('period'):
    g = g.dropna(subset=['vote', 'pm', 'mood'])
    d1 = g[['vote', 'pm', 'mood']].diff()
    if label.startswith('Labour'):
        d1.loc[d1.index == PM_SWITCH, 'pm'] = np.nan   # leader swap is not a change in a person's popularity
    d3 = g[['vote', 'pm', 'mood']].diff(3)
    print(f'\n{label}  (n months = {len(g)})')
    for name, x in [('levels', g[['vote', 'pm', 'mood']]), ('1-mo changes', d1), ('3-mo changes', d3)]:
        c = x.corr()
        print(f'  {name:13s} vote~pm {c.loc["vote","pm"]:+.2f}  vote~mood {c.loc["vote","mood"]:+.2f}  pm~mood {c.loc["pm","mood"]:+.2f}')

# ---- 2. which predicts the vote, given the other?
print('\n=== Change in vote on change in mood and change in PM rating (3-month changes, HAC SEs)')
for label, g in M.groupby('period'):
    g = g.dropna(subset=['vote', 'pm', 'mood'])
    d = g[['vote', 'pm', 'mood']].diff(3).dropna()
    if label.startswith('Labour'):
        d = d[~((d.index >= PM_SWITCH) & (d.index < PM_SWITCH + 3))]
    z = (d - d.mean()) / d.std()
    for f in ('vote ~ mood', 'vote ~ pm', 'vote ~ mood + pm'):
        r = smf.ols(f, data=z).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        coefs = '  '.join(f'{k}={r.params[k]:+.2f} (p={r.pvalues[k]:.3f})' for k in r.params.index if k != 'Intercept')
        print(f'  {label:17s} {f:18s} R2={r.rsquared:.2f}  {coefs}')

# ---- 3. lead/lag: Granger on monthly changes, pooled over both governments
print('\n=== Granger tests on month-to-month changes (both governments, within-period differences)')
D = []
for label, g in M.groupby('period'):
    d = g[['vote', 'pm', 'mood']].diff()
    if label.startswith('Labour'):
        d.loc[d.index == PM_SWITCH, 'pm'] = 0
    D.append(d.dropna())
D = pd.concat(D)
pairs = [('mood', 'vote'), ('vote', 'mood'), ('pm', 'vote'), ('vote', 'pm'), ('mood', 'pm'), ('pm', 'mood')]
for cause, effect in pairs:
    res = grangercausalitytests(D[[effect, cause]], maxlag=2, verbose=False)
    ps = ', '.join(f'lag{k} p={res[k][0]["ssr_ftest"][1]:.3f}' for k in res)
    print(f'  {cause:4s} -> {effect:4s}: {ps}')

# ---- 4. petrol as an outside driver, 2023-26
print('\n=== Petrol (monthly mean board price) vs the three series, National 2023-26')
g = M[M.period == 'National 2023-26'].dropna(subset=['petrol', 'vote', 'pm', 'mood'])
for k in ('vote', 'pm', 'mood'):
    lv = g[[k, 'petrol']].corr().iloc[0, 1]
    ch = g[[k, 'petrol']].diff(3).corr().iloc[0, 1]
    print(f'  petrol ~ {k:4s}: levels r={lv:+.2f}  3-mo changes r={ch:+.2f}')
g26 = g[g.index >= pd.Period('2025-10', 'M')]
print('\n  Since Oct 2025:')
print(g26[['petrol', 'mood', 'vote', 'pm']].round(1).to_string())
