"""National/Luxon 2023-26 against other governments' terms, plus within-term declines
in National's vote, Luxon's preferred-PM rating and right/wrong direction."""
import json, re, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from bs4 import BeautifulSoup
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, '.')
from events import ELECTION_DATES, INCUMBENTS

FIRST_TERMS = {1993: ('National', 'Bolger', '1990-10-27'), 2002: ('Labour', 'Clark', '1999-11-27'),
               2011: ('National', 'Key', '2008-11-08'), 2020: ('Labour', 'Ardern', '2017-09-23'),
               2026: ('National', 'Luxon', '2023-10-14')}
PREV_ELECTION = {y: ELECTION_DATES.get(y - 3) for y in INCUMBENTS}
PREV_ELECTION.update({1993: '1990-10-27', 2002: '1999-11-27', 2005: '2002-07-27',
                      2008: '2005-09-17', 2011: '2008-11-08', 2014: '2011-11-26', 2017: '2014-09-20',
                      2020: '2017-09-23', 2023: '2020-10-17', 2026: '2023-10-14', 1996: '1993-11-06',
                      1999: '1996-10-12'})
MONTHS = [6, 12, 18, 24, 30, 36]
D = 30.44


def norm(s):
    return s.fillna('none').str.replace(r'[^A-Za-z]', '', regex=True).str.lower().str[:8]


def party_trend(cycle, party, frac=0.2):
    polls = json.load(open(f'data/{cycle}_polling.json'))['polls']
    df = pd.DataFrame([dict(date=p['date'], v=p['parties'].get(party)) for p in polls])
    df['date'] = pd.to_datetime(df.date); df = df.dropna().sort_values('date')
    t0 = pd.Timestamp(PREV_ELECTION[cycle])
    df['m'] = (df.date - t0).dt.days / D
    df = df[df.m >= 0]
    fit = lowess(df.v.values, df.m.values, frac=frac)
    return df, lambda m: float(np.interp(m, fit[:, 0], fit[:, 1])) if m <= df.m.max() else np.nan


def section(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


# ---------------------------------------------------------------- A. party vote by term
section('A. Governing party vote: change since election (LOWESS trend, raw polls, all pollsters)')
rows = []
for cycle, party in INCUMBENTS.items():
    if cycle in (1996, 1999, 2017, 2023):   # terms where the PM changed mid-cycle are still terms; keep all
        pass
    df, f = party_trend(cycle, party)
    start = df.v.iloc[0]
    r = dict(cycle=cycle, party=party, start=round(start, 1), first=cycle in FIRST_TERMS)
    for m in MONTHS:
        r[f'm{m}'] = round(f(m) - start, 1)
    r['min'] = round(min(f(m) for m in np.arange(0, df.m.max(), 0.5)) - start, 1)
    if cycle != 2026:
        nxt = json.load(open(f'data/{cycle}_polling.json'))['polls']
        res = [p for p in nxt if p['date'] == ELECTION_DATES[cycle]]
        r['result'] = round(res[-1]['parties'][party] - start, 1) if res else np.nan
    rows.append(r)
tab = pd.DataFrame(rows).set_index('cycle')
print(tab.to_string())
print('\nRank of change at month 30 (most negative first):')
print(tab.sort_values('m30')[['party', 'first', 'm30', 'result']].to_string())

section('A2. Governing party LEVEL at checkpoints (first terms only)')
for cycle, (party, pm, _) in FIRST_TERMS.items():
    df, f = party_trend(cycle, party)
    print(f'{pm:7s} {party:9s} election {df.v.iloc[0]:5.1f} | ' +
          ' '.join(f'm{m}:{f(m):5.1f}' for m in MONTHS) + f' | n={len(df)}')

# ---------------------------------------------------------------- B. preferred PM, 1 News series
section('B. Preferred PM, 1 News (Colmar Brunton/Kantar/Verian) only: first-term PMs')
pm = pd.read_csv('data/pm_polling_all.csv'); pm['date'] = pd.to_datetime(pm.date)
onenews = pm[pm.pollster.str.contains('olmar|erian|antar', na=False)]
for cycle, (party, name, edate) in FIRST_TERMS.items():
    if name == 'Bolger':
        continue
    e = pd.Timestamp(edate)
    x = onenews[onenews.candidate.str.contains(name)].copy()
    pre = x[(x.date < e) & (x.date > e - pd.Timedelta(days=120))]
    x = x[(x.date >= e) & (x.date < e + pd.Timedelta(days=37 * D))]
    x['m'] = (x.date - e).dt.days / D
    bins = x.groupby((x.m // 6 * 6).astype(int)).percent.agg(['mean', 'count']).round(1)
    print(f'\n{name}: last pre-election 1 News rating {pre.percent.iloc[-1] if len(pre) else "n/a"}'
          f' | peak {x.percent.max()} | low {x.percent.min()} | latest-in-term {x.percent.iloc[-1] if len(x) else "n/a"}')
    print('  6-month bins (months since election): ' +
          ', '.join(f'{k}-{k+6}: {r["mean"]} (n={int(r["count"])})' for k, r in bins.iterrows()))

# gap to best opposition figure in the same 1 News poll
section('B2. 1 News: PM minus top-rated other person in same poll, by 6-month bin')
for cycle, (party, name, edate) in FIRST_TERMS.items():
    if name == 'Bolger':
        continue
    e = pd.Timestamp(edate)
    x = onenews[(onenews.date >= e) & (onenews.date < e + pd.Timedelta(days=37 * D))]
    g = []
    for (d, p), grp in x.groupby(['date', 'pollster']):
        me = grp[grp.candidate.str.contains(name)].percent
        oth = grp[~grp.candidate.str.contains(name)].percent
        if len(me) and len(oth):
            g.append(dict(m=(d - e).days / D, gap=me.iloc[0] - oth.max()))
    g = pd.DataFrame(g)
    print(f'{name:7s}: ' + ', '.join(f'{int(k)}-{int(k)+6}: {v:+.0f}' for k, v in
                                     g.groupby((g.m // 6 * 6)).gap.mean().items()))

# ---------------------------------------------------------------- C. right / wrong direction
section('C. Right/wrong direction: parse 2026 Wikipedia table')
MON = {m: i for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 1)}


def end_date(s):
    s = re.sub(r'\[.*?\]', '', s).strip()
    m = re.search(r'(\d{1,2})?\s*([A-Za-z]{3})[a-z]*\s+(\d{4})\s*$', s)
    if not m:
        return None
    day = int(m.group(1)) if m.group(1) else 15
    return pd.Timestamp(int(m.group(3)), MON[m.group(2).lower()], day)


soup = BeautifulSoup(open('raw_data/wikipedia_polling/2026.html').read(), 'lxml')
rw = []
for t in soup.find_all('table'):
    if 'Right direction' not in t.get_text():
        continue
    for tr in t.find_all('tr')[1:]:
        c = [td.get_text(' ', strip=True) for td in tr.find_all(['td', 'th'])]
        if len(c) < 5:
            continue
        try:
            rw.append(dict(date=end_date(c[0]), pollster=c[1], right=float(c[3]), wrong=float(c[4])))
        except ValueError:
            continue
rw = pd.DataFrame(rw).dropna(subset=['date']).assign(source='Wikipedia 2026 opinion polling page')
# polls missing from the Wikipedia table, added from news reports
rw = pd.concat([rw, pd.DataFrame([dict(date=pd.Timestamp('2026-06-11'), pollster='The Post/Freshwater Strategy',
                                       right=27.0, wrong=57.0,
                                       source='Waikato Times/Post 20 Jun 2026 (missing from Wikipedia table)')])])
rw['net'] = rw.right - rw.wrong
print(f'{len(rw)} polls, {rw.date.min().date()} to {rw.date.max().date()}')
print(rw.pollster.value_counts().to_string())
rw.sort_values('date')[['date', 'pollster', 'right', 'wrong', 'net', 'source']].to_csv('data/right_wrong_direction_2026.csv', index=False)

# Roy Morgan is the only pollster asking it across 2017-26: compare terms at the same month
section('C2. Roy Morgan net right direction by months since election (term comparison)')
old = pd.DataFrame(json.load(open('data/govt_approval_right_wrong_track.json')))
old['date'] = pd.to_datetime(old.date)
old = old[old.pollster.str.contains('Roy Morgan')].rename(columns={'right_direction': 'right', 'wrong_direction': 'wrong'})
old = old[old.date > '2017-01-01']
rm_now = rw[rw.pollster.str.contains('Roy Morgan')]
for label, e, x in [('Ardern 2017-20', '2017-09-23', old[old.date < '2020-10-17']),
                    ('Ardern/Hipkins 2020-23', '2020-10-17', old[old.date >= '2020-10-17']),
                    ('Luxon 2023-26', '2023-10-14', rm_now)]:
    x = x.copy(); x['m'] = (x.date - pd.Timestamp(e)).dt.days / D; x = x[x.m >= 0]
    b = x.groupby((x.m // 6 * 6).astype(int)).net.mean().round(0)
    print(f'{label:24s} ' + ', '.join(f'{k}-{k+6}: {v:+.0f}' for k, v in b.items()))
print('Roy Morgan just before the 2023 election:',
      old[(old.date > '2023-08-01') & (old.date < '2023-10-14')][['date', 'right', 'wrong', 'net']].to_string(index=False))


# ---------------------------------------------------------------- D. within-term declines
section('D. Luxon term: declines in National vote, Luxon PM rating, net right direction')


def adjusted(df, frac):
    """House-effect-adjusted LOWESS daily trend (same method as coalition_declines.py)."""
    df = df.dropna(subset=['date', 'v']).sort_values('date').copy(); df['pk'] = norm(df.pollster)
    t = (df.date - df.date.min()).dt.days.values.astype(float); adj = df.v.values.copy()
    for _ in range(5):
        fit = lowess(adj, t, frac=frac, return_sorted=False)
        he = pd.Series(df.v.values - fit, index=df.pk.values).groupby(level=0).mean(); he -= he.mean()
        adj = df.v.values - df.pk.map(he).values
    days = pd.date_range(df.date.min(), df.date.max())
    s = pd.Series(np.interp((days - df.date.min()).days.values.astype(float), np.sort(t),
                            lowess(adj, t, frac=frac)[:, 1]), index=days)
    return s, df


def episodes(s, w, k=5, floor=-0.5):
    ch = (s - s.shift(w)).dropna(); out = []
    while len(out) < k and len(ch):
        e = ch.idxmin()
        if ch[e] > floor:
            break
        out.append((e - pd.Timedelta(days=w), e, ch[e]))
        ch = ch[(ch.index < e - pd.Timedelta(days=w)) | (ch.index > e + pd.Timedelta(days=w))]
    return out


def window_test(df, a, b, pad=45):
    """Before/after test on raw polls: pollster fixed effects + post dummy.
    Pre = polls in [a-pad, a+15d], post = polls in [b-15d, b+pad]."""
    x = df.copy()
    pre = x[(x.date >= a - pd.Timedelta(days=pad)) & (x.date <= a + pd.Timedelta(days=15))].assign(post=0)
    post = x[(x.date >= b - pd.Timedelta(days=15)) & (x.date <= b + pd.Timedelta(days=pad))].assign(post=1)
    x = pd.concat([pre, post])
    if len(pre) < 2 or len(post) < 2:
        return np.nan, np.nan, len(x)
    f = 'v ~ post + C(pk)' if x.pk.nunique() > 1 and x.pk.nunique() < len(x) - 2 else 'v ~ post'
    m = smf.ols(f, data=x).fit(cov_type='HC1')
    return m.params['post'], m.pvalues['post'], len(x)


nat = pd.DataFrame([dict(date=p['date'], pollster=p['pollster'], v=p['parties'].get('National'))
                    for p in json.load(open('data/2026_polling.json'))['polls']])
nat['date'] = pd.to_datetime(nat.date); nat = nat[nat.date > '2023-10-14']
lux = pm[pm.candidate.str.contains('Luxon') & (pm.date > '2023-10-14')].rename(columns={'percent': 'v'})
rwd = rw.rename(columns={'net': 'v'})
series = {'National vote': adjusted(nat, 0.15), 'Luxon pref PM': adjusted(lux, 0.25),
          'Net right direction': adjusted(rwd, 0.25)}
trend = pd.DataFrame({k: v[0] for k, v in series.items()})
print('\nQuarter-end trend (house-adjusted):')
print(trend.resample('QE').last().round(1).to_string())

for name, (s, df) in series.items():
    for w in (90,):
        print(f'\n{name}: steepest non-overlapping {w}-day declines (trend), with before/after test on raw polls')
        allch = (s - s.shift(w)).dropna()
        for a, b, c in episodes(s, w):
            sl, p, n = window_test(df, a, b)
            pct = (allch <= c).mean() * 100
            print(f'  {a.date()} -> {b.date()}: {c:+5.1f} ({s[a]:.1f} -> {s[b]:.1f}); '
                  f'before/after (pollster FE) {sl:+.1f} p={p:.3f} n={n}; bottom {pct:.0f}% of 90-day changes')

# co-movement of the three trends, 90-day changes
section('E. Do they fall together? Correlation of 90-day changes in trend (sampled every 30 days)')
ch = (trend - trend.shift(90)).iloc[::30].dropna()
print(ch.corr().round(2).to_string())
print('\n90-day changes, every 60 days:')
print((trend - trend.shift(90)).iloc[90::60].round(1).to_string())
