#!/usr/bin/env python3
"""
Mortgage rates vs incumbent polling analysis.
Uses RBNZ hb20 2-year fixed rate and party vote polls.
"""
import json, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime

# ── 1. Load mortgage data ──────────────────────────────────────────
mort = pd.read_excel('raw_data/hb20.xlsx', header=None, skiprows=5)
mort = mort.iloc[:, [0, 6]].copy()  # date, 2yr fixed
mort.columns = ['date', 'rate_2yr']
mort['date'] = pd.to_datetime(mort['date'])
mort = mort.dropna(subset=['rate_2yr']).sort_values('date').reset_index(drop=True)
print(f"Mortgage data: {mort['date'].min():%Y-%m} to {mort['date'].max():%Y-%m}, {len(mort)} months")

# ── 2. Load all polling data ───────────────────────────────────────
rows = []
for f in sorted(glob.glob('data/*_polling.json')):
    with open(f) as fh:
        data = json.load(fh)
        for p in data.get('polls', []):
            row = {'date': p['date'], 'pollster': p.get('pollster')}
            row.update(p.get('parties', {}))
            rows.append(row)

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Identify incumbent party per period
# NZ governments: Labour 1999-2008, National 2008-2017, Labour 2017-2023, National 2023-
def get_incumbent(d):
    if d < datetime(2008, 11, 8): return 'Labour'
    if d < datetime(2017, 10, 26): return 'National'
    if d < datetime(2023, 11, 27): return 'Labour'
    return 'National'

df['incumbent'] = df['date'].apply(get_incumbent)
df['inc_vote'] = df.apply(lambda r: r.get(get_incumbent(r['date'])), axis=1)
df['inc_vote'] = pd.to_numeric(df['inc_vote'], errors='coerce')
df = df.dropna(subset=['inc_vote'])

# Keep only polls within mortgage data range
df = df[df['date'] >= mort['date'].min()].copy()
print(f"Polls in mortgage range: {len(df)} ({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m})")

# ── 3. Monthly aggregation ─────────────────────────────────────────
df['month'] = df['date'].dt.to_period('M')
monthly_polls = df.groupby('month')['inc_vote'].mean().reset_index()
monthly_polls['date'] = monthly_polls['month'].dt.to_timestamp()

# Merge with mortgage rates (nearest month)
mort['month'] = mort['date'].dt.to_period('M')
merged = monthly_polls.merge(mort[['month', 'rate_2yr']], on='month', how='inner')
print(f"Matched months: {len(merged)}")

# ── 4. Level correlation ───────────────────────────────────────────
r_level, p_level = stats.pearsonr(merged['rate_2yr'], merged['inc_vote'])
print(f"\n=== LEVEL CORRELATION ===")
print(f"  r = {r_level:.3f}, p = {p_level:.4f}, n = {len(merged)}")

# ── 5. Change correlation (Δrate vs Δvote) ─────────────────────────
merged = merged.sort_values('date')
merged['d_rate'] = merged['rate_2yr'].diff()
merged['d_vote'] = merged['inc_vote'].diff()
changes = merged.dropna(subset=['d_rate', 'd_vote'])

r_change, p_change = stats.pearsonr(changes['d_rate'], changes['d_vote'])
print(f"\n=== CHANGE CORRELATION (month-on-month) ===")
print(f"  r = {r_change:.3f}, p = {p_change:.4f}, n = {len(changes)}")

# ── 6. Lagged correlations (does rate change predict future vote?) ──
print(f"\n=== LAGGED CORRELATIONS (Δrate_t → Δvote_t+lag) ===")
print(f"  {'Lag':>4s}  {'r':>7s}  {'p':>8s}")
best_lag, best_r, best_p = 0, 0, 1
for lag in range(0, 13):
    lagged = pd.DataFrame({
        'd_rate': changes['d_rate'].values[:-lag] if lag > 0 else changes['d_rate'].values,
        'd_vote': changes['d_vote'].values[lag:] if lag > 0 else changes['d_vote'].values
    }).dropna()
    if len(lagged) < 10:
        continue
    r, p = stats.pearsonr(lagged['d_rate'], lagged['d_vote'])
    print(f"  {lag:>4d}  {r:>7.3f}  {p:>8.4f}")
    if abs(r) > abs(best_r):
        best_lag, best_r, best_p = lag, r, p

print(f"\n  Best lag: {best_lag} months (r={best_r:.3f}, p={best_p:.4f})")

# ── 7. Quarterly change (smoother signal) ──────────────────────────
merged['d_rate_3m'] = merged['rate_2yr'].diff(3)
merged['d_vote_3m'] = merged['inc_vote'].diff(3)
q_changes = merged.dropna(subset=['d_rate_3m', 'd_vote_3m'])

r_q, p_q = stats.pearsonr(q_changes['d_rate_3m'], q_changes['d_vote_3m'])
print(f"\n=== QUARTERLY CHANGE CORRELATION (3-month Δ) ===")
print(f"  r = {r_q:.3f}, p = {p_q:.4f}, n = {len(q_changes)}")

# ── 8. By government period ────────────────────────────────────────
print(f"\n=== BY GOVERNMENT PERIOD ===")
for inc in ['Labour', 'National']:
    sub = changes[changes['date'].apply(get_incumbent) == inc]
    if len(sub) > 10:
        r, p = stats.pearsonr(sub['d_rate'], sub['d_vote'])
        print(f"  {inc}: r = {r:.3f}, p = {p:.4f}, n = {len(sub)}")

# ── 9. Granger-style: does rate change predict vote change? ────────
from statsmodels.tsa.stattools import grangercausalitytests
print(f"\n=== GRANGER CAUSALITY (rate → vote, max 6 lags) ===")
gc_data = merged[['d_vote', 'd_rate']].dropna()
try:
    gc = grangercausalitytests(gc_data, maxlag=6, verbose=False)
    for lag, result in gc.items():
        f_stat = result[0]['ssr_ftest'][0]
        p_val = result[0]['ssr_ftest'][1]
        print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# ── 10. Visualization ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Time series (dual axis)
ax1 = axes[0]
ax1r = ax1.twinx()
ax1.plot(merged['date'], merged['inc_vote'], color='steelblue', alpha=0.8, label='Incumbent vote %')
ax1r.plot(merged['date'], merged['rate_2yr'], color='firebrick', alpha=0.8, label='2yr fixed rate %')
ax1r.invert_yaxis()  # Lower rates = good, so invert to see if they track
ax1.set_ylabel('Incumbent vote %', color='steelblue')
ax1r.set_ylabel('2yr fixed rate % (inverted)', color='firebrick')
ax1.set_title('Mortgage Rate vs Incumbent Party Polling')
# Shade government periods
for start, end, party in [
    ('2005-01', '2008-11', 'Labour'), ('2008-11', '2017-10', 'National'),
    ('2017-10', '2023-11', 'Labour'), ('2023-11', '2026-03', 'National')]:
    color = '#d32f2f' if party == 'Labour' else '#1565c0'
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.05, color=color)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# Panel 2: Scatter (quarterly changes)
ax2 = axes[1]
colors = q_changes['date'].apply(get_incumbent).map({'Labour': '#d32f2f', 'National': '#1565c0'})
ax2.scatter(q_changes['d_rate_3m'], q_changes['d_vote_3m'], c=colors, alpha=0.4, s=20)
# Regression line
m, b = np.polyfit(q_changes['d_rate_3m'], q_changes['d_vote_3m'], 1)
x_range = np.linspace(q_changes['d_rate_3m'].min(), q_changes['d_rate_3m'].max(), 50)
ax2.plot(x_range, m * x_range + b, 'k--', alpha=0.7)
ax2.axhline(0, color='gray', lw=0.5); ax2.axvline(0, color='gray', lw=0.5)
ax2.set_xlabel('3-month change in 2yr fixed rate (pp)')
ax2.set_ylabel('3-month change in incumbent vote (pp)')
ax2.set_title(f'Quarterly Rate Change vs Incumbent Vote Change (r={r_q:.3f}, p={p_q:.4f})')
ax2.legend(handles=[
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#d32f2f',label='Labour govt'),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#1565c0',label='National govt')
], fontsize=9)

# Panel 3: Lagged correlation bar chart
ax3 = axes[2]
lag_rs = []
for lag in range(0, 13):
    lagged = pd.DataFrame({
        'd_rate': changes['d_rate'].values[:-lag] if lag > 0 else changes['d_rate'].values,
        'd_vote': changes['d_vote'].values[lag:] if lag > 0 else changes['d_vote'].values
    }).dropna()
    if len(lagged) >= 10:
        r, p = stats.pearsonr(lagged['d_rate'], lagged['d_vote'])
        lag_rs.append((lag, r, p))
lags_df = pd.DataFrame(lag_rs, columns=['lag', 'r', 'p'])
bar_colors = ['darkgreen' if p < 0.05 else 'gray' for p in lags_df['p']]
ax3.bar(lags_df['lag'], lags_df['r'], color=bar_colors, alpha=0.7)
ax3.axhline(0, color='black', lw=0.5)
ax3.set_xlabel('Lag (months)')
ax3.set_ylabel('Correlation (r)')
ax3.set_title('Lagged Correlation: Δ Mortgage Rate → Δ Incumbent Vote')
ax3.legend(handles=[
    plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='darkgreen',label='p < 0.05'),
    plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',label='p ≥ 0.05')
], fontsize=9)

plt.tight_layout()
plt.savefig('reports/mortgage_vs_incumbent.png', dpi=150)
print(f"\nSaved: reports/mortgage_vs_incumbent.png")
