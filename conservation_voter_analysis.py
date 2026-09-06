"""
Conservation voter analysis: testing whether there's a meaningful cohort of
environmentally-concerned voters who are nonetheless available to National.

Theory: Beyond ideological Green/Labour environmentalists, there's a segment of
recreational conservationists (trampers, hunters, fishers etc.) who care about
conservation but are not locked in to the left. They're available to National
depending on specific conservation issues.

Test approach:
  1. Size the non-Green pro-environment segment over time
  2. Profile "would-never-vote-Green" conservationists vs Green loyalists
  3. Natural experiments: Ban1080 (2014/2017) and Outdoor Recreation Party (2020)
  4. Environmental group members who vote National (1996-2005)
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

conn = sqlite3.connect('data/nzesdb.sqlite')

# ---------------------------------------------------------------------------
# Year configs: env_var (1=protect env, 7=develop economy), party_vote_var,
# party coding for National/Labour/Green/NZFirst, and optional extras
# ---------------------------------------------------------------------------
YEARS = [
    {
        'year': 1993, 'table': 'nzes_1993',
        'env_var': 'ENVRSP',  # 1=env, 7=develop
        'vote_var': 'VOT93E',
        'nat': 2, 'lab': 1, 'grn': None, 'nzf': 4,
    },
    {
        'year': 1996, 'table': 'nzes_1996',
        'env_var': 'PENVR',
        'vote_var': 'VOT96P',
        'nat': 2, 'lab': 1, 'grn': None, 'nzf': 3,
        'memenv_var': 'MEMENV',  # 1=yes member
    },
    {
        'year': 1999, 'table': 'nzes_1999_post',
        'env_var': 'SENVR',
        'vote_var': 'CVOT99P',
        'nat': 3, 'lab': 2, 'grn': 7, 'nzf': 4,
        'memenv_var': 'MEMINT',  # member interest/env group
    },
    {
        'year': 2002, 'table': 'nzes_2002',
        'env_var': 'WENPRT',
        'vote_var': 'WVOT02P',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4,
        'memenv_var': 'WMEMENV',
    },
    {
        'year': 2005, 'table': 'nzes_2005',
        'env_var': 'yenprot',
        'vote_var': 'yvot05p',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4,
        'memenv_var': 'ymemenv',
    },
    {
        'year': 2008, 'table': 'nzes_2008',
        'env_var': 'zenvprt',
        'vote_var': 'zvt08p',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4,
        'never_grn_var': 'zpnvgrn',
    },
    {
        'year': 2014, 'table': 'nzes_2014',
        'env_var': 'decovsenv',
        'vote_var': 'dpartyvote',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4, 'ban': 17,
        'never_grn_var': 'dnevervotegrn',
        'never_nat_var': 'dnevervotenat',
        'prior_vote_var': 'dlastpvote',
        'prior_nat': 2, 'prior_lab': 1, 'prior_grn': 3,
    },
    {
        'year': 2017, 'table': 'nzes_2017',
        'env_var': 'renvvseco',
        'vote_var': 'rpartyvote',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4, 'ban': 12,
        'never_grn_var': 'rnevergreen',
        'never_nat_var': 'rnevernational',
        'prior_vote_var': 'rlastpvote',
        'prior_nat': 2, 'prior_lab': 1, 'prior_grn': 3,
    },
    {
        'year': 2020, 'table': 'nzes_2020',
        'env_var': 'C10',
        'vote_var': 'E3',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 4, 'outdoor': 13,
        'never_grn_var': 'E15_3',
        'never_nat_var': 'E15_2',
        'prior_vote_var': 'E19',
        'prior_nat': 2, 'prior_lab': 1, 'prior_grn': 3,
        'urban_var': 'lurban',  # 1=rural, 4=large city
    },
    {
        'year': 2023, 'table': 'nzes_2023',
        'env_var': 'C7',
        'vote_var': 'mpartyvote',
        'nat': 2, 'lab': 1, 'grn': 3, 'nzf': 5,
        'never_grn_var': 'E14d',
        'never_nat_var': 'E14b',
    },
]

# Env score: 1-3 = pro-environment (we treat ≤3 as "green lean")
ENV_THRESHOLD = 3


def load_year(cfg):
    cols = [cfg['env_var'], cfg['vote_var']]
    for k in ('memenv_var', 'never_grn_var', 'never_nat_var',
              'prior_vote_var', 'urban_var'):
        if k in cfg:
            cols.append(cfg[k])
    col_str = ', '.join(f'"{c}"' for c in cols)
    df = pd.read_sql(f'SELECT {col_str} FROM {cfg["table"]}', conn)
    df.columns = cols
    df['env'] = pd.to_numeric(df[cfg['env_var']], errors='coerce')
    df['vote'] = pd.to_numeric(df[cfg['vote_var']], errors='coerce')
    # drop DK/missing on env (coded 8, 9, 0 etc.)
    df = df[df['env'].between(1, 7)]
    df = df[df['vote'].notna() & (df['vote'] > 0)]
    df['pro_env'] = df['env'] <= ENV_THRESHOLD
    df['year'] = cfg['year']
    return df, cfg


# ---------------------------------------------------------------------------
# Section 1: Trend — share of pro-env voters going to each major party
# ---------------------------------------------------------------------------
print("=" * 70)
print("SECTION 1: PRO-ENVIRONMENT VOTERS — PARTY VOTE BREAKDOWN OVER TIME")
print("=" * 70)
print(f"(Pro-environment = env/economy position ≤ {ENV_THRESHOLD} on 1-7 scale)\n")

trend_rows = []
for cfg in YEARS:
    df, cfg = load_year(cfg)
    pro = df[df['pro_env']].copy()
    n = len(pro)
    if n < 30:
        continue

    nat_pct = (pro['vote'] == cfg['nat']).mean() * 100
    lab_pct = (pro['vote'] == cfg['lab']).mean() * 100
    grn_pct = (pro['vote'] == cfg['grn']).mean() * 100 if cfg['grn'] else np.nan
    nzf_pct = (pro['vote'] == cfg['nzf']).mean() * 100

    row = dict(year=cfg['year'], n=n, National=nat_pct, Labour=lab_pct,
               Green=grn_pct, NZFirst=nzf_pct)
    trend_rows.append(row)

    print(f"{cfg['year']}  n={n:4d}  Nat={nat_pct:5.1f}%  Lab={lab_pct:5.1f}%  "
          f"Grn={grn_pct:5.1f}%  NZF={nzf_pct:5.1f}%")

trend_df = pd.DataFrame(trend_rows)

# ---------------------------------------------------------------------------
# Section 2: "Available conservationist" segment — not locked into Green
# and not locked out of National
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2: 'AVAILABLE CONSERVATIONIST' PROFILE (2008-2023)")
print("Pro-env voters who would NEVER vote Green — where do they go?")
print("=" * 70)

avail_rows = []
for cfg in YEARS:
    if 'never_grn_var' not in cfg:
        continue
    df, cfg = load_year(cfg)

    # Load never-vote-Green
    ngv = pd.to_numeric(df[cfg['never_grn_var']], errors='coerce')
    df['never_grn'] = ngv == 1

    # Also never-vote-National if available
    if 'never_nat_var' in cfg:
        nnv = pd.to_numeric(df[cfg['never_nat_var']], errors='coerce')
        df['never_nat'] = nnv == 1
    else:
        df['never_nat'] = False

    pro = df[df['pro_env']].copy()
    n_pro = len(pro)

    # Group A: ideological environmentalists — pro-env + never-vote-National
    groupA = pro[pro['never_nat']] if 'never_nat_var' in cfg else pro[pro['vote'] == cfg.get('grn', -1)]
    # Group B: available conservationists — pro-env + never-vote-Green
    groupB = pro[pro['never_grn']]
    # Group C: uncommitted pro-env (neither lock)
    groupC = pro[~pro['never_grn'] & ~pro['never_nat']] if 'never_nat_var' in cfg else pro[~pro['never_grn']]

    pctB = len(groupB) / n_pro * 100 if n_pro else 0
    pctA = len(groupA) / n_pro * 100 if n_pro else 0

    # Among Group B: where do they actually vote?
    if len(groupB) >= 10:
        bvotes = groupB['vote'].value_counts(normalize=True) * 100
        b_nat = bvotes.get(cfg['nat'], 0)
        b_lab = bvotes.get(cfg['lab'], 0)
        b_nzf = bvotes.get(cfg['nzf'], 0)
    else:
        b_nat = b_lab = b_nzf = np.nan

    print(f"\n{cfg['year']}  (pro-env n={n_pro})")
    print(f"  Group A (never-Nat):  {pctA:5.1f}% of pro-env voters")
    print(f"  Group B (never-Grn):  {pctB:5.1f}% of pro-env voters")
    if len(groupB) >= 10:
        print(f"    Group B votes: Nat={b_nat:.1f}%  Lab={b_lab:.1f}%  NZF={b_nzf:.1f}%  (n={len(groupB)})")

    avail_rows.append(dict(year=cfg['year'], n_pro=n_pro,
                           pct_avail=pctB, pct_ideolog=pctA,
                           avail_nat=b_nat, avail_lab=b_lab))

# ---------------------------------------------------------------------------
# Section 3: Natural experiments — Ban1080 (2014, 2017) and Outdoor (2020)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 3: NATURAL EXPERIMENTS")
print("=" * 70)


def describe_group(df, mask, vote_var, cfg, label, prior_var=None):
    sub = df[mask].copy()
    n = len(sub)
    if n < 5:
        print(f"  {label}: n={n} (too small)")
        return
    env_mean = sub['env'].mean()
    env_sd = sub['env'].std()
    votes = sub['vote'].value_counts(normalize=True) * 100
    print(f"  {label} (n={n})")
    print(f"    Env position mean={env_mean:.2f} sd={env_sd:.2f}  (1=green, 7=develop)")
    top3 = votes.nlargest(4)
    vote_labels = {cfg['nat']: 'Nat', cfg['lab']: 'Lab', cfg.get('grn'): 'Grn',
                   cfg['nzf']: 'NZF', cfg.get('ban'): 'Ban1080',
                   cfg.get('outdoor'): 'Outdoor'}
    vote_str = '  '.join(f"{vote_labels.get(k, 'code' + str(int(k)))}={v:.1f}%"
                         for k, v in top3.items())
    print(f"    Current vote: {vote_str}")
    if prior_var and prior_var in df.columns:
        prior = pd.to_numeric(sub[prior_var], errors='coerce')
        pv = prior.value_counts(normalize=True).head(5) * 100
        pv_str = '  '.join(f"code{int(k)}={v:.1f}%" for k, v in pv.items())
        # map known codes
        prior_nat = cfg.get('prior_nat', cfg['nat'])
        prior_lab = cfg.get('prior_lab', cfg['lab'])
        prior_grn = cfg.get('prior_grn', cfg.get('grn'))
        pnat = (prior == prior_nat).mean() * 100
        plab = (prior == prior_lab).mean() * 100
        pgrn = (prior == prior_grn).mean() * 100 if prior_grn else np.nan
        print(f"    Prior vote: Nat={pnat:.1f}%  Lab={plab:.1f}%  Grn={pgrn:.1f}%")


# 2014: Ban1080
print("\n--- 2014: Ban1080 Party ---")
cfg14 = next(c for c in YEARS if c['year'] == 2014)
df14, cfg14 = load_year(cfg14)
prior_var14 = cfg14.get('prior_vote_var')
if prior_var14:
    df14[prior_var14] = pd.to_numeric(
        pd.read_sql(f'SELECT "{prior_var14}" FROM nzes_2014', conn)[prior_var14],
        errors='coerce')

mask_ban14 = df14['vote'] == cfg14['ban']
mask_grn14 = df14['vote'] == cfg14['grn']
mask_nat14 = df14['vote'] == cfg14['nat']

describe_group(df14, mask_ban14, 'vote', cfg14, "Ban1080 voters", prior_var14)
describe_group(df14, mask_grn14, 'vote', cfg14, "Green voters (comparison)", prior_var14)
describe_group(df14, mask_nat14, 'vote', cfg14, "National voters (comparison)", prior_var14)

# t-test: env position Ban1080 vs Green
ban_env = df14.loc[mask_ban14, 'env'].dropna()
grn_env = df14.loc[mask_grn14, 'env'].dropna()
nat_env = df14.loc[mask_nat14, 'env'].dropna()
if len(ban_env) >= 5 and len(grn_env) >= 5:
    t, p = stats.ttest_ind(ban_env, grn_env)
    print(f"\n  Ban1080 vs Green env position: t={t:.2f}, p={p:.3f}")
if len(ban_env) >= 5 and len(nat_env) >= 5:
    t, p = stats.ttest_ind(ban_env, nat_env)
    print(f"  Ban1080 vs National env position: t={t:.2f}, p={p:.3f}")

# 2017: Ban1080 again — where did they come from (2014)?
print("\n--- 2017: Ban1080 voters ---")
cfg17 = next(c for c in YEARS if c['year'] == 2017)
df17, cfg17 = load_year(cfg17)
prior_var17 = cfg17.get('prior_vote_var')
if prior_var17:
    df17[prior_var17] = pd.to_numeric(
        pd.read_sql(f'SELECT "{prior_var17}" FROM nzes_2017', conn)[prior_var17],
        errors='coerce')

mask_ban17 = df17['vote'] == cfg17['ban']
describe_group(df17, mask_ban17, 'vote', cfg17, "Ban1080 voters 2017", prior_var17)

# Where did 2014 Ban1080 voters go in 2017?
print("\n  2014 Ban1080 voters' 2017 destination (from prior vote field in 2017 data):")
# In 2017 data, rlastpvote=2014 party vote. Ban1080 code in 2014 is 17.
# But in 2017's prior_vote_var, what's the code for Ban1080?
cur = conn.cursor()
cur.execute("SELECT value, label FROM _value_labels WHERE table_name='nzes_2017' AND column_name='rlastpvote' ORDER BY value")
pvlabels = cur.fetchall()
print("  Prior vote value labels:", pvlabels[:15])

# 2020: Outdoor Recreation Party
print("\n--- 2020: Outdoor Recreation Party ---")
cfg20 = next(c for c in YEARS if c['year'] == 2020)
df20, cfg20 = load_year(cfg20)
prior_var20 = cfg20.get('prior_vote_var')
if prior_var20:
    df20[prior_var20] = pd.to_numeric(
        pd.read_sql(f'SELECT "{prior_var20}" FROM nzes_2020', conn)[prior_var20],
        errors='coerce')
# Also load urban var
urban_var20 = cfg20.get('urban_var')
if urban_var20:
    df20[urban_var20] = pd.to_numeric(
        pd.read_sql(f'SELECT "{urban_var20}" FROM nzes_2020', conn)[urban_var20],
        errors='coerce')

mask_out20 = df20['vote'] == cfg20['outdoor']
mask_grn20 = df20['vote'] == cfg20['grn']
mask_nat20 = df20['vote'] == cfg20['nat']

describe_group(df20, mask_out20, 'vote', cfg20, "Outdoor Party voters", prior_var20)
describe_group(df20, mask_grn20, 'vote', cfg20, "Green voters (comparison)", prior_var20)
describe_group(df20, mask_nat20, 'vote', cfg20, "National voters (comparison)", prior_var20)

out_env = df20.loc[mask_out20, 'env'].dropna()
grn_env20 = df20.loc[mask_grn20, 'env'].dropna()
nat_env20 = df20.loc[mask_nat20, 'env'].dropna()
if len(out_env) >= 5 and len(grn_env20) >= 5:
    t, p = stats.ttest_ind(out_env, grn_env20)
    print(f"\n  Outdoor vs Green env position: t={t:.2f}, p={p:.3f}")
if len(out_env) >= 5 and len(nat_env20) >= 5:
    t, p = stats.ttest_ind(out_env, nat_env20)
    print(f"  Outdoor vs National env position: t={t:.2f}, p={p:.3f}")

# Urban/rural breakdown of Outdoor voters vs Green voters
if urban_var20 and len(out_env) >= 5:
    out_urban = df20.loc[mask_out20, urban_var20].dropna()
    grn_urban = df20.loc[mask_grn20, urban_var20].dropna()
    nat_urban = df20.loc[mask_nat20, urban_var20].dropna()
    print(f"\n  Urban/rural (1=rural, 4=city):")
    print(f"  Outdoor mean={out_urban.mean():.2f}  Green mean={grn_urban.mean():.2f}  National mean={nat_urban.mean():.2f}")

# ---------------------------------------------------------------------------
# Section 4: Environmental group membership × party vote (1996-2005)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 4: ENVIRONMENTAL GROUP MEMBERSHIP × PARTY VOTE (1996-2005)")
print("=" * 70)

memenv_configs = [c for c in YEARS if 'memenv_var' in c]
for cfg in memenv_configs:
    df, cfg = load_year(cfg)
    mem_col = cfg['memenv_var']
    mem_raw = pd.to_numeric(
        pd.read_sql(f'SELECT "{mem_col}" FROM {cfg["table"]}', conn)[mem_col],
        errors='coerce')
    df['is_member'] = mem_raw.isin([1, 2, 3, 4, 5, 6])  # any membership level

    members = df[df['is_member']].copy()
    n = len(members)
    if n < 10:
        print(f"{cfg['year']}: too few env group members (n={n})")
        continue

    nat_pct = (members['vote'] == cfg['nat']).mean() * 100
    lab_pct = (members['vote'] == cfg['lab']).mean() * 100
    grn_pct = (members['vote'] == cfg.get('grn', -1)).mean() * 100 if cfg.get('grn') else np.nan
    nzf_pct = (members['vote'] == cfg['nzf']).mean() * 100
    env_mean = members['env'].mean()

    print(f"\n{cfg['year']}  Env group members (n={n}):")
    print(f"  Vote: Nat={nat_pct:.1f}%  Lab={lab_pct:.1f}%  Grn={grn_pct:.1f}%  NZF={nzf_pct:.1f}%")
    print(f"  Avg env position: {env_mean:.2f} (1=protect, 7=develop)")

    # Non-members for comparison
    non = df[~df['is_member']]
    nat_non = (non['vote'] == cfg['nat']).mean() * 100
    grn_non = (non['vote'] == cfg.get('grn', -1)).mean() * 100 if cfg.get('grn') else np.nan
    print(f"  Non-members (n={len(non)}): Nat={nat_non:.1f}%  Grn={grn_non:.1f}%")

# ---------------------------------------------------------------------------
# Section 5: Visualization
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 5: GENERATING VISUALIZATION")
print("=" * 70)

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Conservation Voter Analysis: Is There an\n"
             "'Available-to-National' Environmentalist Cohort?",
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ---- Panel A: Party vote share among pro-env voters over time ----
ax1 = fig.add_subplot(gs[0, :])
td = trend_df.dropna(subset=['Green'])
years_with_grn = td['year'].values

ax1.plot(td['year'], td['National'], 'b-o', label='National', linewidth=2)
ax1.plot(td['year'], td['Labour'], 'r-s', label='Labour', linewidth=2)
ax1.plot(td['year'], td['Green'], 'g-^', label='Green', linewidth=2)
ax1.plot(td['year'], td['NZFirst'], 'k--d', label='NZ First', linewidth=1.5, alpha=0.7)

# Also plot early years without Green separately
early = trend_df[trend_df['Green'].isna()]
ax1.plot(early['year'], early['National'], 'b-o', linewidth=2)
ax1.plot(early['year'], early['Labour'], 'r-s', linewidth=2)

ax1.axhline(25, color='gray', linestyle=':', alpha=0.4)
ax1.set_xlabel('Election year')
ax1.set_ylabel('% of pro-environment voters (env score ≤3)')
ax1.set_title('Panel A: Where do pro-environment voters go? (1993–2023)')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_ylim(0, 65)
ax1.set_xticks(trend_df['year'].values)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# ---- Panel B: Available conservationist segment size over time ----
ax2 = fig.add_subplot(gs[1, 0])
avail_df = pd.DataFrame(avail_rows).dropna(subset=['pct_avail'])
ax2.bar(avail_df['year'] - 0.4, avail_df['pct_avail'], width=0.8,
        color='steelblue', alpha=0.8, label='Never-Green pro-env (available)')
ax2.bar(avail_df['year'] + 0.4, avail_df['pct_ideolog'], width=0.8,
        color='green', alpha=0.6, label='Never-National pro-env (ideological)')
ax2.set_xlabel('Election year')
ax2.set_ylabel('% of pro-environment voters')
ax2.set_title('Panel B: "Available conservationist" segment\n(never-vote-Green pro-env voters)')
ax2.legend(fontsize=8)
ax2.set_xticks(avail_df['year'].values)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# ---- Panel C: Available conservationist vote destination ----
ax3 = fig.add_subplot(gs[1, 1])
avail_valid = avail_df.dropna(subset=['avail_nat', 'avail_lab'])
x = np.arange(len(avail_valid))
w = 0.35
ax3.bar(x - w/2, avail_valid['avail_nat'], width=w, color='blue', alpha=0.7, label='National')
ax3.bar(x + w/2, avail_valid['avail_lab'], width=w, color='red', alpha=0.7, label='Labour')
ax3.set_xlabel('Election year')
ax3.set_ylabel('% of never-Green pro-env voters')
ax3.set_title('Panel C: Where available conservationists\nactually vote')
ax3.set_xticks(x)
ax3.set_xticklabels(avail_valid['year'].astype(int), rotation=45)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ---- Panel D: Env position distribution by party (2023) ----
ax4 = fig.add_subplot(gs[2, 0])
cfg23 = next(c for c in YEARS if c['year'] == 2023)
df23, cfg23 = load_year(cfg23)
parties = {'National': cfg23['nat'], 'Labour': cfg23['lab'],
           'Green': cfg23['grn'], 'NZ First': cfg23['nzf']}
colors = {'National': 'blue', 'Labour': 'red', 'Green': 'green', 'NZ First': 'black'}
for pname, pcode in parties.items():
    sub = df23[df23['vote'] == pcode]['env'].dropna()
    if len(sub) > 20:
        counts = sub.value_counts().sort_index()
        pcts = counts / counts.sum() * 100
        ax4.plot(pcts.index, pcts.values, 'o-', color=colors[pname],
                 label=f'{pname} (n={len(sub)})', linewidth=2, markersize=5)

ax4.axvline(3.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
ax4.set_xlabel('Environment ← 1 ——————————— 7 → Development')
ax4.set_ylabel('% of party voters')
ax4.set_title('Panel D: Environment/Economy position\nby party (2023)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(1, 8))

# ---- Panel E: Natural experiments — env position ----
ax5 = fig.add_subplot(gs[2, 1])

exp_groups = []
# 2014
if len(ban_env) >= 5:
    exp_groups.append(('Ban1080\n2014', ban_env, '#8B4513'))
    exp_groups.append(('Green\n2014', grn_env, 'green'))
    exp_groups.append(('National\n2014', nat_env, 'blue'))
# 2020
if len(out_env) >= 5:
    exp_groups.append(('Outdoor\n2020', out_env, '#228B22'))
    exp_groups.append(('Green\n2020', grn_env20, '#00CC44'))
    exp_groups.append(('National\n2020', nat_env20, '#4444FF'))

if exp_groups:
    positions = range(len(exp_groups))
    labels = [g[0] for g in exp_groups]
    means = [g[1].mean() for g in exp_groups]
    sems = [g[1].sem() for g in exp_groups]
    group_colors = [g[2] for g in exp_groups]

    bars = ax5.barh(positions, means, xerr=sems, color=group_colors, alpha=0.75,
                    capsize=4)
    ax5.set_yticks(positions)
    ax5.set_yticklabels(labels, fontsize=8)
    ax5.set_xlabel('Mean env/economy position (1=env, 7=develop)')
    ax5.set_title('Panel E: Natural experiments\nEnv position ± SE by voter group')
    ax5.axvline(4, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlim(1, 7)
    ax5.grid(True, alpha=0.3, axis='x')

plt.savefig('reports/conservation_voters.png', dpi=150, bbox_inches='tight')
print("Saved: reports/conservation_voters.png")

print("\n" + "=" * 70)
print("DONE")
