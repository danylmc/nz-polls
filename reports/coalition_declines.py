import json,pandas as pd,numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
d=json.load(open('data/2026_polling.json'))
nat=pd.DataFrame([dict(date=p['date'],pollster=p['pollster'],v=p['parties'].get('National')) for p in d['polls']])
pm=pd.read_csv('data/pm_polling_all.csv'); pm=pm[(pm.candidate.str.contains('Luxon'))&(pm.date>='2023-10-15')].rename(columns={'percent':'v'})
def norm(s): return s.str.replace(r'[^A-Za-z]','',regex=True).str.lower().str[:8]
def smooth(df,frac,label):
    df=df.copy(); df['date']=pd.to_datetime(df.date,errors='coerce'); df['v']=pd.to_numeric(df.v,errors='coerce')
    df=df.dropna(subset=['date','v']); df=df[df.date>='2023-10-15'].sort_values('date'); df['pk']=norm(df.pollster)
    t=(df.date-df.date.min()).dt.days.values.astype(float); adj=df.v.values.copy()
    for _ in range(5):
        fit=lowess(adj,t,frac=frac,return_sorted=False)
        he=pd.Series(df.v.values-fit,index=df.pk.values).groupby(level=0).mean()
        he=he-he.mean(); adj=df.v.values-df.pk.map(he).values
    days=pd.date_range(df.date.min(),df.date.max())
    td=(days-df.date.min()).days.values.astype(float)
    s=pd.Series(np.interp(td,np.sort(t),lowess(adj,t,frac=frac)[:,1]),index=days)
    print(f'\n=== {label}: {len(df)} polls, {df.date.min().date()} to {df.date.max().date()}')
    print('house effects:',he.round(1).to_dict())
    print('smoothed, quarterly:',s.resample('QE').last().round(1).to_dict())
    for w in (60,120):
        ch=(s-s.shift(w)).dropna()
        eps=[]; c=ch.copy()
        while len(eps)<4 and len(c):
            e=c.idxmin(); 
            if c[e]>-0.3: break
            eps.append((e-pd.Timedelta(days=w),e,round(c[e],1),round(s[e-pd.Timedelta(days=w)],1),round(s[e],1)))
            c=c[(c.index<e-pd.Timedelta(days=w))|(c.index>e+pd.Timedelta(days=w))]
        print(f' steepest {w}-day declines:')
        for a,b,x,va,vb in eps: print(f'   {a.date()} -> {b.date()}: {x:+} ({va} -> {vb})')
    return s,df
s1,_=smooth(nat,0.15,'National party vote')
s2,p2=smooth(pm,0.25,'Luxon preferred PM')
print(p2.pollster.value_counts().to_dict())
