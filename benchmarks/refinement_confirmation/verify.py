"""Engineering equivalence test; no tuning or evaluation-seed changes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import sawtooth
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from featuregraph_core.behaviors.oscillation import OscillationConfig

TIME=np.arange(1000)/50
TRAIN=range(10000,10600)
EVAL=range(50000,50200)
CONDS=[('sinusoid_in_range','cosine',.20,.35),('triangle_in_range','triangle',.20,.35),
       ('asymmetric_in_range','asymmetric',.20,.35),('flattened_in_range','flattened',.20,.35),
       ('sinusoid_slow','cosine',.12,.19),('sinusoid_fast','cosine',.36,.55),
       ('triangle_fast','triangle',.36,.55)]
COLS=['period','amplitude','rise','symmetry']
TARGETS=['damping_ratio','natural_frequency_hz']
# Frozen archived R2, condition order above, original/refined/amp_period;
# each pair is damping ratio and natural frequency.
EXPECTED=np.array([
[[.7672019287788583,.9806816460381413],[.776178363278717,.9799811046481753],[.7639799992394927,.9802616271475286]],
[[.7568827128169682,.9811390374182558],[.7593168896399024,.9805747959748662],[.7546671993942189,.9806817252408678]],
[[.021825227063803743,.8925624257634929],[.7337736769097403,.9755827808336238],[.7762010616334839,.9815316569756237]],
[[.7799998305221691,.9803556306280962],[.7878377515550792,.9772005891449731],[.7797272413726847,.978226510134132]],
[[-2.819013844773524,-5.373669583374982],[-2.9503220335015468,-11.488391027655304],[-2.1843512954795345,-12.852827145306843]],
[[.8045897755621079,-1.357462281734136],[.8060815117030325,-1.3534597675882676],[.8036640731166331,-1.3555641871770612]],
[[.771789610621136,-1.3501457906961178],[.7854411602681972,-1.343826697666732],[.7766139743075209,-1.3470995932770213]]
])

def generate(seed,shape='cosine',lo=.20,hi=.35):
    rng=np.random.default_rng(seed)
    z=rng.uniform(.005,.04); f=rng.uniform(lo,hi); a=rng.uniform(1,2)
    p=rng.uniform(0,2*np.pi); b=rng.uniform(-.3,.3); n=rng.uniform(.025,.075)
    w=2*np.pi*f; theta=w*np.sqrt(1-z*z)*TIME+p
    carrier={'cosine':lambda:np.cos(theta),'triangle':lambda:-sawtooth(theta,.5),
             'asymmetric':lambda:-sawtooth(theta,.65),
             'flattened':lambda:np.tanh(2*np.cos(theta))/np.tanh(2)}[shape]()
    return a*np.exp(-z*w*TIME)*carrier+b+rng.normal(0,n,len(TIME)),[z,f]

def reference(x):
    # Independent copy of the frozen research implementation.
    s=pd.Series(x); sm=s.rolling(50,center=True).median().rolling(50,center=True).mean()
    r=sm.diff().gt(0); f=sm.diff().lt(0)
    enter=r.astype(int).diff().eq(1); exit=r.astype(int).diff().eq(-1)
    idx=np.arange(len(x))
    df=pd.DataFrame(dict(raw=s,r=r,f=f,peak=pd.Series(np.where(exit,idx,np.nan)).ffill(),
                         trough=pd.Series(np.where(enter,idx,np.nan)).ffill(),group=enter.cumsum()))
    o=df.groupby('group').agg(start=('trough','first'),rise=('r','sum'),fall=('f','sum'),
                              peak=('peak','last'),hi=('raw','max'),lo=('raw','min'))
    o['end']=o.start.shift(-1); o['period']=o.peak.diff()/50
    o['amplitude']=(o.hi-o.lo)/2; o['symmetry']=1-abs(o.rise-o.fall)/(o.rise+o.fall)
    o=o[o.start.notna()&o.end.notna()&(o.peak>=250)&(o.peak<750)].copy()
    anchors=sorted(set([(int(a),False) for a in o.start]+[(int(a),False) for a in o.end]+
                       [(int(a),True) for a in o.peak]))
    rr={}
    for j,(event,peak) in enumerate(anchors):
        a=event-1
        prev=anchors[j-1][0]-1 if j else None
        nxt=anchors[j+1][0]-1 if j+1<len(anchors) else None
        lo=max(0,a-25,(prev+a)//2+1 if prev is not None else 0)
        hi=min(len(x)-1,a+25,(a+nxt)//2 if nxt is not None else len(x)-1)
        rr[event,peak]=None if hi<lo else lo+int(np.argmax(x[lo:hi+1]) if peak else np.argmin(x[lo:hi+1]))
    v=o.copy(); valid=0
    for k,z in o.iterrows():
        a,p,b=int(z.start),int(z.peak),int(z.end)
        ra,rp,rb=rr[a,False],rr[p,True],rr[b,False]
        if not a<p<b or None in (ra,rp,rb) or not ra<rp<rb:
            v.loc[k,['rise','symmetry']]=np.nan; continue
        valid+=1; v.loc[k,'rise']=rp-ra; v.loc[k,'symmetry']=1-abs(2*rp-ra-rb)/(rb-ra)
    return o,v,valid

def integrated(x):
    cfg=OscillationConfig(signal='sig',smooth_window=50)
    df=pd.DataFrame({'rec':0,'sig':x})
    s=cfg.summarize(cfg.add_primitives(df,'rec'),['rec',cfg.trough_event_id_col])
    s=s[s.is_complete & (s.peak_index>=250) & (s.peak_index<750)]
    q=cfg.refine_temporal(df,s,'rec')
    pd.testing.assert_frame_equal(q[s.columns],s)
    o=s.reset_index(drop=True).rename(columns={'start_index':'start','peak_index':'peak',
        'end_index':'end','rising_duration':'rise','amplitude_raw':'amplitude','temporal_symmetry':'symmetry'})
    o['period']=o.period/50
    v=o.copy()
    v['rise']=q.refined_rising_duration.to_numpy()
    v['symmetry']=q.refined_temporal_symmetry.to_numpy()
    return o,v,int(q.refined_temporal_symmetry.notna().sum())

def aggregate(o):
    return np.r_[o.loc[o.peak<500,COLS].mean().values,o.loc[o.peak>=500,COLS].mean().values]

def collect(seeds,label,shape='cosine',lo=.20,hi=.35):
    rows=[]; audit=[]
    for seed in seeds:
        x,y=generate(seed,shape,lo,hi)
        old,new,n=integrated(x)
        ro,rn,m=reference(x)
        try:
            for a,b in [(old,ro),(new,rn)]:
                np.testing.assert_allclose(a[['start','peak','end']+COLS],
                    b[['start','peak','end']+COLS],atol=1e-12,rtol=0,equal_nan=True)
            assert n==m
        except AssertionError as exc:
            raise AssertionError(f'{label} seed={seed}: {exc}') from exc
        rows.append((y,aggregate(old),aggregate(new)))
        audit.append(dict(condition=label,recording=seed,cycles=len(old),refined_valid=n))
    print('Verified cycle-level equivalence:',label,len(rows),flush=True)
    return tuple(np.stack(v) for v in zip(*rows)),audit

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    (y,a,b),audit=collect(TRAIN,'train')
    models={}
    for name,x,ix in [('original',a,list(range(8))),('refined',b,list(range(8))),('amp_period',a,[0,1,4,5])]:
        x=x[:,ix]; fill=np.nanmedian(x,axis=0)
        sc=StandardScaler().fit(np.where(np.isfinite(x),x,fill))
        m=Ridge(alpha=1).fit(sc.transform(np.where(np.isfinite(x),x,fill)),y)
        models[name]=(ix,fill,sc,m)
    rows=[]; preds=[]
    actual=[]
    for label,shape,lo,hi in CONDS:
        (t,c,d),au=collect(EVAL,label,shape,lo,hi); audit+=au
        results=[]
        for name,(ix,fill,sc,m) in models.items():
            z=(d if name=='refined' else c)[:,ix]
            p=m.predict(sc.transform(np.where(np.isfinite(z),z,fill)))
            scores=[]
            for j,target in enumerate(TARGETS):
                r2=r2_score(t[:,j],p[:,j]); scores.append(r2)
                rows.append(dict(condition=label,method=name,target=target,r2=r2,
                    mae=mean_absolute_error(t[:,j],p[:,j]),bias=np.mean(p[:,j]-t[:,j]),
                    coverage=np.isfinite(z).all(1).mean()))
                preds.extend(dict(condition=label,method=name,target=target,seed=s,true=u,pred=v)
                    for s,u,v in zip(EVAL,t[:,j],p[:,j]))
            results.append(scores)
        actual.append(results)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'audit.csv',index=False)
    verification=pd.DataFrame(rows)[['condition','method','target','r2']].copy()
    verification['archived_r2']=EXPECTED.ravel()
    verification['absolute_error']=abs(verification.r2-verification.archived_r2)
    verification.to_csv(out/'verification.csv',index=False)
    print('Max archived R2 difference:',verification.absolute_error.max())
    # Engineering tolerance, chosen before integrated execution; not a scientific threshold.
    np.testing.assert_allclose(actual,EXPECTED,atol=1e-9,rtol=0)
    print('PASS: integrated measurements and archived metrics agree.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
