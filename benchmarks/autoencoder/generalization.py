"""Exploratory transfer benchmark; models fitted only on original B training records."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from scipy.signal import sawtooth
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from featuregraph_core.behaviors.oscillation import OscillationConfig
from data import FS, WIN, HALO, SMOOTH, FEATURES
from system_identification import TRAIN, TARGETS, TIME, simulate, features
from bycycle_comparison import bycycle_features, BAND, FILTER_SECONDS, BAND_AMP_CYCLES
from benchmark import smoothed, spectrum, encode

# New seeds, separate from prior development, matched-pair and validation seeds.
EVAL = tuple(range(40000,40200))
CONDITIONS = (
    ("sinusoid_in_range", "cosine", .20, .35),
    ("triangle_in_range", "triangle", .20, .35),
    ("asymmetric_in_range", "asymmetric", .20, .35),
    ("flattened_in_range", "flattened", .20, .35),
    ("sinusoid_slow", "cosine", .12, .19),
    ("sinusoid_fast", "cosine", .36, .55),
    ("triangle_fast", "triangle", .36, .55),
)

def shifted(seed, shape, lo, hi):
    rng=np.random.default_rng(seed)
    zeta=rng.uniform(.005,.04)
    fn=rng.uniform(lo,hi)
    amp=rng.uniform(1.,2.)
    phase=rng.uniform(0,2*np.pi)
    baseline=rng.uniform(-.3,.3)
    noise=rng.uniform(.025,.075)
    wn=2*np.pi*fn
    theta=wn*np.sqrt(1-zeta*zeta)*TIME+phase
    if shape=="cosine": carrier=np.cos(theta)
    elif shape=="triangle": carrier=-sawtooth(theta,.5)
    elif shape=="asymmetric": carrier=-sawtooth(theta,.65)
    elif shape=="flattened": carrier=np.tanh(2*np.cos(theta))/np.tanh(2)
    else: raise ValueError(shape)
    x=amp*np.exp(-zeta*wn*TIME)*carrier+baseline+rng.normal(0,noise,len(TIME))
    return x,np.array([zeta,fn])

def retained(x):
    # Preserve every complete cycle, even if period is undefined.
    cfg=OscillationConfig(signal="sig",smooth_window=SMOOTH)
    d=cfg.add_primitives(pd.DataFrame({"rec":0,"sig":x}),"rec")
    s=cfg.summarize(d,["rec",cfg.trough_event_id_col]).reset_index()
    s=s[s.is_complete & (s.peak_index>=HALO) & (s.peak_index<HALO+WIN)].copy()
    s["period_s"]=s.period/FS
    v=s[FEATURES]
    left=s.peak_index<HALO+WIN//2
    return v.mean().to_numpy(float),np.r_[v[left].mean(),v[~left].mean()],len(s)

def representations(x):
    f,s,n=retained(x)
    try:
        b,c,m=bycycle_features(x)
        error=""
    except (ValueError,IndexError) as exc:
        # Do not drop failed records; report exception and use training imputation.
        # Unexpected failures are visible in the artifact, not counted as cycles.
        b,c,m=np.full(4,np.nan),np.full(8,np.nan),0
        error=type(exc).__name__+": "+str(exc)
    return {"FG4":f,"FG8":s,"BC4":b,"BC8":c},(n,m,error)

def collect(rows,label):
    ys=[];xs=[];rs=[];audit=[]
    for seed,x,y in rows:
        r,(n,m,error)=representations(x)
        xs.append(x);ys.append(y);rs.append(r)
        audit.append(dict(condition=label,recording=seed,fg_cycles=n,bc_cycles=m,
                          bycycle_error=error))
    return np.stack(xs),np.stack(ys),{k:np.stack([r[k] for r in rs]) for k in rs[0]},audit

def run(out,with_ae):
    assert not set(TRAIN)&set(EVAL)
    for seed in TRAIN[:3]:
        x,y=simulate(seed); a,b=shifted(seed,"cosine",.20,.35)
        np.testing.assert_array_equal(y,b)
        np.testing.assert_allclose(x,a,rtol=0,atol=1e-12)
        f,s,n=retained(x)
        assert f.shape==(4,) and s.shape==(8,)
    x,y,train,audit=collect(((s,*simulate(s)) for s in TRAIN),"train")
    groups={}
    for label,shape,lo,hi in CONDITIONS:
        xx,yy,rr,aa=collect(((s,*shifted(s,shape,lo,hi)) for s in EVAL),label)
        groups[label]=(xx,yy,rr)
        audit.extend(aa)
        print("Constructed",label,flush=True)
    pd.DataFrame(audit).to_csv(out/"construction_audit.csv",index=False)
    preds=[]; cov=[]; coef=[]; diag=[]
    def evaluate(name,a,tests,seed=-1):
        fill=np.nanmedian(np.where(np.isfinite(a),a,np.nan),axis=0)
        if not np.isfinite(fill).all(): raise ValueError("No training measurements for "+name)
        sc=StandardScaler().fit(np.where(np.isfinite(a),a,fill))
        aa=sc.transform(np.where(np.isfinite(a),a,fill))
        models=[Ridge(alpha=1),RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=2)]
        for model in models:
            model.fit(aa,y)
            if isinstance(model,Ridge):
                for j,target in enumerate(TARGETS):
                    for i,w in enumerate(model.coef_[j]):
                        coef.append(dict(method=name,seed=seed,target=target,feature=i,weight=w))
            for label,z in tests.items():
                tt=groups[label][1]
                pp=model.predict(sc.transform(np.where(np.isfinite(z),z,fill)))
                for j,target in enumerate(TARGETS):
                    preds.extend(dict(condition=label,method=name,seed=seed,model=type(model).__name__,
                        target=target,recording=s,true=u,pred=v)
                        for s,u,v in zip(EVAL,tt[:,j],pp[:,j]))
        for label,z in [("train",a),*tests.items()]:
            cov.append(dict(method=name,seed=seed,condition=label,n=len(z),
                            complete=np.isfinite(z).all(1).mean(),
                            finite_fraction=np.isfinite(z).mean()))
    for name,a in train.items():
        evaluate(name,a,{k:r[name] for k,(_,_,r) in groups.items()})
        if a.shape[1]==8:
            evaluate(name+"_amp2",a[:,[1,5]],{k:r[name][:,[1,5]] for k,(_,_,r) in groups.items()})
    for name,transform in [("raw",lambda a:a),("smooth",smoothed),("spectrum",spectrum),
                            ("smooth_spectrum",lambda a:spectrum(smoothed(a)))]:
        a=transform(x); tests={k:transform(xx) for k,(xx,_,_) in groups.items()}
        evaluate(name,a,tests)
        if with_ae:
            mu,sd=a.mean(),a.std()
            a=(a-mu)/sd
            tests={k:(z-mu)/sd for k,z in tests.items()}
            for dim in (4,8):
                for seed in range(5):
                    ae=MLPRegressor(hidden_layer_sizes=(64,32,dim,32,64),activation="tanh",
                                    max_iter=400,random_state=seed).fit(a,a)
                    method=f"{name}_AE_{dim}"
                    evaluate(method,encode(ae,a),{k:encode(ae,z) for k,z in tests.items()},seed)
                    diag.extend(dict(method=method,seed=seed,condition=k,iterations=ae.n_iter_,
                        reconstruction_r2=r2_score(z,ae.predict(z),multioutput="variance_weighted"))
                        for k,z in tests.items())
                    print(method,seed,flush=True)
    p=pd.DataFrame(preds);p.to_csv(out/"predictions.csv",index=False)
    metrics=[]
    for keys,g in p.groupby(["condition","method","seed","model","target"]):
        metrics.append(dict(zip(["condition","method","seed","model","target"],keys),
            n=len(g),r2=r2_score(g.true,g.pred),mae=mean_absolute_error(g.true,g.pred),
            rmse=np.sqrt(mean_squared_error(g.true,g.pred))))
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(cov).to_csv(out/"coverage.csv",index=False)
    pd.DataFrame(coef).to_csv(out/"coefficients.csv",index=False)
    if diag: pd.DataFrame(diag).to_csv(out/"autoencoder_diagnostics.csv",index=False)
    (out/"config.json").write_text(json.dumps(dict(status="exploratory_transfer",train=TRAIN,
        evaluation=EVAL,conditions=CONDITIONS,fs=FS,smooth=SMOOTH,band=BAND,
        filter_seconds=FILTER_SECONDS,band_amp_cycles=BAND_AMP_CYCLES,with_ae=with_ae,
        versions={k:version(k) for k in ("bycycle","neurodsp","numpy","pandas","scipy","scikit-learn")}),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--with-ae",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out,args.with_ae)
