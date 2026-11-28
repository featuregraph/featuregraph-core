"""Exploratory comparison on the original Experiment B development split."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from bycycle.features import compute_shape_features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from system_identification import TRAIN, VALID, TARGETS, simulate, features, impute
from data import FS, HALO, WIN, FEATURES

# Fixed broad band around the simulator's development frequency range.
BAND = (.15, .45)
FILTER_SECONDS = 8
BAND_AMP_CYCLES = 2  # Separate internal filter; band_amp is not a predictor.
NAMES = ["period_s", "amplitude", "rise_samples", "symmetry"]

def bycycle_features(x):
    d = compute_shape_features(x, FS, BAND, center_extrema="peak", n_cycles=BAND_AMP_CYCLES,
        find_extrema_kwargs={"filter_kwargs":{"n_seconds":FILTER_SECONDS, "n_cycles":None}})
    d = d[(d.sample_peak >= HALO) & (d.sample_peak < HALO + WIN)].copy()
    # Native trough-to-trough period, half native peak-to-trough excursion,
    # rise duration in samples and symmetry mapped to the FG scale.
    v = pd.DataFrame(dict(period_s=d.period/FS, amplitude=d.volt_amp/2,
                         rise_samples=d.time_rise,
                         symmetry=1-abs(2*d.time_rdsym-1)))
    left = d.sample_peak < HALO + WIN//2
    return v.mean().to_numpy(float), np.r_[v[left].mean(),v[~left].mean()], len(d)

def collect(seeds):
    rows=[]; counts=[]
    for seed in seeds:
        x,y=simulate(seed)
        f,s,n=features(x)
        b,c,m=bycycle_features(x)
        rows.append((y,f,s,b,c))
        counts.append(dict(recording=seed,fg_cycles=n,bycycle_cycles=m))
    return [np.stack(a) for a in zip(*rows)],pd.DataFrame(counts)

def run(out):
    (y,f,s,b,c),dc=collect(TRAIN)
    (t,g,q,d,e),vc=collect(VALID)
    preds=[]; coefs=[]; coverage=[]; metrics=[]
    def evaluate(name,a,z,cols,mask=None,train_mask=None,rf=True):
        if mask is None: mask=np.ones(len(t),bool)
        if train_mask is None: train_mask=np.ones(len(y),bool)
        a,z=impute(a[train_mask],z)
        sc=StandardScaler().fit(a)
        models=[Ridge(alpha=1)]
        if rf: models.append(RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=2))
        for model in models:
            model.fit(sc.transform(a),y[train_mask])
            p=model.predict(sc.transform(z[mask]))
            for j,target in enumerate(TARGETS):
                actual=t[mask,j]; predicted=p[:,j]
                label=type(model).__name__
                preds.extend(dict(method=name,model=label,target=target,recording=int(i),true=float(u),pred=float(v))
                    for i,u,v in zip(np.asarray(VALID)[mask],actual,predicted))
                metrics.append(dict(method=name,model=label,target=target,n=len(actual),
                    r2=r2_score(actual,predicted),mae=mean_absolute_error(actual,predicted),
                    rmse=np.sqrt(mean_squared_error(actual,predicted))))
                if isinstance(model,Ridge):
                    coefs.extend(dict(method=name,target=target,feature=k,weight_standardized=w,
                        weight_original=w/sd,training_mean=mu,training_scale=sd)
                        for k,w,mu,sd in zip(cols,model.coef_[j],sc.mean_,sc.scale_))
    for label,a,z,cols in [
        ("FG_mean_4",f,g,NAMES),("FG_two_bins_8",s,q,[k+str(i) for i in (1,2) for k in NAMES]),
        ("bycycle_mean_4",b,d,NAMES),("bycycle_two_bins_8",c,e,[k+str(i) for i in (1,2) for k in NAMES])]:
        coverage.append(dict(method=label,train_complete=np.isfinite(a).all(1).mean(),
            validation_complete=np.isfinite(z).all(1).mean()))
        evaluate(label,a,z,cols)
        if a.shape[1]==8:
            for suffix,ix in [("amplitudes",[1,5]),("first_amp",[1]),("second_amp",[5]),("no_amp",[0,2,3,4,6,7])]:
                evaluate(label+"_"+suffix,a[:,ix],z[:,ix],[cols[i] for i in ix],rf=False)
    # Paired complete-case sensitivity: both methods use identical records.
    for dim,a,z,u,v in [(4,f,g,b,d),(8,s,q,c,e)]:
        mask=np.isfinite(z).all(1)&np.isfinite(v).all(1)
        train_mask=np.isfinite(a).all(1)&np.isfinite(u).all(1)
        cols=NAMES if dim==4 else [k+str(i) for i in (1,2) for k in NAMES]
        if mask.sum()>1 and train_mask.sum()>1:
            evaluate(f"FG_complete_{dim}",a,z,cols,mask,train_mask)
            evaluate(f"bycycle_complete_{dim}",u,v,cols,mask,train_mask)
    for name,records in [("metrics",metrics),("predictions",preds),("coefficients",coefs),("coverage",coverage)]:
        pd.DataFrame(records).to_csv(out/(name+".csv"),index=False)
    pd.concat([dc.assign(split="train"),vc.assign(split="validation")]).to_csv(out/"cycle_counts.csv",index=False)
    (out/"config.json").write_text(json.dumps(dict(status="exploratory_development_only",
        band=BAND,filter_seconds=FILTER_SECONDS,band_amp_cycles=BAND_AMP_CYCLES,train_seeds=TRAIN,validation_seeds=VALID,
        versions={p:version(p) for p in ("bycycle","neurodsp","numpy","pandas","scipy","scikit-learn")}),indent=2))
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out)
