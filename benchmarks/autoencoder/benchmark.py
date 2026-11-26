"""Run from this directory with the package installed. Outputs held-out predictions."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data import examples, DEV_SEEDS, TEST_SEEDS, WIN, HALO, FS, SMOOTH
from featuregraph_core.operators.measures import smooth
from matching import peak_metrics

def collect(seeds, stride):
    rows = [r for seed in seeds for r in examples(seed,stride)]
    return rows, np.stack([r["x"] for r in rows]), np.stack([r["y"] for r in rows]), np.stack([r["f"] for r in rows])

def spectrum(x):
    return np.abs(np.fft.rfft(x-x.mean(axis=1,keepdims=True),axis=1))

def smoothed(x):
    # Preserve the common interval and exact FeatureGraph smoothing operator.
    # Edge NaNs are filled from the nearest valid smoothed sample, locally.
    return np.stack([smooth(pd.Series(row), SMOOTH).ffill().bfill().to_numpy()
                     for row in x])

def encode(ae,x):
    for w,b in zip(ae.coefs_[:3],ae.intercepts_[:3]):
        x=np.tanh(x@w+b)
    return x

def run(out,stride):
    dr,x,y,f=collect(DEV_SEEDS,stride)
    tr,z,t,g=collect(TEST_SEEDS,stride)
    assert np.isfinite(y).all() and np.isfinite(t).all()
    meta=pd.DataFrame({"record":[r["seed"] for r in tr],"start":[r["start"] for r in tr]})
    predictions=[]; diagnostics=[]; latent={}
    def evaluate(name,seed,a,b,train_mask=None,test_mask=None):
        train_mask=np.ones(len(y),bool) if train_mask is None else train_mask
        test_mask=np.ones(len(t),bool) if test_mask is None else test_mask
        sc=StandardScaler().fit(a[train_mask])
        for model in [Ridge(alpha=1),RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=2)]:
            model.fit(sc.transform(a[train_mask]),y[train_mask])
            p=model.predict(sc.transform(b[test_mask]))
            frame=meta.loc[test_mask].copy()
            frame["method"]=name; frame["seed"]=seed; frame["model"]=type(model).__name__
            for j,target in enumerate(["period","amplitude"]):
                frame[target+"_true"]=t[test_mask,j]; frame[target+"_pred"]=p[:,j]
            predictions.append(frame)
    fill=np.nanmedian(f,axis=0)
    assert np.isfinite(fill).all()
    fm=np.isfinite(f).all(1); gm=np.isfinite(g).all(1)
    evaluate("FeatureGraph",-1,np.where(np.isfinite(f),f,fill),np.where(np.isfinite(g),g,fill))
    if fm.any() and gm.any():
        evaluate("FeatureGraph complete cases",-1,f,g,fm,gm)
    direct=meta.copy()
    for j,target in enumerate(["period","amplitude"]):
        direct[target+"_true"]=t[:,j]; direct[target+"_measured"]=g[:,j]
    direct["complete"]=gm
    direct.to_csv(out/"direct.csv",index=False)
    pd.concat([meta,pd.DataFrame([peak_metrics(r["detected_peaks"],r["true_peaks"]) for r in tr])],axis=1).to_csv(out/"peaks.csv",index=False)
    for label,transform in [("raw",lambda x:x),("spectrum",spectrum),("smoothed",smoothed),("smoothed_spectrum",lambda x:spectrum(smoothed(x)))]:
        a,b=transform(x),transform(z)
        evaluate(label,-1,a,b)
        mu,sd=a.mean(),a.std()
        a,b=(a-mu)/sd,(b-mu)/sd
        for k in (4,8):
            for seed in range(5):
                ae=MLPRegressor(hidden_layer_sizes=(64,32,k,32,64),activation="tanh",max_iter=400,random_state=seed)
                ae.fit(a,a)
                u,v=encode(ae,a),encode(ae,b)
                name=f"{label}_AE_{k}"
                evaluate(name,seed,u,v)
                diagnostics.append(dict(method=name,seed=seed,reconstruction_r2=r2_score(b,ae.predict(b),multioutput="variance_weighted"),iterations=ae.n_iter_,loss=ae.loss_))
                corr=np.corrcoef(np.column_stack([v,t]).T)[:k,k:]
                pd.DataFrame(corr,columns=["period","amplitude"]).to_csv(out/f"{name}_{seed}_correlations.csv",index=False)
                if seed==0: latent[name]=(u,v)
                else:
                    u0,v0=latent[name]
                    align=LinearRegression().fit(u,u0)
                    diagnostics[-1]["heldout_alignment_r2"]=r2_score(v0,align.predict(v),multioutput="variance_weighted")
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(out/"predictions.csv",index=False)
    metrics=[]
    for keys,group in pred.groupby(["method","seed","model"]):
        for record,block in [("pooled",group),*list(group.groupby("record"))]:
            for target in ["period","amplitude"]:
                a,b=block[target+"_true"],block[target+"_pred"]
                metrics.append(dict(method=keys[0],seed=keys[1],model=keys[2],record=record,target=target,n=len(a),r2=r2_score(a,b),mae=mean_absolute_error(a,b),rmse=np.sqrt(mean_squared_error(a,b))))
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"latent_diagnostics.csv",index=False)
    (out/"config.json").write_text(json.dumps(dict(dev_seeds=DEV_SEEDS,test_seeds=TEST_SEEDS,stride=stride,fs=FS,win=WIN,halo=HALO,smooth_window=SMOOTH,smooth_edge_policy="nearest valid within input interval",dev_coverage=float(fm.mean()),test_coverage=float(gm.mean()),python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--stride",type=int,choices=[50,500],default=500)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out,args.stride)
