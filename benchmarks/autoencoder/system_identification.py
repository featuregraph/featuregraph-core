"""Development-only Experiment B. No final test set is generated or inspected."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data import FS, WIN, HALO, FEATURES, objects
from benchmark import smoothed, spectrum, encode

N = WIN + 2*HALO
TRAIN = tuple(range(10000,10600))
VALID = tuple(range(20000,20200))
TARGETS = ("damping_ratio","natural_frequency_hz")
AE_SEEDS = (0,1,2,3,4)
TIME = np.arange(N)/FS

def simulate(seed):
    rng = np.random.default_rng(seed)
    zeta = rng.uniform(.005,.04)
    fn = rng.uniform(.20,.35)
    amp = rng.uniform(1.,2.)
    phase = rng.uniform(0,2*np.pi)
    baseline = rng.uniform(-.3,.3)
    noise = rng.uniform(.025,.075)
    wn = 2*np.pi*fn
    wd = wn*np.sqrt(1-zeta*zeta)
    x = amp*np.exp(-zeta*wn*TIME)*np.cos(wd*TIME+phase)
    x = x+baseline+rng.normal(0,noise,N)
    return x,np.array([zeta,fn])

def features(x):
    o=objects(x)
    f=o[FEATURES].mean().to_numpy(float)
    left=o[o.peak_index<HALO+WIN//2]
    right=o[o.peak_index>=HALO+WIN//2]
    s=np.concatenate([left[FEATURES].mean().to_numpy(float),
                      right[FEATURES].mean().to_numpy(float)])
    return f,s,len(o)

def collect(seeds):
    xs,ys,fs,ss,counts=[],[],[],[],[]
    for seed in seeds:
        x,y=simulate(seed)
        f,s,c=features(x)
        xs.append(x);ys.append(y);fs.append(f);ss.append(s);counts.append(c)
    return tuple(np.asarray(v) for v in (xs,ys,fs,ss,counts))

def impute(a,b):
    fill=np.nanmedian(a,axis=0)
    if not np.isfinite(fill).all():
        raise ValueError("Feature column has no finite training values")
    return np.where(np.isfinite(a),a,fill),np.where(np.isfinite(b),b,fill)

def physics_fit(x):
    # Equation-aware reference, not an unsupervised representation.
    # Fit decay rate, damped frequency, two quadratures and constant baseline.
    def residual(p):
        decay,fd,u,v,b=p
        w=2*np.pi*fd*TIME
        return np.exp(-decay*TIME)*(u*np.cos(w)+v*np.sin(w))+b-x
    best=None
    for fd in (.20,.275,.35):
        fit=least_squares(residual,[.04,fd,1.,0.,float(x.mean())],
                          bounds=([0,.12,-5,-5,-2],[.25,.45,5,5,2]),
                          max_nfev=250)
        if best is None or fit.cost<best.cost:
            best=fit
    decay,fd=best.x[:2]
    wn=np.hypot(decay,2*np.pi*fd)
    return np.array([decay/wn,wn/(2*np.pi)]),bool(best.success)

def run(out,max_iter):
    assert not set(TRAIN)&set(VALID)
    a,y,f,s,c=collect(TRAIN)
    b,t,g,q,d=collect(VALID)
    assert a.shape==(len(TRAIN),N) and b.shape==(len(VALID),N)
    assert np.isfinite(y).all() and np.isfinite(t).all()
    assert np.array_equal(simulate(TRAIN[0])[0],a[0])
    assert np.array_equal(features(a[0])[0],f[0],equal_nan=True)
    predictions=[]; diagnostics=[]; losses=[]; coverage=[]
    def save(name,seed,model,p):
        for j,target in enumerate(TARGETS):
            predictions.append(pd.DataFrame(dict(method=name,seed=seed,model=model,
                recording=VALID,target=target,true=t[:,j],pred=p[:,j])))
    def evaluate(name,seed,x,z):
        sc=StandardScaler().fit(x)
        for model in (Ridge(alpha=1),RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=2)):
            model.fit(sc.transform(x),y)
            save(name,seed,type(model).__name__,model.predict(sc.transform(z)))
    for name,x,z in (("FG_mean_4",f,g),("FG_two_bins_8",s,q)):
        coverage.append(dict(method=name,train_complete=np.isfinite(x).all(1).mean(),
            validation_complete=np.isfinite(z).all(1).mean()))
        evaluate(name,-1,*impute(x,z))
    # Explicit physics fit provides an informative upper-reference baseline.
    fits=[physics_fit(x) for x in b]
    save("equation_aware_fit",-1,"least_squares",np.stack([r[0] for r in fits]))
    pd.DataFrame({"recording":VALID,"fit_success":[r[1] for r in fits]}).to_csv(out/"physics_fit_status.csv",index=False)
    for name,transform in (("raw",lambda x:x),("smoothed",smoothed),
                           ("spectrum",spectrum),("smoothed_spectrum",lambda x:spectrum(smoothed(x)))):
        x,z=transform(a),transform(b)
        evaluate(name,-1,x,z)
        mu,sd=x.mean(),x.std()
        x,z=(x-mu)/sd,(z-mu)/sd
        for k in (4,8):
            for seed in AE_SEEDS:
                label=f"{name}_AE_{k}"
                ae=MLPRegressor(hidden_layer_sizes=(64,32,k,32,64),activation="tanh",
                                max_iter=max_iter,random_state=seed)
                ae.fit(x,x)
                evaluate(label,seed,encode(ae,x),encode(ae,z))
                diagnostics.append(dict(method=label,seed=seed,iterations=ae.n_iter_,
                    loss=ae.loss_,validation_reconstruction_r2=r2_score(z,ae.predict(z),multioutput="variance_weighted")))
                losses.extend(dict(method=label,seed=seed,iteration=i+1,loss=l)
                              for i,l in enumerate(ae.loss_curve_))
                print(label,seed,"iterations",ae.n_iter_,"loss",ae.loss_,flush=True)
    p=pd.concat(predictions,ignore_index=True)
    p.to_csv(out/"predictions.csv",index=False)
    metrics=[]
    for (method,seed,model,target),r in p.groupby(["method","seed","model","target"]):
        metrics.append(dict(method=method,seed=seed,model=model,target=target,n=len(r),
            r2=r2_score(r.true,r.pred),mae=mean_absolute_error(r.true,r.pred),
            rmse=np.sqrt(mean_squared_error(r.true,r.pred))))
    pd.DataFrame(metrics).to_csv(out/"metrics.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"latent_diagnostics.csv",index=False)
    pd.DataFrame(losses).to_csv(out/"loss_curves.csv",index=False)
    pd.DataFrame(coverage).to_csv(out/"coverage.csv",index=False)
    pd.DataFrame(dict(recording=VALID,cycles=d)).to_csv(out/"cycle_counts.csv",index=False)
    (out/"config.json").write_text(json.dumps(dict(status="development_only",train_seeds=TRAIN,
        validation_seeds=VALID,fs=FS,win=WIN,halo=HALO,max_iter=max_iter,
        features=FEATURES,python=platform.python_version(),numpy=np.__version__,
        pandas=pd.__version__,sklearn=sklearn.__version__),indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-iter",type=int,default=400)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out,args.max_iter)
