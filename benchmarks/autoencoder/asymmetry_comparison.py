"""Post-hoc paired morphology diagnostic; no evaluation data used for fitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from generalization import collect, shifted, EVAL
from system_identification import TRAIN, simulate

NAMES=[f"{name}_{i}" for i in (1,2) for name in ("period","amplitude","rise","symmetry")]
SUBSETS={"full":list(range(8)), "no_symmetry":[0,1,2,4,5,6],
         "amplitude_period":[0,1,4,5],"amplitude_rise":[1,2,5,6],
         "amplitude_only":[1,5],"amplitude_symmetry":[1,3,5,7]}
def run(out):
    _,y,train,audit=collect(((s,*simulate(s)) for s in TRAIN),"train")
    tests={}
    for label,shape in [("sinusoid_in_range","cosine"),("asymmetric_in_range","asymmetric")]:
        _,t,r,a=collect(((s,*shifted(s,shape,.20,.35)) for s in EVAL),label)
        tests[label]=(t,r)
        audit.extend(a)
    sin,asy=tests.values()
    np.testing.assert_array_equal(sin[0],asy[0])
    target=sin[0][:,0]
    features=[]; contrib=[]; stats=[]; metrics=[]; predictions=[]; hybrid=[]
    def score(method,subset,condition,p):
        metrics.append(dict(method=method,subset=subset,condition=condition,
            r2=r2_score(target,p),mae=mean_absolute_error(target,p),
            bias=np.mean(p-target)))
    for method in ("FG8","BC8"):
        a=train[method]
        raw=[a,sin[1][method],asy[1][method]]
        fill=np.nanmedian(np.where(np.isfinite(a),a,np.nan),axis=0)
        assert np.isfinite(fill).all()
        aa,ss,tt=[np.where(np.isfinite(v),v,fill) for v in raw]
        for label,seeds,v in zip(("train",*tests),(TRAIN,EVAL,EVAL),raw):
            features.extend(dict(method=method,condition=label,recording=s,
                **dict(zip(NAMES,row))) for s,row in zip(seeds,v))
        for subset,ix in SUBSETS.items():
            sc=StandardScaler().fit(aa[:,ix])
            model=Ridge(alpha=1).fit(sc.transform(aa[:,ix]),y)
            ps=model.predict(sc.transform(ss[:,ix]))[:,0]
            pt=model.predict(sc.transform(tt[:,ix]))[:,0]
            for label,p in zip(tests,(ps,pt)):
                score(method,subset,label,p)
                predictions.extend(dict(method=method,subset=subset,condition=label,
                    recording=s,true=u,pred=v) for s,u,v in zip(EVAL,target,p))
            if subset!="full": continue
            w=model.coef_[0]
            dz=(tt-ss)/sc.scale_
            c=dz*w
            np.testing.assert_allclose(c.sum(1),pt-ps,atol=1e-12,rtol=0)
            for j,name in enumerate(NAMES):
                stats.append(dict(method=method,feature=name,train_mean=sc.mean_[j],
                    train_scale=sc.scale_[j],train_median=fill[j],
                    weight_standardized=w[j],weight_original=w[j]/sc.scale_[j],
                    sin_mean=ss[:,j].mean(),asym_mean=tt[:,j].mean(),
                    sin_missing=(~np.isfinite(raw[1][:,j])).sum(),
                    asym_missing=(~np.isfinite(raw[2][:,j])).sum(),
                    mean_delta_standardized=dz[:,j].mean(),mean_contribution=c[:,j].mean(),
                    mean_abs_contribution=np.abs(c[:,j]).mean()))
                contrib.extend(dict(method=method,recording=s,feature=name,
                    sin_raw=u,asym_raw=v,delta_standardized=d,contribution=k)
                    for s,u,v,d,k in zip(EVAL,raw[1][:,j],raw[2][:,j],dz[:,j],c[:,j]))
            for name,ix in {"period":[0,4],"amplitude":[1,5],"rise":[2,6],"symmetry":[3,7]}.items():
                # Fixed-model substitution, not a physical counterfactual.
                h=tt.copy();h[:,ix]=ss[:,ix]
                p=model.predict(sc.transform(h))[:,0]
                hybrid.append(dict(method=method,replaced=name,r2=r2_score(target,p),
                    mae=mean_absolute_error(target,p),bias=np.mean(p-target)))
    for name,rows in [("features",features),("paired_contributions",contrib),
                      ("feature_statistics",stats),("subset_metrics",metrics),
                      ("predictions",predictions),("hybrid_metrics",hybrid),("construction_audit",audit)]:
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False)
    print(pd.DataFrame(stats).to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    run(args.out)
