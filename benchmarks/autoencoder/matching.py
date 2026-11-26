"""Maximize match count, then minimize total absolute timing error."""
import numpy as np

def match_peaks(detected, truth, tolerance=50):
    d, t = np.sort(np.asarray(detected)), np.sort(np.asarray(truth))
    n,m = len(d),len(t)
    dp = [[(0,0,()) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            options = [dp[i-1][j],dp[i][j-1]]
            err = abs(int(d[i-1])-int(t[j-1]))
            if err <= tolerance:
                k,c,p = dp[i-1][j-1]
                options.append((k+1,c+err,p+((int(d[i-1]),int(t[j-1])),)))
            dp[i][j] = max(options,key=lambda v:(v[0],-v[1]))
    return dp[n][m][2]

def peak_metrics(detected, truth, tolerance=50, fs=50):
    pairs = match_peaks(detected,truth,tolerance)
    k = len(pairs)
    return dict(tp=k,fp=len(detected)-k,fn=len(truth)-k,
                precision=k/len(detected) if len(detected) else float("nan"),
                recall=k/len(truth) if len(truth) else float("nan"),
                timing_mae_s=sum(abs(a-b) for a,b in pairs)/k/fs if k else float("nan"))

def test_matching():
    assert match_peaks([9,11],[10],2) in (((9,10),),((11,10),))
    assert match_peaks([0,4],[3,7],3) == ((0,3),(4,7))
    assert match_peaks([0],[51],50) == ()
    assert match_peaks([0],[50],50) == ((0,50),)
    assert match_peaks([],[]) == ()
    assert match_peaks([0,8],[7],10) == ((8,7),)

if __name__ == "__main__":
    test_matching()
