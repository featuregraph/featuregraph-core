import numpy as np
from data import simulate, examples, objects, DEV_SEEDS, TEST_SEEDS, FS, WIN, HALO

def test_seed_split():
    assert not set(DEV_SEEDS) & set(TEST_SEEDS)

def test_simulation_repeatability():
    a = simulate(DEV_SEEDS[0])
    b = simulate(DEV_SEEDS[0])
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
    assert not np.array_equal(a[0],simulate(DEV_SEEDS[1])[0])

def test_context_and_targets():
    signal, peaks, amps = simulate(DEV_SEEDS[0])
    rows = list(examples(DEV_SEEDS[0]))
    assert len(rows) == 59
    for r in rows:
        s = r["start"]
        np.testing.assert_array_equal(r["x"],signal[s-HALO:s+WIN+HALO])
        mask = (peaks[1:] >= s) & (peaks[1:] < s+WIN)
        np.testing.assert_allclose(r["y"],[np.diff(peaks)[mask].mean()/FS,amps[1:][mask].mean()])
        assert np.isfinite(r["y"]).all()
        assert ((r["detected_peaks"] >= s) & (r["detected_peaks"] < s+WIN)).all()
    assert np.all(np.diff([r["start"] for r in rows]) == WIN)

def test_local_construction():
    r = next(examples(DEV_SEEDS[0]))
    a = objects(r["x"])
    b = objects(r["x"].copy())
    assert a.equals(b)
    assert ((a.start_index >= 0) & (a.end_index < len(r["x"]))).all()

if __name__ == "__main__":
    test_seed_split()
    test_simulation_repeatability()
    test_context_and_targets()
    test_local_construction()
    for seed in DEV_SEEDS:
        rows = list(examples(seed))
        f = np.stack([r["f"] for r in rows])
        y = np.stack([r["y"] for r in rows])
        valid = np.isfinite(f).all(axis=1)
        print(seed, "windows",len(rows),"coverage",valid.mean(),
              "direct MAE",np.abs(f[valid,:2]-y[valid]).mean(axis=0))
