"""Experiment D: resource cost on a constrained CPU.

Run pinned to one core with single-threaded math libraries to emulate a small gateway:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 python src/cost_bench.py
Measures training time, single-flow latency (p50/p95, features + inference), batch throughput,
model size, and per-flow packet state kept while the window is open.
"""
import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import run_experiments as X                       # reuses data, splits and detector classes
from features import window_features
from data import FULL, RESULTS
from models import model_bytes

prof = json.load(open(f"{RESULTS}/experiments.json"))["profiles"]
early = (8, 0.5)


def cfg(p):
    c = dict(prof[p]["cfg"])
    c["cols"] = [i for i, b in enumerate(c["mask"]) if b]
    return c


dets = [X.Proposed("Proposed (edge)", cfg("edge")), X.Proposed("Proposed (enterprise)", cfg("enterprise")),
        X.Proposed("Proposed (cloud)", cfg("cloud")),
        X.Fixed("Full-flow RF-100", "RF", dict(n_estimators=100, max_depth=None), FULL),
        X.Fixed("Full-flow GBDT (HGB)", "HGB", dict(max_iter=200, max_depth=None), FULL),
        X.Fixed("Full-flow MLP (32,16)", "MLP", dict(hidden_layer_sizes=(32, 16)), FULL),
        X.Fixed("P2: kNN + NearMiss", "kNN", dict(n_neighbors=5), FULL, sampler=X.nearmiss),
        X.Fixed("Fixed early RF-100 (N=8,T=0.5s)", "RF", dict(n_estimators=100, max_depth=None), early),
        X.Fixed("Fixed early DT-8 (N=2,T=0.1s)", "DT", dict(max_depth=8), (2, 0.1)),
        X.OneClass("E1: Isolation Forest (N=8,T=0.5s)", "iforest", early)]

cic, TE = X.cic, X.TE
rng = np.random.default_rng(0)
lat_idx = rng.choice(TE, 300, replace=False)
bat_idx = rng.choice(TE, min(4000, len(TE)), replace=False)


def stages(d):
    """List of (model, n, t, cols, fraction of flows reaching this stage)."""
    if isinstance(d, X.Proposed):
        c = d.cfg
        out = [(d.c.m2, c["N2"], c["T2"], c["cols"], 1.0)]
        if c["cascade"]:
            _, _, deferred = d.c.decide(cic, bat_idx)
            out = [(d.c.m1, c["N1"], c["T1"], d.c.c1, 1.0), (d.c.m2, c["N2"], c["T2"], c["cols"], float(deferred.mean()))]
        return out
    if isinstance(d, X.OneClass):
        return [(X._ScoreWrap(d.sc, d.m), d.w[0], d.w[1], list(range(31)), 1.0)]
    n, t = (32, np.inf) if d.w == FULL else d.w
    return [(d.m, n, t, d.cols, 1.0)]


out = {}
for d in dets:
    t = time.process_time()
    d.fit(cic, X.TR)
    train_s = time.process_time() - t
    st = stages(d)
    lat = []
    casc = isinstance(d, X.Proposed) and d.cfg["cascade"]
    for i in lat_idx:
        s = time.perf_counter()
        if casc:   # Stage 1 always; Stage 2 only when Stage 1 is not confident
            m1, n1, t1, c1, _ = st[0]
            p = m1.predict_proba(window_features(cic.pk[i:i + 1], cic.proto[i:i + 1], n1, t1)[:, c1])[0, -1]
            if d.cfg["tau_lo"] < p < d.cfg["tau_hi"]:
                m2, n2, t2, c2, _ = st[1]
                m2.predict_proba(window_features(cic.pk[i:i + 1], cic.proto[i:i + 1], n2, t2)[:, c2])
        else:
            m, n, tt, cols, _ = st[0]
            m.predict_proba(window_features(cic.pk[i:i + 1], cic.proto[i:i + 1], n, tt)[:, cols])
        lat.append(time.perf_counter() - s)
    s = time.perf_counter()
    for m, n, tt, cols, frac in st:
        k = max(1, int(round(frac * len(bat_idx))))
        m.predict_proba(window_features(cic.pk[bat_idx[:k]], cic.proto[bat_idx[:k]], n, tt)[:, cols])
    el = time.perf_counter() - s
    size = sum(model_bytes(m) for m, *_ in st)
    nmax = max(n for _, n, *_ in st)
    out[d.name] = dict(train_s=train_s, lat_p50_ms=1e3 * float(np.median(lat)), lat_p95_ms=1e3 * float(np.percentile(lat, 95)),
                       throughput_fps=len(bat_idx) / el, size_kb=size / 1024,
                       deferred=(st[-1][4] if len(st) > 1 else None),
                       state_bytes_per_flow=int(10 * 8 * min(nmax, 32)) if d.__dict__.get("w") != FULL else None)
    print(d.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[d.name].items()}, flush=True)
json.dump(out, open(f"{RESULTS}/cost_bench.json", "w"), indent=1)
