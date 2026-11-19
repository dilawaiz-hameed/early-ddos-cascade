"""Model zoo, cost measurement and the confidence-gated two-stage cascade."""
import pickle
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

from features import window_features, FEATURES, MIN_PKTS

# Stage-2 model families and their small hyper-parameter grids (index chosen by the search)
FAMILIES = {
    "DT":  [dict(max_depth=d) for d in (4, 6, 8, 12)],
    "RF":  [dict(n_estimators=n, max_depth=d) for n, d in ((10, 6), (25, 8), (50, 10), (100, None))],
    "HGB": [dict(max_iter=n, max_depth=d) for n, d in ((25, 3), (50, 4), (100, 6), (200, None))],
    "kNN": [dict(n_neighbors=k) for k in (3, 5, 9, 15)],
    "LR":  [dict(C=c) for c in (0.1, 1.0, 10.0, 100.0)],
    "MLP": [dict(hidden_layer_sizes=h) for h in ((8,), (16,), (32,), (32, 16))],
}


def make_model(family, hp, seed=0):
    if family == "DT":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **hp)
    if family == "RF":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1, **hp)
    if family == "ET":
        return ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=1, **hp)
    if family == "HGB":   # gradient-boosted trees (stand-in for XGBoost, which could not be installed)
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **hp)
    if family == "kNN":
        return make_pipeline(RobustScaler(), KNeighborsClassifier(n_jobs=1, **hp))
    if family == "LR":
        return make_pipeline(RobustScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, **hp))
    if family == "MLP":
        return make_pipeline(RobustScaler(), MLPClassifier(max_iter=400, early_stopping=True, random_state=seed, **hp))
    if family == "CNB":
        return make_pipeline(MinMaxScaler(), ComplementNB(**hp))
    if family == "GNB":
        return make_pipeline(RobustScaler(), GaussianNB(**hp))
    raise ValueError(family)


def model_bytes(model):
    return len(pickle.dumps(model, protocol=4))


def proba(model, X):
    p = model.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else (np.full(len(X), float(model.classes_[0])))


def measure_cost(model, pk, proto, n, t, cols, reps_single=30, seed=0):
    """Return dict: per-flow CPU time in batch (feature extraction + inference),
    single-flow latency (features + predict for one flow, median), model size."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pk), min(2000, len(pk)), replace=False)
    t_a = time.process_time()
    X = window_features(pk[idx], proto[idx], n, t)[:, cols]
    t_b = time.process_time()
    model.predict_proba(X)
    t_c = time.process_time()
    lat = []
    for i in idx[:reps_single]:
        s = time.perf_counter()
        x = window_features(pk[i:i + 1], proto[i:i + 1], n, t)[:, cols]
        model.predict_proba(x)
        lat.append(time.perf_counter() - s)
    return dict(cpu_feat=(t_b - t_a) / len(idx), cpu_inf=(t_c - t_b) / len(idx),
                cpu_per_flow=(t_c - t_a) / len(idx), latency=float(np.median(lat)),
                size=model_bytes(model))


class Cascade:
    """Stage 1: tiny decision tree at an early window (N1, T1) that only releases confident flows.
    Stage 2: lightweight model at a later window (N2, T2) for the flows Stage 1 deferred.
    With cascade=False only Stage 2 runs (single-stage detector)."""

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed

    @staticmethod
    def stage1_cols(cfg):
        cols = [i for i in cfg["cols"] if MIN_PKTS[i] <= cfg["N1"]]
        return cols or [i for i in range(len(FEATURES)) if MIN_PKTS[i] <= cfg["N1"]]

    def fit(self, fs, idx):
        c = self.cfg
        y = fs.y[idx]
        X2 = fs.X(c["N2"], c["T2"])[idx][:, c["cols"]]
        self.m2 = make_model(c["fam"], FAMILIES[c["fam"]][c["hp"]], self.seed).fit(X2, y)
        if c["cascade"]:
            self.c1 = self.stage1_cols(c)
            X1 = fs.X(c["N1"], c["T1"])[idx][:, self.c1]
            self.m1 = make_model("DT", dict(max_depth=c["d1"]), self.seed).fit(X1, y)
        return self

    def decide(self, fs, idx):
        """Return (prediction, decision time in s, deferred mask)."""
        c = self.cfg
        p2 = proba(self.m2, fs.X(c["N2"], c["T2"])[idx][:, c["cols"]])
        d2 = fs.dec_time(c["N2"], c["T2"])[idx]
        if not c["cascade"]:
            return (p2 >= 0.5).astype(int), d2, np.ones(len(idx), bool)
        p1 = proba(self.m1, fs.X(c["N1"], c["T1"])[idx][:, self.c1])
        d1 = fs.dec_time(c["N1"], c["T1"])[idx]
        hi, lo = p1 >= c["tau_hi"], p1 <= c["tau_lo"]
        deferred = ~(hi | lo)
        pred = np.where(hi, 1, np.where(lo, 0, (p2 >= 0.5).astype(int)))
        dec = np.where(deferred, np.maximum(d1, d2), d1)
        return pred, dec, deferred

    def cost(self, fs, idx, ref):
        """Normalised resource cost f3 (equal weights on latency, memory, CPU)."""
        c = self.cfg
        k2 = measure_cost(self.m2, fs.pk[idx], fs.proto[idx], c["N2"], c["T2"], c["cols"])
        if c["cascade"]:
            k1 = measure_cost(self.m1, fs.pk[idx], fs.proto[idx], c["N1"], c["T1"], self.c1)
            _, _, deferred = self.decide(fs, idx)
            q = deferred.mean()
            lat = k1["latency"] + q * k2["latency"]
            cpu = k1["cpu_per_flow"] + q * k2["cpu_per_flow"]
            size = k1["size"] + k2["size"]
        else:
            q, lat, cpu, size = 1.0, k2["latency"], k2["cpu_per_flow"], k2["size"]
        buf = 10 * 8 * max(c["N2"], c.get("N1", 0))          # bytes of packet state kept per flow
        f3 = (lat / ref["latency"] + size / ref["size"] + cpu / ref["cpu_per_flow"]) / 3.0
        return dict(f3=f3, latency=lat, cpu_per_flow=cpu, size=size, buf=buf, deferred=float(q))


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)
