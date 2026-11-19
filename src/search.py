"""Configuration search: NSGA-II (proposed), random search and single-objective GA (baselines).

A configuration theta = (cascade on/off, N1, T1, Stage-1 depth, N2, T2, Stage-2 family and
hyper-parameters, confidence thresholds, feature subset). Objectives, all minimised and all
measured on the validation split only:
  f1 = 1 - macro-F1
  f2 = mean decision time over attack flows (s)       -> time-to-detect
  f3 = normalised resource cost (latency, model size, CPU per flow)
"""
import json
import time
import numpy as np

from features import FEATURES
from models import Cascade, FAMILIES, macro_f1, make_model, measure_cost

N1S, T1S = [1, 2, 3, 4], [0.05, 0.1, 0.5]
N2S, T2S = [2, 3, 4, 8, 16, 32], [0.05, 0.1, 0.5, 2.0]
FAMS = list(FAMILIES)
NF = len(FEATURES)


def random_cfg(rng):
    cfg = dict(cascade=int(rng.integers(2)), N1=int(rng.choice(N1S)), T1=float(rng.choice(T1S)),
               d1=int(rng.integers(2, 6)), N2=int(rng.choice(N2S)), T2=float(rng.choice(T2S)),
               fam=str(rng.choice(FAMS)), hp=int(rng.integers(4)),
               tau_hi=float(rng.uniform(0.6, 0.999)), tau_lo=float(rng.uniform(0.001, 0.4)),
               mask=[int(b) for b in (rng.random(NF) < 0.5)])
    return repair(cfg, rng)


def repair(cfg, rng):
    if sum(cfg["mask"]) < 2:
        for j in rng.choice(NF, 2, replace=False):
            cfg["mask"][int(j)] = 1
    if cfg["N2"] <= cfg["N1"]:
        cfg["N2"] = min([n for n in N2S if n > cfg["N1"]] or [32])
    if cfg["T2"] < cfg["T1"]:
        cfg["T2"] = cfg["T1"]
    cfg["cols"] = [i for i, b in enumerate(cfg["mask"]) if b]
    return cfg


def key(cfg):
    c = {k: v for k, v in cfg.items() if k != "cols"}
    if not c["cascade"]:
        c.update(N1=0, T1=0, d1=0, tau_hi=0, tau_lo=0)
    return json.dumps(c, sort_keys=True)


class Evaluator:
    def __init__(self, fs, split, ref):
        self.fs, self.split, self.ref = fs, split, ref
        self.cache, self.log = {}, []
        self.vidx = split["val"]
        self.atk = self.fs.y[self.vidx] == 1

    def __call__(self, cfg):
        k = key(cfg)
        if k in self.cache:
            return self.cache[k]
        t = time.time()
        cas = Cascade(cfg).fit(self.fs, self.split["train"])
        pred, dec, _ = cas.decide(self.fs, self.vidx)
        f1 = 1 - macro_f1(self.fs.y[self.vidx], pred)
        f2 = float(np.mean(dec[self.atk]))
        cost = cas.cost(self.fs, self.vidx, self.ref)
        res = dict(f=(f1, f2, cost["f3"]), cost=cost, secs=time.time() - t)
        self.cache[k] = res
        self.log.append(dict(cfg={kk: vv for kk, vv in cfg.items() if kk != "cols"}, **res))
        return res


def reference_cost(fs, split):
    """Cost of the reference detector (RF-100 on all features at N=32, T=2 s) used for normalising f3."""
    cols = list(range(NF))
    m = make_model("RF", dict(n_estimators=100, max_depth=None)).fit(fs.X(32, 2.0)[split["train"]][:, cols], fs.y[split["train"]])
    return measure_cost(m, fs.pk[split["val"]], fs.proto[split["val"]], 32, 2.0, cols)


# ---------------- NSGA-II ----------------
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(F, idx):
    d = np.zeros(len(idx))
    A = np.array([F[i] for i in idx], float)
    for m in range(A.shape[1]):
        o = np.argsort(A[:, m])
        d[o[0]] = d[o[-1]] = np.inf
        span = A[o[-1], m] - A[o[0], m] or 1.0
        for j in range(1, len(idx) - 1):
            d[o[j]] += (A[o[j + 1], m] - A[o[j - 1], m]) / span
    return d


def crossover(a, b, rng):
    c = {}
    for k in a:
        if k == "cols":
            continue
        if k == "mask":
            c[k] = [x if rng.random() < 0.5 else y for x, y in zip(a[k], b[k])]
        else:
            c[k] = a[k] if rng.random() < 0.5 else b[k]
    return c


def mutate(c, rng, p=0.15):
    c = json.loads(json.dumps({k: v for k, v in c.items() if k != "cols"}))
    if rng.random() < p: c["cascade"] = 1 - c["cascade"]
    if rng.random() < p: c["N1"] = int(rng.choice(N1S))
    if rng.random() < p: c["T1"] = float(rng.choice(T1S))
    if rng.random() < p: c["d1"] = int(rng.integers(2, 6))
    if rng.random() < p: c["N2"] = int(rng.choice(N2S))
    if rng.random() < p: c["T2"] = float(rng.choice(T2S))
    if rng.random() < p: c["fam"] = str(rng.choice(FAMS))
    if rng.random() < p: c["hp"] = int(rng.integers(4))
    if rng.random() < p: c["tau_hi"] = float(np.clip(c["tau_hi"] + rng.normal(0, 0.08), 0.5, 0.999))
    if rng.random() < p: c["tau_lo"] = float(np.clip(c["tau_lo"] + rng.normal(0, 0.08), 0.001, 0.49))
    for j in range(NF):
        if rng.random() < 1.0 / NF:
            c["mask"][j] = 1 - c["mask"][j]
    return repair(c, rng)


def nsga2(ev, pop_size=24, gens=19, seed=0, objectives=(0, 1, 2)):
    """Standard NSGA-II (Deb et al., 2002). objectives selects which of (f1,f2,f3) are optimised,
    so the same loop also runs the single-objective GA baseline (objectives=(0,))."""
    rng = np.random.default_rng(seed)
    pop = [random_cfg(rng) for _ in range(pop_size)]
    fit = [tuple(ev(c)["f"][o] for o in objectives) for c in pop]
    history = []
    for g in range(gens):
        fronts = nondominated_sort(fit)
        rank = np.empty(len(pop), int)
        crowd = np.empty(len(pop))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding(fit, fr)

        def tour():
            i, j = rng.integers(len(pop), size=2)
            if rank[i] != rank[j]:
                return pop[i] if rank[i] < rank[j] else pop[j]
            return pop[i] if crowd[i] > crowd[j] else pop[j]

        kids = [mutate(crossover(tour(), tour(), rng), rng) for _ in range(pop_size)]
        kfit = [tuple(ev(c)["f"][o] for o in objectives) for c in kids]
        allp, allf = pop + kids, fit + kfit
        fronts = nondominated_sort(allf)
        new = []
        for fr in fronts:
            if len(new) + len(fr) <= pop_size:
                new += fr
            else:
                cd = crowding(allf, fr)
                new += [fr[i] for i in np.argsort(-cd)[:pop_size - len(new)]]
                break
        pop, fit = [allp[i] for i in new], [allf[i] for i in new]
        history.append(len(ev.log))
    return pop, fit


def random_search(ev, budget, seed=0):
    rng = np.random.default_rng(seed)
    cfgs = [random_cfg(rng) for _ in range(budget)]
    for c in cfgs:
        ev(c)
    return cfgs


def pareto_front(points):
    pts = [tuple(p) for p in points]
    return [i for i, p in enumerate(pts) if not any(dominates(q, p) for j, q in enumerate(pts) if j != i)]


def hypervolume(points, ref):
    """Exact hypervolume for minimisation in 2 or 3 dimensions (slicing method)."""
    P = np.array([p for p in points if all(x < r for x, r in zip(p, ref))], float)
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 2:
        P = P[np.argsort(P[:, 0])]
        hv, best = 0.0, ref[1]
        for x, y in P:
            if y < best:
                hv += (ref[0] - x) * (best - y)
                best = y
        return hv
    P = P[np.argsort(P[:, 2])]
    hv = 0.0
    for i in range(len(P)):
        z_next = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        if z_next > P[i, 2]:
            hv += hypervolume(P[:i + 1, :2], ref[:2]) * (z_next - P[i, 2])
    return hv
