"""Experiments A-D, baselines, ablations and error analysis (run after src/run_search.py).

Everything is trained on the CIC-DDoS2019 training split; configurations were chosen on the
validation split by the search; test results below are computed once on held-out data.
"""
import glob
import json
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from sklearn.ensemble import IsolationForest, ExtraTreesClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

from data import FlowSet, chrono_split, main_split, FULL, N_GRID, T_GRID, MAIN as MAIN_DATA, RESULTS
from features import FEATURE_NAMES, GROUPS, window_features
from models import Cascade, make_model, measure_cost, proba, FAMILIES
from metrics import accuracy_metrics, at_prevalence, time_metrics, block_bootstrap, paired_test, holm
from search import pareto_front, hypervolume

RES = {}
cic, ext = FlowSet(MAIN_DATA), FlowSet("external")
SP, BLK = main_split(cic, nblocks=10)
SPE, BLKE = chrono_split(ext, nblocks=10)
TR, VA, TE = SP["train"], SP["val"], SP["test"]
TRV = np.r_[TR, VA]
ALL_E = np.arange(len(ext.y))
REF = json.load(open(f"{RESULTS}/reference_cost.json"))
MAIN = "enterprise"          # operating point used for paired tests, transfer, ablations and error analysis


def save():
    json.dump(RES, open(f"{RESULTS}/experiments.json", "w"), indent=1, default=float)


# ------------------------------------------------------------------ choose operating points
def load_search():
    runs = {}
    for f in sorted(glob.glob(f"{RESULTS}/search_*_s*.json")):
        d = json.load(open(f))
        runs.setdefault(d["method"], []).append(d)
    return runs


SCALE = (0.05, 2.0, 3.0)        # normalisation for hypervolume: f1/0.05, f2/2 s, f3/3; ref point (1,1,1)


def norm(f):
    return tuple(min(x / s, 1.0) for x, s in zip(f, SCALE))


def select_profiles(runs):
    """Union of NSGA-II fronts (validation objectives) -> one configuration per deployment profile."""
    log = [e for r in runs["nsga2"] for e in r["log"]]
    F = [tuple(e["f"]) for e in log]
    front = [log[i] for i in pareto_front(F)]
    prof = {}
    # Edge gateway: cheapest configuration that keeps macro-F1 >= 0.995, decides within 100 ms, model <= 64 KB
    c = [e for e in front if e["f"][0] <= 0.005 and e["f"][1] <= 0.1 and e["cost"]["size"] <= 64_000]
    prof["edge"] = min(c or front, key=lambda e: (e["f"][2], e["f"][1]))
    # Enterprise server: earliest decision with macro-F1 >= 0.998 and model <= 1 MB
    c = [e for e in front if e["f"][0] <= 0.002 and e["cost"]["size"] <= 1_000_000]
    prof["enterprise"] = min(c or front, key=lambda e: (round(e["f"][0], 3), e["f"][1], e["f"][2]) if not c else (e["f"][1], e["f"][2]))
    # Cloud: no resource limit; best accuracy, then earliest
    prof["cloud"] = min(front, key=lambda e: (round(e["f"][0], 4), e["f"][1], e["f"][2]))
    return front, prof


def cfg_of(entry):
    c = dict(entry["cfg"])
    c["cols"] = [i for i, b in enumerate(c["mask"]) if b]
    return c


# ------------------------------------------------------------------ uniform evaluation wrappers
class Fixed:
    """A single-model detector at one window (or the complete flow)."""

    def __init__(self, name, fam, hp, window, cols=None, sampler=None):
        self.name, self.fam, self.hp, self.w = name, fam, hp, window
        self.cols = cols if cols is not None else list(range(len(FEATURE_NAMES)))
        self.sampler = sampler

    def fit(self, fs, idx):
        idx = self.sampler(fs, idx, self.w) if self.sampler else idx
        self.m = make_model(self.fam, self.hp).fit(fs.X(*self.w)[idx][:, self.cols], fs.y[idx])
        return self

    def run(self, fs, idx):
        s = proba(self.m, fs.X(*self.w)[idx][:, self.cols])
        return (s >= 0.5).astype(int), s, fs.dec_time(*self.w)[idx]

    def cost(self, fs, idx):
        n, t = (32, np.inf) if self.w == FULL else self.w
        return measure_cost(self.m, fs.pk[idx], fs.proto[idx], n, t, self.cols)


class _ScoreWrap:
    """Scaler + one-class model behind a predict_proba-like call, for timing and size."""

    def __init__(self, sc, m):
        self.sc, self.m = sc, m

    def predict_proba(self, X):
        return self.m.decision_function(self.sc.transform(X))


class OneClass:
    """Unsupervised early detector trained on benign flows only (E1-style)."""

    def __init__(self, name, kind, window):
        self.name, self.kind, self.w = name, kind, window

    def fit(self, fs, idx):
        b = idx[fs.y[idx] == 0]
        self.sc = RobustScaler().fit(fs.X(*self.w)[b])
        Xb = self.sc.transform(fs.X(*self.w)[b])
        self.m = (OneClassSVM(kernel="rbf", nu=0.05, gamma="scale") if self.kind == "ocsvm"
                  else IsolationForest(n_estimators=100, contamination=0.1, random_state=0)).fit(Xb)
        return self

    def run(self, fs, idx):
        X = self.sc.transform(fs.X(*self.w)[idx])
        s = -self.m.decision_function(X)
        return (self.m.predict(X) == -1).astype(int), s, fs.dec_time(*self.w)[idx]

    def cost(self, fs, idx):
        return measure_cost(_ScoreWrap(self.sc, self.m), fs.pk[idx], fs.proto[idx], self.w[0], self.w[1], list(range(len(FEATURE_NAMES))))


class Proposed:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg

    def fit(self, fs, idx):
        self.c = Cascade(self.cfg).fit(fs, idx)
        return self

    def run(self, fs, idx):
        pred, dec, deferred = self.c.decide(fs, idx)
        c = self.cfg
        s2 = proba(self.c.m2, fs.X(c["N2"], c["T2"])[idx][:, c["cols"]])
        if c["cascade"]:
            s1 = proba(self.c.m1, fs.X(c["N1"], c["T1"])[idx][:, self.c.c1])
            s = np.where(deferred, s2, s1)
        else:
            s = s2
        return pred, s, dec

    def cost(self, fs, idx):
        k = self.c.cost(fs, idx, REF)
        return dict(latency=k["latency"], cpu_per_flow=k["cpu_per_flow"], size=k["size"], deferred=k["deferred"])


class Hybrid:
    """Extension added in Phase 3: the cascade's alert OR a benign-trained Isolation Forest alert.
    Aimed at attack types never seen in training (supervised models cannot flag them)."""

    def __init__(self, name, cfg, window=(8, 0.5)):
        self.name, self.p, self.o = name, Proposed(name, cfg), OneClass(name, "iforest", window)

    def fit(self, fs, idx):
        self.p.fit(fs, idx)
        self.o.fit(fs, idx)
        return self

    def run(self, fs, idx):
        p1, s1, d1 = self.p.run(fs, idx)
        p2, s2, d2 = self.o.run(fs, idx)
        pred = np.maximum(p1, p2)
        dec = np.where(p1 == 1, d1, np.where(p2 == 1, d2, np.maximum(d1, d2)))
        return pred, np.maximum(s1, (s2 > 0).astype(float)), dec

    def cost(self, fs, idx):
        a, b = self.p.cost(fs, idx), self.o.cost(fs, idx)
        return dict(latency=a["latency"] + b["latency"], cpu_per_flow=a["cpu_per_flow"] + b["cpu_per_flow"], size=a["size"] + b["size"])


def nearmiss(fs, idx, w, k=3):
    """NearMiss-1 under-sampling (as used by Ebrahem et al., 2026): keep the majority-class samples
    whose mean distance to their k nearest minority samples is smallest."""
    X = RobustScaler().fit_transform(fs.X(*w)[idx])
    y = fs.y[idx]
    maj = 1 if (y == 1).sum() > (y == 0).sum() else 0
    mi, ma = np.where(y != maj)[0], np.where(y == maj)[0]
    d, _ = NearestNeighbors(n_neighbors=min(k, len(mi))).fit(X[mi]).kneighbors(X[ma])
    keep = ma[np.argsort(d.mean(1))[:len(mi)]]
    return idx[np.r_[mi, keep]]


def random_under(fs, idx, w, seed=0):
    rng = np.random.default_rng(seed)
    y = fs.y[idx]
    maj = 1 if (y == 1).sum() > (y == 0).sum() else 0
    mi, ma = np.where(y != maj)[0], np.where(y == maj)[0]
    return idx[np.r_[mi, rng.choice(ma, len(mi), replace=False)]]


def et_top(k=10):
    et = ExtraTreesClassifier(n_estimators=100, random_state=0, class_weight="balanced").fit(cic.full[TR], cic.y[TR])
    return list(np.argsort(-et.feature_importances_)[:k])


def evaluate(det, fit_fs=cic, fit_idx=TR, test_sets=None, with_cost=True):
    det.fit(fit_fs, fit_idx)
    out = {}
    test_sets = test_sets or {"cic_test": (cic, TE, BLK), "external": (ext, ALL_E, None)}
    for tname, (fs, idx, blk) in test_sets.items():
        pred, s, dec = det.run(fs, idx)
        y = fs.y[idx]
        r = accuracy_metrics(y, pred, s)
        r.update(time_metrics(y, pred, dec, fs.t0[idx], fs.seg[idx]))
        if getattr(det, "w", None) == FULL:      # also report the optimistic "decide at last packet" time
            r["flow_end"] = time_metrics(y, pred, fs.dec_time("full", "end")[idx], fs.t0[idx], fs.seg[idx])
        r["recall_by_type"] = {t: float((pred[fs.type[idx] == t] == 1).mean()) for t in np.unique(fs.type[idx]) if t != "BENIGN"}
        r["prevalence"] = [at_prevalence(y, pred, s, p) for p in (0.01, 0.10, 0.30)]
        if blk is not None:
            r["macro_f1_ci"] = block_bootstrap(y, pred, blk[idx], B=500)
        out[tname] = r
        out.setdefault("_pred", {})[tname] = pred.tolist()
    if with_cost:
        out["cost"] = det.cost(cic, VA)
    return out


def summary(name, r):
    c = r["cic_test"]
    e = r.get("external", {})
    print(f"{name:34s} F1 {c['macro_f1']:.4f} FPR {c['fpr']:.4f} TTDmed {c['ttd_median']:.3f}s "
          f"ext F1 {e.get('macro_f1', float('nan')):.3f} size {r.get('cost', {}).get('size', 0)/1024:.1f}KB "
          f"lat {r.get('cost', {}).get('latency', 0)*1e3:.2f}ms", flush=True)


if __name__ == "__main__":
    runs = load_search()
    front, prof = select_profiles(runs)

    # ---------------- Experiment E: search comparison (validation objectives) ----------------
    E = {}
    for m, rr in runs.items():
        hvs = []
        for r in rr:
            F = [tuple(e["f"]) for e in r["log"]]
            pf = [norm(F[i]) for i in pareto_front(F)]
            hvs.append(hypervolume(pf, (1, 1, 1)))
            # anytime curve: hypervolume of everything evaluated so far
        E[m] = dict(hv=hvs, hv_mean=float(np.mean(hvs)), hv_sd=float(np.std(hvs)),
                    front_size=[len(pareto_front([tuple(e["f"]) for e in r["log"]])) for r in rr],
                    unique_evals=[r["n_unique"] for r in rr])
        curves = []
        for r in rr:
            F = [tuple(e["f"]) for e in r["log"]]
            pts = list(range(50, len(F) + 1, 50)) + [len(F)]
            curves.append([(p, hypervolume([norm(F[i]) for i in pareto_front(F[:p])], (1, 1, 1))) for p in pts])
        E[m]["anytime"] = curves
    RES["search"] = E
    RES["front"] = [dict(f=e["f"], cost=e["cost"], cfg=e["cfg"]) for e in front]
    RES["profiles"] = {k: dict(f=v["f"], cost=v["cost"], cfg=v["cfg"]) for k, v in prof.items()}
    print({m: (round(v["hv_mean"], 4), round(v["hv_sd"], 4)) for m, v in E.items()}, flush=True)
    save()

    # ---------------- proposed operating points + baselines on held-out data ----------------
    dets = [Proposed(f"Proposed ({k})", cfg_of(v)) for k, v in prof.items()]
    best_rand = min((e for r in runs["random"] for e in r["log"]), key=lambda e: (round(e["f"][0], 3), e["f"][1], e["f"][2]))
    best_ga = min((e for r in runs["ga_f1"] for e in r["log"]), key=lambda e: (round(e["f"][0], 3), e["f"][1], e["f"][2]))
    dets += [Hybrid(f"Proposed ({MAIN}) + IF fallback", cfg_of(prof[MAIN]))]
    dets += [Proposed("Random-search best", cfg_of(best_rand)), Proposed("GA (F1 only) best", cfg_of(best_ga))]
    top10 = et_top(10)
    early = (8, 0.5)
    dets += [
        Fixed("Full-flow RF-100", "RF", dict(n_estimators=100, max_depth=None), FULL),
        Fixed("Full-flow GBDT (HGB)", "HGB", dict(max_iter=200, max_depth=None), FULL),
        Fixed("Full-flow MLP (32,16)", "MLP", dict(hidden_layer_sizes=(32, 16)), FULL),
        Fixed("P2: CNB + NearMiss", "CNB", {}, FULL, sampler=nearmiss),
        Fixed("P2: kNN + NearMiss", "kNN", dict(n_neighbors=5), FULL, sampler=nearmiss),
        Fixed("P2: RF + NearMiss", "RF", dict(n_estimators=100, max_depth=None), FULL, sampler=nearmiss),
        Fixed("P2: LR + NearMiss", "LR", dict(C=1.0), FULL, sampler=nearmiss),
        Fixed("E5: ET-top10 + RF", "RF", dict(n_estimators=100, max_depth=None), FULL, cols=top10),
        Fixed("E5: ET-top10 + LR", "LR", dict(C=1.0), FULL, cols=top10),
        Fixed("E5: ET-top10 + NB", "GNB", {}, FULL, cols=top10),
        OneClass("E1: One-Class SVM (N=8,T=0.5s)", "ocsvm", early),
        OneClass("E1: Isolation Forest (N=8,T=0.5s)", "iforest", early),
        Fixed("Fixed early RF-100 (N=8,T=0.5s)", "RF", dict(n_estimators=100, max_depth=None), early),
        Fixed("Fixed early DT-8 (N=2,T=0.1s)", "DT", dict(max_depth=8), (2, 0.1)),
    ]
    B = {}
    for d in dets:
        B[d.name] = evaluate(d)
        summary(d.name, B[d.name])
    # paired block tests: main proposed vs each baseline (Holm-corrected)
    main = f"Proposed ({MAIN})"
    yte = cic.y[TE]
    pm = np.array(B[main]["_pred"]["cic_test"])
    names = [n for n in B if n != main]
    tests = [paired_test(yte, pm, np.array(B[n]["_pred"]["cic_test"]), BLK[TE]) for n in names]
    adj = holm(np.array([t["p"] for t in tests]))
    for n, t, a in zip(names, tests, adj):
        t["p_holm"] = float(a)
        B[n]["paired_vs_main"] = t
    for v in B.values():
        v.pop("_pred", None)
    RES["baselines"] = B
    save()

    # reverse transfer: train on the external set, test on CIC (proposed edge config + RF baselines)
    rev = {}
    for d in [Proposed(f"Proposed ({MAIN})", cfg_of(prof[MAIN])),
              Fixed("Full-flow RF-100", "RF", dict(n_estimators=100, max_depth=None), FULL),
              Fixed("Fixed early RF-100 (N=8,T=0.5s)", "RF", dict(n_estimators=100, max_depth=None), early)]:
        r = evaluate(d, ext, SPE["train"], {"external_test": (ext, SPE["test"], BLKE), "cic_all": (cic, np.arange(len(cic.y)), None)}, with_cost=False)
        r.pop("_pred", None)
        rev[d.name] = r
        print("EXT->CIC", d.name, round(r["external_test"]["macro_f1"], 4), round(r["cic_all"]["macro_f1"], 4), flush=True)
    RES["reverse_transfer"] = rev
    save()

    # ---------------- Experiment A: accuracy and time vs window ----------------
    A = []
    for n in N_GRID:
        for t in T_GRID:
            for fam, hp in (("RF", dict(n_estimators=50, max_depth=10)), ("DT", dict(max_depth=6))):
                d = Fixed(f"{fam} N={n} T={t}", fam, hp, (n, t))
                r = evaluate(d, with_cost=False)
                A.append(dict(N=n, T=t, model=fam, macro_f1=r["cic_test"]["macro_f1"],
                              ext_macro_f1=r["external"]["macro_f1"],
                              recall_by_type=r["cic_test"]["recall_by_type"], fpr=r["cic_test"]["fpr"],
                              ttd_median=r["cic_test"]["ttd_median"], ttd_mean=r["cic_test"]["ttd_mean"],
                              decision_mean_all=r["cic_test"]["decision_mean_all_attacks"],
                              onset=r["cic_test"]["onset_to_first_alert"]))
    RES["window_sweep"] = A
    save()
    print("A done", flush=True)

    # ---------------- Experiment C: leave-one-attack-type-out and feature-group transfer ----------------
    C = {}
    pe = cfg_of(prof[MAIN])
    loao_types = sorted(set(cic.type[TR]) & set(cic.type[TE]) - {"BENIGN"})
    for typ in loao_types:
        tr = TR[cic.type[TR] != typ]
        te = TE[(cic.type[TE] == typ) | (cic.y[TE] == 0)]
        row = {}
        for d in [Proposed(f"Proposed ({MAIN})", pe), Hybrid(f"Proposed ({MAIN}) + IF fallback", pe),
                  Fixed("Full-flow RF-100", "RF", dict(n_estimators=100, max_depth=None), FULL),
                  Fixed("Fixed early RF-100 (N=8,T=0.5s)", "RF", dict(n_estimators=100, max_depth=None), early),
                  OneClass("E1: Isolation Forest (N=8,T=0.5s)", "iforest", early)]:
            r = evaluate(d, cic, tr, {"t": (cic, te, None)}, with_cost=False)["t"]
            row[d.name] = dict(recall_unseen=r["recall_attack"], fpr=r["fpr"], macro_f1=r["macro_f1"])
        C[typ] = row
        print("LOAO", typ, {k: round(v["recall_unseen"], 3) for k, v in row.items()}, flush=True)
    RES["leave_one_attack_out"] = C
    W2 = (pe["N2"], pe["T2"])
    G = {}
    for g, cols in list(GROUPS.items()) + [("all", list(range(len(FEATURE_NAMES))))]:
        for mode, cc in (("only", cols), ("without", [i for i in range(len(FEATURE_NAMES)) if i not in cols])):
            if g == "all" and mode == "without":
                continue
            d = Fixed(f"{g}-{mode}", "RF", dict(n_estimators=50, max_depth=10), W2, cols=cc)
            r = evaluate(d, with_cost=False)
            G[f"{g}|{mode}"] = dict(cic=r["cic_test"]["macro_f1"], ext=r["external"]["macro_f1"],
                                    ext_recall_attack=r["external"]["recall_attack"], ext_fpr=r["external"]["fpr"])
    RES["group_transfer"] = G
    save()
    print("C done", flush=True)

    # ---------------- Ablations on the main configuration ----------------
    AB = {}
    base = dict(pe)

    def run_cfg(name, cfg, **kw):
        r = evaluate(Proposed(name, cfg), **kw)
        AB[name] = dict(cic_f1=r["cic_test"]["macro_f1"], ext_f1=r["external"]["macro_f1"],
                        ttd_median=r["cic_test"]["ttd_median"], ttd_mean=r["cic_test"]["ttd_mean"],
                        decision_mean_all=r["cic_test"]["decision_mean_all_attacks"],
                        fpr=r["cic_test"]["fpr"], cost=r["cost"], recall_by_type=r["cic_test"]["recall_by_type"])
        print("ABL", name, {k: (round(v, 4) if isinstance(v, float) else "") for k, v in AB[name].items()}, flush=True)

    run_cfg(f"Full proposed ({MAIN})", base)
    if base["cascade"]:
        run_cfg("No cascade (Stage 2 only)", dict(base, cascade=0))
        s1 = dict(base, tau_hi=0.5, tau_lo=0.5)
        run_cfg("Stage 1 only (no deferral)", s1)
    else:
        casc = dict(base, cascade=1, N1=1, T1=0.05, d1=4, tau_hi=0.9, tau_lo=0.1)
        if casc["N2"] <= 1:
            casc["N2"] = 2
        run_cfg("Add cascade (Stage 1 DT-4 at N=1)", casc)
    for n2 in (2, 4, 8, 16, 32):
        if n2 > base.get("N1", 0) and n2 != base["N2"]:
            run_cfg(f"Stage-2 window N2={n2}", dict(base, N2=n2))
    for t2 in (0.05, 0.1, 0.5, 2.0):
        if t2 != base["T2"] and t2 >= base.get("T1", 0):
            run_cfg(f"Stage-2 window T2={t2}s", dict(base, T2=t2))
    allc = list(range(len(FEATURE_NAMES)))
    run_cfg("Feature selection off (all 31)", dict(base, cols=allc, mask=[1] * len(allc)))
    for g, cols in GROUPS.items():
        cc = [i for i in base["cols"] if i not in cols]
        if cc and len(cc) < len(base["cols"]):
            run_cfg(f"Drop group: {g}", dict(base, cols=cc))
    for fam in FAMILIES:
        if fam != base["fam"]:
            run_cfg(f"Stage-2 family: {fam}", dict(base, fam=fam, hp=1))
    if base["cascade"]:
        for d1 in (2, 3, 4, 5):
            if d1 != base["d1"]:
                run_cfg(f"Stage-1 depth d1={d1}", dict(base, d1=d1))
        run_cfg("Stricter gate (tau 0.02/0.98)", dict(base, tau_lo=0.02, tau_hi=0.98))
        run_cfg("Looser gate (tau 0.30/0.60)", dict(base, tau_lo=0.30, tau_hi=0.60))
        for n1 in (1, 2, 3, 4):
            if n1 != base["N1"] and n1 < base["N2"]:
                run_cfg(f"Stage-1 window N1={n1}", dict(base, N1=n1))
    RES["ablation"] = AB
    save()

    # balancing strategy (on a fixed early RF at the main configuration's Stage-2 window)
    BAL = {}
    for nm, fam_hp, samp in (("class weights", ("RF", dict(n_estimators=50, max_depth=10)), None),
                             ("random under-sampling", ("RF", dict(n_estimators=50, max_depth=10)), random_under),
                             ("NearMiss-1", ("RF", dict(n_estimators=50, max_depth=10)), nearmiss)):
        d = Fixed(nm, fam_hp[0], fam_hp[1], W2, cols=base["cols"], sampler=samp)
        r = evaluate(d, with_cost=False)
        BAL[nm] = dict(cic_f1=r["cic_test"]["macro_f1"], fpr=r["cic_test"]["fpr"], ext_f1=r["external"]["macro_f1"],
                       ext_fpr=r["external"]["fpr"], prev1_precision=r["cic_test"]["prevalence"][0]["precision"])
    # no balancing: RF without class weights
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(50, max_depth=10, random_state=0, n_jobs=1).fit(cic.X(*W2)[TR][:, base["cols"]], cic.y[TR])
    for fs, idx, k in ((cic, TE, "cic"), (ext, ALL_E, "ext")):
        p = m.predict(fs.X(*W2)[idx][:, base["cols"]])
        a = accuracy_metrics(fs.y[idx], p)
        BAL.setdefault("no balancing", {})[f"{k}_f1"] = a["macro_f1"]
        BAL["no balancing"][f"{'fpr' if k == 'cic' else 'ext_fpr'}"] = a["fpr"]
    RES["balancing"] = BAL
    save()
    print("ablation done", flush=True)

    # ---------------- Error analysis ----------------
    ERR = {}
    d = Proposed(MAIN, pe).fit(cic, TR)
    pred, s, dec = d.run(ext, ALL_E)
    fp = (pred == 1) & (ext.y == 0)
    files = ext.file[ext.y == 0]
    ERR["external_fp_by_benign_source"] = {f: dict(n=int((files == f).sum()), fp_rate=float(fp[ext.y == 0][files == f].mean()))
                                           for f in np.unique(files) if (files == f).sum() >= 20}
    ERR["external_recall_by_attack"] = {t: float((pred[ext.type == t] == 1).mean()) for t in np.unique(ext.type) if t != "BENIGN"}
    pred_c, s_c, dec_c = d.run(cic, TE)
    ERR["cic_confusion_by_type"] = {t: dict(n=int((cic.type[TE] == t).sum()), predicted_attack=float((pred_c[cic.type[TE] == t] == 1).mean()))
                                    for t in np.unique(cic.type[TE])}
    # missed attack flows: how many packets did they have, and when would a later window catch them?
    miss = (pred_c == 0) & (cic.y[TE] == 1)
    ERR["cic_missed"] = dict(n=int(miss.sum()), types=dict(zip(*[x.tolist() for x in np.unique(cic.type[TE][miss], return_counts=True)])),
                             median_flow_len=float(np.median(cic.nlen[TE][miss])) if miss.any() else None)
    # permutation importance of Stage 2 on CIC test vs external (stability across datasets)
    W2c = pe["cols"]
    Xc, Xe = cic.X(*W2)[TE][:, W2c], ext.X(*W2)[:, W2c]
    pic = permutation_importance(d.c.m2, Xc, cic.y[TE], scoring="f1_macro", n_repeats=5, random_state=0)
    pie = permutation_importance(d.c.m2, Xe, ext.y, scoring="f1_macro", n_repeats=5, random_state=0)
    ERR["perm_importance"] = dict(features=[FEATURE_NAMES[i] for i in W2c], cic=pic.importances_mean.tolist(), ext=pie.importances_mean.tolist(),
                                  spearman=float(spearmanr(pic.importances_mean, pie.importances_mean).statistic))
    # slow-ramp evasion stress: stretch inter-packet gaps of attack test flows by a factor
    stress = {}
    for factor in (1, 2, 5, 10, 50):
        pk = cic.pk[TE].copy()
        atk = cic.y[TE] == 1
        pk[atk, :, 0] *= factor
        tmp = FlowSet.__new__(FlowSet)
        tmp.__dict__.update(cic.__dict__)
        tmp.pk, tmp._cache = pk, {}
        tmp.y, tmp.proto, tmp.full, tmp.type = cic.y[TE], cic.proto[TE], cic.full[TE], cic.type[TE]
        tmp.dur = cic.dur[TE] * np.where(atk, factor, 1)
        loc = np.arange(len(TE))
        row = {}
        for nm, det in ((f"Proposed ({MAIN})", d),
                        ("Fixed early RF-100 (N=8,T=0.5s)", Fixed("e", "RF", dict(n_estimators=100, max_depth=None), early).fit(cic, TR))):
            p_, _, dd = det.run(tmp, loc)
            row[nm] = dict(recall=float((p_[atk] == 1).mean()), ttd_median=float(np.median(dd[atk & (p_ == 1)])) if (atk & (p_ == 1)).any() else None)
        stress[str(factor)] = row
        print("stress", factor, row, flush=True)
    ERR["slow_ramp_stress"] = stress
    RES["errors"] = ERR
    save()
    print("ALL DONE", flush=True)
