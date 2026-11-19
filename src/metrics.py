"""Evaluation metrics: accuracy, time and prevalence-adjusted measures, block bootstrap, paired tests."""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (f1_score, matthews_corrcoef, average_precision_score, recall_score,
                             confusion_matrix)


def accuracy_metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(macro_f1=f1_score(y, pred, average="macro", zero_division=0),
               recall_attack=tp / max(tp + fn, 1), recall_benign=tn / max(tn + fp, 1),
               fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1),
               mcc=matthews_corrcoef(y, pred) if len(set(pred)) > 1 and len(set(y)) > 1 else 0.0,
               accuracy=(tp + tn) / len(y))
    if score is not None and len(set(y)) > 1:
        out["pr_auc"] = average_precision_score(y, score)
    return out


def at_prevalence(y, pred, score, prev):
    """Expected precision / F1 / PR-AUC if attacks made up `prev` of traffic.
    Benign flows are re-weighted so the test set has the requested attack prevalence;
    TPR and FPR are unchanged, precision falls as attacks become rare."""
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    w0 = (n1 * (1 - prev) / prev) / max(n0, 1)
    w = np.where(y == 1, 1.0, w0)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = w0 * ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(tp + fp, 1e-12)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    out = dict(prevalence=prev, precision=prec, recall=rec, f1_attack=f1)
    if score is not None:
        out["pr_auc"] = average_precision_score(y, score, sample_weight=w)
    return out


def time_metrics(y, pred, dec, t0=None, seg=None, latency=0.0):
    """Per-flow time-to-decision for attack flows, and attack-onset time-to-first-alert per capture."""
    atk = y == 1
    det = atk & (pred == 1)
    out = dict(ttd_mean=float(np.mean(dec[det]) + latency) if det.any() else np.nan,
               ttd_median=float(np.median(dec[det]) + latency) if det.any() else np.nan,
               ttd_p95=float(np.percentile(dec[det], 95) + latency) if det.any() else np.nan,
               decision_mean_all_attacks=float(np.mean(dec[atk]) + latency) if atk.any() else np.nan)
    if t0 is not None and seg is not None:
        onset = {}
        for s in np.unique(seg[atk]):
            m = atk & (seg == s)
            first_attack = t0[m].min()
            alerts = (t0 + dec)[m & (pred == 1)]
            onset[str(s)] = float(alerts.min() - first_attack + latency) if len(alerts) else None
        out["onset_to_first_alert"] = onset
    return out


def block_bootstrap(y, pred, blocks, fn=lambda y, p: f1_score(y, p, average="macro", zero_division=0), B=1000, seed=0):
    """95% CI by resampling whole time blocks (stratified by capture segment), not single flows."""
    rng = np.random.default_rng(seed)
    segs = {}
    for b in np.unique(blocks):
        segs.setdefault(b.rsplit("|", 2)[0], []).append(b)
    idx_of = {b: np.where(blocks == b)[0] for b in np.unique(blocks)}
    vals = []
    for _ in range(B):
        pick = []
        for s, bl in segs.items():
            for b in rng.choice(bl, len(bl), replace=True):
                pick.append(idx_of[b])
        ii = np.concatenate(pick)
        vals.append(fn(y[ii], pred[ii]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def composite_blocks(blocks):
    """Block j of every segment pooled -> composite block j (each holds benign and attack flows)."""
    return np.array([b.rsplit("|", 1)[1] for b in blocks])


def paired_test(y, pa, pb, blocks):
    """Wilcoxon signed-rank on per-composite-block macro-F1 differences (exploratory: blocks are
    contiguous time slices, not independent sessions). Returns p, rank-biserial r, median diff."""
    cb = composite_blocks(blocks)
    da, db = [], []
    for b in np.unique(cb):
        m = cb == b
        da.append(f1_score(y[m], pa[m], average="macro", zero_division=0))
        db.append(f1_score(y[m], pb[m], average="macro", zero_division=0))
    d = np.array(da) - np.array(db)
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(p=1.0, r=0.0, median_diff=0.0, n_blocks=len(d))
    p = wilcoxon(nz).pvalue if len(nz) >= 1 else 1.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    r = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return dict(p=float(p), r=float(r), median_diff=float(np.median(d)), n_blocks=len(d))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * pvals[i])
        adj[i] = min(1.0, running)
    return adj
