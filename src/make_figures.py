"""Draw every figure in the report from results/*.json (no numbers are typed in by hand)."""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import RESULTS, FIGURES

R = json.load(open(f"{RESULTS}/experiments.json"))
COST = json.load(open(f"{RESULTS}/cost_bench.json"))
MAIN = "enterprise"
PM = f"Proposed ({MAIN})"
FULLRUN = RESULTS != "results"

# validated reference palette (dataviz skill, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                     "lines.markersize": 5, "figure.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(f"{FIGURES}/{name}.png", facecolor="white")
    fig.savefig(f"{FIGURES}/{name}.pdf", facecolor="white")
    plt.close(fig)


# 1. accuracy and decision time vs window (Experiment A)
A = [a for a in R["window_sweep"] if a["model"] == "RF"]
Ns = sorted({a["N"] for a in A})
Ts = sorted({a["T"] for a in A})
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1))
for j, t in enumerate(Ts):
    rows = sorted([a for a in A if a["T"] == t], key=lambda a: a["N"])
    ax[0].plot([a["N"] for a in rows], [a["macro_f1"] for a in rows], "-o", color=C[j], label=f"T = {int(t*1000)} ms")
    ax[1].plot([a["N"] for a in rows], [1000 * a["decision_mean_all"] + 0.1 for a in rows], "-o", color=C[j], label=f"T = {int(t*1000)} ms")
ax[0].set(xscale="log", xlabel="Packets observed (N)", ylabel="Macro-F1 (CIC test)", title="Macro-F1 vs packets observed (RF)")
ax[0].set_xticks(Ns, [str(n) for n in Ns])
ax[1].set(xscale="log", yscale="log", xlabel="Packets observed (N)", ylabel="Mean decision time + 0.1 (ms, log)", title="Decision time vs packets observed")
ax[1].set_xticks(Ns, [str(n) for n in Ns])
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.13))
save(fig, "fig1_window_sweep")

# 2. recall by attack type vs N (T = 2 s, RF)
rows = sorted([a for a in A if a["T"] == max(Ts)], key=lambda a: a["N"])
types = sorted(rows[0]["recall_by_type"])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for j, t in enumerate(types):
    ax.plot([a["N"] for a in rows], [a["recall_by_type"][t] for a in rows], "-o", color=C[j], label=t)
ax.set(xscale="log", xlabel="Packets observed (N), T = 2 s", ylabel="Recall (CIC test)", title="Recall by attack family")
ax.set_xticks(Ns, [str(n) for n in Ns])
ax.legend(loc="lower right")
save(fig, "fig2_recall_by_type")

# 3. Pareto front (validation) and the three profiles
front = R["front"]
F = np.array([f["f"] for f in front])
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
ax[0].scatter(1000 * F[:, 1] + 0.1, 1 - F[:, 0], s=22, color=C[0], edgecolor="white", linewidth=0.8, label="NSGA-II front (union of seeds)")
ax[1].scatter(1000 * F[:, 1] + 0.1, F[:, 2], s=22, color=C[0], edgecolor="white", linewidth=0.8)
mk = {"edge": ("s", C[1]), "enterprise": ("D", C[2]), "cloud": ("^", C[6])}
for k, v in R["profiles"].items():
    m, c = mk[k]
    ax[0].scatter(1000 * v["f"][1] + 0.1, 1 - v["f"][0], s=60, marker=m, color=c, edgecolor=INK, linewidth=0.6, label=f"{k} profile", zorder=3)
    ax[1].scatter(1000 * v["f"][1] + 0.1, v["f"][2], s=60, marker=m, color=c, edgecolor=INK, linewidth=0.6, zorder=3)
ax[0].set(xscale="log", xlabel="Mean decision time + 0.1 (ms, log)", ylabel="Validation macro-F1", title="Accuracy vs time-to-detect")
ax[1].set(xscale="log", yscale="log", xlabel="Mean decision time + 0.1 (ms, log)", ylabel="Normalised cost f3 (log)", title="Cost vs time-to-detect")
ax[0].legend(loc="lower left", fontsize=7.5)
save(fig, "fig3_pareto")

# 4. hypervolume vs evaluations: NSGA-II vs random vs single-objective GA
fig, ax = plt.subplots(figsize=(5.2, 3.0))
lab = {"nsga2": "NSGA-II (proposed)", "random": "Random search", "ga_f1": "GA, F1 only"}
for j, m in enumerate(("nsga2", "random", "ga_f1")):
    cur = R["search"][m]["anytime"]
    L = min(len(c) for c in cur)
    x = np.array([c[i][0] for i in range(L) for c in cur[:1]])
    y = np.array([[c[i][1] for i in range(L)] for c in cur])
    ax.plot(x, y.mean(0), color=C[j], label=f"{lab[m]} (HV {R['search'][m]['hv_mean']:.3f})")
    if len(cur) > 1:
        ax.fill_between(x, y.min(0), y.max(0), color=C[j], alpha=0.15, linewidth=0)
ax.set(xlabel="Unique configurations evaluated", ylabel="Hypervolume (normalised)", title=f"Search efficiency ({len(R['search']['nsga2']['anytime'])} seed{'s, mean and min-max' if len(R['search']['nsga2']['anytime']) > 1 else ''})")
ax.legend(loc="lower right", fontsize=7.5)
save(fig, "fig4_hypervolume")

# 5. baselines: in-dataset vs cross-dataset macro-F1
B = R["baselines"]
names = list(B)
fig, ax = plt.subplots(figsize=(8.2, 5.2))
y = np.arange(len(names))
ax.barh(y + 0.2, [B[n]["cic_test"]["macro_f1"] for n in names], height=0.38, color=C[0], label="CIC-DDoS2019 test (in-dataset)")
ax.barh(y - 0.2, [B[n]["external"]["macro_f1"] for n in names], height=0.38, color=C[1], label="External set (cross-dataset)")
for i, n in enumerate(names):
    lo, hi = B[n]["cic_test"].get("macro_f1_ci", (None, None))
    if lo is not None:
        ax.plot([lo, hi], [i + 0.2, i + 0.2], color=INK, linewidth=1)
ax.set_yticks(y, names, fontsize=7.5)
ax.invert_yaxis()
ax.set(xlabel="Macro-F1 (black line: 95% block-bootstrap CI)", xlim=(0, 1.02), title="Macro-F1 in-dataset vs cross-dataset")
ax.legend(loc="lower right", fontsize=7.5)
save(fig, "fig5_baselines_f1")

# 6. time-to-detect vs cost (latency) for the main detectors
fig, ax = plt.subplots(figsize=(6.4, 3.6))
keys = [k for k in COST if k in B]
OFF_FULL = {"E1: Isolation Forest (N=8,T=0.5s)": (2, 9), "Fixed early RF-100 (N=8,T=0.5s)": (-75, -13)}
OFF = {"Full-flow MLP (32,16)": (-40, 9), "P2: kNN + NearMiss": (4, -14), "Full-flow GBDT (HGB)": (6, -14),
       "Full-flow RF-100": (-40, -14), "Proposed (cloud)": (8, 14), "Fixed early DT-8 (N=2,T=0.1s)": (8, -12),
       "Proposed (enterprise)": (6, 6), "E1: Isolation Forest (N=8,T=0.5s)": (-30, 9),
       "Fixed early RF-100 (N=8,T=0.5s)": (-60, -14)}
for j, k in enumerate(keys):
    ttd = B[k]["cic_test"]["ttd_median"] * 1000 + 0.1
    lat = COST[k]["lat_p50_ms"]
    col = C[2] if k.startswith("Proposed") else (C[1] if "Full-flow" in k or "P2" in k else C[0])
    ax.scatter(lat, ttd, s=30 + 3 * np.sqrt(COST[k]["size_kb"]), color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.annotate(k.replace(" (N=8,T=0.5s)", "").replace(" (N=2,T=0.1s)", ""), (lat, ttd), fontsize=6.5, color=INK2,
                xytext=OFF.get(k, (6, 5)) if not FULLRUN else OFF_FULL.get(k, (6, 5) if j % 2 == 0 else (6, -11)), textcoords="offset points")
ax.set_ylim(0.05, 3e5)
ax.set(xscale="log", yscale="log", xlabel="Single-flow latency p50, one CPU core (ms, log)",
       ylabel="Median time-to-detect + 0.1 (ms, log)", title="Time-to-detect vs per-flow latency (marker area ~ model size)")
ax.set_xticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
ax.minorticks_off()
save(fig, "fig6_time_cost")

# 7. prevalence: precision at 1%, 10%, 30% attack share
sel = [PM, f"{PM} + IF fallback", "Full-flow RF-100", "Fixed early RF-100 (N=8,T=0.5s)", "E1: Isolation Forest (N=8,T=0.5s)", "P2: kNN + NearMiss"]
sel = [s for s in sel if s in B]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
w = 0.8 / len(sel)
for j, s in enumerate(sel):
    vals = [p["precision"] for p in B[s]["cic_test"]["prevalence"]]
    ax.bar(np.arange(3) + j * w, vals, width=w * 0.92, color=C[j], label=s)
ax.set_xticks(np.arange(3) + 0.4 - w / 2, ["1% attacks", "10% attacks", "30% attacks"])
ax.set(ylabel="Expected precision", ylim=(0, 1.05), title="Precision when attacks are rare (CIC test, reweighted)")
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
save(fig, "fig7_prevalence")

# 8. transfer: leave-one-attack-out recall
L = R["leave_one_attack_out"]
types = list(L)
dets = list(L[types[0]])
fig, ax = plt.subplots(figsize=(6.4, 2.9))
w = 0.8 / len(dets)
for j, d in enumerate(dets):
    ax.bar(np.arange(len(types)) + j * w, [L[t][d]["recall_unseen"] for t in types], width=w * 0.92, color=C[j], label=d)
ax.set_xticks(np.arange(len(types)) + 0.4 - w / 2, types)
ax.set_xlabel("Attack family left out of training")
ax.set(ylabel="Recall on the unseen type", ylim=(0, 1.05), title="Leave-one-attack-type-out (CIC)")
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
save(fig, "fig8_loao")

# 9. feature-group transfer (H2)
G = R["group_transfer"]
groups = ["intensity", "temporal", "direction", "protocol", "all"]
fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9), sharey=True)
for k, mode in enumerate(("only", "without")):
    gs = [g for g in groups if f"{g}|{mode}" in G]
    x = np.arange(len(gs))
    ax[k].bar(x - 0.2, [G[f"{g}|{mode}"]["cic"] for g in gs], 0.38, color=C[0], label="CIC test")
    ax[k].bar(x + 0.2, [G[f"{g}|{mode}"]["ext"] for g in gs], 0.38, color=C[1], label="External (cross-dataset)")
    ax[k].set_xticks(x, gs)
    ax[k].set(title=f"Feature group {'alone' if mode == 'only' else 'removed'}", ylim=(0, 1.05))
ax[0].set_ylabel("Macro-F1")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=7.5)
save(fig, "fig9_group_transfer")

# 10. ablation
AB = R["ablation"]
full = AB[f"Full proposed ({MAIN})"]
names = [n for n in AB if not n.startswith("Full proposed")]
fig, ax = plt.subplots(1, 2, figsize=(8.2, 0.26 * len(names) + 1.0), sharey=True)
y = np.arange(len(names))
ax[0].barh(y, [AB[n]["cic_f1"] - full["cic_f1"] for n in names], color=C[0], height=0.6)
ax[1].barh(y, [AB[n]["ext_f1"] - full["ext_f1"] for n in names], color=C[1], height=0.6)
ax[0].set_yticks(y, names, fontsize=7)
ax[0].invert_yaxis()
for a, t in zip(ax, ("Change in CIC test macro-F1", "Change in external macro-F1")):
    a.axvline(0, color=INK2, linewidth=0.8)
    a.set(title=t)
save(fig, "fig10_ablation")

# 11. slow-ramp stress
S = R["errors"]["slow_ramp_stress"]
fs = sorted(S, key=float)
fig, ax = plt.subplots(figsize=(5.2, 2.9))
for j, d in enumerate(S[fs[0]]):
    ax.plot([float(f) for f in fs], [S[f][d]["recall"] for f in fs], "-o", color=C[j], label=d)
ax.set(xscale="log", xlabel="Inter-packet gaps stretched by factor", ylabel="Attack recall (CIC test)", ylim=(0, 1.05),
       title="Slow-ramp evasion stress test")
ax.set_xticks([float(f) for f in fs], fs)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
save(fig, "fig11_stress")

# 12. permutation importance CIC vs external
P = R["errors"]["perm_importance"]
o = np.argsort(-np.array(P["cic"]))[:12]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
y = np.arange(len(o))
ax.barh(y - 0.2, np.array(P["cic"])[o], 0.38, color=C[0], label="CIC test")
ax.barh(y + 0.2, np.array(P["ext"])[o], 0.38, color=C[1], label="External set")
ax.set_yticks(y, [P["features"][i] for i in o], fontsize=7.5)
ax.invert_yaxis()
ax.axvline(0, color=INK2, linewidth=0.8)
ax.set(xlabel="Drop in macro-F1 when the feature is shuffled", title=f"Stage-2 feature importance (Spearman rho = {P['spearman']:.2f})")
ax.legend(fontsize=7, loc="lower right")
save(fig, "fig12_importance")

# 13. external false positives by benign source and recall by external attack
E = R["errors"]
fpb = E["external_fp_by_benign_source"]
ks = sorted(fpb, key=lambda k: -fpb[k]["fp_rate"])
rec = E["external_recall_by_attack"]
ka = sorted(rec, key=lambda k: rec[k])
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8), gridspec_kw=dict(wspace=0.9))
ax[0].barh(np.arange(len(ks)), [fpb[k]["fp_rate"] for k in ks], color=C[7], height=0.6)
ax[0].set_yticks(np.arange(len(ks)), [k.replace("malcolm-", "") for k in ks], fontsize=7)
ax[0].invert_yaxis()
ax[0].set(xlabel="False-positive rate", title="External benign: false alarms by source", xlim=(0, 1))
ax[1].barh(np.arange(len(ka)), [rec[k] for k in ka], color=C[0], height=0.6)
ax[1].set_yticks(np.arange(len(ka)), ka, fontsize=6.5)
ax[1].set(xlabel="Recall", title="External attacks: recall by capture", xlim=(0, 1))
save(fig, "fig13_external_errors")
print("figures written")
