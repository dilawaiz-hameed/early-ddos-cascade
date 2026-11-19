"""Generate LaTeX macros and tables for the paper from the experiment result files.

Run from the repository root:  python paper/gen_tables.py
Reads results_full/ (main study) and results/ (pilot). Writes paper/generated/*.tex
so that no number in the paper is typed by hand.
"""
import json
import os

R = json.load(open("results_full/experiments.json"))
K = json.load(open("results_full/cost_bench.json"))
PIL = json.load(open("results/experiments.json"))
OUT = "paper/generated"
os.makedirs(OUT, exist_ok=True)
B, AB, L, G, S, E = R["baselines"], R["ablation"], R["leave_one_attack_out"], R["group_transfer"], R["search"], R["errors"]
PM, ED, CL, DT8, RF, ERF = ("Proposed (enterprise)", "Proposed (edge)", "Proposed (cloud)",
                            "Fixed early DT-8 (N=2,T=0.1s)", "Full-flow RF-100", "Fixed early RF-100 (N=8,T=0.5s)")
HY = PM + " + IF fallback"
f4 = lambda x: f"{x:.4f}"
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"
pct = lambda x, d=2: f"{100 * x:.{d}f}\\%"


def ms(s):
    v = 1000 * s
    return "$<$0.01" if v < 0.01 else (f"{v:.2f}" if v < 10 else f"{v:.0f}")


def sec(s):
    return f"{s:.1f}\\,s" if s >= 1 else f"{ms(s)}\\,ms"


T = lambda n: B[n]["cic_test"]
X = lambda n: B[n]["external"]
conf = E["cic_confusion_by_type"]
n_test = sum(v["n"] for v in conf.values())
n_ben = conf["BENIGN"]["n"]
TRAIN = {"BENIGN": 6863, "DNS": 28000, "LDAP": 4000, "MSSQL": 4000, "NTP": 2319, "NetBIOS": 4000, "SNMP": 4000,
         "SSDP": 3000, "SynFlood": 2000, "TFTP": 4000, "UDP": 3000, "UDPLag": 134, "WebDDoS": 866}   # flows_full.py build log
n_train = sum(TRAIN.values())
p2 = ["P2: CNB + NearMiss", "P2: kNN + NearMiss", "P2: RF + NearMiss", "P2: LR + NearMiss"]
best_p2 = max(p2, key=lambda n: T(n)["macro_f1"])
w = {(a["N"], a["T"]): a for a in R["window_sweep"] if a["model"] == "RF"}
ext_best = max(B, key=lambda n: X(n)["macro_f1"])
prof = R["profiles"]

M = {
    "nTrain": f"{n_train:,}", "nTest": f"{n_test:,}", "nTestBen": f"{n_ben:,}", "nTestAtk": f"{n_test - n_ben:,}",
    "nTrainBen": f"{TRAIN['BENIGN']:,}", "attackShare": pct(1 - n_ben / n_test, 0),
    "mainFone": f4(T(PM)["macro_f1"]), "mainCIlo": f3(T(PM)["macro_f1_ci"][0]), "mainCIhi": f3(T(PM)["macro_f1_ci"][1]),
    "mainMCC": f3(T(PM)["mcc"]), "mainFPR": pct(T(PM)["fpr"]), "mainFNR": pct(T(PM)["fnr"], 3),
    "mainTTDmed": ms(T(PM)["ttd_median"]), "mainTTDmean": ms(T(PM)["ttd_mean"]), "mainPone": f3(T(PM)["prevalence"][0]["precision"]),
    "mainExt": f3(X(PM)["macro_f1"]), "mainSize": f"{K[PM]['size_kb']:.0f}", "mainLat": f2(K[PM]["lat_p50_ms"]),
    "mainThr": f"{K[PM]['throughput_fps']:,.0f}", "mainDefer": pct(K[PM]["deferred"], 1),
    "mainPortMap": pct(T(PM)["recall_by_type"]["PortMap"], 1),
    "edgeFone": f4(T(ED)["macro_f1"]), "edgeFPR": pct(T(ED)["fpr"]), "edgeSize": f"{K[ED]['size_kb']:.1f}", "edgeLat": f2(K[ED]["lat_p50_ms"]),
    "cloudFone": f4(T(CL)["macro_f1"]),
    "dtFone": f4(T(DT8)["macro_f1"]), "dtFPR": pct(T(DT8)["fpr"]), "dtPone": f3(T(DT8)["prevalence"][0]["precision"]),
    "dtSize": f"{K[DT8]['size_kb']:.1f}", "dtLat": f2(K[DT8]["lat_p50_ms"]), "dtHolm": f2(B[DT8]["paired_vs_main"]["p_holm"]),
    "rfFone": f4(T(RF)["macro_f1"]), "rfFPR": pct(T(RF)["fpr"]), "rfTTDmed": f"{T(RF)['ttd_median']:.1f}",
    "rfTTDend": ms(T(RF)["flow_end"]["ttd_median"]), "rfPone": f3(T(RF)["prevalence"][0]["precision"]),
    "rfSize": f"{K[RF]['size_kb']:.0f}", "rfLat": f2(K[RF]["lat_p50_ms"]), "rfThr": f"{K[RF]['throughput_fps']:,.0f}",
    "rfExt": f3(X(RF)["macro_f1"]),
    "latRatio": f"{K[RF]['lat_p50_ms'] / K[PM]['lat_p50_ms']:.0f}", "ttdRatio": f"{T(RF)['ttd_median'] / T(PM)['ttd_median']:,.0f}",
    "ifPone": f3(T("E1: Isolation Forest (N=8,T=0.5s)")["prevalence"][0]["precision"]), "hyFPR": pct(T(HY)["fpr"], 1),
    "bestPtwo": f4(T(best_p2)["macro_f1"]), "bestPtwoName": best_p2.replace("P2: ", "").replace(" + NearMiss", ""),
    "impAbs": f"{T(PM)['macro_f1'] - T(best_p2)['macro_f1']:+.4f}",
    "impRel": f"{100 * (T(PM)['macro_f1'] - T(best_p2)['macro_f1']) / T(best_p2)['macro_f1']:+.1f}\\%",
    "knnFone": f4(T("P2: kNN + NearMiss")["macro_f1"]), "knnSynRec": pct(T("P2: kNN + NearMiss")["recall_by_type"]["SynFlood"], 1),
    "knnLagRec": pct(T("P2: kNN + NearMiss")["recall_by_type"]["UDPLag"], 1),
    "knnLat": f2(K["P2: kNN + NearMiss"]["lat_p50_ms"]), "knnSize": f"{K['P2: kNN + NearMiss']['size_kb']:,.0f}",
    "knnSpeed": f"{K['P2: kNN + NearMiss']['lat_p50_ms'] / K[PM]['lat_p50_ms']:.1f}",
    "winOne": f4(w[(1, 0.05)]["macro_f1"]), "winTwo": f4(w[(2, 0.05)]["macro_f1"]),
    "winThreeA": ms(w[(3, 0.05)]["decision_mean_all"]), "winThreeB": f"{w[(3, 2.0)]['decision_mean_all']:.2f}",
    "hvN": f3(S["nsga2"]["hv_mean"]), "hvR": f3(S["random"]["hv_mean"]), "hvG": f3(S["ga_f1"]["hv_mean"]),
    "frontN": str(len(R["front"])), "evals": str(S["nsga2"]["unique_evals"][0]),
    "frontTrees": str(sum(1 for e in R["front"] if e["cfg"]["fam"] == "DT" or (e["cfg"]["cascade"] and e["cfg"]["fam"] in ("LR", "DT")))),
    "loaoMin": pct(min(L[t][PM]["recall_unseen"] for t in L), 1), "loaoTypes": str(len(L)),
    "extBest": f3(X(ext_best)["macro_f1"]), "extBestName": ext_best,
    "revRF": f3(R["reverse_transfer"][RF]["cic_all"]["macro_f1"]), "revMain": f3(R["reverse_transfer"][PM]["cic_all"]["macro_f1"]),
    "revRFin": f3(R["reverse_transfer"][RF]["external_test"]["macro_f1"]),
    "dirExt": f3(G["direction|only"]["ext"]), "dirFPR": pct(G["direction|only"]["ext_fpr"], 0),
    "tempExt": f3(G["temporal|only"]["ext"]), "noTempExt": f3(G["temporal|without"]["ext"]),
    "abSTwoTTD": ms(AB["No cascade (Stage 2 only)"]["ttd_mean"]), "abSTwoLat": f2(AB["No cascade (Stage 2 only)"]["cost"]["latency"] * 1000),
    "abSTwoF": f4(AB["No cascade (Stage 2 only)"]["cic_f1"]), "abSOneF": f4(AB["Stage 1 only (no deferral)"]["cic_f1"]),
    "abFullTTD": ms(AB["Full proposed (enterprise)"]["ttd_mean"]), "abFullLat": f2(AB["Full proposed (enterprise)"]["cost"]["latency"] * 1000),
    "abNoneF": f4(AB["Stage-1 window N1=1"]["cic_f1"]), "abAllExt": f3(AB["Feature selection off (all 31)"]["ext_f1"]),
    "abIntF": f4(AB["Drop group: intensity"]["cic_f1"]), "abIntFPR": pct(AB["Drop group: intensity"]["fpr"], 1),
    "abTempExt": f3(AB["Drop group: temporal"]["ext_f1"]),
    "balNM": f4(R["balancing"]["NearMiss-1"]["cic_f1"]), "balCW": f4(R["balancing"]["class weights"]["cic_f1"]),
    "spear": f2(E["perm_importance"]["spearman"]), "pilotSpear": f2(PIL["errors"]["perm_importance"]["spearman"]),
    "stressMain": pct(E["slow_ramp_stress"]["50"][PM]["recall"], 2), "stressRF": pct(E["slow_ramp_stress"]["10"][ERF]["recall"], 1),
    "missed": str(E["cic_missed"]["n"]), "dnsFP": pct(E["external_fp_by_benign_source"]["malcolm-DNS"]["fp_rate"], 0),
    "pilotFone": f4(PIL["baselines"][PM]["cic_test"]["macro_f1"]), "pilotSyn": pct(PIL["leave_one_attack_out"]["SynFlood"][PM]["recall_unseen"], 1),
    "pilotExt": f3(PIL["baselines"][PM]["external"]["macro_f1"]),
    "mainSyn": pct(L["SynFlood"][PM]["recall_unseen"], 1),
    "pilotHvN": f3(PIL["search"]["nsga2"]["hv_mean"]), "pilotHvR": f3(PIL["search"]["random"]["hv_mean"]),
    "eOneD": str(prof["enterprise"]["cfg"]["d1"]), "eNone": str(prof["enterprise"]["cfg"]["N1"]),
    "eTone": f"{prof['enterprise']['cfg']['T1'] * 1000:.0f}", "eNtwo": str(prof["enterprise"]["cfg"]["N2"]),
    "eTtwo": f"{prof['enterprise']['cfg']['T2'] * 1000:.0f}", "eFeat": str(sum(prof["enterprise"]["cfg"]["mask"])),
    "eTlo": f3(prof["enterprise"]["cfg"]["tau_lo"]), "eThi": f3(prof["enterprise"]["cfg"]["tau_hi"]),
    "nDetectors": str(len(B)),
}
with open(f"{OUT}/macros.tex", "w") as f:
    f.write("% generated by paper/gen_tables.py -- do not edit by hand\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")

# ---- Table: main results on the held-out test day
rows = [(PM, "Proposed cascade (enterprise)"), (ED, "Proposed (edge)"), (CL, "Proposed (cloud)"),
        (DT8, "Fixed early DT-8 ($N{=}2$, 100\\,ms)"), (ERF, "Fixed early RF-100 ($N{=}8$, 500\\,ms)"),
        (RF, "Full-flow RF-100"), ("Full-flow GBDT (HGB)", "Full-flow GBDT"), ("Full-flow MLP (32,16)", "Full-flow MLP"),
        ("E5: ET-top10 + RF", "ET-top10 + RF~\\cite{nawaz2025}"), ("E1: Isolation Forest (N=8,T=0.5s)", "Isolation Forest~\\cite{lamptey2025}"),
        ("E1: One-Class SVM (N=8,T=0.5s)", "One-class SVM~\\cite{lamptey2025}"), ("Random-search best", "Random-search best"),
        ("GA (F1 only) best", "GA (F1 only) best")]
with open(f"{OUT}/tab_main.tex", "w") as f:
    for n, lab in rows:
        t, x = T(n), X(n)
        ci = f"[{t['macro_f1_ci'][0]:.3f}, {t['macro_f1_ci'][1]:.3f}]"
        pv = B[n].get("paired_vs_main")
        pstr = "--" if pv is None else f"{pv['p_holm']:.3f}"
        f.write(f"{lab} & {t['macro_f1']:.4f} & {ci} & {t['mcc']:.3f} & {100 * t['fpr']:.2f} & {sec(t['ttd_median'])} & "
                f"{t['prevalence'][0]['precision']:.3f} & {x['macro_f1']:.3f} & {pstr} \\\\\n")

# ---- Table: cost
cost_rows = [(ED, "Proposed (edge)"), (PM, "Proposed (enterprise)"), (CL, "Proposed (cloud)"), (DT8, "Fixed early DT-8"),
             (ERF, "Fixed early RF-100"), (RF, "Full-flow RF-100"), ("Full-flow GBDT (HGB)", "Full-flow GBDT"),
             ("P2: kNN + NearMiss", "$k$NN+NearMiss~\\cite{ebrahem2026}"), ("E1: Isolation Forest (N=8,T=0.5s)", "Isolation Forest")]
with open(f"{OUT}/tab_cost.tex", "w") as f:
    for n, lab in cost_rows:
        k = K[n]
        st = f"{k['state_bytes_per_flow']}" if k["state_bytes_per_flow"] else "flow"
        f.write(f"{lab} & {k['lat_p50_ms']:.2f} & {k['lat_p95_ms']:.2f} & {k['throughput_fps'] / 1000:.1f} & {k['size_kb']:.1f} & {st} \\\\\n")

# ---- Table: base-paper quantitative comparison (re-implemented rows)
with open(f"{OUT}/tab_p2.tex", "w") as f:
    for n in p2 + [RF, PM]:
        t = T(n)
        lab = {"P2: CNB + NearMiss": "Ebrahem~\\cite{ebrahem2026} CNB+NearMiss$^\\dagger$", "P2: kNN + NearMiss": "Ebrahem~\\cite{ebrahem2026} $k$NN+NearMiss$^\\dagger$",
               "P2: RF + NearMiss": "Ebrahem~\\cite{ebrahem2026} RF+NearMiss$^\\dagger$", "P2: LR + NearMiss": "Ebrahem~\\cite{ebrahem2026} LR+NearMiss$^\\dagger$",
               RF: "Full-flow RF, class weights$^\\dagger$", PM: "\\textbf{Proposed cascade}"}[n]
        lat = K[n]["lat_p50_ms"] if n in K else B[n]["cost"]["latency"] * 1000
        size = K[n]["size_kb"] if n in K else B[n]["cost"]["size"] / 1024
        d = T(PM)["macro_f1"] - t["macro_f1"]
        imp = "--" if n == PM else f"{d:+.4f} ({100 * d / t['macro_f1']:+.1f}\\%)"
        f.write(f"{lab} & {t['macro_f1']:.4f} & {100 * t['fpr']:.2f} & {100 * t['fnr']:.2f} & {sec(t['ttd_median'])} & {lat:.2f} & {size:.0f} & {imp} \\\\\n")

# ---- Table: transfer
with open(f"{OUT}/tab_transfer.tex", "w") as f:
    for n, lab in [(PM, "Proposed (enterprise)"), (ED, "Proposed (edge)"), (DT8, "Fixed early DT-8"), (RF, "Full-flow RF-100"),
                   ("E5: ET-top10 + RF", "ET-top10 + RF"), ("GA (F1 only) best", "GA (F1 only) best")]:
        x = X(n)
        f.write(f"CIC$\\rightarrow$Ext & {lab} & {x['macro_f1']:.3f} & {x['recall_attack']:.3f} & {100 * x['fpr']:.1f} \\\\\n")
    for n, lab in [(PM, "Proposed (enterprise)"), (RF, "Full-flow RF-100"), (ERF, "Fixed early RF-100")]:
        x = R["reverse_transfer"][n]["cic_all"]
        f.write(f"Ext$\\rightarrow$CIC & {lab} & {x['macro_f1']:.3f} & {x['recall_attack']:.3f} & {100 * x['fpr']:.1f} \\\\\n")

# ---- Table: ablation (selected)
sel = [("Full proposed (enterprise)", "Full cascade (validation-selected)"), ("No cascade (Stage 2 only)", "Stage 2 only"),
       ("Stage 1 only (no deferral)", "Stage 1 only"), ("Stage-1 window N1=1", "Stage 1 at $N_1{=}1$"),
       ("Looser gate (tau 0.30/0.60)", "Looser gate (0.30/0.60)"), ("Stage-2 window T2=0.05s", "Stage 2 at $T_2{=}50$\\,ms"),
       ("Feature selection off (all 31)", "All 31 features"), ("Drop group: intensity", "$-$ intensity"),
       ("Drop group: temporal", "$-$ temporal"), ("Drop group: protocol", "$-$ protocol"),
       ("Drop group: direction", "$-$ direction"), ("Stage-2 family: DT", "Stage 2 = tree"),
       ("Stage-2 family: LR", "Stage 2 = LR"), ("Stage-2 family: kNN", "Stage 2 = $k$NN")]
with open(f"{OUT}/tab_ablation.tex", "w") as f:
    for n, lab in sel:
        a = AB[n]
        f.write(f"{lab} & {a['cic_f1']:.4f} & {100 * a['fpr']:.2f} & {ms(a['ttd_mean'])} & {a['ext_f1']:.3f} & "
                f"{a['cost']['size'] / 1024:.1f} & {a['cost']['latency'] * 1000:.2f} \\\\\n")
for name in ("tab_main", "tab_cost", "tab_p2", "tab_transfer", "tab_ablation"):   # rule inside the file: \input + \bottomrule clash
    with open(f"{OUT}/{name}.tex", "a") as f:
        f.write("\\bottomrule\n")
print("wrote", OUT, len(M), "macros")
