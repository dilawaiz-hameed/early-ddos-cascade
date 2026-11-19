"""Assemble bidirectional flows from packet captures and store their first packets.

Output: data/flows_<dataset>.npz with
  pk   float32 [n_flows, PMAX, 10]  per-packet fields of the first PMAX packets (NaN padded)
  full float32 [n_flows, F]         full-flow feature vector (whole flow, for the accuracy-ceiling baseline)
  meta (json) per flow: dataset, file, attack type, label, block, onset time, length
Per-packet fields: 0 ts (relative to flow start), 1 ip_len, 2 dir (0=initiator, 1=responder),
3 tcp flags, 4 ip header len, 5 l4 header len, 6 tcp window, 7 payload len, 8 fragment bit, 9 abs ts.
"""
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pcapio import read_packets
from features import window_features, FEATURE_NAMES

PMAX = 32               # packets kept per flow for early windows
FULL_CAP = 5000         # packets used for full-flow (complete flow) features
IDLE_TIMEOUT = 60.0     # seconds of silence that closes a flow

CIC_ATTACKER = {"172.16.0.5"}
CIC_VICTIMS = {"192.168.50.1", "192.168.50.4"}

# Chris Sanders' library: keep only captures of ordinary (non-malicious) activity
SANDERS_EXCLUDE = ("arppoison", "aurora", "cryptowall", "ek_to_", "dvwa", "ratinfected",
                   "sessionhijacking", "synscan", "osfingerprinting", "isp_hijack", "80211", "dns.pcapclear")


def _flows_from_file(path, label_fn, attack_name, max_flows=None, seed=0):
    flows = {}            # key -> dict
    done = []
    for p in read_packets(path):
        a, b = (p.src, p.sport), (p.dst, p.dport)
        key = (a, b, p.proto) if a <= b else (b, a, p.proto)
        f = flows.get(key)
        if f is not None and p.ts - f["last"] > IDLE_TIMEOUT:
            done.append(f)
            f = None
        if f is None:
            f = {"init": a, "t0": p.ts, "last": p.ts, "pk": [], "n": 0,
                 "src": p.src, "dst": p.dst, "proto": p.proto}
            flows[key] = f
        f["last"] = p.ts
        f["n"] += 1
        if len(f["pk"]) < FULL_CAP:
            d = 0 if (p.src, p.sport) == f["init"] else 1
            f["pk"].append((p.ts, p.ip_len, d, p.flags, p.ip_hl, p.l4_hl, p.win, p.payload, p.frag, p.proto))
    done.extend(flows.values())
    if max_flows and len(done) > max_flows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(done), max_flows, replace=False)
        done = [done[i] for i in sorted(idx)]
    out = []
    for f in done:
        lab, typ = label_fn(f, attack_name)
        out.append((f, lab, typ))
    return out


def _pack(records, dataset, fname, segments):
    pk = np.full((len(records), PMAX, 10), np.nan, dtype=np.float64)
    full = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float32)
    meta = []
    for i, (f, lab, typ) in enumerate(records):
        arr = np.asarray(f["pk"], dtype=np.float64)
        proto = arr[0, 9]
        rel = arr.copy()
        rel[:, 9] = arr[:, 0]                       # keep absolute ts in column 9
        rel[:, 0] = arr[:, 0] - f["t0"]
        k = min(PMAX, len(rel))
        pk[i, :k] = rel[:k]
        full[i] = window_features(rel[None, :, :], np.array([proto]), n=len(rel), t=np.inf)[0]
        seg = next(s for s in segments if s[0] <= f["t0"] <= s[1])
        meta.append(dict(dataset=dataset, file=fname, type=typ, label=int(lab), proto=int(proto),
                         t0=float(f["t0"]), dur=float(f["last"] - f["t0"]), n=int(f["n"]), seg=f"{fname}#{segments.index(seg)}",
                         seg_start=float(seg[0]), seg_end=float(seg[1])))
    return pk, full, meta


def _segments(path, gap=60.0):
    ts = np.sort(np.fromiter((p.ts for p in read_packets(path)), dtype=np.float64))
    cut = np.where(np.diff(ts) > gap)[0]
    starts = np.r_[ts[0], ts[cut + 1]]
    ends = np.r_[ts[cut], ts[-1]]
    return list(zip(starts, ends))


def cic_label(f, attack_name):
    ends = {f["src"], f["dst"]}
    is_attack = bool(ends & CIC_ATTACKER) and bool(ends & CIC_VICTIMS)
    return (1, attack_name) if is_attack else (0, "BENIGN")


def build(dataset):
    recs_all, pks, fulls, metas = [], [], [], []
    if dataset == "cic2019":
        files = sorted(glob.glob("data/cic2019/*.pcap"))
        for path in files:
            name = os.path.basename(path).replace("CIC-DDoS-2019-", "").replace(".pcap", "")
            attack = "BENIGN" if name == "Benign" else name
            recs = _flows_from_file(path, cic_label, attack)
            pk, full, meta = _pack(recs, dataset, name, _segments(path))
            pks.append(pk); fulls.append(full); metas += meta
    elif dataset == "external":
        # attack side: StopDDoS real-world captures (one-way, >=99% attack traffic per the curators)
        for path in sorted(glob.glob("data/stopddos/*.pcap*")):
            name = os.path.basename(path).split(".pcap")[0]
            recs = _flows_from_file(path, lambda f, a: (1, a), name, max_flows=1500, seed=1)
            pk, full, meta = _pack(recs, dataset, name, _segments(path))
            pks.append(pk); fulls.append(full); metas += meta
        # benign side: everyday traffic captures from a different network
        for path in sorted(glob.glob("data/sanders/*.pcap*")):
            name = os.path.basename(path).split(".pcap")[0]
            if any(x in os.path.basename(path) for x in SANDERS_EXCLUDE):
                continue
            if not any(True for _ in read_packets(path)):
                continue
            recs = _flows_from_file(path, lambda f, a: (0, "BENIGN"), "BENIGN")
            pk, full, meta = _pack(recs, dataset, name, _segments(path))
            pks.append(pk); fulls.append(full); metas += meta
        # more benign: everyday-protocol sample captures from the Malcolm project (HTTP, DNS, TLS, QUIC, SSH, ...)
        for path in sorted(glob.glob("data/malcolm/*.pcap")):
            name = "malcolm-" + os.path.basename(path).split(".pcap")[0]
            recs = _flows_from_file(path, lambda f, a: (0, "BENIGN"), "BENIGN", max_flows=1500, seed=2)
            pk, full, meta = _pack(recs, dataset, name, _segments(path))
            pks.append(pk); fulls.append(full); metas += meta
    pk = np.concatenate(pks).astype(np.float32)
    full = np.concatenate(fulls)
    np.savez_compressed(f"data/flows_{dataset}.npz", pk=pk, full=full, meta=json.dumps(metas))
    from collections import Counter
    print(dataset, pk.shape, Counter((m["type"], m["label"]) for m in metas))


if __name__ == "__main__":
    for d in sys.argv[1:] or ["cic2019", "external"]:
        build(d)
