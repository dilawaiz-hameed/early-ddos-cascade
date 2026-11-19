"""Build flows from the FULL CIC-DDoS2019 captures (selected 200 MB files of both days).

Usage:
    python src/flows_full.py "D:\\Cnet project\\PCAP-01-12_0-0249" "D:\\Cnet project\\PCAP-03-11"

Writes data/flows_cicfull.npz in the same format as flows.py, plus a 'day' field:
  1 Dec 2018 (file prefix SAT-01-12) = training day, 3 Nov 2018 (SAT-03-11) = testing day.

Labels
  * attack vs benign: a flow is an attack if one endpoint is the attacker 172.16.0.5;
  * attack type: from the file's position in the capture, using the bursts we verified
    against the published schedule (see FILE_TYPES below); stretches whose type is unclear
    are marked UNKNOWN and their attack flows are dropped (benign flows are kept).

Memory safety (8 GB laptops)
  * files are streamed one at a time; flows idle for 60 s are finalised as we go;
  * spoofed floods create a new flow for almost every packet, so only 1 in ATTACK_SAMPLE
    attack flows (chosen by a stable hash of the flow key) is tracked;
  * at most ATTACK_CAP attack flows and BENIGN_CAP benign flows are kept per file.
Flows that cross a file boundary are split in two (each 200 MB file covers seconds to minutes).
"""
import glob
import json
import os
import re
import sys
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcapio import read_packets
from features import window_features, FEATURE_NAMES

PMAX, FULL_CAP, IDLE_TIMEOUT = 32, 5000, 60.0
ATTACKER = "172.16.0.5"
ATTACK_SAMPLE = int(os.environ.get("CN_ATTACK_SAMPLE", 20))
ATTACK_CAP = int(os.environ.get("CN_ATTACK_CAP", 1000))
BENIGN_CAP = int(os.environ.get("CN_BENIGN_CAP", 5000))

# file-number ranges -> attack type (inclusive), verified from capture times + attacker share
FILE_TYPES = {
    "01-12": [(0, 0, "UNKNOWN"), (1, 188, "NTP"), (189, 196, "UNKNOWN"), (197, 377, "DNS"),
              (378, 442, "LDAP"), (443, 473, "MSSQL"), (474, 485, "NetBIOS"), (486, 570, "SNMP"),
              (571, 591, "SSDP"), (592, 616, "UDP"), (617, 617, "UDPLag|WebDDoS"),
              (618, 619, "SynFlood"), (620, 818, "TFTP")],
    "03-11": [(0, 0, "PortMap"), (1, 11, "NetBIOS"), (12, 68, "LDAP"), (69, 106, "MSSQL"),
              (107, 136, "UDP"), (137, 137, "UDPLag"), (138, 145, "SynFlood")],
}
DAY = {"01-12": "train_day", "03-11": "test_day"}


def file_info(path):
    m = re.search(r"SAT-(\d\d-\d\d)-2018_(\d+)$", os.path.basename(path))
    if not m:
        return None
    day, digits = m.group(1), m.group(2)
    num = int(digits[1:]) if len(digits) > 1 else 0      # "_0"=0, "_01"=1, "_0215"=215
    return day, num


def attack_type(day, num, proto):
    for a, b, t in FILE_TYPES[day]:
        if a <= num <= b:
            if t == "UDPLag|WebDDoS":                       # file 617: both low-volume attacks
                return "WebDDoS" if proto == 6 else "UDPLag"
            return t
    return "UNKNOWN"


def finalise(f, day, num, fname):
    arr = np.asarray(f["pk"], dtype=np.float64)
    proto = int(arr[0, 9])
    rel = arr.copy()
    rel[:, 9] = arr[:, 0]
    rel[:, 0] = arr[:, 0] - f["t0"]
    pk = np.full((PMAX, 10), np.nan)
    k = min(PMAX, len(rel))
    pk[:k] = rel[:k]
    full = window_features(rel[None], np.array([proto]), n=len(rel), t=np.inf)[0]
    if f["att"]:
        typ = attack_type(day, num, proto)
        if typ == "UNKNOWN":
            return None
        lab = 1
    else:
        typ, lab = "BENIGN", 0
    meta = dict(dataset="cicfull", file=fname, type=typ, label=lab, proto=proto, t0=float(f["t0"]),
                dur=float(f["last"] - f["t0"]), n=int(f["n"]), seg=fname, day=DAY[day])
    return pk.astype(np.float32), full, meta


def process(path, rng):
    day, num = file_info(path)
    fname = os.path.basename(path)
    flows, done = {}, []
    for i, p in enumerate(read_packets(path)):
        att = p.src == ATTACKER or p.dst == ATTACKER
        a, b = (p.src, p.sport), (p.dst, p.dport)
        key = (a, b, p.proto) if a <= b else (b, a, p.proto)
        if att and zlib.crc32(repr(key).encode()) % ATTACK_SAMPLE:
            continue
        f = flows.get(key)
        if f is not None and p.ts - f["last"] > IDLE_TIMEOUT:
            done.append(f)
            f = None
        if f is None:
            f = {"init": a, "t0": p.ts, "last": p.ts, "pk": [], "n": 0, "att": att}
            flows[key] = f
        f["last"] = p.ts
        f["n"] += 1
        if len(f["pk"]) < FULL_CAP:
            d = 0 if a == f["init"] else 1
            f["pk"].append((p.ts, p.ip_len, d, p.flags, p.ip_hl, p.l4_hl, p.win, p.payload, p.frag, p.proto))
        if i % 200_000 == 199_999:                          # sweep flows that went idle
            idle = [k for k, v in flows.items() if p.ts - v["last"] > IDLE_TIMEOUT]
            for k in idle:
                done.append(flows.pop(k))
    done.extend(flows.values())
    att = [f for f in done if f["att"]]
    ben = [f for f in done if not f["att"]]
    if len(att) > ATTACK_CAP:
        att = [att[i] for i in sorted(rng.choice(len(att), ATTACK_CAP, replace=False))]
    if len(ben) > BENIGN_CAP:
        ben = [ben[i] for i in sorted(rng.choice(len(ben), BENIGN_CAP, replace=False))]
    recs = [r for r in (finalise(f, day, num, fname) for f in att + ben) if r is not None]
    return recs


def main(folders):
    rng = np.random.default_rng(0)
    files = []
    for fo in folders:
        files += [p for p in glob.glob(os.path.join(fo, "SAT-*")) if file_info(p)]
    files.sort(key=lambda p: (file_info(p)[0], file_info(p)[1]))
    pks, fulls, metas = [], [], []
    for j, path in enumerate(files):
        recs = process(path, rng)
        for pk, full, meta in recs:
            pks.append(pk); fulls.append(full); metas.append(meta)
        n_att = sum(m["label"] for _, _, m in recs)
        print(f"[{j + 1}/{len(files)}] {os.path.basename(path)}: {len(recs)} flows ({n_att} attack)", flush=True)
    # segment bounds (needed by the split code)
    for m in metas:
        m["seg_start"] = m["seg_end"] = 0.0
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/flows_cicfull.npz", pk=np.stack(pks), full=np.stack(fulls).astype(np.float32),
                        meta=json.dumps(metas))
    from collections import Counter
    print("saved data/flows_cicfull.npz", len(metas), "flows")
    for d in ("train_day", "test_day"):
        print(d, sorted(Counter(m["type"] for m in metas if m["day"] == d).items()))


if __name__ == "__main__":
    main(sys.argv[1:])
