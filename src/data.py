"""Load flow tables, cache early-window feature matrices, and build leak-free splits."""
import json
import os
import numpy as np

from features import window_features, decision_mask, FEATURE_NAMES

N_GRID = [1, 2, 3, 4, 8, 16, 32]
T_GRID = [0.05, 0.1, 0.5, 2.0]            # seconds (50, 100, 500, 2000 ms)
FULL = ("full", "full")
IDLE_TIMEOUT = 60.0                       # must match flows.IDLE_TIMEOUT


class FlowSet:
    def __init__(self, name):
        z = np.load(f"data/flows_{name}.npz")
        self.name = name
        self.pk = z["pk"].astype(np.float64)
        self.full = z["full"].astype(np.float32)
        self.meta = json.loads(str(z["meta"]))
        self.y = np.array([m["label"] for m in self.meta])
        self.type = np.array([m["type"] for m in self.meta])
        self.file = np.array([m["file"] for m in self.meta])
        self.seg = np.array([m["seg"] for m in self.meta])
        self.t0 = np.array([m["t0"] for m in self.meta])
        self.nlen = np.array([m["n"] for m in self.meta])
        self.proto = np.array([m["proto"] for m in self.meta])
        self.dur = np.array([m["dur"] for m in self.meta])
        self.day = np.array([m.get("day", "") for m in self.meta])
        fl = np.nan_to_num(self.pk[:, :, 3]).astype(np.int64)
        has_finrst = ((fl & 0x05) > 0).any(1)
        self.closed = (self.proto == 6) & has_finrst & (self.nlen <= self.pk.shape[1])
        self._cache = {}

    def X(self, n, t):
        """Feature matrix for window (n packets, t seconds)."""
        if (n, t) == FULL:
            return self.full
        key = (n, t)
        if key not in self._cache:
            self._cache[key] = window_features(self.pk, self.proto, n, t)
        return self._cache[key]

    def dec_time(self, n, t):
        """Seconds from the flow's first packet to the moment the window closes."""
        if (n, t) == ("full", "end"):         # optimistic variant: decide at the last packet
            return self.dur.copy()
        if (n, t) == FULL:
            # complete-flow detector: a flow exporter only emits the flow once it is closed by
            # TCP FIN/RST or after the idle timeout (60 s here) with no further packets
            return self.dur + np.where(self.closed, 0.0, IDLE_TIMEOUT)
        return decision_mask(self.pk, n, t)[1]


def chrono_split(fs, fracs=(0.6, 0.2, 0.2), nblocks=5):
    """Within every capture segment, order flows by start time and cut 60/20/20.

    Test flows are always later in time than training flows of the same capture, so bursts
    never straddle train and test. Each split is further cut into contiguous time blocks
    (used by the block bootstrap)."""
    part = np.empty(len(fs.y), dtype=object)
    block = np.empty(len(fs.y), dtype=object)
    for s in np.unique(fs.seg):
        idx = np.where(fs.seg == s)[0]
        idx = idx[np.argsort(fs.t0[idx], kind="stable")]
        a = int(round(fracs[0] * len(idx)))
        b = int(round((fracs[0] + fracs[1]) * len(idx)))
        for name, sl in (("train", idx[:a]), ("val", idx[a:b]), ("test", idx[b:])):
            part[sl] = name
            for j, chunk in enumerate(np.array_split(sl, nblocks)):
                block[chunk] = f"{s}|{name}|{j}"
    return {k: np.where(part == k)[0] for k in ("train", "val", "test")}, block


# Which flow table is the main dataset: "cic2019" (LUCID sample, default) or "cicfull"
# (the full CIC-DDoS2019 captures built by flows_full.py). Set with the CN_MAIN variable.
MAIN = os.environ.get("CN_MAIN", "cic2019")
RESULTS = os.environ.get("CN_RESULTS", "results")
FIGURES = os.environ.get("CN_FIGURES", "figures")


def main_split(fs, nblocks=10):
    """Split used by every experiment.

    LUCID sample: chronological 60/20/20 within each capture (chrono_split).
    Full captures: train on one day, test on the other. Training and validation come from
    1 Dec 2018 (first 80% / last 20% of each capture file by time); the whole of 3 Nov 2018
    is the test set, including PortMap, an attack type that never occurs on 1 Dec."""
    if not len(fs.day) or not (fs.day == "train_day").any():
        return chrono_split(fs, nblocks=nblocks)
    part = np.empty(len(fs.y), dtype=object)
    block = np.empty(len(fs.y), dtype=object)
    for s in np.unique(fs.seg):
        idx = np.where(fs.seg == s)[0]
        idx = idx[np.argsort(fs.t0[idx], kind="stable")]
        if fs.day[idx[0]] == "train_day":
            a = int(round(0.8 * len(idx)))
            pieces = (("train", idx[:a]), ("val", idx[a:]))
        else:
            pieces = (("test", idx),)
        for name, sl in pieces:
            part[sl] = name
            for j, chunk in enumerate(np.array_split(sl, nblocks)):
                block[chunk] = f"{s}|{name}|{j}"
    return {k: np.where(part == k)[0] for k in ("train", "val", "test")}, block
