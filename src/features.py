"""Early-window features: computed only from packets seen up to the decision point.

A window is (N, T): the detector decides when the N-th packet of a flow arrives or when
T seconds have passed since the flow's first packet, whichever comes first. Every feature
is computable from those packets alone; nothing needs the completed flow.
"""
import numpy as np

# name, semantic group (following Elangovan et al., 2026), packets needed before it is defined
FEATURES = [
    ("n_pkts", "intensity", 1), ("bytes_sum", "intensity", 1), ("len_mean", "intensity", 1),
    ("len_std", "intensity", 2), ("len_max", "intensity", 1), ("len_min", "intensity", 1),
    ("byte_rate", "intensity", 2), ("pkt_rate", "intensity", 2),
    ("duration", "temporal", 2), ("iat_mean", "temporal", 2), ("iat_std", "temporal", 3),
    ("iat_min", "temporal", 2), ("iat_max", "temporal", 2),
    ("fwd_pkt_frac", "direction", 1), ("fwd_byte_frac", "direction", 1),
    ("bwd_pkts", "direction", 1), ("dir_changes", "direction", 2),
    ("is_tcp", "protocol", 1), ("is_udp", "protocol", 1), ("is_icmp", "protocol", 1),
    ("syn_frac", "protocol", 1), ("ack_frac", "protocol", 1), ("fin_frac", "protocol", 1),
    ("rst_frac", "protocol", 1), ("psh_frac", "protocol", 1), ("win_mean", "protocol", 1),
    ("iphl_mean", "protocol", 1), ("l4hl_mean", "protocol", 1), ("frag_frac", "protocol", 1),
    ("payload_mean", "protocol", 1), ("zero_payload_frac", "protocol", 1),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
GROUPS = {g: [i for i, f in enumerate(FEATURES) if f[1] == g] for g in ("intensity", "temporal", "direction", "protocol")}
MIN_PKTS = np.array([f[2] for f in FEATURES])


def decision_mask(pk, n, t):
    """Boolean mask of packets visible at decision time, and the decision time per flow (s)."""
    ts = pk[:, :, 0]
    P = ts.shape[1]
    idx = np.arange(P)[None, :]
    present = ~np.isnan(ts)
    mask = present & (idx < n) & (np.nan_to_num(ts, nan=np.inf) <= t)
    # decision time: N-th packet if it arrives within T, otherwise T (the detector cannot know the flow ended)
    n_eff = min(n, P)
    nth = ts[:, n_eff - 1] if n_eff >= 1 else np.full(len(ts), np.nan)
    has_nth = ~np.isnan(nth) & (nth <= t)
    last = np.nanmax(np.where(present, ts, np.nan), axis=1)
    dec = np.where(has_nth, nth, t if np.isfinite(t) else last)
    return mask, dec


def window_features(pk, proto, n, t):
    """pk: [flows, P, 10] (ts relative to flow start, NaN padded). Returns [flows, F] float32."""
    mask, _ = decision_mask(pk, n, t)
    m = mask.astype(np.float64)
    k = m.sum(1)
    k1 = np.maximum(k, 1)
    g = lambda c: np.where(mask, pk[:, :, c], 0.0)
    ts, ln, dr, fl = pk[:, :, 0], g(1), g(2), g(3).astype(np.int64)
    out = {}
    out["n_pkts"] = k
    out["bytes_sum"] = ln.sum(1)
    out["len_mean"] = out["bytes_sum"] / k1
    out["len_std"] = np.sqrt(np.maximum((m * (ln - out["len_mean"][:, None]) ** 2).sum(1) / k1, 0))
    out["len_max"] = np.where(mask, pk[:, :, 1], -np.inf).max(1)
    out["len_min"] = np.where(mask, pk[:, :, 1], np.inf).min(1)
    tsm = np.where(mask, ts, np.nan)
    first, lastv = np.nanmin(np.where(mask, ts, np.inf), 1), np.nanmax(np.where(mask, ts, -np.inf), 1)
    dur = np.where(k >= 2, lastv - first, 0.0)
    out["duration"] = dur
    out["byte_rate"] = np.where(k >= 2, out["bytes_sum"] / np.maximum(dur, 1e-6), 0.0)
    out["pkt_rate"] = np.where(k >= 2, k / np.maximum(dur, 1e-6), 0.0)
    iat = np.diff(tsm, axis=1)                     # NaN where either packet invisible
    vi = ~np.isnan(iat)
    ni = vi.sum(1)
    iz = np.where(vi, iat, 0.0)
    out["iat_mean"] = np.where(ni > 0, iz.sum(1) / np.maximum(ni, 1), -1.0)
    out["iat_std"] = np.where(ni > 1, np.sqrt(np.maximum((np.where(vi, (iat - out["iat_mean"][:, None]) ** 2, 0)).sum(1) / np.maximum(ni, 1), 0)), -1.0)
    out["iat_min"] = np.where(ni > 0, np.where(vi, iat, np.inf).min(1, initial=np.inf), -1.0)
    out["iat_max"] = np.where(ni > 0, np.where(vi, iat, -np.inf).max(1, initial=-np.inf), -1.0)
    fwd = m * (1 - dr)
    out["fwd_pkt_frac"] = fwd.sum(1) / k1
    out["fwd_byte_frac"] = (fwd * ln).sum(1) / np.maximum(out["bytes_sum"], 1)
    out["bwd_pkts"] = (m * dr).sum(1)
    drm = np.where(mask, dr, np.nan)
    ch = np.abs(np.diff(drm, axis=1))
    out["dir_changes"] = np.nansum(ch, axis=1)
    out["is_tcp"] = (proto == 6).astype(float)
    out["is_udp"] = (proto == 17).astype(float)
    out["is_icmp"] = (proto == 1).astype(float)
    for name, bit in (("fin_frac", 0x01), ("syn_frac", 0x02), ("rst_frac", 0x04), ("psh_frac", 0x08), ("ack_frac", 0x10)):
        out[name] = (m * ((fl & bit) > 0)).sum(1) / k1
    out["win_mean"] = g(6).sum(1) / k1
    out["iphl_mean"] = g(4).sum(1) / k1
    out["l4hl_mean"] = g(5).sum(1) / k1
    out["frag_frac"] = g(8).sum(1) / k1
    out["payload_mean"] = g(7).sum(1) / k1
    out["zero_payload_frac"] = (m * (pk[:, :, 7] == 0)).sum(1) / k1
    X = np.stack([out[nm] for nm in FEATURE_NAMES], axis=1)
    X[~np.isfinite(X)] = 0.0
    return X.astype(np.float32)
