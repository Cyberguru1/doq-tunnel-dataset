"""Unified DoQ flow feature extractor.

Converts raw traffic from two sources into a single, direction-aware,
28-feature flow representation so that benign and malicious DoQ traffic share
one feature space (EXECUTION_PLAN.md, milestone M3):

  * our pcaps  -> pcap_to_events()
  * IEEE DoQ+QUIC packet-level CSV -> ieee_to_events()

Intermediate "packet event" schema (produced by both readers):

    {
      "ts": float,          # seconds (relative is fine; only diffs are used)
      "length": int,        # on-wire packet length in bytes
      "direction": None,    # filled in later: "out" | "in"
      "src_ip": str, "src_port": int,
      "dst_ip": str, "dst_port": int,
    }

Direction is inferred per flow from the first packet's source endpoint
(the QUIC client = the connection initiator), which keeps the pcap reader and
the IEEE CSV reader fully consistent regardless of the CSV's own direction
column.

The 28 features (per time window) are defined in FEATURE_COLS. Aligning these
column names exactly with CIRA-CIC-DoHBrw-2020 is deferred until that CSV is
downloaded; the important property here is that BOTH sources produce identical
columns.
"""

import hashlib

import numpy as np
import pandas as pd

# Canonical feature order (28 columns). Kept in one place so the merger and any
# model code can import it instead of hard-coding names.
FEATURE_COLS = [
    "duration", "num_packets", "num_bytes", "packets_per_sec", "bytes_per_sec",
    "pkts_out", "pkts_in", "bytes_out", "bytes_in",
    "len_mean", "len_std", "len_max", "len_min",
    "len_out_mean", "len_in_mean", "len_out_std", "len_in_std",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "iat_out_mean", "iat_in_mean",
    "burst_max", "burst_mean",
    "ratio_bytes_out_in", "ratio_pkts_out_in", "len_cv",
]


def pcap_to_events(path):
    """Read a pcap/pcapng and return a list of packet events."""
    from scapy.all import rdpcap, IP, IPv6, UDP, TCP

    events = []
    for p in rdpcap(str(path)):
        if IP in p:
            src_ip, dst_ip = p[IP].src, p[IP].dst
        elif IPv6 in p:
            src_ip, dst_ip = p[IPv6].src, p[IPv6].dst
        else:
            continue
        if UDP in p:
            src_port, dst_port = p[UDP].sport, p[UDP].dport
        elif TCP in p:
            src_port, dst_port = p[TCP].sport, p[TCP].dport
        else:
            continue
        events.append({
            "ts": float(p.time),
            "length": int(len(p)),
            "direction": None,
            "src_ip": src_ip, "src_port": int(src_port),
            "dst_ip": dst_ip, "dst_port": int(dst_port),
        })
    return events


def ieee_to_events(path, doq_only=True):
    """Read an IEEE DoQ+QUIC packet-level CSV into packet events.

    Keeps only protocol == 1 (DoQ DNS packets) when doq_only is True; the
    protocol == 0 rows are HTTP/3 content, not DNS, and must not be used for
    the benign-DoQ class.
    """
    df = pd.read_csv(path)
    if doq_only and "protocol" in df.columns:
        df = df[df["protocol"] == 1]
    events = []
    for row in df.to_dict("records"):
        events.append({
            "ts": float(row["relative_time"]),
            "length": int(row["length"]),
            "direction": None,
            "src_ip": str(row["src_ip"]), "src_port": int(row["src_port"]),
            "dst_ip": str(row["dst_ip"]), "dst_port": int(row["dst_port"]),
        })
    return events


def _flow_key(e):
    """Bidirectional 5-tuple key (endpoints sorted) for grouping."""
    a = (e["src_ip"], e["src_port"])
    b = (e["dst_ip"], e["dst_port"])
    lo, hi = sorted([a, b])
    return (lo[0], lo[1], hi[0], hi[1])


def group_into_flows(events, idle_timeout=120.0):
    """Split events into flows: same endpoints + gap <= idle_timeout."""
    events = sorted(events, key=lambda e: e["ts"])
    flows = {}
    last_ts = {}
    for e in events:
        k = _flow_key(e)
        if k not in flows or (e["ts"] - last_ts[k]) > idle_timeout:
            flows[k] = []
        flows[k].append(e)
        last_ts[k] = e["ts"]
    return list(flows.values())


def _annotate_direction(flow):
    """Set each event's direction relative to the first packet's source."""
    client = (flow[0]["src_ip"], flow[0]["src_port"])
    for e in flow:
        e["direction"] = "out" if (e["src_ip"], e["src_port"]) == client else "in"
    return flow


def split_windows(flow, window_sec=10.0):
    """Segment one flow's events into fixed-width time windows.

    Persistent DoQ tunnels are long-lived; windowing turns one long connection
    into many labeled samples (primary ML unit). Session-level aggregates are
    a separate concern (kept by flow_id).
    """
    if not flow:
        return []
    windows, cur, start = [], [], flow[0]["ts"]
    for e in flow:
        if e["ts"] - start > window_sec and cur:
            windows.append(cur)
            cur, start = [], e["ts"]
        cur.append(e)
    if cur:
        windows.append(cur)
    return windows


def features_from_window(events):
    """Compute the 28 FEATURE_COLS statistics for one window of events."""
    events = sorted(events, key=lambda e: e["ts"])
    ts = np.array([e["ts"] for e in events], dtype=float)
    ln = np.array([e["length"] for e in events], dtype=float)
    n = len(events)
    dur = float(ts[-1] - ts[0])

    f = {}
    f["duration"] = dur
    f["num_packets"] = n
    f["num_bytes"] = int(ln.sum())
    f["packets_per_sec"] = (n / dur) if dur > 0 else 0.0
    f["bytes_per_sec"] = (f["num_bytes"] / dur) if dur > 0 else 0.0

    out = [e for e in events if e["direction"] == "out"]
    inn = [e for e in events if e["direction"] == "in"]
    out_ln = np.array([e["length"] for e in out], dtype=float)
    in_ln = np.array([e["length"] for e in inn], dtype=float)

    f["pkts_out"] = len(out)
    f["pkts_in"] = len(inn)
    f["bytes_out"] = int(out_ln.sum()) if out_ln.size else 0
    f["bytes_in"] = int(in_ln.sum()) if in_ln.size else 0

    f["len_mean"] = float(ln.mean())
    f["len_std"] = float(ln.std()) if n > 1 else 0.0
    f["len_max"] = float(ln.max())
    f["len_min"] = float(ln.min())

    f["len_out_mean"] = float(out_ln.mean()) if out_ln.size else 0.0
    f["len_in_mean"] = float(in_ln.mean()) if in_ln.size else 0.0
    f["len_out_std"] = float(out_ln.std()) if out_ln.size > 1 else 0.0
    f["len_in_std"] = float(in_ln.std()) if in_ln.size > 1 else 0.0

    iat = np.diff(ts)
    if iat.size >= 1:
        f["iat_mean"] = float(iat.mean())
        f["iat_std"] = float(iat.std()) if iat.size > 1 else 0.0
        f["iat_max"] = float(iat.max())
        f["iat_min"] = float(iat.min())
    else:
        f["iat_mean"] = f["iat_std"] = f["iat_max"] = f["iat_min"] = 0.0

    def _iat_mean(seq):
        t = np.array([e["ts"] for e in sorted(seq, key=lambda e: e["ts"])], dtype=float)
        return float(np.diff(t).mean()) if t.size >= 2 else 0.0

    f["iat_out_mean"] = _iat_mean(out)
    f["iat_in_mean"] = _iat_mean(inn)

    # bursts: maximal runs of consecutive packets in the same direction
    runs, prev, cnt = [], None, 0
    for e in events:
        if e["direction"] == prev:
            cnt += 1
        else:
            if cnt:
                runs.append(cnt)
            cnt, prev = 1, e["direction"]
    if cnt:
        runs.append(cnt)
    f["burst_max"] = int(max(runs)) if runs else 0
    f["burst_mean"] = float(np.mean(runs)) if runs else 0.0

    f["ratio_bytes_out_in"] = (f["bytes_out"] / f["bytes_in"]) if f["bytes_in"] > 0 else 0.0
    f["ratio_pkts_out_in"] = (f["pkts_out"] / f["pkts_in"]) if f["pkts_in"] > 0 else 0.0
    f["len_cv"] = (f["len_std"] / f["len_mean"]) if f["len_mean"] > 0 else 0.0

    return f


def _flow_id(flow):
    k = _flow_key(flow[0])
    return f"{k[0]}:{k[1]}-{k[2]}:{k[3]}"


def extract_features(events, window_sec=10.0, idle_timeout=120.0, min_packets=2):
    """Group events into flows -> windows -> feature rows.

    Returns a DataFrame with FEATURE_COLS plus flow_id and window_start_ts.
    """
    rows = []
    for flow in group_into_flows(events, idle_timeout):
        flow = _annotate_direction(flow)
        fid = _flow_id(flow)
        for win in split_windows(flow, window_sec):
            if len(win) < min_packets:
                continue
            row = features_from_window(win)
            row["flow_id"] = fid
            row["window_start_ts"] = float(win[0]["ts"])
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS + ["flow_id", "window_start_ts"])
    return pd.DataFrame(rows)
