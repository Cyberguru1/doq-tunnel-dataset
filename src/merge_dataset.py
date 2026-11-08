"""Merge generated + public DoQ traffic into one labeled dataset CSV.

Reads a JSON config that lists sources (our pcaps, IEEE DoQ+QUIC CSVs), runs
each through the shared extract_features() pipeline, attaches class/tool/source
labels, and concatenates everything into one CSV.

Example config (sources.json):

    {
      "window_sec": 10.0,
      "idle_timeout": 120.0,
      "min_packets": 2,
      "sources": [
        {"kind": "pcap", "paths": ["malicious_doq_*.pcap"],
         "class": "malicious_doq", "tool": "dnstt", "source": "lab_dnstt"},
        {"kind": "pcap", "paths": ["benign_doq_lab_*.pcap"],
         "class": "benign_doq", "tool": null, "source": "lab_dnsproxy"},
        {"kind": "ieee_csv", "paths": ["doq_quic/*.csv"],
         "class": "benign_doq", "tool": null, "source": "ieee_doqquic"},
        {"kind": "pcap", "paths": ["plain_dns_*.pcap"],
         "class": "non_doq", "tool": null, "source": "lab"}
      ]
    }

Usage:
    python merge_dataset.py --config sources.json --out combined.csv
"""

import argparse
import glob
import json

import pandas as pd

from extract_features import FEATURE_COLS, doq_jsonl_to_events, extract_features, ieee_to_events, pcap_to_events

LABEL_COLS = ["class", "tool", "source"]


def expand_paths(patterns):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise FileNotFoundError(f"no files matched {patterns}")
    return paths


def build(config):
    frames = []
    for src in config["sources"]:
        kind = src["kind"]
        paths = expand_paths(src["paths"])
        events = []
        for p in paths:
            if kind == "pcap":
                events += pcap_to_events(p)
            elif kind == "ieee_csv":
                events += ieee_to_events(p)
            elif kind == "doq_jsonl":
                events += doq_jsonl_to_events(p)
            else:
                raise ValueError(f"unknown kind: {kind}")

        df = extract_features(
            events,
            window_sec=config.get("window_sec", 10.0),
            idle_timeout=config.get("idle_timeout", 120.0),
            min_packets=config.get("min_packets", 2),
        )
        df["class"] = src["class"]
        df["tool"] = src.get("tool")
        df["source"] = src["source"]
        frames.append(df)
        print(f"[{kind}] {len(paths)} file(s) -> {len(df)} windows (class={src['class']}, source={src['source']})")

    out = pd.concat(frames, ignore_index=True)
    cols = FEATURE_COLS + ["flow_id", "window_start_ts"] + LABEL_COLS
    out = out[[c for c in cols if c in out.columns]]
    return out


def balance_classes(df, ratio=1.0, seed=0):
    """Subsample majority classes so no class exceeds ratio * (smallest class).

    Uses random subsampling of real captured windows (not synthetic
    oversampling) so the balanced CSV stays faithful to observed traffic.
    ratio=1.0 gives an even split; ratio=2.0 keeps up to 2x the minority class.
    """
    counts = df["class"].value_counts()
    cap = int(ratio * counts.min())
    parts = []
    for cls, g in df.groupby("class"):
        if len(g) > cap:
            g = g.sample(n=cap, random_state=seed)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--balance", action="store_true",
                    help="subsample majority classes toward the smallest class")
    ap.add_argument("--balance-ratio", type=float, default=1.0,
                    help="max class size as a multiple of the smallest class (default 1.0)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    df = build(config)

    if args.balance:
        full_out = args.out.replace(".csv", "_full.csv")
        df.to_csv(full_out, index=False)
        print(f"wrote {len(df)} unbalanced rows to {full_out}")
        before = df["class"].value_counts().to_dict()
        df = balance_classes(df, ratio=args.balance_ratio, seed=args.seed)
        after = df["class"].value_counts().to_dict()
        print(f"balanced (ratio={args.balance_ratio}): {before} -> {after}")

    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df.groupby(["class", "tool", "source"]).size().to_string())


if __name__ == "__main__":
    main()
