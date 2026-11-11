#!/usr/bin/env python3
"""Build the comparison table: our runs (results/*/metrics.json) vs published
results on CIRA-CIC-DoHBrw-2020.

Usage: python src/model/compare.py results/cira_* > results/comparison.md
"""

from __future__ import annotations

import json
import os
import sys

# Published results on CIRA-CIC-DoHBrw-2020 (numbers as reported by the papers; protocols differ,
# see the "protocol" column). Values in percent.
LITERATURE = [
    # task, method, paper, protocol, accuracy, f1_macro, f1_weighted, notes
    ("tool", "FF-MR (ResNet byte-seq + stats, multi-head attention)",
     "Wang et al., Applied Sciences 12(24):12644, 2022", "flow-level, 3 tunnel tools (+benign)",
     None, 99.78, None, "macro P 99.72 / R 99.73"),
    ("tool", "MTL-DoHTA task-3 (2D-CNN + GradNorm multi-task)",
     "Jung & Kwak, Sensors 25(4):993, 2025", "8:2 split, flow-level",
     None, 98.37, None, "task-3 F1 (tool identification)"),
    ("tool", "EnhancedRLGWO + Random Forest (12 selected features)",
     "Sammour et al., Frontiers in Computer Science 7, 2026", "4-class (benign/iodine/dns2tcp/dnscat2), 5-fold CV",
     99.82, None, 99.82, "per-tool P/R/F1 reported as 0.98"),
    ("tool", "Random Forest (all 28 features)",
     "reported in Sammour et al. 2026 (Table 4)", "4-class",
     98.50, None, 98.45, ""),
    ("tool", "CNN-LSTM",
     "reported in Sammour et al. 2026 (Table 4)", "4-class",
     97.90, None, 97.85, ""),
    ("tool", "Contrastive sequence learning",
     "Computers 15(9):617, 2026", "per-tool recall dns2tcp 96.74 / dnscat2 93.89 / iodine 89.12",
     None, None, None, "sequence input"),
    ("tool", "Hierarchical GBDT (tool level)",
     "Mitsuhashi et al., IEEE TNSM, 2023", "6 tools incl. dnstt/tcp-over-dns/tuns on their own captures",
     98.02, None, None, "not the same class set"),
    ("binary", "LSTM (time-series)", "MontazeriShatoori et al., DASC 2020", "layer 2, flow-level", None, None, 99.90, ""),
    ("binary", "Balanced stacked RF", "Zebin et al., 2022", "layer 2", None, None, 99.90, ""),
    ("binary", "Ensemble ML", "Aggarwal et al.", "layer 2", None, None, 99.99, ""),
    ("binary", "MTL-DoHTA task-2", "Jung & Kwak, Sensors 2025", "8:2 split", None, None, 99.88, ""),
    ("binary", "BiLSTM", "Casanova et al.", "layer 2", None, None, 99.90, ""),
    ("l1", "Random Forest", "Singh et al.", "layer 1 (DoH vs non-DoH)", None, None, 100.0, ""),
    ("l1", "MTL-DoHTA task-1", "Jung & Kwak, Sensors 2025", "8:2 split", None, None, 98.91, ""),
]


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def ours(path):
    m = json.load(open(os.path.join(path, "metrics.json")))
    a = m["args"]
    rows = []
    level = "flow" if "flow_level" in m else "window"
    agg = m["flow_level"]["aggregate"] if "flow_level" in m else m["window_level"]["aggregate"]
    name = f"**Proposed TD-DilConv-BiLSTM ({a['variant']}, T={a['seq_len']})**"
    proto = f"{a['split']} split" + (f", {a['cv']}-fold CV" if a.get("cv") else f", test={a['test_size']}")
    rows.append((a["task"], name, "this work", f"{proto}, {level}-level",
                 agg["accuracy"]["mean"] * 100, agg["f1_macro"]["mean"] * 100, agg["f1_weighted"]["mean"] * 100,
                 f"± {agg['f1_macro']['std']*100:.2f} (macro-F1 std)" if a.get("cv") else ""))
    for b, agg in m.get("baselines", {}).items():
        rows.append((a["task"], f"{b} (same split, T={a['seq_len']})", "this work", proto,
                     agg["accuracy"]["mean"] * 100, agg["f1_macro"]["mean"] * 100, agg["f1_weighted"]["mean"] * 100, ""))
    return rows


def main():
    rows = list(LITERATURE)
    for p in sys.argv[1:]:
        if os.path.exists(os.path.join(p, "metrics.json")):
            rows += ours(p)
    for task in ("tool", "binary", "l1"):
        sub = [r for r in rows if r[0] == task]
        if not sub:
            continue
        print(f"\n## Task: {task}\n")
        print("| Method | Source | Protocol | Accuracy | F1 macro | F1 weighted | Notes |")
        print("|---|---|---|---|---|---|---|")
        for r in sub:
            print(f"| {r[1]} | {r[2]} | {r[3]} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} | {r[7]} |")


if __name__ == "__main__":
    main()
