#!/usr/bin/env python3
"""Download CIRA-CIC-DoHBrw-2020 (original DoHMeter CSVs with IPs/timestamps and
per-tool files) from the Kaggle mirror into a persistent directory, then summarise.

Usage: python fetch_cira.py <dest_dir> [kaggle_handle ...]
Default handles tried in order:
  dhoogla/cicdohbrw2020/versions/1          (18 original CSVs, 393 MB)
  peterfriedrich1/dns-test-traffic-dohbrw2020
"""
import glob
import os
import shutil
import sys

import pandas as pd

dest = sys.argv[1]
handles = sys.argv[2:] or ["dhoogla/cicdohbrw2020/versions/1", "peterfriedrich1/dns-test-traffic-dohbrw2020"]
os.makedirs(dest, exist_ok=True)

have = glob.glob(os.path.join(dest, "**", "*.csv"), recursive=True)
if not have:
    import kagglehub
    for h in handles:
        try:
            p = kagglehub.dataset_download(h)
            files = [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(f)]
            print(f"{h}: {len(files)} files at {p}")
            for f in files:
                rel = os.path.relpath(f, p)
                print(f"   {os.path.getsize(f)/1e6:8.1f} MB  {rel}")
            if any(f.endswith(".csv") for f in files):
                for f in files:
                    d = os.path.join(dest, os.path.relpath(f, p))
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(f, d)
                print("copied CSVs to", dest)
                break
        except Exception as e:  # noqa: BLE001
            print(f"{h} failed: {e}")

print("\n== summary of", dest)
for f in sorted(glob.glob(os.path.join(dest, "**", "*"), recursive=True)):
    if not os.path.isfile(f):
        continue
    try:
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, low_memory=False, nrows=200000)
    except Exception as e:  # noqa: BLE001
        print("skip", f, e); continue
    lab = df["Label"].value_counts().to_dict() if "Label" in df.columns else "-"
    ip = "SourceIP" in df.columns
    print(f"{os.path.relpath(f, dest):55s} rows>={len(df):7d} cols={df.shape[1]:3d} ip={ip} labels={lab}")
