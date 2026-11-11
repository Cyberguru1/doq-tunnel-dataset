"""Data loading, session-sequence construction, scaling and augmentation.

Two datasets share one code path:

* CIRA-CIC-DoHBrw-2020 (DoHMeter CSVs)  -> load_cira()
    layer-2 task "binary": benign-DoH vs malicious-DoH
    task "tool":           benign / dns2tcp / dnscat2 / iodine (4 classes)
    task "l1":             DoH vs non-DoH
* Our DoQ dataset (combined.csv from merge_dataset.py) -> load_doq()
    task "binary":         benign_doq vs malicious_doq
    task "tool":           benign + per-tunnel-config classes (if present)

Every loader returns a DataFrame with:
    feature columns (numeric), 'label' (int), 'label_name' (str),
    'group' (session key used for sequence building and grouped splits),
    'ts' (float, ordering inside a group)

Sequences: consecutive records of the same session group, ordered by 'ts',
are stacked into windows of `seq_len` steps (stride `stride`). Short groups
are zero-padded and masked. seq_len=1 reproduces per-flow classification.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# DoHMeter identifier / non-feature columns in CIRA-CIC-DoHBrw-2020.
CIRA_ID_COLS = ["SourceIP", "DestinationIP", "SourcePort", "DestinationPort", "TimeStamp", "Label", "DoH"]

TOOL_PATTERNS = {  # file/dir name -> tool class
    "dns2tcp": "dns2tcp",
    "dnscat": "dnscat2",
    "iodine": "iodine",
}


# --------------------------------------------------------------------------- #
# CIRA-CIC-DoHBrw-2020
# --------------------------------------------------------------------------- #
def _find_csvs(root: str):
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not files:  # parquet mirror (dhoogla/cicdohbrw2020 v3: no IPs, no per-tool files)
        files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    return files


def _tool_from_path(p: str):
    low = p.lower()
    for pat, name in TOOL_PATTERNS.items():
        if pat in low:
            return name
    return None


def load_cira(root: str, task: str = "tool", max_rows_per_class: int | None = None,
              seed: int = 0) -> pd.DataFrame:
    """Load CIRA-CIC-DoHBrw-2020 CSVs under `root`.

    Expected layout (official download): .../CSVs/Total-CSVs/{l1-doh,l1-nondoh,l2-benign,l2-malicious}.csv
    and, for the tool task, per-tool CSV folders/files whose path contains
    dns2tcp / dnscat2 / iodine. If only l2-malicious.csv exists and the
    'Label' column itself carries tool names, that is used instead.
    """
    files = _find_csvs(root)
    if not files:
        raise FileNotFoundError(f"no CSVs under {root}")

    frames = []
    if task == "l1":
        for f in files:
            b = os.path.basename(f).lower()
            if "l1-doh-nondoh" in b:  # single mirror file, label inside
                df = _read_cira(f, None); df["label_name"] = df["Label"].astype(str).str.lower(); frames.append(df)
            elif b.startswith("l1-doh"):
                frames.append(_read_cira(f, "doh"))
            elif b.startswith("l1-nondoh"):
                frames.append(_read_cira(f, "nondoh"))
    elif task == "binary":
        for f in files:
            b = os.path.basename(f).lower()
            if "l2-benigndoh-maliciousdoh" in b:  # single mirror file, label inside
                df = _read_cira(f, None); df["label_name"] = df["Label"].astype(str).str.lower(); frames.append(df)
            elif b.startswith("l2-benign"):
                frames.append(_read_cira(f, "benign"))
            elif b.startswith("l2-malicious"):
                frames.append(_read_cira(f, "malicious"))
    elif task == "tool":
        tool_files = [f for f in files if _tool_from_path(f) and "total" not in f.lower()]
        benign = [f for f in files if os.path.basename(f).lower().startswith("l2-benign")]
        for f in benign:
            frames.append(_read_cira(f, "benign"))
        if tool_files:
            for f in tool_files:
                frames.append(_read_cira(f, _tool_from_path(f)))
        else:  # fall back to tool names inside the Label column of l2-malicious.csv
            for f in files:
                if os.path.basename(f).lower().startswith("l2-malicious"):
                    df = _read_cira(f, None)
                    df["label_name"] = df["Label"].astype(str).str.lower().map(
                        lambda s: next((v for k, v in TOOL_PATTERNS.items() if k in s), "malicious"))
                    frames.append(df)
    else:
        raise ValueError(task)
    if not frames:
        raise FileNotFoundError(f"no CSVs matching task={task} under {root}")

    df = pd.concat(frames, ignore_index=True)
    df = _finalize_cira(df)
    if max_rows_per_class:
        df = (df.groupby("label_name", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_rows_per_class), random_state=seed)))
        df = df.reset_index(drop=True)
    return df


def _read_cira(path: str, label_name):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "DoH" in df.columns:
        # per-tool / per-browser DoHMeter CSVs carry a boolean 'DoH' column instead of 'Label'.
        # Keep only DoH flows (this is exactly how the official l2-*.csv files were derived) and
        # never let the column reach the feature set.
        is_doh = df["DoH"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        n0 = len(df)
        df = df[is_doh].copy()
        print(f"  {os.path.basename(path)}: kept {len(df):,} DoH flows of {n0:,} (DoH column dropped)")
    if label_name is not None:
        df["label_name"] = label_name
    df["src_file"] = os.path.basename(path)
    return df


def _finalize_cira(df: pd.DataFrame) -> pd.DataFrame:
    # session key: client<->resolver pair; order inside it by timestamp
    if "SourceIP" in df.columns and "DestinationIP" in df.columns:
        df["group"] = (df["SourceIP"].astype(str) + ">" + df["DestinationIP"].astype(str)
                       + "|" + df["label_name"].astype(str))
    else:  # no session identifiers in this mirror -> per-record groups (seq_len must be 1)
        df["group"] = np.arange(len(df)).astype(str)
    if "TimeStamp" in df.columns:
        ts = pd.to_datetime(df["TimeStamp"], errors="coerce")
        df["ts"] = (ts - ts.min()).dt.total_seconds().fillna(0.0)
    else:
        df["ts"] = 0.0
    feat = [c for c in df.columns
            if c not in CIRA_ID_COLS + ["label_name", "src_file", "group", "ts", "label"]]
    df[feat] = df[feat].apply(pd.to_numeric, errors="coerce")
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    names = sorted(df["label_name"].unique(), key=lambda s: (s != "benign", s != "doh", s))
    df["label"] = df["label_name"].map({n: i for i, n in enumerate(names)}).astype(int)
    df.attrs["feature_cols"] = feat
    df.attrs["class_names"] = names
    return df


# --------------------------------------------------------------------------- #
# Our DoQ dataset (combined.csv)
# --------------------------------------------------------------------------- #
def load_doq(csv_path: str, task: str = "binary") -> pd.DataFrame:
    from extract_features import FEATURE_COLS  # same 28 features as the extractor

    df = pd.read_csv(csv_path)
    if task == "binary":
        df["label_name"] = df["class"].astype(str)
    elif task == "tool":
        tool = df["tool"].astype(str) if "tool" in df.columns else "unknown"
        df["label_name"] = np.where(df["class"].astype(str).str.startswith("malicious"),
                                    "malicious_" + tool, df["class"].astype(str))
    else:
        raise ValueError(task)
    df["group"] = df["flow_id"].astype(str) if "flow_id" in df.columns else np.arange(len(df)).astype(str)
    df["ts"] = pd.to_numeric(df.get("window_start_ts", 0.0), errors="coerce").fillna(0.0)
    feat = [c for c in FEATURE_COLS if c in df.columns]
    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    names = sorted(df["label_name"].unique(), key=lambda s: ("benign" not in s, s))
    df["label"] = df["label_name"].map({n: i for i, n in enumerate(names)}).astype(int)
    df.attrs["feature_cols"] = feat
    df.attrs["class_names"] = names
    return df


# --------------------------------------------------------------------------- #
# Sequence construction
# --------------------------------------------------------------------------- #
def make_sequences(df: pd.DataFrame, feature_cols, seq_len: int = 8, stride: int | None = None):
    """Stack consecutive records of each session group into (N, T, F) windows.

    Returns X (N,T,F) float32, M (N,T) bool mask, y (N,) int, g (N,) group id,
    idx (N,T) row indices into df (-1 for padding).
    """
    if stride is None:
        stride = max(1, seq_len // 2)
    Fd = len(feature_cols)
    if seq_len == 1:
        X = df[feature_cols].to_numpy(np.float32)[:, None, :]
        M = np.ones((len(df), 1), dtype=bool)
        return X, M, df["label"].to_numpy(), df["group"].to_numpy(), np.arange(len(df))[:, None]

    order = df.sort_values(["group", "ts"], kind="mergesort")
    feats = order[feature_cols].to_numpy(np.float32)
    labels = order["label"].to_numpy()
    groups = order["group"].to_numpy()
    rows = order.index.to_numpy()

    Xs, Ms, ys, gs, ids = [], [], [], [], []
    # boundaries of each group in the sorted array
    change = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(groups)]])
    for s, e in zip(starts, ends):
        n = e - s
        if n <= seq_len:
            w_starts = [s]
        else:
            w_starts = list(range(s, e - seq_len + 1, stride))
            if w_starts[-1] + seq_len < e:
                w_starts.append(e - seq_len)
        for ws in w_starts:
            we = min(ws + seq_len, e)
            x = np.zeros((seq_len, Fd), np.float32)
            m = np.zeros(seq_len, bool)
            i = np.full(seq_len, -1, np.int64)
            x[: we - ws] = feats[ws:we]
            m[: we - ws] = True
            i[: we - ws] = rows[ws:we]
            Xs.append(x); Ms.append(m); ys.append(labels[ws]); gs.append(groups[ws]); ids.append(i)
    return (np.stack(Xs), np.stack(Ms), np.asarray(ys), np.asarray(gs), np.stack(ids))


# --------------------------------------------------------------------------- #
# Scaling & augmentation
# --------------------------------------------------------------------------- #
class FeatureScaler:
    """signed log1p compression followed by standardisation (fit on train only)."""

    def __init__(self, log_compress: bool = True):
        self.log_compress = log_compress
        self.mean = None
        self.std = None

    def _pre(self, X):
        if self.log_compress:
            return np.sign(X) * np.log1p(np.abs(X))
        return X

    def fit(self, X, mask=None):
        Z = self._pre(X.reshape(-1, X.shape[-1]))
        if mask is not None:
            Z = Z[mask.reshape(-1)]
        self.mean = Z.mean(0)
        self.std = Z.std(0) + 1e-6
        return self

    def transform(self, X, mask=None):
        Z = (self._pre(X) - self.mean) / self.std
        Z = np.clip(Z, -10, 10).astype(np.float32)
        if mask is not None:
            Z = Z * mask[..., None]
        return Z

    def state(self):
        return {"log_compress": self.log_compress, "mean": self.mean.tolist(), "std": self.std.tolist()}


def augment(x, mask, jitter: float = 0.05, step_drop: float = 0.1, feat_drop: float = 0.05, rng=None):
    """Training-time augmentation on a scaled batch (numpy): Gaussian jitter,
    random time-step dropout (masked), random feature dropout."""
    rng = rng or np.random.default_rng()
    x = x.copy()
    if jitter > 0:
        x += rng.normal(0, jitter, x.shape).astype(np.float32)
    if step_drop > 0 and x.shape[1] > 1:
        drop = rng.random(x.shape[:2]) < step_drop
        keep_one = mask.copy()
        drop &= keep_one
        # never drop all valid steps of a window
        all_dropped = (mask & ~drop).sum(1) == 0
        drop[all_dropped] = False
        x[drop] = 0.0
    if feat_drop > 0:
        fd = rng.random((x.shape[0], 1, x.shape[2])) < feat_drop
        x = np.where(fd, 0.0, x).astype(np.float32)
    return x * mask[..., None]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_indices(y, groups, mode: str = "stratified", test_size: float = 0.2,
                  val_size: float = 0.1, seed: int = 0):
    """Return (train_idx, val_idx, test_idx).

    mode='stratified': random stratified split (the protocol used by the
                       published CIRA-CIC-DoHBrw-2020 results).
    mode='group':      GroupShuffleSplit on the session key (stricter: no
                       session appears in both train and test).
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    idx = np.arange(len(y))
    if mode == "stratified":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(sss.split(idx, y))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        tr2, va = next(sss2.split(tr, y[tr]))
        return tr[tr2], tr[va], te
    elif mode == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(idx, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
        tr2, va = next(gss2.split(tr, y[tr], groups[tr]))
        return tr[tr2], tr[va], te
    raise ValueError(mode)


def train_val_split(y, groups, mode: str = "stratified", val_size: float = 0.1, seed: int = 0):
    """Single train/val split (used inside K-fold CV to carve a validation set)."""
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

    idx = np.arange(len(y))
    if mode == "stratified":
        sp = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        return next(sp.split(idx, y))
    sp = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    return next(sp.split(idx, y, groups))
