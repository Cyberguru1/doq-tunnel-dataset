#!/usr/bin/env python3
"""Train / evaluate the proposed TD-DilatedConv-BiLSTM-Residual detector.

Examples
--------
# CIRA-CIC-DoHBrw-2020, tool identification (benign/dns2tcp/dnscat2/iodine),
# per-flow protocol (T=1, 80/20 stratified) -> comparable with published results
python src/model/train.py --dataset cira --root $DS/cira --task tool --seq-len 1 --out results/cira_tool_T1

# same, persistent-session windows of 8 consecutive flows
python src/model/train.py --dataset cira --root $DS/cira --task tool --seq-len 8 --out results/cira_tool_T8

# 5-fold CV, base (no residual) variant for ablation
python src/model/train.py --dataset cira --root $DS/cira --task tool --seq-len 8 --cv 5 --variant base

# our DoQ dataset
python src/model/train.py --dataset doq --csv $DS/combined.csv --task binary --seq-len 4 --out results/doq_T4

Outputs in --out: metrics.json, report.md, confusion.csv, model.pt, scaler.json, history.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from data import (FeatureScaler, augment, load_cira, load_doq, make_sequences,  # noqa: E402
                  split_indices, train_val_split)
from net import build_model, count_params  # noqa: E402


# --------------------------------------------------------------------------- #
def metrics_from_preds(y_true, y_pred, class_names, proba=None):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    out = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p); out[f"recall_{avg}"] = float(r); out[f"f1_{avg}"] = float(f)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)), zero_division=0)
    out["per_class"] = {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                        for i, n in enumerate(class_names)}
    out["confusion"] = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist()
    if proba is not None:
        try:
            if len(class_names) == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return out


def to_flow_level(pred_windows, proba_windows, idx_windows, mask, n_rows):
    """Map window predictions back to the individual records (flows) inside them.
    Each record gets the average probability of all windows that contain it."""
    n_cls = proba_windows.shape[1]
    acc = np.zeros((n_rows, n_cls), np.float64)
    cnt = np.zeros(n_rows, np.int64)
    ids = idx_windows[mask]
    pr = np.repeat(proba_windows, mask.sum(1), axis=0)
    np.add.at(acc, ids, pr)
    np.add.at(cnt, ids, 1)
    seen = cnt > 0
    acc[seen] /= cnt[seen, None]
    return acc, seen


# --------------------------------------------------------------------------- #
def train_one(X, M, y, idx_tr, idx_va, idx_te, args, class_names, device, log):
    n_features, n_classes = X.shape[-1], len(class_names)
    scaler = FeatureScaler(log_compress=not args.no_log).fit(X[idx_tr], M[idx_tr])
    Xs = scaler.transform(X, M)

    model = build_model(n_features, n_classes, variant=args.variant,
                        channels=args.channels, embed_dim=args.embed_dim,
                        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
                        dropout=args.dropout).to(device)
    log(f"model={args.variant} params={count_params(model):,} T={X.shape[1]} F={n_features} classes={class_names}")

    counts = np.bincount(y[idx_tr], minlength=n_classes).astype(np.float64)
    w = (counts.sum() / (n_classes * np.maximum(counts, 1))) ** args.class_weight_pow
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                               label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(idx_tr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed)

    def predict(indices, bs=4096):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(indices), bs):
                b = indices[s:s + bs]
                xb = torch.from_numpy(Xs[b]).to(device)
                mb = torch.from_numpy(M[b]).to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outs.append(torch.softmax(model(xb, mb).float(), 1).cpu().numpy())
        return np.concatenate(outs)

    from sklearn.metrics import f1_score
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(idx_tr)
        tot, t0 = 0.0, time.time()
        for s in range(0, len(perm), args.batch_size):
            b = perm[s:s + args.batch_size]
            xb = Xs[b]
            if not args.no_augment:
                xb = augment(xb, M[b], jitter=args.jitter, step_drop=args.step_drop,
                             feat_drop=args.feat_drop, rng=rng)
            xb = torch.from_numpy(xb).to(device)
            mb = torch.from_numpy(M[b]).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(xb, mb), yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt); scaler_amp.update(); sched.step()
            tot += float(loss) * len(b)
        pv = predict(idx_va)
        f1v = f1_score(y[idx_va], pv.argmax(1), average="macro")
        history.append({"epoch": ep, "train_loss": tot / len(idx_tr), "val_f1_macro": f1v, "sec": time.time() - t0})
        log(f"epoch {ep:3d} loss {tot/len(idx_tr):.4f} val_f1_macro {f1v:.4f} ({time.time()-t0:.1f}s)")
        if f1v > best_f1 + 1e-5:
            best_f1, best_epoch, bad = f1v, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep} (best {best_epoch})")
                break
    model.load_state_dict(best_state)
    pt = predict(idx_te)
    return model, scaler, pt, history, best_epoch


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["cira", "doq"], required=True)
    ap.add_argument("--root", help="CIRA CSV root directory")
    ap.add_argument("--csv", help="DoQ combined.csv")
    ap.add_argument("--task", default="tool", help="cira: l1|binary|tool ; doq: binary|tool")
    ap.add_argument("--seq-len", type=int, default=1)
    ap.add_argument("--stride", type=int, default=None, help="default = seq_len (non-overlapping windows)")
    ap.add_argument("--split", choices=["stratified", "group"], default="stratified")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--cv", type=int, default=0, help="K-fold CV instead of a single split")
    ap.add_argument("--max-rows-per-class", type=int, default=None)
    ap.add_argument("--variant", choices=["enhanced", "base"], default="enhanced")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=96)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--class-weight-pow", type=float, default=0.5, help="0=no weighting, 1=inverse frequency")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--step-drop", type=float, default=0.1)
    ap.add_argument("--feat-drop", type=float, default=0.05)
    ap.add_argument("--no-log", action="store_true", help="disable log1p feature compression")
    ap.add_argument("--baselines", action="store_true", help="also run RF/XGB/MLP on the same split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "train.log"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"== {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)} device={device}")

    t0 = time.time()
    if args.dataset == "cira":
        df = load_cira(args.root, args.task, args.max_rows_per_class, args.seed)
    else:
        df = load_doq(args.csv, args.task)
    feature_cols, class_names = df.attrs["feature_cols"], df.attrs["class_names"]
    log(f"loaded {len(df):,} records in {time.time()-t0:.1f}s; classes={dict(df['label_name'].value_counts())}")
    log(f"features ({len(feature_cols)}): {feature_cols}")
    gs = df.groupby("label_name")["group"].agg(["nunique"])
    gsize = df.groupby(["label_name", "group"]).size().groupby("label_name").describe()[["50%", "max"]]
    log("session groups per class:\n" + gs.join(gsize).to_string())

    stride = args.stride or args.seq_len
    X, M, y, g, idx = make_sequences(df, feature_cols, args.seq_len, stride)
    log(f"windows: {X.shape} (T={args.seq_len}, stride={stride}); per class={np.bincount(y).tolist()}")

    if args.cv and args.cv > 1:
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        kf = (StratifiedKFold(args.cv, shuffle=True, random_state=args.seed) if args.split == "stratified"
              else GroupKFold(args.cv))
        folds = list(kf.split(X, y, g))
    else:
        tr, va, te = split_indices(y, g, args.split, args.test_size, args.val_size, args.seed)
        folds = [(np.concatenate([tr, va]), te)]

    all_window, all_flow, all_base = [], [], []
    for k, (trval, te) in enumerate(folds):
        if args.cv and args.cv > 1:
            tr, va = train_val_split(y[trval], g[trval], args.split, args.val_size, args.seed + k)
            tr, va = trval[tr], trval[va]
        log(f"--- fold {k}: train {len(tr):,} val {len(va):,} test {len(te):,}")
        model, scaler, pt, history, best_epoch = train_one(X, M, y, tr, va, te, args, class_names, device, log)

        mw = metrics_from_preds(y[te], pt.argmax(1), class_names, pt)
        mw["best_epoch"] = best_epoch
        all_window.append(mw)
        log(f"[fold {k}] window-level: acc {mw['accuracy']:.4f} f1_macro {mw['f1_macro']:.4f} f1_weighted {mw['f1_weighted']:.4f}")

        if args.seq_len > 1:  # per-record (flow) level, directly comparable with per-flow literature
            acc, seen = to_flow_level(pt.argmax(1), pt, idx[te], M[te], len(df))
            yf = df["label"].to_numpy()[seen]
            mf = metrics_from_preds(yf, acc[seen].argmax(1), class_names, acc[seen])
            all_flow.append(mf)
            log(f"[fold {k}] flow-level:   acc {mf['accuracy']:.4f} f1_macro {mf['f1_macro']:.4f} f1_weighted {mf['f1_weighted']:.4f}")

        if args.baselines:
            from baselines import run_baselines
            Xs = scaler.transform(X, M)
            flat = Xs.reshape(len(Xs), -1)
            res = run_baselines(flat[np.concatenate([tr, va])], y[np.concatenate([tr, va])], flat[te], y[te],
                                class_names, seed=args.seed, log=log)
            all_base.append(res)

        if k == 0:
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "class_names": class_names,
                        "feature_cols": feature_cols, "scaler": scaler.state()}, os.path.join(args.out, "model.pt"))
            json.dump(scaler.state(), open(os.path.join(args.out, "scaler.json"), "w"))
            pd.DataFrame(history).to_csv(os.path.join(args.out, "history.csv"), index=False)
            pd.DataFrame(mw["confusion"], index=class_names, columns=class_names).to_csv(
                os.path.join(args.out, "confusion.csv"))

    def agg(ms, keys=("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")):
        return {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))}
                for k in keys if all(k in m for m in ms)}

    summary = {
        "args": vars(args), "class_names": class_names, "n_records": int(len(df)), "n_windows": int(len(X)),
        "feature_cols": feature_cols, "folds": len(folds),
        "window_level": {"per_fold": all_window, "aggregate": agg(all_window)},
    }
    if all_flow:
        summary["flow_level"] = {"per_fold": all_flow, "aggregate": agg(all_flow)}
    if all_base:
        summary["baselines"] = {name: agg([r[name] for r in all_base]) for name in all_base[0]}
    json.dump(summary, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)

    # Markdown report
    lines = [f"# Results: {args.dataset} / task={args.task} / T={args.seq_len} / variant={args.variant} / split={args.split}"
             + (f" / {args.cv}-fold CV" if args.cv else f" / test={args.test_size}"), "",
             f"records={len(df):,} windows={len(X):,} classes={class_names}", "",
             "| Model | Level | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |",
             "|---|---|---|---|---|---|---|"]

    def row(name, level, a):
        f = lambda k: f"{a[k]['mean']*100:.3f} ± {a[k]['std']*100:.3f}" if k in a else "-"
        return f"| {name} | {level} | {f('accuracy')} | {f('precision_macro')} | {f('recall_macro')} | {f('f1_macro')} | {f('f1_weighted')} |"

    lines.append(row(f"Proposed ({args.variant})", "window", summary["window_level"]["aggregate"]))
    if all_flow:
        lines.append(row(f"Proposed ({args.variant})", "flow", summary["flow_level"]["aggregate"]))
    for name, a in summary.get("baselines", {}).items():
        lines.append(row(name, "window", a))
    lines += ["", "## Per-class (fold 0, window level)", "", "| class | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for n, v in all_window[0]["per_class"].items():
        lines.append(f"| {n} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    lines += ["", "## Confusion matrix (fold 0)", "", pd.DataFrame(all_window[0]["confusion"], index=class_names,
                                                                 columns=class_names).to_markdown()]
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
