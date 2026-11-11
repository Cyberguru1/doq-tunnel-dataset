"""Classical baselines evaluated on exactly the same split / features as the
proposed model (Random Forest, XGBoost, MLP, plus a plain LSTM for sequence
inputs). Used by train.py --baselines and by compare.py."""

from __future__ import annotations

import time

import numpy as np


def run_baselines(Xtr, ytr, Xte, yte, class_names, seed=0, log=print, models=("rf", "xgb", "mlp")):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    from train import metrics_from_preds

    out = {}
    for name in models:
        t0 = time.time()
        if name == "rf":
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
        elif name == "xgb":
            import xgboost as xgb
            clf = xgb.XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
                                    colsample_bytree=0.9, tree_method="hist", random_state=seed,
                                    n_jobs=-1, eval_metric="mlogloss")
        elif name == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True, random_state=seed)
        else:
            raise ValueError(name)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        m = metrics_from_preds(yte, proba.argmax(1), class_names, proba)
        m["fit_seconds"] = time.time() - t0
        out[name.upper()] = m
        log(f"[baseline {name.upper()}] acc {m['accuracy']:.4f} f1_macro {m['f1_macro']:.4f} "
            f"f1_weighted {m['f1_weighted']:.4f} ({m['fit_seconds']:.0f}s)")
    return out
