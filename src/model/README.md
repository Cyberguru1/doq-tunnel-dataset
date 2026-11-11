# Proposed detector — persistent-session DoH/DoQ tunnel-tool recognition

`net.py` implements the model from the proposal (§5.3) and the two additions found necessary during validation:

```
flow stats x_t (29 or 28 per step)
  └─ PLE feature tokens   : per-feature piecewise-linear quantile embedding -> (d_token=8, F)      [added]
  └─ Time-distributed dilated Conv1d stack over the F feature tokens (dilations 1,2,4, GroupNorm, GELU)
       · inter-block residuals + multi-scale 1x1 fusion + raw-input skip  (enhanced variant)
  └─ per-step embedding (128) ─┐
  └─ BiLSTM (96x2) over T steps ─┴─ attention pooling of both paths (residual fusion) -> MLP head -> tool class
```

Input = a window of **T consecutive flows of one client->resolver session** ("persistent traffic"); T=1 reduces
the model to a per-flow classifier. Training uses leak-free sliding windows (`--train-stride`), class-weighted CE,
label smoothing, jitter / step-drop / feature-drop augmentation, AdamW + cosine LR, early stopping on val macro-F1.

| file | purpose |
|---|---|
| `net.py` | model (`DilatedResBiLSTM`, `PiecewiseLinearEmbedding`, `quantile_bin_edges`) |
| `data.py` | CIRA-CIC-DoHBrw-2020 / our DoQ loaders, session-window builder, scaler, augmentation, splits |
| `train.py` | training + evaluation (window- and flow-level metrics, K-fold CV, baselines, report.md) |
| `baselines.py` | RF / XGBoost / MLP on the identical windows |
| `compare.py` | table: our runs vs published results |
| `fetch_cira.py` | downloads the original CIC CSVs from the Kaggle mirrors |

Reproduce: `notebooks/cira_validation.ipynb` (Colab, T4). Results and the full experiment log live in the
project root (`RESULTS_CIRA.md`, `PROGRESS_LOG.md`).
